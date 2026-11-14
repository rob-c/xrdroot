"""The head of a distributed graph: the data, how it is cut into tasks, and running it all.

:func:`get_headnode` reads ``RDataFrame``'s arguments as ROOT does and makes
the head node for the data they name: :class:`EmptySourceHeadNode` for a
number of empty entries, :class:`TreeHeadNode` for a tree or chain - by name
and files, or handed over - :class:`RDatasetSpecHeadNode` for an
``RDatasetSpec`` and :class:`RNTupleHeadNode` for RNTuples. The head node
keeps every node of the graph and, when a result is asked for, sends the
graph to the backend with the ranges to run it over
(:meth:`HeadNode.execute_graph`), checks what came back and hands each
action its value.

Each task's frame is xrdroot's own, over a
:class:`~xrdroot.rdf.distributed.Window` of the data: the first task of a run
in a process makes it, and the next ones point the window at their own
range and data, keeping the graph.
"""

from __future__ import annotations

import uuid
import warnings
from abc import ABC, abstractmethod
from collections import Counter, deque
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial, singledispatch
from typing import TYPE_CHECKING, Any, cast

from ...errors import UnsupportedFeatureError
from ...rdf.distributed import Window, source_of
from ...rdf.sources import Empty, Joined, Table, open_named
from ..rdf import treeutils
from ..rdf.experimental import RDatasetSpec
from . import ComputationGraphGenerator, Ranges, _graph_cache
from .Backends import Utils
from .Backends.Base import distrdf_mapper, distrdf_reducer
from .Node import Node
from .Operation import Action, InstantAction, Operation

if TYPE_CHECKING:
    from .Backends.Base import BaseBackend, TaskResult

__all__ = [
    "TaskObjects",
    "HeadNode",
    "get_headnode",
    "EmptySourceHeadNode",
    "TreeHeadNode",
    "RDatasetSpecHeadNode",
    "RNTupleHeadNode",
]


@singledispatch
def _append_node_to_actions(operation: Any, node: Node, actions: list[Node]) -> None:
    """Keep the node if its operation is an action."""


@_append_node_to_actions.register(Action)
@_append_node_to_actions.register(InstantAction)
def _(operation: Operation, node: Node, actions: list[Node]) -> None:
    actions.append(node)


@dataclass
class TaskObjects:
    """A task's frame - ``None`` for a task of a tree with nothing in its range - and, for a
    tree, the entries it will process of which trees."""

    rdf: Any
    entries_in_trees: Ranges.TaskTreeEntries | None


def _frame_over(source: Any) -> Any:
    """A PyROOT frame of xrdroot's over ``source``."""
    from ... import rdf as _engine
    from ..rdf.frame import RDataFrame

    return RDataFrame._of(_engine.RDataFrame(source))


def _cached_frame(exec_id: Any, inner: Any, start: int, stop: int) -> Any:
    """The run's frame in this process, its window on ``inner`` from ``start`` to ``stop``."""
    frame = _graph_cache._RDF_REGISTER.get(exec_id)
    if frame is None:
        frame = _graph_cache._RDF_REGISTER[exec_id] = _frame_over(Window(inner, start, stop))
    else:
        _window(frame).change(inner, start, stop)
    return frame


def _window(frame: Any) -> Window:
    """The window a task's frame is over."""
    return cast(Window, source_of(frame._inner))


class HeadNode(Node, ABC):
    """The head of a graph: every node in it, and the backend that runs it.

    ``graph_nodes`` holds the nodes newest first, so that pruning meets a
    child before its parent; ``rdf_node`` is a frame of the whole data in
    this process, which answers for column names and types.
    """

    def __init__(self, backend: BaseBackend | None, npartitions: int | None, localdf: Any) -> None:
        super().__init__(lambda: self)
        self.backend = backend
        self.node_counter = 0
        self.graph_nodes: deque[Node] = deque([self])
        self.rdf_uuid: uuid.UUID | str = uuid.uuid4()
        self.exec_id: _graph_cache.ExecutionIdentifier | None = None
        self._npartitions = npartitions
        self._user_specified_npartitions = npartitions is not None
        self.rdf_node = localdf

    @property
    def npartitions(self) -> int | None:
        return self._npartitions

    @npartitions.setter
    def npartitions(self, value: int | None) -> None:
        """The backend's number of partitions, unless the user gave one."""
        if not self._user_specified_npartitions:
            self._npartitions = value

    def is_prunable(self) -> bool:
        """The head stays: every other node hangs from it."""
        return False

    def _prune_graph(self) -> None:
        self.graph_nodes = deque(node for node in self.graph_nodes if not node.is_prunable())

    def _get_action_nodes(self) -> list[Node]:
        """The graph's actions, oldest first: the order the tasks give their results in."""
        found: list[Node] = []
        for node in reversed(self.graph_nodes):
            _append_node_to_actions(node.operation, node, found)
        return found

    def _generate_graph_dict(self) -> dict[int, Node]:
        """The pruned graph, every node by its id, a parent before its children."""
        self._prune_graph()
        return {node.node_id: node for node in reversed(self.graph_nodes)}

    @abstractmethod
    def _build_ranges(self) -> list[Any]:
        """The ranges of this run's tasks."""

    @abstractmethod
    def _generate_rdf_creator(self) -> Callable[[Any], TaskObjects]:
        """How a task makes the frame of its range."""

    @abstractmethod
    def _handle_returned_values(self, values: TaskResult) -> Iterable[Any]:
        """The results the tasks added up to, checked."""

    def execute_graph(self) -> None:
        """Run the graph on the backend, one task a range, and give every action its value."""
        assert self.backend is not None, "a graph runs on the backend its frame was made with"
        self.npartitions = self.backend.optimize_npartitions()
        self.exec_id = _graph_cache.ExecutionIdentifier(self.rdf_uuid, uuid.uuid4())
        mapper = partial(
            distrdf_mapper,
            build_rdf_from_range=self._generate_rdf_creator(),
            computation_graph_callable=partial(
                ComputationGraphGenerator.trigger_computation_graph, self._generate_graph_dict()
            ),
            initialization_fn=type(self.backend).initialization,
            code_to_declare="\n".join(self.backend.strings_to_declare.values()),
        )
        local_nodes = self._get_action_nodes()
        try:
            returned = self.backend.ProcessAndMerge(self._build_ranges(), mapper, distrdf_reducer)
        finally:
            self.backend.cleanup_cache(self.exec_id)
        for node, value in zip(local_nodes, self._handle_returned_values(returned), strict=True):
            Utils.set_value_on_node(value, node, self.backend)


def _local_frame(args: tuple[Any, ...]) -> Any:
    """The frame of the whole data in this process; a ``TypeError`` for arguments no
    ``RDataFrame`` constructor takes."""
    from ..rdf.frame import RDataFrame

    try:
        if not args:
            raise TypeError("no arguments")
        return RDataFrame(*args)
    except TypeError:
        raise TypeError(
            "The arguments provided are not accepted by any RDataFrame constructor. "
            "See the RDataFrame documentation for the accepted constructor argument types."
        ) from None


def get_headnode(backend: BaseBackend | None, npartitions: int | None, *args: Any) -> HeadNode:
    """The head node for ``RDataFrame(*args)``'s data."""
    localdf = _local_frame(args)
    if isinstance(args[0], RDatasetSpec):
        return RDatasetSpecHeadNode(backend, npartitions, localdf, *args)
    source = source_of(localdf._inner)
    if isinstance(source, Empty):
        return EmptySourceHeadNode(backend, npartitions, localdf, args[0])
    if isinstance(source, Joined) or _is_rntuple(source):
        return RNTupleHeadNode(backend, npartitions, localdf, *args)
    return TreeHeadNode(backend, npartitions, localdf, *args)


def _is_rntuple(source: Any) -> bool:
    from ...rntuple import RNTuple

    return isinstance(source, Table) and isinstance(source.table, RNTuple)


class EmptySourceHeadNode(HeadNode):
    """The head of a graph over ``RDataFrame(nentries)``'s empty entries."""

    def __init__(
        self, backend: BaseBackend | None, npartitions: int | None, localdf: Any, nentries: int
    ) -> None:
        super().__init__(backend, npartitions, localdf)
        self.nentries = int(nentries)

    def _build_ranges(self) -> list[Any]:
        if not self.nentries:
            raise RuntimeError(
                "Cannot build a distributed RDataFrame with zero entries. "
                "Distributed computation will fail. "
            )
        assert self.npartitions is not None, "the backend says how many partitions there are"
        if self.npartitions > self.nentries:
            warnings.warn(
                f"Number of partitions {self.npartitions} is greater than number of entries "
                f"{self.nentries} in the dataframe. Using {self.nentries} partition(s)",
                UserWarning,
                stacklevel=2,
            )
            self.npartitions = self.nentries
        return Ranges.get_balanced_ranges(self.nentries, self.npartitions, self.exec_id)

    def _generate_rdf_creator(self) -> Callable[[Any], TaskObjects]:
        nentries = self.nentries

        def build_rdf_from_range(current_range: Ranges.EmptySourceRange) -> TaskObjects:
            frame = _graph_cache._RDF_REGISTER.get(current_range.exec_id)
            inner = Empty(nentries) if frame is None else _window(frame).inner
            made = _cached_frame(current_range.exec_id, inner, current_range.start,
                                 current_range.end)  # fmt: skip
            return TaskObjects(made, None)

        return build_rdf_from_range

    def _handle_returned_values(self, values: TaskResult) -> Iterable[Any]:
        assert values.mergeables is not None, "every task of empty entries has results"
        return values.mergeables


def _checked_entries(inputfiles: list[str], treenames: list[str], values: TaskResult) -> None:
    """Did the tasks read exactly these trees, and every entry of them once?"""
    if values.mergeables is None:
        raise RuntimeError(
            "The distributed execution returned no values. "
            "This can happen if all files in your dataset contain empty trees."
        )
    counts = Counter(f"{file}/{tree}" for file, tree in zip(inputfiles, treenames, strict=True))
    entries = values.entries_in_trees
    assert entries is not None, "a task of trees says which it read"
    if counts.keys() != entries.trees_with_entries.keys():
        raise RuntimeError(
            "The specified input files and the files that were actually processed are not "
            f"the same:\nInput files: {list(counts)}\n"
            f"Processed files: {list(entries.trees_with_entries)}"
        )
    total = sum(entries.trees_with_entries[path] * times for path, times in counts.items())
    if entries.processed_entries != total:
        raise RuntimeError(
            f"The dataset has {total} entries, but {entries.processed_entries} were processed."
        )


def _tree_of(args: tuple[Any, ...]) -> tuple[Any, Any]:
    """The tree or chain ``RDataFrame``'s arguments name, and the default columns after it."""
    from ..trees import TChain, TTree

    if isinstance(args[0], TTree):
        return args[0], deepcopy(args[1]) if len(args) == 2 else None
    defaults = deepcopy(args[2]) if len(args) == 3 else None
    chain = TChain(str(args[0]))
    for filename in [args[1]] if isinstance(args[1], str) else args[1]:
        chain.Add(str(filename))
    return chain, defaults


class TreeHeadNode(HeadNode):
    """The head of a graph over a tree or chain: by name and files, or handed over with its
    friends."""

    def __init__(
        self, backend: BaseBackend | None, npartitions: int | None, localdf: Any, *args: Any
    ) -> None:
        super().__init__(backend, npartitions, localdf)
        tree, self.defaultbranches = _tree_of(args)
        self.friendinfo: treeutils.RFriendInfo | None = None
        if args[0] is tree:
            indexed = treeutils.indexed_friend(tree)
            if indexed is not None:
                raise ValueError(
                    f"Friend tree '{indexed}' has a TTreeIndex. "
                    "This is not supported in distributed mode."
                )
            self.friendinfo = treeutils.friend_info(tree) or None
        self.maintreename = str(tree.GetName())
        self.subtreenames = treeutils.tree_names(tree)
        self.inputfiles = treeutils.file_names(tree)

    def _build_ranges(self) -> list[Any]:
        assert self.npartitions is not None, "the backend says how many partitions there are"
        return Ranges.get_percentage_ranges(self.subtreenames, self.inputfiles, self.npartitions,
                                            self.friendinfo, self.exec_id, None)  # fmt: skip

    def _generate_rdf_creator(self) -> Callable[[Any], TaskObjects]:
        return _tree_rdf_from_range

    def _handle_returned_values(self, values: TaskResult) -> Iterable[Any]:
        _checked_entries(self.inputfiles, self.subtreenames, values)
        return values.mergeables  # type: ignore[return-value]


def _tree_rdf_from_range(current_range: Ranges.TreeRangePerc) -> TaskObjects:
    """The frame of a task's range of a chain: its trees chained again, with their friends,
    and the window on them from the range's first entry to its last."""
    from ..rdf.frame import RDataFrame

    clustered, entries = Ranges.get_clustered_range_from_percs(current_range)
    if clustered is None:
        return TaskObjects(None, entries)
    chain = treeutils.chain_of(clustered.treenames, clustered.filenames, clustered.friendinfo)
    inner = source_of(RDataFrame(chain)._inner)
    made = _cached_frame(current_range.exec_id, inner, clustered.globalstart, clustered.globalend)
    return TaskObjects(made, entries)


class RDatasetSpecHeadNode(TreeHeadNode):
    """The head of a graph over an ``RDatasetSpec``'s samples."""

    def __init__(
        self, backend: BaseBackend | None, npartitions: int | None, localdf: Any, *args: Any
    ) -> None:
        HeadNode.__init__(self, backend, npartitions, localdf)
        spec: RDatasetSpec = args[0]
        self.defaultbranches = None
        self.friendinfo = spec.GetFriendInfo() or None
        if spec.GetEntryRangeBegin() != 0:
            warnings.warn(
                "Setting global range is not supported in distributed parsing of RDatasetSpec, "
                "the set range will be ignored and all events will be processed.",
                stacklevel=2,
            )
        self.subtreenames = spec.GetTreeNames()
        self.inputfiles = spec.GetFileNameGlobs()
        self.maintreename = self.subtreenames[0] if self.subtreenames else ""


class RNTupleHeadNode(HeadNode):
    """The head of a graph over RNTuples of one name in several files."""

    def __init__(
        self, backend: BaseBackend | None, npartitions: int | None, localdf: Any, *args: Any
    ) -> None:
        super().__init__(backend, npartitions, localdf)
        if len(args) < 2:
            raise UnsupportedFeatureError(
                "distributed RDataFrame reads an RNTuple by its name and the files it is in, "
                "and was handed the RNTuple itself; give RDataFrame its name and its files"
            )
        self.mainntuplename = str(args[0])
        self.inputfiles = [args[1]] if isinstance(args[1], str) else [str(f) for f in args[1]]
        self.subnames = [self.mainntuplename] * len(self.inputfiles)

    def _build_ranges(self) -> list[Any]:
        assert self.exec_id is not None and self.npartitions is not None
        # RNTuple tasks build their graph afresh: the sentinel says so to the generator.
        self.exec_id = _graph_cache.ExecutionIdentifier("RNTuple", self.exec_id.graph_uuid)
        return Ranges.get_percentage_ranges(self.subnames, self.inputfiles, self.npartitions,
                                            None, self.exec_id, None)  # fmt: skip

    def _generate_rdf_creator(self) -> Callable[[Any], TaskObjects]:
        return _rntuple_rdf_from_range

    def _handle_returned_values(self, values: TaskResult) -> Iterable[Any]:
        _checked_entries(self.inputfiles, self.subnames, values)
        return values.mergeables  # type: ignore[return-value]


def _rntuple_rdf_from_range(current_range: Ranges.TreeRangePerc) -> TaskObjects:
    """The frame of a task's range of RNTuples: a frame of its own, made for this task."""
    clustered, entries = Ranges.get_clustered_range_from_percs(current_range)
    if clustered is None:
        return TaskObjects(None, entries)
    inner = open_named(clustered.treenames[0], clustered.filenames)
    window = Window(inner, clustered.globalstart, clustered.globalend)
    return TaskObjects(_frame_over(window), entries)
