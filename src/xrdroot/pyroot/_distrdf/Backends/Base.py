"""A backend's contract, and the map and reduce steps every backend runs.

A task is :func:`distrdf_mapper` applied to one range of entries: it sets the
process up, builds the frame of its range, runs the graph over it and sends
back a :class:`TaskResult`. :func:`distrdf_reducer` adds two of those up. A
backend - :class:`BaseBackend`'s subclasses - decides where the tasks run
(``ProcessAndMerge``), how files reach the workers and how many partitions
to cut the data into; the code every worker needs before it starts is kept
on the class, so it is there for every backend made afterwards.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

from .. import Ranges, _graph_cache
from . import Utils

if TYPE_CHECKING:
    from .._graph_cache import ExecutionIdentifier
    from ..HeadNode import TaskObjects

__all__ = [
    "setup_mapper",
    "get_mergeable_values",
    "TaskResult",
    "distrdf_mapper",
    "merge_values",
    "distrdf_reducer",
    "BaseBackend",
]


def _root() -> Any:
    from .... import pyroot

    return pyroot


def setup_mapper(initialization_fn: Callable[[], Any], code_to_declare: str) -> None:
    """Every task's first steps: no graphics, the user's initialization, the user's code."""
    root = _root()
    root.gROOT.SetBatch(True)
    initialization_fn()
    if code_to_declare:
        root.gInterpreter.Declare(code_to_declare)


def get_mergeable_values(
    starting_node: Any,
    range_id: int,
    computation_graph_callable: Callable[..., list[Any]],
    exec_id: ExecutionIdentifier,
) -> list[Any]:
    """Run the graph on the task's frame; what each of its results sends back."""
    actions = computation_graph_callable(starting_node, range_id, exec_id)
    return [Utils.get_mergeablevalue(action) for action in actions]


@dataclass
class TaskResult:
    """What a task sends back: its results' partials - ``None`` for a task of a tree that had
    nothing in its range - and, for a tree, the entries it processed of which trees."""

    mergeables: list[Any] | None
    entries_in_trees: Ranges.TaskTreeEntries | None


def distrdf_mapper(
    current_range: Ranges.DataRange,
    build_rdf_from_range: Callable[[Any], TaskObjects],
    computation_graph_callable: Callable[..., list[Any]],
    initialization_fn: Callable[[], Any],
    code_to_declare: str,
) -> TaskResult:
    """One task: the graph run over one range of entries."""
    setup_mapper(initialization_fn, code_to_declare)
    with _graph_cache.LOCK:
        rdf_plus = build_rdf_from_range(current_range)
        mergeables = None
        if rdf_plus.rdf is not None:
            mergeables = get_mergeable_values(
                rdf_plus.rdf, current_range.id, computation_graph_callable, current_range.exec_id
            )
    return TaskResult(mergeables, rdf_plus.entries_in_trees)


def merge_values(mergeables_out: Any, mergeables_in: Any) -> Any:
    """The second task's results added into the first's, which are given back."""
    if mergeables_out is None:
        return mergeables_in
    if mergeables_in is not None:
        for mergeable_out, mergeable_in in zip(mergeables_out, mergeables_in, strict=True):
            Utils.merge_values(mergeable_out, mergeable_in)
    return mergeables_out


def distrdf_reducer(results_inout: TaskResult, results_in: TaskResult) -> TaskResult:
    """Two tasks' results as one: their partials added, and their entries - a tree two tasks
    read counted once, the entries each processed added."""
    entries_out, entries_in = results_inout.entries_in_trees, results_in.entries_in_trees
    if entries_out is not None and entries_in is not None:
        entries_out.trees_with_entries.update(entries_in.trees_with_entries)
        entries_out.processed_entries += entries_in.processed_entries
    merged = merge_values(results_inout.mergeables, results_in.mergeables)
    return TaskResult(merged, entries_out)


def _nothing() -> None:
    """No initialization."""


class BaseBackend(ABC):
    """A distributed backend: where tasks run, and what the workers need first.

    ``initialization`` is run before anything else in every task; the
    headers, files and shared libraries registered are sent to every worker,
    and the C++ registered declared there, for every backend.
    """

    initialization: ClassVar[Callable[[], Any]] = staticmethod(_nothing)
    headers: ClassVar[set[str]] = set()
    files: ClassVar[set[str]] = set()
    pcms: ClassVar[set[str]] = set()
    shared_libraries: ClassVar[set[str]] = set()
    strings_to_declare: ClassVar[dict[str, str]] = {}

    @classmethod
    def register_initialization(cls, fun: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Run ``fun(*args, **kwargs)`` first in every task - and here, now."""
        cls.initialization = staticmethod(partial(fun, *args, **kwargs))
        fun(*args, **kwargs)

    @classmethod
    def register_declaration(cls, code_to_declare: str) -> None:
        """Declare C++ here and in every task, once however often it is registered."""
        stripped = code_to_declare.strip()
        guard = "ROOT_RDF_DISTRIBUTED_CODE_" + hashlib.sha256(stripped.encode()).hexdigest()
        if guard not in cls.strings_to_declare:
            cls.strings_to_declare[guard] = f"#ifndef {guard}\n#define {guard}\n{stripped}\n#endif"
        _root().gInterpreter.Declare(cls.strings_to_declare[guard])

    @classmethod
    def register_shared_lib(cls, paths_to_shared_libraries: str | Iterable[str]) -> None:
        libraries, pcms = Utils.register_shared_libs(paths_to_shared_libraries)
        cls.shared_libraries.update(libraries)
        cls.pcms.update(pcms)

    @classmethod
    def register_headers(cls, paths_to_headers: str | Iterable[str]) -> None:
        cls.headers.update(Utils.register_headers(paths_to_headers))

    @classmethod
    def register_files(cls, paths_to_files: str | Iterable[str]) -> None:
        cls.files.update(Utils.register_files(paths_to_files))

    @abstractmethod
    def ProcessAndMerge(
        self,
        ranges: list[Any],
        mapper: Callable[[Any], TaskResult],
        reducer: Callable[[TaskResult, TaskResult], TaskResult],
    ) -> TaskResult:
        """Run ``mapper`` on every range, and add the results up with ``reducer``."""

    @abstractmethod
    def distribute_unique_paths(self, paths: Iterable[str]) -> None:
        """Send these files to every worker."""

    @abstractmethod
    def optimize_npartitions(self) -> int:
        """How many partitions to cut the data into, when the user said none."""

    @abstractmethod
    def make_dataframe(self, *args: Any, **kwargs: Any) -> Any:
        """A distributed frame on this backend."""

    def cleanup_cache(self, exec_id: ExecutionIdentifier) -> None:  # noqa: B027
        """Let go of what the workers kept of a run that is over: nothing, unless the backend
        keeps something."""
