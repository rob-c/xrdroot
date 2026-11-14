"""What distributed ``RDataFrame`` asks of a frame: a window of the data, and partial results.

ROOT's distributed ``RDataFrame`` cuts the data into ranges of entries and
runs the whole computation graph over each range in a task of its own; the
tasks' results are then added up, and only the sum is read. This is the
part of that a frame does:

- a :class:`Window` is a source showing only the entries from one number to
  another of the data beneath it - which a task's frame is made over, and
  which the next task in the same process points at its own range and data
  rather than building the graph again, as ROOT's ``ChangeEmptyEntryRange``
  and ``ChangeSpec`` do;
- :func:`trigger` runs a frame's loop over what has been booked since the
  last, leaving each result's :class:`Partial` - its accumulator, not yet
  finished - on it, and :class:`Partial` adds two of them up and finishes the
  sum, as ROOT's ``RMergeableValue`` is merged and read;
- :func:`rebook` books a result again on the same frame, for the next range,
  as ROOT's ``CloneResultAndAction`` does.

Adding accumulators rather than results is what keeps a ``Mean`` or a
``StdDev`` exact: each range's sums and counts are added, never its means.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, cast

from ..file import open_root
from ..rntuple import RNTuple
from . import actions
from .frame import RDataFrame, Result, _Reopened
from .graph import Definition
from .loop import Plan, accumulate
from .report import CutFlowReport, CutInfo
from .snapshot import Snapshot
from .sources import SampleInfo, Source

__all__ = [
    "Window",
    "Partial",
    "trigger",
    "partial_of",
    "rebook",
    "source_of",
    "clusters_and_entries",
]

#: The actions whose partials are lists of batches, added up by putting the lists end to end.
LISTED = (actions.Take, actions.AsNumpy, actions.FillObject)

#: The actions a task finishes itself: what a ``Snapshot`` writes is closed where it was
#: written, and a ``Report`` counts the filters that task's loop saw.
FINISHED_IN_TASK = (Snapshot, actions.Report)


class Window(Source):
    """The entries from ``start`` up to ``stop`` of ``inner``, numbered as ``inner`` numbers them.

    Each file of ``inner`` stays where it is among the boundaries, cut to the
    window - an empty span where it is outside it - so that which file an
    entry is from, and ``rdfentry_``, are what they are in the whole data.
    """

    def __init__(self, inner: Source, start: int, stop: int) -> None:
        self.inner = inner
        self.start = start
        self.stop = stop

    def change(self, inner: Source, start: int, stop: int) -> None:
        """Show another range, of the same data or of other data with the same columns."""
        self.inner, self.start, self.stop = inner, start, stop

    def __len__(self) -> int:
        return len(self.inner)

    def names(self) -> list[str]:
        return self.inner.names()

    def readable(self) -> list[str]:
        return self.inner.readable()

    def read(self, name: str, start: int, stop: int) -> Any:
        return self.inner.read(name, start, stop)

    def available(self, name: str, start: int, stop: int) -> Any:
        return self.inner.available(name, start, stop)

    def boundaries(self) -> list[int]:
        return [max(self.start, min(bound, self.stop)) for bound in self.inner.boundaries()]

    def sample(self, index: int) -> SampleInfo:
        return self.inner.sample(index)

    def cxx_type(self, name: str) -> str | None:
        return self.inner.cxx_type(name)

    def describe(self) -> str:
        return f"entries {self.start} to {self.stop} of {self.inner.describe()}"

    def root_description(self) -> str:
        return self.inner.root_description()

    def close(self) -> None:
        if self.inner.owned:
            self.inner.close()


class _Named(Definition):
    """A column only by name: all a partial's action needs of its inputs to be finished."""

    __slots__ = ()


def _light(action: actions.Action) -> actions.Action:
    """The action without its graph - nodes, expressions, a script's callables - which
    finishing and adding up its partials never use, and which need not travel."""
    made = copy.copy(action)
    made.node = None  # type: ignore[assignment]
    made.inputs = [_Named(each.name) for each in action.inputs]
    return made


class Partial:
    """``RMergeableValue``: what one task's loop made of a result, to be added to others'.

    A ``Snapshot`` or ``Report`` is finished in the task, and kept as its
    value; any other result is kept as its accumulator, added up with
    :meth:`Merge` and finished by :meth:`GetValue`.
    """

    __slots__ = ("action", "acc", "finished", "value")

    def __init__(self, action: actions.Action, acc: Any, flow: dict[int, tuple[int, int]]) -> None:
        self.finished = isinstance(action, FINISHED_IN_TASK)
        self.value = action.finish(acc, flow) if self.finished else None
        self.action = _light(action)
        self.acc = None if self.finished else acc

    def __repr__(self) -> str:
        return f"<Partial {self.action.kind}>"

    def Merge(self, other: Partial) -> None:
        """Add ``other``'s partial into this one."""
        if isinstance(self.value, CutFlowReport):
            self.value = _reports(self.value, cast(CutFlowReport, other.value))
        elif self.finished:
            raise TypeError(
                f"a distributed {self.action.kind} is finished in each task, and gathered "
                f"as it is rather than added up"
            )
        else:
            self.acc = _added(self.action, self.acc, other.acc)

    def GetValue(self) -> Any:
        """The result of everything added up so far."""
        if self.finished:
            return self.value
        return self.action.finish(self.acc, {})


def _added(action: actions.Action, first: Any, second: Any) -> Any:
    if first is None or second is None:
        return second if first is None else first
    if isinstance(action, LISTED):
        return [*first, *second]
    return action.add(first, second)


def _reports(first: CutFlowReport, second: CutFlowReport) -> CutFlowReport:
    """Two tasks' cut flows of the same filters, their counts added."""
    return CutFlowReport([
        CutInfo(mine.name, mine.passed + theirs.passed, mine.all + theirs.all)
        for mine, theirs in zip(first, second, strict=True)
    ])  # fmt: skip


def source_of(frame: RDataFrame) -> Source:
    """The data a frame's whole graph reads."""
    return frame._graph.source


def trigger(frame: RDataFrame) -> None:
    """Run the loop of ``frame``'s graph over every result booked since the last, once, in this
    process, leaving each result's :class:`Partial` on it for :func:`partial_of`."""
    graph = frame._graph
    pending, graph.pending = graph.pending, []
    plan = Plan(graph.source, [result._action for result in pending], graph.filters,
                graph.step, 1)  # fmt: skip
    accs, flow = accumulate(plan)
    graph.runs += 1
    for result, acc in zip(pending, accs, strict=True):
        result._partial = Partial(result._action, acc, flow)


def partial_of(result: Any) -> Partial:
    """The partial :func:`trigger` left on a result, or on a ``Snapshot``'s result."""
    if isinstance(result, _Reopened):
        result = result._written
    found = result._partial
    if found is None:
        raise RuntimeError(
            "this result has no partial: its frame's loop has not run as a distributed "
            "task's since it was booked"
        )
    return found  # type: ignore[no-any-return]


def rebook(result: Any, filename: str | None = None) -> Any:
    """``CloneResultAndAction``: the same result booked again on the same frame - a
    ``Snapshot``'s to write into ``filename`` this time."""
    if isinstance(result, _Reopened):
        written = rebook(result._written, filename)
        return _Reopened(written, result._graph, result._columns)
    action = result._action
    if filename is not None:
        action = copy.copy(action)
        action.filename = filename
    booked: Result = result._graph.book(action)
    return booked


def clusters_and_entries(name: str, filename: str) -> tuple[list[int], int]:
    """``GetClustersAndEntries``: where each cluster of a tree or RNTuple begins, then where
    the last one ends - ``[0, ..., entries]`` - and how many entries it has."""
    with open_root(filename) as file:
        found = file[name]
        if isinstance(found, RNTuple):
            entries = len(found)
            starts = [cluster.first_entry for cluster in found._store.covering(0, entries)]
            return _bounds(starts, entries), entries
        entries = len(found)
        return _bounds([start for start, _ in found.clustering.ranges()], entries), entries


def _bounds(starts: Sequence[int], entries: int) -> list[int]:
    return sorted({0, *starts, entries})
