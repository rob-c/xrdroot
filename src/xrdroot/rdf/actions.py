"""The results a frame can be asked for, each as a partial per batch and a way to add them up.

Every action turns one batch into a *partial* - a count, a sum, a histogram
of just those entries, a run of values - in :meth:`Action.partial`, which is
what a worker process computes, and adds partials up in task order in
:meth:`Action.merge`, which is what the process that booked it does. A
partial is always made the same way from the same batch and always merged in
the same order, so the result is the same to the last bit however many
processes made the partials.

A column that is a collection per entry is taken element by element, as
ROOT's actions take an ``RVec``: ``Sum`` of ``jet_pt`` adds every jet,
``Histo1D`` of it fills one entry per jet, and a weight that is one number
per entry goes with each of its entry's elements.
"""

from __future__ import annotations

import copy
import functools
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..graph import Graph as TGraph
from ..hist import Histogram
from ..tree import concatenate, take
from .graph import Definition, Selector, chain_of, named_filters
from .report import CutFlowReport, CutInfo, Display
from .values import align, from_column, no_text

if TYPE_CHECKING:
    from .loop import Batch

__all__ = [
    "Action",
    "Count",
    "Sum",
    "Mean",
    "Extreme",
    "StdDev",
    "Stats",
    "Histo",
    "AutoHisto",
    "GraphAction",
    "Take",
    "AsNumpy",
    "Reduce",
    "Aggregate",
    "Foreach",
    "DisplayAction",
    "Report",
]

Array = Any


def elements(name: str, values: Any, what: str) -> Array:
    """A column's values one element at a time: a collection per entry is flattened."""
    value = from_column(name, values)
    if value.items is not None:
        raise ValueError(f"{what} takes one column, and {name!r} is several collections")
    data = np.asarray(value.data)
    no_text(data.dtype, what)
    return data


def joined(pieces: Sequence[Any]) -> Any:
    """Several batches of one column as one: what Combinations gives, member by member."""
    if isinstance(pieces[0], tuple):
        return tuple(concatenate(list(member)) for member in zip(*pieces))
    return concatenate(pieces)


class Action:
    """One booked result."""

    #: What ROOT calls it, for messages.
    kind = "action"
    #: Whether its partials may be computed in worker processes.
    parallel = True

    def __init__(self, node: Selector, inputs: Sequence[Definition]) -> None:
        self.node = node
        self.inputs = list(inputs)

    def columns(self, batch: Batch) -> list[Any]:
        return [batch.column(each, self.node) for each in self.inputs]

    def start(self) -> Any:
        return None

    def partial(self, batch: Batch) -> Any:
        raise NotImplementedError

    def merge(self, acc: Any, part: Any) -> Any:
        return part if acc is None else self.add(acc, part)

    def add(self, acc: Any, part: Any) -> Any:
        raise NotImplementedError

    def done(self, acc: Any) -> bool:
        """Is the result complete already, so the loop need go no further for it?"""
        return False

    def finish(self, acc: Any, flow: dict[int, tuple[int, int]]) -> Any:
        return acc

    def abort(self, acc: Any) -> None:
        """The loop failed: let go of anything the result holds on to."""
        return None


class Count(Action):
    """``Count``: how many entries reach the node."""

    kind = "Count"

    def partial(self, batch: Batch) -> Any:
        return batch.size(self.node)

    def add(self, acc: Any, part: Any) -> Any:
        return acc + part


def _exact_sum(data: Array) -> int | float:
    """A batch's sum: exactly, of integers; in ``double``, of anything else."""
    if data.dtype.kind in "biu":
        kind = np.uint64 if data.dtype.kind == "u" else np.int64
        return int(np.sum(data, dtype=kind))
    return float(np.sum(data, dtype=np.float64))


class Sum(Action):
    """``Sum``: every value - every element, of a collection - added up.

    Integers are added exactly, and anything else in ``double``: each
    batch's sum, then the batches' sums added with :func:`math.fsum`, so the
    total does not depend on how the entries were cut into batches.
    """

    kind = "Sum"

    def partial(self, batch: Batch) -> Any:
        (values,) = self.columns(batch)
        return [_exact_sum(elements(self.inputs[0].name, values, self.kind))]

    def add(self, acc: Any, part: Any) -> Any:
        return acc + part

    def finish(self, acc: Any, flow: dict[int, tuple[int, int]]) -> Any:
        if all(isinstance(each, int) for each in acc):
            return sum(acc)
        return math.fsum(acc)


class Mean(Action):
    """``Mean``: the average of every value, in ``double``; zero when there is none."""

    kind = "Mean"

    def partial(self, batch: Batch) -> Any:
        (values,) = self.columns(batch)
        data = elements(self.inputs[0].name, values, self.kind).astype(np.float64)
        return [float(np.sum(data))], len(data)

    def add(self, acc: Any, part: Any) -> Any:
        return acc[0] + part[0], acc[1] + part[1]

    def finish(self, acc: Any, flow: dict[int, tuple[int, int]]) -> Any:
        sums, count = acc
        return math.fsum(sums) / count if count else 0.0


def _limit(dtype: Any, top: bool) -> Any:
    """What ROOT's ``Min`` (``Max``) starts from: the largest (lowest) of the type."""
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        info: Any = np.finfo(dtype)
    else:
        info = np.iinfo(np.int32 if dtype.kind == "b" else dtype)
    return np.asarray(info.max if top else info.min).item()


class Extreme(Action):
    """``Min`` and ``Max``: of every value, in the column's own type."""

    def __init__(self, node: Selector, inputs: Sequence[Definition], largest: bool) -> None:
        super().__init__(node, inputs)
        self.largest = largest
        self.kind = "Max" if largest else "Min"

    def partial(self, batch: Batch) -> Any:
        (values,) = self.columns(batch)
        data = elements(self.inputs[0].name, values, self.kind)
        found = None
        if len(data):
            found = (np.max(data) if self.largest else np.min(data)).item()
        return found, data.dtype

    def add(self, acc: Any, part: Any) -> Any:
        if acc[0] is None or part[0] is None:
            return acc if part[0] is None else part
        pick = max if self.largest else min
        return pick(acc[0], part[0]), acc[1]

    def finish(self, acc: Any, flow: dict[int, tuple[int, int]]) -> Any:
        value, dtype = acc
        return _limit(dtype, not self.largest) if value is None else value


def _moments(data: Array) -> tuple[int, float, float]:
    count = len(data)
    mean = float(np.mean(data)) if count else 0.0
    return count, mean, float(np.sum((data - mean) ** 2)) if count else 0.0


def _joined(first: tuple[int, float, float], second: tuple[int, float, float]) -> Any:
    """Chan's rule: the count, mean and summed squared deviation of two samples as one."""
    (na, ma, sa), (nb, mb, sb) = first, second
    total = na + nb
    if not total:
        return first
    delta = mb - ma
    return total, ma + delta * nb / total, sa + sb + delta * delta * na * nb / total


class StdDev(Action):
    """``StdDev``: the spread of every value about their mean, over ``n - 1``."""

    kind = "StdDev"

    def partial(self, batch: Batch) -> Any:
        (values,) = self.columns(batch)
        return _moments(elements(self.inputs[0].name, values, self.kind).astype(np.float64))

    def add(self, acc: Any, part: Any) -> Any:
        return _joined(acc, part)

    def finish(self, acc: Any, flow: dict[int, tuple[int, int]]) -> Any:
        count, _, squares = acc
        return math.sqrt(squares / (count - 1)) if count > 1 else 0.0


class Statistic:
    """What ``Stats`` gives: ROOT's ``TStatistic`` of a column, weighted or not."""

    def __init__(self, name: str, parts: Sequence[tuple[Array, Array]]) -> None:
        values = np.concatenate([part[0] for part in parts]).astype(np.float64)
        weights = np.concatenate([part[1] for part in parts]).astype(np.float64)
        self.name = name
        self.n = len(values)
        self.sum_w = float(np.sum(weights))
        self.sum_w2 = float(np.sum(weights * weights))
        self.mean = float(np.sum(weights * values) / self.sum_w) if self.sum_w else 0.0
        self.m2 = float(np.sum(weights * (values - self.mean) ** 2))
        self.min = float(np.min(values)) if self.n else math.inf
        self.max = float(np.max(values)) if self.n else -math.inf

    def __repr__(self) -> str:
        return f"<Statistic {self.name!r}: n={self.n} mean={self.mean} rms={self.GetRMS()}>"

    def GetN(self) -> int:
        return self.n

    def GetW(self) -> float:
        return self.sum_w

    def GetW2(self) -> float:
        return self.sum_w2

    def GetMean(self) -> float:
        return self.mean

    def GetVar(self) -> float:
        """The variance, over the effective number of entries less one, as ``TStatistic``."""
        if self.n < 2 or not self.sum_w:
            return 0.0
        return self.m2 / self.sum_w * self.n / (self.n - 1)

    def GetRMS(self) -> float:
        return math.sqrt(self.GetVar())

    def GetMeanErr(self) -> float:
        effective = self.sum_w * self.sum_w / self.sum_w2 if self.sum_w2 else 0.0
        return self.GetRMS() / math.sqrt(effective) if effective else 0.0

    def GetMin(self) -> float:
        return self.min

    def GetMax(self) -> float:
        return self.max


class Stats(Action):
    """``Stats``: count, mean, spread and range of a column, weighted by another if given."""

    kind = "Stats"

    def partial(self, batch: Batch) -> Any:
        found = [
            from_column(each.name, values) for each, values in zip(self.inputs, self.columns(batch))
        ]
        arrays, _ = align(found, self.kind)
        weights = arrays[1] if len(arrays) > 1 else np.ones(len(arrays[0]))
        return [(arrays[0], np.broadcast_to(weights, arrays[0].shape))]

    def add(self, acc: Any, part: Any) -> Any:
        return acc + part

    def finish(self, acc: Any, flow: dict[int, tuple[int, int]]) -> Any:
        return Statistic(self.inputs[0].name, acc)


def _aligned(action: Action, batch: Batch) -> list[Array]:
    """An action's columns lined up element by element, as a histogram is filled from them."""
    found = [
        from_column(each.name, values) for each, values in zip(action.inputs, action.columns(batch))
    ]
    arrays, _ = align(found, action.kind)
    return arrays


class Histo(Action):
    """``Histo1D``/``2D``/``3D`` and ``Profile1D``/``2D``: a booked model, filled.

    Each batch fills a copy of the model, and the copies are merged in
    order as ``hadd`` merges them, so every sum is ROOT's.
    """

    def __init__(
        self,
        node: Selector,
        inputs: Sequence[Definition],
        model: Histogram,
        weighted: bool,
        kind: str,
    ) -> None:
        super().__init__(node, inputs)
        self.model = model
        self.weighted = weighted
        self.kind = kind

    def partial(self, batch: Batch) -> Any:
        arrays = _aligned(self, batch)
        made = self.model.copy()
        weight = arrays.pop() if self.weighted else None
        made.fill(*arrays, weight=weight)
        return made

    def add(self, acc: Any, part: Any) -> Any:
        return type(acc).merge([acc, part])


#: How many bins ``Histo1D`` books when it is given no model, as ROOT books 128.
AUTO_BINS = 128


class AutoHisto(Action):
    """``Histo1D`` of a column alone: 128 bins, spanning every value it was filled with.

    The values are kept until the end, when the range is known, and then
    filled in the order they came, as ROOT fills its buffer.
    """

    kind = "Histo1D"

    def __init__(self, node: Selector, inputs: Sequence[Definition], weighted: bool) -> None:
        super().__init__(node, inputs)
        self.weighted = weighted

    def partial(self, batch: Batch) -> Any:
        return [_aligned(self, batch)]

    def add(self, acc: Any, part: Any) -> Any:
        return acc + part

    def finish(self, acc: Any, flow: dict[int, tuple[int, int]]) -> Any:
        values = np.concatenate([part[0] for part in acc]).astype(np.float64)
        weights = np.concatenate([part[1] for part in acc]) if self.weighted else None
        finite = values[np.isfinite(values)]
        low, high = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 0.0)
        if low == high:
            low, high = low - 1.0, high + 1.0
        name = self.inputs[0].name
        made = Histogram.book(name, (AUTO_BINS, low, float(np.nextafter(high, np.inf))), title=name)
        made.fill(values, weight=weights)
        return made


class GraphAction(Action):
    """``Graph``: a point per entry - per element, of collections - in the order they came."""

    kind = "Graph"

    def partial(self, batch: Batch) -> Any:
        return [_aligned(self, batch)]

    def add(self, acc: Any, part: Any) -> Any:
        return acc + part

    def finish(self, acc: Any, flow: dict[int, tuple[int, int]]) -> Any:
        xs = np.concatenate([part[0] for part in acc])
        ys = np.concatenate([part[1] for part in acc])
        return TGraph.new("Graph", xs, ys, title="Graph")


class Take(Action):
    """``Take``: a column's values, entry by entry, as one array, ``Jagged`` or list."""

    kind = "Take"

    def partial(self, batch: Batch) -> Any:
        return self.columns(batch)[0]

    def merge(self, acc: Any, part: Any) -> Any:
        return [part] if acc is None else [*acc, part]

    def finish(self, acc: Any, flow: dict[int, tuple[int, int]]) -> Any:
        return joined(acc)


class AsNumpy(Action):
    """``AsNumpy``: several columns at once, as a dict of what ``Take`` would give each."""

    kind = "AsNumpy"

    def partial(self, batch: Batch) -> Any:
        return self.columns(batch)

    def merge(self, acc: Any, part: Any) -> Any:
        return [part] if acc is None else [*acc, part]

    def finish(self, acc: Any, flow: dict[int, tuple[int, int]]) -> Any:
        return {
            each.name: joined([part[at] for part in acc]) for at, each in enumerate(self.inputs)
        }


class Reduce(Action):
    """``Reduce``: every value folded together by a function of two, from ``init``.

    A NumPy ufunc - ``np.add``, ``np.maximum`` - reduces each batch in C;
    any other function is called value by value, as ROOT calls it. Each batch
    starts from ``init``, as each of ROOT's threads does, so it should be the
    function's identity.
    """

    kind = "Reduce"

    def __init__(
        self,
        node: Selector,
        inputs: Sequence[Definition],
        function: Callable[[Any, Any], Any],
        init: Any,
    ) -> None:
        super().__init__(node, inputs)
        self.function = function
        self.init = init

    def partial(self, batch: Batch) -> Any:
        (values,) = self.columns(batch)
        data = elements(self.inputs[0].name, values, self.kind)
        if isinstance(self.function, np.ufunc):
            return self.function.reduce(data, initial=self.init)
        return functools.reduce(self.function, data.tolist(), self.init)

    def add(self, acc: Any, part: Any) -> Any:
        return self.function(acc, part)


class Aggregate(Action):
    """``Aggregate``: each batch folded into ``init`` by ``aggregator``, batches by ``merger``.

    ``aggregator(accumulated, values)`` is called once per batch, with a
    fresh copy of ``init`` and the batch's values as an array - the whole
    batch, not an entry at a time - and gives back what it made;
    ``merger(first, second)`` adds two of those up.
    """

    kind = "Aggregate"

    def __init__(
        self,
        node: Selector,
        inputs: Sequence[Definition],
        aggregator: Callable[[Any, Any], Any],
        merger: Callable[[Any, Any], Any],
        init: Any,
    ) -> None:
        super().__init__(node, inputs)
        self.aggregator = aggregator
        self.merger = merger
        self.init = init

    def partial(self, batch: Batch) -> Any:
        return self.aggregator(copy.deepcopy(self.init), *self.columns(batch))

    def add(self, acc: Any, part: Any) -> Any:
        return self.merger(acc, part)


class Foreach(Action):
    """``Foreach`` and ``ForeachSlot``: a callable run on every batch, for what it does."""

    kind = "Foreach"
    parallel = False

    def __init__(
        self, node: Selector, inputs: Sequence[Definition], function: Callable[..., Any], slot: bool
    ) -> None:
        super().__init__(node, inputs)
        self.function = function
        self.slot = slot

    def partial(self, batch: Batch) -> Any:
        columns = self.columns(batch)
        self.function(*([batch.slot] if self.slot else []), *columns)
        return None

    def merge(self, acc: Any, part: Any) -> Any:
        return None


def _entries(values: Any) -> int:
    """How many entries a column's batch holds; of what Combinations gives, its first member's."""
    return len(values[0]) if isinstance(values, tuple) else len(values)


def _first(values: Any, wanted: Array) -> Any:
    if isinstance(values, tuple):
        return tuple(take(member, wanted) for member in values)
    return take(values, wanted)


class DisplayAction(Action):
    """``Display``: the first entries of some columns, kept until there are enough."""

    kind = "Display"

    def __init__(
        self, node: Selector, inputs: Sequence[Definition], rows: int, elements: int
    ) -> None:
        super().__init__(node, inputs)
        self.rows = rows
        self.elements = elements

    def partial(self, batch: Batch) -> Any:
        wanted = np.arange(min(batch.size(self.node), self.rows + 1))
        return [_first(values, wanted) for values in self.columns(batch)]

    def _length(self, acc: Any) -> int:
        return sum(_entries(part[0]) for part in acc) if acc and self.inputs else 0

    def merge(self, acc: Any, part: Any) -> Any:
        acc = [] if acc is None else acc
        return acc if self.done(acc) else [*acc, part]

    def done(self, acc: Any) -> bool:
        return acc is not None and self._length(acc) > self.rows

    def finish(self, acc: Any, flow: dict[int, tuple[int, int]]) -> Any:
        columns = {
            each.name: joined([part[at] for part in acc]) for at, each in enumerate(self.inputs)
        }
        shown = min(self._length(acc), self.rows)
        return Display(columns, shown, self.elements, self._length(acc) > self.rows)


class Report(Action):
    """``Report``: the named filters' counts, up to the node - or all of them, at the root."""

    kind = "Report"

    def __init__(self, node: Selector, every: Callable[[], list[Any]] | None) -> None:
        super().__init__(node, [])
        #: At the root, how to find every named filter of the graph when the loop is done.
        self.every = every

    def partial(self, batch: Batch) -> Any:
        batch.rows(self.node)
        return None

    def merge(self, acc: Any, part: Any) -> Any:
        return None

    def finish(self, acc: Any, flow: dict[int, tuple[int, int]]) -> Any:
        filters = self.every() if self.every is not None else named_filters(chain_of(self.node))
        cuts = []
        for node in filters:
            seen, passed = flow.get(id(node), (0, 0))
            cuts.append(CutInfo(node.name, passed, seen))
        return CutFlowReport(cuts)

    def __getstate__(self) -> dict[str, Any]:
        return {"node": self.node, "inputs": [], "every": None}
