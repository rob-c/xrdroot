"""The event loop: one pass over the data that computes every booked result at once.

The entries are cut into *tasks* - ``step`` consecutive entries each, never
straddling two files - and a task is one batch: every column any result
needs is read for it once, every filter is evaluated once over its whole
batch, and every defined column is computed once, for exactly the entries
that reach the node it was defined at. Each result turns a batch into a
*partial* - a histogram of that batch, a sum, a run of values - and the
partials are added up in the order of the tasks.

That order is what makes the answer the same however many processes compute
it. With ``workers=1`` the tasks run here, one after another. With more,
they run in a pool of worker processes - each opening the files again for
itself, as a chain or a tree pickled from its URL does - and come back in
task order, where they are added up exactly as they would have been here:
the same partials, merged the same way, give the same bits.

What depends on the order entries arrive in cannot be split up like that: a
``Range`` counts the entries that reached it, and a ``Foreach`` runs for its
side effects, which a worker process would have on its own copy of the
world. Those are refused with more than one worker, by name.
"""

from __future__ import annotations

import importlib
import multiprocessing
import pickle
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from ..tree import Jagged, take
from .expression import Expression, Scope
from .graph import (
    Defined,
    Definition,
    EntryColumn,
    Filter,
    PerSample,
    Range,
    Selector,
    SlotColumn,
    SourceColumn,
    chain_of,
)
from .sources import SampleInfo, Source
from .values import from_column, to_column, truth

__all__ = ["Batch", "Plan", "execute", "run_task"]

Array = Any

#: Entries per task, when a frame is not told otherwise.
DEFAULT_STEP = 100_000


def _positions(inner: Array | None, outer: Array | None) -> Array | None:
    """Where the rows of a node sit among the rows of a node above it."""
    if outer is None:
        return inner
    assert inner is not None  # a node below never lets through more than one above it
    return np.searchsorted(outer, inner)


def _checked(name: str, out: Any, count: int) -> Any:
    """What a callable gave for a batch, if it is one value (or collection) per entry."""
    sized = isinstance(out, (Jagged, list)) or (isinstance(out, np.ndarray) and out.ndim > 0)
    if sized and len(out) == count:
        return out
    if isinstance(out, tuple) and all(isinstance(item, Jagged) for item in out):
        return out
    size = f"{len(out)} values" if hasattr(out, "__len__") else f"a {type(out).__name__}"
    raise ValueError(
        f"{name} is given a batch of {count} entries at a time and has to give back one "
        f"value per entry - an array, a Jagged or a list of {count} - and gave {size}"
    )


class Batch:
    """One task's entries, and everything worked out for them so far."""

    def __init__(
        self,
        source: Source,
        span: tuple[int, int],
        slot: int,
        sample: SampleInfo | None,
        ranges: dict[int, int],
    ) -> None:
        self.source = source
        self.start, self.stop = span
        self.count = self.stop - self.start
        self.slot = slot
        self.sample = sample
        #: How many entries each ``Range`` has seen, carried from one batch to the next.
        self.ranges = ranges
        #: Each named filter's entries seen and passed in this batch.
        self.counts: dict[int, tuple[int, int]] = {}
        self._read: dict[str, Any] = {}
        self._rows: dict[int, Array | None] = {}
        self._values: dict[tuple[int, int], Any] = {}

    # -- which entries reach a node --------------------------------------------------

    def rows(self, node: Selector) -> Array | None:
        """The rows of the batch that reach ``node``; ``None`` for every row."""
        key = id(node)
        if key not in self._rows:
            self._rows[key] = self._select(node)
        return self._rows[key]

    def size(self, node: Selector) -> int:
        rows = self.rows(node)
        return self.count if rows is None else len(rows)

    def entries(self, node: Selector) -> Array:
        """The entry numbers of the rows that reach ``node``."""
        numbers = np.arange(self.start, self.stop, dtype=np.int64)
        rows = self.rows(node)
        return numbers if rows is None else numbers[rows]

    def _select(self, node: Selector) -> Array | None:
        if isinstance(node, Filter):
            return self._filter(node)
        if isinstance(node, Range):
            return self._range(node)
        return None

    def _kept(self, parent: Selector, keep: Array) -> Array | None:
        rows = self.rows(parent)
        if keep.all():
            return rows
        return np.flatnonzero(keep) if rows is None else rows[keep]

    def _filter(self, node: Filter) -> Array | None:
        assert node.parent is not None
        seen = self.size(node.parent)
        keep = self._condition(node, seen) if seen else np.zeros(0, np.bool_)
        self.counts[id(node)] = (seen, int(np.count_nonzero(keep)))
        return self._kept(node.parent, keep)

    def _condition(self, node: Filter, seen: int) -> Array:
        assert node.parent is not None
        if isinstance(node.condition, Expression):
            value = node.condition.evaluate(self._scope(node.inputs, node.parent))
            if value.offsets is not None or value.items is not None:
                raise ValueError(
                    f"Filter({node.condition.text!r}) gives a collection per entry - an RVec "
                    f"- rather than one answer; reduce it to one, with Any(...), All(...), "
                    f"Sum(...) or .size()"
                )
            data = value.data
        else:
            columns = [self.column(each, node.parent) for each in node.inputs.values()]
            data = _checked("a Filter's callable", node.condition(*columns), seen)
        return np.broadcast_to(truth(data, "Filter"), (seen,))

    def _range(self, node: Range) -> Array | None:
        assert node.parent is not None
        seen = self.size(node.parent)
        before = self.ranges.get(id(node), 0)
        self.ranges[id(node)] = before + seen
        counted = before + np.arange(seen, dtype=np.int64)
        keep = (counted >= node.begin) & ((counted - node.begin) % node.stride == 0)
        if node.end is not None:
            keep &= counted < node.end
        return self._kept(node.parent, keep)

    # -- the columns ---------------------------------------------------------------

    def column(self, definition: Definition, node: Selector) -> Any:
        """One column's values for the rows that reach ``node``."""
        key = (id(definition), id(node))
        if key not in self._values:
            self._values[key] = self._compute(definition, node)
        return self._values[key]

    def _scope(self, inputs: dict[str, Definition], node: Selector) -> Scope:
        def fetch(name: str) -> Any:
            return from_column(name, self.column(inputs[name], node))

        return Scope(fetch, self.size(node), self.entries(node))

    def _compute(self, definition: Definition, node: Selector) -> Any:
        if isinstance(definition, Defined):
            return self._defined(definition, node)
        count = self.size(node)
        if isinstance(definition, SourceColumn):
            return self._source(definition.name, node, count)
        if isinstance(definition, EntryColumn):
            return self.entries(node).astype(np.uint64)
        if isinstance(definition, SlotColumn):
            return np.full(count, self.slot, np.uint32)
        assert isinstance(definition, PerSample)
        value = np.asarray(definition.compute(self.sample))
        return np.full(count, value, dtype=value.dtype)

    def _source(self, name: str, node: Selector, count: int) -> Any:
        if not count:  # nothing reaches here: the right type, for nothing read
            return self.source.read(name, 0, 0)  # at the start, so no basket is touched
        if name not in self._read:
            self._read[name] = self.source.read(name, self.start, self.stop)
        rows = self.rows(node)
        return self._read[name] if rows is None else take(self._read[name], rows)

    def _defined(self, definition: Defined, node: Selector) -> Any:
        home = definition.selector
        if node is not home:
            values = self.column(definition, home)
            where = _positions(self.rows(node), self.rows(home))
            return values if where is None else take(values, where)
        count = self.size(home)
        if isinstance(definition.compute, Expression):
            value = definition.compute.evaluate(self._scope(definition.inputs, home))
            return to_column(value, count)
        columns = [self.column(each, home) for each in definition.inputs.values()]
        name = f"the callable defining {definition.name!r}"
        return _checked(name, definition.compute(*columns), count)


class Plan:
    """One event loop's work: where the entries come from, and what to make of them."""

    def __init__(
        self,
        source: Source,
        actions: Sequence[Any],
        filters: Sequence[Filter],
        step: int,
        workers: int,
    ) -> None:
        self.source = source
        self.actions = list(actions)
        #: The named filters, counted for ``Report`` whether a result needs them or not.
        self.filters = list(filters)
        self.step = step
        self.workers = workers

    def __getstate__(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "actions": self.actions,
            "filters": self.filters,
            "step": self.step,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(  # type: ignore[misc]
            state["source"], state["actions"], state["filters"], state["step"], 1
        )

    def leaves(self) -> list[Selector]:
        return [action.node for action in self.actions] + list(self.filters)

    def limit(self) -> int | None:
        """How far into the data any result needs to read, when a ``Range`` says."""
        bounds = []
        for leaf in self.leaves():
            chain = chain_of(leaf)
            first = chain[1] if len(chain) > 1 else None
            if not isinstance(first, Range) or first.end is None:
                return None
            bounds.append(first.end)
        return max(bounds, default=None)

    def tasks(self) -> list[tuple[int, int, int]]:
        """Every task's first entry, one past its last, and the file it is in."""
        bounds = self.source.boundaries()
        limit = self.limit()
        stop = bounds[-1] if limit is None else min(bounds[-1], limit)
        found = []
        for index, (low, high) in enumerate(zip(bounds, bounds[1:])):
            for at in range(low, min(high, stop), self.step):
                found.append((at, min(at + self.step, high, stop), index))
        return found or [(0, 0, 0)]

    def ranges(self) -> list[Range]:
        return [
            node for leaf in self.leaves() for node in chain_of(leaf) if isinstance(node, Range)
        ]


def run_task(
    plan: Plan, task: tuple[int, int, int], slot: int = 0, ranges: dict[int, int] | None = None
) -> tuple[list[Any], list[tuple[int, int]]]:
    """One batch: each result's partial, and each named filter's counts."""
    start, stop, index = task
    sample = plan.source.sample(index) if stop > start else None
    batch = Batch(plan.source, (start, stop), slot, sample, {} if ranges is None else ranges)
    for node in plan.filters:
        batch.rows(node)
    parts = [action.partial(batch) for action in plan.actions]
    return parts, [batch.counts.get(id(node), (0, 0)) for node in plan.filters]


def _exhausted(plan: Plan, ranges: dict[int, int], accs: list[Any]) -> bool:
    """Is nothing left to do: every result filled, or past the end of a ``Range`` above it?"""
    for number, leaf in enumerate(plan.leaves()):
        spans = [node for node in chain_of(leaf) if isinstance(node, Range)]
        over = any(node.end is not None and ranges.get(id(node), 0) >= node.end for node in spans)
        done = number < len(accs) and plan.actions[number].done(accs[number])
        if not (over or done):
            return False
    return True


def _serial(plan: Plan, tasks: list[tuple[int, int, int]], accs: list[Any]) -> Iterator[Any]:
    ranges: dict[int, int] = {}
    for task in tasks:
        yield run_task(plan, task, 0, ranges)
        if _exhausted(plan, ranges, accs):
            return


#: The plan a worker process runs, and the slot it is, once it has been started.
_INSTALLED: list[Any] = []


def install(payload: bytes, slots: Any) -> None:
    """Start a worker: unpack the plan it runs, and take a slot number of its own."""
    _INSTALLED[:] = [pickle.loads(payload), slots.get()]


def work(task: tuple[int, int, int]) -> tuple[list[Any], list[tuple[int, int]]]:
    """One task, in a worker process."""
    plan, slot = _INSTALLED
    return run_task(plan, task, slot)


def _refuse_ordered(plan: Plan) -> None:
    if plan.ranges():
        raise UnsupportedFeatureError(
            "Range counts the entries that reach it in the order they arrive, which "
            "workers processing parts of the data side by side cannot do; run this frame "
            "with workers=1, as ROOT refuses Range with implicit multithreading"
        )
    for action in plan.actions:
        if not action.parallel:
            raise UnsupportedFeatureError(
                f"{action.kind} is run for what it does rather than for a result, and in a "
                f"worker process it would do it to that process's copy of everything; run "
                f"with workers=1, or use Aggregate, whose result comes back"
            )


def _payload(plan: Plan) -> bytes:
    try:
        return pickle.dumps(plan)
    except Exception as why:
        raise UnsupportedFeatureError(
            f"workers={plan.workers} sends the frame's work to other processes, and it "
            f"could not be sent: {why}. A lambda, or a function defined inside another, "
            f"cannot be; define callables at the top level of a module, use string "
            f"expressions, or run with workers=1"
        ) from None


def _parallel(plan: Plan, tasks: list[tuple[int, int, int]]) -> Iterator[Any]:
    _refuse_ordered(plan)
    payload = _payload(plan)
    context = multiprocessing.get_context()
    slots = context.Queue()
    for slot in range(plan.workers):
        slots.put(slot)
    # Looked up as the pool is made rather than when this module was: a process
    # started with ``spawn`` pickles the pool's worker function by its module's
    # name, and a harness that imports that module afresh - coverage under
    # pytest-xdist does - would otherwise leave us holding the old one's.
    pools = importlib.import_module("concurrent.futures.process").ProcessPoolExecutor
    with pools(
        plan.workers, mp_context=context, initializer=install, initargs=(payload, slots)
    ) as pool:
        yield from pool.map(work, tasks)


def _add_counts(totals: list[list[int]], counts: list[tuple[int, int]]) -> None:
    for total, (seen, passed) in zip(totals, counts):
        total[0] += seen
        total[1] += passed


def execute(plan: Plan) -> list[Any]:
    """Run the loop, and give every action's result, in the order of ``plan.actions``."""
    accs = [action.start() for action in plan.actions]
    totals = [[0, 0] for _ in plan.filters]
    try:
        _fold(plan, accs, totals)
    except BaseException:
        for action, acc in zip(plan.actions, accs):
            action.abort(acc)
        raise
    flow = {id(node): (total[0], total[1]) for node, total in zip(plan.filters, totals)}
    return [action.finish(acc, flow) for action, acc in zip(plan.actions, accs)]


def _fold(plan: Plan, accs: list[Any], totals: list[list[int]]) -> None:
    """Every task's partials, added up in task order into ``accs``, in place."""
    tasks = plan.tasks()
    parallel = plan.workers > 1 and len(tasks) > 1
    for parts, counts in _parallel(plan, tasks) if parallel else _serial(plan, tasks, accs):
        for at, (action, part) in enumerate(zip(plan.actions, parts)):
            accs[at] = action.merge(accs[at], part)
        _add_counts(totals, counts)
