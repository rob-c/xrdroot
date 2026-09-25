"""``RDataFrame``: ROOT's declarative analysis, lazily, in one pass, a batch at a time.

    >>> df = RDataFrame("Events", "events.root")                        # doctest: +SKIP
    >>> two = df.Filter("nMuon == 2", "two muons")                      # doctest: +SKIP
    >>> mass = two.Define("m", "InvariantMass(Muon_pt, Muon_eta, Muon_phi, Muon_mass)")
    >>> h = mass.Histo1D(("m", "dimuon mass", 300, 0.25, 300), "m")     # doctest: +SKIP
    >>> h.GetValue().plot()                                             # doctest: +SKIP

Nothing is read until a result is asked for. ``Define`` and ``Filter``
describe a graph; ``Histo1D``, ``Count``, ``Sum`` and the other actions book
results on it and hand back a :class:`Result`, a promise of the value; the
first one asked for runs one event loop that fills every result booked so
far, reading each column once. The difference from ROOT is what a batch is:
ROOT calls the code of a ``Define`` or ``Filter`` once per entry, and here
expressions - and callables - work on a whole batch of entries at once, as
NumPy arrays and :class:`~xrdroot.Jagged` collections, so nothing loops over
entries in Python.

Every method has ROOT's name, and a snake_case one beside it: ``Define`` and
``define``, ``Histo1D`` and ``histo1d``, ``AsNumpy`` and ``as_numpy``.
"""

from __future__ import annotations

import inspect
import keyword
import os
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from ..library import convert
from . import actions
from .expression import Expression
from .graph import (
    Defined,
    Definition,
    EntryColumn,
    Filter,
    PerSample,
    Range,
    Root,
    Selector,
    SlotColumn,
    SourceColumn,
    chain_of,
    named_filters,
)
from .loop import DEFAULT_STEP, Batch, Plan, execute
from .models import model_of, split_arguments
from .snapshot import MODES, Snapshot
from .sources import Empty, Source, open_named, wrap

__all__ = [
    "RDataFrame",
    "RNode",
    "Result",
    "RunGraphs",
    "EnableImplicitMT",
    "DisableImplicitMT",
    "IsImplicitMTEnabled",
    "GetThreadPoolSize",
]

#: The number of worker processes a frame uses when it is not told: ``EnableImplicitMT`` sets it.
_IMPLICIT = [1]

#: The columns every frame has, that ROOT's ``GetColumnNames`` leaves out.
SPECIAL = {"rdfentry_": EntryColumn("rdfentry_"), "rdfslot_": SlotColumn("rdfslot_")}

#: What a name ``Define`` makes may look like: C++'s identifiers.
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def EnableImplicitMT(workers: int = 0) -> None:
    """``ROOT::EnableImplicitMT``: frames made from now on use this many worker processes.

    ``0`` means one per CPU, as ROOT's default is a thread per core. Here they
    are processes rather than threads, which is what lets Python code in a
    ``Define`` run side by side, and what makes callables have to be picklable.
    """
    _IMPLICIT[0] = int(workers) or (os.cpu_count() or 1)


def DisableImplicitMT() -> None:
    """``ROOT::DisableImplicitMT``: back to one process, the deterministic default."""
    _IMPLICIT[0] = 1


def IsImplicitMTEnabled() -> bool:
    return _IMPLICIT[0] > 1


def GetThreadPoolSize() -> int:
    """How many worker processes a frame made now would use."""
    return _IMPLICIT[0]


class Result:
    """``RResultPtr``: a result booked on a frame, computed the first time it is asked for.

        >>> n = df.Count()                         # doctest: +SKIP
        >>> n.GetValue()                           # runs the event loop, once
        >>> int(n), n.value                        # there already

    Asking for the value - :meth:`GetValue`, :attr:`value`, ``float()``,
    ``int()``, iterating, indexing, or any attribute of the value itself -
    runs the frame's event loop if it has not run since the result was
    booked, and that one loop computes every result booked so far.
    """

    __slots__ = ("_graph", "_action", "_value", "_ready", "_error")

    def __init__(self, graph: _Graph, action: actions.Action) -> None:
        self._graph = graph
        self._action = action
        self._value: Any = None
        self._ready = False
        #: Why the loop computing this failed, raised again whenever it is asked for.
        self._error: BaseException | None = None

    def __repr__(self) -> str:
        if not self._ready:
            return f"<Result of {self._action.kind}, not yet computed>"
        return f"<Result of {self._action.kind}: {self._value!r}>"

    def IsReady(self) -> bool:
        """Has the loop that computes this run yet?"""
        return self._ready

    def GetValue(self) -> Any:
        """The value, running the event loop first if it has not run yet."""
        if self._error is not None:
            raise self._error
        if not self._ready:
            self._graph.run()
        return self._value

    @property
    def value(self) -> Any:
        return self.GetValue()

    def _set(self, value: Any) -> None:
        self._value = value
        self._ready = True

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.GetValue(), name)

    def __float__(self) -> float:
        return float(self.GetValue())

    def __int__(self) -> int:
        return int(self.GetValue())

    def __index__(self) -> int:
        return int(self.GetValue())

    def __iter__(self) -> Iterator[Any]:
        return iter(self.GetValue())

    def __len__(self) -> int:
        return len(self.GetValue())

    def __getitem__(self, key: Any) -> Any:
        return self.GetValue()[key]

    def __str__(self) -> str:
        return str(self.GetValue())

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        return np.asarray(self.GetValue(), dtype=dtype)

    get_value = GetValue
    is_ready = IsReady


class _Graph:
    """What every frame derived from one ``RDataFrame`` shares: the data and the bookings."""

    def __init__(self, source: Source, workers: int, step: int) -> None:
        if step <= 0:
            raise ValueError("step must be at least one entry")
        if workers < 1:
            raise ValueError("workers must be at least one process")
        self.source = source
        self.workers = workers
        self.step = step
        self.root = Root()
        #: The results booked since the loop last ran.
        self.pending: list[Result] = []
        #: Every named filter, in the order they were made.
        self.filters: list[Filter] = []
        self.runs = 0

    def book(self, action: actions.Action) -> Result:
        result = Result(self, action)
        self.pending.append(result)
        return result

    def run(self) -> None:
        """One event loop: every result booked since the last."""
        _run_together([self])


def _run_together(graphs: Sequence[_Graph]) -> None:
    """One loop over the data several graphs share, filling what each has booked."""
    pending = [result for graph in graphs for result in graph.pending]
    for graph in graphs:
        graph.pending = []
    values = _computed(graphs, pending)
    for graph in graphs:
        graph.runs += 1
    for result, value in zip(pending, values):
        result._set(value)


def _computed(graphs: Sequence[_Graph], pending: list[Result]) -> list[Any]:
    """Every pending result's value; if the loop fails, it has failed for every one of them."""
    try:
        return execute(_plan_of(graphs, pending))
    except Exception as why:
        for result in pending:
            result._error = why
        raise


def _plan_of(graphs: Sequence[_Graph], pending: list[Result]) -> Plan:
    first = graphs[0]
    filters = [node for graph in graphs for node in graph.filters]
    work = [result._action for result in pending]
    return Plan(first.source, work, filters, first.step, first.workers)


def _sharing(graph: _Graph) -> tuple[int, int, int]:
    source = graph.source
    underlying = getattr(source, "table", source)
    return id(underlying), graph.step, graph.workers


def RunGraphs(results: Iterable[Result]) -> int:
    """``ROOT::RDF::RunGraphs``: compute these results, in as few event loops as there can be.

    Results of one frame come from one loop, as always. Frames made
    separately over the very same tree, chain or RNTuple object - with the
    same ``step`` and ``workers`` - share one loop too, reading each column
    once for all of them. Gives the number of loops run.
    """
    graphs: dict[int, _Graph] = {}
    for result in results:
        if not result._ready:
            graphs.setdefault(id(result._graph), result._graph)
    groups: dict[tuple[int, int, int], list[_Graph]] = {}
    for graph in graphs.values():
        groups.setdefault(_sharing(graph), []).append(graph)
    for group in groups.values():
        _run_together(group)
    return len(groups)


def _source_of(args: tuple[Any, ...]) -> Source:
    if len(args) == 1 and isinstance(args[0], (int, np.integer)) and not isinstance(args[0], bool):
        return Empty(int(args[0]))
    if len(args) == 2 and isinstance(args[0], str):
        return open_named(args[0], args[1])
    if len(args) == 1:
        return wrap(args[0])
    raise TypeError(
        "an RDataFrame is made from a TTree, Chain or RNTuple; from a tree's name and a "
        "file, a list of them or a glob; or from a number of empty entries"
    )


def _parameters(function: Callable[..., Any], what: str) -> list[str]:
    """The columns a callable reads, when not told: the names of its parameters."""
    try:
        found = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        found = None
    plain = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if found is None or any(each.kind not in plain for each in found):
        raise TypeError(
            f"{what} was given a callable without the columns it reads, and they cannot be "
            f"read off its parameters; give columns=[...]"
        )
    return [each.name for each in found]


def _as_names(columns: Any) -> list[str]:
    if isinstance(columns, str):
        return [columns]
    return list(columns)


class RDataFrame:
    """ROOT's ``RDataFrame``, over a tree, a chain, an RNTuple, or empty entries.

        >>> RDataFrame(tree)                                  # doctest: +SKIP
        >>> RDataFrame("Events", "events.root")               # doctest: +SKIP
        >>> RDataFrame("Events", ["a.root", "run2/*.root"])   # doctest: +SKIP
        >>> RDataFrame(1_000_000).Define("x", "rdfentry_ * 2.")   # doctest: +SKIP

    ``workers`` is the number of processes the event loop is shared across -
    :func:`EnableImplicitMT` sets what it is when not given, and it is one
    unless that was called - and ``step`` the number of entries in a batch.
    A frame made from a tree's name and files opens them, and closes them
    with :meth:`close` or a ``with`` block; one handed a tree leaves it be.

    Every transformation gives a new frame, sharing the graph with this one:
    ``Define``, ``Redefine``, ``DefinePerSample``, ``Alias``, ``Filter`` and
    ``Range``. Every action books a result and gives a :class:`Result`.
    """

    _graph: _Graph
    _node: Selector
    _columns: dict[str, Definition]
    _defined: list[str]

    def __init__(self, *args: Any, workers: int | None = None, step: int = DEFAULT_STEP) -> None:
        source = _source_of(args)
        graph = _Graph(source, _IMPLICIT[0] if workers is None else int(workers), step)
        columns: dict[str, Definition] = {name: SourceColumn(name) for name in source.names()}
        self._set(graph, graph.root, {**columns, **SPECIAL}, [])

    def _set(
        self, graph: _Graph, node: Selector, columns: dict[str, Definition], defined: list[str]
    ) -> None:
        self._graph = graph
        self._node = node
        self._columns = columns
        self._defined = defined

    def _derived(
        self,
        node: Selector | None = None,
        columns: dict[str, Definition] | None = None,
        defined: list[str] | None = None,
    ) -> RDataFrame:
        made = object.__new__(RDataFrame)
        made._set(
            self._graph,
            self._node if node is None else node,
            self._columns if columns is None else columns,
            self._defined if defined is None else defined,
        )
        return made

    def __repr__(self) -> str:
        kind = "RDataFrame" if isinstance(self._node, Root) else "RNode"
        return (
            f"<{kind} over {self._graph.source.describe()}, {len(self.GetColumnNames())} columns>"
        )

    def close(self) -> None:
        """Close the files this frame opened itself; a tree it was handed is left open."""
        if self._graph.source.owned:
            self._graph.source.close()

    def __enter__(self) -> RDataFrame:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- columns -------------------------------------------------------------------

    def _column(self, name: str, what: str) -> Definition:
        """A column by name, or - as ROOT's actions do not, but it costs nothing - an expression."""
        found = self._columns.get(name)
        if found is not None:
            return found
        compiled = Expression(name, list(self._columns))
        return Defined(name, compiled, self._inputs(compiled), self._node)

    def _inputs(self, expression: Expression) -> dict[str, Definition]:
        return {name: self._columns[name] for name in expression.columns}

    def _compute(
        self, compute: str | Callable[..., Any], columns: Sequence[str] | None, what: str
    ) -> tuple[Any, dict[str, Definition]]:
        if isinstance(compute, str):
            if columns is not None:
                raise TypeError(f"{what} reads the columns its expression names; give no columns")
            compiled = Expression(compute, list(self._columns))
            return compiled, self._inputs(compiled)
        if not callable(compute):
            raise TypeError(
                f"{what} takes an expression or a callable, not {type(compute).__name__}"
            )
        return compute, self._named(compute, columns, what)

    def _named(
        self, compute: Callable[..., Any], columns: Sequence[str] | None, what: str
    ) -> dict[str, Definition]:
        """The columns a callable reads, in the order it takes them: named, or its parameters'.

        They are keyed by position, since a callable may be given one column twice.
        """
        names = _parameters(compute, what) if columns is None else _as_names(columns)
        missing = [name for name in names if name not in self._columns]
        if missing:
            raise KeyError(
                f"{what} reads {', '.join(map(repr, missing))}, which this frame has no column "
                f"called; there is {', '.join(self.GetColumnNames()) or 'none'}"
            )
        return {str(at): self._columns[name] for at, name in enumerate(names)}

    def _check_new(self, name: str, what: str) -> None:
        if not isinstance(name, str) or not IDENTIFIER.fullmatch(name) or keyword.iskeyword(name):
            raise ValueError(f"{what} makes a column called {name!r}, which is not a C++ name")
        if name in SPECIAL or name.startswith("rdf"):
            raise ValueError(f"{what} cannot make {name!r}: names beginning rdf are ROOT's own")
        if name in self._columns:
            raise ValueError(
                f"{what} makes {name!r}, and this frame has a column called that already; "
                f"Redefine replaces one"
            )

    def Define(
        self, name: str, expression: str | Callable[..., Any], columns: Sequence[str] | None = None
    ) -> RDataFrame:
        """A new column, made from others by a C++ expression or a Python callable.

            >>> df.Define("pt2", "pt * pt")                              # doctest: +SKIP
            >>> df.Define("good", "jet_pt[jet_pt > 30]")                 # doctest: +SKIP
            >>> df.Define("r", lambda x, y: np.hypot(x, y))              # doctest: +SKIP
            >>> df.Define("r", np.hypot, ["x", "y"])                     # doctest: +SKIP

        A callable is given the columns a batch at a time - NumPy arrays,
        :class:`~xrdroot.Jagged` collections, lists of strings - in the
        order of ``columns``, or of its own parameters' names when that is
        not given, and gives back one value per entry of the batch: an
        array, a ``Jagged`` of a collection per entry, or a list.
        """
        self._check_new(name, "Define")
        return self._defining(name, expression, columns, "Define")

    def _defining(
        self,
        name: str,
        expression: str | Callable[..., Any],
        columns: Sequence[str] | None,
        what: str,
    ) -> RDataFrame:
        compute, inputs = self._compute(expression, columns, f"{what}({name!r})")
        made = Defined(name, compute, inputs, self._node)
        defined = self._defined if name in self._defined else [*self._defined, name]
        return self._derived(columns={**self._columns, name: made}, defined=defined)

    def Redefine(
        self, name: str, expression: str | Callable[..., Any], columns: Sequence[str] | None = None
    ) -> RDataFrame:
        """``Define`` of a column that is there already, which nodes after this see instead."""
        if name not in self._columns or name in SPECIAL:
            raise ValueError(f"Redefine replaces a column, and this frame has none called {name!r}")
        return self._defining(name, expression, columns, "Redefine")

    def DefinePerSample(self, name: str, function: Callable[[Any], Any]) -> RDataFrame:
        """A column of one value per file, from what ``function`` makes of the file.

        ``function`` is given a :class:`~.sources.SampleInfo` - ``id``,
        ``Contains(text)``, ``EntryRange()`` - once per batch, and every
        batch lies within one file, and gives back a number.
        """
        self._check_new(name, "DefinePerSample")
        if not callable(function) or isinstance(function, str):
            raise UnsupportedFeatureError(
                "DefinePerSample takes a callable of the sample's information; a string "
                "expression is not given the sample to look at"
            )
        made = PerSample(name, function)
        return self._derived(columns={**self._columns, name: made}, defined=[*self._defined, name])

    def Alias(self, alias: str, column: str) -> RDataFrame:
        """Another name for a column, which expressions and actions after this may use."""
        self._check_new(alias, "Alias")
        if column not in self._columns:
            raise KeyError(f"Alias names {column!r}, which this frame has no column called")
        columns = {**self._columns, alias: self._columns[column]}
        return self._derived(columns=columns, defined=[*self._defined, alias])

    def Filter(
        self,
        condition: str | Callable[..., Any],
        columns: Sequence[str] | str | None = None,
        name: str = "",
    ) -> RDataFrame:
        """Only the entries for which ``condition`` is true go on.

            >>> df.Filter("nMuon == 2", "two muons")                     # doctest: +SKIP
            >>> df.Filter(lambda pt: pt.max() > 30)                      # doctest: +SKIP

        As in ROOT, a string condition's second argument is the filter's
        name, which ``Report`` counts it under; a callable's is its columns.
        """
        if isinstance(columns, str):
            name, columns = columns, None
        compute, inputs = self._compute(condition, columns, "Filter")
        node = Filter(self._node, compute, inputs, name)
        if name:
            self._graph.filters.append(node)
        return self._derived(node=node)

    def Range(self, begin: int, end: int | None = None, stride: int = 1) -> RDataFrame:
        """``Range(end)`` or ``Range(begin, end, stride)``: a span of the entries reaching here.

        ``end`` of ``0`` or ``None`` is no end. Once every result is past the
        end of a ``Range`` above it, the loop stops, reading no further.
        """
        if end is None:
            begin, end = 0, begin
        if begin < 0 or end < 0 or stride < 1 or (end and end < begin):
            raise ValueError(
                f"Range({begin}, {end}, {stride}) is not a span: begin and end are counts from "
                f"zero, end after begin, and the stride at least one"
            )
        return self._derived(node=Range(self._node, begin, end or None, stride))

    # -- what there is ---------------------------------------------------------------

    def GetColumnNames(self) -> list[str]:
        """Every column this frame can use: those defined here first, then the data's."""
        mine = [name for name in self._defined if name in self._columns]
        return mine + [name for name in self._columns if name not in SPECIAL and name not in mine]

    @property
    def columns(self) -> list[str]:
        return self.GetColumnNames()

    def GetDefinedColumnNames(self) -> list[str]:
        return [name for name in self._defined if self._columns[name].name == name]

    def HasColumn(self, name: str) -> bool:
        return name in self._columns

    def GetFilterNames(self) -> list[str]:
        """The names of the named filters between the head of the graph and here."""
        return [node.name for node in named_filters(chain_of(self._node))]

    def GetNRuns(self) -> int:
        """How many event loops the graph has run."""
        return self._graph.runs

    def GetNSlots(self) -> int:
        """How many worker processes the loop is shared across."""
        return self._graph.workers

    def GetColumnType(self, name: str) -> str:
        """A column's C++ type, as ROOT names it: ``Float_t``, ``ROOT::VecOps::RVec<double>``."""
        from .types import cxx_type

        definition = self._columns.get(name)
        if definition is None:
            raise KeyError(f"this frame has no column called {name!r}")
        declared = (
            self._graph.source.cxx_type(name) if isinstance(definition, SourceColumn) else None
        )
        if declared:
            return declared
        probe = Batch(self._graph.source, (0, 0), 0, None, {})
        return cxx_type(definition, probe.column(definition, self._node))

    def Describe(self) -> str:
        """A description of the frame: its data, its columns and where each comes from."""
        from .types import describe

        return describe(self)

    # -- actions ---------------------------------------------------------------------

    def _book(self, action: actions.Action) -> Result:
        return self._graph.book(action)

    def _single(self, column: str, what: str) -> list[Definition]:
        if not isinstance(column, str):
            raise TypeError(f"{what} takes the name of one column, not {type(column).__name__}")
        return [self._column(column, what)]

    def Count(self) -> Result:
        """How many entries reach here."""
        return self._book(actions.Count(self._node, []))

    def Sum(self, column: str) -> Result:
        """Every value of a column added up; every element, of a collection."""
        return self._book(actions.Sum(self._node, self._single(column, "Sum")))

    def Mean(self, column: str) -> Result:
        return self._book(actions.Mean(self._node, self._single(column, "Mean")))

    def Min(self, column: str) -> Result:
        return self._book(actions.Extreme(self._node, self._single(column, "Min"), largest=False))

    def Max(self, column: str) -> Result:
        return self._book(actions.Extreme(self._node, self._single(column, "Max"), largest=True))

    def StdDev(self, column: str) -> Result:
        return self._book(actions.StdDev(self._node, self._single(column, "StdDev")))

    def Stats(self, column: str, weight: str | None = None) -> Result:
        """A ``TStatistic`` of a column: count, mean, spread, least and greatest."""
        inputs = self._single(column, "Stats") + (
            [] if weight is None else self._single(weight, "Stats")
        )
        return self._book(actions.Stats(self._node, inputs))

    def _histogram(
        self,
        args: tuple[Any, ...],
        model: Any,
        weight: str | None,
        shape: tuple[int, bool],
        what: str,
    ) -> Result:
        dimensions, profile = shape
        count = dimensions + (1 if profile else 0)
        model, names, weighted = split_arguments(args, model, count, what)
        weight = weighted if weight is None else weight
        inputs = [self._column(name, what) for name in names]
        if weight is not None:
            inputs.append(self._column(weight, what))
        if model is None and what == "Histo1D":
            return self._book(actions.AutoHisto(self._node, inputs, weight is not None))
        if model is None:
            raise TypeError(
                f"{what} needs a model: ('name', 'title', bins...) or a booked histogram"
            )
        booked = model_of(model, dimensions, profile, names[0], what)
        return self._book(actions.Histo(self._node, inputs, booked, weight is not None, what))

    def Histo1D(self, *args: Any, model: Any = None, weight: str | None = None) -> Result:
        """A histogram of a column: ``Histo1D(model, column, weight)``, or ``Histo1D(column)``.

        A model is ``("name", "title", nbins, low, high)`` - or edges for the
        binning, or a booked :class:`~xrdroot.Histogram`; without one, 128
        bins spanning every value filled, as ROOT books them.
        """
        return self._histogram(args, model, weight, (1, False), "Histo1D")

    def Histo2D(self, *args: Any, model: Any = None, weight: str | None = None) -> Result:
        """``Histo2D(model, x, y, weight)``: a histogram of two columns against each other."""
        return self._histogram(args, model, weight, (2, False), "Histo2D")

    def Histo3D(self, *args: Any, model: Any = None, weight: str | None = None) -> Result:
        """``Histo3D(model, x, y, z, weight)``."""
        return self._histogram(args, model, weight, (3, False), "Histo3D")

    def Profile1D(self, *args: Any, model: Any = None, weight: str | None = None) -> Result:
        """``Profile1D(model, x, value, weight)``: the mean of one column in bins of another."""
        return self._histogram(args, model, weight, (1, True), "Profile1D")

    def Profile2D(self, *args: Any, model: Any = None, weight: str | None = None) -> Result:
        """``Profile2D(model, x, y, value, weight)``."""
        return self._histogram(args, model, weight, (2, True), "Profile2D")

    def Graph(self, x: str, y: str) -> Result:
        """A ``TGraph`` of one column against another, a point per entry (or element)."""
        inputs = self._single(x, "Graph") + self._single(y, "Graph")
        return self._book(actions.GraphAction(self._node, inputs))

    def Take(self, column: str) -> Result:
        """A column's values, as one array, ``Jagged`` or list, in the order of the entries."""
        return self._book(actions.Take(self._node, self._single(column, "Take")))

    def _names(self, columns: Any, exclude: Any, what: str) -> list[str]:
        """The columns an action takes: every one readable, those a pattern finds, or a list."""
        if columns is None:
            chosen = self._readable()
        elif isinstance(columns, str):
            pattern = re.compile(columns)
            chosen = [name for name in self.GetColumnNames() if pattern.search(name)]
        else:
            chosen = list(columns)
        left_out = set(_as_names(exclude or []))
        return [name for name in chosen if name not in left_out]

    def _readable(self) -> list[str]:
        """Every column that can be read or computed: the data's that decode, and the made."""
        readable = set(self._graph.source.readable())
        return [name for name in self.GetColumnNames() if name in readable or self._made(name)]

    def _made(self, name: str) -> bool:
        return not isinstance(self._columns[name], SourceColumn)

    def AsNumpy(
        self, columns: Sequence[str] | str | None = None, exclude: Sequence[str] | None = None
    ) -> Result:
        """Columns as a dict of arrays - ``Jagged`` for collections - every one unless told.

        ``columns`` is a list of names, or a regular expression the names are
        matched against; ``exclude`` names columns to leave out.
        """
        names = self._names(columns, exclude, "AsNumpy")
        inputs = [self._column(name, "AsNumpy") for name in names]
        return self._book(actions.AsNumpy(self._node, inputs))

    def _library(self, library: str, columns: Any) -> Any:
        return convert(self.AsNumpy(columns).GetValue(), library)

    def to_numpy(self, columns: Sequence[str] | str | None = None) -> Any:
        """``AsNumpy``, computed now: a dict of arrays."""
        return self._library("np", columns)

    def to_pandas(self, columns: Sequence[str] | str | None = None) -> Any:
        """The columns as a pandas DataFrame, computed now."""
        return self._library("pd", columns)

    def to_awkward(self, columns: Sequence[str] | str | None = None) -> Any:
        return self._library("ak", columns)

    def to_arrow(self, columns: Sequence[str] | str | None = None) -> Any:
        return self._library("pa", columns)

    def to_polars(self, columns: Sequence[str] | str | None = None) -> Any:
        return self._library("pl", columns)

    def Reduce(self, function: Callable[[Any, Any], Any], column: str, init: Any = 0) -> Result:
        """Every value folded together by ``function``: ``np.add``, ``max``, any of two."""
        return self._book(
            actions.Reduce(self._node, self._single(column, "Reduce"), function, init)
        )

    def Aggregate(
        self,
        aggregator: Callable[..., Any],
        merger: Callable[[Any, Any], Any],
        column: str,
        init: Any = None,
    ) -> Result:
        """``aggregator(acc, batch_values)`` per batch, from ``init``; ``merger(a, b)`` across."""
        inputs = self._single(column, "Aggregate")
        return self._book(actions.Aggregate(self._node, inputs, aggregator, merger, init))

    def _foreach(self, function: Callable[..., Any], columns: Any, slot: bool) -> None:
        what = "ForeachSlot" if slot else "Foreach"
        if columns is None:
            names = _parameters(function, what)[1 if slot else 0 :]
        else:
            names = _as_names(columns)
        inputs = [self._column(name, what) for name in names]
        self._book(actions.Foreach(self._node, inputs, function, slot)).GetValue()

    def Foreach(self, function: Callable[..., Any], columns: Sequence[str] | None = None) -> None:
        """Run ``function`` on the columns of every batch, now: ROOT's instant action."""
        self._foreach(function, columns, slot=False)

    def ForeachSlot(
        self, function: Callable[..., Any], columns: Sequence[str] | None = None
    ) -> None:
        """``Foreach``, with the slot number first."""
        self._foreach(function, columns, slot=True)

    def Display(
        self, columns: Sequence[str] | str | None = None, rows: int = 5, elements: int = 10
    ) -> Result:
        """A box of the first ``rows`` entries of the columns, as ROOT's ``Display`` draws."""
        names = self._names(columns, None, "Display")
        inputs = [self._column(name, "Display") for name in names]
        return self._book(actions.DisplayAction(self._node, inputs, rows, elements))

    def Report(self) -> Result:
        """The cut flow of the named filters above here, or of all of them at the head."""
        every = (lambda: list(self._graph.filters)) if isinstance(self._node, Root) else None
        return self._book(actions.Report(self._node, every))

    def Snapshot(
        self,
        tree_name: str,
        filename: str,
        columns: Sequence[str] | str | None = None,
        *,
        rntuple: bool = False,
        compression: str | None = "zlib",
        mode: str = "RECREATE",
        lazy: bool = False,
    ) -> Any:
        """Write the entries reaching here, and the columns asked for, to a new file.

        The columns are every one this frame has unless told - a list, or a
        regular expression - and go into a ``TTree`` called ``tree_name``, or
        an RNTuple with ``rntuple=True``. As in ROOT this runs the event loop
        now, with every other result booked, and gives back a frame over what
        it wrote; ``lazy=True`` gives a :class:`Result` of that frame instead.
        """
        if mode not in MODES:
            raise ValueError(f"Snapshot's mode is one of {', '.join(MODES)}, not {mode!r}")
        names = self._names(columns, None, "Snapshot")
        if not names:
            raise ValueError("Snapshot has no columns to write: name at least one")
        inputs = [self._column(name, "Snapshot") for name in names]
        action = Snapshot(self._node, inputs, (tree_name, filename), rntuple, compression, mode)
        result = _Reopened(self._graph.book(action), self._graph)
        return result if lazy else result.GetValue()

    # -- the snake_case spellings ----------------------------------------------------

    define = Define
    redefine = Redefine
    define_per_sample = DefinePerSample
    alias = Alias
    filter = Filter
    range = Range
    count = Count
    sum = Sum
    mean = Mean
    min = Min
    max = Max
    std_dev = StdDev
    stddev = StdDev
    stats = Stats
    histo1d = Histo1D
    histo2d = Histo2D
    histo3d = Histo3D
    profile1d = Profile1D
    profile2d = Profile2D
    graph = Graph
    take = Take
    as_numpy = AsNumpy
    reduce = Reduce
    aggregate = Aggregate
    foreach = Foreach
    foreach_slot = ForeachSlot
    display = Display
    report = Report
    snapshot = Snapshot
    get_column_names = GetColumnNames
    get_defined_column_names = GetDefinedColumnNames
    get_column_type = GetColumnType
    get_filter_names = GetFilterNames
    has_column = HasColumn
    get_n_runs = GetNRuns
    get_n_slots = GetNSlots
    describe = Describe


#: ROOT's name for a frame derived from another: here they are the same class.
RNode = RDataFrame


class _Reopened:
    """``Snapshot``'s result: a frame over the file written, opened when first asked for."""

    def __init__(self, written: Result, graph: _Graph) -> None:
        self._written = written
        self._graph = graph
        self._frame: RDataFrame | None = None

    def IsReady(self) -> bool:
        return self._written.IsReady()

    def GetValue(self) -> RDataFrame:
        if self._frame is None:
            tree_name, filename = self._written.GetValue()
            self._frame = RDataFrame(
                tree_name, filename, workers=self._graph.workers, step=self._graph.step
            )
        return self._frame

    @property
    def value(self) -> RDataFrame:
        return self.GetValue()

    get_value = GetValue
    is_ready = IsReady
