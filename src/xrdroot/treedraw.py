"""``TTree::Draw``: a histogram filled from a tree by expression, a batch at a time.

    >>> h = tree.draw("jet_pt", "Sum$(jet_pt > 30) > 1")        # doctest: +SKIP
    >>> h.entries, h.mean(), h.selected

The expressions are compiled once and evaluated over one batch of entries at
a time - every axis, the selection and the weight in one shared loop, as
``TTreeFormulaManager`` runs them - so memory is a batch and never the tree.
What is filled is what ``TSelectorDraw`` fills, in the order it fills it: the
selection's value multiplies the weight, a fill of weight zero is not made,
and the rest go into a ``TH1F``, ``TH2F`` or ``TH3F`` - or a ``TProfile`` or
``TProfile2D`` - booked as ROOT books it.

A draw not told its binning books ROOT's ``htemp``: the first
``estimate`` fills - ``TTree::GetEstimate()``, a million unless changed - are
held back, :mod:`~xrdroot.limits` finds the axis they need, and after them
every value is filled as it is read, the axis doubling to take in any that
falls off it, as :mod:`~xrdroot.extending` has it. So a draw of a hundred
entries and one of a hundred million end up with exactly the axes, the bins
and the moments ROOT would give them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from typing import Any, NamedTuple

import numpy as np

from . import drawspec
from .booking import histogram_members, profile_members
from .extending import _cell_bins, fill_in_order
from .filling import running
from .formula import Formula, compile_formula
from .formula.nodes import Alt, Node, Reduce, Ref, Size, Special
from .formula.select import TreeNames, _Read, _shaped, _underlying
from .formula.together import evaluate_together
from .graph import Graph
from .hist import Histogram
from .limits import DBL_MAX, Limits, good_axes
from .profile import Profile
from .tree import Jagged

__all__ = ["ESTIMATE", "DrawnGraph", "DrawnHistogram", "DrawnProfile", "draw"]

Array = Any

#: ``TTree::fEstimate``'s default: how many fills a draw looks at to find its axes.
ESTIMATE = 1_000_000
#: How many entries a draw reads at a time, unless told otherwise.
STEP = 100_000
#: ``gEnv``'s ``Hist.Binning`` defaults: the bins of each axis of what a draw
#: books, by what it books and how many axes it bins along.
DEFAULT_BINS = {
    ("hist", 1): (100,),
    ("hist", 2): (40, 40),
    ("hist", 3): (20, 20, 20),
    ("prof", 1): (100,),
    ("prof", 2): (20, 20),
}

class DrawnHistogram(Histogram):
    """A :class:`~.hist.Histogram` a draw booked, with the count of what it selected.

    ``selected`` is the number of fills the draw made - ``TTree::Draw``'s
    return value - and ``extendable`` whether its axes still double to take
    in a value off their ends, as ROOT's ``htemp`` does.
    """

    __slots__ = ("selected", "extendable")

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        super().__init__(classname, members)
        self.selected = 0
        self.extendable = False


class DrawnProfile(Profile):
    """A :class:`~.profile.Profile` a draw booked, with ``selected`` and ``extendable`` too."""

    __slots__ = ("selected", "extendable")

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        super().__init__(classname, members)
        self.selected = 0
        self.extendable = False


class DrawnGraph(Graph):
    """The scatter of a ``y:x`` draw, as a :class:`~.graph.Graph`, with ``selected`` too."""

    __slots__ = ("selected",)

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        super().__init__(classname, members)
        self.selected = 0


#: Everything a draw makes, which says how many fills it selected.
DRAWN = (DrawnHistogram, DrawnProfile, DrawnGraph)


# -- what the call asks for --------------------------------------------------


class Plan(NamedTuple):
    """Everything the strings and keywords say, before anything is read."""

    #: ``"hist"``, ``"prof"`` or ``"graph"``.
    kind: str
    #: The expressions in the order they fill: x, y, z - and what a profile averages last.
    parts: list[str]
    name: str
    title: str
    #: The axis titles to book with, or ``None`` for those the title gives.
    labels: list[str] | None
    options: drawspec.Options
    #: The histogram ``>>name`` or ``>>+name`` fills, when it is already there.
    existing: Histogram | None


def _dimensions(parts: list[str], varexp: str) -> int:
    if not 1 <= len(parts) <= 3:
        raise ValueError(
            f"{varexp!r} has {len(parts)} parts, and a draw fills one, two or three axes "
            f"- written 'x', 'y:x' or 'z:y:x'"
        )
    if not all(part.strip() for part in parts):
        raise ValueError(f"{varexp!r} has an empty part between its colons")
    return len(parts)


def _existing(
    found: drawspec.Target, histograms: Mapping[str, Any] | None
) -> Histogram | None:
    """The histogram ``>>name`` fills, as ``gDirectory->Get(name)`` finds it."""
    held = None if found.name is None or histograms is None else histograms.get(found.name)
    if found.add and held is None:
        raise KeyError(
            f"'>>+{found.name}' adds to a histogram called {found.name!r}, and there is none "
            f"in histograms=; pass the one to add to, or draw into '>>{found.name}'"
        )
    if held is None or found.binning is not None:
        return None  # ROOT makes a new one when brackets give the binning
    if not isinstance(held, Histogram):
        raise TypeError(
            f"{found.name!r} in histograms= is a {type(held).__name__}, not a histogram "
            f"or profile a draw can fill"
        )
    return held


def _kind(options: drawspec.Options, existing: Histogram | None, named: bool) -> str:
    if existing is not None:
        if options.profile is not None and not isinstance(existing, Profile):
            raise ValueError(
                f"option 'prof' fills a profile, and {existing.name!r} is a "
                f"{existing.classname}; ROOT would delete it and book a profile in its place"
            )
        return "prof" if isinstance(existing, Profile) else "hist"
    if options.profile is not None:
        return "prof"
    return "graph" if options.graph and not named else "hist"


def _fits(existing: Histogram | None, kind: str, dimensions: int) -> None:
    """Refuse a histogram of a different shape, which ROOT would delete and replace."""
    if existing is None:
        return
    binned = dimensions - (kind == "prof")
    if len(existing.axes) != binned or binned == 0:
        raise ValueError(
            f"{existing.name!r} is a {existing.classname} of {len(existing.axes)} "
            f"ax{'is' if len(existing.axes) == 1 else 'es'}, and a draw of {dimensions} "
            f"expression{'s' if dimensions > 1 else ''} fills "
            f"{'a profile' if kind == 'prof' else 'a histogram'} of {max(binned, 0)}"
        )


def _named(found: drawspec.Target, name: str | None) -> str:
    if found.name is not None and name is not None and name != found.name:
        raise ValueError(
            f"name={name!r} and '>>{found.name}' name the histogram twice; give one of them"
        )
    return found.name or name or "htemp"


def plan(
    varexp: str,
    selection: str,
    option: str,
    name: str | None,
    title: str | None,
    histograms: Mapping[str, Any] | None,
) -> tuple[Plan, drawspec.Target]:
    """Read the three strings and the keywords into what to fill, and how."""
    found = drawspec.target(varexp)
    parts = drawspec.split_names(found.varexp)
    dimensions = _dimensions(parts, varexp)
    options = drawspec.options(option, dimensions)
    existing = _existing(found, histograms)
    kind = _kind(options, existing, found.name is not None)
    _fits(existing, kind, dimensions)
    if options.norm and kind == "prof":
        raise ValueError("option 'norm' scales a histogram, and a profile holds means")
    ordered = [part.strip() for part in reversed(parts)]
    drawn = found.varexp + (f" {{{selection}}}" if len(selection) > 1 else "")
    keep_titles = kind == "hist" and found.name is None and title is None
    return (
        Plan(
            kind=kind,
            parts=ordered,
            name=_named(found, name),
            title=drawn if title is None else title,
            labels=ordered if keep_titles else None,
            options=options,
            existing=existing,
        ),
        found,
    )


# -- the binning --------------------------------------------------------------


class Axes(NamedTuple):
    """How the booked axes are binned: fixed from the start, or found from the fills."""

    #: The number of bins of each axis, x first.
    nbins: list[int]
    #: What to book each axis with when it is fixed: ``(n, low, high)`` or edges.
    fixed: list[Any] | None


def _from_brackets(numbers: Sequence[float | None], defaults: Sequence[int]) -> Axes:
    """The binning ``>>h(nx, xlo, xhi, ny, ...)`` gives, the way ``TSelectorDraw`` reads it."""
    nbins, ends = [], []
    for axis, default in enumerate(defaults):
        count, low, high = numbers[3 * axis : 3 * axis + 3]
        nbins.append(default if count is None else int(count))
        ends.append((0.0 if low is None else low, 0.0 if high is None else high))
    if all(low < high for low, high in ends):
        return Axes(nbins, [(n, low, high) for n, (low, high) in zip(nbins, ends)])
    return Axes(nbins, None)  # the ends given are ignored, as ROOT ignores them


def _one(spec: Any, default: int) -> tuple[int, Any]:
    """One axis of ``bins=``: its count, and what to book it with if it is fixed."""
    if spec is None:
        return default, None
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        return int(spec), None
    if isinstance(spec, tuple) and len(spec) == 3 and isinstance(spec[0], (int, np.integer)):
        return int(spec[0]), (int(spec[0]), float(spec[1]), float(spec[2]))
    edges = np.asarray(spec, dtype=np.float64)
    return len(edges) - 1, edges


def _per_axis(bins: Any, defaults: Sequence[int]) -> list[Any]:
    """``bins=`` as one spec per axis: itself for one axis, a spec each for more."""
    specs = [bins] if len(defaults) == 1 else list(bins)
    if len(specs) != len(defaults):
        raise ValueError(
            f"bins= gives {len(specs)} axes for a draw that bins along {len(defaults)}: give "
            f"one (n, low, high), count or run of edges per axis, x first"
        )
    return specs


def _from_keyword(bins: Any, defaults: Sequence[int]) -> Axes:
    specs = _per_axis(bins, defaults)
    found = [_one(spec, default) for spec, default in zip(specs, defaults)]
    counts = [count for count, _book in found]
    booked = [book for _count, book in found]
    if all(book is not None for book in booked):
        return Axes(counts, booked)
    if any(isinstance(book, np.ndarray) for book in booked):
        raise ValueError(
            "bins= gives edges for one axis and leaves another to be found from the "
            "values; edges are only booked with every axis's binning given"
        )
    return Axes(counts, None)


def axes_for(found: drawspec.Target, bins: Any, kind: str, binned: int) -> Axes:
    """The binning of what a draw books, from ``>>h(...)``, ``bins=`` or ROOT's defaults."""
    defaults = DEFAULT_BINS[(kind, binned)]
    if found.binning is not None and bins is not None:
        raise ValueError(
            "the binning is given twice, in the brackets after '>>' and as bins=; give one"
        )
    if found.binning is not None:
        return _from_brackets(found.binning, defaults)
    if bins is not None:
        return _from_keyword(bins, defaults)
    return Axes(list(defaults), None)


# -- whole numbers, which get bins a whole number wide --------------------------

#: The ``TTreeFormula`` lookups ``IsInteger`` counts as whole numbers.
INTEGER_SPECIALS = ("Entry$", "Entries$", "LocalEntry$", "Iteration$", "Length$")


#: The types of a branch ``IsLeafInteger`` counts: every integer, and ``Bool_t``.
INTEGER_TYPES = {"bool", *(f"{sign}int{bits}" for sign in ("", "u") for bits in (8, 16, 32, 64))}


def _integer_branch(branch: Any) -> bool:
    name = (branch.typename or "").replace("list[", "").replace("]", "")
    return name in INTEGER_TYPES


def is_integer(formula: Formula, source: Any) -> bool:
    """``TTreeFormula::IsInteger``: a lone integer branch, or a count ROOT keeps."""
    root: Node = formula._root
    while isinstance(root, Alt):
        root = root.primary
    if isinstance(root, Special):
        return root.name in INTEGER_SPECIALS
    if isinstance(root, Reduce):
        return root.kind == "Length$"
    if isinstance(root, Ref):
        return _integer_branch(source[root.column])
    return isinstance(root, Size)


# -- reading ------------------------------------------------------------------


def _numbers(values: Any, valid: Any) -> tuple[Array, Array]:
    """A result flattened to one value per fill, as doubles, and where it is real."""
    if isinstance(values, Jagged):
        values, valid = values.content, valid.content
    values = np.asarray(values)
    if values.dtype.kind in "USO":
        raise ValueError(
            "a draw fills numbers, and this expression is strings; ROOT bins strings by "
            "label, which this does not - scan them instead, or compare them in the selection"
        )
    return values.astype(np.float64), np.asarray(valid, dtype=bool)


def batches(
    source: Any, formulas: Sequence[Formula], first: int, count: int | None, step: int
) -> Iterator[list[tuple[Any, Any]]]:
    """Every formula's value for one batch of entries after another, in one loop each."""
    total = len(source)
    stop = total if count is None else min(total, first + count)
    branches = list(dict.fromkeys(name for f in formulas for name in f.branches))
    for start in range(first, stop, step):
        read = _Read(source, start, min(start + step, stop), None)
        columns = {n: _shaped(_underlying(source[n]), read.column(n)) for n in branches}
        yield evaluate_together(
            formulas,
            columns,
            entries=total,
            rows=len(read.numbers),
            entry_numbers=read.numbers,
            local_entries=read.local,
        )


class Weighing(NamedTuple):
    """The selection and weight of a draw, as formulas or a number."""

    selection: Formula | None
    weight: Formula | float


def _fills(
    results: list[tuple[Any, Any]], parts: int, weighing: Weighing
) -> tuple[list[Array], Array]:
    """One batch's fills: a value per axis and a weight, those of weight zero left out."""
    found = [_numbers(*result) for result in results]
    keep = np.logical_and.reduce([valid for _values, valid in found])
    at = parts
    weight: Array = weighing.weight
    if isinstance(weighing.weight, Formula):
        weight, at = found[at][0], at + 1
    if weighing.selection is not None:
        weight = weight * found[at][0]  # fWeight * fSelect->EvalInstance()
    weights = np.broadcast_to(np.asarray(weight, dtype=np.float64), keep.shape)
    keep &= weights != 0
    return [values[keep] for values, _valid in found[:parts]], weights[keep]


# -- filling --------------------------------------------------------------------


def _extent(values: Array) -> tuple[float, float]:
    """``TakeEstimate``'s smallest and largest value, which no NaN is either of."""
    values = values[~np.isnan(values)]
    if not len(values):
        return DBL_MAX, -DBL_MAX
    return float(values.min()), float(values.max())


class Filling:
    """What a draw fills: held back until the axes are found, then filled as it is read."""

    def __init__(
        self,
        book: Callable[[list[Limits] | None], Histogram],
        axes: Axes,
        integers: list[bool],
        estimate: int,
        histogram: Histogram | None,
    ) -> None:
        self.book, self.axes, self.integers, self.estimate = book, axes, integers, estimate
        self.histogram = histogram
        self.extendable = getattr(histogram, "extendable", False)
        if histogram is None and axes.fixed is not None:
            self.histogram = book(None)
        self.held: list[list[Array]] = []
        self.count = 0

    def feed(self, columns: list[Array], weights: Array) -> None:
        if self.histogram is None:
            room = self.estimate - self.count
            self.held.append([column[:room] for column in [*columns, weights]])
            self.count += min(room, len(weights))
            if self.count < self.estimate:
                return
            self._booked()
            columns, weights = [column[room:] for column in columns], weights[room:]
        self._fill(columns, weights)

    def _fill(self, columns: list[Array], weights: Array) -> None:
        assert self.histogram is not None
        binned = len(self.histogram.axes)
        values = columns[binned] if len(columns) > binned else None
        self.extendable = fill_in_order(
            self.histogram, columns[:binned], values, weights, self.extendable
        )

    def _booked(self) -> None:
        """``TakeEstimate``: the axes from what was held back, which is then filled."""
        held = [np.concatenate(column) for column in zip(*self.held)] if self.held else []
        binned = len(self.axes.nbins)
        ranges = [_extent(column) for column in held[:binned]] or [(DBL_MAX, -DBL_MAX)] * binned
        self.histogram = self.book(good_axes(self.axes.nbins, ranges, self.integers))
        self.extendable = True
        self.held = []
        if held:
            self._fill(held[:-1], held[-1])

    def finish(self) -> Histogram:
        if self.histogram is None:
            self._booked()
        assert self.histogram is not None
        if isinstance(self.histogram, (DrawnHistogram, DrawnProfile)):
            self.histogram.extendable = self.extendable
        return self.histogram


class Scatter:
    """The points of a ``y:x`` draw, kept for a graph: every fill, in order."""

    def __init__(self) -> None:
        self.points: list[list[Array]] = []

    def feed(self, columns: list[Array], weights: Array) -> None:
        self.points.append(columns)

    def finish(self, name: str, title: str) -> DrawnGraph:
        """The graph, named and titled ``Graph`` as ``TGraph(n, x, y)`` is unless told otherwise."""
        if self.points:
            x, y = (np.concatenate(axis) for axis in zip(*self.points))
        else:
            x, y = np.zeros(0), np.zeros(0)
        graph = DrawnGraph.new(name, x, y, title=title)
        assert isinstance(graph, DrawnGraph)
        return graph


def _booker(chosen: Plan, axes: Axes) -> Callable[[list[Limits] | None], Histogram]:
    """How to book what the draw fills, once its axes are known."""

    def book(found: list[Limits] | None) -> Histogram:
        specs = axes.fixed if found is None else [tuple(limits) for limits in found]
        assert specs is not None
        if chosen.kind == "prof":
            error = chosen.options.profile or ""
            made: Histogram = DrawnProfile(
                *profile_members(chosen.name, specs, chosen.title, error, None, None)
            )
        else:
            made = DrawnHistogram(
                *histogram_members(chosen.name, specs, chosen.title, "F", chosen.labels)
            )
        if chosen.options.errors:
            made._ensure_sumw2()
        return made

    return book


def _sum_of_weights(histogram: Histogram) -> float:
    """``GetSumOfWeights``: the bins on every axis added in turn, as ROOT adds them."""
    bins = _cell_bins(histogram._widths)
    inner = np.logical_and.reduce(
        [(at >= 1) & (at <= axis.nbins) for at, axis in zip(bins, histogram.axes)]
    )
    return running(0.0, histogram._bins[inner].astype(np.float64))


def _normalised(histogram: Histogram) -> None:
    total = _sum_of_weights(histogram)
    if total != 0:
        histogram.scale(1.0 / total)


def _checked_range(first_entry: int, entries: int | None, step: int, estimate: int) -> None:
    if first_entry < 0 or (entries is not None and entries < 0):
        raise ValueError("first_entry and entries count entries, and neither can be negative")
    if step < 1 or estimate < 1:
        raise ValueError("step and estimate are numbers of entries, at least one each")


def _compiled(
    source: Any, chosen: Plan, selection: str, weight: Any, aliases: Any
) -> tuple[list[Formula], Weighing]:
    names = TreeNames(source)
    formulas = [compile_formula(part, names, aliases=aliases) for part in chosen.parts]
    weighed: Formula | float
    if isinstance(weight, str):
        weighed = compile_formula(weight, names, aliases=aliases)
        formulas.append(weighed)
    else:
        weighed = 1.0 if weight is None else float(weight)
    cut = compile_formula(selection, names, aliases=aliases) if selection else None
    if cut is not None:
        formulas.append(cut)
    return formulas, Weighing(cut, weighed)


def _filler(
    source: Any, chosen: Plan, found: drawspec.Target, bins: Any, estimate: int,
    formulas: list[Formula],
) -> Filling | Scatter:
    if chosen.kind == "graph":
        if bins is not None:
            raise ValueError("a scatter of points has no bins; drop bins= or the point option")
        return Scatter()
    binned = len(chosen.parts) - (chosen.kind == "prof")
    if chosen.existing is not None:
        if bins is not None:
            raise ValueError(
                f"{chosen.name!r} is already binned, and bins= would bin it again"
            )
        if not found.add:
            chosen.existing.reset()  # '>>h' starts it again; '>>+h' adds to it
        none = Axes([], None)
        return Filling(_booker(chosen, none), none, [], estimate, chosen.existing)
    axes = axes_for(found, bins, chosen.kind, binned)
    integers = [is_integer(formula, source) for formula in formulas[:binned]]
    return Filling(_booker(chosen, axes), axes, integers, estimate, None)


def _made(filler: Filling | Scatter, chosen: Plan, title: str | None) -> Any:
    """What the draw filled, finished: a graph named as ROOT names one unless told."""
    if isinstance(filler, Filling):
        return filler.finish()
    if chosen.name == "htemp":
        return filler.finish("Graph", title or "Graph")
    return filler.finish(chosen.name, chosen.title)


def _run(
    source: Any,
    formulas: list[Formula],
    filler: Filling | Scatter,
    parts: int,
    weighing: Weighing,
    reading: tuple[int, int | None, int],
) -> int:
    """Read the tree a batch at a time into ``filler``: how many fills were made."""
    selected = 0
    for results in batches(source, formulas, *reading):
        columns, weights = _fills(results, parts, weighing)
        filler.feed(columns, weights)
        selected += len(weights)
    return selected


def draw(
    source: Any,
    varexp: str,
    selection: str = "",
    option: str = "",
    *,
    entries: int | None = None,
    first_entry: int = 0,
    bins: Any = None,
    name: str | None = None,
    title: str | None = None,
    weight: Any = None,
    ax: Any = None,
    step: int | None = None,
    estimate: int = ESTIMATE,
    histograms: MutableMapping[str, Any] | None = None,
    aliases: Mapping[str, str] | None = None,
) -> Any:
    """``TTree::Draw`` on a tree, a chain or anything that reads like one; see ``TTree.draw``."""
    step = STEP if step is None else step
    _checked_range(first_entry, entries, step, estimate)
    chosen, found = plan(varexp, selection, option, name, title, histograms)
    formulas, weighing = _compiled(source, chosen, selection, weight, aliases)
    filler = _filler(source, chosen, found, bins, estimate, formulas)
    reading = (first_entry, entries, step)
    selected = _run(source, formulas, filler, len(chosen.parts), weighing, reading)
    made = _made(filler, chosen, title)
    if chosen.options.norm and not isinstance(made, Graph):
        _normalised(made)
    if isinstance(made, DRAWN):
        made.selected = selected
    if histograms is not None and not isinstance(made, Graph):
        histograms[chosen.name] = made
    if ax is not None:
        made.plot(ax=ax)
    return made
