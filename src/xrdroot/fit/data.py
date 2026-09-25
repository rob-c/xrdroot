"""``ROOT::Fit::BinData``: the points a fit is made to, chosen the way ``FillData`` chooses them.

What ROOT fits is not the histogram but a list of points taken from it, and
which points are on the list is most of what a fit option means. For a
histogram ``FillData`` walks the bins inside the axis range - x outermost,
then y, then z - and keeps each bin at its centre, or at its edges when the
function is to be integrated over it, with its content and its error; an
empty bin, whose error is zero, is left out unless the fit is a likelihood
(``L``), Pearson's chi-square (``P``) or ``WW``, which give it an error of
one. ``W`` sets every error to one. A range, given or the function's own
with ``R``, keeps the bins whose centres are inside it.

For a graph the error bars decide what kind of data it is: no errors at
all, errors in y, errors in x too - then the chi-square is the effective
variance's - or asymmetric errors in y, one side or the other taken by the
sign of the residual. ``GetDataType`` decides which from the class and from
which of its bars are zero, and a ``TMultiGraph`` takes the most elaborate
kind any of its graphs has, which leaves a plain ``TGraph`` in it with no
points at all. Each graph's points are taken in order of ``x``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import UnsupportedFeatureError
from ..fillrandom import bin_edges, first_last

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from ..hist import Axis, Histogram

__all__ = ["DataOptions", "FitData", "from_histogram", "from_graphs", "NO_ERROR", "VALUE_ERROR"]

#: ``BinData::ErrorType``, in ROOT's order, which a multigraph takes the largest of.
NO_ERROR, VALUE_ERROR, COORD_ERROR, ASYM_ERROR = 0, 1, 2, 3
#: The graph classes a fit reads, and the members their bars are in.
SYMMETRIC, ASYMMETRIC = "TGraphErrors", "TGraphAsymmErrors"


@dataclass
class DataOptions:
    """``ROOT::Fit::DataOptions``: how the points are chosen and weighed."""

    integral: bool = False
    bin_volume: bool = False
    norm_bin_volume: bool = False
    use_empty: bool = False
    errors1: bool = False
    exp_errors: bool = False
    coord_errors: bool = True
    asym_errors: bool = True


@dataclass
class FitData:
    """The points: coordinates ``(n, ndim)``, values and whichever errors the kind has."""

    x: Any
    y: Any
    error: Any
    kind: int
    options: DataOptions
    upper: Any = None
    xerr: Any = None
    ylow: Any = None
    yhigh: Any = None
    sum_content: float = 0.0
    sum_error2: float = 0.0
    weighted: bool = False
    ref_volume: float = 1.0

    @property
    def size(self) -> int:
        """``Size``: how many points there are."""
        return len(self.y)

    @property
    def ndim(self) -> int:
        """``NDim``: how many coordinates each point has."""
        return int(self.x.shape[1])

    def coordinates(self) -> Any:
        """The points as a function takes them: ``(n,)`` in one dimension, ``(n, ndim)`` else."""
        return self.x[:, 0] if self.ndim == 1 else self.x


def _adjusted(options: DataOptions, error: Any, value: Any) -> tuple[Any, Any]:
    """``AdjustError``: which points are kept, and the error each is kept with.

    A point of no error is kept, with an error of one, only for a fit that
    uses the empty bins, or one setting every error to one when the point
    has a value; with ``W`` every error is one.
    """
    empty = error <= 0
    keep = ~empty | options.use_empty | (options.errors1 & (np.abs(value) > 0))
    adjusted = np.where(empty | options.errors1, 1.0, error)
    return keep, adjusted


def _sums(data: FitData) -> None:
    """``BinData``'s running totals: of the values, of the errors squared, and whether weighted."""
    if data.error is None:
        data.sum_content = float(np.add.accumulate(np.concatenate(([0.0], data.y)))[-1])
        return
    y, e = data.y, data.error
    squares = np.where((y != 0) | (e != 1.0), e * e, 0.0)
    data.sum_content = float(np.add.accumulate(np.concatenate(([0.0], y)))[-1])
    data.sum_error2 = float(np.add.accumulate(np.concatenate(([0.0], squares)))[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        data.weighted = bool(np.any((y != 0) & (np.abs(e * e / y - 1.0) > 1e-12)))


# -- histograms --------------------------------------------------------------------------


def examine_range(axis: Axis, span: tuple[float, float], first: int, last: int) -> tuple[int, int]:
    """``ExamineRange``: the bins of ``[first, last]`` whose centres are inside ``span``."""
    low, high = span
    ilow, ihigh = int(axis.find_bin(low)), int(axis.find_bin(high))
    if ilow > last or ihigh < first:
        warnings.warn(
            f"fit range is outside histogram range, no fit data for {axis.name}",
            RuntimeWarning,
            stacklevel=6,
        )
    first = min(max(ilow, first), last + 1)
    last = max(min(ihigh, last), first - 1)
    if first < last:
        centres = axis.root_centers()
        first += int(centres[first] < low)
        last -= int(centres[last] > high)
    return first, last


def _axis_bins(histogram: Histogram, index: int, span: Any) -> Any:
    """The bins of one axis the fit takes: its range, cut to ``span`` if there is one."""
    axis = histogram.axes[index]
    first, last = first_last(histogram._core[f"f{'XYZ'[index]}axis"])
    if span is not None:
        first, last = examine_range(axis, span, first, last)
    return np.arange(first, last + 1)


def _bin_columns(histogram: Histogram, grid: list[Any], edges: bool) -> tuple[Any, Any]:
    """The points' coordinates - centres, or low edges - and their high edges, per axis."""
    columns, uppers = [], []
    for axis, found in zip(histogram.axes, grid):
        lows, ups, _ = bin_edges(axis, found)
        columns.append(lows if edges else axis.root_centers()[found])
        uppers.append(ups)
    return np.stack(columns, axis=1), np.stack(uppers, axis=1)


def from_histogram(histogram: Histogram, options: DataOptions, spans: list[Any]) -> FitData:
    """``FillData`` for a histogram or a profile: its bins, in ROOT's order, as points."""
    edges = options.integral or options.bin_volume
    bins = [_axis_bins(histogram, i, spans[i]) for i in range(len(histogram.axes))]
    grid = [column.ravel() for column in np.meshgrid(*bins, indexing="ij")]
    values = histogram.values(flow=True)[tuple(grid)].astype(np.float64)
    errors = histogram.errors(flow=True)[tuple(grid)].astype(np.float64)
    keep, adjusted = _adjusted(options, errors, values)
    x, upper = _bin_columns(histogram, [found[keep] for found in grid], edges)
    kind = NO_ERROR if options.errors1 else VALUE_ERROR
    data = FitData(x=x, y=values[keep], error=adjusted[keep], kind=kind, options=options)
    if options.errors1:
        data.error = None
    if edges:
        data.upper = upper
        volumes = np.prod(np.abs(upper - x), axis=1)
        data.ref_volume = float(np.min(volumes, initial=np.inf))
    _sums(data)
    return data


# -- graphs ------------------------------------------------------------------------------


def _bars(graph: Any) -> dict[str, Any]:
    """What ``TGraph``'s getters give for every point: ``-1`` where the class keeps no bar.

    ``ex``, ``ey``, ``eyl`` and ``eyh`` are the arrays ``GetEX``, ``GetEY``,
    ``GetEYlow`` and ``GetEYhigh`` return, ``None`` for a class without
    them, which is what ``GetDataType`` asks about.
    """
    n = len(graph)
    none = np.full(n, -1.0)
    found = {"xl": none, "xh": none, "y": none, "yl": none, "yh": none}
    found.update(ex=None, ey=None, eyl=None, eyh=None)
    if graph.classname == SYMMETRIC:
        ex, ey = graph.xerr[0], graph.yerr[0]
        found.update(xl=ex, xh=ex, y=ey, yl=ey, yh=ey, ex=ex, ey=ey)
    elif graph.classname == ASYMMETRIC:
        (xl, xh), (yl, yh) = graph.xerr, graph.yerr
        found.update(xl=xl, xh=xh, yl=yl, yh=yh, eyl=yl, eyh=yh)
        found["y"] = np.sqrt(0.5 * (yl * yl + yh * yh))
    elif graph.classname != "TGraph":
        raise UnsupportedFeatureError(
            f"{graph.name!r} is a {graph.classname}, and a fit reads the errors of a TGraph, "
            f"a TGraphErrors or a TGraphAsymmErrors; a TGraphMultiErrors's layers are one "
            f"error only once you have said how to add them up"
        )
    return found


def _asymmetric_kind(bars: dict[str, Any], options: DataOptions) -> int:
    """``GetDataType`` for asymmetric bars: which of x and y are all zero decides it."""
    zero_x = bool(np.all(bars["xl"] + bars["xh"] <= 0))
    zero_y = bool(np.all(bars["eyl"] + bars["eyh"] <= 0))
    if zero_x and zero_y:
        return NO_ERROR
    if zero_y:
        return COORD_ERROR
    if zero_x:
        options.coord_errors = False
    return ASYM_ERROR


def _has_no_errors(bars: dict[str, Any], options: DataOptions) -> bool:
    """Whether ``GetDataType`` makes it ``kNoError``: told to, or with no y bars at all."""
    return options.errors1 or (bars["ey"] is None and bars["eyl"] is None)


def data_kind(bars: dict[str, Any], options: DataOptions) -> int:
    """``GetDataType``: what kind of data a graph is, from its class and its bars."""
    if _has_no_errors(bars, options):
        return NO_ERROR
    kind = VALUE_ERROR
    if bars["ex"] is not None and options.coord_errors:
        kind = COORD_ERROR if np.any(bars["ex"] > 0) else VALUE_ERROR
    elif bars["eyl"] is not None and options.asym_errors:
        kind = _asymmetric_kind(bars, options)
    zero_y = bars["ey"] is not None and not np.any(bars["ey"] > 0)
    return NO_ERROR if zero_y and kind != COORD_ERROR else kind


def _graph_points(graph: Any, bars: dict[str, Any], data: dict[str, list[Any]], state: Any) -> None:
    """``DoFillData``: one graph's points in order of x, each kept as its kind keeps it."""
    options, kind, span = state
    order = sorted(range(len(graph)), key=lambda i: (graph.x[i], i))
    x, y = graph.x[order], graph.y[order]
    inside = np.ones(len(x), dtype=bool)
    if span is not None:
        inside = (x >= span[0]) & (x <= span[1])
    picked = {name: np.asarray(value)[order] for name, value in bars.items() if value is not None}
    if options.errors1:
        keep, ey = inside, np.ones(len(x))
        ex = np.zeros(len(x))
    elif kind == VALUE_ERROR:
        keep, ey = _adjusted(options, picked["y"], np.ones(len(x)))
        keep &= inside
        ex = np.zeros(len(x))
    else:
        # errors in x too, or asymmetric ones: a point is dropped only when it has neither
        ex = np.maximum(0.5 * (picked["xl"] + picked["xh"]), 0.0) if options.coord_errors else 0 * x
        ey = np.maximum(picked["y"], 0.0)
        keep = inside & ~((ex <= 0) & (ey <= 0))
    for name, column in (("x", x), ("y", y), ("ex", ex), ("ey", ey)):
        data[name].append(column[keep])
    data["yl"].append(picked["yl"][keep])
    data["yh"].append(picked["yh"][keep])


def _joined(columns: dict[str, Any], kind: int, options: DataOptions) -> FitData:
    """The graphs' points as one set, with the errors their kind keeps."""
    made = FitData(
        x=columns["x"].reshape(-1, 1),
        y=columns["y"],
        error=columns["ey"],
        kind=kind,
        options=options,
    )
    if kind == NO_ERROR:
        made.error = None
    if kind >= COORD_ERROR:
        made.xerr = columns["ex"]
    if kind == ASYM_ERROR:
        made.ylow, made.yhigh = columns["yl"], columns["yh"]
    _sums(made)
    return made


def from_graphs(graphs: list[Any], options: DataOptions, span: Any) -> FitData:
    """``FillData`` for a graph, or for every graph of a multigraph, as one set of points."""
    bars = [_bars(graph) for graph in graphs]
    kind = max(data_kind(found, options) for found in bars)
    options.errors1 = kind == NO_ERROR
    options.coord_errors &= kind in (COORD_ERROR, ASYM_ERROR)
    options.asym_errors &= kind == ASYM_ERROR
    data: dict[str, list[Any]] = {
        name: [np.empty(0)] for name in ("x", "y", "ex", "ey", "yl", "yh")
    }
    for graph, found in zip(graphs, bars):
        _graph_points(graph, found, data, (options, kind, span))
    return _joined({name: np.concatenate(parts) for name, parts in data.items()}, kind, options)
