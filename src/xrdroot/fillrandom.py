"""``TH1::FillRandom``: a histogram filled from a function or another histogram, as ROOT fills it.

From a function, ROOT integrates it over each bin of the histogram's axis
range - in closed form for ``gaus``, ``expo``, ``landau`` and ``polN``,
numerically otherwise - adds the integrals up in bin order into a
cumulative table, and for each ``Rndm()`` finds the bin with
``TMath::BinarySearch`` and fills the point on the straight line across it.
The table is the function's integrals, not its value at the bin centres,
and the points filled are those, not the centres: a histogram filled here
from ROOT's seed is ROOT's histogram, entry for entry and sum for sum.

From a histogram, ROOT draws each entry with ``TH1::GetRandom`` - unless
more than ten entries a bin are asked for, when it draws a Poisson count
for every bin, of mean the source's share times ``n``, and then draws single
entries with ``GetRandom`` to add or take away until there are exactly
``n``, and makes the running sums again from the bins. Both ways are here,
draw for draw. ROOT's own code draws the entries it adds from ``gRandom``
even when it was given another generator; here they come from the
generator given, which is the same thing whenever that is ``gRandom``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .errors import UnsupportedFeatureError
from .filling import add_to_cells

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from .function import Function
    from .hist import Axis, Histogram

__all__ = ["bin_edges", "fill_random", "first_last", "from_parent", "standard_function"]

#: ``TAxis::kAxisRange``: the axis has a range set, from ``fFirst`` to ``fLast``.
AXIS_RANGE = 1 << 11
#: Past this many entries a bin, ``FillRandom(h, n)`` draws a Poisson count per bin.
POISSON_PER_BIN = 10
#: The functions ``TF1::InitStandardFunctions`` and ``TF2::InitStandardFunctions`` put
#: in ``gROOT``'s list, with the parameters they are given: each over ``(-1, 1)``,
#: along each of its axes.
STANDARD: dict[str, tuple[float, ...]] = {
    "gaus": (1.0, 0.0, 1.0),
    "gausn": (1.0, 0.0, 1.0),
    "landau": (1.0, 0.0, 1.0),
    "landaun": (1.0, 0.0, 1.0),
    "expo": (1.0, 1.0),
    **{f"pol{degree}": (1.0,) * (degree + 1) for degree in range(10)},
    "xygaus": (1.0, 0.0, 1.0, 0.0, 1.0),
    "bigaus": (1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
    "xyexpo": (1.0, 0.0, 1.0),
    "xylandau": (1.0, 0.0, 1.0, 1.0, 0.0),
    "xylandaun": (1.0, 0.0, 1.0, 1.0),
}
#: The range ``InitStandardFunctions`` gives each of them, along each axis.
STANDARD_RANGE = (-1.0, 1.0)
#: The standard functions of two variables.
STANDARD_2D = ("xygaus", "bigaus", "xyexpo", "xylandau", "xylandaun")


def standard_function(name: str) -> Function:
    """``gROOT->GetFunction(name)`` for one of ROOT's standard functions, as ROOT makes it."""
    from .function import Function

    span = (STANDARD_RANGE, STANDARD_RANGE) if name in STANDARD_2D else STANDARD_RANGE
    return Function(name, name, range=span, parameters=STANDARD[name])


def first_last(row: dict[str, Any]) -> tuple[int, int]:
    """``TAxis::GetFirst`` and ``GetLast``: the axis's range if one is set, else every bin."""
    nbins = int(row["fNbins"])
    bits = int(row.get("TNamed", {}).get("fBits", 0) or 0)
    first, last = int(row.get("fFirst", 0) or 0), int(row.get("fLast", 0) or 0)
    if bits & AXIS_RANGE and 1 <= first <= last <= nbins:
        return first, last
    return 1, nbins


def bin_edges(axis: Axis, bins: Any) -> tuple[Any, Any, Any]:
    """``GetBinLowEdge``, ``GetBinUpEdge`` and ``GetBinWidth`` of bins, as ``TAxis`` has them.

    An even axis works each out from its ends - the upper edge as the next
    bin's lower one, not as the lower edge plus the width - which is where
    the last bit of each comes from in ROOT.
    """
    bins = np.asarray(bins, dtype=np.int64)
    if axis.even:
        width = (axis.high - axis.low) / axis.nbins
        low, up = axis.low + (bins - 1) * width, axis.low + bins * width
        return low, up, np.full(len(bins), width)
    edges = axis.edges()
    return edges[bins - 1], edges[bins], edges[bins] - edges[bins - 1]


def _one_axis(histogram: Histogram) -> Axis:
    if len(histogram.axes) != 1:
        raise UnsupportedFeatureError(
            f"{histogram.name!r} has {len(histogram.axes)} axes, and filling one at random is "
            f"done here for a histogram of one axis, TH1::FillRandom; TH2's and TH3's integrate "
            f"a TF2 or TF3 over every cell, which is not here"
        )
    return histogram.axes[0]


def _cumulative(steps: Any, what: str) -> Any:
    """The running sum of the bins' integrals over its total, refusing a total of zero."""
    total = np.add.accumulate(np.concatenate(([0.0], np.asarray(steps, dtype=np.float64))))
    if total[-1] == 0:
        raise ValueError(
            f"{what} integrates to zero over the histogram's bins, so there is nothing to draw "
            f"from; ROOT's FillRandom stops with an error here"
        )
    total[1:] = total[1:] / total[-1]
    return total


def _from_function(histogram: Histogram, function: Function, count: int, rng: Any) -> None:
    axis = _one_axis(histogram)
    if function.dimensions != 1:
        raise UnsupportedFeatureError(
            f"{function.name!r} is a function of {function.dimensions} variables, and "
            f"{histogram.name!r} has one axis to fill"
        )
    first, last = first_last(histogram._core["fXaxis"])
    lows, ups, widths = bin_edges(axis, np.arange(first, last + 1))
    integral = _cumulative(
        [function.integral(float(a), float(b), 0.0) for a, b in zip(lows, ups)],
        repr(function.name),
    )
    draws = np.asarray(rng.rndm(count), dtype=np.float64)
    step = _search(integral, len(lows), draws)
    below = integral[step]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = lows[step] + widths[step] * (draws - below) / (integral[step + 1] - below)
    histogram.fill(x)


def _search(integral: Any, n: int, r: Any) -> Any:
    """``TMath::BinarySearch(n, integral, r)``: the first equal entry, else the one below."""
    at = np.searchsorted(integral[:n], r, side="left")
    exact = (at < n) & (integral[np.minimum(at, n - 1)] == r)
    return np.where(exact, at, at - 1)


def from_parent(
    cells: Any, squares: Any, axis: Axis, parent: Any, parent_axis: Axis, count: int, rng: Any
) -> None:
    """ROOT's Poisson way of drawing ``count`` entries from ``parent`` into ``cells``.

    ``cells`` are the bins of one axis, flow and all, and ``squares`` their
    sums of squared weights or ``None``; ``parent`` is the source's bins
    without their flow, binned as ``cells`` are. Every bin gets a Poisson
    count - added to its square too, as ROOT adds it - and then single
    entries drawn with ``GetRandom`` are added, or taken away from bins
    that have one, until the total is ``count``.
    """
    nbins = axis.nbins
    sumw = float(np.add.accumulate(np.asarray(parent, dtype=np.float64))[-1])
    means = np.asarray(parent, dtype=np.float64) * count / sumw
    counts = np.zeros(len(cells))
    counts[1 : nbins + 1] = [rng.poisson(float(mean)) for mean in means]
    add_to_cells(cells, counts)
    if squares is not None:
        squares += counts
    made = int(float(np.add.accumulate(counts)[-1]) + 0.5)
    if made < count:
        extra = axis.find_bin(rng.from_distribution(parent, parent_axis, count - made))
        np.add.at(cells, extra, 1)
        if squares is not None:
            np.add.at(squares, extra, 1.0)
    while made > count:
        for found in axis.find_bin(rng.from_distribution(parent, parent_axis, made - count)):
            if cells[found] > 0:
                cells[found] = cells[found] - 1
                made -= 1


def _from_histogram(histogram: Histogram, source: Histogram, count: int, rng: Any) -> None:
    axis, parent_axis = _one_axis(histogram), _one_axis(source)
    parent = source.values().astype(np.float64)
    if not np.all(parent >= 0):
        raise ValueError(
            f"{source.name!r} has a negative or NaN bin, so it is not a distribution to draw "
            f"from; ROOT's FillRandom refuses it too"
        )
    if not parent.any():
        raise ValueError(f"{source.name!r} is empty, so there is nothing to draw from")
    first, last = first_last(histogram._core["fXaxis"])
    if count <= POISSON_PER_BIN * (last - first + 1):
        histogram.fill(rng.from_distribution(parent, parent_axis, count))
        return
    if axis != parent_axis or (first, last) != (1, axis.nbins):
        raise ValueError(
            f"{count} entries is more than ten a bin, which ROOT fills by a Poisson count per "
            f"bin, and that needs {histogram.name!r} and {source.name!r} binned the same over "
            f"the whole axis; ROOT does nothing at all when they are not"
        )
    from .arithmetic import reset_statistics

    from_parent(histogram._cells(), histogram._sumw2(), axis, parent, parent_axis, count, rng)
    reset_statistics(histogram)


def fill_random(histogram: Histogram, source: Any, count: int, rng: Any) -> None:
    """``TH1::FillRandom(source, count, rng)``: from a function, a shape's name, or a histogram."""
    from .function import Function
    from .hist import Histogram

    if rng is None:
        from .random import gRandom as rng
    count = int(count)
    if count < 0:
        raise ValueError(f"{count} is not a number of entries to fill")
    if isinstance(source, Histogram):
        _from_histogram(histogram, source, count, rng)
        return
    if isinstance(source, str):
        source = _named(histogram, source)
    if not isinstance(source, Function):
        raise TypeError(
            f"FillRandom draws from a Function, the name of one of ROOT's shapes, or a "
            f"Histogram, not {type(source).__name__}"
        )
    _from_function(histogram, source, count, rng)


def _named(histogram: Histogram, name: str) -> Function:
    """``gROOT->GetFunction(name)`` for a standard shape; any other formula over the axis."""
    from .function import Function

    if name in STANDARD:
        return standard_function(name)
    axis = _one_axis(histogram)
    return Function(name, name, range=(axis.low, axis.high))
