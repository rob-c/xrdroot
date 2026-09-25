"""A histogram as a distribution: ROOT's ``GetCumulative``, ``GetQuantiles`` and ``Smooth``.

Each of these reads a histogram as the shape of something rather than as a
count of it: how much lies below each bin, where a given fraction of it is
passed, and what it looks like with the fluctuations taken out. Each is a
port of ROOT's own loop, down to the order it adds in and what it does at the
edges - the empty bins a quantile falls across, the ends a running median
cannot reach - so the numbers are ROOT's.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np

from .arithmetic import copied, reset
from .errors import UnsupportedFeatureError
from .filling import store_cells
from .reshaping import _low_edge
from .stats import smooth_array

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from .hist import Axis, Histogram

__all__ = ["cumulative", "quantiles", "smooth"]

#: The neighbours ``GetCumulative`` adds and takes away, in its order - z, y,
#: x, then the pairs, then all three - by the axes each steps back along,
#: with the sign it goes in with: inclusion and exclusion.
TERMS = (
    ((2,), 1.0), ((1,), 1.0), ((0,), 1.0),
    ((2, 1), -1.0), ((1, 0), -1.0), ((0, 2), -1.0),
    ((2, 1, 0), 1.0),
)  # fmt: skip


def _one_axis(histogram: Histogram, what: str) -> Axis:
    if len(histogram.axes) != 1:
        raise ValueError(
            f"{what} a histogram of {len(histogram.axes)} axes is not offered, as ROOT offers "
            f"it only for one"
        )
    return histogram.axes[0]


def _counting(histogram: Histogram, what: str) -> None:
    if histogram.kind == "MEAN":
        raise UnsupportedFeatureError(
            f"{what} {histogram.name!r} is not offered: it is a profile, whose bins are means "
            f"rather than counts"
        )


# -- GetCumulative ----------------------------------------------------------


def cumulative(histogram: Histogram, forward: bool, suffix: str) -> Histogram:
    """``TH1::GetCumulative``: each bin the sum of every bin up to it - or, backward, from it.

    With more than one axis it is every bin below and to the left of it,
    worked out by inclusion and exclusion in ROOT's order, each neighbour as
    it was stored, so a ``TH1F`` rounds as ROOT's does. The flow bins are
    left out and left empty; the squares of the weights add up the same way;
    and, each bin being set with ``SetBinContent``, the entries are the
    number of bins and the running sums are made from the bins.
    """
    _counting(histogram, "a cumulative of")
    made = copied(histogram, histogram.name + suffix)
    reset(made)
    source, errors = histogram._bins.astype(np.float64), histogram._variance_cells()
    if len(histogram.axes) == 1:
        _cumulated_line(made, source, errors, forward)
    else:
        _cumulated_grid(made, source, errors, forward)
    made._core["fEntries"] = float(len(histogram))
    made._moment_homes()["fTsumw"]["fTsumw"] = 0.0
    return made


def _cumulated_line(made: Histogram, source: Any, errors: Any, forward: bool) -> None:
    """One axis: a running sum in turn, in the storage's own precision as ROOT stores each."""
    cells, squares = made._cells(), made._sumw2()
    inner = slice(1, len(cells) - 1)
    order = slice(None) if forward else slice(None, None, -1)
    kind = cells.dtype if cells.dtype.kind == "f" else np.dtype(np.float64)
    store_cells(cells[inner], np.add.accumulate(source[inner][order].astype(kind))[order])
    if squares is not None:
        squares[inner] = np.add.accumulate(errors[inner][order])[order]


def _cumulated_grid(made: Histogram, source: Any, errors: Any, forward: bool) -> None:
    """Two or three axes: ``GetCumulative``'s loop, z outermost, a bin at a time.

    Each bin is its own content plus the neighbours inclusion and exclusion
    name, read back as they were stored; the squares of the weights, where
    they are kept, add up alongside.
    """
    cells, squares = made._cells(), made._sumw2()
    held = squares if squares is not None else np.zeros(len(cells))
    walk = _Walk(made._widths, forward)
    for bin, near in walk:
        total, spread = float(source[bin]), float(errors[bin])
        for neighbour, sign in near:
            total += sign * float(cells[neighbour])
            spread += sign * float(held[neighbour])
        store_cells(cells[bin : bin + 1], np.array([total]))
        held[bin] = spread


class _Walk:
    """The bins ``GetCumulative`` visits, in its order, each with the neighbours it takes."""

    def __init__(self, widths: list[int], forward: bool) -> None:
        self.strides = [int(np.prod(widths[:axis])) for axis in range(len(widths))]
        self.terms = [(axes, sign) for axes, sign in TERMS if max(axes) < len(widths)]
        self.step = 1 if forward else -1
        self.edges = [1 if forward else width - 2 for width in widths]
        self.ranges = [range(1, width - 1)[:: self.step] for width in widths]

    def _neighbours(self, at: tuple[int, ...], bin: int) -> list[tuple[int, float]]:
        return [
            (bin - self.step * sum(self.strides[axis] for axis in axes), sign)
            for axes, sign in self.terms
            if all(at[axis] != self.edges[axis] for axis in axes)
        ]

    def __iter__(self) -> Any:
        for backwards in itertools.product(*reversed(self.ranges)):
            at = backwards[::-1]
            bin = sum(index * stride for index, stride in zip(at, self.strides))
            yield bin, self._neighbours(at, bin)


# -- GetQuantiles -----------------------------------------------------------


def _up_edge(axis: Axis, bin: int) -> float:
    """``TAxis::GetBinUpEdge``, which works an edge out as if even past the uneven ones."""
    if not axis.even and 0 < bin <= axis.nbins:
        return float(axis.edges()[bin])
    return axis.low + bin * ((axis.high - axis.low) / axis.nbins)


def _search(integral: np.ndarray[Any, Any], value: float) -> int:
    """``TMath::BinarySearch``: the value's index if it is there, else the one before it."""
    found = int(np.searchsorted(integral, value, side="left"))
    return found if found < len(integral) and integral[found] == value else found - 1


def _landing_on_an_edge(axis: Axis, integral: Any, bin: int, p: float) -> float:
    """A probability the cumulative distribution reaches exactly at the end of a bin.

    Zero goes past every empty bin at the start; one is the last bin's upper
    edge; anything else is the bin's centre - or, if the bins after it are
    empty, the middle of the empty stretch.
    """
    if p == 0.0:
        while bin + 1 <= axis.nbins and integral[bin + 1] == 0.0:
            bin += 1
        return _up_edge(axis, bin)
    if p == 1.0:
        return _up_edge(axis, bin)
    widths, width, later = axis.root_widths(), 0.0, bin + 1
    while p == integral[later]:  # the distribution ends at one, and p is not one
        width += float(widths[later])
        later += 1
    return float(axis.root_centers()[bin]) if width == 0 else _up_edge(axis, bin) + width / 2.0


def _quantile(axis: Axis, integral: Any, p: float) -> float:
    """``GetQuantiles`` for one probability: on an edge, or along a line across a bin."""
    bin = _search(integral[: axis.nbins], p)
    if integral[bin] == p:
        return _landing_on_an_edge(axis, integral, bin, p)
    found = _low_edge(axis, bin + 1)
    rise = integral[bin + 1] - integral[bin]
    # ROOT guards against a bin that does not rise, which a probability
    # found between two points of the distribution cannot land in.
    found += float(axis.root_widths()[bin + 1]) * (p - integral[bin]) / rise if rise > 0 else 0.0
    return float(found)


def quantiles(histogram: Histogram, probabilities: Any) -> np.ndarray[Any, Any]:
    """``TH1::GetQuantiles``: where the distribution passes each of the ``probabilities``.

    The cumulative distribution is ``ComputeIntegral``'s - the bins on the
    axis summed in turn and divided by their total - and a quantile inside a
    bin is on the straight line across it. Without probabilities they are
    those of the cumulative distribution itself, zero first, one per bin and
    one more.
    """
    _counting(histogram, "the quantiles of")
    axis = _one_axis(histogram, "the quantiles of")
    contents = histogram._bins.astype(np.float64)[1 : axis.nbins + 1]
    summed = np.add.accumulate(np.concatenate(([0.0], contents)))
    if summed[-1] == 0:
        raise ValueError(
            f"{histogram.name!r} has nothing in its bins, so it has no distribution to take "
            f"quantiles of"
        )
    integral = summed / summed[-1]
    if probabilities is None:
        wanted = integral.copy()
    else:
        wanted = np.asarray(probabilities, dtype=np.float64).ravel()
        if not np.all((wanted >= 0) & (wanted <= 1)):
            raise ValueError(
                "a quantile is taken at a probability from 0 to 1, and not every one given is"
            )
    return np.array([_quantile(axis, integral, float(p)) for p in wanted])


# -- Smooth -----------------------------------------------------------------


def smooth(histogram: Histogram, ntimes: int) -> None:
    """``TH1::Smooth``: the bins replaced by their 353QH-twice smoothing, in place.

    Only the contents change: the errors, the running sums and the entries
    are left as they were, as ROOT leaves them, so they still describe what
    was filled rather than what is now drawn.
    """
    _counting(histogram, "smoothing")
    axis = _one_axis(histogram, "smoothing")
    if axis.nbins < 3:
        raise ValueError(
            f"smoothing needs at least 3 bins, and {histogram.name!r} has {axis.nbins}"
        )
    cells = histogram._cells()
    inner = slice(1, axis.nbins + 1)
    store_cells(cells[inner], smooth_array(cells[inner].astype(np.float64), ntimes))
