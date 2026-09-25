"""Statistics: what ROOT's ``GetMean``, ``GetStdDev`` and ``Integral`` give back.

ROOT does not work a histogram's mean out of its bins. Every fill adds to a
handful of running sums - of the weights, their squares, and the weights times
each coordinate and its square - and the mean, the spread and the effective
number of entries are made from those, so they are the mean of what was
filled rather than of the bin centres it landed near. ``GetStats`` hands the
sums over, and everything here starts from it, exactly as ROOT's does.

Only when the sums are gone does ROOT fall back on the bins: a histogram whose
contents were set by hand, or whose statistics were reset, has a total weight
of zero and entries all the same, and then the sums are made again from the
bin centres and contents. That fallback is here too, with ROOT's own centres
and ROOT's order of adding, so a mean made from bins is ROOT's to the bit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .filling import arrays, global_bins, running

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from .hist import Histogram

__all__ = [
    "NAMES",
    "PROFILE_NAMES",
    "central_moment",
    "effective_entries",
    "extreme",
    "extreme_index",
    "find_bin",
    "integral",
    "interpolate",
    "mean",
    "mean_error",
    "statistics",
    "std",
    "std_error",
]

#: ``kNstat``: how many numbers ``GetStats`` fills, whatever the class.
NSTAT = 13

#: The moments ``GetStats`` gives for a histogram of one, two and three
#: dimensions, in its order: the sums of the weights and their squares, then
#: of the weights times each coordinate, its square, and each product of two.
NAMES = {
    1: ("fTsumw", "fTsumw2", "fTsumwx", "fTsumwx2"),
    2: ("fTsumw", "fTsumw2", "fTsumwx", "fTsumwx2", "fTsumwy", "fTsumwy2", "fTsumwxy"),
    3: (
        "fTsumw", "fTsumw2", "fTsumwx", "fTsumwx2", "fTsumwy", "fTsumwy2", "fTsumwxy",
        "fTsumwz", "fTsumwz2", "fTsumwxz", "fTsumwyz",
    ),
}  # fmt: skip

#: A profile's, which add the sums of what is averaged and of its square.
PROFILE_NAMES = {
    1: (*NAMES[1], "fTsumwy", "fTsumwy2"),
    2: (*NAMES[2], "fTsumwz", "fTsumwz2"),
    3: (*NAMES[3], "fTsumwt", "fTsumwt2"),
}

#: Where the sum of ``w`` times each axis's coordinate sits in ``GetStats``,
#: x first: its square is the next one along. The fourth is what a
#: three-dimensional profile averages.
INDEX = (2, 4, 7, 11)


def statistics(histogram: Histogram) -> list[float]:
    """ROOT's ``GetStats``: the running sums, or the same made from the bins.

    The bins are the fallback ROOT takes when the sums have been thrown away,
    which it can tell from a total weight of zero; the sums from bins leave
    out the flow bins, as the running sums leave out what fell off the axes.
    """
    names = histogram._moment_names()
    homes = histogram._moment_homes()
    found = [float(homes[name][name]) for name in names]
    if histogram._from_bins(found[0]):
        terms = histogram._bin_terms()
        found = [running(0.0, terms[name]) for name in names]
    return found + [0.0] * (NSTAT - len(found))


def _index(histogram: Histogram, axis: int) -> int:
    """Where one axis's sums are in ``GetStats``, refusing an axis there is not."""
    count = histogram._moment_axes()
    if not 0 <= axis < count:
        raise ValueError(
            f"axis={axis} is not an axis of {histogram.name!r}, which has {count}, counted "
            f"from zero"
        )
    return INDEX[axis]


def mean(histogram: Histogram, axis: int) -> float:
    """``GetMean``: the weighted mean of what was filled along one axis."""
    at = _index(histogram, axis)
    found = statistics(histogram)
    return found[at] / found[0] if found[0] else 0.0


def _variance(histogram: Histogram, axis: int) -> float:
    at = _index(histogram, axis)
    found = statistics(histogram)
    if not found[0]:
        return 0.0
    average = found[at] / found[0]
    return abs(found[at + 1] / found[0] - average * average)


def std(histogram: Histogram, axis: int) -> float:
    """``GetStdDev``: the spread, from the same sums, never below zero."""
    return math.sqrt(_variance(histogram, axis))


def effective_entries(histogram: Histogram) -> float:
    """``GetEffectiveEntries``: ``(sum w)**2 / sum w**2``, the count as good as the weights."""
    found = statistics(histogram)
    return found[0] * found[0] / found[1] if found[1] else abs(found[0])


def mean_error(histogram: Histogram, axis: int) -> float:
    """``GetMeanError``: the spread over the root of the effective entries."""
    spread, count = std(histogram, axis), effective_entries(histogram)
    return spread / math.sqrt(count) if count > 0 else 0.0


def std_error(histogram: Histogram, axis: int) -> float:
    """``GetStdDevError``: the Gaussian ``sigma / sqrt(2 n)`` ROOT quotes."""
    variance, count = _variance(histogram, axis), effective_entries(histogram)
    return math.sqrt(variance / (2 * count)) if count > 0 else 0.0


def _looped(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Bins in the order ``GetSkewness`` visits them: x outermost, then z, then y."""
    order = (0, 2, 1)[: values.ndim] if values.ndim == 3 else tuple(range(values.ndim))
    return np.ascontiguousarray(values.transpose(order)).ravel()


def central_moment(histogram: Histogram, axis: int, power: int) -> float:
    """``GetSkewness`` for a power of three and ``GetKurtosis`` for four, as ROOT has them.

    The mean and the spread are the running ones; the sum over the bins is
    of the bin contents at the bin centres, in ROOT's order, and divided as
    ROOT divides - so a histogram with nothing in it gives NaN, as it does.
    """
    if not 0 <= axis < len(histogram.axes):
        raise ValueError(
            f"axis={axis} is not a binned axis of {histogram.name!r}, which has "
            f"{len(histogram.axes)}, counted from zero"
        )
    middle, spread = mean(histogram, axis), std(histogram, axis)
    values = histogram.values().astype(np.float64)
    shape = [1] * values.ndim
    shape[axis] = -1
    centres = np.broadcast_to(
        histogram.axes[axis].root_centers()[1:-1].reshape(shape), values.shape
    )
    weights, apart = _looped(values), _looped(centres) - middle
    terms = weights * apart * apart * apart
    scale = spread * spread * spread
    if power == 4:
        terms, scale = terms * apart, scale * spread
    with np.errstate(all="ignore"):
        found = np.float64(running(0.0, terms)) / (running(0.0, weights) * scale)
    return float(found) - (3.0 if power == 4 else 0.0)


def _range(low: Any, high: Any, nbins: int) -> tuple[int, int]:
    """One axis's bins to sum, clamped the way ``TH1::DoIntegral`` clamps them."""
    first = 1 if low is None else max(int(low), 0)
    last = nbins if high is None else int(high)
    if last > nbins + 1 or last < first:
        last = nbins + 1
    return first, last


def _per_axis(given: Any, count: int) -> Sequence[Any]:
    """One bound per axis: a single number is x's, and the other axes take their default."""
    if given is None or np.ndim(given) == 0:
        return [given] + [None] * (count - 1)
    return list(given)


def integral(histogram: Histogram, low: Any, high: Any, width: bool) -> tuple[float, float]:
    """``TH1::IntegralAndError``: the sum of the bins in a range, and its error.

    ``low`` and ``high`` are ROOT's bin numbers, the flow bins zero and
    ``nbins + 1``, each one number for one axis or one per axis; the default
    is every bin on the axis and none of its flow. ``width`` multiplies each
    bin by its width - its area, its volume - first. The sums are added in
    ROOT's order, x outermost.
    """
    count = len(histogram.axes)
    lows, highs = _per_axis(low, count), _per_axis(high, count)
    ranges = [_range(a, b, axis.nbins) for a, b, axis in zip(lows, highs, histogram.axes)]
    cut = tuple(slice(first, last + 1) for first, last in ranges)
    values = histogram.values(flow=True).astype(np.float64)[cut]
    squares = histogram.variances(flow=True)[cut]
    if width:
        size = np.ones(())
        for axis, (first, last) in zip(histogram.axes, ranges):
            size = np.multiply.outer(size, axis.root_widths()[first : last + 1])
        values, squares = values * size, squares * size * size
    return running(0.0, values.ravel()), math.sqrt(running(0.0, squares.ravel()))


def find_bin(histogram: Histogram, coordinates: Sequence[Any]) -> Any:
    """``TH1::FindBin``: ROOT's global bin number, flow counted, x fastest."""
    if len(coordinates) != len(histogram.axes):
        raise ValueError(
            f"{histogram.name!r} has {len(histogram.axes)} axes, and a bin is found from "
            f"one coordinate per axis, not {len(coordinates)}"
        )
    scalar = all(np.ndim(value) == 0 for value in coordinates)
    flat, _weights = arrays(coordinates, None)
    bins = [axis.find_bin(value) for axis, value in zip(histogram.axes, flat)]
    found = global_bins(bins, histogram._widths)
    if scalar:
        return int(found[0])
    return found.reshape(np.broadcast(*coordinates).shape)


def interpolate(histogram: Histogram, x: Any) -> Any:
    """``TH1::Interpolate``: a straight line between the two nearest bin centres.

    Below the first centre it is the first bin's content and above the last
    the last's, as ROOT has it; a NaN is a NaN.
    """
    if len(histogram.axes) != 1:
        raise ValueError(
            f"interpolating {histogram.name!r} along one axis needs one axis, and it has "
            f"{len(histogram.axes)}"
        )
    axis = histogram.axes[0]
    points = np.asarray(x, dtype=np.float64)
    centres, values = axis.root_centers(), histogram.values(flow=True).astype(np.float64)
    found = axis.find_bin(points)
    low = np.clip(np.where(points <= centres[found], found - 1, found), 0, axis.nbins)
    y0, y1, x0, x1 = values[low], values[low + 1], centres[low], centres[low + 1]
    line = y0 + (points - x0) * ((y1 - y0) / (x1 - x0))
    ends = np.where(points >= centres[axis.nbins], values[axis.nbins], line)
    result = np.where(points <= centres[1], values[1], ends)
    return float(result) if result.ndim == 0 else result


def extreme(histogram: Histogram, highest: bool) -> float:
    """``GetMaximum`` or ``GetMinimum``: what was set with ``SetMaximum``, or the bins'.

    ROOT keeps -1111 for a limit that was never set, and then looks at every
    bin on the axes - the flow bins not among them.
    """
    stored = float(histogram._core["fMaximum" if highest else "fMinimum"])
    if stored != -1111.0:
        return stored
    values = histogram.values()
    return float(values.max() if highest else values.min())


def extreme_index(histogram: Histogram, highest: bool) -> Any:
    """Where the largest or smallest bin is, as an index into ``values()``.

    The first one ROOT's loop would meet when two are equal, which is the
    first with x running fastest.
    """
    values = histogram.values()
    flat = values.ravel(order="F")
    at = int(np.argmax(flat) if highest else np.argmin(flat))
    found = tuple(int(i) for i in np.unravel_index(at, values.shape, order="F"))
    return found[0] if len(found) == 1 else found
