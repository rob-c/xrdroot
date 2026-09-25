"""``TH1::ExtendAxis``: an axis that doubles to take in whatever it is filled with.

The histogram ``TTree::Draw`` books for itself is extendable. Its axis comes
from the first ``GetEstimate()`` values, and when a value after those falls
off either end, ``TAxis::FindBin`` does not send it to the flow bin: it
doubles the axis's range towards the value - again and again, until the value
is inside - and gathers every bin's contents into the bin their centre now
falls in, two old bins to each new one. The number of bins stays the same,
the moments and the count of entries stay what they were, and anything that
had been in the flow bins of that axis is lost, as ROOT warns it is.

A NaN cannot be reached by doubling, and a histogram - not a profile - that
meets one stops being extendable altogether, as ``SetCanExtend(kNoAxis)``
makes it. An infinity, which doubling never reaches either, goes to the flow
bin and leaves the axis as it was.

Which value arrives first decides how far each end moves, so the fills are
made in order: every value up to the next that would move an axis is filled
as an array, the axis is moved, and the filling carries on from there.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .filling import global_bins
from .hist import Axis, Histogram
from .profile import Profile

__all__ = ["fill_in_order", "new_limits"]

Array = Any

#: How many doublings ``FindNewAxisLimits`` tries before it gives up.
DOUBLINGS = 64


def new_limits(low: float, high: float, point: float) -> tuple[float, float] | None:
    """``TH1::FindNewAxisLimits``: the axis doubled until ``point`` is on it, or ``None``.

        >>> new_limits(0.0, 1.0, 2.5)
        (0.0, 4.0)
    """
    if low >= high:
        return None
    width = high - low
    tries = 0
    while point < low:
        if tries > DOUBLINGS:
            return None
        tries += 1
        low -= width
        width *= 2
    while point >= high:
        if tries > DOUBLINGS:
            return None
        tries += 1
        high += width
        width *= 2
    return low, high


def _cell_bins(widths: Sequence[int]) -> list[Array]:
    """For every cell, flow and all, its bin along each axis: x runs fastest."""
    cells = np.arange(int(np.prod(widths)), dtype=np.int64)
    bins = []
    for width in widths:
        bins.append(cells % width)
        cells = cells // width
    return bins


def _moved(histogram: Histogram, old: tuple[Axis, ...], index: int) -> tuple[Array, Array]:
    """Where each old cell's contents go on the new axes, and which of them are kept."""
    bins = _cell_bins(histogram._widths)
    found = [
        new.find_bin(axis.root_centers()[at]) for at, axis, new in zip(bins, old, histogram.axes)
    ]
    extended = bins[index]
    kept = (extended >= 1) & (extended <= old[index].nbins)
    return global_bins(found, histogram._widths), kept


def extend_axis(histogram: Histogram, index: int, point: float) -> bool:
    """``TH1::ExtendAxis`` along axis ``index`` towards ``point``: whether it could.

    A histogram's bins are gathered where their content is not zero; a
    profile's where the sum of the weights is not, as ``TProfileHelper``
    has it. What was in the flow bins of the axis moved is dropped.
    """
    axis = histogram.axes[index]
    found = new_limits(axis.low, axis.high, point)
    if found is None:
        return False
    old = histogram.axes
    row = histogram._core[f"f{'XYZ'[index]}axis"]
    row["fXmin"], row["fXmax"] = found
    histogram.axes = tuple(Axis(histogram._core[f"f{letter}axis"]) for letter in "XYZ"[: len(old)])
    target, kept = _moved(histogram, old, index)
    profile = isinstance(histogram, Profile)
    arrays = [histogram._cells(), *histogram._per_cell()]
    kept &= (arrays[2] if profile else arrays[0]) != 0
    for cells in arrays:
        held = cells[kept].copy()
        cells[:] = 0
        np.add.at(cells, target[kept], held)
    return True


def _first_off(histogram: Histogram, coordinates: Sequence[Array], start: int) -> int | None:
    """The first fill from ``start`` that would move an axis, or ``None`` if none would.

    A histogram also stops at a NaN, which ends its extending altogether.
    """
    moves = np.zeros(len(coordinates[0]) - start, dtype=bool)
    for axis, values in zip(histogram.axes, coordinates):
        part = values[start:]
        moves |= np.isfinite(part) & ((part < axis.low) | (part >= axis.high))
        if not isinstance(histogram, Profile):
            moves |= np.isnan(part)
    found = np.flatnonzero(moves)
    return start + int(found[0]) if len(found) else None


def _meet(histogram: Histogram, point: Sequence[float]) -> bool:
    """``FindBin`` along each axis in turn for one fill: whether the axes can still extend."""
    for index, (axis, value) in enumerate(zip(histogram.axes, point)):
        if np.isnan(value) and not isinstance(histogram, Profile):
            return False
        if np.isfinite(value) and not axis.low <= value < axis.high:
            extend_axis(histogram, index, value)
    return True


def _fill(histogram: Histogram, coordinates: Sequence[Array], values: Any, weights: Array) -> None:
    given = [*coordinates] if values is None else [*coordinates, values]
    histogram.fill(*given, weight=weights)


def fill_in_order(
    histogram: Histogram,
    coordinates: Sequence[Array],
    values: Array | None,
    weights: Array,
    extendable: bool,
) -> bool:
    """Fill as ROOT would one entry at a time, extending the axes if they are ``extendable``.

    ``values`` are what a profile averages, and ``None`` for a histogram.
    What comes back is whether the axes can still extend, which a NaN
    filling a histogram puts an end to.
    """
    start, count = 0, len(weights)
    while extendable and start < count:
        stop = _first_off(histogram, coordinates, start)
        if stop is None:
            break
        _fill_part(histogram, coordinates, values, weights, slice(start, stop))
        extendable = _meet(histogram, [float(c[stop]) for c in coordinates])
        _fill_part(histogram, coordinates, values, weights, slice(stop, stop + 1))
        start = stop + 1
    _fill_part(histogram, coordinates, values, weights, slice(start, count))
    return extendable


def _fill_part(
    histogram: Histogram,
    coordinates: Sequence[Array],
    values: Array | None,
    weights: Array,
    part: slice,
) -> None:
    """Fill the fills in ``part``, which a histogram does all at once."""
    _fill(
        histogram,
        [c[part] for c in coordinates],
        None if values is None else values[part],
        weights[part],
    )
