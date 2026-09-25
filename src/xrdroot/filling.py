"""Filling: ROOT's bookkeeping for every entry, done an array at a time.

``TH1::Fill`` is a dozen lines, and every one of them is a promise about the
numbers in the file afterwards: the entry is counted whether or not it lands
on the axis, the bin gets the weight and the sum of squares its square, the
sums of squares come into being at the first weight that is not one, and the
moments - the sums of ``w``, ``w*x``, ``w*x*x`` and the rest - take only what
fell inside the axes. This keeps each of those promises for a whole array at
once, and in the same order ROOT would have met the entries one by one.

The order matters to the last bit. A double sum depends on the order it is
added in, and a sum NumPy is left to itself adds pairwise, which is more
accurate than ROOT and therefore not ROOT; so every running total here is
added up strictly in turn, the way a loop over fills adds it, and a histogram
filled here holds the same doubles one ROOT filled with the same entries. The
bins are added to in turn too, which is what ``numpy.add.at`` does.

What falls outside an axis goes to its flow bin, the upper edge of the last
bin included; a NaN goes to the overflow, because ROOT's ``FindBin`` asks
``!(x < xmax)`` and a NaN is never below anything. Integer storage saturates
as ROOT's does, a bin of a ``TH1C`` stopping at 127 however much more is put
in it, and a weight going into one is cut to a whole number first.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from .hist import Axis, Histogram
    from .profile import Profile

__all__ = [
    "add_to_cells",
    "axis_terms",
    "arrays",
    "fill_histogram",
    "fill_profile",
    "global_bins",
    "running",
    "store_cells",
]

#: How far each integer storage goes before it saturates: ROOT stops one
#: short of the type's most negative value, so a bin never flips sign.
LIMITS = {
    np.dtype(np.int8): 127,
    np.dtype(np.int16): 32767,
    np.dtype(np.int32): 2147483647,
    np.dtype(np.int64): 9223372036854775807,
}

#: The widest step a weight can take into an integer bin: past this it only
#: saturates, and cutting it here keeps the conversion to a whole number defined.
_STEP = float(2**62)


def running(start: float, values: Any) -> float:
    """``start`` with every value added to it in turn, as a loop over fills adds.

    ``numpy.add.accumulate`` is sequential by definition - every partial sum
    is an answer - so its last partial sum is the loop's to the last bit.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if not len(values):
        return float(start)
    return float(np.add.accumulate(np.concatenate(([float(start)], values)))[-1])


def arrays(given: Sequence[Any], weight: Any) -> tuple[list[np.ndarray[Any, Any]], Any]:
    """The coordinates and the weights of the fills, as flat arrays of one length.

    Scalars and arrays mix as NumPy broadcasts them, so one weight goes with
    every entry and one ``x`` with every ``y``. No weight is a weight of one,
    which is what ROOT's ``Fill(x)`` adds - to the last bit, as it happens,
    ``1.0 * x`` being ``x``.
    """
    made = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in given),
        np.asarray(1.0 if weight is None else weight, dtype=np.float64),
    )
    flat = [np.ravel(value) for value in made]
    return flat[:-1], flat[-1]


def global_bins(bins: Sequence[Any], widths: Sequence[int]) -> Any:
    """ROOT's global bin number from one per axis: x runs fastest, flow counted."""
    cell = np.zeros_like(bins[0])
    for found, stride in zip(reversed(bins), reversed(widths)):
        cell = cell * stride + found
    return cell


def _inside(bins: Sequence[Any], axes: Sequence[Axis]) -> Any:
    """Which fills landed on every axis, rather than in a flow bin of one."""
    inside = np.ones(len(bins[0]), dtype=bool)
    for found, axis in zip(bins, axes):
        inside &= (found >= 1) & (found <= axis.nbins)
    return inside


def _unit_until(weights: Any) -> int:
    """How many fills come before the first whose weight is not one."""
    heavier = np.flatnonzero(weights != 1.0)
    return int(heavier[0]) if len(heavier) else len(weights)


def _steps(weights: Any) -> Any:
    """What each weight adds to an integer bin: ``Int_t(w)``, cut towards zero."""
    return np.trunc(np.clip(np.nan_to_num(weights), -_STEP, _STEP)).astype(np.int64)


def _fill_cells(cells: Any, index: Any, weights: Any) -> None:
    """Add each weight to its bin, in order, the way the storage's ``AddBinContent`` does.

    A float bin adds the weight in its own precision, as a ``TH1F`` adds a
    ``Float_t``. An integer bin saturates, and adding a run of steps of one
    sign saturates where adding them one by one would; only a run of steps of
    both signs, which can leave the limit and come back, is walked in turn.
    """
    if cells.dtype.kind == "f":
        np.add.at(cells, index, weights.astype(cells.dtype))
        return
    steps, top = _steps(weights), LIMITS[cells.dtype]
    if np.all(steps >= 0) or np.all(steps <= 0):
        totals = np.zeros(len(cells), dtype=np.int64)
        np.add.at(totals, index, steps)
        cells[:] = np.clip(cells + totals, -top, top)
        return
    for at, step in zip(index.tolist(), steps.tolist()):
        cells[at] = min(max(int(cells[at]) + step, -top), top)


def add_to_cells(cells: Any, values: Any) -> None:
    """Add one value to every bin at once: ``AddBinContent`` for each cell."""
    if cells.dtype.kind == "f":
        cells += np.asarray(values, dtype=np.float64).astype(cells.dtype)
        return
    top = LIMITS[cells.dtype]
    cells[:] = np.clip(cells + _steps(values), -top, top)


def store_cells(cells: Any, values: Any) -> None:
    """Replace every bin at once: ``UpdateBinContent``, converting as C++ converts.

    An integer storage takes the whole number towards zero, kept inside what
    the type holds rather than wrapped round it.
    """
    values = np.asarray(values, dtype=np.float64)
    if cells.dtype.kind == "f":
        cells[:] = values.astype(cells.dtype)
        return
    top = LIMITS[cells.dtype]
    cells[:] = np.clip(_steps(values), -top, top)


def axis_terms(weights: Any, coordinates: Sequence[Any]) -> list[tuple[str, Any]]:
    """The moments one set of fills adds to, with what each fill adds to it.

    Each product is formed in ROOT's order - ``w*x*x`` is ``(w*x)*x`` - so it
    is the same double ROOT added.
    """
    letters = "xyz"
    terms = [("fTsumw", weights), ("fTsumw2", weights * weights)]
    for letter, value in zip(letters, coordinates):
        terms += [
            (f"fTsumw{letter}", weights * value),
            (f"fTsumw{letter}2", weights * value * value),
        ]
    for first in range(len(coordinates)):
        for second in range(first + 1, len(coordinates)):
            product = weights * coordinates[first] * coordinates[second]
            terms.append((f"fTsumw{letters[first]}{letters[second]}", product))
    return terms


def _add_moments(homes: dict[str, dict[str, Any]], terms: list[tuple[str, Any]]) -> None:
    for name, values in terms:
        home = homes[name]
        home[name] = running(home[name], values)


def fill_histogram(histogram: Histogram, coordinates: Sequence[Any], weights: Any) -> None:
    """``TH1::Fill``, ``TH2::Fill`` or ``TH3::Fill`` for every entry, in order."""
    bins = [axis.find_bin(value) for axis, value in zip(histogram.axes, coordinates)]
    cells = global_bins(bins, histogram._widths)
    inside = _inside(bins, histogram.axes)
    split = len(weights) if histogram._sumw2() is not None else _unit_until(weights)
    for part in (slice(0, split), slice(split, None)):
        if part.start == split and split < len(weights):
            histogram._ensure_sumw2()  # before the first weight that is not one
        _histogram_part(histogram, cells[part], weights[part], coordinates, inside, part)


def _histogram_part(
    histogram: Histogram,
    cells: Any,
    weights: Any,
    coordinates: Sequence[Any],
    inside: Any,
    part: slice,
) -> None:
    core = histogram._core
    core["fEntries"] = float(core["fEntries"]) + len(cells)
    _fill_cells(histogram._cells(), cells, weights)
    squares = histogram._sumw2()
    if squares is not None:
        np.add.at(squares, cells, weights * weights)
    kept = inside[part]
    placed = [value[part][kept] for value in coordinates]
    _add_moments(histogram._moment_homes(), axis_terms(weights[kept], placed))


def fill_profile(profile: Profile, coordinates: Sequence[Any], values: Any, weights: Any) -> None:
    """``TProfile::Fill`` and its two- and three-dimensional kin, for every entry.

    A profile given a range for what it averages drops a value outside it -
    or a NaN - before counting it at all, as ROOT does.
    """
    low, high = profile._value_range()
    if low != high:
        kept = ~((values < low) | (values > high) | np.isnan(values))
        coordinates = [value[kept] for value in coordinates]
        values, weights = values[kept], weights[kept]
    bins = [axis.find_bin(value) for axis, value in zip(profile.axes, coordinates)]
    cells = global_bins(bins, profile._widths)
    inside = _inside(bins, profile.axes)
    split = len(weights) if profile._bin_sumw2() is not None else _unit_until(weights)
    for part in (slice(0, split), slice(split, None)):
        if part.start == split and split < len(weights):
            profile._ensure_sumw2()  # before the first weight that is not one
        _profile_part(profile, cells[part], values[part], weights[part], coordinates, inside, part)


def _profile_part(
    profile: Profile,
    cells: Any,
    values: Any,
    weights: Any,
    coordinates: Sequence[Any],
    inside: Any,
    part: slice,
) -> None:
    core = profile._core
    core["fEntries"] = float(core["fEntries"]) + len(cells)
    np.add.at(profile._cells(), cells, weights * values)
    np.add.at(profile._value_squares(), cells, weights * values * values)
    squares = profile._bin_sumw2()
    if squares is not None:
        np.add.at(squares, cells, weights * weights)
    np.add.at(profile._bin_weights(), cells, weights)
    kept = inside[part]
    placed = [value[part][kept] for value in coordinates]
    heavy, averaged = weights[kept], values[kept]
    letter = profile._value_letter()
    terms = axis_terms(heavy, placed)
    terms += [
        (f"fTsumw{letter}", heavy * averaged),
        (f"fTsumw{letter}2", heavy * averaged * averaged),
    ]
    _add_moments(profile._moment_homes(), terms)
