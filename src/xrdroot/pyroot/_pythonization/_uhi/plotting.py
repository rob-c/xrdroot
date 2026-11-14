"""UHI's plotting protocol on ROOT's histograms: what ``mplhep`` and ``hist`` read of one.

A histogram is a :class:`uhi.typing.plottable.PlottableHistogram`: its
``kind`` (``COUNT``, or ``MEAN`` for a profile), its ``axes``, and its
``values``, ``variances`` and ``counts`` as arrays shaped as the axes are,
x first. ``values()`` is the histogram's own bins seen through NumPy,
read-only unless ``writable=True`` is asked for, when writing to it writes
the bins. A histogram of bytes or a profile cannot be seen that way - a
profile's bins are means worked out when asked - so theirs is a copy, and
asking for it writable is refused, as ROOT refuses it.
"""

from __future__ import annotations

import enum
from collections.abc import Iterator
from itertools import product
from typing import Any

import numpy as np

from .tags import _get_axis, _get_axis_len

__all__: list[str] = []


class Kind(str, enum.Enum):
    """What a bin holds: a count of entries, or the mean of a value."""

    COUNT = "COUNT"
    MEAN = "MEAN"


class PlottableAxisTraits:
    """A ROOT axis is never circular, always has its two flow bins, and is discrete only
    when every bin has a label."""

    def __init__(self, circular: bool = False, discrete: bool = False) -> None:
        self.circular = circular
        self.discrete = discrete
        self.underflow = True
        self.overflow = True


class PlottableAxisContinuous:
    """One ``TAxis`` as UHI's axis of ranges: each bin its ``(low, high)`` edges."""

    discrete = False

    def __init__(self, tAxis: Any) -> None:
        self.tAx = tAxis

    @property
    def traits(self) -> PlottableAxisTraits:
        return PlottableAxisTraits(discrete=self.discrete)

    def __len__(self) -> int:
        return int(self.tAx.GetNbins())

    def __getitem__(self, index: int) -> Any:
        return (self.tAx.GetBinLowEdge(index + 1), self.tAx.GetBinUpEdge(index + 1))

    def __iter__(self) -> Iterator[Any]:
        return (self[i] for i in range(len(self)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlottableAxisContinuous):
            return False
        return len(self) == len(other) and all(a == b for a, b in zip(self, other, strict=True))

    __hash__ = None  # type: ignore[assignment]  # equal by their bins, which can change


class PlottableAxisDiscrete(PlottableAxisContinuous):
    """An axis of categories: each bin its label."""

    discrete = True

    def __getitem__(self, index: int) -> Any:
        return str(self.tAx.GetBinLabel(index + 1))


#: ROOT's name for what both kinds of axis are.
PlottableAxisBase = PlottableAxisContinuous


def _plottable_axis(tAxis: Any) -> PlottableAxisBase:
    """An axis whose every bin has a label is categories; any other, ranges."""
    labelled = all(tAxis.GetBinLabel(i + 1) for i in range(tAxis.GetNbins()))
    return PlottableAxisDiscrete(tAxis) if labelled else PlottableAxisContinuous(tAxis)


def _axes(self: Any) -> tuple[PlottableAxisBase, ...]:
    return tuple(_plottable_axis(_get_axis(self, i)) for i in range(self.GetDimension()))


def _kind(self: Any) -> Kind:
    """A profile's bins are means; every other histogram's are counts."""
    return Kind.MEAN if type(self).__name__.startswith("TProfile") else Kind.COUNT


def _shape(hist: Any, flow: bool = True) -> tuple[int, ...]:
    """The bins along each axis, x first - the flow bins counted unless told otherwise."""
    return tuple(_get_axis_len(hist, i, flow) for i in range(hist.GetDimension()))


def _inner(flow: bool, dimensions: int) -> tuple[slice, ...]:
    """Every bin, or every bin but the flow, along each axis."""
    return (slice(None) if flow else slice(1, -1),) * dimensions


def _values_default(self: Any, flow: bool = False, *, writable: bool = False) -> Any:
    """The bins themselves as an array, x first, read-only unless ``writable`` is asked for."""
    cells = self.GetArray()
    shaped = cells[: self.GetNcells()].reshape(_shape(self), order="F")
    view = shaped[_inner(flow, self.GetDimension())]
    view.setflags(write=writable)
    return view


def _grid_by_copy(self: Any, getter: Any, flow: bool) -> Any:
    """``getter`` of each bin, by its numbers along each axis, as a new read-only array."""
    offset = 0 if flow else 1
    ranges = [
        range(offset, _get_axis_len(self, axis, flow) + offset)
        for axis in range(self.GetDimension())
    ]
    found = [getter(*bins) for bins in product(*ranges)]
    out = np.array(found, dtype=np.float64).reshape(_shape(self, flow))
    out.setflags(write=False)
    return out


def _values_by_copy(self: Any, flow: bool = False, *, writable: bool = False) -> Any:
    """Each bin's ``GetBinContent``, as a new array that is never the histogram's own."""
    if writable:
        raise TypeError(
            "values(writable=True) is not supported for this histogram type, it cannot "
            "return a writable view into the histogram"
        )
    return _grid_by_copy(self, self.GetBinContent, flow)


def _by_copy(hist: Any) -> bool:
    """Whether the histogram's values are a copy: a profile's, or bins of one byte each."""
    return _kind(hist) == Kind.MEAN or type(hist).__name__[-1:] == "C"


def _values(self: Any, flow: bool = False, *, writable: bool = False) -> Any:
    chosen = _values_by_copy if _by_copy(self) else _values_default
    return chosen(self, flow, writable=writable)


def _hasWeights(hist: Any) -> bool:
    """Whether the histogram keeps the squares of its weights."""
    return bool(hist.GetSumw2N())


def _get_sum_of_weights_squared(self: Any, flow: bool = False) -> Any:
    """The squared weights, shaped as the bins are."""
    squares = np.asarray(self.GetSumw2(), dtype=np.float64)
    return squares.reshape(_shape(self), order="F")[_inner(flow, self.GetDimension())]


def _variances(self: Any, flow: bool = False) -> Any:
    """Each bin's variance: its squared weights, or for unweighted counts the count itself."""
    return np.asarray(self._xrd.variances(flow))


def _counts(self: Any, flow: bool = False) -> Any:
    """How many entries each bin is worth: its content, or with weights the effective count."""
    return np.asarray(self._xrd.counts(flow))
