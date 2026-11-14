"""``RAxisVariant`` and ``Internal::RAxes``: any one axis, and the axes of a histogram.

The variant holds a regular, a variable or a categorical axis and hands
each of ROOT's common questions to it. The axes of a histogram number
its bins: the global index of a bin is its linearized index along the
first axis, times the second axis's bins, plus its index along the
second, and on, as ``ROOT::Experimental::Internal::RAxes`` counts them.
"""

from __future__ import annotations

from typing import Any, NoReturn

from .axes import RCategoricalAxis, RRegularAxis, RVariableBinAxis
from .index import RBinIndexMultiDimRange, RBinIndexRange, RLinearizedIndex, as_index
from .slices import RSliceSpec

__all__ = ["RAxes", "RAxisVariant", "AXES"]

#: The axis types a variant holds, in ``std::variant``'s order: its ``index()``.
AXES = (RRegularAxis, RVariableBinAxis, RCategoricalAxis)


class _Variant:
    """What ``GetVariant()`` gives: ``index()``, the alternative held."""

    def __init__(self, axis: Any) -> None:
        self._axis = axis

    def index(self) -> int:
        return AXES.index(type(self._axis))


class RAxisVariant:
    """``RAxisVariant(axis)``: one axis of any of ROOT 7's kinds."""

    def __init__(self, axis: Any) -> None:
        self.axis: Any = axis.axis if isinstance(axis, RAxisVariant) else axis
        if not isinstance(self.axis, AXES):
            raise ValueError(f"An RAxisVariant holds an RRegularAxis, an RVariableBinAxis or an "
                             f"RCategoricalAxis, not {axis!r}.")  # fmt: skip

    def GetVariant(self) -> _Variant:
        return _Variant(self.axis)

    def GetRegularAxis(self) -> RRegularAxis | None:
        return self.axis if isinstance(self.axis, RRegularAxis) else None

    def GetVariableBinAxis(self) -> RVariableBinAxis | None:
        return self.axis if isinstance(self.axis, RVariableBinAxis) else None

    def GetCategoricalAxis(self) -> RCategoricalAxis | None:
        return self.axis if isinstance(self.axis, RCategoricalAxis) else None

    def GetNNormalBins(self) -> int:
        return int(self.axis.GetNNormalBins())

    def GetTotalNBins(self) -> int:
        return int(self.axis.GetTotalNBins())

    def GetNormalRange(self, begin: Any = None, end: Any = None) -> RBinIndexRange:
        return self.axis.GetNormalRange(begin, end)  # type: ignore[no-any-return]

    def GetFullRange(self) -> RBinIndexRange:
        return self.axis.GetFullRange()  # type: ignore[no-any-return]

    def Slice(self, spec: Any) -> RAxisVariant:
        return RAxisVariant(self.axis.Slice(spec))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RAxisVariant) and bool(self.axis == other.axis)

    __hash__ = None  # type: ignore[assignment]

    def Streamer(self, *buffer: Any) -> NoReturn:
        """ROOT's streamer refuses to store a variant; so does this."""
        raise RuntimeError("unable to store RAxisVariant")

    #: Writing it into a file runs its streamer, which refuses, as ROOT's does.
    _write_into = Streamer

    def __repr__(self) -> str:
        return f"RAxisVariant({self.axis!r})"


def variant_vector(axes: list[RAxisVariant]) -> Any:
    """A ``std::vector<RAxisVariant>`` of ``axes``, as ``GetAxes()`` returns them."""
    from ...cppstd.containers import std

    return std.vector["ROOT::Experimental::RAxisVariant"](list(axes))


class RAxes:
    """``Internal::RAxes(axes)``: the axes of a histogram, and the global index of its bins."""

    def __init__(self, axes: Any) -> None:
        self._axes = [RAxisVariant(axis) for axis in axes]
        if not self._axes:
            raise ValueError("must have at least 1 axis object")
        self._raw = [variant.axis for variant in self._axes]
        self._totals = [axis.GetTotalNBins() for axis in self._raw]

    def GetNDimensions(self) -> int:
        return len(self._axes)

    def Get(self) -> Any:
        return variant_vector(self._axes)

    def variants(self) -> list[RAxisVariant]:
        return list(self._axes)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RAxes) and self._axes == other._axes

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = None  # type: ignore[assignment]

    def ComputeTotalNBins(self) -> int:
        total = 1
        for count in self._totals:
            total *= count
        return total

    def ComputeGlobalIndex(self, given: Any) -> RLinearizedIndex:
        """``ComputeGlobalIndex``: of a tuple of values (``std::make_tuple``), or of a vector or
        array of bin indices."""
        if isinstance(given, tuple):
            if len(given) != len(self._raw):
                raise ValueError("invalid number of arguments to ComputeGlobalIndex")
            return RLinearizedIndex(*self.values_index(given))
        indices = list(given)
        if len(indices) != len(self._raw):
            raise ValueError("invalid number of indices passed to ComputeGlobalIndex")
        return RLinearizedIndex(*self.indices_index(indices))

    def values_index(self, values: Any) -> tuple[int, bool]:
        """The global index of the bin of ``values``, one per axis, and whether there is one.

        An axis that cannot take its argument refuses it when it is reached, so an
        argument past a value outside every bin is not looked at, as in ROOT.
        """
        index = 0
        for axis, total, value in zip(self._raw, self._totals, values, strict=False):
            at, valid = axis.value_index(value)
            if not valid:
                return 0, False
            index = index * total + at
        return index, True

    def indices_index(self, indices: Any) -> tuple[int, bool]:
        """The global index of the bin of ``indices``, one ``RBinIndex`` per axis."""
        index = 0
        for axis, total, given in zip(self._raw, self._totals, indices, strict=False):
            at, valid = axis.bin_index(as_index(given))
            if not valid:
                return 0, False
            index = index * total + at
        return index, True

    def GetFullMultiDimRange(self) -> RBinIndexMultiDimRange:
        return RBinIndexMultiDimRange([axis.GetFullRange() for axis in self._raw])

    def Slice(self, specs: list[RSliceSpec]) -> list[RAxisVariant]:
        """The axes a slice leaves: each sliced, but those summed over, which disappear."""
        return [axis.Slice(spec) for axis, spec in zip(self._axes, specs, strict=False)
                if spec.GetOperationSum() is None]  # fmt: skip

    def Streamer(self, *buffer: Any) -> NoReturn:
        """ROOT's streamer refuses to store the axes; so does this."""
        raise RuntimeError("unable to store RAxes")

    #: Writing it into a file runs its streamer, which refuses, as ROOT's does.
    _write_into = Streamer
