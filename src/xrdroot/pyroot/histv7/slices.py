"""``RSliceSpec`` and ``RSliceBinIndexMapper``: what a slice of a histogram keeps, and where
each of its bins goes.

A specification per dimension gives a range of bins - none for all of them
- and at most one operation: a rebin, which groups bins, or a sum, which
makes the dimension disappear. The mapper takes a bin's indices to its
indices in the sliced histogram: a bin below the range to the underflow,
past it to the overflow, a bin in it shifted to the range's start; a bin
outside a summed range is dropped.
"""

from __future__ import annotations

from typing import Any

from .index import RBinIndex, RBinIndexRange, as_index

__all__ = ["ROperationRebin", "ROperationSum", "RSliceBinIndexMapper", "RSliceSpec", "as_spec"]


class RSliceSpec:
    """``RSliceSpec([range][, operation])``: the slice of one dimension."""

    class ROperationRebin:
        """``RSliceSpec::ROperationRebin(nGroup)``: ``nGroup`` bins into each new one."""

        def __init__(self, nGroup: Any) -> None:
            self._group = int(nGroup) % (1 << 64)
            if self._group == 0:
                raise ValueError("nGroup must be > 0")

        def GetNGroup(self) -> int:
            return self._group


    class ROperationSum:
        """``RSliceSpec::ROperationSum``: the bins along a dimension summed, which removes it."""

    def __init__(self, *given: Any) -> None:
        self._range, self._operation = RBinIndexRange(), None
        if len(given) == 1 and isinstance(given[0], RSliceSpec):
            self._range, self._operation = given[0]._range, given[0]._operation
            return
        for each in given:
            if isinstance(each, RBinIndexRange):
                self._range = each
            elif isinstance(each, (RSliceSpec.ROperationRebin, RSliceSpec.ROperationSum)):
                self._operation = each
            else:
                raise ValueError(f"An RSliceSpec is made of a range and an operation, not "
                                 f"{each!r}.")  # fmt: skip

    def GetRange(self) -> RBinIndexRange:
        return self._range

    def HasOperation(self) -> bool:
        return self._operation is not None

    def GetOperationRebin(self) -> ROperationRebin | None:
        return self._operation if isinstance(self._operation, ROperationRebin) else None

    def GetOperationSum(self) -> ROperationSum | None:
        return self._operation if isinstance(self._operation, ROperationSum) else None


#: ``RSliceSpec::ROperationRebin`` and ``RSliceSpec::ROperationSum``, by their own names.
ROperationRebin, ROperationSum = RSliceSpec.ROperationRebin, RSliceSpec.ROperationSum


def as_spec(given: Any) -> RSliceSpec:
    """What C++ converts to an ``RSliceSpec``: a range, an operation, or the specification."""
    return given if isinstance(given, RSliceSpec) else RSliceSpec(given)


def _in_range(index: RBinIndex, found: RBinIndexRange) -> tuple[RBinIndex, bool]:
    """Where ``index`` goes in a sliced range, and whether the range holds it.

    The flow bins map to themselves, though a range need not hold them; a
    normal bin before the range goes to the underflow, one after it to the
    overflow, and one in it is shifted down by the range's start.
    """
    begin, end = found.GetBegin(), found.GetEnd()
    if index.IsUnderflow():
        return index, begin.IsUnderflow()
    if index.IsOverflow():
        return index, end.IsInvalid()
    if begin.IsNormal() and index < begin:
        return RBinIndex.Underflow(), False
    if end.IsNormal() and index >= end:
        return RBinIndex.Overflow(), False
    return (index - begin.fIndex if begin.IsNormal() else index), True


class RSliceBinIndexMapper:
    """``Internal::RSliceBinIndexMapper(sliceSpecs)``: a bin's indices in the sliced histogram."""

    def __init__(self, sliceSpecs: Any) -> None:
        self._specs = [as_spec(spec) for spec in sliceSpecs]
        if not self._specs:
            raise ValueError("must have at least 1 slice specification")
        self._kept = sum(spec.GetOperationSum() is None for spec in self._specs)

    def GetSliceSpecs(self) -> Any:
        from ...cppstd.containers import std

        return std.vector["ROOT::Experimental::RSliceSpec"](list(self._specs))

    def GetMappedDimensionality(self) -> int:
        return self._kept

    def Map(self, original: Any, mapped: Any) -> bool:
        """``Map(original, mapped)``: ``mapped`` filled in, and whether the bin is kept at all."""
        if len(original) != len(self._specs):
            raise ValueError("invalid number of original indices passed to "
                             "RSliceBinIndexMapper::Map")  # fmt: skip
        if len(mapped) != self._kept:
            raise ValueError("invalid size of mapped indices passed to RSliceBinIndexMapper::Map")
        found = self.mapped([as_index(each) for each in original])
        for at, index in enumerate(found or ()):
            mapped[at] = index
        return found is not None

    def mapped(self, original: list[RBinIndex]) -> list[RBinIndex] | None:
        """The indices of ``original`` in the sliced histogram, or ``None`` for a bin dropped."""
        made = []
        for index, spec in zip(original, self._specs, strict=True):
            if index.IsInvalid():
                raise ValueError("invalid bin index passed to RSliceBinIndexMapper::Map")
            contained = True
            if not spec.GetRange().IsInvalid():
                index, contained = _in_range(index, spec.GetRange())
            if spec.GetOperationSum() is not None:
                if not contained:
                    return None
                continue
            rebin = spec.GetOperationRebin()
            if rebin is not None and index.IsNormal():
                index = RBinIndex(index.fIndex // rebin.GetNGroup())
            made.append(index)
        return made
