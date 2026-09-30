"""``RBinIndex``, ``RRegularAxis`` and ``RVariableBinAxis``: ROOT 7's histogram axes.

An axis finds a value's bin as ``ROOT::Experimental``'s does - below the low
edge the underflow, at or past the high edge (or NaN) the overflow, which
are the first and last of its bins when it has them and nowhere when it has
not - and gives the "linearized" index: where the bin sits among the axis's
bins, and whether it is one.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

__all__ = ["RBinIndex", "RBinIndexRange", "RRegularAxis", "RVariableBinAxis", "linearized"]

#: ``RBinIndex``'s special values: the underflow, the overflow and no bin at all.
UNDERFLOW, OVERFLOW, INVALID = -3, -2, -1


class RBinIndex:
    """``RBinIndex``: a normal bin's number from 0, or the underflow, the overflow or nothing."""

    def __init__(self, index: int = INVALID) -> None:
        self._index = int(index)

    @staticmethod
    def Underflow() -> RBinIndex:
        return RBinIndex(UNDERFLOW)

    @staticmethod
    def Overflow() -> RBinIndex:
        return RBinIndex(OVERFLOW)

    def IsNormal(self) -> bool:
        return self._index >= 0

    def IsUnderflow(self) -> bool:
        return self._index == UNDERFLOW

    def IsOverflow(self) -> bool:
        return self._index == OVERFLOW

    def IsInvalid(self) -> bool:
        return self._index == INVALID

    def GetIndex(self) -> int:
        return self._index

    def __add__(self, step: int) -> RBinIndex:
        return RBinIndex(self._index + int(step) if self.IsNormal() else INVALID)

    def __sub__(self, step: int) -> RBinIndex:
        moved = self._index - int(step)
        return RBinIndex(moved if self.IsNormal() and moved >= 0 else INVALID)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RBinIndex) and self._index == other._index

    def __lt__(self, other: RBinIndex) -> bool:
        return self.IsNormal() and other.IsNormal() and self._index < other._index

    def __ge__(self, other: RBinIndex) -> bool:
        return self == other or other < self

    def __hash__(self) -> int:
        return hash(self._index)

    def __repr__(self) -> str:
        names = {UNDERFLOW: "Underflow", OVERFLOW: "Overflow", INVALID: "Invalid"}
        return f"RBinIndex({names.get(self._index, self._index)})"


class RBinIndexRange:
    """The bins a ``for`` over ``GetNormalRange()`` or ``GetFullRange()`` walks, in order."""

    def __init__(self, indices: list[RBinIndex]) -> None:
        self._indices = indices

    def __iter__(self) -> Iterator[RBinIndex]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def as_index(given: Any) -> RBinIndex:
    """A bin given as an ``RBinIndex`` or as a normal bin's number."""
    return given if isinstance(given, RBinIndex) else RBinIndex(int(given))


class _Axis:
    """What both axes share: their bins, the flow bins or not, and a bin's linearized index."""

    def __init__(self, normal: int, flow: bool) -> None:
        self._normal, self._flow = int(normal), bool(flow)

    def GetNNormalBins(self) -> int:
        return self._normal

    def GetTotalNBins(self) -> int:
        return self._normal + 2 if self._flow else self._normal

    def HasFlowBins(self) -> bool:
        return self._flow

    def GetNormalRange(self, begin: Any = None, end: Any = None) -> RBinIndexRange:
        low = 0 if begin is None else as_index(begin).GetIndex()
        high = self._normal if end is None else as_index(end).GetIndex()
        if not 0 <= low <= high <= self._normal:
            raise ValueError(f"The bins from {low} to {high} are not a range of this axis's "
                             f"{self._normal} bins.")  # fmt: skip
        return RBinIndexRange([RBinIndex(i) for i in range(low, high)])

    def GetFullRange(self) -> RBinIndexRange:
        normal = list(self.GetNormalRange())
        flows = [RBinIndex.Underflow(), *normal, RBinIndex.Overflow()]
        return RBinIndexRange(flows if self._flow else normal)

    def _normal_bin(self, x: float) -> int:
        raise NotImplementedError

    def _low(self) -> float:
        raise NotImplementedError

    def _high(self) -> float:
        raise NotImplementedError

    def value_index(self, x: float) -> tuple[int, bool]:
        """``ComputeLinearizedIndex``: where the bin holding ``x`` sits, and whether it is one."""
        if x < self._low():
            return 0, self._flow
        if not x < self._high():
            return self._normal + 1, self._flow
        return self._normal_bin(x) + int(self._flow), True

    def bin_index(self, index: RBinIndex) -> tuple[int, bool]:
        """``GetLinearizedIndex``: where a bin, given by its ``RBinIndex``, sits."""
        if index.IsUnderflow():
            return 0, self._flow
        if index.IsOverflow():
            return self._normal + 1, self._flow
        if index.IsInvalid() or index.GetIndex() >= self._normal:
            return 0, False
        return index.GetIndex() + int(self._flow), True


class RRegularAxis(_Axis):
    """``RRegularAxis(nNormalBins, {low, high}, enableFlowBins)``: bins of equal width."""

    def __init__(self, nNormalBins: int, interval: Any, enableFlowBins: bool = True) -> None:
        low, high = (float(v) for v in interval)
        if int(nNormalBins) <= 0 or not (np.isfinite(low) and np.isfinite(high)) or low >= high:
            raise ValueError(f"An RRegularAxis of {nNormalBins} bins from {low} to {high} needs "
                             "at least one bin and finite edges, low below high.")  # fmt: skip
        super().__init__(int(nNormalBins), enableFlowBins)
        self._range = (low, high)
        self._inverse = int(nNormalBins) / (high - low)

    def GetLow(self) -> float:
        return self._range[0]

    def GetHigh(self) -> float:
        return self._range[1]

    _low, _high = GetLow, GetHigh

    def ComputeLowEdge(self, bin: int) -> float:
        low, high = self._range
        return low + (high - low) * int(bin) / self._normal

    def ComputeHighEdge(self, bin: int) -> float:
        return self.ComputeLowEdge(int(bin) + 1)

    def edges(self) -> list[float]:
        return [self.ComputeLowEdge(i) for i in range(self._normal + 1)]

    def _normal_bin(self, x: float) -> int:
        return min(int((x - self._range[0]) * self._inverse), self._normal - 1)


class RVariableBinAxis(_Axis):
    """``RVariableBinAxis(binEdges, enableFlowBins)``: bins between the edges given."""

    def __init__(self, binEdges: Any, enableFlowBins: bool = True) -> None:
        edges = [float(v) for v in binEdges]
        if len(edges) < 2 or any(not a < b for a, b in zip(edges, edges[1:])):
            raise ValueError(f"An RVariableBinAxis needs two or more edges, each above the one "
                             f"before it, and was given {edges}.")  # fmt: skip
        super().__init__(len(edges) - 1, enableFlowBins)
        self._edges = edges

    def GetBinEdges(self) -> list[float]:
        return list(self._edges)

    edges = GetBinEdges

    def _low(self) -> float:
        return self._edges[0]

    def _high(self) -> float:
        return self._edges[-1]

    def _normal_bin(self, x: float) -> int:
        return next((b for b in range(self._normal - 1) if x < self._edges[b + 1]),
                    self._normal - 1)  # fmt: skip


def linearized(axes: list[_Axis], values: tuple[Any, ...]) -> tuple[tuple[int, ...], bool]:
    """Where the bin of ``values`` - one per axis - sits along every axis, and whether it is one."""
    found = [axis.bin_index(v) if isinstance(v, RBinIndex) else axis.value_index(float(v))
             for axis, v in zip(axes, values)]  # fmt: skip
    return tuple(i for i, _ in found), all(ok for _, ok in found)
