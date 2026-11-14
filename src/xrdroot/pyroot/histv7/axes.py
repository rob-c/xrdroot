"""ROOT 7's histogram axes: ``RRegularAxis``, ``RVariableBinAxis``, ``RCategoricalAxis``.

An axis finds a value's bin as ``ROOT::Experimental``'s does - below the low
edge the underflow, at or past the high edge (or NaN) the overflow, which
are the first and last of its bins when it has them and nowhere when it has
not - and gives the "linearized" index: where the bin sits among the axis's
bins, and whether it is one. A categorical axis has no underflow, and
its overflow takes every name it does not know. Each refuses, as ROOT's
does with ``std::invalid_argument``, an argument it cannot take: a regular
or variable axis a name, a categorical axis a number.
"""

from __future__ import annotations

import bisect
import math
from typing import Any, NoReturn

import numpy as np

from .index import CreateBinIndexRange, RBinIndex, RBinIndexRange, RLinearizedIndex, as_index
from .slices import as_spec

__all__ = ["RCategoricalAxis", "RRegularAxis", "RVariableBinAxis", "number", "to_string"]

#: What C++ converts to a ``double`` argument: Python's and NumPy's numbers.
NUMBERS = (int, float, np.integer, np.floating)


def to_string(value: float) -> str:
    """``std::to_string(double)``: ``%f``, which ROOT's messages are made with."""
    return f"{value:f}"


def number(value: Any) -> float:
    """A regular or variable axis's argument, which must be a number."""
    if isinstance(value, NUMBERS):
        return float(value)
    raise ValueError("invalid type of argument")


def _normal_range(begin: Any, end: Any, normal: int) -> RBinIndexRange:
    """``GetNormalRange(begin, end)``: ``[begin, end)`` of normal bins, checked as ROOT checks."""
    begin, end = as_index(begin), as_index(end)
    if not begin.IsNormal():
        raise ValueError("begin must be a normal bin")
    if begin.GetIndex() >= normal:
        raise ValueError("begin must be inside the axis")
    if not end.IsNormal():
        raise ValueError("end must be a normal bin")
    if end.GetIndex() > normal:
        raise ValueError("end must be inside or past the axis")
    if not end >= begin:
        raise ValueError("end must be >= begin")
    return CreateBinIndexRange(begin, end, 0)


class _Axis:
    """What every axis shares: its ranges, and its linearized indices as ``RLinearizedIndex``."""

    _normal: int
    _flow: bool

    def GetNNormalBins(self) -> int:
        return self._normal

    def GetNormalRange(self, begin: Any = None, end: Any = None) -> RBinIndexRange:
        if begin is None and end is None:
            return CreateBinIndexRange(0, self._normal, 0)
        return _normal_range(begin, end, self._normal)

    def ComputeLinearizedIndex(self, x: Any) -> RLinearizedIndex:
        return RLinearizedIndex(*self.value_index(x))

    def GetLinearizedIndex(self, index: Any) -> RLinearizedIndex:
        return RLinearizedIndex(*self.bin_index(as_index(index)))

    def value_index(self, x: Any) -> tuple[int, bool]:
        raise NotImplementedError

    def bin_index(self, index: RBinIndex) -> tuple[int, bool]:
        raise NotImplementedError

    def Streamer(self, *buffer: Any) -> NoReturn:
        """ROOT's streamer refuses to store an axis; so does this."""
        raise RuntimeError(f"unable to store {type(self).__name__}")

    #: Writing it into a file runs its streamer, which refuses, as ROOT's does.
    _write_into = Streamer


class _FlowAxis(_Axis):
    """A regular or variable axis: an underflow and an overflow bin, or neither."""

    def GetTotalNBins(self) -> int:
        return self._normal + 2 if self._flow else self._normal

    def HasFlowBins(self) -> bool:
        return self._flow

    def GetFullRange(self) -> RBinIndexRange:
        if self._flow:
            return CreateBinIndexRange(RBinIndex.Underflow(), RBinIndex(), self._normal)
        return self.GetNormalRange()

    def _low(self) -> float:
        raise NotImplementedError

    def _high(self) -> float:
        raise NotImplementedError

    def _normal_bin(self, x: float) -> int:
        raise NotImplementedError

    def value_index(self, x: Any) -> tuple[int, bool]:
        """``ComputeLinearizedIndex``: NaN, like a value at or past the high edge, overflows."""
        x = number(x)
        if x < self._low():
            return 0, self._flow
        if not x < self._high():
            return self._normal + 1, self._flow
        return self._normal_bin(x) + int(self._flow), True

    def bin_index(self, index: RBinIndex) -> tuple[int, bool]:
        """``GetLinearizedIndex``: an index past the normal bins is no bin, at its own number."""
        if index.IsUnderflow():
            return 0, self._flow
        if index.IsOverflow():
            return self._normal + 1, self._flow
        if index.IsInvalid():
            return 0, False
        if index.fIndex >= self._normal:
            return index.fIndex, False
        return index.fIndex + int(self._flow), True

    def _sliced(self, spec: Any) -> tuple[int, int, int]:
        """The normal bins ``[begin, end)`` a slice keeps, and how many go into each new bin."""
        spec = as_spec(spec)
        if spec.GetOperationSum() is not None:
            raise RuntimeError("sum operation makes dimension disappear")
        begin, end, found = 0, self._normal, spec.GetRange()
        if not found.IsInvalid():
            begin = found.GetBegin().fIndex if found.GetBegin().IsNormal() else begin
            end = found.GetEnd().fIndex if found.GetEnd().IsNormal() else end
        rebin = spec.GetOperationRebin()
        group = 1 if rebin is None else rebin.GetNGroup()
        if (end - begin) % group != 0:
            raise RuntimeError("rebin operation does not divide number of normal bins")
        return begin, end, group


class RRegularAxis(_FlowAxis):
    """``RRegularAxis(nNormalBins, {low, high}, enableFlowBins)``: bins of equal width."""

    def __init__(self, nNormalBins: Any, interval: Any, enableFlowBins: bool = True) -> None:
        low, high = (float(v) for v in interval)
        self._normal, self._flow = int(nNormalBins) % (1 << 64), bool(enableFlowBins)
        if self._normal == 0:
            raise ValueError("nNormalBins must be > 0")
        if not math.isfinite(low):
            raise ValueError(f"low must be a finite value, but is {to_string(low)}")
        if not math.isfinite(high):
            raise ValueError(f"high must be a finite value, but is {to_string(high)}")
        if low >= high:
            raise ValueError(f"high must be > low, but {to_string(low)} >= {to_string(high)}")
        self.fLow, self.fHigh = low, high
        self._inverse = self._normal / (high - low)

    def GetLow(self) -> float:
        return self.fLow

    def GetHigh(self) -> float:
        return self.fHigh

    def _low(self) -> float:
        return self.fLow

    def _high(self) -> float:
        return self.fHigh

    def ComputeLowEdge(self, bin: Any) -> float:
        if int(bin) >= self._normal:
            raise ValueError("bin must be inside the axis")
        return self.fLow + (self.fHigh - self.fLow) * int(bin) / self._normal

    def ComputeHighEdge(self, bin: Any) -> float:
        if int(bin) >= self._normal:
            raise ValueError("bin must be inside the axis")
        return self.fLow + (self.fHigh - self.fLow) * (int(bin) + 1) / self._normal

    def _normal_bin(self, x: float) -> int:
        return min(int((x - self.fLow) * self._inverse), self._normal - 1)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RRegularAxis) and (self._normal, self.fLow, self.fHigh,
                                                    self._flow) == (
            other._normal, other.fLow, other.fHigh, other._flow)  # fmt: skip

    __hash__ = None  # type: ignore[assignment]

    def Slice(self, spec: Any) -> RRegularAxis:
        """The axis of the bins a slice keeps, regrouped; with flow bins, which keep the rest.

        A new edge is worked out only where the range moves one, so that an edge
        kept is the edge given, not one recomputed with its rounding.
        """
        begin, end, group = self._sliced(spec)
        low = self.ComputeLowEdge(begin) if begin > 0 else self.fLow
        high = self.ComputeHighEdge(end - 1) if end < self._normal else self.fHigh
        return RRegularAxis((end - begin) // group, (low, high), True)

    def __repr__(self) -> str:
        return f"RRegularAxis({self._normal}, ({self.fLow}, {self.fHigh}), {self._flow})"


def _edges(given: Any) -> list[float]:
    """Bin edges, checked as ``RVariableBinAxis`` checks them: two or more, none NaN, rising."""
    edges = [float(v) for v in given]
    if len(edges) < 2:
        raise ValueError("must have >= 2 bin edges")
    for at, edge in enumerate(edges):
        if math.isnan(edge):
            raise ValueError(f"bin egde {at} is NaN")  # sic, as ROOT spells it
        if at and edges[at - 1] >= edge:
            raise ValueError(f"binEdges must be sorted, but for bin {at - 1}: "
                             f"{to_string(edges[at - 1])} >= {to_string(edge)}")  # fmt: skip
    return edges


class RVariableBinAxis(_FlowAxis):
    """``RVariableBinAxis(binEdges, enableFlowBins)``: bins between the edges given."""

    def __init__(self, binEdges: Any, enableFlowBins: bool = True) -> None:
        self._edges = _edges(binEdges)
        self._normal, self._flow = len(self._edges) - 1, bool(enableFlowBins)

    def GetBinEdges(self) -> Any:
        from ...cppstd.containers import std

        return std.vector["double"](self._edges)

    def edges(self) -> list[float]:
        return list(self._edges)

    def _low(self) -> float:
        return self._edges[0]

    def _high(self) -> float:
        return self._edges[-1]

    def _normal_bin(self, x: float) -> int:
        """The first bin whose high edge is above ``x``: ROOT's walk, by bisection."""
        return bisect.bisect_right(self._edges, x) - 1

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RVariableBinAxis) and (self._edges, self._flow) == (
            other._edges, other._flow)  # fmt: skip

    __hash__ = None  # type: ignore[assignment]

    def Slice(self, spec: Any) -> RVariableBinAxis:
        """The axis of the bins a slice keeps, every ``group``-th edge; with flow bins."""
        begin, end, group = self._sliced(spec)
        return RVariableBinAxis([*self._edges[begin:end:group], self._edges[end]], True)

    def __repr__(self) -> str:
        return f"RVariableBinAxis({self._edges}, {self._flow})"


class RCategoricalAxis(_Axis):
    """``RCategoricalAxis(categories, enableOverflowBin)``: a bin for each name, and one for
    the names not among them."""

    def __init__(self, categories: Any, enableOverflowBin: bool = True) -> None:
        self._categories = [str(name) for name in categories]
        self._normal, self._flow = len(self._categories), bool(enableOverflowBin)
        if not self._categories:
            raise ValueError("must have at least one category")
        self._bins: dict[str, int] = {}
        for at, name in enumerate(self._categories):
            if name in self._bins:
                raise ValueError(f"duplicate category '{name}' for bin {at}")
            self._bins[name] = at

    def GetTotalNBins(self) -> int:
        return self._normal + 1 if self._flow else self._normal

    def GetCategories(self) -> Any:
        from ...cppstd.containers import std

        return std.vector["std::string"](self._categories)

    def HasOverflowBin(self) -> bool:
        return self._flow

    def GetFullRange(self) -> RBinIndexRange:
        if self._flow:
            return CreateBinIndexRange(0, RBinIndex(), self._normal)
        return self.GetNormalRange()

    def value_index(self, x: Any) -> tuple[int, bool]:
        """``ComputeLinearizedIndex``: a name's bin, or the overflow for a name not known."""
        if isinstance(x, NUMBERS):
            raise ValueError("invalid type of argument")
        found = self._bins.get(str(x))
        return (self._normal, self._flow) if found is None else (found, True)

    def bin_index(self, index: RBinIndex) -> tuple[int, bool]:
        if index.IsOverflow():
            return self._normal, self._flow
        if not index.IsNormal():  # no underflow bin, and the invalid index
            return 0, False
        return index.fIndex, index.fIndex < self._normal

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RCategoricalAxis) and (self._categories, self._flow) == (
            other._categories, other._flow)  # fmt: skip

    __hash__ = None  # type: ignore[assignment]

    def Slice(self, spec: Any) -> RCategoricalAxis:
        """Only the whole axis: a categorical axis is neither cut nor regrouped."""
        spec = as_spec(spec)
        if spec.GetOperationSum() is not None:
            raise RuntimeError("sum operation makes dimension disappear")
        if not spec.GetRange().IsInvalid():
            raise RuntimeError("slicing of RCategoricalAxis not implemented")
        if spec.GetOperationRebin() is not None:
            raise RuntimeError("cannot rebin RCategoricalAxis")
        return RCategoricalAxis(self._categories, True)

    def __repr__(self) -> str:
        return f"RCategoricalAxis({self._categories}, {self._flow})"
