"""``RHistStats``: a ROOT 7 histogram's entries and moments, summed over every fill.

The sums take every value filled, in the flow bins or out of any bin, as
``ROOT::Experimental::RHistStats`` takes them - the entries, the sums of
the weights and of their squares, and for each dimension the weighted sums
of the value's first four powers - so the mean and the spread are those of
what was filled, not of the bins. A dimension can be disabled (a
categorical axis's, whose names have no mean), and the whole object
tainted (after a bin is set by hand), after which reading it throws until
it is cleared. Each product is formed in ROOT's order, ``w * x * x``, so
the sums round as ROOT's do.
"""

from __future__ import annotations

import copy
import math
import threading
from typing import Any, NoReturn

from .axes import NUMBERS

__all__ = ["RDimensionStats", "RHistStats", "RWeight", "split_weight", "ATOMIC"]

#: What makes an ``...Atomic`` operation whole against other threads: one lock for them all.
ATOMIC = threading.RLock()


class RWeight:
    """``RWeight(w)``: the weight a fill is given."""

    __slots__ = ("fValue",)

    def __init__(self, value: Any) -> None:
        self.fValue = float(value)


def split_weight(args: tuple[Any, ...]) -> tuple[tuple[Any, ...], float | None]:
    """A fill's arguments, and its weight when the last is an ``RWeight``. A single tuple -
    ``std::make_tuple``'s - is the arguments."""
    weight = None
    if args and isinstance(args[-1], RWeight):
        args, weight = args[:-1], args[-1].fValue
    if len(args) == 1 and isinstance(args[0], tuple):
        args = args[0]
    return args, weight


class RHistStats:
    """``RHistStats(nDimensions)``: the running sums a histogram's statistics are worked out
    from."""

    class RDimensionStats:
        """``RHistStats::RDimensionStats``: one dimension's weighted sums of ``x`` to ``x**4``."""

        __slots__ = ("fEnabled", "fSumWX", "fSumWX2", "fSumWX3", "fSumWX4")

        def __init__(self) -> None:
            self.fEnabled = True
            self.Clear()

        def Add(self, x: Any, w: float | None = None) -> None:
            """``Add(x[, w])``, or ``Add(other)``: the powers of one value, or another's sums."""
            if isinstance(x, RDimensionStats):
                self.fSumWX += x.fSumWX
                self.fSumWX2 += x.fSumWX2
                self.fSumWX3 += x.fSumWX3
                self.fSumWX4 += x.fSumWX4
            elif w is None:
                self.fSumWX += x
                self.fSumWX2 += x * x
                self.fSumWX3 += x * x * x
                self.fSumWX4 += x * x * x * x
            else:
                self.fSumWX += w * x
                self.fSumWX2 += w * x * x
                self.fSumWX3 += w * x * x * x
                self.fSumWX4 += w * x * x * x * x

        def AddAtomic(self, other: RDimensionStats) -> None:
            with ATOMIC:
                self.Add(other)

        def Clear(self) -> None:
            self.fSumWX = self.fSumWX2 = self.fSumWX3 = self.fSumWX4 = 0.0

        def Scale(self, factor: float) -> None:
            self.fSumWX *= factor
            self.fSumWX2 *= factor
            self.fSumWX3 *= factor
            self.fSumWX4 *= factor

    def __init__(self, nDimensions: Any) -> None:
        if int(nDimensions) == 0:
            raise ValueError("nDimensions must be > 0")
        self.fNEntries = 0
        self.fSumW = self.fSumW2 = 0.0
        self.fDimensionStats = [RDimensionStats() for _ in range(int(nDimensions))]
        self.fTainted = False

    def _ThrowIfTainted(self) -> None:
        if self.fTainted:
            raise RuntimeError("statistics are tainted, for example after setting bin contents "
                               "or slicing")  # fmt: skip

    def GetNDimensions(self) -> int:
        return len(self.fDimensionStats)

    def GetNEntries(self) -> int:
        self._ThrowIfTainted()
        return self.fNEntries

    def GetSumW(self) -> float:
        self._ThrowIfTainted()
        return self.fSumW

    def GetSumW2(self) -> float:
        self._ThrowIfTainted()
        return self.fSumW2

    def GetDimensionStats(self, dim: Any = 0) -> RDimensionStats:
        self._ThrowIfTainted()
        found = self._at(dim)
        if not found.fEnabled:
            raise ValueError("dimension is disabled")
        return found

    def _at(self, dim: Any) -> RDimensionStats:
        """``fDimensionStats.at(dim)``: past the end is ``std::out_of_range``."""
        if not 0 <= int(dim) < len(self.fDimensionStats):
            raise IndexError(f"vector::_M_range_check: __n (which is {int(dim)}) >= this->size() "
                             f"(which is {len(self.fDimensionStats)})")  # fmt: skip
        return self.fDimensionStats[int(dim)]

    def DisableDimension(self, dim: Any) -> None:
        self._at(dim).fEnabled = False

    def IsEnabled(self, dim: Any) -> bool:
        return self._at(dim).fEnabled

    def Taint(self) -> None:
        self.fTainted = True

    def IsTainted(self) -> bool:
        return self.fTainted

    def ComputeNEffectiveEntries(self) -> float:
        self._ThrowIfTainted()
        if self.fSumW2 == 0:
            return math.nan
        return self.fSumW * self.fSumW / self.fSumW2

    def ComputeMean(self, dim: Any = 0) -> float:
        found = self.GetDimensionStats(dim)
        if self.fSumW == 0:
            return math.nan
        return found.fSumWX / self.fSumW

    def ComputeVariance(self, dim: Any = 0) -> float:
        found = self.GetDimensionStats(dim)
        if self.fSumW == 0:
            return math.nan
        mean = self.ComputeMean(dim)
        return found.fSumWX2 / self.fSumW - mean * mean

    def ComputeStdDev(self, dim: Any = 0) -> float:
        """``std::sqrt`` of the variance: NaN, not an error, for one rounded below zero."""
        variance = self.ComputeVariance(dim)
        return math.sqrt(variance) if variance >= 0 else math.nan

    def _moments(self, dim: Any) -> tuple[RDimensionStats, float, float] | None:
        """The sums, mean and variance a skewness or kurtosis needs, or ``None`` for NaN."""
        found = self.GetDimensionStats(dim)
        if self.fSumW == 0:
            return None
        mean, variance = self.ComputeMean(dim), self.ComputeVariance(dim)
        return None if variance == 0 else (found, mean, variance)

    def ComputeSkewness(self, dim: Any = 0) -> float:
        moments = self._moments(dim)
        if moments is None:
            return math.nan
        found, mean, var = moments
        ewx3, ewx2 = found.fSumWX3 / self.fSumW, found.fSumWX2 / self.fSumW
        return (ewx3 - 3 * ewx2 * mean + 2 * mean * mean * mean) / _pow(var, 1.5)

    def ComputeKurtosis(self, dim: Any = 0) -> float:
        moments = self._moments(dim)
        if moments is None:
            return math.nan
        found, mean, var = moments
        ewx4, ewx3 = found.fSumWX4 / self.fSumW, found.fSumWX3 / self.fSumW
        ewx2 = found.fSumWX2 / self.fSumW
        return (ewx4 - 4 * ewx3 * mean + 6 * ewx2 * mean * mean
                - 3 * mean * mean * mean * mean) / (var * var) - 3  # fmt: skip

    def _checked(self, args: tuple[Any, ...]) -> tuple[list[float | None], float | None]:
        """A fill's values - ``None`` for a disabled dimension's - and its weight, each argument
        checked before anything changes, as ROOT checks them."""
        values, weight = split_weight(args)
        if len(values) != len(self.fDimensionStats):
            raise ValueError("invalid number of arguments to Fill")
        made: list[float | None] = []
        for value, found in zip(values, self.fDimensionStats, strict=False):
            if found.fEnabled and not isinstance(value, NUMBERS):
                raise ValueError("invalid type of argument in RHistStats")
            made.append(float(value) if found.fEnabled else None)
        return made, weight

    def Fill(self, *args: Any) -> None:
        """``Fill(values...[, RWeight])`` or ``Fill(tuple[, RWeight])``: one entry."""
        values, weight = self._checked(args)
        self.fNEntries += 1
        w = 1.0 if weight is None else weight
        self.fSumW += w
        self.fSumW2 += w * w
        for value, found in zip(values, self.fDimensionStats, strict=False):
            if value is not None:
                found.Add(value, weight)

    def _check_add(self, other: RHistStats, name: str) -> None:
        if len(self.fDimensionStats) != len(other.fDimensionStats):
            raise ValueError(f"number of dimensions not identical in {name}")
        for mine, theirs in zip(self.fDimensionStats, other.fDimensionStats, strict=False):
            if mine.fEnabled != theirs.fEnabled:
                raise ValueError("the same dimensions must be enabled to combine statistics "
                                 f"with {name}")  # fmt: skip

    def Add(self, other: RHistStats, name: str = "Add") -> None:
        """``Add(other)``: another's entries, read even if tainted, whose taint comes along."""
        self._check_add(other, name)
        self.fNEntries += other.fNEntries
        self.fSumW += other.fSumW
        self.fSumW2 += other.fSumW2
        for mine, theirs in zip(self.fDimensionStats, other.fDimensionStats, strict=False):
            if mine.fEnabled:
                mine.Add(theirs)
        self.fTainted |= other.fTainted

    def AddAtomic(self, other: RHistStats) -> None:
        with ATOMIC:
            self.Add(other, "AddAtomic")

    def Clear(self) -> None:
        self.fNEntries = 0
        self.fSumW = self.fSumW2 = 0.0
        for found in self.fDimensionStats:
            found.Clear()
        self.fTainted = False

    def Scale(self, factor: Any) -> None:
        factor = float(factor)
        self.fSumW *= factor
        self.fSumW2 *= factor * factor
        for found in self.fDimensionStats:
            if found.fEnabled:
                found.Scale(factor)

    def copy(self) -> RHistStats:
        """A copy of every sum, as C++'s ``h.fStats = fStats`` makes one."""
        return copy.deepcopy(self)

    def Streamer(self, *buffer: Any) -> NoReturn:
        """ROOT's streamer refuses to store the statistics; so does this."""
        raise RuntimeError("unable to store RHistStats")

    #: Writing it into a file runs its streamer, which refuses, as ROOT's does.
    _write_into = Streamer


#: ``RHistStats::RDimensionStats``, by its own name.
RDimensionStats = RHistStats.RDimensionStats


def _pow(base: float, exponent: float) -> float:
    """``std::pow``: NaN for a negative base and a fractional exponent, as C's is."""
    return math.pow(base, exponent) if base >= 0 else math.nan
