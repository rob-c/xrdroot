"""A histogram's statistics and extremes, as ``TH1`` gives them - axis ranges honoured.

The running sums a histogram keeps are xrdroot's, which are ROOT's; but
once an axis has a range set (``SetRangeUser``), ROOT works the mean and
the spread out again from the bins in that range, and so does this. The
extremes - ``GetMaximum``, ``GetMaximumBin`` - look only at the bins drawn,
as ROOT's do, unless ``SetMaximum`` fixed them.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...fillrandom import AXIS_RANGE
from .refs import store

__all__: list[str] = []

#: ``FLT_MAX``, what ``GetMaximum`` and ``GetMinimum`` start from.
FLT_MAX = 3.40282346638528859811704183484516925440e38
#: What ``fMaximum`` and ``fMinimum`` hold when never set.
UNSET = -1111.0


class Stats:
    """``GetMean``, ``GetStdDev``, ``Integral``, ``GetMaximum`` and the rest."""

    _xrd: Any
    _core: Any
    _contents: Any
    _widths: Any
    GetXaxis: Any
    GetYaxis: Any
    GetZaxis: Any
    SetBit: Any
    ResetBit: Any

    def _axes(self) -> list[Any]:
        return [self.GetXaxis(), self.GetYaxis(), self.GetZaxis()][: len(self._xrd.axes)]

    def _displayed(self) -> np.ndarray[Any, Any]:
        """The global bins in every axis's displayed range, x fastest."""
        ranges = [np.arange(axis.GetFirst(), axis.GetLast() + 1) for axis in self._axes()]
        grids = np.meshgrid(*ranges, indexing="ij")
        found = np.zeros(grids[0].shape, dtype=np.int64)
        for grid, width in zip(reversed(grids), reversed(self._widths())):
            found = found * width + grid
        return found.ravel(order="F")

    def _ranged(self) -> bool:
        return any(axis.TestBit(AXIS_RANGE) for axis in self._axes())

    def _range_moments(self, axis: int) -> tuple[float, float, float]:
        """The sum of weights, the mean and the spread along ``axis``, from the bins in range."""
        bins = self._displayed()
        weights = self._contents()[bins].astype(np.float64)
        parts = self._bin_parts(bins)[axis]
        centres = np.array([self._axes()[axis].GetBinCenter(int(b)) for b in parts])
        total = float(np.sum(weights))
        if total == 0:
            return 0.0, 0.0, 0.0
        mean = float(np.sum(weights * centres) / total)
        variance = float(np.sum(weights * centres * centres) / total) - mean * mean
        return total, mean, math.sqrt(max(variance, 0.0))

    def _bin_parts(self, bins: np.ndarray[Any, Any]) -> list[np.ndarray[Any, Any]]:
        """The bin along each axis of every global bin in ``bins``."""
        found, rest = [], np.asarray(bins, dtype=np.int64)
        for width in self._widths():
            rest, part = np.divmod(rest, width)
            found.append(part)
        return found

    # -- counts and sums ---------------------------------------------------------------------

    def GetEntries(self) -> float:
        return float(self._core()["fEntries"])

    def SetEntries(self, n: float) -> None:
        self._core()["fEntries"] = float(n)

    def GetEffectiveEntries(self) -> float:
        return float(self._xrd.effective_entries)

    def GetSumOfWeights(self) -> float:
        """``GetSumOfWeights``: every bin on the axes, flow left out, range or no range."""
        return float(np.sum(self._xrd.values().astype(np.float64)))

    def _bin_ranges(self, args: tuple[Any, ...]) -> tuple[list[int], list[int], bool]:
        """The first and last bin per axis an ``Integral`` call names, and whether ``"width"``."""
        width = any(isinstance(arg, str) and "width" in arg.lower() for arg in args)
        numbers = [int(arg) for arg in args if not isinstance(arg, str)]
        lows = [axis.GetFirst() for axis in self._axes()]
        highs = [axis.GetLast() for axis in self._axes()]
        for at in range(min(len(numbers) // 2, len(lows))):
            lows[at], highs[at] = numbers[2 * at], numbers[2 * at + 1]
        return lows, highs, width

    def Integral(self, *args: Any) -> float:
        """``Integral([binx1, binx2[, biny1, biny2]][, "width"])``: the range drawn by default."""
        lows, highs, width = self._bin_ranges(args)
        return float(self._xrd.integral(lows, highs, width))

    def IntegralAndError(self, *args: Any) -> float:
        """``IntegralAndError(binx1, binx2, [...,] error[, option])``: the error into ``error``."""
        at = next(i for i, arg in enumerate(args) if not isinstance(arg, (int, float, np.integer)))
        lows, highs, width = self._bin_ranges(args[:at] + args[at + 1 :])
        store(args[at], float(self._xrd.integral_error(lows, highs, width)))
        return float(self._xrd.integral(lows, highs, width))

    # -- moments --------------------------------------------------------------------------

    def GetMean(self, axis: int = 1) -> float:
        """``GetMean(axis)``: 1, 2, 3 for x, y, z; 11, 12, 13 for the error on each."""
        if axis > 10:
            return self.GetMeanError(axis - 10)
        if self._ranged():
            return self._range_moments(axis - 1)[1]
        return float(self._xrd.mean(axis - 1))

    def GetMeanError(self, axis: int = 1) -> float:
        return float(self._xrd.mean_error(axis - 1))

    def GetStdDev(self, axis: int = 1) -> float:
        if axis > 10:
            return self.GetStdDevError(axis - 10)
        if self._ranged():
            return self._range_moments(axis - 1)[2]
        return float(self._xrd.std(axis - 1))

    def GetStdDevError(self, axis: int = 1) -> float:
        return float(self._xrd.std_error(axis - 1))

    GetRMS = GetStdDev
    GetRMSError = GetStdDevError

    def GetSkewness(self, axis: int = 1) -> float:
        return float(self._xrd.skewness_error() if axis > 10 else self._xrd.skewness(axis - 1))

    def GetKurtosis(self, axis: int = 1) -> float:
        return float(self._xrd.kurtosis_error() if axis > 10 else self._xrd.kurtosis(axis - 1))

    def GetCovariance(self, axis1: int = 1, axis2: int = 2) -> float:
        """``GetCovariance``: of two axes, from the running sums."""
        names = {(1, 2): "fTsumwxy", (1, 3): "fTsumwxz", (2, 3): "fTsumwyz"}
        low, high = sorted((axis1, axis2))
        if low == high:
            return self.GetStdDev(low) ** 2
        sums = self._xrd._moment_homes()
        total = float(sums["fTsumw"]["fTsumw"])
        product = float(sums[names[(low, high)]][names[(low, high)]])
        return product / total - self.GetMean(low) * self.GetMean(high) if total else 0.0

    def GetCorrelationFactor(self, axis1: int = 1, axis2: int = 2) -> float:
        spread = self.GetStdDev(axis1) * self.GetStdDev(axis2)
        return self.GetCovariance(axis1, axis2) / spread if spread else 0.0

    # -- contours -------------------------------------------------------------------------

    def SetContour(self, nlevels: int, levels: Any = None) -> None:
        """``SetContour(n, levels)``: the levels ``CONT`` draws - those given, a user's, or
        ``n`` evenly from the lowest content up to below the highest."""
        self.ResetBit(1 << 10)
        if int(nlevels) <= 0:
            self._core()["fContour"] = np.zeros(0)
            return
        if levels is not None:
            self.SetBit(1 << 10)
            chosen = [float(levels[i]) for i in range(int(nlevels))]
        else:
            low, high = self.GetMinimum(), self.GetMaximum()
            step = (high - low) / int(nlevels)
            chosen = [low + step * i for i in range(int(nlevels))]
        self._core()["fContour"] = np.asarray(chosen, dtype=np.float64)

    def GetContour(self, levels: Any = None) -> int:
        found = self._core().get("fContour")
        count = 0 if found is None else len(found)
        for i in range(count if levels is not None else 0):
            levels[i] = float(found[i])
        return count

    def GetContourLevel(self, level: int) -> float:
        found = self._core().get("fContour")
        return float(found[level]) if found is not None and 0 <= level < len(found) else 0.0

    # -- extremes -------------------------------------------------------------------------

    def SetMaximum(self, maximum: float = UNSET) -> None:
        self._core()["fMaximum"] = float(maximum)

    def SetMinimum(self, minimum: float = UNSET) -> None:
        self._core()["fMinimum"] = float(minimum)

    def GetMaximumStored(self) -> float:
        return float(self._core()["fMaximum"])

    def GetMinimumStored(self) -> float:
        return float(self._core()["fMinimum"])

    def GetMaximum(self, maxval: float = FLT_MAX) -> float:
        """``GetMaximum``: what ``SetMaximum`` said, else the largest bin drawn below ``maxval``."""
        if self.GetMaximumStored() != UNSET:
            return self.GetMaximumStored()
        values = self._contents()[self._displayed()].astype(np.float64)
        chosen = values[values < maxval]
        return float(max(np.max(chosen), -FLT_MAX)) if len(chosen) else -FLT_MAX

    def GetMinimum(self, minval: float = -FLT_MAX) -> float:
        if self.GetMinimumStored() != UNSET:
            return self.GetMinimumStored()
        values = self._contents()[self._displayed()].astype(np.float64)
        chosen = values[values > minval]
        return float(min(np.min(chosen), FLT_MAX)) if len(chosen) else FLT_MAX

    def GetMaximumBin(self, *refs: Any) -> int:
        """``GetMaximumBin``: the global bin of the first largest bin drawn."""
        bins = self._displayed()
        found = int(bins[int(np.argmax(self._contents()[bins]))])
        return self._located(found, refs)

    def GetMinimumBin(self, *refs: Any) -> int:
        bins = self._displayed()
        found = int(bins[int(np.argmin(self._contents()[bins]))])
        return self._located(found, refs)

    def _located(self, found: int, refs: tuple[Any, ...]) -> int:
        if refs:
            self.GetBinXYZ(found, *refs)  # type: ignore[attr-defined]
        return found

    # -- the x axis, from the histogram -----------------------------------------------------

    def GetBinCenter(self, bin: int) -> float:
        return float(self.GetXaxis().GetBinCenter(bin))

    def GetBinLowEdge(self, bin: int) -> float:
        return float(self.GetXaxis().GetBinLowEdge(bin))

    def GetBinWidth(self, bin: int) -> float:
        return float(self.GetXaxis().GetBinWidth(bin))

    def FindBin(self, x: Any, y: float = 0.0, z: float = 0.0) -> int:
        """``FindBin(x[, y[, z]])``: the global bin, flow counted."""
        found = [axis.FindBin(value) for axis, value in zip(self._axes(), (x, y, z))]
        return self.GetBin(*found)  # type: ignore[attr-defined,no-any-return]

    FindFixBin = FindBin

    def Interpolate(self, x: float, *rest: float) -> float:
        return float(self._xrd.interpolate(x))
