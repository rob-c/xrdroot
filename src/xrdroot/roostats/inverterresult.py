"""``HypoTestInverterResult``: the tests of a scan, and the interval they invert to.

It keeps a :class:`~.hypotest.HypoTestResult` for each value of the scanned
parameter tried - ``CLs``, or ``CLs+b``, their ``y`` - and finds where the
curve through them crosses one less the confidence level: the upper limit,
or the lower, by MathCore's root finder on the straight lines between the
points. Expected limits come from each point's expected p-values - the
background toys', or the asymptotic formulae's at so many sigma.
"""

from __future__ import annotations

import math
from typing import Any

from ..roofit.messages import ERROR, log
from .intervals import SimpleInterval

__all__ = ["HypoTestInverterResult"]


class HypoTestInverterResult(SimpleInterval):
    """The scan: each value tried, its test, and the limits found from them."""

    kLinear, kSpline = 0, 1

    def __init__(self, name: Any = None, scannedVariable: Any = None, cl: float = 0.0) -> None:
        var = scannedVariable.clone(scannedVariable.GetName()) if scannedVariable else None
        super().__init__(name or "", var, math.nan, math.nan, cl)
        self._use_cls = self._two_sided = False
        self._interpolate = [True, True]  # lower, upper
        self._fitted = [False, False]
        self._interpolation = self.kLinear
        self._errors = [-1.0, -1.0]
        self._cleanup = 0.005
        self._x: list[float] = []
        self._results: list[Any] = []
        self._expected: list[Any] = []

    def Clone(self, name: Any = None) -> HypoTestInverterResult:
        """A copy - ``TObject::Clone``'s, streamed: the limits found so far too."""
        made = HypoTestInverterResult.__new__(HypoTestInverterResult)
        made.__dict__.update(self.__dict__)
        made._name = self._name if name is None else str(name)
        made._interpolate, made._fitted = list(self._interpolate), list(self._fitted)
        made._errors, made._x = list(self._errors), list(self._x)
        made._results = [one.Clone() for one in self._results]
        made._expected = []
        return made

    # -- the settings ---------------------------------------------------------------

    def UseCLs(self, on: bool = True) -> None:
        self._use_cls = bool(on)

    def SetTestSize(self, size: float) -> None:
        self._cl = 1.0 - float(size)

    def SetCLsCleanupThreshold(self, threshold: float) -> None:
        self._cleanup = float(threshold)

    def IsOneSided(self) -> bool:
        return not self._two_sided

    def IsTwoSided(self) -> bool:
        return self._two_sided

    def SetInterpolationOption(self, option: int) -> None:
        self._interpolation = int(option)

    def GetInterpolationOption(self) -> int:
        return self._interpolation

    # -- the points -----------------------------------------------------------------

    def ArraySize(self) -> int:
        return len(self._x)

    def _valid(self, index: int) -> bool:
        if 0 <= index < self.ArraySize():
            return True
        log(self, ERROR, "InputArguments", "Problem: You are asking for an impossible array "
            "index value")  # fmt: skip
        return False

    def GetXValue(self, index: int) -> float:
        return self._x[index] if self._valid(index) else -999.0

    def GetResult(self, index: int) -> Any:
        return self._results[index] if self._valid(index) else None

    def _of(self, index: int, what: str) -> float:
        found = self.GetResult(index)
        return float(getattr(found, what)()) if found is not None else -999.0

    def GetYValue(self, index: int) -> float:
        return self._of(index, "CLs" if self._use_cls else "CLsplusb")

    def GetYError(self, index: int) -> float:
        return self._of(index, "CLsError" if self._use_cls else "CLsplusbError")

    def CLb(self, index: int) -> float:
        return self._of(index, "CLb")

    def CLsplusb(self, index: int) -> float:
        return self._of(index, "CLsplusb")

    def CLs(self, index: int) -> float:
        return self._of(index, "CLs")

    def CLbError(self, index: int) -> float:
        return self._of(index, "CLbError")

    def CLsplusbError(self, index: int) -> float:
        return self._of(index, "CLsplusbError")

    def CLsError(self, index: int) -> float:
        return self._of(index, "CLsError")

    def GetLastXValue(self) -> float:
        return self.GetXValue(self.ArraySize() - 1)

    def GetLastYValue(self) -> float:
        return self.GetYValue(self.ArraySize() - 1)

    def GetLastYError(self) -> float:
        return self.GetYError(self.ArraySize() - 1)

    def GetLastResult(self) -> Any:
        return self.GetResult(self.ArraySize() - 1)

    def FindIndex(self, xvalue: float) -> int:
        """The index of the point at ``xvalue`` - to a relative 1e-12 above one, an absolute one
        below - or -1."""
        for i, x in enumerate(self._x):
            if _same(xvalue, x):
                return i
        return -1

    def Add(self, x: Any, result: Any = None) -> bool:
        """``Add(x, result)``: the test at ``x``, merged into one already there - or ``Add(other)``,
        every point of another scan."""
        if result is None:
            return self._add_scan(x)
        index = self.FindIndex(float(x))
        if index < 0:
            self._x.append(float(x))
            self._results.append(result.Clone())
        else:
            self._results[index].Append(result)
        self._lower = self._upper = math.nan
        return True

    def _add_scan(self, other: Any) -> bool:
        """``Add(otherResult)``: each of its points appended to the one here at the same value -
        of those here before - or added, and the limits to be found again."""
        from ..roofit.messages import INFO

        before = self.ArraySize()
        if not other.ArraySize():
            return True
        log(self, INFO, "Eval", f"HypoTestInverterResult::Add - merging result from "
            f"{other.GetName()} in {self.GetName()}")  # fmt: skip
        for x, result in zip(other._x, other._results, strict=False):
            index = next((j for j in range(before) if _same(x, self._x[j])), -1)
            if index >= 0:
                self._results[index].Append(result)
            else:
                self._results.append(result.Clone())
                self._x.append(x)
        if self.ArraySize() > before:
            log(self, INFO, "Eval", "HypoTestInverterResult::Add  - new number of points is "
                f"{self.ArraySize()}")  # fmt: skip
        else:
            toys = self._results[0].GetNullDistribution().GetSize()
            log(self, INFO, "Eval", f"HypoTestInverterResult::Add  - new toys/point is {toys}")
        self._lower = self._upper = math.nan
        return True

    # -- the distributions of each point --------------------------------------------

    def GetBackgroundTestStatDist(self, index: int) -> Any:
        found = self._results[index] if 0 <= index < len(self._results) else None
        if found is None:
            return None
        return found.GetAltDistribution() if found.GetBackGroundIsAlt() else (
            found.GetNullDistribution())  # fmt: skip

    def GetSignalAndBackgroundTestStatDist(self, index: int) -> Any:
        found = self._results[index] if 0 <= index < len(self._results) else None
        if found is None:
            return None
        return found.GetNullDistribution() if found.GetBackGroundIsAlt() else (
            found.GetAltDistribution())  # fmt: skip

    def GetNullTestStatDist(self, index: int) -> Any:
        return self.GetSignalAndBackgroundTestStatDist(index)

    def GetAltTestStatDist(self, index: int) -> Any:
        return self.GetBackgroundTestStatDist(index)

    def GetExpectedPValueDist(self, index: int) -> Any:
        from .inverterlimits import expected_p_value_dist

        return expected_p_value_dist(self, index)

    def GetLowerLimitDistribution(self) -> Any:
        from .inverterlimits import limit_distribution

        return limit_distribution(self, True)

    def GetUpperLimitDistribution(self) -> Any:
        from .inverterlimits import limit_distribution

        return limit_distribution(self, False)

    def GetExpectedLowerLimit(self, nsig: float = 0.0, opt: str = "") -> float:
        from .inverterlimits import expected_limit

        return expected_limit(self, nsig, True, opt)

    def GetExpectedUpperLimit(self, nsig: float = 0.0, opt: str = "") -> float:
        from .inverterlimits import expected_limit

        return expected_limit(self, nsig, False, opt)

    # -- the limits -----------------------------------------------------------------

    def _limit(self, lower: bool) -> float:
        from .inverterlimits import closest_point_index, find_interpolated_limit

        side = 0 if lower else 1
        if self._fitted[side]:
            return self._lower if lower else self._upper
        if not self._interpolate[side]:
            found = self.GetXValue(closest_point_index(self, 1 - self._cl))
            self._lower, self._upper = (found, self._upper) if lower else (self._lower, found)
        elif math.isnan(self._lower if lower else self._upper):
            find_interpolated_limit(self, 1 - self._cl, lower)
        return self._lower if lower else self._upper

    def LowerLimit(self, *args: Any) -> float:
        return self._limit(True)

    def UpperLimit(self, *args: Any) -> float:
        return self._limit(False)

    def _limit_error(self, lower: bool) -> float:
        from .inverterlimits import estimated_error

        if math.isnan(self._lower if lower else self._upper):
            self._limit(lower)
        found = self._errors[0 if lower else 1]
        return found if found >= 0 else estimated_error(self, 1 - self._cl, lower)

    def LowerLimitEstimatedError(self) -> float:
        return self._limit_error(True)

    def UpperLimitEstimatedError(self) -> float:
        return self._limit_error(False)

    def FindInterpolatedLimit(self, target: float, lowSearch: bool = False, xmin: float = 1.0,
                              xmax: float = 0.0) -> float:  # fmt: skip
        from .inverterlimits import find_interpolated_limit

        return find_interpolated_limit(self, target, lowSearch, xmin, xmax)

    def ExclusionCleanup(self) -> int:
        from .inverterlimits import exclusion_cleanup

        return exclusion_cleanup(self)


def _same(x: float, other: float) -> bool:
    """``TMath::AreEqualRel`` to 1e-12 for a value above one, ``AreEqualAbs`` below."""
    if abs(x) > 1:
        return abs(x - other) <= 1e-12 * (abs(x) + abs(other)) / 2
    return abs(x - other) < 1e-12
