"""``HypoTestResult``: the p-values of a test of a null hypothesis against an alternate.

The p-values of the null and the alternate are both on the same side of the
test statistic's value on the data - the right tail, unless it is said
otherwise - and both include that value itself, as limits need for discrete
distributions. ``CL_b`` is the background's p-value, ``CL_s+b`` the
other's, and ``CL_s`` their ratio; which is which follows whether the
background is the alternate. From toys, each p-value has a binomial error.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from ..roofit import cout
from ..roofit.printing import g
from .intervals import Named
from .utils import PValueToSignificance

__all__ = ["HypoTestResult"]


class _Cell:
    value = 0.0


class HypoTestResult(Named):
    """The result of a hypothesis test: two p-values, and the distributions they came from."""

    def __init__(self, name: Any = "", nullp: float = math.nan, altp: float = math.nan) -> None:
        super().__init__(name)
        self._null_p, self._alt_p = float(nullp), float(altp)
        self._null_error = self._alt_error = 0.0
        self._data_value = math.nan
        self._all_data: Any = None
        self._null: Any = None
        self._alt: Any = None
        self._null_detailed: Any = None
        self._alt_detailed: Any = None
        self._fit_info: Any = None
        self._right_tail, self._background_is_alt = True, False

    def Clone(self, name: str = "") -> HypoTestResult:
        found = HypoTestResult(self._name, self._null_p, self._alt_p)
        found._title = self._title
        found._null_error, found._alt_error = self._null_error, self._alt_error
        found._data_value = self._data_value
        found._right_tail, found._background_is_alt = self._right_tail, self._background_is_alt
        found.Append(self)
        if name:
            found.SetName(name)
        return found

    def Append(self, other: HypoTestResult) -> None:
        """Another result's toys added to this one's, and the p-values again."""
        self._null = _joined(self._null, other.GetNullDistribution())
        self._alt = _joined(self._alt, other.GetAltDistribution())
        self._null_detailed = _appended(self._null_detailed, other.GetNullDetailedOutput())
        self._alt_detailed = _appended(self._alt_detailed, other.GetAltDetailedOutput())
        self._fit_info = _appended(self._fit_info, other.GetFitInfo())
        if math.isnan(self._data_value):
            self._data_value = other.GetTestStatisticData()
        self._update()

    # -- the p-values -------------------------------------------------------------

    def _p_value(self, distribution: Any) -> tuple[float, float] | None:
        """``UpdatePValue``: the closed tail beyond the data's value, and its error."""
        if math.isnan(self._data_value) or distribution is None:
            return None
        error = _Cell()
        if self._right_tail:
            found = distribution.IntegralAndError(error, self._data_value, math.inf, True, True,
                                                  True)  # fmt: skip
        else:
            found = distribution.IntegralAndError(error, -math.inf, self._data_value, True, True,
                                                  True)  # fmt: skip
        return float(found), float(error.value)

    def _update(self) -> None:
        null, alt = self._p_value(self._null), self._p_value(self._alt)
        if null is not None:
            self._null_p, self._null_error = null
        if alt is not None:
            self._alt_p, self._alt_error = alt

    def NullPValue(self) -> float:
        return self._null_p

    def AlternatePValue(self) -> float:
        return self._alt_p

    def CLb(self) -> float:
        return self.AlternatePValue() if self._background_is_alt else self.NullPValue()

    def CLsplusb(self) -> float:
        return self.NullPValue() if self._background_is_alt else self.AlternatePValue()

    def CLs(self) -> float:
        clb = self.CLb()
        if clb == 0:
            cout.write("Error: Cannot compute CLs because CLb = 0. Returning CLs = -1\n")
            return -1.0
        return self.CLsplusb() / clb

    def Significance(self) -> float:
        return float(PValueToSignificance(self.NullPValue()))

    # -- errors -------------------------------------------------------------------

    def NullPValueError(self) -> float:
        return self._null_error

    def CLbError(self) -> float:
        return self._alt_error if self._background_is_alt else self._null_error

    def CLsplusbError(self) -> float:
        return self._null_error if self._background_is_alt else self._alt_error

    def SignificanceError(self) -> float:
        z = self.Significance()
        return self.NullPValueError() / (math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi))

    def CLsError(self) -> float:
        if self._alt is None or self._null is None:
            return 0.0
        if self.CLb() == 0:
            return -1.0
        clb2, clsb2 = self.CLbError() ** 2, self.CLsplusbError() ** 2
        return math.sqrt(clsb2 + clb2 * self.CLs() ** 2) / self.CLb()

    # -- setting ------------------------------------------------------------------

    def SetNullPValue(self, value: float) -> None:
        self._null_p = float(value)

    def SetNullPValueError(self, value: float) -> None:
        self._null_error = float(value)

    def SetAltPValue(self, value: float) -> None:
        self._alt_p = float(value)

    def SetAltPValueError(self, value: float) -> None:
        self._alt_error = float(value)

    def SetAltDistribution(self, distribution: Any) -> None:
        self._alt = distribution
        self._update()

    def SetNullDistribution(self, distribution: Any) -> None:
        self._null = distribution
        self._update()

    def SetAltDetailedOutput(self, data: Any) -> None:
        self._alt_detailed = data

    def SetNullDetailedOutput(self, data: Any) -> None:
        self._null_detailed = data

    def SetFitInfo(self, data: Any) -> None:
        self._fit_info = data

    def SetTestStatisticData(self, value: float) -> None:
        self._data_value = float(value)
        self._update()

    def SetAllTestStatisticsData(self, values: Any) -> None:
        if values is not None:
            self._all_data = values.snapshot() if hasattr(values, "snapshot") else values
        if self._all_data is not None and len(self._all_data):
            self.SetTestStatisticData(self._all_data[0].getVal())

    def SetPValueIsRightTail(self, right: bool) -> None:
        self._right_tail = bool(right)
        self._update()

    def GetPValueIsRightTail(self) -> bool:
        return self._right_tail

    def SetBackgroundAsAlt(self, alt: bool = True) -> None:
        self._background_is_alt = bool(alt)

    def GetBackGroundIsAlt(self) -> bool:
        return self._background_is_alt

    def GetNullDistribution(self) -> Any:
        return self._null

    def GetAltDistribution(self) -> Any:
        return self._alt

    def GetNullDetailedOutput(self) -> Any:
        return self._null_detailed

    def GetAltDetailedOutput(self) -> Any:
        return self._alt_detailed

    def GetFitInfo(self) -> Any:
        return self._fit_info

    def GetTestStatisticData(self) -> float:
        return self._data_value

    def GetAllTestStatisticsData(self) -> Any:
        return self._all_data

    def HasTestStatisticData(self) -> bool:
        return not math.isnan(self._data_value)

    # -- printing -----------------------------------------------------------------

    def Print(self, option: str = "") -> None:
        """The p-values, significance and CLs - with their errors, from toys."""
        toys = self._alt is not None or self._null is not None
        cout.write("\n".join(self._lines(toys)) + "\n")
        for label, value, error in ((" - CL_b: ", self.CLb, self.CLbError),
                                    (" - CL_s+b: ", self.CLsplusb, self.CLsplusbError),
                                    (" - CL_s: ", self.CLs, self.CLsError)):  # fmt: skip
            cout.write(label)  # before the value, which may complain as C++ reckons it
            cout.write(g(value()))
            cout.write((f" +/- {g(error())}" if toys else "") + "\n")

    def _lines(self, toys: bool) -> list[str]:
        """The printout's first lines: the p-value, significance, and toys and data if any."""
        lines = ["", f"Results {self._name}: "]
        lines.append(f" - Null p-value = {g(self.NullPValue())}"
                     + (f" +/- {g(self.NullPValueError())}" if toys else ""))  # fmt: skip
        lines.append(f" - Significance = {g(self.Significance())}"
                     + (f" +/- {g(self.SignificanceError())} sigma" if toys else ""))  # fmt: skip
        if self._alt is not None:
            lines.append(f" - Number of Alt toys: {self._alt.GetSize()}")
        if self._null is not None:
            lines.append(f" - Number of Null toys: {self._null.GetSize()}")
        if self.HasTestStatisticData():
            lines.append(f" - Test statistic evaluated on data: {g(self._data_value)}")
        return lines


def _joined(mine: Any, other: Any) -> Any:
    """``fNullDistr->Add(other)``, or a copy of the other's where there is none."""
    if mine is not None:
        mine.Add(other)
        return mine
    return copy.deepcopy(other) if other is not None else None


def _appended(mine: Any, other: Any) -> Any:
    if other is None:
        return mine
    if mine is not None:
        mine.append(other)
        return mine
    return other.Clone() if hasattr(other, "Clone") else copy.copy(other)
