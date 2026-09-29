"""``HypoTestResult``: p-values, CLs and their errors from two sampling distributions, as ROOT's.

The results below were made in ROOT through PyROOT from the same
hand-written distributions - ten null toys, ten weighted alternate toys -
and printed: the right tail and the left, the background as the null and as
the alternate. Joining two results, copying one, and the corners where a
p-value is zero are said as ROOT says them.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.roostats.hypotest import HypoTestResult
from xrdroot.roostats.sampling import SamplingDistribution


@pytest.fixture(autouse=True)
def _fresh(tmp_path: Any) -> Iterator[None]:
    yield from fresh(tmp_path)


def _toys() -> HypoTestResult:
    """Null toys 1 to 10; alternate toys 5 to 14 weighing 1 and 2 in turn; the data at 7."""
    result = HypoTestResult("toys")
    result.SetNullDistribution(SamplingDistribution("null", "null title", range(1, 11), "q"))
    result.SetAltDistribution(SamplingDistribution("alt", "alt title", range(5, 15),
                                                   [1, 2] * 5, "q"))  # fmt: skip
    result.SetTestStatisticData(7.0)
    return result


def _printed(*lines: str) -> str:
    return "\n".join(lines) + "\n"


RIGHT = _printed(
    "", "Results toys: ",
    " - Null p-value = 0.4 +/- 0.154919",
    " - Significance = 0.253347 +/- 0.40099 sigma",
    " - Number of Alt toys: 10",
    " - Number of Null toys: 10",
    " - Test statistic evaluated on data: 7",
    " - CL_b: 0.4 +/- 0.154919",
    " - CL_s+b: 0.8 +/- 0.133333",
    " - CL_s: 2 +/- 0.843274",
)  # fmt: skip


def test_a_result_without_toys_prints_its_two_p_values_alone(capfd: Any) -> None:
    """``HypoTestResult("plain", 0.1, 0.3)``: ROOT's printout, no errors, CLs of 3."""
    result = HypoTestResult("plain", 0.1, 0.3)
    result.Print()
    assert capfd.readouterr().out == _printed(
        "", "Results plain: ", " - Null p-value = 0.1", " - Significance = 1.28155",
        " - CL_b: 0.1", " - CL_s+b: 0.3", " - CL_s: 3")  # fmt: skip
    assert (result.CLs(), result.CLsError()) == (2.9999999999999996, 0.0)
    assert (result.Significance(), result.SignificanceError()) == (1.2815515655446004, 0.0)
    assert not result.HasTestStatisticData()


def test_toys_give_roots_closed_right_tails_and_their_binomial_errors(capfd: Any) -> None:
    """Both tails beyond 7, 7 included: ROOT's p-values, errors and printout."""
    result = _toys()
    result.Print()
    assert capfd.readouterr().out == RIGHT
    assert (result.NullPValue(), result.NullPValueError()) == (0.4, 0.15491933384829668)
    assert result.CLsError() == pytest.approx(0.8432740427115677, rel=1e-14)
    assert result.SignificanceError() == pytest.approx(0.40098958933175705, rel=1e-14)
    assert result.CLsplusbError() == pytest.approx(0.1333333333333333, rel=1e-14)


def test_the_left_tail_and_the_background_as_the_alternate(capfd: Any) -> None:
    """Left tails from 7 down; CL_b and CL_s+b swapped with the background as the alternate."""
    result = _toys()
    result.SetPValueIsRightTail(False)
    result.Print()
    left = capfd.readouterr().out
    assert " - Null p-value = 0.7 +/- 0.144914\n" in left
    assert " - CL_s: 0.380952 +/- 0.218498\n" in left
    result.SetBackgroundAsAlt(True)
    result.Print()
    swapped = capfd.readouterr().out
    assert swapped.endswith(" - CL_b: 0.266667 +/- 0.142638\n - CL_s+b: 0.7 +/- 0.144914\n"
                            " - CL_s: 2.625 +/- 1.50559\n")  # fmt: skip
    assert (result.CLb(), result.CLbError()) == pytest.approx(
        (0.26666666666666666, 0.14263828031894413), rel=1e-14)
    assert (result.CLsplusb(), result.CLsplusbError()) == pytest.approx(
        (0.7, 0.14491376746189438), rel=1e-14)
    assert (result.GetBackGroundIsAlt(), result.GetPValueIsRightTail()) == (True, False)


def _more() -> HypoTestResult:
    """Two null toys more, at 7 and 8, and one alternate toy at 0.5."""
    more = HypoTestResult("more")
    more.SetNullDistribution(SamplingDistribution("null", "null title", [7, 8], "q"))
    more.SetAltDistribution(SamplingDistribution("alt", "alt title", [0.5], "q"))
    more.SetTestStatisticData(7.0)
    return more


def test_appending_a_result_joins_the_toys_and_a_clone_keeps_everything(capfd: Any) -> None:
    """Twelve null toys and eleven alternate, the p-values again; the clone prints the same."""
    result = _toys()
    result.SetPValueIsRightTail(False)
    result.SetBackgroundAsAlt(True)
    result.Append(_more())
    result.Print()
    joined = capfd.readouterr().out
    assert joined == _printed(
        "", "Results toys: ",
        " - Null p-value = 0.666667 +/- 0.136083",
        " - Significance = -0.430727 +/- 0.374265 sigma",
        " - Number of Alt toys: 11",
        " - Number of Null toys: 12",
        " - Test statistic evaluated on data: 7",
        " - CL_b: 0.3125 +/- 0.142029",
        " - CL_s+b: 0.666667 +/- 0.136083",
        " - CL_s: 2.13333 +/- 1.06288",
    )  # fmt: skip
    copy = result.Clone("copy")
    copy.Print()
    assert capfd.readouterr().out == joined.replace("Results toys", "Results copy")
    assert (copy.GetName(), copy.GetNullDistribution().GetSize()) == ("copy", 12)
    assert copy.GetNullDistribution() is not result.GetNullDistribution()
    assert result.Clone().GetName() == "toys"


def test_a_result_without_data_takes_the_appended_ones_and_its_details(capfd: Any) -> None:
    """The data's value and each detailed output come from the result appended."""
    x = ROOT.RooRealVar("x", "", 0, 10)
    rows = []
    for value in (1.0, 2.0, 3.0):
        x.setVal(value)
        row = ROOT.RooDataSet(f"row{value:g}", "", ROOT.RooArgSet(x))
        row.add(ROOT.RooArgSet(x))
        rows.append(row)
    more = _more()
    more.SetNullDetailedOutput(rows[0])
    more.SetAltDetailedOutput(rows[1])
    info = ROOT.RooArgSet(x)
    more.SetFitInfo(info)
    result = HypoTestResult("empty")
    result.Append(more)
    assert result.GetTestStatisticData() == 7.0
    assert result.GetNullDetailedOutput().numEntries() == 1
    assert result.GetAltDetailedOutput().GetName() == "row2"
    assert result.GetFitInfo() is not info and len(result.GetFitInfo()) == 1
    again = HypoTestResult("again")
    again.SetNullDetailedOutput(rows[2])
    result.Append(again)
    assert list(result.GetNullDetailedOutput().column("x")) == [1.0, 3.0]
    assert result.GetAltDetailedOutput().numEntries() == 1


def test_a_background_p_value_of_zero_makes_no_cls(capfd: Any) -> None:
    """``CLb`` = 0: CLs is -1, said so as it is printed, and its error -1."""
    result = HypoTestResult("zero")
    result.SetNullDistribution(SamplingDistribution("null", "", [1, 2]))
    result.SetAltDistribution(SamplingDistribution("alt", "", [1, 2]))
    result.SetTestStatisticData(5.0)
    assert result.CLsError() == -1.0
    capfd.readouterr()
    assert result.CLs() == -1.0
    assert capfd.readouterr().out == (
        "Error: Cannot compute CLs because CLb = 0. Returning CLs = -1\n")


def test_the_p_values_and_data_set_by_hand() -> None:
    """Each value set is the value got; all the statistics' values set the data's."""
    result = HypoTestResult("hand")
    for setter, getter, value in (("SetNullPValue", "NullPValue", 0.2),
                                  ("SetNullPValueError", "NullPValueError", 0.01),
                                  ("SetAltPValue", "AlternatePValue", 0.6),
                                  ("SetAltPValueError", "CLsplusbError", 0.02)):  # fmt: skip
        getattr(result, setter)(value)
        assert getattr(result, getter)() == value
    result.SetAllTestStatisticsData(None)
    assert result.GetAllTestStatisticsData() is None
    values = [ROOT.RooRealVar("ts0", "", 3.5), ROOT.RooRealVar("ts1", "", 1.0)]
    result.SetAllTestStatisticsData(values)
    assert result.GetAllTestStatisticsData() is values and result.GetTestStatisticData() == 3.5
    result.SetAllTestStatisticsData(ROOT.RooArgList(*values))
    assert [v.GetName() for v in result.GetAllTestStatisticsData()] == ["ts0", "ts1"]
    result.SetAllTestStatisticsData([])
    assert result.GetTestStatisticData() == 3.5
