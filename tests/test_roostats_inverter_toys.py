"""HypoTestInverter with toy-based results: the paths a frequentist scan takes.

A stand-in for the frequentist calculator gives, at each value of ``mu``,
toy distributions of known shape - the signal-plus-background one's upper
tail beyond the data shrinking as ``mu`` grows - so each step of the scan,
the limit's error fit, the expected limits and the automatic search can be
followed without the fits that real toys would cost.
"""

from __future__ import annotations

from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roofit.messages import service
from xrdroot.roostats.calculators import HypoTestCalculatorGeneric
from xrdroot.roostats.hypotest import HypoTestResult
from xrdroot.roostats.sampling import SamplingDistribution


@pytest.fixture(autouse=True)
def _quiet() -> Any:
    service().reset()
    yield
    service().reset()


class _Sampler:
    def __init__(self) -> None:
        self.stat: Any = "stat"

    def GetTestStatistic(self) -> Any:
        return self.stat

    def SetTestStatistic(self, stat: Any) -> None:
        self.stat = stat


class FrequentistCalculator(HypoTestCalculatorGeneric):
    """Toys of a known shape: the null's tail beyond the data at one is ``1 - mu / 4``."""

    def __init__(self, sb: Any, b: Any, toys: int = 40, data: float = 1.0) -> None:
        self._null, self._alt, self._data = sb, b, None
        self._sampler = _Sampler()
        self.toys, self.value, self.calls = toys, data, 0

    def SetToys(self, null: int, alt: int) -> None:
        self.toys = max(int(null), 40)

    def GetHypoTest(self) -> Any:
        self.calls += 1
        mu = self._null.GetSnapshot().first().getVal()
        n = self.toys
        null = [4.0 * (k + 0.5) / (n * max(mu, 1e-3)) for k in range(n)]
        alt = [4.0 * (k + 0.5) / n for k in range(n)]
        made = HypoTestResult("toys")
        made.SetNullDistribution(SamplingDistribution("null", "null", null))
        made.SetAltDistribution(SamplingDistribution("alt", "alt", alt))
        made.SetTestStatisticData(self.value)
        return made


def models() -> tuple[Any, Any, Any]:
    w = ROOT.RooWorkspace("w")
    w.factory("Gaussian::g(x[0,-5,5], mu[1,0,6], 1)")
    sb = ROOT.RooStats.ModelConfig("sb", w)
    sb.SetPdf("g")
    sb.SetObservables("x")
    sb.SetParametersOfInterest("mu")
    sb.SetSnapshot(ROOT.RooArgSet(w.var("mu")))
    b = sb.Clone("b")
    w.var("mu").setVal(0)
    b.SetSnapshot(ROOT.RooArgSet(w.var("mu")))
    return w, sb, b


def inverter(toys: int = 40) -> tuple[Any, Any]:
    _, sb, b = models()
    calc = FrequentistCalculator(sb, b, toys)
    it = ROOT.RooStats.HypoTestInverter(calc)
    it.UseCLs(True)
    return it, calc


def test_a_toy_scan_names_each_points_distributions_and_counts_its_toys() -> None:
    it, calc = inverter()
    it.SetFixedScan(4, 1, 4)
    result = it.GetInterval()
    assert result.ArraySize() == 4 and calc.calls == 4
    assert result.GetNullTestStatDist(1).GetName() == "null_mu_2.00"
    assert result.GetAltTestStatDist(1).GetName() == "alt_mu_2.00"
    assert it._toys_run == 320
    assert [result.GetYValue(i) for i in range(4)] == [1.0, 2 / 3, 1 / 3, 0.0]
    assert result.UpperLimit() == pytest.approx(3.85, rel=1e-8)
    assert result.UpperLimitEstimatedError() == 0.0  # the nearest point has no error to scale
    assert (result.LowerLimit(), result.LowerLimitEstimatedError()) == (0.0, 0.0)


def test_the_expected_limits_come_from_the_background_toys() -> None:
    """Each background toy as data: its CLs - forty of them a point - the limits of their
    quantiles, or with ``P`` the limit of the curve through each point's quantile."""
    it, _ = inverter()
    it.SetFixedScan(4, 1, 4)
    result = it.GetInterval()
    dist = result.GetExpectedPValueDist(1)
    assert dist.GetSize() == 40 and dist.GetSamplingDistribution()[0] == 0.975
    assert [result.GetExpectedUpperLimit(s) for s in (-1, 0, 1)] == pytest.approx(
        [1.95, 1.973749777720621, 6.0], rel=1e-6)  # fmt: skip
    assert result.GetExpectedUpperLimit(0, "P") == pytest.approx(1.9731724721405375, rel=1e-6)
    assert result.GetUpperLimitDistribution().GetSize() == 40
    assert result.GetLowerLimitDistribution().GetName() == "Expected lower Limit"
    assert result.GetExpectedPValueDist(9) is None


def test_the_limit_error_is_the_nearest_points_error_over_the_fitted_slope() -> None:
    it, _ = inverter()
    it.SetFixedScan(5, 3.5, 3.9)
    result = it.GetInterval()
    assert result.UpperLimit() == pytest.approx(3.8, abs=0.1)
    assert result.UpperLimitEstimatedError() > 0


def test_an_automatic_toy_scan_brackets_bisects_and_fits(capsys: Any) -> None:
    """The top of the range where CLs is zero, the bottom at one - CLs from zero - then points
    run with toys until precise, and the limit where they cross the size."""
    it, _calc = inverter()
    it.SetVerbose(2)
    result = it.GetInterval()
    assert result.UpperLimit() == pytest.approx(3.85, abs=0.19)  # the accuracy asked for
    assert result.LowerLimit() == 0.0
    out = capsys.readouterr().out
    assert "Search for upper limit to the limit" in out
    assert "HypoTestInverter::RunLimit - Now doing proper bracketing & bisection" in out
    assert "\tLimit: mu < 3.9375 +/- 0.1875 @ 95% CL\n" in out
    assert it.GetLimitPlot() is None or it.GetLimitPlot().GetN() > 0


def test_the_settings_are_kept_and_passed_to_the_results() -> None:
    it, calc = inverter()
    it.SetFixedScan(2, 1, 2)
    assert it.GetInterval().ConfidenceLevel() == pytest.approx(0.95)
    it.SetTestSize(0.1)
    assert (it.Size(), it._results.ConfidenceLevel()) == (0.1, 0.9)
    it.SetConfidenceLevel(0.68)
    assert it.ConfidenceLevel() == 0.68 and it._results.ConfidenceLevel() == 0.68
    it.UseCLs(False)
    assert not it._results._use_cls
    it.SetMaximumToys(10)
    it.SetNumErr(0.5)
    assert it.GetHypoTestCalculator() is calc
    assert it.GetTestStatistic() == "stat"
    assert it.SetTestStatistic("other") and calc._sampler.stat == "other"
    it.SetData("data")
    assert calc.GetData() == "data"
    it.SetCloseProof(True)
    it.SetAutoScan()
    assert (it._nbins, it._xmin, it._xmax) == (0, 1.0, -1.0)
    it.Clear()
    assert it._results is None and it.GetLimitPlot() is None


def test_a_second_interval_is_a_copy_of_the_first(capsys: Any) -> None:
    it, calc = inverter()
    it.SetFixedScan(2, 1, 2)
    first = it.GetInterval()
    again = it.GetInterval()
    assert again is not first and again.ArraySize() == 2 and calc.calls == 2
    assert "return an already existing interval" in capsys.readouterr().out


def test_the_limit_distributions_need_results_and_are_not_rebuilt(capsys: Any) -> None:
    from xrdroot.errors import UnsupportedFeatureError

    it, _ = inverter()
    assert it.GetUpperLimitDistribution() is None
    assert it.GetLowerLimitDistribution() is None
    out = capsys.readouterr().out
    assert "HypoTestInverter::GetUpperLimitDistribution(false) - result not existing" in out
    with pytest.raises(UnsupportedFeatureError, match="rebuilding them from new toys"):
        it.GetUpperLimitDistribution(True)
    with pytest.raises(UnsupportedFeatureError):
        it.RebuildDistributions()
    it.SetFixedScan(3, 1, 3)
    it.GetInterval()
    assert it.GetUpperLimitDistribution().GetSize() == 40


@pytest.mark.parametrize(("args", "said"), [
    ((0, 1, 2), "HypoTestInverter::RunFixedScan - Please provide nBins>0"),
    ((1, 1, 2), "nBins==1 -> I will run for xMin (1)"),
    ((3, 2, 2), "xMin==xMax -> I will enforce nBins==1"),
    ((3, 3, 2), "Please provide xMin (3) smaller than xMax (2)"),
    ((3, -1, 2), "xMin < lower bound, using xmin = 0"),
    ((3, 1, 9), "xMax > upper bound, using xmax = 6"),
    ((3, 0, 2, True), "cannot go in log steps if xMin <= 0"),
])  # fmt: skip
def test_a_fixed_scan_checks_its_range_as_roostats_does(capsys: Any, args: Any, said: str) -> None:
    it, _ = inverter()
    it.RunFixedScan(*args)
    assert said in capsys.readouterr().out


def test_a_log_scan_steps_evenly_in_the_log() -> None:
    it, _ = inverter()
    it.RunFixedScan(3, 1, 4, True)
    assert it._results._x == pytest.approx([1.0, 2.0, 4.0])


class _Failing(FrequentistCalculator):
    """No result at all, or p-values that are not probabilities."""

    def __init__(self, sb: Any, b: Any, mode: str) -> None:
        super().__init__(sb, b)
        self.mode = mode

    def GetHypoTest(self) -> Any:
        if self.mode == "none":
            return None
        return HypoTestResult("bad", float("nan"), 0.5)


@pytest.mark.parametrize(("mode", "said"), [
    ("none", "HypoTestInverter::Eval - HypoTest failed"),
    ("nan", "HypoTestInverter - Skipping invalid result for  point mu = 2. null p-value=nan"),
])  # fmt: skip
def test_a_failed_point_is_skipped_and_said(capsys: Any, mode: str, said: str) -> None:
    _, sb, b = models()
    it = ROOT.RooStats.HypoTestInverter(_Failing(sb, b, mode))
    assert not it.RunOnePoint(2.0)
    it.RunFixedScan(1, 2, 2)
    out = capsys.readouterr().out
    assert said in out
    assert "HypoTestInverter::RunFixedScan - The hypo test for point 2 failed. Skipping." in out


def test_a_point_outside_the_range_is_moved_to_its_end_and_a_repeat_merged(capsys: Any) -> None:
    it, _calc = inverter()
    it.SetVerbose(1)
    assert it.RunOnePoint(-1.0) and it.RunOnePoint(7.0) and it.RunOnePoint(6.0 * (1 + 1e-13))
    out = capsys.readouterr().out
    assert ("HypoTestInverter::RunOnePoint - Out of range: using the lower bound 0 on the scanned "
            "variable rather than -1") in out
    assert "using the upper bound 6 on the scanned variable rather than 7" in out
    assert "HypoTestInverter::RunOnePoint - Merge with previous result for mu = 6" in out
    assert "Running for mu = 6" in out and "P values for  mu =  6\n\tCLs      = " in out
    assert it._results.ArraySize() == 2
    assert it._results.GetResult(1).GetNullDistribution().GetSize() == 80
