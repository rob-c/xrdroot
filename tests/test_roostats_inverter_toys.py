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


def test_a_search_more_accurate_than_the_toys_ends_in_a_fit(capsys: Any) -> None:
    """Asked for more accuracy than the toys give: the edges moved in about the crossing, then
    an exponential fitted to the points there, and the picture drawn."""
    ROOT.TCanvas("c", "c")
    it, _ = inverter()
    it.SetVerbose(1)
    found, limit, _ = it.RunLimit(None, None, 1e-4, 1e-5)
    assert found and limit == pytest.approx(3.85, abs=0.1)
    out = capsys.readouterr().out
    assert "Trying to move the interval edges closer" in out
    assert "HypoTestInverter::RunLimit - Before fit   --- \nLimit: mu < " in out
    assert "Fit to " in out
    assert it.GetLimitPlot().GetN() > 3


class AsymptoticCalculator(HypoTestCalculatorGeneric):
    """The same p-values at every ``mu`` - or none at all."""

    def __init__(self, sb: Any, b: Any, clsb: Any) -> None:
        self._null, self._alt, self._data = sb, b, None
        self._sampler, self.clsb = _Sampler(), clsb

    def IsTwoSided(self) -> bool:
        return False

    def GetHypoTest(self) -> Any:
        if self.clsb is None:
            return None
        made = HypoTestResult("flat", self.clsb, 1.0)
        made.SetBackgroundAsAlt(True)
        return made


def flat(clsb: Any, cls: bool = True) -> Any:
    _, sb, b = models()
    it = ROOT.RooStats.HypoTestInverter(AsymptoticCalculator(sb, b, clsb))
    it.UseCLs(cls)
    return it


@pytest.mark.parametrize(("clsb", "cls", "said"), [
    (0.5, True, "Cannot determine upper limit of scan range. At mu = 384  still getting CLs = 0.5"),
    (0.01, False, "Cannot determine lower limit of scan range. At mu = -96 still get CLsplusb"),
    (None, True, "Hypo test failed at x=2.20528 when trying to find limit."),
    (None, False, "Hypotest failed at lower limit of scan range: 0"),
])  # fmt: skip
def test_an_automatic_search_that_cannot_bracket_the_limit_fails(
        capsys: Any, clsb: Any, cls: bool, said: str) -> None:  # fmt: skip
    it = flat(clsb, cls)
    assert it.RunLimit()[0] is False
    assert said in capsys.readouterr().out
    flat(clsb, cls).GetInterval()
    assert "HypoTestInverter::GetInterval - error running an auto scan " in capsys.readouterr().out


def test_a_hint_narrows_the_search_and_a_toy_limit_stops_it(capsys: Any) -> None:
    it, _ = inverter()
    it.SetMaximumToys(1)
    it.RunLimit(None, None, 0, 0, 3.0)
    out = capsys.readouterr().out
    assert "HypoTestInverter::RunLimit - Use hint value 3 search in interval 0.9 , 6" in out
    assert "HypoTestInverter::RunLimit - maximum number of toys reached" in out


def test_a_point_with_no_error_estimate_stops_the_search(monkeypatch: Any, capsys: Any) -> None:
    from xrdroot.roostats import inverterscan

    it, _ = inverter()
    real = inverterscan._last
    calls = iter(range(100))
    monkeypatch.setattr(inverterscan, "_last",
                        lambda results: real(results) if next(calls) < 1 else (0.05, -1.0))
    assert it.RunLimit()[0] is False
    assert "[#0] ERROR:Eval -- Hypotest failed\n" in capsys.readouterr().out


def test_references_given_are_filled_with_the_limit_and_its_error() -> None:
    class Ref:
        value = 0.0

    it, _ = inverter()
    limit, error = Ref(), Ref()
    assert it.RunLimit(limit, error) is True
    assert limit.value == pytest.approx(3.85, abs=0.2) and error.value > 0


def _toy_scan() -> Any:
    it, _ = inverter()
    it.SetFixedScan(4, 1, 4)
    return it.GetInterval()


@pytest.mark.parametrize("option", ["", "OBS", "EXP", "SAME", "CLB 2CL", "EXP CLB"])
def test_the_scan_is_drawn_with_its_bands_size_line_and_legend(tmp_path: Any, option: str) -> None:
    canvas = ROOT.TCanvas("c", "c")
    result = _toy_scan()
    result.UseCLs(option != "EXP CLB")
    plot = ROOT.RooStats.HypoTestInverterPlot(result)
    assert plot.GetName() == result.GetName()
    plot.Draw(option)
    legend = plot._kept[-1]
    assert legend.ClassName() == "TLegend"
    canvas.SaveAs(str(tmp_path / "scan.png"))
    assert (tmp_path / "scan.png").stat().st_size > 1000


def test_the_plots_of_each_level_and_of_each_points_statistic(capfd: Any) -> None:
    result = _toy_scan()
    plot = ROOT.RooStats.HypoTestInverterPlot("p", "t", result)
    names = [plot.MakePlot(opt).GetName() for opt in ("CLb", "CLs+b", "CLsplusb", "CLs", "")]
    assert names == ["CLb_observed", "CLs+b_observed", "CLs+b_observed", "CLs_observed",
                     "CLs_observed"]  # fmt: skip
    bands = plot.MakeExpectedPlot(1.5, 0.5)
    assert [one.GetTitle() for one in bands.GetListOfGraphs()] == [
        "Expected CLs #pm 1.5 #sigma", "Expected CLs - Median"]  # fmt: skip
    assert len(list(plot.MakeExpectedPlot(0.0, 0.0).GetListOfGraphs())) == 1
    assert plot.MakeTestStatPlot(1, 0).ClassName().endswith("HypoTestPlot")
    assert plot.MakeTestStatPlot(1, 1) is not None and plot.MakeTestStatPlot(1, 2) is not None
    assert plot.MakeTestStatPlot(1, 3) is None and plot.MakeTestStatPlot(9, 0) is None
    result._results[2] = HypoTestResult("bad", float("nan"), 0.5)
    result._results[2].SetBackgroundAsAlt(True)
    assert plot.MakePlot().GetN() == 3
    assert ("Warning in <HypoTestInverterPlot::MakePlot>: Got a confidence level of nan at "
            "x=3.000000 (failed fit?). Skipping this point.") in capfd.readouterr().err


def test_a_scan_of_no_named_parameter_and_no_pad_draws_all_the_same(monkeypatch: Any) -> None:
    from xrdroot.pyroot.graphics import pads
    from xrdroot.roostats.inverterresult import HypoTestInverterResult

    ROOT.TCanvas("c", "c")
    bare = HypoTestInverterResult("bare")
    for x in (1.0, 2.0):
        bare.Add(x, HypoTestResult("p", 0.3 / x, 0.5))
    plot = ROOT.RooStats.HypoTestInverterPlot(bare)
    plot.Draw("OBS")
    monkeypatch.setattr(pads, "current", lambda: None)
    plot.Draw("OBS")
    assert plot._kept[0].GetXaxis().GetTitle() == ""


def test_the_expected_plot_leaves_out_a_point_with_no_expected_p_values() -> None:
    from xrdroot.roostats.inverterresult import HypoTestInverterResult

    mu = ROOT.RooRealVar("mu", "mu", 1, 0, 10)
    r = HypoTestInverterResult("r", mu, 0.95)
    r._two_sided = True
    for x, p in ((1.0, 1.0), (2.0, 0.1)):
        r.Add(x, HypoTestResult("p", p, 0.5))
    bands = ROOT.RooStats.HypoTestInverterPlot(r).MakeExpectedPlot()
    assert bands.GetListOfGraphs().At(2).GetN() == 1


def test_the_inverter_is_made_from_models_or_calculators_and_checks_them(capsys: Any) -> None:
    """Each calculator made from the models, the scanned variable guessed, the models checked."""
    from xrdroot.roostats import inverter as inv

    w, sb, b = models()
    data = ROOT.RooDataSet("d", "d", ROOT.RooArgSet(w.var("x")))
    data.add(ROOT.RooArgSet(w.var("x")))
    kinds = []
    for kind in (inv.kFrequentist, inv.kHybrid):
        made = ROOT.RooStats.HypoTestInverter(data, sb, b, None, kind)
        kinds.append((made._kind, type(made.GetHypoTestCalculator()).__name__))
    assert kinds == [(2, "FrequentistCalculator"), (1, "HybridCalculator")]
    none = ROOT.RooStats.HypoTestInverter(data, sb, b, None, inv.kUndefined)
    assert none.GetHypoTestCalculator() is None and none.GetTestStatistic() is None
    assert not none.SetTestStatistic("s")
    none.SetData(data)
    assert ROOT.RooStats.HypoTestInverter()._calc is None
    w.var("mu").setVal(1)
    b.SetSnapshot(ROOT.RooArgSet(w.var("mu")))
    ROOT.RooStats.HypoTestInverter(HypoTestCalculatorGeneric(data, sb, b))
    out = capsys.readouterr().out
    assert "HypoTestInverter - Cannot guess the variable to scan " in out
    assert "HypoTestInverter - Type of hypotest calculator is not supported " in out
    assert "using a B model  with POI mu not equal to zero  user must check input" in out


def test_the_scanned_variable_is_the_alternates_if_the_null_has_none(capsys: Any) -> None:
    from xrdroot.roostats import inverter as inv

    w, sb, b = models()
    empty = ROOT.RooStats.ModelConfig("empty", w)
    calc = FrequentistCalculator(empty, b)
    assert inv.variable_to_scan(calc).GetName() == "mu"
    assert inv.variable_to_scan(FrequentistCalculator(empty, empty)) is None
    inv.check_input_models(FrequentistCalculator(sb, None), w.var("mu"))
    inv.check_input_models(FrequentistCalculator(sb, empty), w.var("mu"))
    inv.check_input_models(FrequentistCalculator(sb, b), w.var("x"))
    out = capsys.readouterr().out
    assert "FATAL:InputArguments -- HypoTestInverter - model are not existing" in out
    assert "HypoTestInverter - B model has no pdf or observables defined" in out


def test_settings_before_any_result_and_a_fixed_scan_that_fails(capsys: Any) -> None:
    it, _ = inverter()
    it.SetTestSize(0.1)
    it.SetConfidenceLevel(0.9)
    assert it._results is None
    it.SetFixedScan(2, 3, 1)
    it.GetInterval()
    assert "HypoTestInverter::GetInterval - error running a fixed scan " in capsys.readouterr().out
    bare = ROOT.RooStats.HypoTestInverter()
    bare._var = ROOT.RooRealVar("v", "v", 0, 0, 1)
    assert bare._create_results().GetName() == "result_v"


def test_toy_points_with_errors_their_merges_cleanup_and_limit_distributions(capsys: Any) -> None:
    """Eleven toys a point: the most precise point near the target, the toys per point after a
    merge, the cleanup's quantiles of the toys, the limit distribution's minimum of ten."""
    from xrdroot.roostats import inverterlimits as limits

    it, _ = inverter(11)
    it.SetFixedScan(4, 1, 4)
    result = it.GetInterval()
    assert limits.closest_point_index(result, 0.6) == 2
    again, _ = inverter(11)
    again.SetFixedScan(4, 1, 4)
    result.Add(again.GetInterval())
    assert "HypoTestInverterResult::Add  - new toys/point is 22" in capsys.readouterr().out
    small, _ = inverter(5)
    small.SetFixedScan(3, 1, 3)
    assert small.GetInterval().GetUpperLimitDistribution().GetSize() == 10
    assert "set a minimum size of 10 for limit distribution" in capsys.readouterr().out
    eleven, _ = inverter(11)
    eleven.SetFixedScan(4, 1, 4)
    assert eleven.GetInterval().ExclusionCleanup() >= 0
    forty, _ = inverter(40)
    forty.SetFixedScan(3, 1, 3)
    assert forty.GetInterval().ExclusionCleanup() == 0
    assert "ExclusionCleanup - invalid size of sampling distribution" in capsys.readouterr().out


def test_the_limit_error_fit_for_a_lower_limit_and_with_too_few_points(capsys: Any) -> None:
    from xrdroot.roostats import inverterlimits as limits

    it, _ = inverter()
    it.SetFixedScan(5, 3.5, 3.9)
    result = it.GetInterval()
    result.UpperLimit()
    result._lower = 3.6
    assert limits.estimated_error(result, 0.05, True, 3.5, 3.9) >= 0.0
    assert limits.estimated_error(result, 0.05, False, 3.55, 3.75) >= 0.0
    assert limits.estimated_error(result, 0.05, False, 3.55, 3.65) == 0.0
    result._upper = float("nan")
    assert limits.estimated_error(result, 0.05, False, 3.5, 3.9) == 0.0
    result.UpperLimit()
    assert limits.estimated_error(result, 0.05, False, 3.7, 3.9) >= 0.0
    result.GetYError = lambda i: 0.0
    assert limits.estimated_error(result, 0.05, False, 3.8, 3.95) == 0.0
    assert "no valid points - cannot estimate  the upper limit error" in capsys.readouterr().out
    assert result.FindInterpolatedLimit(0.05, False, 3.5, 3.9) == pytest.approx(3.85, abs=0.05)
