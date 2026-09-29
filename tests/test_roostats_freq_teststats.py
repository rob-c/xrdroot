"""RooStats' test statistics on ten events of a Gaussian, as ROOT 6.40 evaluates them.

The profile likelihood ratio - two-sided, one-sided, for discovery, signed
- its detailed output of both fits, the simple likelihood ratio, the ratio
of two profiles, the number of events and the maximum-likelihood estimate
were evaluated in ROOT through PyROOT on the same events, drawn after
``RooRandom::randomGenerator()->SetSeed(4357)``; the values are ROOT's to
Minuit's tolerance. The retries RooStats makes when a fit fails are driven
by a minimizer that reports failures, since these models' fits never fail.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.roofit.fitting.minimizer import RooMinimizer
from xrdroot.roostats import moretests, teststats
from xrdroot.roostats.moretests import (
    MaxLikelihoodEstimateTestStat,
    NumEventsTestStat,
    RatioOfProfiledLikelihoodsTestStat,
    SimpleLikelihoodRatioTestStat,
)
from xrdroot.roostats.teststats import ProfileLikelihoodTestStat, TestStatistic, fit_nll

REL = 1e-6


@pytest.fixture(autouse=True)
def _fresh(tmp_path: Any) -> Iterator[None]:
    """A clean ROOT session; the likelihoods reused, as RooStats' default is."""
    yield from fresh(tmp_path)
    ProfileLikelihoodTestStat.SetAlwaysReuseNLL(True)
    SimpleLikelihoodRatioTestStat.SetAlwaysReuseNLL(True)


def _gauss() -> tuple[Any, ...]:
    """Ten events of ``Gauss(y; m, s)``, drawn after seed 4357."""
    ROOT.RooRandom.randomGenerator().SetSeed(4357)
    y = ROOT.RooRealVar("y", "", -5, 5)
    m = ROOT.RooRealVar("m", "", 0.5, -1, 2)
    s = ROOT.RooRealVar("s", "", 1, 0.5, 2)
    gs = ROOT.RooGaussian("gs", "", y, m, s)
    data = gs.generate(ROOT.RooArgSet(y), 10)
    return y, m, s, gs, data


def _at(ts: Any, m: Any, data: Any, value: float, kind: int = 0) -> float:
    m.setVal(value)
    return float(ts.EvaluateProfileLikelihood(kind, data, ROOT.RooArgSet(m)))


def test_the_profile_likelihood_ratio_two_sided_and_its_two_minima() -> None:
    """ROOT's ratio at three points, and the unconditional and conditional minima alone."""
    _y, m, _s, gs, data = _gauss()
    ts = ProfileLikelihoodTestStat(gs)
    found = [_at(ts, m, data, v) for v in (-0.5, 0.2, 1.0)]
    assert found == pytest.approx([7.3531024710172925, 1.6880574377519242, 2.157993061880589],
                                  rel=REL)  # fmt: skip
    kinds = (_at(ts, m, data, 0.2, 1), _at(ts, m, data, 0.2, 2))
    assert kinds == pytest.approx((8.821666171278956, 10.509723609023835), rel=REL)
    assert (ts.GetVarName(), ts.IsTwoSided(), ts.IsOneSidedDiscovery()) == (
        "Profile Likelihood Ratio", True, False)
    assert ts.GetPdf() is gs


@pytest.mark.parametrize(("discovery", "signed", "wanted"), [
    (False, False, [0.0, 2.157993061880589]),
    (False, True, [-7.3531024710172925, 2.157993061880589]),
    (True, False, [7.3531024710172925, 0.0]),
    (True, True, [7.3531024710172925, -2.157993061880589]),
])  # fmt: skip
def test_one_sided_statistics_are_roots_on_each_side_of_the_best_fit(
    discovery: bool, signed: bool, wanted: list[float]
) -> None:
    """A limit's statistic is nothing below the best fit, discovery's above; signed, negated."""
    _y, m, _s, gs, data = _gauss()
    ts = ProfileLikelihoodTestStat(gs)
    ts.SetOneSidedDiscovery(True) if discovery else ts.SetOneSided(True)
    ts.SetSigned(signed)
    assert [_at(ts, m, data, v) for v in (-0.5, 1.0)] == pytest.approx(wanted, rel=REL)
    assert ts.IsOneSidedDiscovery() == discovery and not ts.IsTwoSided()
    ts.SetOneSided(False)
    ts.SetOneSidedDiscovery(False)
    assert ts.IsTwoSided()


#: ROOT's detailed output at ``m`` = 1, with errors and pulls: both fits.
DETAILED = [
    ("fitUncond_m", "fitUncond_", 0.5703887558557946),
    ("fitUncond_m_pull", "fitUncond_m_pull", -2.3294824568673174),
    ("fitUncond_s", "fitUncond_", 0.5846819411562666),
    ("fitUncond_s_pull", "fitUncond_s_pull", -3.2535179432175148),
    ("fitUncond_minNLL", "fitUncond_minNLL", 8.821538694900177),
    ("fitUncond_fitStatus", "fitUncond_fitStatus", 0.0),
    ("fitUncond_covQual", "fitUncond_covQual", 3.0),
    ("fitUncond_numInvalidNLLEval", "fitUncond_numInvalidNLLEval", 0.0),
    ("fitCond_s", "fitCond_", 0.7254294459828824),
    ("fitCond_s_pull", "fitCond_s_pull", 0.8810687481132163),
    ("fitCond_minNLL", "fitCond_minNLL", 10.979531756780766),
    ("fitCond_fitStatus", "fitCond_fitStatus", 0.0),
    ("fitCond_covQual", "fitCond_covQual", 3.0),
    ("fitCond_numInvalidNLLEval", "fitCond_numInvalidNLLEval", 0.0),
]


def test_the_detailed_output_is_both_fits_with_their_pulls() -> None:
    """``EnableDetailedOutput(True, True)``: each fit's parameters, pulls, minimum and status."""
    _y, m, _s, gs, data = _gauss()
    ts = ProfileLikelihoodTestStat(gs)
    assert ts.GetDetailedOutput() is None
    ts.EnableDetailedOutput(True, True)
    m.setVal(1.0)
    assert ts.Evaluate(data, ROOT.RooArgSet(m)) == pytest.approx(2.157993061880589, rel=REL)
    found = [(v.GetName(), v.GetTitle(), v.getVal()) for v in ts.GetDetailedOutput()]
    assert [one[:2] for one in found] == [one[:2] for one in DETAILED]
    assert [one[2] for one in found] == pytest.approx([one[2] for one in DETAILED], rel=1e-5)
    ts.EnableDetailedOutput(False)
    assert ts.GetDetailedOutput() is None


def test_without_parameters_of_interest_or_free_parameters_there_is_little_to_fit() -> None:
    """No point: both fits the same, ROOT's rounding apart; nothing free: the value itself."""
    _y, m, s, gs, data = _gauss()
    ts = ProfileLikelihoodTestStat(gs)
    assert ts.Evaluate(data, ROOT.RooArgSet()) == pytest.approx(0.0, abs=1e-5)
    s.setConstant(True)
    m.setConstant(True)
    m.setVal(1.0)
    assert ts.Evaluate(data, ROOT.RooArgSet(m)) == 0.0


def test_a_negative_signed_ratio_is_zero_and_said_so_when_messages_are_shown(capfd: Any) -> None:
    """At print level 3 nothing is held back: the fits' rounding below zero is warned of."""
    _y, _m, _s, gs, data = _gauss()
    ts = ProfileLikelihoodTestStat(gs)
    ts.SetSigned(True)
    ts.SetPrintLevel(3)
    capfd.readouterr()
    assert ts.Evaluate(data, ROOT.RooArgSet()) == 0.0
    assert "[#0] WARNING:Eval -- pll is negative - setting it to zero \n" in capfd.readouterr().out


def test_a_likelihood_not_reused_is_made_again_for_each_data() -> None:
    """Neither reuse set: each evaluation makes its own likelihood, with the settings given."""
    _y, m, _s, gs, data = _gauss()
    ts = ProfileLikelihoodTestStat(gs)
    ProfileLikelihoodTestStat.SetAlwaysReuseNLL(False)
    ts.SetReuseNLL(False)
    ts.SetLOffset("initial")
    ts.SetLOffset(False)
    ts.SetMinimizer("Minuit2")
    ts.SetStrategy(1)
    ts.SetTolerance(1.0)
    ts.SetConditionalObservables(ROOT.RooArgSet())
    ts.SetGlobalObservables(ROOT.RooArgSet())
    ts.SetVarName("q")
    first = _at(ts, m, data, 1.0)
    made = ts._nll
    assert _at(ts, m, data, 1.0) == pytest.approx(first, rel=1e-9)
    assert ts._nll is not made and ts.GetVarName() == "q" and ts._offset == "none"
    ts.SetReuseNLL(True)
    ts.SetLOffset(True)
    made = ts._nll
    _at(ts, m, data, 1.0)
    assert ts._nll is made and ts._offset == "initial"


def test_a_signed_ratio_below_zero_is_zero_quietly_and_foreign_points_are_not_held() -> None:
    """ROOT's 0 for the signed ratio of no point; a point's variable the model lacks is left."""
    _y, m, _s, gs, data = _gauss()
    ts = ProfileLikelihoodTestStat(gs)
    ts.SetSigned(True)
    assert ts.Evaluate(data, ROOT.RooArgSet()) == 0.0
    ts.SetSigned(False)
    k = ROOT.RooRealVar("k", "", 3.0)
    m.setVal(1.0)
    both = ts.Evaluate(data, ROOT.RooArgSet(m, k))
    assert both == pytest.approx(2.157993061880589, rel=REL)


class _Failing(RooMinimizer):
    """A minimizer whose first fits report ``statuses``, and which says what it was asked."""

    statuses: ClassVar[list[int]] = []
    calls: ClassVar[list[tuple[str, str]]] = []

    def minimize(self, type: str = "", alg: str = "") -> int:
        _Failing.calls.append((type, alg))
        real = super().minimize(type, alg)
        self._last = _Failing.statuses.pop(0) if _Failing.statuses else real
        return self._last

    def save(self, name: Any = None, title: Any = None) -> Any:
        found = super().save(name, title)
        found._status = self._last
        return found


def _failing(monkeypatch: Any, where: Any, statuses: list[int]) -> list[tuple[str, str]]:
    monkeypatch.setattr(where, "RooMinimizer", _Failing)
    _Failing.statuses, _Failing.calls = list(statuses), []
    return _Failing.calls


@pytest.mark.parametrize(("strategy", "wanted"), [
    (0, [("", "Minimize"), ("", "Scan"), ("", "Minimize"), ("", "Scan"), ("", "Minimize"),
         ("", "Scan"), ("Minuit", "migradimproved")]),
    (1, [("", "Minimize"), ("", "Scan"), ("", "Minimize"), ("", "Scan"),
         ("Minuit", "migradimproved")]),
])  # fmt: skip
def test_a_failing_fit_is_tried_again_as_roostats_tries_it(
    monkeypatch: Any, strategy: int, wanted: list[tuple[str, str]]
) -> None:
    """A scan after each failure, strategy 1 on the second, the improved Migrad last - and a
    statistic whose fit never succeeds is ``-1``."""
    _y, m, _s, gs, data = _gauss()
    ts = ProfileLikelihoodTestStat(gs)
    ts.SetStrategy(strategy)
    calls = _failing(monkeypatch, teststats, [3] * 7)
    m.setVal(1.0)
    assert ts.Evaluate(data, ROOT.RooArgSet(m)) == -1.0
    assert calls[:len(wanted)] == wanted


def test_a_fit_that_fails_only_in_improve_is_a_success(monkeypatch: Any) -> None:
    """Status 1000 and up are Improve's, which RooStats ignores."""
    _y, _m, _s, gs, data = _gauss()
    calls = _failing(monkeypatch, teststats, [1000])
    nll = gs.createNLL(data)
    assert fit_nll(nll, 1, 1.0, 1).status() == 1000
    assert calls == [("", "Minimize")]


def test_the_base_statistic_asks_for_an_evaluation_and_says_little() -> None:
    """``TestStatistic``: no value of its own, no name, the right tail, no detail."""
    base = TestStatistic()
    with pytest.raises(NotImplementedError):
        base.Evaluate(None, None)
    base.SetConditionalObservables(None)
    base.SetGlobalObservables(None)
    assert (base.GetVarName(), base.PValueIsRightTail(), base.GetDetailedOutput()) == (
        "", True, None)


def _two_points(m: Any, s: Any) -> tuple[Any, Any]:
    """The null at ``m`` = 0.5, ``s`` = 1; the alternate at 1.2, 1.3."""
    null = ROOT.RooArgSet(m, s).snapshot()
    m.setVal(1.2)
    s.setVal(1.3)
    return null, ROOT.RooArgSet(m, s).snapshot()


def test_the_simple_likelihood_ratio_of_two_fixed_points_is_roots() -> None:
    """``-log L(null) + log L(alt)`` whatever point it is asked at, and both halves detailed."""
    _y, m, s, gs, data = _gauss()
    null, alt = _two_points(m, s)
    ts = SimpleLikelihoodRatioTestStat(gs, gs, null, alt)
    ts.EnableDetailedOutput(True)
    assert ts.GetDetailedOutput() is None
    m.setVal(0.0)
    assert ts.Evaluate(data, ROOT.RooArgSet(m)) == pytest.approx(-3.056127486953663, rel=1e-12)
    detail = [(v.GetName(), v.GetTitle(), v.getVal()) for v in ts.GetDetailedOutput()]
    assert detail == [("nullNLL", "null NLL", pytest.approx(10.923136154595287, rel=1e-12)),
                      ("altNLL", "alternate NLL", pytest.approx(13.97926364154895, rel=1e-12))]
    m.setVal(0.3)
    assert ts.Evaluate(data, ROOT.RooArgSet(m)) == pytest.approx(-3.056127486953663, rel=1e-12)
    assert ts.GetVarName() == "log(L(#mu_{1}) / L(#mu_{0}))"


def test_the_same_parameters_for_both_hypotheses_are_warned_of(capfd: Any) -> None:
    """Without points, both are the model's variables: equal, said, and a ratio of nothing."""
    _y, m, _s, gs, data = _gauss()
    ts = SimpleLikelihoodRatioTestStat(gs, gs)
    assert ts.ParamsAreEqual()
    capfd.readouterr()
    assert ts.Evaluate(data, ROOT.RooArgSet(m)) == 0.0
    assert capfd.readouterr().out == (
        "[#0] WARNING:InputArguments -- Same RooArgSet used for null and alternate, so you must "
        "explicitly SetNullParameters and SetAlternateParameters or the likelihood ratio will "
        "always be 1.\n"
    )


def test_a_simple_ratio_set_up_afterwards_and_not_reusing_its_likelihoods() -> None:
    """Points set after construction; unlike names are unequal; each data a new likelihood."""
    _y, m, s, gs, data = _gauss()
    null, alt = _two_points(m, s)
    ts = SimpleLikelihoodRatioTestStat(gs, gs, null, alt)
    ts.SetNullParameters(ROOT.RooArgSet(m))
    assert not ts.ParamsAreEqual()
    ts.SetNullParameters(null)
    ts.SetAltParameters(alt)
    SimpleLikelihoodRatioTestStat.SetAlwaysReuseNLL(False)
    ts.SetReuseNLL(False)
    ts.SetConditionalObservables(ROOT.RooArgSet())
    ts.SetGlobalObservables(ROOT.RooArgSet())
    assert ts.Evaluate(data, ROOT.RooArgSet(m)) == pytest.approx(-3.056127486953663, rel=1e-12)
    assert (ts._nll_null, ts._nll_alt) == (None, None)
    bare = SimpleLikelihoodRatioTestStat()
    assert (bare._null, bare._alt) == (None, None)


#: ROOT's detailed output of the ratio of profiles at ``m`` = 0.2, ``m2`` = 1: names and values.
PROFILES = [
    ("nullprof_fitUncond_m", 0.5706948070662136), ("nullprof_fitUncond_s", 0.582545952166153),
    ("nullprof_fitUncond_minNLL", 8.821666171278956), ("nullprof_fitUncond_fitStatus", 0.0),
    ("nullprof_fitUncond_covQual", 3.0), ("nullprof_fitUncond_numInvalidNLLEval", 0.0),
    ("nullprof_fitCond_s", 0.692139757130555), ("nullprof_fitCond_minNLL", 10.50972360903088),
    ("nullprof_fitCond_fitStatus", 0.0), ("nullprof_fitCond_covQual", 3.0),
    ("nullprof_fitCond_numInvalidNLLEval", 0.0),
    ("altprof_fitUncond_m2", 0.5703887558557946), ("altprof_fitUncond_s", 0.5846819411562666),
    ("altprof_fitUncond_minNLL", 8.821538694900177), ("altprof_fitUncond_fitStatus", 0.0),
    ("altprof_fitUncond_covQual", 3.0), ("altprof_fitUncond_numInvalidNLLEval", 0.0),
    ("altprof_fitCond_s", 0.7254294459828824), ("altprof_fitCond_minNLL", 10.979531756780766),
    ("altprof_fitCond_fitStatus", 0.0), ("altprof_fitCond_covQual", 3.0),
    ("altprof_fitCond_numInvalidNLLEval", 0.0),
]


def _profiles() -> tuple[Any, ...]:
    """The null ``Gauss(y; m, s)`` and the alternate ``Gauss(y; m2, s)`` at ``m2`` = 1."""
    y, m, s, gs, data = _gauss()
    m2 = ROOT.RooRealVar("m2", "", 1.0, -1, 2)
    gs2 = ROOT.RooGaussian("gs2", "", y, m2, s)
    return y, m, s, gs, data, gs2, RatioOfProfiledLikelihoodsTestStat(gs, gs2, ROOT.RooArgSet(m2))


def test_the_ratio_of_two_profiles_and_its_detail_are_roots() -> None:
    """The null's profile ratio less the alternate's, each fit of both kept, titled as ROOT's."""
    _y, m, _s, _gs, data, _gs2, ts = _profiles()
    ts.EnableDetailedOutput(True)
    m.setVal(0.2)
    assert ts.Evaluate(data, ROOT.RooArgSet(m)) == pytest.approx(-0.46993562412866474, rel=1e-5)
    found = [(v.GetName(), v.getVal()) for v in ts.GetDetailedOutput()]
    assert [one[0] for one in found] == [one[0] for one in PROFILES]
    assert [one[1] for one in found] == pytest.approx([one[1] for one in PROFILES], rel=1e-5)
    assert ts.GetDetailedOutput().find("altprof_fitCond_s").GetTitle() == "fitCond_ for null"
    assert ts.GetVarName() == "log(L(#mu_{1},#hat{#nu}_{1}) / L(#mu_{0},#hat{#nu}_{0}))"
    ts.EnableDetailedOutput(False)
    m.setVal(0.2)
    ts.Evaluate(data, ROOT.RooArgSet(m))
    assert ts.GetDetailedOutput() is None


def test_without_subtracting_the_maxima_the_ratio_is_of_the_conditional_minima(capfd: Any) -> None:
    """``SetSubtractMLE(False)``: ROOT's difference, each model's minimum - and NaN for another."""
    y, m, s, gs, data, gs2, ts = _profiles()
    ts.SetSubtractMLE(False)
    for setter, value in (("SetReuseNLL", True), ("SetMinimizer", ""), ("SetStrategy", 1),
                          ("SetTolerance", 1.0), ("SetPrintLevel", 0),
                          ("SetConditionalObservables", ROOT.RooArgSet()),
                          ("SetGlobalObservables", ROOT.RooArgSet())):  # fmt: skip
        getattr(ts, setter)(value)
    RatioOfProfiledLikelihoodsTestStat.SetAlwaysReuseNLL(True)
    m.setVal(0.2)
    assert ts.Evaluate(data, ROOT.RooArgSet(m)) == pytest.approx(-0.469808147461249, rel=1e-5)
    m.setVal(0.2)
    assert ts.ProfiledLikelihood(data, ROOT.RooArgSet(m), gs) == pytest.approx(
        10.509723609023835, rel=1e-6)
    m2 = gs2.getVariables().find("m2")
    assert ts.ProfiledLikelihood(data, ROOT.RooArgSet(m2), gs2) == pytest.approx(
        10.979531756780766, rel=1e-6)
    capfd.readouterr()
    other = ROOT.RooGaussian("x", "", y, m, s)
    nan = ts.ProfiledLikelihood(data, ROOT.RooArgSet(m), other)
    assert nan != nan
    assert capfd.readouterr().out == (
        "[#0] ERROR:InputArguments -- RatioOfProfiledLikelihoods::ProfileLikelihood - invalid "
        "pdf used for computing the profiled likelihood - return NaN\n"
    )
    assert len(RatioOfProfiledLikelihoodsTestStat(gs, gs2)._alt_poi) == 0


def test_the_number_of_events_is_counted_as_the_data_and_model_have_it(capfd: Any) -> None:
    """Weighted: the weights; extended or no model: the entries; counting: the one event's sum."""
    y, m, _s, gs, data = _gauss()
    y.setBins(5)
    binned = ROOT.RooDataHist("binned", "", ROOT.RooArgSet(y), data)
    n = ROOT.RooRealVar("n", "", 10)
    extended = ROOT.RooExtendPdf("ext", "", gs, n)
    ts = NumEventsTestStat(gs)
    assert (ts.GetTestStatistic(), ts.GetVarName()) == (gs, "Number of events")
    assert ts.Evaluate(binned) == 10.0
    assert NumEventsTestStat().Evaluate(data) == 10.0
    assert NumEventsTestStat(extended).Evaluate(data) == 10.0
    x = ROOT.RooRealVar("x", "", 7, 0, 50)
    count = ROOT.RooDataSet("count", "", ROOT.RooArgSet(x))
    count.add(ROOT.RooArgSet(x))
    assert NumEventsTestStat(ROOT.RooPoisson("p", "", x, m)).Evaluate(count) == 7.0
    capfd.readouterr()
    assert ts.Evaluate(data, ROOT.RooArgSet(m)) == 0.0
    assert capfd.readouterr().out == "Data set is invalid\n"


def test_the_maximum_likelihood_estimate_is_roots_and_its_tail_is_settable() -> None:
    """ROOT's fitted ``m``; the p-value's tail the right one unless it is said otherwise."""
    _y, m, _s, gs, data = _gauss()
    ts = MaxLikelihoodEstimateTestStat(gs, m)
    ts.SetConditionalObservables(ROOT.RooArgSet())
    assert ts.Evaluate(data, ROOT.RooArgSet(m)) == pytest.approx(0.5705992229984852, rel=1e-9)
    assert ts.GetVarName() == "Maximum Likelihood Estimate of m"
    assert ts.PValueIsRightTail() is True
    assert ts.PValueIsRightTail(False) is None
    assert ts.PValueIsRightTail() is False


def test_a_maximum_likelihood_fit_that_keeps_failing_is_minus_one(
    monkeypatch: Any, capfd: Any
) -> None:
    """Five tries, rescans from the third, strategy 1 from the fourth - said as ROOT says them."""
    import xrdroot.roofit.fitting.minimizer as minimizer

    _y, m, _s, gs, data = _gauss()
    calls = _failing(monkeypatch, minimizer, [3] * 8)
    capfd.readouterr()
    assert MaxLikelihoodEstimateTestStat(gs, m).Evaluate(data) == -1.0
    rescan, strategy = "    ----> Doing a re-scan first\n", "    ----> trying with strategy = 1\n"
    assert capfd.readouterr().out == rescan + (rescan + strategy) * 2
    assert [alg for kind, alg in calls] == ["Minimize"] * 3 + ["Scan", "Minimize"] * 2 + ["Scan"]
    assert moretests.MaxLikelihoodEstimateTestStat is MaxLikelihoodEstimateTestStat
