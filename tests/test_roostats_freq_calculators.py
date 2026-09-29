"""``HybridCalculator`` and ``FrequentistCalculator``: toy hypothesis tests as ROOT 6.40 runs them.

HybridStandardForm's counting model - a flat observable, ``s + b`` events,
``b`` drawn from its Poisson prior for each toy - and a frequentist version
with a Gaussian constraint and a global observable were run in ROOT through
PyROOT after ``RooRandom::randomGenerator()->SetSeed(4357)``, with a few
toys each: the test statistic of the data, every toy's statistic, the fit
information and the printed result are ROOT's. The calculators' refusals
and hooks are driven directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.roostats.calculators import (
    FrequentistCalculator,
    HybridCalculator,
    HypoTestCalculatorGeneric,
)
from xrdroot.roostats.moretests import NumEventsTestStat


@pytest.fixture(autouse=True)
def _fresh(tmp_path: Any) -> Iterator[None]:
    yield from fresh(tmp_path)


def _hybrid_models(events: int = 115) -> tuple[Any, Any, Any, Any]:
    """HybridStandardForm's workspace, data of ``events`` events, the B and S+B models."""
    ROOT.RooRandom.randomGenerator().SetSeed(4357)
    w = ROOT.RooWorkspace("w")
    w.factory("Uniform::f(m[0,1])")
    w.factory("ExtendPdf::px(f,sum::splusb(s[0,0,100],b[100,0.1,300]))")
    w.factory("Poisson::py(y[100,0.1,500],prod::taub(tau[1.],b))")
    w.factory("PROD::model(px,py)")
    w.defineSet("obs", "m")
    w.defineSet("poi", "s")
    data = w.pdf("px").generate(w.set("obs"), events)
    models = []
    for name, value in (("B_model", 0.0), ("S+B_model", 50.0)):
        mc = ROOT.RooStats.ModelConfig(name, w)
        mc.SetPdf(w.pdf("px"))
        mc.SetObservables(w.set("obs"))
        mc.SetParametersOfInterest(w.set("poi"))
        w.var("s").setVal(value)
        mc.SetSnapshot(w.set("poi"))
        models.append(mc)
    return w, data, models[0], models[1]


#: ROOT's printed result of 40 null and 20 alternate toys.
HYBRID = (
    "\nResults HypoTestCalculator_result: \n"
    " - Null p-value = 0.15 +/- 0.0564579\n"
    " - Significance = 1.03643 +/- 0.242144 sigma\n"
    " - Number of Alt toys: 20\n"
    " - Number of Null toys: 40\n"
    " - Test statistic evaluated on data: 115\n"
    " - CL_b: 0.15 +/- 0.0564579\n"
    " - CL_s+b: 1 +/- 0\n"
    " - CL_s: 6.66667 +/- 2.50924\n"
)


def test_a_hybrid_test_of_the_number_of_events_is_roots(capfd: Any) -> None:
    """``b`` drawn from ``py`` for each toy: ROOT's toys, p-values and printout."""
    w, data, b_model, sb_model = _hybrid_models()
    hc = HybridCalculator(data, sb_model, b_model)
    hc.GetTestStatSampler().SetTestStatistic(NumEventsTestStat(w.pdf("px")))
    hc.SetToys(40, 20)
    hc.ForcePriorNuisanceAlt(w.pdf("py"))
    hc.ForcePriorNuisanceNull(w.pdf("py"))
    capfd.readouterr()
    result = hc.GetHypoTest()
    out = capfd.readouterr().out
    assert out.endswith(
        "[#0] PROGRESS:Generation -- Test Statistic on data: 115\n"
        "[#1] INFO:InputArguments -- Using a ToyMCSampler. Now configuring for Null.\n"
        "[#1] INFO:InputArguments -- Using randomized nuisance parameters.\n"
        "[#1] INFO:InputArguments -- Using a ToyMCSampler. Now configuring for Alt.\n"
        "[#1] INFO:InputArguments -- Using randomized nuisance parameters.\n"
    )
    null = list(result.GetNullDistribution().GetSamplingDistribution())
    alt = list(result.GetAltDistribution().GetSamplingDistribution())
    assert null[:10] == [76.0, 76.0, 77.0, 79.0, 81.0, 82.0, 84.0, 84.0, 86.0, 86.0]
    assert alt[:10] == [121.0, 133.0, 136.0, 139.0, 142.0, 148.0, 149.0, 151.0, 155.0, 156.0]
    result.Print()
    assert capfd.readouterr().out == HYBRID
    assert result.CLsError() == pytest.approx(2.509242175696937, rel=1e-12)


def _frequentist_models() -> tuple[Any, Any, Any, Any]:
    """115 flat events of ``s + b``, ``b`` constrained by ``Gauss(b0; b, 10)``, ``b0`` global."""
    ROOT.RooRandom.randomGenerator().SetSeed(4357)
    w = ROOT.RooWorkspace("w")
    w.factory("Uniform::f(m[0,1])")
    w.factory("ExtendPdf::px(f,sum::splusb(s[0,0,100],b[100,0.1,300]))")
    w.factory("Gaussian::cons(b0[100,0,300],b,10)")
    w.factory("PROD::model(px,cons)")
    for name, members in (("obs", "m"), ("poi", "s"), ("glob", "b0"), ("nuis", "b")):
        w.defineSet(name, members)
    data = w.pdf("px").generate(w.set("obs"), 115)
    models = []
    for name, value in (("B_model", 0.0), ("S+B_model", 50.0)):
        mc = ROOT.RooStats.ModelConfig(name, w)
        mc.SetPdf(w.pdf("model"))
        mc.SetObservables(w.set("obs"))
        mc.SetParametersOfInterest(w.set("poi"))
        mc.SetNuisanceParameters(w.set("nuis"))
        mc.SetGlobalObservables(w.set("glob"))
        w.var("s").setVal(value)
        mc.SetSnapshot(w.set("poi"))
        models.append(mc)
    return w, data, models[0], models[1]


#: ROOT's fits of ``b`` to the data with ``s`` held at each hypothesis.
FIT_INFO = {
    "fitNull_b": 107.16166759361596, "fitNull_s": 0.0, "fitNull_minNLL": -426.9093010879194,
    "fitNull_fitStatus": 0.0, "fitAlt_b": 85.12248420275901, "fitAlt_s": 50.0,
    "fitAlt_minNLL": -424.7601804009173, "fitAlt_fitStatus": 0.0,
}


def test_a_frequentist_test_of_the_ratio_of_profiles_is_roots(capfd: Any) -> None:
    """The default statistic, ``b`` at each hypothesis' best fit, the global observable drawn."""
    _w, data, b_model, sb_model = _frequentist_models()
    fc = FrequentistCalculator(data, sb_model, b_model)
    fc.SetToys(10, 5)
    fc.StoreFitInfo(True)
    capfd.readouterr()
    result = fc.GetHypoTest()
    assert capfd.readouterr().out.endswith(
        "[#0] PROGRESS:Generation -- Test Statistic on data: -2.14867\n"
        "[#1] INFO:InputArguments -- Profiling conditional MLEs for Null.\n"
        "[#1] INFO:InputArguments -- Using a ToyMCSampler. Now configuring for Null.\n"
        "[#1] INFO:InputArguments -- Profiling conditional MLEs for Alt.\n"
        "[#1] INFO:InputArguments -- Using a ToyMCSampler. Now configuring for Alt.\n"
    )
    assert (result.NullPValue(), result.AlternatePValue()) == (0.0, 1.0)
    assert result.GetTestStatisticData() == pytest.approx(-2.1486674045054315, rel=1e-6)
    null = list(result.GetNullDistribution().GetSamplingDistribution())
    alt = list(result.GetAltDistribution().GetSamplingDistribution())
    assert null == pytest.approx([
        -7.863325102218255, -7.518183793600599, -7.390069085951211, -7.344074023961525,
        -7.006459012860148, -6.342182727680836, -4.443533375298898, -3.3788506966474188,
        -2.554274428172903, -2.2505746180401616], rel=1e-5)  # fmt: skip
    assert alt == pytest.approx([3.46019328537011, 4.359209788947624, 6.10044661033146,
                                 8.274418355717444, 8.279003236504309], rel=1e-5)  # fmt: skip
    info = {v.GetName(): v.getVal() for v in fc.GetFitInfo()}
    assert {name: info[name] for name in FIT_INFO} == pytest.approx(FIT_INFO, rel=1e-6)


def test_given_conditional_estimates_and_adaptive_tails(capfd: Any) -> None:
    """The null's ``b`` given, the alternate's profiled; toys in each tail asked for."""
    w, data, b_model, sb_model = _frequentist_models()
    statistic = ROOT.RooStats.ProfileLikelihoodTestStat(sb_model.GetPdf())
    statistic.SetOneSided(True)
    fc = FrequentistCalculator(data, sb_model, b_model, ROOT.RooStats.ToyMCSampler(statistic, 10))
    w.var("b").setVal(110)
    fc.SetConditionalMLEsNull(ROOT.RooArgSet(w.var("b")))
    fc.SetConditionalMLEsAlt(None)
    fc.SetToys(6, 4)
    fc.SetNToysInTails(2, 2)
    capfd.readouterr()
    result = fc.GetHypoTest()
    assert capfd.readouterr().out.endswith(
        "[#0] PROGRESS:Generation -- Test Statistic on data: 0\n"
        "[#1] INFO:InputArguments -- Using given conditional MLEs for Null.\n"
        "[#1] INFO:InputArguments -- Using a ToyMCSampler. Now configuring for Null.\n"
        "[#1] INFO:InputArguments -- Adaptive Sampling\n"
        "[#1] INFO:InputArguments -- Profiling conditional MLEs for Alt.\n"
        "[#1] INFO:InputArguments -- Using a ToyMCSampler. Now configuring for Alt.\n"
        "[#1] INFO:InputArguments -- Adaptive Sampling\n"
    )
    assert (result.NullPValue(), result.AlternatePValue()) == (1.0, 1.0)
    assert list(result.GetNullDistribution().GetSamplingDistribution()) == [0.0] * 6
    assert list(result.GetAltDistribution().GetSamplingDistribution()) == [0.0] * 4
    assert fc.GetFitInfo() is None


def test_the_default_sampler_samples_the_ratio_of_profiles_binned_for_binned_data() -> None:
    """A thousand toys of ``RatioOfProfiledLikelihoods``; binned data is the toys' prototype."""
    w, data, b_model, sb_model = _hybrid_models(20)
    w.var("m").setBins(4)
    binned = ROOT.RooDataHist("binned", "", w.set("obs"), data)
    calc = HypoTestCalculatorGeneric(binned, sb_model, b_model)
    toys = calc.GetTestStatSampler()
    assert (toys.GetNToys(), toys._proto, toys._binned) == (1000, binned, True)
    assert toys.GetTestStatistic().GetVarName().startswith("log(L(#mu_{1},#hat{#nu}_{1})")
    unbinned = HypoTestCalculatorGeneric(data, sb_model, b_model).GetTestStatSampler()
    assert (unbinned._proto, unbinned._binned) == (None, False)
    calc.SetData(data)
    calc.SetNullModel(sb_model)
    calc.SetAlternateModel(b_model)
    assert (calc.GetData(), calc.GetNullModel(), calc.GetAlternateModel()) == (
        data, sb_model, b_model)
    assert calc.GetFitInfo() is None


def test_a_null_model_without_a_snapshot_or_a_sampler_without_a_statistic_is_refused(
    capfd: Any,
) -> None:
    """Each said as ROOT says it, and no result."""
    w, data, b_model, sb_model = _hybrid_models(20)
    bare = ROOT.RooStats.ModelConfig("bare", w)
    bare.SetPdf(w.pdf("px"))
    bare.SetParametersOfInterest(w.set("poi"))
    capfd.readouterr()
    assert FrequentistCalculator(data, sb_model, bare).GetHypoTest() is None
    assert capfd.readouterr().out.endswith(
        "[#0] ERROR:Generation -- Null model needs a snapshot. Set using "
        "modelconfig->SetSnapshot(poi).\n")  # fmt: skip
    empty = ROOT.RooStats.ToyMCSampler()
    assert FrequentistCalculator(data, sb_model, b_model, empty).GetHypoTest() is None
    assert capfd.readouterr().out.endswith(
        "[#0] ERROR:InputArguments -- Test Statistic Sampler or Test Statistics not defined. "
        "Stop.\n")  # fmt: skip


@pytest.mark.parametrize(("which", "end"), [("Null", "."), ("Alt", "")])
def test_a_hybrid_prior_without_nuisance_parameters_is_refused(
    capfd: Any, which: str, end: str
) -> None:
    """A prior forced on a model that names no nuisance parameters stops the test."""
    w, data, b_model, sb_model = _hybrid_models(20)
    w.var("b").setConstant(True)
    hc = HybridCalculator(data, sb_model, b_model)
    getattr(hc, f"ForcePriorNuisance{which}")(w.pdf("py"))
    capfd.readouterr()
    assert hc.GetHypoTest() is None
    assert capfd.readouterr().out.endswith(
        "[#0] ERROR:InputArguments -- HybridCalculator - Nuisance PDF has been specified, but is "
        "unaware of which parameters are the nuisance parameters. Must set nuisance parameters "
        f"in the {which} ModelConfig{end}\n"
        "[#0] ERROR:Generation -- There was an error in CheckHook(). Stop.\n"
    )


def _counted(hc: Any, w: Any, null: int = 3, alt: int = 2) -> Any:
    hc.GetTestStatSampler().SetTestStatistic(NumEventsTestStat(w.pdf("px")))
    hc.SetToys(null, alt)
    return hc


def test_a_hybrid_test_without_priors_says_how_it_treats_the_nuisance_parameters(
    capfd: Any,
) -> None:
    """Nuisance parameters and no prior: a uniform one; none at all: a simple test."""
    w, data, b_model, sb_model = _hybrid_models(20)
    hc = _counted(HybridCalculator(data, sb_model, b_model), w)
    hc.SetNullModel(b_model)
    hc.SetAlternateModel(sb_model)
    capfd.readouterr()
    hc.GetHypoTest()
    out = capfd.readouterr().out
    for which in ("Null", "Alt"):
        assert (f"[#1] INFO:InputArguments -- HybridCalculator - Using uniform prior on nuisance "
                f"parameters ({which} model).\n") in out  # fmt: skip
    w2, data2, b2, sb2 = _hybrid_models(20)
    w2.var("b").setConstant(True)
    capfd.readouterr()
    _counted(HybridCalculator(data2, sb2, b2), w2).GetHypoTest()
    out = capfd.readouterr().out
    for which in ("Null", "Alt"):
        assert (f"[#1] INFO:InputArguments -- HybridCalculator - No nuisance parameters specified "
                f"for {which} model and no prior forced. Case is reduced to simple hypothesis "
                "testing with no uncertainty.\n") in out  # fmt: skip


def test_the_same_alternate_toys_each_time_when_asked() -> None:
    """``UseSameAltToys``: the alternate's toys are drawn from one seed every time."""
    w, data, b_model, sb_model = _hybrid_models(20)
    hc = _counted(HybridCalculator(data, sb_model, b_model), w, 2, 4)
    hc.UseSameAltToys()
    first = list(hc.GetHypoTest().GetAltDistribution().GetSamplingDistribution())
    second = list(hc.GetHypoTest().GetAltDistribution().GetSamplingDistribution())
    assert first == second and len(first) == 4


def test_every_statistic_of_the_sampler_is_evaluated_on_the_data() -> None:
    """A second statistic: all the data's values kept in the result, the first its value."""
    w, data, b_model, sb_model = _hybrid_models(20)
    hc = _counted(HybridCalculator(data, sb_model, b_model), w)
    hc.GetTestStatSampler().AddTestStatistic(NumEventsTestStat())
    result = hc.GetHypoTest()
    values = [v.getVal() for v in result.GetAllTestStatisticsData()]
    assert values == [20.0, 20.0] and result.GetTestStatisticData() == 20.0


class _Sampler:
    """A sampler that is not a ``ToyMCSampler`` and samples nothing."""

    def __init__(self, statistic: Any) -> None:
        self._statistic = statistic
        self._events = 0

    def __getattr__(self, name: str) -> Any:
        return lambda *args: None

    def GetTestStatistic(self) -> Any:
        return self._statistic

    def nEventsPerToy(self) -> int:
        return self._events

    def SetNEventsPerToy(self, events: int) -> None:
        self._events = int(events)

    def EvaluateAllTestStatistics(self, data: Any, poi: Any) -> Any:
        return [ROOT.RooRealVar("ts", "", self._statistic.Evaluate(data, poi))]


class _Refusing(HypoTestCalculatorGeneric):
    """A calculator whose hooks say they failed."""

    def _pre_null(self, point: Any, observed: float) -> int:
        return 1

    def _pre_alt(self, point: Any, observed: float) -> int:
        return 2


def test_hooks_that_fail_are_said_and_a_sampler_that_samples_nothing_gives_no_distributions(
    capfd: Any,
) -> None:
    """Each hook's failure named; neither distribution; the data's statistic still kept."""
    _w, data, b_model, sb_model = _hybrid_models(20)
    sampler = _Sampler(NumEventsTestStat())
    capfd.readouterr()
    result = _Refusing(data, sb_model, b_model, sampler).GetHypoTest()
    out = capfd.readouterr().out
    assert "[#0] ERROR:Generation -- PreNullHook did not return 0.\n" in out
    assert "[#0] ERROR:Generation -- PreAltHook did not return 0.\n" in out
    assert (result.GetNullDistribution(), result.GetAltDistribution()) == (None, None)
    assert result.GetTestStatisticData() == 20.0
    assert sampler.nEventsPerToy() == 0
    hybrid = HybridCalculator(data, sb_model, b_model, _Sampler(NumEventsTestStat()))
    assert hybrid.GetHypoTest().GetNullDistribution() is None


def test_the_generic_calculators_own_hooks_do_nothing() -> None:
    """``HypoTestCalculatorGeneric`` itself: no checks, no hooks, just the sampler's toys."""
    _w, data, b_model, sb_model = _hybrid_models(20)
    sampler = _Sampler(NumEventsTestStat())
    result = HypoTestCalculatorGeneric(data, sb_model, b_model, sampler).GetHypoTest()
    assert result.GetTestStatisticData() == 20.0 and result.GetNullDistribution() is None


def test_a_counting_model_without_nuisance_parameters_gets_toys_of_the_datas_size(
    capfd: Any,
) -> None:
    """Not extended: each toy as many events as the data; nothing to profile; toys as set."""
    ROOT.RooRandom.randomGenerator().SetSeed(4357)
    w = ROOT.RooWorkspace("w")
    w.factory("Poisson::pois(x[7,0,50],sum::mean(s[0,0,20],b[3]))")
    x = w.var("x")
    data = ROOT.RooDataSet("data", "", ROOT.RooArgSet(x))
    data.add(ROOT.RooArgSet(x))
    models = []
    for name, value in (("B", 0.0), ("SB", 5.0)):
        mc = ROOT.RooStats.ModelConfig(name, w)
        mc.SetPdf(w.pdf("pois"))
        mc.SetParametersOfInterest(ROOT.RooArgSet(w.var("s")))
        w.var("s").setVal(value)
        mc.SetSnapshot(ROOT.RooArgSet(w.var("s")))
        models.append(mc)
    toys = ROOT.RooStats.ToyMCSampler(NumEventsTestStat(w.pdf("pois")), 3)
    capfd.readouterr()
    result = FrequentistCalculator(data, models[1], models[0], toys).GetHypoTest()
    assert "Profiling" not in capfd.readouterr().out
    assert toys.nEventsPerToy() == 1
    assert result.GetNullDistribution().GetSize() == 3
    assert result.GetAltDistribution().GetSize() == 3


def test_estimates_given_for_other_parameters_still_profile_the_nuisance_parameters(
    capfd: Any,
) -> None:
    """Given MLEs that miss ``b``: said, and ``b`` profiled all the same."""
    w, data, b_model, sb_model = _frequentist_models()
    toys = ROOT.RooStats.ToyMCSampler(NumEventsTestStat(w.pdf("model")), 2)
    fc = FrequentistCalculator(data, sb_model, b_model, toys)
    fc.SetConditionalMLEsAlt(ROOT.RooArgSet(w.var("s")))
    capfd.readouterr()
    fc.GetHypoTest()
    out = capfd.readouterr().out
    assert ("[#1] INFO:InputArguments -- Using given conditional MLEs for Alt.\n"
            "[#1] INFO:InputArguments -- Profiling conditional MLEs for Alt.\n") in out
