"""RooStats' ProfileLikelihoodCalculator and the LikelihoodInterval it hands back.

A Gaussian of ten fixed values, its mean the parameter of interest and its
width a nuisance: the interval's ends, the point left where MINOS leaves it,
the contour of the two, the p-values of the null hypotheses, and the
messages for a fit that cannot be done or a minimum that is not valid, as
ROOT 6.40 gave them through PyROOT for the same calls.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterator
from typing import Any, ClassVar

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.fit import defaults
from xrdroot.roofit.messages import service
from xrdroot.roostats import combined
from xrdroot.roostats.combined import CombinedCalculator, ProfileLikelihoodCalculator

VALUES = [0.3, 1.2, -0.4, 2.1, 0.8, 1.5, 0.1, 1.9, 0.6, 1.1]


@pytest.fixture(autouse=True)
def _messages() -> Iterator[None]:
    """Every test starts from RooFit's message streams and Minuit's defaults as they start."""
    saved = dict(defaults.DEFAULTS)
    service().reset()
    yield
    service().reset()
    defaults.DEFAULTS.clear()
    defaults.DEFAULTS.update(saved)


def gaussian(spec: str = "Gaussian::g(x[-10,10],mu[0,-5,5],s[1,0.1,5])") -> tuple[Any, Any]:
    """The workspace of ``spec`` and the ten values of ``x``."""
    w = ROOT.RooWorkspace("w")
    w.factory(spec)
    x = w.var("x")
    data = ROOT.RooDataSet("d", "d", ROOT.RooArgSet(x))
    for value in VALUES:
        x.setVal(value)
        data.add(ROOT.RooArgSet(x))
    return w, data


def config(w: Any, poi: str = "mu") -> Any:
    """A ModelConfig of ``g`` over ``x``, its snapshot the null ``mu = 0``."""
    mc = ROOT.RooStats.ModelConfig("mc", w)
    mc.SetPdf("g")
    mc.SetParametersOfInterest(poi)
    mc.SetObservables("x")
    if poi == "mu":
        mc.SetNuisanceParameters("s")
        null = ROOT.RooArgSet(w.var("mu")).snapshot()
        null.setRealValue("mu", 0.0)
        mc.SetSnapshot(null)
    return mc


def test_the_interval_of_the_mean_is_minos_on_the_profile(capsys: Any) -> None:
    """ROOT 6.40: [0.674806296452188, 1.1651937035400877] at 68.27%, [0.4025888, 1.4374112]
    at 95%, with ``mu`` and ``s`` left where ROOT's MINOS leaves them."""
    w, data = gaussian()
    mu, s = w.var("mu"), w.var("s")
    plc = ROOT.RooStats.ProfileLikelihoodCalculator(data, config(w), 0.3173)
    assert (plc.Size(), plc.ConfidenceLevel()) == (0.3173, pytest.approx(0.6827))
    interval = plc.GetInterval()
    assert (interval.GetName(), interval.ClassName()) == (
        "LikelihoodInterval_", "RooStats::LikelihoodInterval")  # fmt: skip
    assert interval.ConfidenceLevel() == pytest.approx(0.6827)
    best = interval.GetBestFitParameters().find("mu")
    assert (best.getVal(), best.getError()) == pytest.approx(
        (0.9198614877074255, 0.23900079790202644), rel=1e-9)  # fmt: skip
    assert interval.GetLikelihoodRatio().getVariables().find("mu") is mu
    assert interval.LowerLimit(mu) == pytest.approx(0.674806296452188, rel=1e-9)
    assert interval.UpperLimit(mu) == pytest.approx(1.1651937035400877, rel=1e-9)
    assert (mu.getVal(), s.getVal()) == pytest.approx(
        (1.1648941606047998, 0.7942849999333178), rel=1e-6)  # fmt: skip


def test_the_interval_at_another_level_is_found_again(capsys: Any) -> None:
    """At 95%, the limits found again - ``[0.4025888, 1.4374112]`` - and what lies within."""
    w, data = gaussian()
    mu, s = w.var("mu"), w.var("s")
    interval = ROOT.RooStats.ProfileLikelihoodCalculator(data, config(w), 0.3173).GetInterval()
    interval.SetConfidenceLevel(0.95)
    lower, upper = ctypes.c_double(0), ctypes.c_double(0)
    assert interval.FindLimits(mu, lower, upper)
    assert (lower.value, upper.value) == pytest.approx(
        (0.402588822071688, 1.437411177722059), rel=1e-9)  # fmt: skip
    assert interval.FindLimits(mu)
    for value in (0.2, 1.5):
        mu.setVal(value)
        assert not interval.IsInInterval(ROOT.RooArgSet(mu))
    mu.setVal(1.0)
    assert interval.IsInInterval(ROOT.RooArgSet(mu))
    assert interval.LowerLimit(s) == pytest.approx(0.5157138093813148, rel=1e-9)


def test_the_null_of_no_mean_is_rejected_at_three_sigma(capsys: Any) -> None:
    """ROOT 6.40: p 0.0012881267498952272, Z 3.0142383167614115, the global fit's -log L
    11.3928, ``mu`` put back where the global fit left it, and what it says as it goes."""
    w, data = gaussian()
    plc = ProfileLikelihoodCalculator(data, config(w))
    result = plc.GetHypoTest()
    assert result.GetName() == "ProfileLRHypoTestResult_"
    assert result.NullPValue() == pytest.approx(0.0012881267498952272, rel=1e-9)
    assert result.Significance() == pytest.approx(3.0142383167614115, rel=1e-9)
    assert plc.GetFitResult().minNll() == pytest.approx(11.3928, abs=5e-5)  # as ROOT prints it
    assert not w.var("mu").isConstant()
    assert w.var("mu").getVal() == pytest.approx(0.9198614877074255, rel=1e-9)
    said = [line for line in capsys.readouterr().out.splitlines() if "PROGRESS" in line]
    assert said == [
        "[#0] PROGRESS:Minimization -- ProfileLikelihoodCalcultor::DoGLobalFit - find MLE ",
        "[#0] PROGRESS:Minimization -- ProfileLikelihoodCalcultor::DoMinimizeNLL - using "
        "Minuit2 /  with strategy 1",
        "[#0] PROGRESS:Minimization -- ProfileLikelihoodCalcultor::GetHypoTest - do "
        "conditional fit ",
        "[#0] PROGRESS:Minimization -- ProfileLikelihoodCalcultor::DoMinimizeNLL - using "
        "Minuit2 /  with strategy 1",
    ]


CONTOUR_X = [0.2351389086370116, 0.47657951822460665, 0.9200000053367641, 1.363454619603536,
             1.6048610901507905, 1.349427925609061, 0.9200000053367641, 0.49053623566378945]
CONTOUR_Y = [1.0200247663445392, 0.6208071753617884, 0.47573297187502994, 0.6208317249853179,
             1.020116784869318, 1.3784934218672076, 1.4747824491980963, 1.3784748785866454]


def test_the_contour_of_mean_and_width_is_minuits(capsys: Any) -> None:
    """ROOT 6.40: MnContours' eight points at the two-dimensional 95% level, each end of the
    two, and the null ``(0, 1)`` of both at p 0.007547196109229864."""
    w, data = gaussian()
    mu, s = w.var("mu"), w.var("s")
    plc = ProfileLikelihoodCalculator(data, config(w, "mu,s"))
    interval = plc.GetInterval()
    xs, ys = [0.0] * 10, [0.0] * 10
    assert interval.GetContourPoints(mu, s, xs, ys, 8) == 8
    assert xs[:8] == pytest.approx(CONTOUR_X, rel=1e-7)
    assert ys[:8] == pytest.approx(CONTOUR_Y, rel=1e-7)
    assert "LikelihoodInterval - Finding the contour of mu ( 0 ) and s ( 1 ) " in (
        capsys.readouterr().out)  # fmt: skip
    assert interval.LowerLimit(mu) == pytest.approx(0.402588822071688, rel=1e-9)
    assert interval.UpperLimit(s) == pytest.approx(1.2625358137868057, rel=1e-9)
    null = ROOT.RooArgSet(mu, s).snapshot()
    null.setRealValue("mu", 0.0)
    null.setRealValue("s", 1.0)
    plc.SetNullParameters(null)
    assert plc.GetHypoTest().NullPValue() == pytest.approx(0.007547196109229864, rel=1e-9)


def test_ends_at_the_range_and_a_contour_minuit_cannot_close(capsys: Any) -> None:
    """ROOT 6.40: ``mu`` in [0.8, 1.1] has the range for its ends; MnContours gives six points
    for five, which Minuit2 refuses, so there are none."""
    w, data = gaussian("Gaussian::g(x[-10,10],mu[1,0.8,1.1],s[1,0.1,5])")
    mu, s = w.var("mu"), w.var("s")
    plc = ProfileLikelihoodCalculator(data, w.pdf("g"), ROOT.RooArgSet(mu), 0.3173)
    interval = plc.GetInterval()
    assert (interval.LowerLimit(mu), interval.UpperLimit(mu)) == (0.8, 1.1)
    capsys.readouterr()
    assert interval.GetContourPoints(mu, s, [0.0] * 5, [0.0] * 5, 5) == 0
    said = capsys.readouterr()
    assert said.err == (
        "Error in <Minuit2>: Minuit2Minimizer::Contour Invalid result from MnContours\n")
    assert said.out.endswith(
        "[#0] ERROR:Minimization -- LikelihoodInterval - Error finding contour for parameters "
        "mu and s\n"
    )


def test_wrong_parameters_find_no_ends_and_no_contour(capsys: Any) -> None:
    """ROOT 6.40: an unknown parameter is refused by name, for the ends and the contour; a
    point of other parameters is in no interval, quietly."""
    w, data = gaussian()
    mu, s = w.var("mu"), w.var("s")
    interval = ProfileLikelihoodCalculator(data, w.pdf("g"), [mu]).GetInterval()
    other = ROOT.RooRealVar("other", "other", 0.0)
    capsys.readouterr()
    assert interval.LowerLimit(other) == 0.0
    lower = ctypes.c_double(7.0)
    assert not interval.FindLimits(other, lower)
    assert lower.value == 7.0
    assert interval.GetContourPoints(mu, other, [0.0], [0.0], 1) == 0
    assert not interval.IsInInterval(ROOT.RooArgSet(s))
    assert not interval.IsInInterval(ROOT.RooArgSet(mu, s))
    assert capsys.readouterr().out == (
        "Error - invalid parameter other specified for finding the interval limits \n" * 2
        + "[#0] ERROR:InputArguments -- LikelihoodInterval - Error - invalid parameters "
        "specified for finding the contours; parX = mu parY = other\n"
    )


def test_a_null_with_nothing_left_to_fit_takes_the_likelihood_as_it_is(capsys: Any) -> None:
    """ROOT 6.40: with ``s`` constant, no conditional fit: p 0.40014100705924033 for the null
    ``mu = 1``; a constant null has no degrees of freedom, and p 0."""
    w, data = gaussian("Gaussian::g(x[-10,10],mu[1,0.8,1.1],s[1,0.1,5])")
    mu, s = w.var("mu"), w.var("s")
    s.setConstant(True)
    null = ROOT.RooArgSet(mu).snapshot()
    plc = ProfileLikelihoodCalculator(data, w.pdf("g"), ROOT.RooArgSet(mu), 0.05, null)
    assert plc.GetHypoTest().NullPValue() == pytest.approx(0.40014100705924033, rel=1e-9)
    assert "GetHypoTest - do conditional fit" not in capsys.readouterr().out
    w, data = gaussian("Gaussian::g(x[-10,10],mu[1,-5,5],s[1,0.1,5])")
    mu = w.var("mu")
    null = ROOT.RooArgSet(mu).snapshot()
    null.setRealValue("mu", 0.5)
    null.find("mu").setConstant(True)
    plc = ProfileLikelihoodCalculator(data, w.pdf("g"), ROOT.RooArgSet(mu), 0.05, null)
    assert plc.GetHypoTest().NullPValue() == 0.0
    assert (mu.isConstant(), mu.getVal()) == (False, pytest.approx(0.920290007543507, rel=1e-6))


def retries(out: str) -> list[str]:
    return [line.split("-- ")[1] for line in out.splitlines() if "---->" in line]


def test_without_anything_to_fit_there_is_no_interval_and_no_test(capsys: Any) -> None:
    """ROOT 6.40: every parameter constant, MIGRAD is tried four times, a scan between, then
    the save fails - no fit, so no interval and no test."""
    w, data = gaussian()
    mu = w.var("mu")
    mu.setConstant(True)
    w.var("s").setConstant(True)
    plc = ProfileLikelihoodCalculator(data, w.pdf("g"), [mu], 0.05, [mu])
    assert plc.GetInterval() is None
    out = capsys.readouterr().out
    assert retries(out) == [
        "    ----> Doing a re-scan first", "    ----> Doing a re-scan first",
        "    ----> trying with improve",
    ]
    assert out.count("FCN function has zero parameters") == 5
    assert out.endswith(
        "[#0] WARNING:Minimization -- RooMinimizer::save: Error, run minimization before!\n")
    assert plc.GetHypoTest() is None
    assert plc.GetFitResult() is None


def test_a_calculator_short_of_data_a_model_or_its_sets_gives_nothing() -> None:
    """ROOT 6.40: no parameters of interest, no null, no data or no model: ``None``."""
    w, data = gaussian()
    mu = w.var("mu")
    plc = ProfileLikelihoodCalculator(data, w.pdf("g"), ROOT.RooArgSet(mu))
    assert plc.GetHypoTest() is None
    plc.SetConfidenceLevel(0.9)
    assert (plc.Size(), plc.ConfidenceLevel()) == (pytest.approx(0.1), 0.9)
    plc.SetParameters(ROOT.RooArgSet())
    assert plc.GetInterval() is None
    empty = ProfileLikelihoodCalculator()
    assert (empty.GetInterval(), empty.GetHypoTest(), empty.GetData(), empty.GetPdf()) == (
        None, None, None, None)  # fmt: skip
    empty.SetData(data)
    empty.SetPdf(w.pdf("g"))
    empty.SetParameters([mu])
    assert (empty.GetData(), empty.GetPdf()) == (data, w.pdf("g"))
    assert empty.GetInterval() is not None


def test_a_calculator_takes_its_sets_from_a_model_config_or_one_by_one() -> None:
    """``SetModel`` takes the density and every set the ModelConfig names; each setter one."""
    w, data = gaussian()
    mu, s, x = w.var("mu"), w.var("s"), w.var("x")
    mc = config(w)
    mc.SetConditionalObservables(ROOT.RooArgSet())
    mc.SetGlobalObservables(ROOT.RooArgSet())
    calc = CombinedCalculator(data, mc)
    assert calc.GetPdf() is w.pdf("g") and calc.Size() == 0.05
    assert [p.GetName() for p in calc._sets["null"]] == ["mu"]
    calc = CombinedCalculator(data, w.pdf("g"), [mu], 0.1, None, [mu], [s])
    assert calc.Size() == 0.1 and not len(calc._sets["null"])
    calc.SetTestSize(0.2)
    calc.SetAlternateParameters([s])
    calc.SetNuisanceParameters([mu])
    calc.SetConditionalObservables([x])
    calc.SetGlobalObservables([x])
    calc.SetModel(ROOT.RooStats.ModelConfig("bare", w))  # names nothing, changes nothing
    assert calc.Size() == 0.2
    assert [[p.GetName() for p in calc._sets[k]] for k in ("alt", "nuis", "cond", "glob")] == [
        ["s"], ["mu"], ["x"], ["x"]]  # fmt: skip
    assert CombinedCalculator(data, w.pdf("g"), [mu]).Size() == 0.05


class Result:
    """What ``RooMinimizer::save`` hands back, as far as the calculator reads it."""

    def __init__(self, status: int) -> None:
        self._status = status

    def status(self) -> int:
        return self._status

    def minNll(self) -> float:
        return 1.0

    def floatParsFinal(self) -> list[Any]:
        return []

    def defaultPrintContents(self, option: Any) -> int:
        return 0

    def defaultPrintStyle(self, option: Any) -> int:
        return 0

    def printStream(self, contents: Any, style: Any) -> str:
        return f"  fit of status {self._status}\n"


class Scripted:
    """A RooMinimizer whose MIGRADs end with the statuses it is given, one by one."""

    statuses: ClassVar[list[int]] = []
    calls: ClassVar[list[tuple[str, str, int]]] = []
    minimizer_type = "Minuit2"

    def __init__(self, nll: Any) -> None:
        self._strategy = 1
        self._last = -1

    def setStrategy(self, strategy: int) -> None:
        self._strategy = strategy

    def setEps(self, eps: float) -> None:
        pass

    def setPrintLevel(self, level: int) -> None:
        pass

    def optimizeConst(self, level: int) -> None:
        pass

    def minimize(self, kind: str, algorithm: str) -> int:
        self.calls.append((kind, algorithm, self._strategy))
        if algorithm != "Scan":
            self._last = self.statuses.pop(0)
        return self._last

    def save(self) -> Result:
        return Result(self._last)


@pytest.fixture
def scripted(monkeypatch: Any) -> type[Scripted]:
    """The calculator's minimizer scripted: its calls kept, its statuses given."""
    monkeypatch.setattr(combined, "RooMinimizer", Scripted)
    Scripted.calls = []
    return Scripted


def test_with_strategy_zero_a_failing_fit_is_tried_with_one_then_improved(
    scripted: Any, capsys: Any
) -> None:
    """``DoMinimizeNLL``: a re-scan after each failure, strategy 1 at the third try and
    Minuit's improved MIGRAD at the fourth."""
    ROOT.Math.MinimizerOptions.SetDefaultStrategy(0)
    scripted.statuses = [3, 3, 3, 1003]
    assert combined.minimize_nll(None, "Owner").status() == 1003
    assert scripted.calls == [
        ("", "", 0), ("", "Scan", 0), ("", "", 0), ("", "Scan", 0), ("", "", 1),
        ("", "Scan", 1), ("Minuit", "migradimproved", 1),
    ]
    assert retries(capsys.readouterr().out) == [
        "    ----> Doing a re-scan first", "    ----> Doing a re-scan first",
        "    ----> trying with strategy = 1", "    ----> Doing a re-scan first",
        "    ----> trying with improve",
    ]


def test_a_fit_that_succeeds_on_a_retry_is_kept(scripted: Any) -> None:
    """A second MIGRAD that converges ends the tries; its status is the fit's."""
    scripted.statuses = [4, 0]
    assert combined.minimize_nll(None).status() == 0
    assert [c[1] for c in scripted.calls] == ["", "Scan", ""]


def test_failed_fits_are_said_and_the_test_goes_on(scripted: Any, capsys: Any) -> None:
    """ROOT's warnings for a global fit and a conditional one that fail; a null parameter the
    model does not have is left alone."""
    w, data = gaussian()
    mu = w.var("mu")
    scripted.statuses = [0, 1, 1, 1]
    plc = ProfileLikelihoodCalculator(data, w.pdf("g"), [mu], 0.05, [mu])
    assert plc.GetHypoTest().NullPValue() == 0.5
    out = capsys.readouterr().out
    assert "[#1] INFO:Minimization --   fit of status 0\n" in out
    assert ("[#0] WARNING:Minimization -- ProfileLikelihoodCalcultor::GetHypotest -  "
            "Conditional fit failed - status = 1\n") in out  # fmt: skip
    scripted.statuses = [1000, 0]  # an error of Improve's alone ends the tries
    zz = ROOT.RooRealVar("zz", "zz", 0.0, -1.0, 1.0)
    plc.SetNullParameters([zz])
    assert plc.GetHypoTest().NullPValue() == 0.5
    assert ("[#0] WARNING:Minimization -- ProfileLikelihoodCalcultor::DoGlobalFit -  Global "
            "fit failed - status = 1000\n") in capsys.readouterr().out  # fmt: skip


def test_a_constant_parameter_of_interest_keeps_its_own_value_as_the_best() -> None:
    """A constant parameter of interest is not among the fit's: the best is itself."""
    w, data = gaussian()
    mu = w.var("mu")
    mu.setVal(0.5)
    mu.setConstant(True)
    interval = ProfileLikelihoodCalculator(data, w.pdf("g"), [mu]).GetInterval()
    assert interval.GetBestFitParameters().getRealValue("mu") == 0.5
