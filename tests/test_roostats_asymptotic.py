"""The AsymptoticCalculator on a counting model with a constrained background, against ROOT.

``n`` events of ``mu s + nu b`` with ``nu`` constrained by a Gaussian of its
global observable ``nom``: the numbers are ROOT 6.40's for the same model -
to the last bit, each model made afresh.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roofit.messages import service
from xrdroot.roostats import asymptotic
from xrdroot.roostats.asympformulae import expected_p_values, p_values


@pytest.fixture(autouse=True)
def _quiet() -> Any:
    service().reset()
    ROOT.RooStats.AsymptoticCalculator.SetPrintLevel(0)
    yield
    service().reset()


def counting(n: float = 8.0) -> tuple[Any, Any, Any, Any]:
    """The workspace, the data, and the S+B and B models."""
    w = ROOT.RooWorkspace("w")
    w.factory("Poisson::pois(n[0,30], sum::lam(prod::sig(mu[1,0,10], s[3]), "
              "prod::bkg(nu[1,0,3], b[5])))")  # fmt: skip
    w.factory("Gaussian::cons(nom[1,0,3], nu, sigma[0.2])")
    w.factory("PROD::model(pois, cons)")
    obs = w.var("n")
    obs.setVal(n)
    data = ROOT.RooDataSet("data", "data", ROOT.RooArgSet(obs))
    data.add(ROOT.RooArgSet(obs))
    sb = ROOT.RooStats.ModelConfig("sb", w)
    sb.SetPdf("model")
    sb.SetObservables("n")
    sb.SetParametersOfInterest("mu")
    sb.SetNuisanceParameters("nu")
    sb.SetGlobalObservables("nom")
    mu = w.var("mu")
    mu.setVal(1)
    sb.SetSnapshot(ROOT.RooArgSet(mu))
    b = sb.Clone("b")
    mu.setVal(0)
    b.SetSnapshot(ROOT.RooArgSet(mu))
    mu.setVal(1)
    return w, data, sb, b


def test_a_discovery_test_gives_roots_p_values_to_the_bit(capsys: Any) -> None:
    """The null at the parameter's minimum: one-sided discovery, said, and ROOT's numbers."""
    ROOT.RooStats.AsymptoticCalculator.SetPrintLevel(1)
    _, data, sb, b = counting()
    calc = ROOT.RooStats.AsymptoticCalculator(data, sb, b)
    result = calc.GetHypoTest()
    assert (result.NullPValue(), result.AlternatePValue(), result.CLs()) == (
        0.1323978114646034, 0.49999163636481636, 3.776434299282125)  # fmt: skip
    assert calc.IsOneSidedDiscovery() and not calc.IsTwoSided()
    out = capsys.readouterr().out
    assert "[#0] PROGRESS:Eval -- Best fitted POI value = 1 +/- 0.981559\n" in out
    assert ("[#1] INFO:InputArguments -- AsymptotiCalculator: Minimum of POI is 0 corresponds to "
            "null  snapshot   - default configuration is  one-sided discovery formulae  ") in out
    assert "RooArgSet:: = (n)\n" in out and "RooArgSet:: = (nom)\n" in out


def test_a_limit_test_uses_qtilde_and_zeroes_q_above_the_fit(capsys: Any) -> None:
    """The null at one, the alternate at the minimum: the ``qtilde`` formulae, said."""
    _, data, sb, b = counting()
    calc = ROOT.RooStats.AsymptoticCalculator(data, b, sb)
    calc.SetOneSided(True)
    result = calc.GetHypoTest()
    assert result.NullPValue() == 0.5
    assert result.AlternatePValue() == 0.8528773942114782
    out = capsys.readouterr().out
    assert ("Minimum of POI is 0 corresponds to alt  snapshot   - using qtilde asymptotic "
            "formulae") in out
    calc.SetTwoSided()
    two = calc.GetHypoTest()
    assert (two.NullPValue(), two.AlternatePValue()) == (1.0, 1.0)


def test_nominal_asimov_data_keeps_the_nuisance_parameters_where_they_are() -> None:
    """``nominalAsimov``: no conditional fit - the global observable stays at its value."""
    w, data, sb, b = counting()
    calc = ROOT.RooStats.AsymptoticCalculator(data, b, sb, True)
    calc.SetOneSided(True)
    calc.SetQTilde(False)
    result = calc.GetHypoTest()
    assert result.AlternatePValue() == 0.8602865572138911
    assert w.var("nom").getVal() == 1.0
    asimov = calc.GetAsimovData()
    assert asimov.GetName() == "CountingAsimovData0" and asimov.numEntries() == 1
    assert calc.GetNLL() == pytest.approx(calc.GetNLL()) and calc.GetExpectedNLL() < 10
    assert calc.GetBestFitPoi().first().GetName() == "mu"


def test_the_calculator_refuses_what_it_cannot_start_from(capsys: Any) -> None:
    """No density, no data, no parameter of interest, no snapshot of either model: said."""
    w, data, sb, b = counting()
    bare = ROOT.RooStats.ModelConfig("bare", w)
    ROOT.RooStats.AsymptoticCalculator(data, sb, bare)
    no_poi = sb.Clone("no_poi")
    no_poi.SetParametersOfInterest(ROOT.RooArgSet())
    ROOT.RooStats.AsymptoticCalculator(data, sb, no_poi)
    unsnapped = ROOT.RooStats.ModelConfig("unsnapped", w)
    unsnapped.SetPdf("model")
    unsnapped.SetObservables("n")
    unsnapped.SetParametersOfInterest("mu")
    ROOT.RooStats.AsymptoticCalculator(data, sb, unsnapped)
    ROOT.RooStats.AsymptoticCalculator(data, unsnapped, sb)
    calc = ROOT.RooStats.AsymptoticCalculator(data, sb, b)
    calc.SetData(None)
    calc._initialized = False
    assert calc.GetHypoTest() is None
    out = capsys.readouterr().out
    for text in ("ModelConfig has not a pdf defined", "data set has not been defined",
                 "ModelConfig has not POI defined.", "Null model needs a snapshot.",
                 "Alt (Background)  model needs a snapshot.",
                 "Error initializing Asymptotic calculator - return nullptr result"):
        assert text in out


def test_two_parameters_of_interest_are_warned_of_and_the_first_is_taken(capsys: Any) -> None:
    w, data, sb, b = counting()
    both = sb.Clone("both")
    both.SetParametersOfInterest("mu,nu")
    both.SetSnapshot(ROOT.RooArgSet(w.var("mu"), w.var("nu")))
    calc = ROOT.RooStats.AsymptoticCalculator(data, b, both)
    calc.GetHypoTest()
    out = capsys.readouterr().out
    assert "ModelConfig has more than one POI defined" in out
    assert "snapshot has more than one POI - assume as POI first parameter" in out


def _scripted(monkeypatch: Any, values: list[float]) -> Any:
    """The calculator, made, then each of its likelihood minima from ``values`` in turn."""
    monkeypatch.undo()
    _, data, sb, b = counting()
    calc = ROOT.RooStats.AsymptoticCalculator(data, b, sb)
    calc.SetOneSided(True)
    calc._nll_obs, calc._nll_asimov = 10.0, 10.0
    given = iter(values)
    monkeypatch.setattr(asymptotic, "evaluate_nll", lambda *args: next(given))
    return calc


@pytest.mark.parametrize(("values", "said", "dummy"), [
    ([9.0, 8.0, 10.5], "New minimum  found for" + " " * 27 + "NLL = 8    muHat  ", False),
    ([9.0, 11.0], "qmu is still < 0  for mu = 1 return a dummy result", True),
    ([10.5, 9.0, 8.0], "Found a better unconditional minimum for Asimov data set", False),
    ([10.5, 9.0, 11.0], "qmu_A is still < 0  for mu = 1 return a dummy result", True),
])  # fmt: skip
def test_a_negative_ratio_refits_unconditionally(monkeypatch: Any, capsys: Any, values: Any,
                                                 said: str, dummy: bool) -> None:  # fmt: skip
    """A ratio below zero: the unconditional fit again - kept if better, else a dummy result."""
    calc = _scripted(monkeypatch, values)
    result = calc.GetHypoTest()
    assert said in capsys.readouterr().out
    assert (result.GetName() == "") is dummy


def test_a_failed_fit_is_retried_and_a_nan_ratio_gives_a_dummy_result(
        monkeypatch: Any, capsys: Any) -> None:  # fmt: skip
    calc = _scripted(monkeypatch, [math.nan, math.nan])
    calc._nll_obs = math.nan
    assert calc.GetHypoTest().GetName() == ""
    out = capsys.readouterr().out
    assert "unconditional fit failed before - retry to do it now" in out
    assert "failure in fitting for qmu or qmuA 1 return a dummy result" in out
    calc = _scripted(monkeypatch, [10.5, 10.5, 12.0])
    calc._nll_asimov = math.nan
    calc.GetHypoTest()
    assert "Fit failed for  unconditional the qmu Asimov- retry" in capsys.readouterr().out


def test_the_calculator_says_its_tests_at_print_level_one(monkeypatch: Any, capsys: Any) -> None:
    calc = _scripted(monkeypatch, [10.5, 10.5, 12.0, 11.0])
    asymptotic.PRINT_LEVEL[0] = 1
    calc.GetHypoTest()
    out = capsys.readouterr().out
    assert "\t OBSERVED DATA :  qmu   = 1 condNLL = 10.5 uncond 10" in out
    assert ("poi = 1 qmu = 1 qmu_A = 1 sigma = 1  CLsplusb = 0.158655 CLb = 0.5 CLs = "
            "3.15149") in out


def test_a_one_sided_discovery_zeroes_q_below_the_fit(monkeypatch: Any, capsys: Any) -> None:
    calc = _scripted(monkeypatch, [10.5, 12.0])
    calc.SetOneSided(False)
    calc.SetOneSidedDiscovery(True)
    next(iter(calc._best_poi)).setVal(0.0)
    assert calc.GetHypoTest().NullPValue() == 0.5
    assert "Using one-sided discovery qmu - setting qmu to zero  muHat = 0 muTest = 1" in (
        capsys.readouterr().out)  # fmt: skip


def test_the_formulae_give_roots_p_values_one_and_two_sided_with_and_without_tilde() -> None:
    """``qmu = 4``, ``qmu_A = 1``: the four forms, ROOT's numbers."""
    tol = 2e-3
    assert p_values(4.0, 1.0, True, False, False, tol) == (0.022750131948179216,
                                                           0.15865525393145707)  # fmt: skip
    assert p_values(4.0, 1.0, False, False, False, tol) == (0.04550026389635843,
                                                            0.16000515196308718)  # fmt: skip
    assert p_values(4.0, 1.0, True, False, True, tol) == (0.006209665325776138,
                                                          0.06680720126885809)  # fmt: skip
    assert p_values(4.0, 1.0, False, False, True, tol) == (0.028959797273955354,
                                                           0.06815709930048817)  # fmt: skip


def test_expected_p_values_come_from_the_observed_pair() -> None:
    """One-sided in closed form; two-sided by two roots; -1 where there is no tail to invert."""
    calc = ROOT.RooStats.AsymptoticCalculator
    assert calc.GetExpectedPValues(0.05, 0.5, 1.0, True) == 0.3084479032573671
    assert calc.GetExpectedPValues(0.05, 0.5, 1.0, False) == 0.259511022841444
    assert calc.GetExpectedPValues(0.05, 0.5, -1e3, True) == -1.0
    assert calc.GetExpectedPValues(0.05, 0.5, 1.0, True, False) == pytest.approx(
        0.3335911989367035, rel=1e-9)  # fmt: skip
    assert calc.GetExpectedPValues(1.0, 0.5, 1.0, True, False) == -1.0


def test_an_expected_p_value_that_cannot_be_found_is_minus_one(capfd: Any) -> None:
    """No root in [0, 20]: MathCore's messages, then RooStats' -1."""
    assert expected_p_values(1e-300, 1.0 - 1e-16, 30.0, True, False) == -1.0
    said = capfd.readouterr()
    assert said.err == (
        "Info in <ROOT::Math::BrentMethods::MinimStep>: Grid search failed to find a root in the "
        " interval \nInfo in <ROOT::Math::BrentMethods::MinimStep>: xmin = 0 xmax = 20 npts = 100\n"
        "Error in <ROOT::Math::BrentRootFinder>: Interval does not contain a root\n")
    assert "[#0] ERROR:Eval -- Error finding expected p-values - return -1\n" in said.out
