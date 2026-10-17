"""Integration settings change the calculation, with ROOT's numeric results as references."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from refmachine import roots
from xrdroot.roofit import numintconfig
from xrdroot.roofit.integratorconfig import configuration, configured


@pytest.fixture(autouse=True)
def _fresh_configuration(monkeypatch):
    monkeypatch.setattr(numintconfig, "DEFAULT", [])


CASES = json.loads((Path(__file__).parent / "data/roofit-romberg-controls-6.40.json").read_text())


@pytest.mark.parametrize("case", CASES)
def test_romberg_refinement_controls_match_root(case):
    x = ROOT.RooRealVar("x", "x", -1.0, 1.0)
    f = ROOT.RooFormulaVar("f", "f", case["formula"], ROOT.RooArgList(x))
    config = ROOT.RooNumIntConfig(ROOT.RooAbsReal.defaultIntegratorConfig())
    section = config.getConfigSection("RooIntegrator1D")
    section.setRealValue("maxSteps", 6)
    section.setRealValue("fixSteps", case["fixed"])
    section.setRealValue("minSteps", case["minimum"])
    section.setCatLabel("sumRule", case["rule"])
    section.setCatLabel("extrapolation", case["extrap"])
    found = f.createIntegral({x}, NumIntConfig=config).getVal()
    # exp can round one ulp differently on other platforms; refinement and
    # extrapolation propagate that difference into the final integral.
    assert found == roots(case["value"], rel=2e-12, abs=1e-14)


def test_explicit_configuration_overrides_the_objects_private_copy():
    x = ROOT.RooRealVar("x", "x", -10.0, 10.0)
    f = ROOT.RooLandau("landau", "landau", x, 0.0, 0.1)
    f.forceNumInt()
    default = ROOT.RooAbsReal.defaultIntegratorConfig()
    default.setEpsAbs(1e-6)
    default.setEpsRel(1e-6)
    custom = ROOT.RooNumIntConfig(default)
    custom.method1D().setLabel("RooAdaptiveGaussKronrodIntegrator1D")
    f.setIntegratorConfig(custom)
    assert f.createIntegral({x}).getVal() == roots(0.09895710292189497, rel=1e-13)
    assert f.createIntegral({x}, default).getVal() == roots(0.09896533620544187, rel=1e-13)
    assert f.getIntegratorConfig().method1D().getLabel() == "RooAdaptiveGaussKronrodIntegrator1D"


def test_scoped_settings_restore_the_previous_configuration_after_an_exception():
    x = ROOT.RooRealVar("x", "x", -1.0, 1.0)
    f = ROOT.RooFormulaVar("f", "f", "x*x", ROOT.RooArgList(x))
    a, b = ROOT.RooNumIntConfig(), ROOT.RooNumIntConfig()
    default = configuration(f)
    with configured(f, a):
        assert configuration(f) is a
        with pytest.raises(RuntimeError, match="deliberate"):
            with configured(f, b):
                assert configuration(f) is b
                raise RuntimeError("deliberate")
        assert configuration(f) is a
    assert configuration(f) is default


def test_adaptive_integration_subdivides_independently_for_each_event():
    x = ROOT.RooRealVar("x", "x", -1.0, 1.0)
    y = ROOT.RooRealVar("y", "y", 0.0, -1.0, 1.0)
    f = ROOT.RooFormulaVar("f", "f", "exp(-100*(x-y)*(x-y))", ROOT.RooArgList(x, y))
    config = f.specialIntegratorConfig(True)
    config.method1D().setLabel("RooAdaptiveGaussKronrodIntegrator1D")
    points = np.array([0.0, 0.5, 0.9])
    many = f.integrate({"x"}, {"y": points})
    separate = np.array([f.integrate({"x"}, {"y": value}) for value in points])
    assert np.array_equal(many, separate)


@pytest.mark.parametrize("method", ("RooSegmentedIntegrator1D", "RooIntegrator1D"))
def test_segmented_and_ordinary_integration_use_the_configured_refinement_rule(method):
    x = ROOT.RooRealVar("x", "x", -1.0, 1.0)
    f = ROOT.RooFormulaVar("f", "f", "x*x*x*x", ROOT.RooArgList(x))
    c = f.specialIntegratorConfig(True)
    c.method1D().setLabel(method)
    c.getConfigSection("RooIntegrator1D").setRealValue("fixSteps", 3)
    # Fixed-step trapezoids: each of three segments is independently refined.
    expected = 0.4184670781893004 if method == "RooSegmentedIntegrator1D" else 0.5625
    assert f.createIntegral({x}).getVal() == roots(expected, rel=1e-14)


@pytest.mark.parametrize("method", ("RooMCIntegrator", "RooGaussKronrodIntegrator1D"))
def test_unimplemented_numeric_algorithms_are_refused_instead_of_silently_substituted(method):
    from xrdroot.errors import UnsupportedFeatureError

    x = ROOT.RooRealVar("x", "x", -1.0, 1.0)
    f = ROOT.RooFormulaVar("f", "f", "x*x", ROOT.RooArgList(x))
    f.specialIntegratorConfig(True).method1D().setLabel(method)
    with pytest.raises(UnsupportedFeatureError, match=method):
        f.createIntegral({x}).getVal()


@pytest.mark.parametrize("opened", (False, True))
def test_unimplemented_gauss_kronrod_extrapolation_and_open_ends_are_refused(opened):
    from xrdroot.errors import UnsupportedFeatureError

    x = ROOT.RooRealVar("x", "x", -1.0, 1.0)
    f = ROOT.RooFormulaVar("f", "f", "exp(-x*x)", ROOT.RooArgList(x))
    c = f.specialIntegratorConfig(True)
    if opened:
        x.setRange(-float("inf"), float("inf"))
        c.method1DOpen().setLabel("RooAdaptiveGaussKronrodIntegrator1D")
    else:
        c.method1D().setLabel("RooAdaptiveGaussKronrodIntegrator1D")
        c.getConfigSection("RooAdaptiveGaussKronrodIntegrator1D").setCatLabel(
            "method", "WynnEpsilon"
        )
    with pytest.raises(UnsupportedFeatureError, match="Wynn-epsilon"):
        f.createIntegral({x}).getVal()


@pytest.mark.parametrize("rule,maximum", (("Trapezoid", 20), ("Midpoint", 14)))
def test_nonpositive_refinement_limits_use_roots_rule_specific_default(rule, maximum):
    from xrdroot.roofit.integratorconfig import romberg_settings

    c = ROOT.RooNumIntConfig(ROOT.RooAbsReal.defaultIntegratorConfig())
    section = c.getConfigSection("RooIntegrator1D")
    section.setCatLabel("sumRule", rule)
    section.setRealValue("maxSteps", 0)
    assert romberg_settings(c).max_steps == maximum


def test_fixed_steps_are_clipped_to_the_configured_maximum_and_reported(capsys):
    x = ROOT.RooRealVar("x", "x", -1.0, 1.0)
    f = ROOT.RooFormulaVar("f", "f", "x*x", ROOT.RooArgList(x))
    c = f.specialIntegratorConfig(True)
    s = c.getConfigSection("RooIntegrator1D")
    s.setRealValue("maxSteps", 3)
    s.setRealValue("fixSteps", 4)
    assert f.createIntegral({x}).getVal() == 0.75
    assert "fixSteps>maxSteps, fixSteps set to maxSteps" in capsys.readouterr().out


def test_zero_tolerances_use_roots_romberg_fallback_accuracy():
    x = ROOT.RooRealVar("x", "x", -10.0, 10.0)
    f = ROOT.RooLandau("landau", "landau", x, 0.0, 0.1)
    f.forceNumInt()
    c = f.specialIntegratorConfig(True)
    c.setEpsAbs(0.0)
    c.setEpsRel(0.0)
    assert f.createIntegral({x}).getVal() == roots(0.09896533620544187, rel=1e-13)


def test_bin_integration_respects_the_configured_number_of_fallback_bins():
    x = ROOT.RooRealVar("x", "x", -1.0, 1.0)
    f = ROOT.RooFormulaVar("f", "f", "x*x", ROOT.RooArgList(x))
    c = f.specialIntegratorConfig(True)
    c.method1D().setLabel("RooBinIntegrator")
    c.getConfigSection("RooBinIntegrator").setRealValue("numBins", 2)
    assert f.createIntegral({x}).getVal() == 0.5


@pytest.mark.parametrize(
    "method,expected",
    (
        ("RooIntegrator2D", 0.421875),
        ("RooSegmentedIntegrator2D", 0.28285274729461973),
    ),
)
def test_two_dimensional_romberg_uses_its_dependency_regardless_of_the_one_dimensional_choice(
    method, expected
):
    x = ROOT.RooRealVar("x", "x", -1.0, 1.0)
    y = ROOT.RooRealVar("y", "y", -1.0, 1.0)
    f = ROOT.RooFormulaVar("f", "f", "x*x*x*x*y*y", ROOT.RooArgList(x, y))
    c = f.specialIntegratorConfig(True)
    c.method2D().setLabel(method)
    c.method1D().setLabel("RooAdaptiveGaussKronrodIntegrator1D")
    c.getConfigSection("RooIntegrator1D").setRealValue("fixSteps", 3)
    assert f.createIntegral({x, y}).getVal() == roots(expected, rel=1e-14)
