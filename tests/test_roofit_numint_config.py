"""Numeric integration settings and ownership checked against ROOT 6.40.04."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roofit import numintconfig
from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.variables import RooRealVar


@pytest.fixture(autouse=True)
def _fresh_configuration(monkeypatch):
    monkeypatch.setattr(numintconfig, "DEFAULT", [])
    monkeypatch.setattr(numintconfig, "MISSING", RooArgSet())


def test_an_empty_configuration_has_no_registered_integrators():
    config = ROOT.RooNumIntConfig()
    for name in (
        "method1D",
        "method2D",
        "methodND",
        "method1DOpen",
        "method2DOpen",
        "methodNDOpen",
    ):
        method = getattr(config, name)()
        assert (method.getLabel(), method.getIndex(), method.numTypes()) == ("N/A", 0, 1)
    assert (config.epsAbs(), config.epsRel(), config.printEvalCounter()) == (1e-7, 1e-7, False)


def test_the_default_is_shared_by_functions_and_has_roots_method_indices():
    config = ROOT.RooAbsReal.defaultIntegratorConfig()
    assert config is ROOT.RooNumIntConfig.defaultConfig()
    assert config is ROOT.RooAbsPdf.defaultIntegratorConfig()
    expected = (
        ("method1D", "RooIntegrator1D", 2, 8),
        ("method2D", "RooAdaptiveIntegratorND", 5, 6),
        ("methodND", "RooAdaptiveIntegratorND", 3, 4),
        ("method1DOpen", "RooImproperIntegrator1D", 1, 4),
    )
    for name, label, index, count in expected:
        method = getattr(config, name)()
        assert (method.getLabel(), method.getIndex(), method.numTypes()) == (label, index, count)


def test_verbose_configuration_output_matches_root_character_for_character(capsys):
    ROOT.RooNumIntConfig.defaultConfig().Print("v")
    expected = (Path(__file__).parent / "data/roofit-numint-config-6.40.txt").read_text()
    assert capsys.readouterr().out == expected


def test_nonverbose_output_omits_sections_and_reports_the_evaluation_counter(capsys):
    config = ROOT.RooNumIntConfig()
    config.setPrintEvalCounter(True)
    assert config.printEvalCounter()
    config.Print()
    text = capsys.readouterr().out
    assert "Printing of function evaluation counter for each integration enabled\n\n" in text
    assert "1-D integration method: N/A\n" in text
    assert "Available integration methods" not in text


@pytest.mark.parametrize("method,getter", (("setEpsAbs", "epsAbs"), ("setEpsRel", "epsRel")))
@pytest.mark.parametrize("value", (0.0, 1e-8, math.inf, math.nan))
def test_nonnegative_and_nonfinite_precision_values_follow_roots_setters(method, getter, value):
    config = ROOT.RooNumIntConfig()
    getattr(config, method)(value)
    result = getattr(config, getter)()
    assert math.isnan(result) if math.isnan(value) else result == value


@pytest.mark.parametrize("method,getter", (("setEpsAbs", "epsAbs"), ("setEpsRel", "epsRel")))
def test_negative_precision_is_reported_and_keeps_the_previous_value(method, getter, capsys):
    config = ROOT.RooNumIntConfig()
    getattr(config, method)(-1.0)
    assert getattr(config, getter)() == 1e-7
    assert "target absolute precision must be greater or equal than zero" in capsys.readouterr().out


def test_a_configuration_copy_owns_independent_method_choices_and_parameters():
    original = ROOT.RooNumIntConfig.defaultConfig()
    copied = ROOT.RooNumIntConfig(original)
    copied.setEpsAbs(3e-6)
    copied.method1D().setLabel("RooAdaptiveGaussKronrodIntegrator1D")
    copied.getConfigSection("RooIntegrator1D").setRealValue("maxSteps", 30)
    assert original.epsAbs() == 1e-7
    assert original.method1D().getLabel() == "RooIntegrator1D"
    assert original.getConfigSection("RooIntegrator1D").getRealValue("maxSteps") == 20
    assert copied.getConfigSection("RooIntegrator1D").getRealValue("maxSteps") == 30


def test_sections_clone_the_supplied_parameters_and_register_each_capability():
    config = ROOT.RooNumIntConfig()
    parameter = RooRealVar("steps", "steps", 17.0)
    assert config.addConfigSection("custom", RooArgSet(parameter), True, True, True, True) is False
    parameter.setVal(21.0)
    section = config.getConfigSection("custom")
    assert section.GetName() == "custom" and section.getRealValue("steps") == 17.0
    for name in (
        "method1D",
        "method2D",
        "methodND",
        "method1DOpen",
        "method2DOpen",
        "methodNDOpen",
    ):
        assert getattr(config, name)().lookupIndex("custom") == 1


def test_an_unknown_section_returns_the_shared_empty_set_and_reports_the_error(capsys):
    a, b = ROOT.RooNumIntConfig(), ROOT.RooNumIntConfig()
    section = a.getConfigSection("missing")
    assert section is b.getConfigSection("other") and len(section) == 0
    assert "no configuration stored for integrator 'missing'" in capsys.readouterr().out


def test_private_function_settings_are_cloned_and_can_be_reset_to_the_defaults():
    x = ROOT.RooRealVar("x", "x", -10.0, 10.0)
    gaussian = ROOT.RooGaussian("g", "g", x, 0.0, 1.0)
    default = ROOT.RooAbsReal.defaultIntegratorConfig()
    assert gaussian.getIntegratorConfig() is default
    assert gaussian.specialIntegratorConfig() is None
    private = gaussian.specialIntegratorConfig(True)
    private.setEpsRel(2e-6)
    assert gaussian.getIntegratorConfig() is private and default.epsRel() == 1e-7
    custom = ROOT.RooNumIntConfig(default)
    custom.setEpsAbs(3e-6)
    gaussian.setIntegratorConfig(custom)
    custom.setEpsAbs(4e-6)
    assert gaussian.getIntegratorConfig().epsAbs() == 3e-6
    gaussian.setIntegratorConfig()
    assert gaussian.specialIntegratorConfig() is None
    assert gaussian.getIntegratorConfig() is default
    gaussian.forceNumInt()
    assert gaussian.getForceNumInt()
    gaussian.forceNumInt(False)
    assert not gaussian.getForceNumInt()
