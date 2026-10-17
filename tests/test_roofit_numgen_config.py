"""Sampler configuration and ownership follow ROOT 6.40.04, including lazy registration."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roofit import numgenconfig, printing, rng
from xrdroot.roofit.collections import RooArgSet

#: Native configurations before and after numeric sampling initializes the factory.
REFERENCE = json.loads((Path(__file__).parent / "data/roofit-numgen-config-6.40.json").read_text())


@pytest.fixture(autouse=True)
def _fresh_configuration(monkeypatch):
    monkeypatch.setattr(numgenconfig, "DEFAULT", [])
    monkeypatch.setattr(numgenconfig, "REGISTERED", [False])
    monkeypatch.setattr(numgenconfig, "MISSING", RooArgSet())
    monkeypatch.setattr(printing, "PRECISION", [6])
    monkeypatch.setattr(rng, "_GENERATOR", [])


def snapshot(config):
    found = {}
    for dimension, conditional, categories in itertools.product(
        ("1D", "2D", "ND"), (False, True), (False, True)
    ):
        method = getattr(config, "method" + dimension)(conditional, categories)
        found[f"{dimension}/{conditional}/{categories}"] = {
            "name": method.GetName(),
            "title": method.GetTitle(),
            "label": method.getLabel(),
            "index": method.getIndex(),
            "states": [[str(p.first), int(p.second)] for p in method],
        }
    return found


def test_empty_and_default_configurations_start_with_roots_unregistered_methods():
    assert snapshot(ROOT.RooNumGenConfig()) == REFERENCE["empty"]
    default = ROOT.RooNumGenConfig.defaultConfig()
    assert default is ROOT.RooAbsPdf.defaultGeneratorConfig()
    assert snapshot(default) == REFERENCE["before_factory"]


def test_numeric_sampling_registers_the_existing_default_and_roots_parameters():
    default = ROOT.RooAbsPdf.defaultGeneratorConfig()
    x = ROOT.RooRealVar("x", "x", 0.0, 10.0)
    model = ROOT.RooChebychev("model", "model", x, [0.0, 0.5, -0.1])
    assert model.generate({x}, 1).numEntries() == 1
    assert default is ROOT.RooAbsPdf.defaultGeneratorConfig()
    assert snapshot(default) == REFERENCE["default"]
    for name, expected in REFERENCE["sections"].items():
        found = [
            {
                "name": v.GetName(),
                "title": v.GetTitle(),
                "value": v.getVal(),
                "min": v.getMin(),
                "max": v.getMax(),
                "constant": v.isConstant(),
            }
            for v in default.getConfigSection(name)
        ]
        assert found == expected


def test_verbose_and_standard_configuration_printing_match_roots_choices(capsys):
    numgenconfig.initialize_factory()
    default = ROOT.RooAbsPdf.defaultGeneratorConfig()
    default.Print("v")
    assert (
        capsys.readouterr().out
        == (Path(__file__).parent / "data/roofit-numgen-config-6.40.txt").read_text()
    )
    default.Print()
    standard = capsys.readouterr().out
    assert "Available sampling methods" not in standard
    default.method1D(False, False).setLabel("RooAcceptReject")
    default.Print()
    assert (
        "(RooAcceptReject if conditional)"
        not in capsys.readouterr().out.split("2-D sampling method:")[0]
    )


def test_configuration_copies_keep_their_method_choices_and_values_independent():
    numgenconfig.initialize_factory()
    original = ROOT.RooNumGenConfig.defaultConfig()
    copied = ROOT.RooNumGenConfig(original)
    copied.method1D(False, False).setLabel("RooAcceptReject")
    copied.getConfigSection("RooFoamGenerator").setRealValue("nSample", 23)
    assert original.method1D(False, False).getLabel() == "RooFoamGenerator"
    assert original.getConfigSection("RooFoamGenerator").getRealValue("nSample") == 200
    assert copied.method1D(False, False).getLabel() == "RooAcceptReject"
    assert copied.getConfigSection("RooFoamGenerator").getRealValue("nSample") == 23


def test_function_specific_settings_are_cloned_and_can_be_reset():
    numgenconfig.initialize_factory()
    x = ROOT.RooRealVar("x", "x", -1.0, 1.0)
    pdf = ROOT.RooGaussian("g", "g", x, 0.0, 1.0)
    default = ROOT.RooAbsPdf.defaultGeneratorConfig()
    assert pdf.getGeneratorConfig() is default and pdf.specialGeneratorConfig() is None
    private = pdf.specialGeneratorConfig(True)
    assert private is pdf.specialGeneratorConfig(True) and private is not default
    private.method1D(False, False).setLabel("RooAcceptReject")
    assert default.method1D(False, False).getLabel() == "RooFoamGenerator"
    custom = ROOT.RooNumGenConfig(default)
    pdf.setGeneratorConfig(custom)
    custom.method1D(False, False).setLabel("RooAcceptReject")
    assert pdf.getGeneratorConfig().method1D(False, False).getLabel() == "RooFoamGenerator"
    pdf.setGeneratorConfig()
    assert pdf.specialGeneratorConfig() is None and pdf.getGeneratorConfig() is default


@pytest.mark.parametrize(
    "conditional,categories", list(itertools.product((False, True), (False, True)))
)
def test_custom_sections_register_capabilities_and_own_copied_parameters(conditional, categories):
    class Prototype:
        def generatorName(self):
            return "custom"

        def canSampleConditional(self):
            return conditional

        def canSampleCategories(self):
            return categories

    config = ROOT.RooNumGenConfig()
    variable = ROOT.RooRealVar("steps", "steps", 17.0)
    assert config.addConfigSection(Prototype(), ROOT.RooArgSet(variable)) is False
    variable.setVal(23.0)
    assert config.getConfigSection("custom").getRealValue("steps") == 17.0
    for dimension, cond, cat in itertools.product(("1D", "2D", "ND"), (False, True), (False, True)):
        method = getattr(config, "method" + dimension)(cond, cat)
        expected = 2 if (not cond or conditional) and (not cat or categories) else 1
        assert method.numTypes() == expected


def test_unknown_sections_share_roots_empty_placeholder_and_error_message(capsys):
    a, b = ROOT.RooNumGenConfig(), ROOT.RooNumGenConfig()
    missing = a.getConfigSection("missing")
    assert missing is b.getConfigSection("other") and len(missing) == 0
    assert (
        "RooNumGenConfig::getIntegrator: ERROR: no configuration stored for integrator 'missing'"
        in capsys.readouterr().out
    )


def test_an_empty_private_configuration_is_not_silently_replaced_with_global_defaults():
    x = ROOT.RooRealVar("x", "x", 0.0, 1.0)
    model = ROOT.RooGenericPdf("p", "p", "1+@0", ROOT.RooArgList(x))
    model.setGeneratorConfig(ROOT.RooNumGenConfig())
    with pytest.raises(ValueError, match="No numerical event generator was selected"):
        model.generate({x}, 1)


def test_an_unimplemented_custom_sampler_is_refused_instead_of_substituted():
    from xrdroot.errors import UnsupportedFeatureError

    numgenconfig.initialize_factory()
    x = ROOT.RooRealVar("x", "x", 0.0, 1.0)
    model = ROOT.RooGenericPdf("p", "p", "1+@0", ROOT.RooArgList(x))
    config = model.specialGeneratorConfig(True)
    config._register("custom", ROOT.RooArgSet(), False, False)
    config.method1D(False, False).setLabel("custom")
    with pytest.raises(UnsupportedFeatureError, match="generation with custom is not implemented"):
        model.generate({x}, 1)


def test_four_dimensional_foam_options_use_the_nd_cell_setting():
    from xrdroot.roofit.generation.samplerconfig import foam_options

    x = ROOT.RooRealVar("x", "x", 0.0, 1.0)
    model = ROOT.RooGenericPdf("p", "p", "1+@0", ROOT.RooArgList(x))
    assert foam_options(model, 4) == {"n_cells": 10000, "n_sample": 200, "chat_level": 0}
