"""Generator descriptions and dimensional settings are checked against native ROOT datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roofit import numgenconfig, printing, rng

#: Native direct, mixed, component and multidimensional generation references.
CASES = json.loads((Path(__file__).parent / "data/roofit-generation-verbose-6.40.json").read_text())


@pytest.fixture(autouse=True)
def _fresh_configuration(monkeypatch):
    monkeypatch.setattr(numgenconfig, "DEFAULT", [])
    monkeypatch.setattr(numgenconfig, "REGISTERED", [False])
    monkeypatch.setattr(printing, "PRECISION", [6])
    monkeypatch.setattr(rng, "_GENERATOR", [])


def model_for(kind):
    x = ROOT.RooRealVar("x", "x", -1.0, 1.0)
    if kind == "direct":
        return ROOT.RooGaussian("g", "g", x, 0.0, 0.4), [x]
    if kind == "mixed":
        y = ROOT.RooRealVar("y", "y", 0.0, 1.0)
        mean = ROOT.RooFormulaVar("mean", "mean", "0.2*@0", ROOT.RooArgList(y))
        return ROOT.RooGaussian("g", "g", x, mean, 0.4), [x, y]
    if kind in ("sum", "product"):
        return gaussian_components(kind, x)
    return dimensional_model(kind)


def gaussian_components(kind, x):
    y = ROOT.RooRealVar("y", "y", -1.0, 1.0)
    first = ROOT.RooGaussian("g1", "g1", x, -0.3, 0.2)
    second = ROOT.RooGaussian("g2", "g2", y if kind == "product" else x, 0.3, 0.4)
    if kind == "product":
        return ROOT.RooProdPdf("model", "model", ROOT.RooArgList(first, second)), [x, y]
    fraction = ROOT.RooRealVar("fraction", "fraction", 0.4)
    return ROOT.RooAddPdf(
        "model", "model", ROOT.RooArgList(first, second), ROOT.RooArgList(fraction)
    ), [x]


def dimensional_model(kind):
    variables = [
        ROOT.RooRealVar(name, name, 0.0, 1.0) for name in ("a", "b", "c", "d")[: int(kind[-1])]
    ]
    formula = "1+" + "+".join("@" + str(i) for i in range(len(variables)))
    model = ROOT.RooGenericPdf("p", "p", formula, ROOT.RooArgList(*variables))
    numgenconfig.initialize_factory()
    config = model.specialGeneratorConfig(True)
    for key in ("nCell3D", "nCellND"):
        config.getConfigSection("RooFoamGenerator").setRealValue(key, 1)
    config.getConfigSection("RooFoamGenerator").setRealValue("nSample", 8)
    return model, variables


@pytest.mark.parametrize("case", CASES)
def test_verbose_sampling_preserves_roots_coordinates_and_random_stream(case, capsys):
    model, variables = model_for(case["kind"])
    generator = ROOT.RooRandom.randomGenerator()
    generator.SetSeed(37)
    capsys.readouterr()
    data = model.generate(ROOT.RooArgSet(*variables), 6, AutoBinned=False, Verbose=True)
    shown = capsys.readouterr().out
    found = [
        [data.get(i).getRealValue(v.GetName()) for v in variables] for i in range(data.numEntries())
    ]
    assert found == case["events"]
    assert [generator.Rndm() for _ in range(16)] == case["next"]
    if case["kind"] != "product":
        assert (
            "\n".join(line for line in shown.splitlines() if not line.startswith("[#")) + "\n"
            == case["description"]
        )
