"""Configured generation matches ROOT's complete datasets after empirical-maximum resampling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roofit import numgenconfig, printing, rng

#: Native datasets include cases that trim already-generated rows after finding a larger maximum.
CASES = json.loads(
    (Path(__file__).parent / "data/roofit-generation-controls-6.40.json").read_text()
)


@pytest.fixture(autouse=True)
def _fresh_configuration(monkeypatch):
    monkeypatch.setattr(numgenconfig, "DEFAULT", [])
    monkeypatch.setattr(numgenconfig, "REGISTERED", [False])
    monkeypatch.setattr(printing, "PRECISION", [6])
    monkeypatch.setattr(rng, "_GENERATOR", [])


@pytest.mark.parametrize("case", CASES)
def test_configured_accept_reject_datasets_and_the_remaining_random_stream_match_root(case):
    numgenconfig.initialize_factory()
    x = ROOT.RooRealVar("x", "x", 0.0, 1.0)
    y = ROOT.RooRealVar("y", "y", -1.0, 1.0)
    variables = [x, y] if case["kind"] == 4 else [x]
    formula = {0: "1+0.2*@0", 1: "exp(4*@0)", 2: "0.1+(@0<0.1)", 4: "1+@0*@0+@1*@1"}[case["kind"]]
    model = ROOT.RooGenericPdf("p", "p", formula, ROOT.RooArgList(*variables))
    config = model.specialGeneratorConfig(True)
    config.method1D(False, False).setLabel("RooAcceptReject")
    config.method2D(False, False).setLabel("RooAcceptReject")
    for dimension in range(4):
        config.getConfigSection("RooAcceptReject").setRealValue(
            f"nTrial{dimension}D", case["trials"]
        )
    generator = ROOT.RooRandom.randomGenerator()
    generator.SetSeed(case["seed"])
    data = model.generate(ROOT.RooArgSet(*variables), case["count"], AutoBinned=False)
    found = [
        [data.get(i).getRealValue(v.GetName()) for v in variables] for i in range(data.numEntries())
    ]
    assert found == case["events"]
    assert [generator.Rndm() for _ in range(16)] == case["next"]
