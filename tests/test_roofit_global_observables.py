"""Global observables: a constraint's observed value, taken from the dataset or from the model."""

from __future__ import annotations

from typing import Any

import pytest

from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.pdfs.prodpdf import RooProdPdf
from xrdroot.roofit.rng import RooRandom
from xrdroot.roofit.variables import RooRealVar

START = "[#1] INFO:Minimization --  Including the following constraint terms in minimization: "


def constrained() -> tuple[Any, Any, Any, Any]:
    """rf613's model: ``mu`` constrained by ``mu_obs``, seed 5, 30 events, ``mu_obs`` 0.5 kept."""
    x = RooRealVar("x", "x", -10, 10)
    mu = RooRealVar("mu", "mu", 0.0, -10, 10)
    gauss = RooGaussian("gauss", "gauss", x, mu, RooRealVar("sigma", "sigma", 1.0, 0.1, 2.0))
    mu_obs = RooRealVar("mu_obs", "mu_obs", 1.0, -10, 10)
    mu_obs.setConstant()
    constraint = RooGaussian("constraint", "constraint", mu_obs, mu, 0.1)
    model = RooProdPdf("model", "model", [gauss, constraint])
    RooRandom.randomGenerator().SetSeed(5)
    data = model.generate([x], 30)
    mu_obs.setVal(0.5)
    data.setGlobalObservables([mu_obs])
    mu_obs.setVal(1.0)
    return model, data, mu, mu_obs


def fitted(model: Any, data: Any, **options: Any) -> tuple[float, float, Any]:
    params = model.getParameters(data.get())
    saved = params.snapshot()
    result = model.fitTo(data, PrintLevel=-1, Save=True, **options)
    mu = params.find("mu").getVal()
    params.assign(saved)
    return result.minNll(), mu, result


def said(out: str) -> list[str]:
    return [line for line in out.splitlines() if "Minimization --" in line and "fitFCN" not in line]


def test_the_datasets_global_observables_are_taken_without_being_named(capsys: Any) -> None:
    """ROOT 6.40: -log L 46.60723133250949 at mu 0.441466987362248, and what it says."""
    model, data, _mu, _obs = constrained()
    capsys.readouterr()
    nll, mu, _ = fitted(model, data)
    assert (nll, mu) == pytest.approx((46.60723133250949, 0.44146698736224826), rel=1e-9)
    assert said(capsys.readouterr().out) == [
        START + "(constraint)",
        "[#1] INFO:Minimization -- The following global observables have been automatically "
        "defined according to the dataset which also provides their values: (mu_obs)",
    ]


def test_named_global_observables_take_the_datasets_values(capsys: Any) -> None:
    """Named, and in the dataset: the same fit, said as ROOT says it."""
    model, data, _mu, mu_obs = constrained()
    capsys.readouterr()
    nll, mu, _ = fitted(model, data, GlobalObservables=mu_obs)
    assert (nll, mu) == pytest.approx((46.60723133250949, 0.44146698736224826), rel=1e-9)
    assert said(capsys.readouterr().out)[1] == (
        "[#1] INFO:Minimization -- The following global observables have been defined: "
        "(mu_obs), with the values of (mu_obs) obtained from the dataset and the other values "
        "from the model."
    )


def test_the_models_values_are_taken_when_the_source_says_so(capsys: Any) -> None:
    """``GlobalObservablesSource("model")``: mu_obs at 1, ROOT's 51.1585120001842 at 0.8824."""
    model, data, _mu, mu_obs = constrained()
    capsys.readouterr()
    nll, mu, _ = fitted(model, data, GlobalObservables=mu_obs, GlobalObservablesSource="model")
    assert (nll, mu) == pytest.approx((51.1585120001842, 0.882432707556216), rel=1e-9)
    assert said(capsys.readouterr().out)[1] == (
        "[#1] INFO:Minimization -- The following global observables have been defined and "
        "their values are taken from the model: (mu_obs)"
    )


def test_parameters_restored_as_rf613_restores_them_start_the_next_fit_where_the_first_did(
    capsys: Any,
) -> None:
    """rf613 fills sets of its own - ``getParameters(obs, set)``, ``snapshot(set)`` - and
    assigns the snapshot back between fits: values and errors, so the second fit starts, and
    ends, where the first did - as in ROOT, whose rf613 prints the same result twice."""
    from xrdroot.roofit.collections import RooArgSet

    model, data, mu, mu_obs = constrained()
    params, saved = RooArgSet(), RooArgSet()
    model.getParameters(data.get(), params)
    params.snapshot(saved)
    first = model.fitTo(data, GlobalObservables=mu_obs, PrintLevel=-1, Save=True)
    params.assign(saved)
    assert (mu.getVal(), mu.getError()) == (0.0, 0.0)
    second = model.fitTo(data, PrintLevel=-1, Save=True)
    assert (second.minNll(), second.edm()) == (first.minNll(), first.edm())
    capsys.readouterr()
