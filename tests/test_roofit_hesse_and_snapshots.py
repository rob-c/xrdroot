"""HESSE's failure codes as Minuit2Minimizer scores them, and a workspace's parameter snapshots."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from xrdroot.roofit.fitting import minimizer
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.rng import RooRandom
from xrdroot.roofit.variables import RooRealVar
from xrdroot.roofit.workspace import RooWorkspace


def fmin(**flags: bool) -> Any:
    base = {"has_covariance": False, "hesse_failed": False, "has_posdef_covar": True}
    return SimpleNamespace(**{**base, **flags})


def test_hesse_scores_what_it_could_not_do_as_minuit2_does() -> None:
    """0 with a covariance; without one, 3 not positive definite, 1 failed, 4 otherwise - and a
    failed HESSE has none, whatever iminuit still shows."""
    codes = [
        minimizer._hesse_code(fmin(has_covariance=True)),
        minimizer._hesse_code(fmin(hesse_failed=True)),
        minimizer._hesse_code(fmin(has_posdef_covar=False)),
        minimizer._hesse_code(fmin()),
        minimizer._hesse_code(fmin(has_covariance=True, hesse_failed=True, has_posdef_covar=False)),
    ]
    assert codes == [0, 1, 3, 4, 3]


def test_a_hessian_a_parameter_does_not_move_fails_with_roots_status_302(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """rf510's background-only fit: a signal shape under a zero fraction. ROOT's 302, and error."""
    from xrdroot.roofit.pdfs.addpdf import RooAddPdf
    from xrdroot.roofit.pdfs.basic import RooPolynomial

    x = RooRealVar("x", "x", 0, 10)
    mean = RooRealVar("mean", "mean", 5, 0, 10)
    g = RooGaussian("g", "g", x, mean, RooRealVar("s", "s", 1))
    a1 = RooRealVar("a1", "a1", 0.2, -1.0, 1.0)
    frac = RooRealVar("f", "f", 0.5, 0, 1)
    model = RooAddPdf("model", "model", [RooPolynomial("p", "p", x, [a1]), g], [frac])
    RooRandom.randomGenerator().SetSeed(4357)
    data = model.generate([x], 1000)
    frac.setVal(1)
    frac.setConstant(True)
    capsys.readouterr()
    result = model.fitTo(data, PrintLevel=-1, Save=True)
    assert result.status() == 302
    assert "Error when calculating Hessian" in capsys.readouterr().out


def test_a_hesse_without_a_covariance_is_reported_and_scored(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``calculateHessErrors() Error when calculating Hessian``, and 100 times the code added."""
    x = RooRealVar("x", "x", -5, 5)
    m, s = RooRealVar("m", "m", 0, -2, 2), RooRealVar("s", "s", 1, 0.1, 3)
    g = RooGaussian("g", "g", x, m, s)
    RooRandom.randomGenerator().SetSeed(3)
    data = g.generate([x], 50)
    monkeypatch.setattr(minimizer, "_hesse_code", lambda found: 1)
    result = g.fitTo(data, PrintLevel=-1, Save=True)
    assert result.status() == 100
    assert (
        "[#0] ERROR:Minimization -- RooMinimizer::calculateHessErrors() Error when calculating "
        "Hessian\n" in capsys.readouterr().out
    )


def test_a_snapshot_keeps_values_errors_and_constness_and_prints_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``reference_fit = (m=1.5 +/- 0.2,s=2[C],t=3)``, in the workspace's order of nodes."""
    x = RooRealVar("x", "x", -5, 5)
    m, s = RooRealVar("m", "m", 1.5, -2, 2), RooRealVar("s", "s", 2, 0.1, 3)
    t = RooRealVar("t", "t", 3, 0, 5)
    w = RooWorkspace("w", "w")
    w.Import(RooGaussian("g", "g", x, m, s), Silence=True)
    w.Import(t, Silence=True)
    m, s = w.var("m"), w.var("s")  # the workspace's copies, which its snapshots are of
    m.setError(0.2)
    s.setConstant(True)
    w.saveSnapshot("reference_fit", "t,s,m")
    m.setVal(0.0)
    s.setConstant(False)
    assert w.loadSnapshot("reference_fit") is True
    assert (m.getVal(), m.getError(), s.isConstant()) == (1.5, 0.2, True)
    capsys.readouterr()
    w.Print()
    assert (
        "parameter snapshots\n-------------------\nreference_fit = (m=1.5 +/- 0.2,s=2[C],t=3)\n\n"
        in capsys.readouterr().out
    )
