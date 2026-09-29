"""The Asimov values of global observables, and the Asimov event of a counting model.

Each constraint term's global observable goes to its nuisance parameter's
server - divided by a Gamma's scale - and a term that cannot be read so is
skipped as RooStats says; a counting model's observables go to the values
their Poisson or Gaussian terms expect.
"""

from __future__ import annotations

from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roofit.messages import service
from xrdroot.roostats.asimovcount import counting_asimov_data
from xrdroot.roostats.asimovglobs import set_global_observables


@pytest.fixture(autouse=True)
def _fresh() -> Any:
    service().reset()
    yield
    service().reset()


def _model(w: Any, pdf: str, obs: str, nuis: str, globs: str) -> Any:
    mc = ROOT.RooStats.ModelConfig("mc", w)
    mc.SetPdf(pdf)
    mc.SetObservables(obs)
    mc.SetNuisanceParameters(nuis)
    mc.SetGlobalObservables(globs)
    return mc


def _globals(w: Any, mc: Any) -> None:
    set_global_observables(mc, ROOT.RooArgSet(list(mc.GetGlobalObservables())),
                           ROOT.RooArgSet(list(mc.GetNuisanceParameters())))  # fmt: skip


def test_each_supported_term_puts_its_global_observable_at_its_nuisance_parameter() -> None:
    """A Gaussian's mean, a Poisson's rate - its rounding switched off - a Gamma's rate over
    its scale ``theta``."""
    w = ROOT.RooWorkspace("w")
    w.factory("Gaussian::x_pdf(x[0,-5,5], mean[0], 1)")
    w.factory("Gaussian::g(g0[0,-5,5], a[0.5,-5,5], 1)")
    w.factory("Poisson::p(p0[10,0,50], prod::rate(t[10], k[1.2,0,3]))")
    w.factory("Gamma::gm(beta[1.5,0,10], sum::kg(y0[4,0,10], 1), theta_s[0.25], 0)")
    w.factory("PROD::model(x_pdf, g, p, gm)")
    mc = _model(w, "model", "x", "a,k,beta", "g0,p0,y0")
    _globals(w, mc)
    assert (w.var("g0").getVal(), w.var("p0").getVal(), w.var("y0").getVal()) == (
        0.5, 12.0, 6.0)  # fmt: skip
    assert w.pdf("p").getNoRounding()


def test_a_term_that_cannot_be_read_is_skipped_said_as_roostats_says(capsys: Any) -> None:
    w = ROOT.RooWorkspace("w")
    w.factory("Gaussian::x_pdf(x[0,-5,5], mean[0], 1)")
    w.factory("Gaussian::two(sum::gg(g1[0,-5,5], g2[0,-5,5]), a[0.5,-5,5], 1)")
    w.factory("Gaussian::none(a, b[0.3,-5,5], 1)")
    w.factory("Gaussian::pair(g3[0,-5,5], sum::ab(a, c[0.1,-5,5]), 1)")
    w.factory("Landau::odd(g4[0,-5,5], c, 1)")
    w.factory("Gaussian::far(prod::g5s(g5[0,-5,5], 1), a, 1)")
    w.factory("Gaussian::shared(g6[0,-5,5], prod::aa(a, 2), prod::a3(a, 0.5))")
    w.factory("Uniform::flat(c)")
    w.factory("PROD::model(x_pdf, two, none, pair, odd, far, shared, flat)")
    mc = _model(w, "model", "x", "a,b,c", "g1,g2,g3,g4,g5,g6")
    _globals(w, mc)
    out = capsys.readouterr().out
    where = "AsymptoticCalculator::MakeAsimovData"
    assert f"{where}: constraint term  two has multiple global observables" in out
    assert f"{where}: constraint term  none has no global observables - skip it" in out
    assert f"{where}:constraint term pair has multiple floating params" in out
    assert f"{where}:constraint term odd of type RooLandau is a non-supported type" in out
    assert "constraint term shared constraint term has more server depending on nuisance" in out
    assert f"{where}:constraint term far has no direct dependence on global observable" in out


def test_a_gamma_with_no_theta_has_a_scale_of_one_and_no_constraint_no_nuisance_pdf(
        capsys: Any) -> None:  # fmt: skip
    w = ROOT.RooWorkspace("w")
    w.factory("Gaussian::x_pdf(x[0,-5,5], mean[0], 1)")
    w.factory("Gamma::gm(beta[1.5,0,10], sum::kg(y0[4,0,10], 1), scale[0.25], 0)")
    w.factory("PROD::model(x_pdf, gm)")
    _globals(w, _model(w, "model", "x", "beta", "y0"))  # said on Generation, which INFO hides
    assert w.var("y0").getVal() == 1.5
    capsys.readouterr()
    _globals(w, _model(w, "x_pdf", "x", "mean", "y0"))
    out = capsys.readouterr().out
    assert "RooStatsUtils::MakeNuisancePdf - no constraints found" in out
    assert "model has nuisance parameters and global obs but no nuisance pdf" in out


def test_a_counting_models_asimov_event_is_at_each_terms_expectation(capsys: Any) -> None:
    """A Poisson's rate, unrounded, and a Gaussian's mean, in a product - the category's index
    in the name - and a multivariate Gaussian's means, said at print level three."""
    from xrdroot.roostats import asimov

    w = ROOT.RooWorkspace("w")
    w.factory("Poisson::p(n[0,50], prod::rate(t[10], k[1.25,0,3]))")
    w.factory("Gaussian::g(x[0,-5,5], m[0.5,-5,5], 1)")
    w.factory("PROD::both(p, g)")
    obs = ROOT.RooArgSet(w.var("n"), w.var("x"))
    cat = ROOT.RooCategory("c", "c")
    cat.defineType("A", 3)
    asimov._LEVEL[0] = 3
    made = counting_asimov_data(w.pdf("both"), obs, cat)
    assert made.GetName() == "CountingAsimovData3" and made.numEntries() == 1
    assert (w.var("n").getVal(), w.var("x").getVal()) == (12.5, 0.5)
    assert w.pdf("p").getNoRounding()
    capsys.readouterr()  # said on Generation, which the INFO stream leaves out
    x = [ROOT.RooRealVar(f"y{i}", "", 0, -5, 5) for i in range(2)]
    mu = [ROOT.RooRealVar(f"mu{i}", "", 0.1 * (i + 1), -5, 5) for i in range(2)]
    cov = ROOT.TMatrixDSym(2)
    cov[0, 0] = cov[1, 1] = 1.0
    mvg = ROOT.RooMultiVarGaussian("mvg", "mvg", x, mu, cov)
    assert counting_asimov_data(mvg, ROOT.RooArgSet(*x)).GetName() == "CountingAsimovData0"
    assert (x[0].getVal(), x[1].getVal()) == (0.1, 0.2)


def test_a_model_that_is_not_a_counting_model_has_no_asimov_event(capsys: Any) -> None:
    w = ROOT.RooWorkspace("w")
    w.factory("Landau::l(x[0,-5,5], 0, 1)")
    w.factory("Gaussian::g(y[0,-5,5], m[0.5,-5,5], 1)")
    w.factory("PROD::lg(l, g)")
    w.factory("Gaussian::xy(x, y, 1)")
    w.factory("Gaussian::free(y, m, s[1,0.1,2])")
    w.factory("Gaussian::fixed(y, 0, 1)")
    xy = ROOT.RooArgSet(w.var("x"), w.var("y"))
    assert counting_asimov_data(w.pdf("l"), ROOT.RooArgSet(w.var("x"))) is None
    assert counting_asimov_data(w.pdf("lg"), xy) is None
    assert counting_asimov_data(w.pdf("xy"), xy) is None
    assert counting_asimov_data(w.pdf("free"), ROOT.RooArgSet(w.var("y"))) is None
    assert counting_asimov_data(w.pdf("fixed"), ROOT.RooArgSet(w.var("y"))) is None
    out = capsys.readouterr().out
    assert ("A counting model pdf must be either a RooProdPdf or a RooPoisson or a "
            "RooGaussian") in out
    assert "Illegal term in counting model: the PDF l depends on the observables" in out
    assert "AsymptoticCalculator::SetObsExpected( RooGaussian ) : Has two observables ?? " in out
    assert "SetObsExpected( RooGaussian ) : Has two non-const arguments  " in out
    assert "SetObsExpected( RooGaussian ) : No observable?" in out
