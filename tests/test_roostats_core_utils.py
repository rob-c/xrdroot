"""RooStats' utilities: significances, parameter sets, and a model's constraint terms.

The significances and p-values, what ``FactorizePdf`` splits a product, an
extended density and a simultaneous one into, and the names and classes
``MakeNuisancePdf``, ``MakeUnconstrainedPdf`` and ``StripConstraints`` hand
back - and the messages when they cannot - are what ROOT 6.40 gave for the
same models through PyROOT.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roofit.messages import service
from xrdroot.roostats import utils

RS = ROOT.RooStats
NO_OBSERVABLES = (
    "[#0] ERROR:InputArguments -- RooStats::MakeUnconstrainedPdf - invalid observable list "
    "passed (observables not found in original pdf) or invalid pdf passed (without "
    "observables)\n"
)


@pytest.fixture(autouse=True)
def _messages() -> Iterator[None]:
    """Every test starts from the message streams RooFit starts with."""
    service().reset()
    yield
    service().reset()


def workspace() -> Any:
    """``g(x)`` with two constraints, in products, extended and simultaneous densities."""
    w = ROOT.RooWorkspace("w")
    w.factory("Gaussian::g(x[0,-5,5],mu[1,-3,3],s[1,0.1,3])")
    w.factory("Gaussian::c(m0[1,-3,3],mu,0.5)")
    w.factory("Gaussian::c2(m1[1,-3,3],s,0.5)")
    w.factory("Gaussian::h(y[0,-5,5],mu,1)")
    w.factory("PROD::model(g,c,c2)")
    w.factory("PROD::onlyc(c,c2)")
    w.factory("PROD::gc(g,c)")
    w.factory("PROD::two(g,h,c)")
    w.factory("ExtendPdf::ext(model,n[10,0,100])")
    w.factory("ExtendPdf::extc(onlyc,n)")
    w.factory("SIMUL::sim(cat[a,b],a=model,b=gc)")
    w.factory("SIMUL::simc(cat,a=model,b=onlyc)")
    w.factory("SIMUL::sim3(cat3[a,b,c],a=model,b=gc)")
    return w


def names(items: Any) -> list[str]:
    return [one.GetName() for one in items]


def test_significances_and_p_values_are_the_normal_tails() -> None:
    """ROOT 6.40: ``inf``, ``-inf``, ``-0``, 6.3613409 for 1e-10 and 2.8665e-07 beyond 5."""
    assert RS.PValueToSignificance(0.0) == math.inf
    assert RS.PValueToSignificance(1.0) == -math.inf
    assert RS.PValueToSignificance(0.5) == 0.0
    assert RS.PValueToSignificance(1e-10) == pytest.approx(6.361340902404056, rel=1e-12)
    assert RS.PValueToSignificance(0.05) == pytest.approx(1.6448536269514729, rel=1e-12)
    assert RS.SignificanceToPValue(5.0) == pytest.approx(2.866515718791945e-07, rel=1e-12)
    assert RS.SignificanceToPValue(2.0) == 0.022750131948179216


def test_the_asimov_significance_with_and_without_an_uncertainty() -> None:
    """ROOT 6.40: 0.9839916447569484 for 10 over 100, 0.8772455271177728 with 5 on it."""
    assert RS.AsimovSignificance(10, 100) == 0.9839916447569484
    assert RS.AsimovSignificance(10, 100, 5) == 0.8772455271177728


def test_the_parameter_set_helpers_set_remove_and_fix(capsys: Any) -> None:
    """``SetParameters``, ``RemoveConstantParameters`` and ``SetAllConstant`` as ROOT's:
    ``SetAllConstant`` says whether it changed any (True, False, then True freeing)."""
    w = workspace()
    params = ROOT.RooArgSet(w.var("mu"), w.var("s"), w.var("x"))
    w.var("s").setConstant(True)
    RS.RemoveConstantParameters(params)
    assert names(params) == ["mu", "x"]
    assert RS.SetAllConstant(params) is True
    assert RS.SetAllConstant(params) is False
    assert w.var("mu").isConstant()
    assert RS.SetAllConstant([*params, ROOT.RooFit.RooConst(2.0)], False) is True
    values = ROOT.RooArgSet(ROOT.RooRealVar("mu", "mu", 2.5))
    RS.SetParameters(values, params)
    assert w.var("mu").getVal() == 2.5
    copied = utils.copy_of(params)
    assert names(copied) == ["mu", "x"] and copied.find("mu") is w.var("mu")


#: ROOT 6.40: each density's terms with ``x``, its constraints, and what is left of it
#: without them (name, title, class) - ``None`` when nothing is.
FACTORS = {
    "model": (["g"], ["c", "c2"], ("g_unconstrained_unconstrained", "g", "RooGaussian")),
    "ext": (["g"], ["c", "c2"], ("ext_unconstrained", "ext without constraints",
                                 "RooExtendPdf")),
    "sim": (["g"], ["c", "c2"], ("sim_unconstrained", "sim without constraints",
                                 "RooSimultaneous")),
    "g": (["g"], [], ("g_unconstrained", "g", "RooGaussian")),
    "c": ([], ["c"], None),
    "onlyc": ([], ["c", "c2"], None),
    "extc": ([], ["c", "c2"], None),
    "simc": (["g"], ["c", "c2"], None),
    "gc": (["g"], ["c"], ("g_unconstrained_unconstrained", "g", "RooGaussian")),
    "sim3": (["g"], ["c", "c2"], None),
}


@pytest.mark.parametrize("name", list(FACTORS))
def test_a_density_is_split_into_its_observables_terms_and_its_constraints(name: str) -> None:
    """``FactorizePdf`` and ``StripConstraints`` of each kind of density, as ROOT 6.40's."""
    w = workspace()
    obs = ROOT.RooArgSet(w.var("x"))
    terms, constraints = ROOT.RooArgList(), ROOT.RooArgList()
    RS.FactorizePdf(obs, w.pdf(name), terms, constraints)
    want_terms, want_constraints, stripped = FACTORS[name]
    assert (names(terms), names(constraints)) == (want_terms, want_constraints)
    found = RS.StripConstraints(w.pdf(name), obs)
    assert (found and (found.GetName(), found.GetTitle(), found.ClassName())) == stripped


def test_the_nuisance_density_is_the_product_of_the_constraints(capsys: Any) -> None:
    """``MakeNuisancePdf``: ``nuis`` of ``(c, c2)``; a warning and none for ``g`` alone."""
    w = workspace()
    obs = ROOT.RooArgSet(w.var("x"))
    made = RS.MakeNuisancePdf(w.pdf("sim"), obs, "nuis")
    assert (made.GetName(), made.ClassName(), names(made.pdfList())) == (
        "nuis", "RooProdPdf", ["c", "c2"])  # fmt: skip
    assert RS.MakeNuisancePdf(w.pdf("g"), obs, "nuis") is None
    assert capsys.readouterr().out == (
        "[#0] WARNING:Eval -- RooStatsUtils::MakeNuisancePdf - no constraints found on nuisance "
        "parameters in the input model\n"
    )


def test_the_unconstrained_density_is_named_as_asked_or_refused(capsys: Any) -> None:
    """``MakeUnconstrainedPdf``: its own name, or the one given; an error for constraints only."""
    w = workspace()
    obs = ROOT.RooArgSet(w.var("x"))
    assert RS.MakeUnconstrainedPdf(w.pdf("ext"), obs).GetName() == "ext_unconstrained"
    named = RS.MakeUnconstrainedPdf(w.pdf("ext"), obs, "named")
    assert (named.GetName(), named.GetTitle()) == ("named", "ext without constraints")
    assert RS.MakeUnconstrainedPdf(w.pdf("onlyc"), obs, "named") is None
    assert capsys.readouterr().out == NO_OBSERVABLES


def test_a_product_of_two_observables_terms_stays_a_product() -> None:
    """``two = g * h * c`` over ``x`` and ``y``: ``two_unconstrained`` of the clones of both."""
    w = workspace()
    found = RS.StripConstraints(w.pdf("two"), ROOT.RooArgSet(w.var("x"), w.var("y")))
    assert (found.GetName(), found.GetTitle(), found.ClassName()) == (
        "two_unconstrained", "two without constraints", "RooProdPdf")  # fmt: skip
    assert names(found.pdfList()) == ["g_unconstrained", "h_unconstrained"]


def test_a_model_config_hands_over_its_density_and_observables(capsys: Any) -> None:
    """ROOT 6.40: an error and none without them; ``nn`` and ``uu`` with them."""
    w = workspace()
    mc = RS.ModelConfig("mc", w)
    assert RS.MakeNuisancePdf(mc, "x") is None
    assert RS.MakeUnconstrainedPdf(mc, "x") is None
    assert capsys.readouterr().out == (
        "[#0] ERROR:InputArguments -- RooStatsUtils::MakeNuisancePdf - invalid input model: "
        "missing pdf and/or observables\n"
        "[#0] ERROR:InputArguments -- RooStatsUtils::MakeUnconstrainedPdf - invalid input "
        "model: missing pdf and/or observables\n"
    )
    mc.SetPdf("model")
    mc.SetObservables("x")
    assert names(RS.MakeNuisancePdf(mc, "nn").pdfList()) == ["c", "c2"]
    made = RS.MakeUnconstrainedPdf(mc, "uu")
    assert (made.GetName(), made.GetTitle(), made.ClassName()) == ("uu", "g", "RooGaussian")
    terms, constraints = ROOT.RooArgList(), ROOT.RooArgList()
    RS.FactorizePdf(mc, w.pdf("model"), terms, constraints)
    assert (names(terms), names(constraints)) == (["g"], ["c", "c2"])
