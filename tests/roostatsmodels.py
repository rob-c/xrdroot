"""The small models the RooStats tests share: counting with a constrained background."""

from __future__ import annotations

from typing import Any

import xrdroot.pyroot as ROOT


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
