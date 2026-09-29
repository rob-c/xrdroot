"""A simultaneous density's global observables and constraints, and a few corners near them.

Three channels of a counting model, two constrained backgrounds, and a
state with no channel: ``generateSimGlobal`` draws each channel's global
observables from its own density in the states' order, as ROOT 6.40 drew
them from seed 11, and a constraint two channels share is counted once.
"""

from __future__ import annotations

from typing import Any

import xrdroot.pyroot as ROOT


def _model() -> Any:
    w = ROOT.RooWorkspace("w")
    w.factory("Gaussian::ca(ga[0,-5,5], a[0,-5,5], 1)")
    w.factory("Gaussian::cb(gb[0,-5,5], b[0,-5,5], 1)")
    w.factory("Poisson::pa(na[0,100], sum::la(10, a))")
    w.factory("Poisson::pb(nb[0,100], sum::lb(10, b))")
    w.factory("PROD::ma(pa, ca)")
    w.factory("PROD::mb(pb, cb)")
    w.factory("PROD::mc(pa, ca, cb)")
    w.factory("SIMUL::sim(c[A,B,C,D], A=ma, B=mb, C=mc)")
    return w


def test_each_channels_global_observables_are_drawn_from_its_own_density_as_root() -> None:
    w = _model()
    ROOT.RooRandom.randomGenerator().SetSeed(11)
    made = w.pdf("sim").generateSimGlobal(ROOT.RooArgSet(w.var("ga"), w.var("gb")), 2)
    rows = [(made.get(i).getRealValue("ga"), made.get(i).getRealValue("gb")) for i in range(2)]
    assert made.GetName() == "gensimglobal"
    assert rows == [(-0.3856208539649484, -1.3882312434725463),
                    (-1.2103503081016243, 2.010606200663221)]  # fmt: skip
    counted = w.pdf("sim").generateSimGlobal(ROOT.RooArgSet(w.var("na")), 1)
    assert counted.get(0).getRealValue("na") == 9.0  # the channel without na skipped


def test_a_constraint_two_channels_share_is_counted_once() -> None:
    w = _model()
    params = [w.var("a"), w.var("b")]
    terms = w.pdf("sim").constraint_terms(frozenset({"na", "nb"}), params, False)
    assert [t.GetName() for t in terms] == ["ca", "cb"]


def test_a_sums_normalisation_for_a_fit_is_not_announced() -> None:
    x = ROOT.RooRealVar("x", "x", 0, 1)
    f = ROOT.RooFormulaVar("f", "f", "1+x", [x])
    s = ROOT.RooRealSumPdf("s", "s", [f], [ROOT.RooRealVar("k", "k", 1)], True)
    assert s.expected(frozenset({"x"}), fit=True) == 1.5


def test_the_chi_squares_library_is_said_once(capsys: Any, monkeypatch: Any) -> None:
    from xrdroot.roofit.fitting import fit

    monkeypatch.setattr(fit, "_SAID_LIBRARY", [False])
    x = ROOT.RooRealVar("x", "x", 0, 1)
    x.setBins(2)
    g = ROOT.RooGaussian("g", "g", x, ROOT.RooFit.RooConst(0.5), ROOT.RooFit.RooConst(1))
    data = g.generateBinned({x}, 10)
    g.createChi2(data)
    g.createChi2(data)
    out = capsys.readouterr().out
    assert out.count("using generic CPU library compiled with no vectorizations") == 1
