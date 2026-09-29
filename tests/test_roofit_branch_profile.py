"""``RooProfileLL`` - ``nll.createProfile(poi)`` - and the ``RooMinimizer`` calls it makes,
against ROOT 6.40.

Every number and message was printed by ROOT through PyROOT for a Gaussian's
likelihood of 50 events drawn after ``SetSeed(4357)``: the profile's name,
its values - one MIGRAD each, a ``.`` of progress each, asked again or not -
the minimum it finds first, found again when a nuisance parameter is fixed,
its plotted copy, and the minimizer's constant-term switch and call count.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.fitting.minimizer import RooMinimizer
from xrdroot.roofit.fitting.nll import RooNLLVar
from xrdroot.roofit.fitting.profile import create_profile
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooRealVar

NAME = "RooEvaluatorWrapper_Profile[m]"
FIT = "[#1] INFO:Minimization -- [fitFCN] No discrete parameters, performing continuous " \
    "minimization only\n"  # fmt: skip


def _nll() -> tuple[Any, Any, Any, Any]:
    """The likelihood of a Gaussian's 50 events, its mean ``m`` and width ``s``."""
    generator().SetSeed(4357)
    x = RooRealVar("x", "x", -10, 10)
    m = RooRealVar("m", "m", 0, -5, 5)
    s = RooRealVar("s", "s", 2, 0.1, 10)
    g = RooGaussian("g", "g", x, m, s)
    data = g.generate(RooArgSet(x), 50)
    return g.createNLL(data), g, m, s


def test_a_profile_is_named_after_the_wrapper_and_minimises_first(capsys: Any) -> None:
    """Its name and title, its variables, and what it says making its minimizer and minimum."""
    nll, _, m, _ = _nll()
    prof = nll.createProfile(RooArgSet(m))
    assert (prof.GetName(), prof.GetTitle()) == (NAME, "Profile of RooEvaluatorWrapper")
    assert [v.GetName() for v in prof.getVariables()] == ["m", "s"]
    capsys.readouterr()
    m.setVal(0.5)
    assert prof.getVal() == 1.414842748782121
    assert capsys.readouterr().out == (
        f"[#1] INFO:Minimization -- RooProfileLL::evaluate({NAME}) Creating instance of MINUIT\n"
        "[#1] INFO:Fitting -- RooAddition::defaultErrorLevel(nll_g_over_g_Int[x]_gData) "
        "Summation contains a RooNLLVar, using its error level\n"
        f"[#1] INFO:Minimization -- RooProfileLL::evaluate({NAME}) determining minimum "
        "likelihood for current configurations w.r.t all observable\n"
        f"{FIT}[#1] INFO:Minimization -- RooProfileLL::evaluate({NAME}) minimum found at "
        f"(m=0.0481818)\n.{FIT}"
    )


def test_asking_again_fits_again_while_a_nuisance_parameter_is_free(capsys: Any) -> None:
    """MIGRAD's moves leave the value dirty: the same value, from another fit - a dot each."""
    nll, _, m, s = _nll()
    prof = nll.createProfile(RooArgSet(m))
    m.setVal(0.5)
    prof.getVal()
    capsys.readouterr()
    assert prof.getVal() == 1.414842748782121
    m.setVal(-0.3)
    assert prof.getVal() == 0.8498003585962266
    assert capsys.readouterr().out == f".{FIT}.{FIT}"
    assert [(v.GetName(), v.getVal()) for v in prof.bestFitParams()] == [("s", 1.8724494468091268)]
    assert [(v.GetName(), v.getVal()) for v in prof.bestFitObs()] == [("m", 0.04818179140205403)]
    s.setConstant(True)
    m.setVal(0.1)
    prof.getVal()
    capsys.readouterr()
    assert prof.getVal() == pytest.approx(0.019145764736293813, rel=1e-9)
    assert capsys.readouterr().out == ""  # nothing free: the value is kept


def test_fixing_a_nuisance_parameter_finds_the_minimum_again(capfd: Any) -> None:
    """From the last fit's values when not starting from the minimum; a nuisance parameter's
    change of constness is said, and the minimum found again - the parameter of interest's
    copy already there, as RooFit's ``addClone`` says."""
    nll, _, m, s = _nll()
    prof = nll.createProfile(RooArgSet(m))
    assert [p.GetName() for p in prof.getParameters(RooArgSet())] == ["m", "s"]
    for value in (0.5, -0.3):  # ROOT's sequence: two fits from the minimum first
        m.setVal(value)
        prof.getVal()
    assert prof.alwaysStartFromMin()
    prof.setAlwaysStartFromMin(False)
    assert not prof.alwaysStartFromMin()
    m.setVal(0.2)
    assert prof.getVal() == pytest.approx(0.16380763208245241, rel=1e-9)
    s.setConstant(True)
    m.setVal(0.1)
    capfd.readouterr()
    assert prof.getVal() == pytest.approx(0.019145764736293813, rel=1e-9)
    out = capfd.readouterr()
    assert (out.out + out.err) == (
        f"\n[#1] INFO:Minimization -- RooProfileLL::evaluate({NAME}) constant status of parameter "
        "s has changed from floating to fixed, recalculating absolute minimum\n"
        f"[#1] INFO:Minimization -- RooProfileLL::evaluate({NAME}) determining minimum "
        f"likelihood for current configurations w.r.t all observable\n{FIT}"
        "[#0] ERROR:InputArguments -- RooArgSet::checkForDup: ERROR argument with name m is "
        "already in this set\n"
        f"[#1] INFO:Minimization -- RooProfileLL::evaluate({NAME}) minimum found at "
        f"(m=0.0481824)\n.{FIT}"
    )


def test_a_profile_of_a_column_of_values_fits_once_for_each() -> None:
    """``compute`` with an array: one fit per value, in order - a scalar is one fit."""
    nll, _, m, _ = _nll()
    prof = nll.createProfile(RooArgSet(m))
    found = prof.compute({"m": np.array([0.5, -0.3])})
    assert found.tolist() == pytest.approx([1.414842748782121, 0.8498003585962266], rel=1e-9)
    assert prof.compute({"m": 0.5}) == pytest.approx(1.414842748782121, rel=1e-9)
    assert prof.compute({}) == pytest.approx(1.414842748782121, rel=1e-9)


def test_the_minimum_asked_for_first_makes_the_minimizer(capsys: Any) -> None:
    """``bestFitParams`` before any value: the minimizer made, the minimum found; ``minimizer``
    and ``nll`` hand back what the profile holds, and a profile's profile is the likelihood's."""
    nll, _, m, _ = _nll()
    prof = nll.createProfile(RooArgSet(m))
    assert [v.GetName() for v in prof.bestFitParams()] == ["s"]
    assert "Creating instance of MINUIT" in capsys.readouterr().out
    assert prof.minimizer() is prof.minimizer()
    assert prof.nll() is nll
    assert prof.createProfile(RooArgSet(m)).GetName() == NAME
    fresh = nll.createProfile(RooArgSet(m))
    assert fresh.minimizer() is not None


def test_a_likelihood_made_directly_names_its_profile_after_itself() -> None:
    """With no evaluator wrapper round it, the profile takes the likelihood's own name."""
    _, g, m, _ = _nll()
    data = g.generate(RooArgSet(g.x), 5)
    prof = create_profile(RooNLLVar(g, data, name="mine"), RooArgSet(m))
    assert (prof.GetName(), prof.GetTitle()) == ("mine_Profile[m]", "Profile of -log(likelihood)")


def test_a_plotted_profile_is_a_copy_that_puts_the_parameter_back() -> None:
    """RooFit draws a copy - its own minimum found again - named as ROOT names the curve."""
    nll, _, m, _ = _nll()
    prof = nll.createProfile(RooArgSet(m))
    m.setVal(0.1)
    prof.getVal()
    frame = m.frame(Range=(-0.5, 0.5))
    prof.plotOn(frame)
    curve = frame.getObject(0)
    assert (curve.GetName(), curve.GetN()) == (f"{NAME}_Norm[m]", 106)
    assert m.getVal() == 0.1


def test_the_constant_term_switch_says_each_change_and_each_repeat(capsys: Any) -> None:
    """``optimizeConst``: activating, already active, deactivating, not active - and nothing at
    all below print level -2; ``zeroEvalCount`` counts from nothing again."""
    nll, _, _, _ = _nll()
    minimizer = RooMinimizer(nll)
    capsys.readouterr()
    for flag in (1, 1, 0, 0):
        minimizer.optimizeConst(flag)
    minimizer.setPrintLevel(-3)
    minimizer.optimizeConst(1)
    prefix = "[#1] INFO:Minimization -- RooAbsMinimizerFcn::setOptimizeConst: "
    texts = ("activating const optimization", "const optimization already active",
             "deactivating const optimization", "const optimization wasn't active")  # fmt: skip
    assert capsys.readouterr().out == "".join(f"{prefix}{text}\n" for text in texts)
    minimizer.setPrintLevel(-1)
    minimizer.migrad()
    assert minimizer.evalCounter() == 31
    minimizer.zeroEvalCount()
    assert minimizer.evalCounter() == 0
