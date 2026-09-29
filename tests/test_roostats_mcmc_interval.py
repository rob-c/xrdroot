"""``MCMCInterval`` of chains made by hand, and ``MCMCCalculator``'s settings.

A chain of four points in ``x`` over four bins, weights 1, 3, 5 and 1: the
shortest interval gathers the heaviest bins until they hold the level -
strictly, or ROOT's way of stepping back - and the tail-fraction interval
walks in from each end. The chains that cannot give an interval - all
burn-in, a level out of range, an interval type unset - say so as ROOT does.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.roostats.markov import MarkovChain
from xrdroot.roostats.mcmc import MCMCCalculator, MCMCInterval, kTailFraction


def chain(points: list[tuple[float, float]]) -> tuple[Any, Any]:
    x = ROOT.RooRealVar("x", "x", 0, 4)
    x.setBins(4)
    made = MarkovChain()
    for value, weight in points:
        x.setVal(value)
        made.Add(ROOT.RooArgSet(x), 0.0, weight)
    return x, made


def interval(points: Any = None, cl: float = 0.7, **settings: Any) -> tuple[Any, Any]:
    x, made = chain(points or [(0.5, 1), (1.5, 3), (2.5, 5), (3.5, 1)])
    found = MCMCInterval("i", ROOT.RooArgSet(x), made)
    for name, value in settings.items():
        getattr(found, name)(value)
    found.SetConfidenceLevel(cl)
    return x, found


def test_the_shortest_interval_is_the_heaviest_bins_holding_the_level() -> None:
    x, found = interval()
    assert (found.LowerLimit(x), found.UpperLimit(x)) == (1.5, 2.5)
    assert (found.GetHistCutoff(), found.GetActualConfidenceLevel()) == (3.0, 0.8)
    x.setVal(2.5)
    inside = found.IsInInterval(ROOT.RooArgSet(x))
    x.setVal(0.5)
    assert (inside, found.IsInInterval(ROOT.RooArgSet(x))) == (True, False)
    found.SetConfidenceLevel(1.5)  # the bins made once; a level no bins reach
    assert found.GetActualConfidenceLevel() == 1.0
    assert (found.CheckParameters(ROOT.RooArgSet(x)), len(found.GetParameters())) == (True, 1)
    other = ROOT.RooRealVar("y", "y", -1, 1)
    assert (found.LowerLimit(other), found.UpperLimit(other)) == (-1.0, 1.0)


def test_a_loose_interval_steps_back_and_raises_its_cutoff() -> None:
    _, found = interval(SetHistStrict=False)
    assert (found.GetHistCutoff(), found.GetActualConfidenceLevel()) == (5.0, 0.5)
    even = [(0.5, 2), (1.5, 2), (2.5, 2), (3.5, 2)]
    _, loose = interval(even, 0.7, SetHistStrict=False)
    assert (loose.GetHistCutoff(), loose.GetActualConfidenceLevel()) == (3.0, 0.0)
    _, strict = interval(even, 0.5)  # ties with the cutoff all counted in
    assert (strict.GetHistCutoff(), strict.GetActualConfidenceLevel()) == (2.0, 1.0)


def test_a_chain_all_burn_in_has_no_interval(capsys: Any) -> None:
    x, found = interval(SetNumBurnInSteps=10)
    assert (found.GetHistCutoff(), found.LowerLimit(x), found.UpperLimit(x)) == (-1.0, 0.0, 4.0)
    out = capsys.readouterr().out
    assert "MCMCInterval::CreateDataHist: creation of histogram failed" in out
    assert "In MCMCInterval::LowerLimitByDataHist: couldn't determine cutoff." in out
    assert "Returning param.getMax()." in out
    x, tail = interval(SetNumBurnInSteps=10, SetIntervalType=kTailFraction,
                       SetLeftSideTailFraction=0.5)  # fmt: skip
    assert (tail.LowerLimit(x), tail.UpperLimit(x)) == (-math.inf, math.inf)
    assert "MCMCInterval::CreateVector: creation of vector failed" in capsys.readouterr().out


def test_the_tail_fraction_interval_walks_in_from_each_end() -> None:
    x, found = interval(cl=0.6, SetIntervalType=kTailFraction, SetLeftSideTailFraction=0.5)
    found.SetConfidenceLevel(0.6)  # the vector made once
    assert (found.LowerLimit(x), found.UpperLimit(x)) == (0.5, 3.5)
    assert found.GetActualConfidenceLevel() == pytest.approx(0.8)
    x.setVal(2.0)
    assert (found.IsInInterval(ROOT.RooArgSet(x)), found.GetIntervalType()) == (True, kTailFraction)
    x, everything = interval(cl=0.0, SetIntervalType=kTailFraction, SetLeftSideTailFraction=1.0)
    assert everything.LowerLimit(x) == 3.5  # the low tail takes every point


def test_what_an_interval_cannot_be_is_said(capsys: Any) -> None:
    interval(SetIntervalType=kTailFraction, SetLeftSideTailFraction=1.5)
    interval(SetIntervalType=99)
    x, made = chain([(0.5, 1)])
    y = ROOT.RooRealVar("y", "y", 0, 1)
    two = MCMCInterval("two", ROOT.RooArgSet(x, y), made)
    two.SetIntervalType(kTailFraction)
    two.SetLeftSideTailFraction(0.5)
    two.SetConfidenceLevel(0.9)
    two.SetAxes(ROOT.RooArgList(x))
    two.SetAxes(ROOT.RooArgList(y, x))
    assert [a.GetName() for a in two.GetAxes()] == ["y", "x"]
    out = capsys.readouterr().out
    for said in ("Fraction must be in the range [0, 1].  1.5is not allowed.",
                 "MCMCInterval::DetermineInterval(): Error: Interval type not set",
                 "Can only find a tail-fraction interval for 1-D intervals",
                 "number of variables in axes (1) doesn't match number of parameters (2)"):
        assert said in out
    assert MCMCInterval("empty").IsInInterval(ROOT.RooArgSet(x)) is False
    _, found = interval()
    assert found.GetPosteriorHist().GetName() == "MCMCposterior_hist"


def test_keys_and_sparse_histograms_are_refused() -> None:
    with pytest.raises(UnsupportedFeatureError, match="a keys density"):
        interval(SetUseKeys=True)
    with pytest.raises(UnsupportedFeatureError, match="a sparse histogram"):
        interval(SetUseSparseHist=True, SetEpsilon=0.1, SetDelta=0.1)


def test_the_calculator_passes_its_settings_to_the_chain_and_the_interval(capsys: Any) -> None:
    from test_roostats_bayes_mcmc import gaussian, helped

    w, _config, data = gaussian("mu,sigma")
    mc = MCMCCalculator()
    assert mc.GetInterval() is None
    mc.SetData(data)
    mc.SetPdf(w["normal"])
    mc.SetPriorPdf(None)
    mc.SetParameters(ROOT.RooArgSet(w["mu"], w["sigma"]))
    mc.SetNuisanceParameters(ROOT.RooArgSet())
    mc.SetTestSize(-1)
    assert mc.GetInterval() is None
    assert "Test size/Confidence level not set." in capsys.readouterr().out
    mc.SetTestSize(0.1)
    mc.SetChainParameters(ROOT.RooArgSet(w["mu"], w["sigma"]))
    mc.SetProposalFunction(helped(w).GetProposalFunction())
    mc.SetNumIters(30)
    mc.SetNumBurnInSteps(0)
    mc.SetNumBins(0)
    mc.SetAxes(ROOT.RooArgList(w["sigma"], w["mu"]))
    mc.SetUseKeys(False)
    mc.SetUseSparseHist(False)
    mc.SetEpsilon(0.1)
    mc.SetDelta(0.1)
    mc.SetIntervalType(0)
    mc.SetLeftSideTailFraction(2.0)
    assert "Fraction must be in the range [0, 1].  2is not allowed." in capsys.readouterr().out
    found = mc.GetInterval()
    assert (mc.Size(), mc.ConfidenceLevel(), found.GetNumBurnInSteps()) == (0.1, 0.9, 0)
    assert [a.GetName() for a in found.GetAxes()] == ["sigma", "mu"]
