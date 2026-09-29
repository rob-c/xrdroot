"""``MCMCIntervalPlot``: a chain's posterior with its interval shaded, and the chain's walk.

The chains are those of the MCMCCalculator tests - IntervalExamples'
Gaussian, 300 steps, 20 of them burn-in - drawn in one parameter, as the
shortest interval or the central one, and in two, as the posterior's
contour at the cutoff.
"""

from __future__ import annotations

from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from test_roostats_bayes_mcmc import calculator
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.roostats import mcmcplot
from xrdroot.roostats.mcmc import kTailFraction


@pytest.fixture(autouse=True)
def _canvas() -> None:
    ROOT.TCanvas("c", "c", 400, 300)


def test_the_shortest_interval_is_the_posteriors_bins_above_the_cutoff_shaded() -> None:
    w, mc = calculator()
    interval = mc.GetInterval()
    plot = ROOT.RooStats.MCMCIntervalPlot(interval)
    plot.SetLineColor("kRed")
    plot.SetShadeColor(3)
    plot.SetLineWidth(2)
    plot.SetTitle("posterior")
    plot.Draw()
    copy, low, high = plot._kept
    mu = w["mu"]
    assert (low.GetX1(), high.GetX1()) == (interval.LowerLimit(mu), interval.UpperLimit(mu))
    assert (low.GetLineColor(), low.GetLineWidth(), copy.GetFillColor()) == (632, 2, 3)
    assert plot._posterior.GetMaximum() == 1.0
    assert plot._posterior.GetYaxis().GetTitle() == "Posterior for parameter mu"
    again = ROOT.RooStats.MCMCIntervalPlot(interval)
    again._posterior = posterior = interval.GetPosteriorHist()
    again.Draw("SAME")
    assert again._posterior is posterior


def test_a_central_interval_shades_the_bins_between_its_ends() -> None:
    w, mc = calculator()
    mc.SetLeftSideTailFraction(0.5)
    interval = mc.GetInterval()
    assert interval.GetIntervalType() == kTailFraction
    plot = ROOT.RooStats.MCMCIntervalPlot()
    plot.SetMCMCInterval(interval)
    plot.Draw()
    copy = plot._kept[0]
    low, high = interval.LowerLimit(w["mu"]), interval.UpperLimit(w["mu"])
    for i in range(1, copy.GetNbinsX() + 1):
        if not low <= copy.GetBinCenter(i) <= high:
            assert copy.GetBinContent(i) == 0


def test_two_parameters_are_the_posteriors_contour_at_the_cutoff(capsys: Any) -> None:
    _, mc = calculator("mu,sigma")
    interval = mc.GetInterval()
    plot = ROOT.RooStats.MCMCIntervalPlot(interval)
    plot.Draw()
    hist = plot._posterior
    assert (hist.ClassName(), hist.GetContourLevel(0)) == ("TH2F", interval.GetHistCutoff())
    plot.Draw("CONT2 SAME")
    assert plot._posterior is hist
    mc.SetLeftSideTailFraction(0.5)
    ROOT.RooStats.MCMCIntervalPlot(mc.GetInterval()).Draw()
    assert "DrawTailFractionInterval:  Sorry: 2-D plots not currently supported" in (
        capsys.readouterr().out)  # fmt: skip


def test_the_chains_walk_is_drawn_with_its_burn_in_or_without() -> None:
    w, mc = calculator("mu,sigma")
    interval = mc.GetInterval()
    plot = ROOT.RooStats.MCMCIntervalPlot(interval)
    plot.DrawChainScatter(w["mu"], w["sigma"])
    walk, early, first = plot._kept
    size = interval.GetChain().Size()
    assert (walk.GetN(), early.GetN(), first.GetN()) == (size - 20, 19, 1)
    assert walk.GetTitle() == "2-D Scatter Plot of Markov chain for mu, sigma"
    plot = ROOT.RooStats.MCMCIntervalPlot(interval)
    plot.SetShowBurnIn(False)
    plot.DrawChainScatter(w["mu"], w["sigma"])
    assert [g.GetN() for g in plot._kept] == [size, 1]


class _Chain:
    def Size(self) -> int:
        return 5


class _Interval:
    def __init__(self, axes: list[Any], burn: int) -> None:
        self.axes, self.burn = axes, burn

    def GetAxes(self) -> list[Any]:
        return self.axes

    def GetChain(self) -> _Chain:
        return _Chain()

    def GetNumBurnInSteps(self) -> int:
        return self.burn


def test_a_chain_all_burn_in_or_of_three_parameters_has_no_histogram(capsys: Any) -> None:
    x = ROOT.RooRealVar("x", "x", 0, 1)
    assert mcmcplot.posterior_hist(_Interval([x], 5)) is None
    assert "Number of burn-in steps (num steps to ignore) >= number of steps" in (
        capsys.readouterr().out)  # fmt: skip
    with pytest.raises(UnsupportedFeatureError, match="not 3"):
        mcmcplot.posterior_hist(_Interval([x, x, x], 0))
