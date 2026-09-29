"""``MCMCIntervalPlot``: a Markov chain's posterior histogram with its interval, and its walk.

The posterior is the chain's weights histogrammed in the parameter of
interest - its burn-in dropped - scaled to one at its highest bin, the
interval's bins filled grey beneath it, and a line at each end; the walk
is the chain's points in two parameters, its burn-in in pink and its first
point green, as ``MCMCIntervalPlot`` draws them.
"""

from __future__ import annotations

from typing import Any

from ...roofit.messages import ERROR, log
from ...roostats.intervals import Named

__all__ = ["MCMCIntervalPlot", "posterior_hist"]

#: ``kBlack``, ``kGray``, ``kGreen``, ``kPink``, ``kViolet``.
BLACK, GRAY, GREEN, PINK, VIOLET = 1, 920, 416, 900, 880


def posterior_hist(interval: Any) -> Any:
    """``GetPosteriorHist``: ``MCMCposterior_hist``, a ``TH1F`` - or ``TH2F`` - of the chain."""
    from ..core import TH1F, TH2F

    axes = list(interval.GetAxes())
    chain = interval.GetChain()
    start = interval.GetNumBurnInSteps()
    if start >= chain.Size():
        log(None, ERROR, "InputArguments", "MCMCInterval::CreateHist: creation of histogram "
            "failed: Number of burn-in steps (num steps to ignore) >= number of steps in Markov "
            "chain.")  # fmt: skip
        return None
    shape = [(a.numBins(), a.getMin(), a.getMax()) for a in axes]
    if len(axes) == 1:
        hist = TH1F("posterior", "MCMC Posterior Histogram", *shape[0])
    elif len(axes) == 2:
        hist = TH2F("posterior", "MCMC Posterior Histogram", *shape[0], *shape[1])
    else:
        from ...errors import UnsupportedFeatureError

        raise UnsupportedFeatureError(
            f"MCMCInterval's posterior histogram is drawn here in one or two parameters, not "
            f"{len(axes)}"
        )
    hist.SetDirectory(0)
    columns = [chain.values(a.GetName(), start) for a in axes]
    for point, weight in zip(zip(*columns), chain.weights(start)):
        hist.Fill(*point, weight)
    hist.GetXaxis().SetTitle(axes[0].GetName())
    if len(axes) > 1:
        hist.GetYaxis().SetTitle(axes[1].GetName())
    made = hist.Clone("MCMCposterior_hist")
    made.SetDirectory(0)
    return made


def _colour(value: Any) -> int:
    from ...roofit.names import named_constant

    return int(named_constant(value)) if isinstance(value, str) else int(value)


class MCMCIntervalPlot(Named):
    """The picture of an :class:`~xrdroot.roostats.MCMCInterval`."""

    def __init__(self, interval: Any = None) -> None:
        super().__init__("")
        self._line_color, self._shade_color, self._line_width = BLACK, GRAY, 1
        self._show_burn_in = True
        self._interval: Any = None
        self._posterior: Any = None
        self._kept: list[Any] = []
        if interval is not None:
            self.SetMCMCInterval(interval)

    def SetMCMCInterval(self, interval: Any) -> None:
        self._interval = interval

    def SetLineColor(self, color: Any) -> None:
        self._line_color = _colour(color)

    def SetShadeColor(self, color: Any) -> None:
        self._shade_color = _colour(color)

    def SetLineWidth(self, width: int) -> None:
        self._line_width = int(width)

    def SetShowBurnIn(self, flag: bool) -> None:
        self._show_burn_in = bool(flag)

    def Draw(self, options: Any = "") -> None:
        """The interval: the shortest's highest bins, or the tail fraction's middle."""
        from ...roostats.mcmc import kShortest

        shortest = self._interval.GetIntervalType() == kShortest
        dimension = len(list(self._interval.GetAxes()))
        if dimension == 2 and shortest:
            self._draw_2d(str(options or ""))
        elif dimension == 1:
            self._draw_1d(str(options or ""), shortest)
        else:
            where = "DrawHistInterval" if shortest else "DrawTailFractionInterval"
            log(None, ERROR, "InputArguments", f"MCMCIntervalPlot::{where}:  Sorry: "
                f"{dimension}-D plots not currently supported")  # fmt: skip

    def _draw_2d(self, options: str) -> None:
        """``DrawHistInterval`` of two parameters: the posterior's contour at the cutoff."""
        import array

        if self._posterior is None:
            self._posterior = self._interval.GetPosteriorHist()
        hist = self._posterior
        hist.SetTitle(self._title if self._title else "")
        hist.SetStats(False)
        if "CONT2" not in options:
            options += "CONT2"
        hist.SetContour(1, array.array("d", [self._interval.GetHistCutoff()]))
        hist.SetLineColor(self._line_color)
        hist.SetLineWidth(self._line_width)
        hist.Draw(options)

    def _draw_1d(self, options: str, shortest: bool) -> None:
        from ..graphics.shapes import TLine

        param = next(iter(self._interval.GetAxes()))
        upper, lower = self._interval.UpperLimit(param), self._interval.LowerLimit(param)
        if self._posterior is None:
            self._posterior = self._interval.GetPosteriorHist()
        hist = self._posterior
        hist.SetTitle(self._title if self._title else "")
        hist.GetYaxis().SetTitle(f"Posterior for parameter {param.GetName()}")
        hist.SetStats(False)
        copy = hist.Clone(f"{hist.GetTitle()}_copy")
        cutoff = self._interval.GetHistCutoff() if shortest else 0.0
        for i in range(1, copy.GetNbinsX() + 1):
            outside = (copy.GetBinContent(i) < cutoff) if shortest else (
                copy.GetBinCenter(i) < lower or copy.GetBinCenter(i) > upper)  # fmt: skip
            if outside:
                copy.SetBinContent(i, 0)
                copy.SetBinError(i, 0)
        top = hist.GetMaximumBin()
        hist.Scale(1 / hist.GetBinContent(top))
        copy.Scale(1 / copy.GetBinContent(top))
        copy.SetFillStyle(1001)
        copy.SetFillColor(self._shade_color)
        hist.Draw(options)
        copy.Draw("HIST SAME")
        lines = [TLine(lower, 0, lower, 1), TLine(upper, 0, upper, 1)]
        for line in lines:
            line.SetLineColor(self._line_color)
            line.SetLineWidth(self._line_width)
            line.Draw(options)
        self._kept = [copy, *lines]

    def DrawChainScatter(self, xVar: Any, yVar: Any) -> None:
        """The chain's walk in two parameters: its points in violet, its burn-in pink, its
        first point a green star."""
        import numpy as np

        from ..core import TGraph

        chain = self._interval.GetChain()
        burn = self._interval.GetNumBurnInSteps() if self._show_burn_in else 0
        xs = np.asarray(chain.values(xVar.GetName()), dtype=np.float64)
        ys = np.asarray(chain.values(yVar.GetName()), dtype=np.float64)
        walk = TGraph(len(xs) - burn, xs[burn:], ys[burn:])
        walk.SetTitle(self._title or f"2-D Scatter Plot of Markov chain for {xVar.GetName()}, "
                      f"{yVar.GetName()}")  # fmt: skip
        for axis, var in ((walk.GetXaxis(), xVar), (walk.GetYaxis(), yVar)):
            axis.Set(var.numBins(), var.getMin(), var.getMax())
            axis.SetTitle(var.GetName())
        walk.SetLineColor(GRAY)
        walk.SetMarkerStyle(6)
        walk.SetMarkerColor(VIOLET)
        walk.Draw("A,L,P,same")
        kept = [walk]
        if burn > 0:
            early = TGraph(burn - 1, xs[:burn], ys[:burn])
            early.SetLineColor(PINK)
            early.SetMarkerStyle(6)
            early.SetMarkerColor(PINK)
            early.Draw("L,P,same")
            kept.append(early)
        first = TGraph(1, xs[:1], ys[:1])
        first.SetLineColor(GREEN)
        first.SetMarkerStyle(3)
        first.SetMarkerSize(2)
        first.SetMarkerColor(GREEN)
        first.Draw("L,P,same")
        self._kept.extend([*kept, first])
