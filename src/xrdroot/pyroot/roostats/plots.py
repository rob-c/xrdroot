"""RooStats' plots: the pictures of its intervals and tests, drawn with pyroot's graphics.

``LikelihoodIntervalPlot`` draws the profile likelihood ratio of one
parameter with the interval's cut and ends, or the contour of two; the
others draw posteriors, Markov chains, sampling distributions and Brazil
bands. They live here rather than in the engine because what they make is
``TLine``, ``TGraph``, ``TH1F`` and ``TLegend`` - pyroot's objects - on the
current pad.
"""

from __future__ import annotations

from typing import Any

from ...errors import UnsupportedFeatureError
from ...roofit.cmdargs import RooCmdArg
from ...roofit.collections import RooArgSet, as_list
from ...roofit.messages import ERROR, WARNING, log, log_plain
from ...roostats.intervals import Named

__all__ = ["LikelihoodIntervalPlot"]

#: ``kBlue``, ``kGreen``, ``kBlack``, ``kGray``: the colours RooStats' plots default to.
BLUE, GREEN, BLACK, GRAY = 600, 416, 1, 920


def _chi2_quantile(p: float, ndf: int) -> float:
    """``ROOT::Math::chisquared_quantile``: the exact inverse of the chi-square's cdf."""
    from ..core.rmath import chisquared_quantile

    return float(chisquared_quantile(p, ndf))


class LikelihoodIntervalPlot(Named):
    """The profile likelihood ratio of a :class:`~xrdroot.roostats.LikelihoodInterval`."""

    def __init__(self, interval: Any = None) -> None:
        super().__init__("")
        self._color = 0
        self._fill_style = 4050
        self._line_color = 0
        self._npoints = 0
        self._maximum = -1.0
        self._x = (0.0, -1.0)
        self._y = (0.0, -1.0)
        self._precision = -1.0
        self._interval: Any = None
        self._params: Any = None
        self._plotted: Any = None
        if interval is not None:
            self.SetLikelihoodInterval(interval)

    def SetLikelihoodInterval(self, interval: Any) -> None:
        self._interval = interval
        self._params = interval.GetParameters()

    def SetPlotParameters(self, params: Any) -> None:
        self._params = RooArgSet(as_list(params))

    def SetRange(self, *ends: float) -> None:
        if len(ends) == 2:
            self._x = (float(ends[0]), float(ends[1]))
        else:
            self._x, self._y = (float(ends[0]), float(ends[2])), (float(ends[1]), float(ends[3]))

    def SetPrecision(self, eps: float) -> None:
        self._precision = float(eps)

    def SetLineColor(self, color: Any) -> None:
        self._line_color = _colour(color)

    def SetFillStyle(self, style: int) -> None:
        self._fill_style = int(style)

    def SetContourColor(self, color: Any) -> None:
        self._color = _colour(color)

    def SetMaximum(self, maximum: float) -> None:
        self._maximum = float(maximum)

    def SetNPoints(self, npoints: int) -> None:
        self._npoints = int(npoints)

    def GetPlottedObject(self) -> Any:
        return self._plotted

    def Draw(self, options: Any = "") -> None:
        """``LikelihoodIntervalPlot::Draw``: the profile of one parameter, or a contour of two."""
        own = self._interval.GetParameters()
        self._drop_foreign(own)
        if len(self._params) > 2:
            log_plain(self, ERROR, "InputArguments", f"LikelihoodIntervalPlot::Draw("
                      f"{self._name}) ERROR: contours for more than 2 dimensions not "
                      "implemented!\n")  # fmt: skip
            return
        profile = self._interval.GetLikelihoodRatio()
        if len(self._params) != len(own):
            profile = profile.nll().createProfile(self._params)
        best = self._interval.GetBestFitParameters()
        if best is not None:
            self._params.assign(best)
            profile.getVal()
        option = str(options or "").lower()
        if len(self._params) == 1:
            self._draw_1d(profile, option)
        else:
            self._draw_2d(profile, option)

    def _drop_foreign(self, own: Any) -> None:
        """The parameters to plot that are not the interval's: dropped, said."""
        for par in [p for p in self._params if own.find(p.GetName()) is None]:
            log_plain(self, ERROR, "InputArguments", f"Parameter {par.GetName()}is not in the "
                      "list of LikelihoodInterval parameters  - do not use for plotting \n")
            self._params.remove(par)

    def _draw_1d(self, profile: Any, option: str) -> None:
        """The profile on a frame - or as a ``TF1`` for ``"tf1"`` - with the cut and the ends."""
        param = self._params[0]
        tf1 = "tf1" in option and "rooplot" not in option
        option = option.replace("rooplot", "").replace("tf1", "")
        npoints = self._npoints if self._npoints > 0 else 100
        low, high = self._interval.LowerLimit(param), self._interval.UpperLimit(param)
        var = profile.getVariables().find(param.GetName())
        self._color = self._color or BLUE
        self._line_color = self._line_color or GREEN
        if tf1:
            x1, x2 = self._tf1(profile, var, param, (low, high), npoints, option)
            lines = self._cut_lines(low, high, x1, x2)
            for line in (lines[2], lines[0], lines[1]):
                line.Draw()
            return
        x1, x2, frame = self._frame(profile, var, param, npoints)
        for line in self._cut_lines(low, high, x1, x2):
            frame.addObject(line)
        frame.Draw(option)

    def _cut_lines(self, low: float, high: float, x1: float, x2: float) -> list[Any]:
        """The interval's ends up to the cut, and the cut across."""
        from ..graphics.shapes import TLine

        level = 0.5 * _chi2_quantile(self._interval.ConfidenceLevel(), 1)
        lines = [TLine(low, 0.0, low, level), TLine(high, 0.0, high, level),
                 TLine(x1, level, x2, level)]  # fmt: skip
        for line in lines:
            line.SetLineColor(self._line_color)
        return lines

    def _frame(self, profile: Any, var: Any, param: Any, npoints: int) -> tuple[float, float, Any]:
        xmin, xmax = (param.getMin(), param.getMax()) if self._x[0] >= self._x[1] else self._x
        before = var.getBins()
        if self._npoints > 0:
            var.setBins(self._npoints)
        frame = var.frame(xmin, xmax, npoints)
        frame.SetTitle(self._title)
        frame.GetYaxis().SetTitle(f"- log #lambda({param.GetName()})")
        commands = [RooCmdArg("Precision", self._precision)] if self._precision > 0 else []
        profile.plotOn(frame, *commands, RooCmdArg("LineColor", self._color))
        frame.SetMaximum(self._maximum)
        frame.SetMinimum(0.0)
        var.setBins(before)
        self._plotted = frame
        return xmin, xmax, frame

    def _tf1(self, profile: Any, var: Any, param: Any, ends: Any, npoints: int, option: str) -> Any:
        """``asTF``: the profile as a function of the parameter, over twice the interval."""
        from ..core import TF1

        low, high = ends
        xmin, xmax = max(var.getMin(), 2 * low - high), min(var.getMax(), 2 * high - low)
        if self._x[0] < self._x[1]:
            xmin, xmax = self._x

        def at(x: Any, p: Any = None) -> float:
            var.setVal(float(x[0]))
            return float(profile.getVal())

        f1 = TF1(f"{self._name}_PLL_{var.GetName()}", at, xmin, xmax, 0)
        f1.SetNpx(npoints)
        f1.SetTitle(self._title)
        x1, x2 = xmin, xmax
        if self._maximum > 0 and self._x[0] >= self._x[1]:
            x0 = f1.GetX(0, xmin, xmax)
            if x1 < x0 < x2:
                x1, x2 = f1.GetX(self._maximum, xmin, x0), f1.GetX(self._maximum, x0, xmax)
                f1.SetMaximum(self._maximum)
        f1.SetRange(x1, x2)
        f1.SetLineColor(BLUE)
        f1.GetXaxis().SetTitle(var.GetName())
        f1.GetYaxis().SetTitle(f"- log #lambda({param.GetName()})")
        f1.Draw(option)
        self._plotted = f1.GetHistogram()
        return x1, x2


    def _draw_2d(self, profile: Any, option: str) -> None:
        """Minuit's contour of the two parameters at the level, and the best fit as a marker."""
        if "nominuit" in option or ("hist" in option and "nohist" not in option):
            raise UnsupportedFeatureError(
                "LikelihoodIntervalPlot draws a two-parameter interval from Minuit's contour; "
                "its scan into a histogram ('nominuit', 'hist') is not here yet"
            )
        option = option.replace("nohist", "").replace("minuit", "")
        px, py = self._params[0], self._params[1]
        best = self._interval.GetBestFitParameters()
        _at_best(profile, best)
        npoints = self._npoints if self._npoints > 0 else 40
        graph = self._contour(px, py, npoints)
        if "c" not in option:
            option += "L"
        if "same" not in option:
            self._frame_2d(px, py, npoints)
        self._contour_style(graph, option)
        self._best_marker(best, px, py)

    def _contour(self, px: Any, py: Any, npoints: int) -> Any:
        """``GetContourPoints``' contour as a closed ``TGraph``, said if short of points."""
        from ..core import TGraph

        graph = TGraph(npoints + 1)
        xs, ys = [0.0] * (npoints + 1), [0.0] * (npoints + 1)
        found = self._interval.GetContourPoints(px, py, xs, ys, npoints)
        if found < npoints:
            log(self, WARNING, "Eval", f"Warning - Less points calculated in contours np = "
                f"{found} / {npoints}")  # fmt: skip
        for index in range(found):
            graph.SetPoint(index, xs[index], ys[index])
        for index in range(found, npoints):  # ROOT's loop, which skips as the points shift
            graph.RemovePoint(index)
        graph.SetPoint(found, xs[0], ys[0])
        return graph

    def _frame_2d(self, px: Any, py: Any, npoints: int) -> None:
        """``_hist2D``: the axes, over the range set or the parameters'."""
        from ..core import TH2F

        title = self._title or f"Contour of {py.GetName()} vs {px.GetName()}"
        title = f"{title};{px.GetName()};{py.GetName()}"
        xmin, xmax = (px.getMin(), px.getMax()) if self._x[0] >= self._x[1] else self._x
        ymin, ymax = (py.getMin(), py.getMax()) if self._y[0] >= self._y[1] else self._y
        frame = TH2F("_hist2D", title, npoints, xmin, xmax, npoints, ymin, ymax)
        frame.GetXaxis().SetTitle(px.GetName())
        frame.GetYaxis().SetTitle(py.GetName())
        frame.SetStats(False)
        frame.SetFillStyle(self._fill_style)
        frame.SetMaximum(1)
        frame.Draw("AXIS")
        self._kept_frame = frame

    def _contour_style(self, graph: Any, option: str) -> None:
        if self._line_color:
            graph.SetLineColor(self._line_color)
        if self._color:
            graph.SetFillColor(self._color)
            option += "F"
        graph.SetLineWidth(3)
        if "same" in option:
            graph.SetFillStyle(self._fill_style)
        graph.Draw(option)
        graph.SetName(f"Graph_of_{self._interval.GetName()}")
        self._plotted = graph

    def _best_marker(self, best: Any, px: Any, py: Any) -> None:
        from ..core import TGraph

        if best is None:
            return
        marker = TGraph(1)
        marker.SetPoint(0, best.getRealValue(px.GetName()), best.getRealValue(py.GetName()))
        marker.SetMarkerStyle(33)
        if self._color:
            marker.SetMarkerColor(self._color + 4 if self._color != BLACK else GRAY)
        marker.Draw("P")


def _at_best(profile: Any, best: Any) -> None:
    """The profile's variables at the best fit, and the profile evaluated there."""
    for par in profile.getVariables():
        found = best.find(par.GetName()) if best is not None else None
        if found is not None:
            par.setVal(found.getVal())
    profile.getVal()


def _colour(color: Any) -> int:
    """A colour number, from a number or one of ``kRed``'s names as PyROOT takes it."""
    if isinstance(color, str):
        from ...roofit.names import named_constant

        return int(named_constant(color))
    return int(color)
