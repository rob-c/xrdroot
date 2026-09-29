"""``SamplingDistPlot`` and ``HypoTestPlot``: sampling distributions drawn as histograms.

Each distribution added is a ``TH1F`` of its values - weighted, if they are
- over its range padded by a bin and a half, in the next marker and colour;
a shaded copy keeps only the bins beyond a cut. ``Draw`` puts every
histogram and line on a ``RooPlot`` of a variable spanning them all.
``HypoTestPlot`` adds a result's null and alternate distributions, shaded
beyond the data's test statistic, and a line at it.
"""

from __future__ import annotations

import math
from typing import Any

from ...roofit.messages import WARNING, log
from ...roostats.intervals import Named

__all__ = ["HypoTestPlot", "SamplingDistPlot"]

#: ``kBlack``, ``kRed``, ``kBlue``.
BLACK, RED, BLUE = 1, 632, 600


class SamplingDistPlot(Named):
    """Histograms of sampling distributions, with lines and a legend, on one frame."""

    def __init__(self, nbins: int = 100, low: float = math.nan, high: float = math.nan) -> None:
        super().__init__("")
        self._bins = int(nbins)
        self._marker, self._color, self._fill_style = 20, 1, 3004
        self._items: list[Any] = []
        self._others: list[Any] = []
        self._hist: Any = None
        self._legend: Any = None
        self._log_x = self._log_y = False
        self._apply_style = True
        self._x = (float(low), float(high))
        self._y = (math.nan, math.nan)
        self._var_name = ""
        self._plot: Any = None

    # -- adding -------------------------------------------------------------------

    def AddSamplingDistribution(self, samplingDist: Any,
                                drawOptions: str = "NORMALIZE HIST") -> float:  # fmt: skip
        """A histogram of the distribution: its scale, one over its integral if normalised."""
        from ..core import TH1F

        values = list(samplingDist.GetSamplingDistribution())
        if not values:
            log(None, WARNING, "Plotting", "Empty sampling distribution given to plot. Skipping.")
            return 0.0
        weights = list(samplingDist.GetSampleWeights())
        finite = [v for v in values if not math.isinf(v)]
        low, high = (min(finite), max(finite)) if finite else (math.inf, -math.inf)
        if low >= high:
            log(None, WARNING, "Plotting", "Could not determine xmin and xmax of sampling "
                "distribution that was given to plot.")  # fmt: skip
            low, high = -1.0, 1.0
        width = (high - low) / self._bins
        xlow = self._x[0] if not math.isnan(self._x[0]) else low - 1.5 * width
        xup = self._x[1] if not math.isnan(self._x[1]) else high + 1.5 * width
        hist = TH1F(samplingDist.GetName(), samplingDist.GetTitle(), self._bins, xlow, xup)
        hist.SetDirectory(0)
        self._var_name = self._var_name or samplingDist.GetVarName()
        hist.GetXaxis().SetTitle(self._var_name)
        for i, value in enumerate(values):
            hist.Fill(value, weights[i]) if weights else hist.Fill(value)
        hist.Sumw2()
        options = str(drawOptions).upper()
        total = 1.0
        if "NORMALIZE" in options:
            total = float(hist.Integral("width"))
            hist.Scale(1.0 / total)
        hist.SetMarkerStyle(self._marker)
        hist.SetMarkerColor(self._color)
        hist.SetLineColor(self._color)
        self._marker += 1
        self._color += 1
        hist.SetStats(False)
        self._hist = hist
        self._items.append(hist)
        if self._legend is not None and samplingDist.GetTitle():
            self._legend.AddEntry(hist, samplingDist.GetTitle(), "L")
        return 1.0 / total

    def AddSamplingDistributionShaded(self, samplingDist: Any, minShaded: float, maxShaded: float,
                                      drawOptions: str = "NORMALIZE HIST") -> float:  # fmt: skip
        """The distribution, and a copy filled only between ``minShaded`` and ``maxShaded``."""
        if not list(samplingDist.GetSamplingDistribution()):
            log(None, WARNING, "Plotting", "Empty sampling distribution given to plot. Skipping.")
            return 0.0
        scale = self.AddSamplingDistribution(samplingDist, drawOptions)
        shaded = self._hist.Clone(f"{samplingDist.GetName()}_shaded")
        shaded.SetDirectory(0)
        shaded.SetFillStyle(self._fill_style)
        self._fill_style += 1
        shaded.SetLineWidth(1)
        for i in range(shaded.GetNbinsX()):  # from the underflow, and short of the last: ROOT's
            if shaded.GetBinCenter(i) < minShaded or shaded.GetBinCenter(i) > maxShaded:
                shaded.SetBinContent(i, 0)
        self._items.append(shaded)
        return scale

    def AddLine(self, x1: float, y1: float, x2: float, y2: float, title: Any = None) -> None:
        from ..graphics.shapes import TLine

        line = TLine(x1, y1, x2, y2)
        line.SetLineWidth(3)
        line.SetLineColor(BLACK)
        if self._legend is not None and title:
            self._legend.AddEntry(line, title, "L")
        self._others.append(line)

    def AddTH1(self, h: Any, drawOptions: str = "") -> None:
        if self._legend is not None and h.GetTitle():
            self._legend.AddEntry(h, h.GetTitle(), "L")
        copy = h.Clone()
        copy.SetDirectory(0)
        self._items.append(copy)

    def AddTF1(self, f: Any, title: Any = None, drawOptions: str = "SAME") -> None:
        if self._legend is not None and title:
            self._legend.AddEntry(f, title, "L")
        self._others.append(f.Clone())

    def SetAxisTitle(self, name: str) -> None:
        self._var_name = str(name)

    def SetLegend(self, legend: Any) -> None:
        self._legend = legend

    def SetApplyStyle(self, flag: bool) -> None:
        self._apply_style = bool(flag)

    def SetLogXaxis(self, flag: bool) -> None:
        self._log_x = bool(flag)

    def SetLogYaxis(self, flag: bool) -> None:
        self._log_y = bool(flag)

    def SetXRange(self, low: float, high: float) -> None:
        self._x = (float(low), float(high))

    def SetYRange(self, low: float, high: float) -> None:
        self._y = (float(low), float(high))

    def GetTH1F(self, samplingDist: Any = None) -> Any:
        if samplingDist is None:
            return self._hist
        return next((h for h in self._items if h.GetName() == samplingDist.GetName()), None)

    def GetPlot(self) -> Any:
        return self._plot

    # -- styles -------------------------------------------------------------------

    def _named(self, dist: Any) -> list[Any]:
        """The histograms of ``dist`` - the last one added, if none is named."""
        name = self._hist.GetName() if dist is None else dist.GetName()
        return [h for h in self._items if h.GetName() == name] if dist is not None else [self._hist]

    def _shaded(self, dist: Any) -> list[Any]:
        name = (self._hist.GetName() if dist is None else dist.GetName()) + "_shaded"
        return [h for h in self._items if h.GetName() == name]

    def SetLineColor(self, color: Any, sampleDist: Any = None) -> None:
        for hist in self._named(sampleDist):
            hist.SetLineColor(color)
        for hist in self._shaded(sampleDist):
            hist.SetLineColor(color)
            hist.SetFillColor(color)

    def SetLineWidth(self, width: Any, sampleDist: Any = None) -> None:
        for hist in self._named(sampleDist)[:1]:
            hist.SetLineWidth(width)

    def SetLineStyle(self, style: Any, sampleDist: Any = None) -> None:
        for hist in self._named(sampleDist)[:1]:
            hist.SetLineStyle(style)

    def SetMarkerStyle(self, style: Any, sampleDist: Any = None) -> None:
        for hist in self._named(sampleDist)[:1]:
            hist.SetMarkerStyle(style)

    def SetMarkerColor(self, color: Any, sampleDist: Any = None) -> None:
        for hist in self._named(sampleDist)[:1]:
            hist.SetMarkerColor(color)

    def SetMarkerSize(self, size: Any, sampleDist: Any = None) -> None:
        for hist in self._named(sampleDist)[:1]:
            hist.SetMarkerSize(size)

    # -- drawing ------------------------------------------------------------------

    def _extent(self) -> tuple[float, float, float]:
        """``GetAbsoluteInterval``: the histograms' x range, and their highest bin with a tenth
        more - of the last histogram higher than the running figure, as ROOT compares them."""
        low, high, top = math.inf, -math.inf, -math.inf
        for hist in self._items:
            axis = hist.GetXaxis()
            low, high = min(low, axis.GetXmin()), max(high, axis.GetXmax())
            if hist.GetMaximum() > top:
                top = hist.GetMaximum() + 0.1 * hist.GetMaximum()
        return low, high, top

    def ApplyDefaultStyle(self) -> None:
        """RooStats' plain style on ``gStyle``, if it is to be applied."""
        if not self._apply_style:
            return
        from ..graphics.style import gStyle

        for setter in ("SetFrameBorderMode", "SetCanvasBorderMode", "SetPadBorderMode",
                       "SetPadColor", "SetCanvasColor", "SetStatColor"):  # fmt: skip
            getattr(gStyle, setter)(0)
        gStyle.SetFrameFillStyle(0)  # and a paper size of 20 x 26, which no picture here has
        if self._legend is not None:
            self._legend.SetFillColor(0)
            self._legend.SetBorderSize(1)

    def Draw(self, options: Any = "") -> None:
        """Every histogram and line on a frame of a variable spanning them - log axes as set."""
        from ...roofit.variables import RooRealVar
        from ..graphics.pads import current

        self.ApplyDefaultStyle()
        low, high, top = self._extent()
        bottom = math.nan
        low = self._x[0] if not math.isnan(self._x[0]) else low
        high = self._x[1] if not math.isnan(self._x[1]) else high
        bottom = self._y[0] if not math.isnan(self._y[0]) else bottom
        top = self._y[1] if not math.isnan(self._y[1]) else top
        plot = RooRealVar("xaxis", self._var_name, low, low, high).frame()
        plot.SetTitle("")
        if not math.isnan(top):
            plot.SetMaximum(top)
        if not math.isnan(bottom):
            plot.SetMinimum(bottom)
        for hist in self._items:
            copy = hist.Clone()
            if not math.isnan(top):
                copy.SetMaximum(top)
            if not math.isnan(bottom):
                copy.SetMinimum(bottom)
            copy.SetDirectory(0)
            plot.addTH1(copy, "")  # the histogram's own option, which is empty
        for other in self._others:
            plot.addObject(other.Clone(), "")
        if self._legend is not None:
            plot.addObject(self._legend)
        self._log_axes()
        self._plot = plot
        plot.Draw()
        pad = current()
        if pad is not None:
            pad.SetLogx(self._log_x)
            pad.SetLogy(self._log_y)

    def _log_axes(self) -> None:
        """``gStyle``'s log axes as the plot's - said, if the style is not applied."""
        from ..graphics.style import gStyle

        for get, set_, flag, axis in (("GetOptLogx", "SetOptLogx", self._log_x, "x"),
                                      ("GetOptLogy", "SetOptLogy", self._log_y, "y")):  # fmt: skip
            if bool(getattr(gStyle, get)()) != flag:
                if not self._apply_style:
                    log(None, WARNING, "Plotting", f"gStyle will be changed to adjust "
                        f"SetOptLog{axis}(...)")  # fmt: skip
                getattr(gStyle, set_)(int(flag))


class HypoTestPlot(SamplingDistPlot):
    """A hypothesis test's null and alternate distributions, shaded beyond the data's value."""

    def __init__(self, result: Any, bins: int = 100, *args: Any) -> None:
        ends = [float(a) for a in args if isinstance(a, (int, float))]
        option = next((a for a in args if isinstance(a, str)), "NORMALIZE HIST")
        super().__init__(bins, *(ends[:2] if len(ends) >= 2 else ()))
        self._result = result
        self.ApplyResult(result, option)

    def ApplyResult(self, result: Any, opt: str = "NORMALIZE HIST") -> None:
        from ..graphics.legend import TLegend

        self._legend = TLegend(0.55, 0.95 - 0.3 * 0.66, 0.95, 0.95)
        alt, null = result.GetAltDistribution(), result.GetNullDistribution()
        if not result.HasTestStatisticData():
            for dist in (alt, null):
                if dist is not None:
                    self.AddSamplingDistribution(dist, opt)
        else:
            value = result.GetTestStatisticData()
            ends = (value, math.inf) if result.GetPValueIsRightTail() else (-math.inf, value)
            for dist in (alt, null):
                if dist is not None:
                    self.AddSamplingDistributionShaded(dist, *ends, opt)
            top = self._extent()[2]
            self.AddLine(value, 0, value, top * 0.66, "test statistic data")
        self.ApplyDefaultStyle()

    def ApplyDefaultStyle(self) -> None:
        """The alternate in blue, the null in red, both two pixels wide."""
        result = getattr(self, "_result", None)
        if result is None:
            return
        for dist, color in ((result.GetAltDistribution(), BLUE),
                            (result.GetNullDistribution(), RED)):  # fmt: skip
            if dist is not None:
                self.SetLineWidth(2, dist)
                self.SetLineColor(color, dist)
