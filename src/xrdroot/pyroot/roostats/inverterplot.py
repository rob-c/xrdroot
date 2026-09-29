"""``HypoTestInverterPlot``: a scan's ``CLs`` against the parameter, observed and expected.

``MakePlot`` is the observed curve - a ``TGraphErrors`` of each point's
``CLs`` (or ``CLs+b``, ``CLb``) and its error; ``MakeExpectedPlot`` the
"Brazil" bands - a ``TMultiGraph`` of the median expected curve and the one
and two sigma bands about it, green and yellow. ``Draw`` puts them on the
pad with a red line at the test size and a legend, as RooStats draws them.
"""

from __future__ import annotations

import math
import sys
from typing import Any

from ...roostats.intervals import Named
from ...roostats.inverterlimits import MAX_SIGMA, quantile, sorted_order

__all__ = ["HypoTestInverterPlot"]

#: ``kRed``, ``kGreen``, ``kYellow``, ``kBlue``.
RED, GREEN, YELLOW, BLUE = 632, 416, 400, 600

_MODES = (("CLB", "CLb"), ("CLS+B", "CLsplusb"), ("CLSPLUSB", "CLsplusb"), ("CLS", "CLs"))


def _mode(option: str) -> str:
    upper = str(option or "").upper()
    return next((mode for word, mode in _MODES if word in upper), "")


def _observed_points(results: Any, mode: str) -> list[tuple[float, float, float]]:
    """Each point's level and error, in order - those that are not a probability left out."""
    kept = []
    for i in sorted_order(results._x):
        value = results.GetYValue(i) if not mode else getattr(results, mode)(i)
        error = results.GetYError(i) if not mode else getattr(results, f"{mode}Error")(i)
        if value < 0 or not math.isfinite(value):
            sys.stderr.write(f"Warning in <HypoTestInverterPlot::MakePlot>: Got a confidence level "
                             f"of {value:f} at x={results.GetXValue(i):f} (failed fit?). Skipping "
                             "this point.\n")  # fmt: skip
            continue
        kept.append((results.GetXValue(i), value, error))
    return kept


def make_plot(results: Any, option: str = "") -> Any:
    """``MakePlot``: the observed ``CLs`` - or what ``option`` names - against the parameter,
    the points it could not compute skipped with a warning."""
    from ..core.graphs import TGraphErrors

    mode = _mode(option)
    kept = _observed_points(results, mode)
    graph: Any = TGraphErrors(len(kept))
    for i, (x, y, e) in enumerate(kept):
        graph.SetPoint(i, x, y)
        graph.SetPointError(i, 0.0, e)
    name = {"CLb": "CLb", "CLsplusb": "CLs+b"}.get(mode, "CLs" if mode or results._use_cls
                                                    else "CLs+b")  # fmt: skip
    graph.SetName(f"{name}_observed")
    graph.SetTitle(f"Observed {name}")
    graph.SetMarkerStyle(20)
    graph.SetLineWidth(2)
    return graph


def _band_title(name: str, nsig: float) -> str:
    if nsig - int(nsig) < 0.01:
        return f"Expected {name} #pm {int(nsig)} #sigma"
    return f"Expected {name} #pm {nsig:3.1f} #sigma"


def _quantiles(values: list[float], asymptotic: bool, nsig1: float, nsig2: float) -> list[float]:
    """The expected p-values at -nsig2, -nsig1, the median, nsig1 and nsig2."""
    from ...function.analytic import gaussian_cdf

    sigmas = (-nsig2, -nsig1, 0.0, nsig1, nsig2)
    if asymptotic:
        step = 2 * MAX_SIGMA / (len(values) - 1)
        return [values[math.floor((s + MAX_SIGMA) / step + 0.5)] for s in sigmas]
    return [quantile(values, 0.5 if s == 0.0 else gaussian_cdf(s)) for s in sigmas]


def _fill_bands(results: Any, median: Any, bands: list[Any], nsig: tuple[float, float]) -> None:
    """Each point with expected p-values: the median, and the bands' ends about it."""
    asymptotic = results.GetNullTestStatDist(0) is None and results.GetAltTestStatDist(0) is None
    at = 0
    for i in sorted_order(results._x):
        dist = results.GetExpectedPValueDist(i)
        if dist is None:
            continue
        q = _quantiles(dist.GetSamplingDistribution(), asymptotic, *nsig)
        x = results.GetXValue(i)
        median.SetPoint(at, x, q[2])
        for band, (low, high) in zip(bands, ((q[1], q[3]), (q[0], q[4]))):
            _band_point(band, at, x, (low, q[2], high))
        at += 1


def _band_point(band: Any, at: int, x: float, q: tuple[float, float, float]) -> None:
    if band is not None:
        band.SetPoint(at, x, q[1])
        band.SetPointEYlow(at, q[1] - q[0])
        band.SetPointEYhigh(at, q[2] - q[1])


def make_expected_plot(plot: Any, results: Any, nsig1: float = 1.0, nsig2: float = 2.0) -> Any:
    """``MakeExpectedPlot``: the median expected curve, dashed, and its bands."""
    from ..core.graphs import TGraph, TGraphAsymmErrors
    from ..core.stacks import TMultiGraph

    first, second = nsig1 > 0, nsig2 > nsig1
    nsig1, nsig2 = abs(nsig1), abs(nsig2)
    name = "CLs" if results._use_cls else "CLs+b"
    median: Any = TGraph()
    median.SetTitle(f"Expected {name} - Median")
    bands: list[Any] = [TGraphAsymmErrors() if on else None for on in (first, second)]
    for band, nsig in zip(bands, (nsig1, nsig2)):
        if band is not None:
            band.SetTitle(_band_title(name, nsig))
    _fill_bands(results, median, bands, (nsig1, nsig2))
    made: Any = TMultiGraph(f"{plot.GetName()}_expected", f"Expected {plot.GetTitle()}")
    for band, colour in ((bands[1], YELLOW), (bands[0], GREEN)):
        if band is not None:
            band.SetFillColor(colour)
            made.Add(band, "3")
    median.SetLineStyle(2)
    median.SetLineWidth(2)
    made.Add(median, "L")
    return made


class HypoTestInverterPlot(Named):
    """The picture of a :class:`~xrdroot.roostats.HypoTestInverterResult`."""

    def __init__(self, *args: Any) -> None:
        results = args[-1]
        name, title = (args[0], args[1]) if len(args) == 3 else (results.GetName(),
                                                                 results.GetTitle())  # fmt: skip
        super().__init__(name, title)
        self._results = results
        self._kept: list[Any] = []

    def MakePlot(self, opt: str = "") -> Any:
        return make_plot(self._results, opt)

    def MakeExpectedPlot(self, sig1: float = 1.0, sig2: float = 2.0) -> Any:
        return make_expected_plot(self, self._results, sig1, sig2)

    def MakeTestStatPlot(self, index: int, type: int = 0, nbins: int = 100) -> Any:
        """The test statistic's distributions at a point: both (0), the null's (1), the
        alternate's (2)."""
        from .sdplot import HypoTestPlot, SamplingDistPlot

        results = self._results
        if type == 0:
            found = results._results[index] if 0 <= index < len(results._results) else None
            return HypoTestPlot(found, nbins) if found is not None else None
        dist = {1: results.GetSignalAndBackgroundTestStatDist,
                2: results.GetBackgroundTestStatDist}.get(type, lambda i: None)(index)  # fmt: skip
        if dist is None:
            return None
        made = SamplingDistPlot(nbins)
        made.AddSamplingDistribution(dist)
        return made

    def Draw(self, opt: str = "") -> None:
        """The observed curve - axes too, unless ``SAME`` - the expected bands, the size's red
        line, ``CLb`` and the other CL if asked, and a legend."""
        from .inverterdraw import draw

        self._kept = draw(self, str(opt or "").upper())


def draw_limit_plot(it: Any, target: float, limit: float, error: float, expo: Any) -> None:
    """``RunLimit``'s picture: the points near the target, the fit, the target and the limit."""
    from ..graphics.shapes import TLine

    graph = it._limit_plot
    graph.Sort()
    graph.SetLineWidth(2)
    xmin, xmax = it._var.getMin(), it._var.getMax()
    xs, ys = list(graph.GetX()), list(graph.GetY())
    for x, y in zip(xs, ys):
        if 0.6 * target <= y <= 1.4 * target:
            xmin, xmax = min(x, xmin), max(x, xmax)
    graph.GetXaxis().SetRangeUser(xmin, xmax)
    graph.GetYaxis().SetRangeUser(0.5 * target, 1.5 * target)
    graph.Draw("AP")
    expo.Draw("SAME")  # the picture is made only by the fit, which the function is
    line = TLine(xs[0], target, xs[-1], target)
    line.SetLineColor(RED)
    line.SetLineWidth(2)
    line.Draw()
    kept = [line, line.DrawLine(limit, 0, limit, ys[0])]
    line.SetLineWidth(1)
    line.SetLineStyle(2)
    kept += [line.DrawLine(limit - error, 0, limit - error, ys[0]),
             line.DrawLine(limit + error, 0, limit + error, ys[0])]  # fmt: skip
    it._drawn = kept
