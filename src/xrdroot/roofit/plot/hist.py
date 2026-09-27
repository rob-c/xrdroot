"""``RooHist``: data on a plot - a point per bin, with Poisson or sum-of-weights error bars.

A dataset drawn on a frame is binned in the frame's bins and each bin
becomes a point at its centre, its bar the central 68% Poisson interval of
its count (``RooHistError::getPoissonInterval``: the chi-square quantiles
of ``2n`` and ``2n+2`` degrees of freedom) - or, for weighted data, the
square root of the sum of the squared weights. A bin wider than the plot's
nominal bin is scaled down to it, so a density is drawn, not a count.

It is an :class:`xrdroot.Graph`, a ``TGraphAsymmErrors``, so the canvas
draws it as any graph.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...graph import Graph
from ...stats import incomplete_gamma
from ..messages import WARNING, log

__all__ = ["RooHist", "poisson_interval"]

#: ``RooAbsData::ErrorType``.
POISSON, SUMW2, NONE, AUTO, EXPECTED = 0, 1, 2, 3, 4


def _gamma_quantile(p: float, shape: float) -> float:
    """The ``x`` where the regularised lower incomplete gamma of ``shape`` reaches ``p``."""
    low, high = 0.0, max(1.0, shape)
    while incomplete_gamma(shape, high) < p:
        high *= 2.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if incomplete_gamma(shape, mid) < p:
            low = mid
        else:
            high = mid
        if high - low <= 1e-15 * high:
            break
    return 0.5 * (low + high)


def poisson_interval(n: int, sigmas: float = 1.0) -> tuple[float, float]:
    """``RooHistError::getPoissonInterval``: the interval of ``sigmas`` round a count ``n``."""
    beta = math.erf(sigmas / math.sqrt(2.0))
    alpha = 1.0 - beta
    if n == 0:
        return 0.0, _gamma_quantile(1.0 - alpha, n + 1.0)
    return _gamma_quantile(0.5 * alpha, float(n)), _gamma_quantile(1.0 - 0.5 * alpha, n + 1.0)


class RooHist(Graph):
    """Points with asymmetric error bars, as RooFit draws data."""

    #: The events plotted - their values and weights - for counting those in a range.
    data_values: Any = None
    data_weights: Any = None

    def __init__(
        self,
        name: str = "",
        title: str = "",
        x: Any = (),
        y: Any = (),
        xlow: Any = (),
        xhigh: Any = (),
        ylow: Any = (),
        yhigh: Any = (),
    ) -> None:
        made = Graph.new(
            name,
            x,
            y,
            title=title,
            xerr=(np.asarray(xlow, float), np.asarray(xhigh, float)),
            yerr=(np.asarray(ylow, float), np.asarray(yhigh, float)),
        )
        super().__init__("TGraphAsymmErrors", made.members)
        self._core["TAttMarker"]["fMarkerStyle"] = 8
        #: How many events are in the plot's bins, and in all.
        self.entries = 0.0
        self.raw_entries = -1.0
        self.nominal_width = 0.0
        self.y_label = ""

    def fit_range_events(self) -> float:
        """``getFitRangeNEvt``: what curves normalise to."""
        return self.entries if self.raw_entries == -1 else self.raw_entries

    def events_between(self, low: float, high: float) -> float:
        """``getFitRangeNEvt(low, high)``: the events the data had between ``low`` and ``high``."""
        inside = (self.x >= low) & (self.x <= high)  # the bins whose centres are inside
        return float(np.sum(self.y[inside]))

    def fit_range_bin_width(self) -> float:
        return self.nominal_width

    def GetN(self) -> int:
        return len(self.x)

    def ClassName(self) -> str:
        return "RooHist"

    def GetName(self) -> str:
        return self.name

    def SetName(self, name: str) -> None:
        self._core["TNamed"]["fName"] = str(name)

    def errors(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        core = self.members
        return np.asarray(core["fEYlow"]), np.asarray(core["fEYhigh"])


def from_counts(
    name: str,
    edges: Any,
    counts: Any,
    sumw2: Any,
    etype: int,
    nominal: float,
    x_error: float = 1.0,
    scale: float = 1.0,
) -> RooHist:
    """``RooHist(const TH1&, ...)``: a point per bin of ``counts`` over ``edges``."""
    edges = np.asarray(edges, dtype=np.float64)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nominal = nominal or float((edges[-1] - edges[0]) / len(widths))
    factor = np.where(widths > 0, nominal / widths, 1.0) * scale
    counts = np.asarray(counts, dtype=np.float64)
    low, high = _bars(name, counts, np.asarray(sumw2, dtype=np.float64), etype)
    half = 0.5 * widths * x_error
    made = RooHist(name, "", centers, counts * factor, half, half, low * factor, high * factor)
    made.entries = float(np.sum(counts))
    made.nominal_width = nominal
    return made


def _bars(name: str, counts: Any, sumw2: Any, etype: int) -> tuple[Any, Any]:
    if etype == SUMW2:
        error = np.sqrt(sumw2)
        return error, error
    if etype != POISSON:
        return np.zeros_like(counts), np.zeros_like(counts)
    low, high = np.empty_like(counts), np.empty_like(counts)
    for i, n in enumerate(counts):
        low[i], high[i] = _poisson_bar(name, float(n))
    return low, high


def _poisson_bar(name: str, n: float) -> tuple[float, float]:
    """``RooHist::addBin``: the Poisson bar - interpolated between integers for a fractional
    count."""
    if n < 0:
        log(
            None,
            WARNING,
            "Plotting",
            f"RooHist::addBin({name}) WARNING: negative entry set to zero "
            "when Poisson error bars are requested",
        )
    whole = int(n)
    if abs(n - whole) > 1e-5:
        low1, high1 = poisson_interval(max(whole, 0))
        low2, high2 = poisson_interval(max(whole + 1, 0))
        log(
            None,
            WARNING,
            "Plotting",
            f"RooHist::addBin({name}) WARNING: non-integer bin entry {n:g} "
            "with Poisson errors, interpolating between Poisson errors of adjacent integer",
        )
        ym, yp = low1 + (n - whole) * (low2 - low1), high1 + (n - whole) * (high2 - high1)
    else:
        ym, yp = poisson_interval(max(whole, 0))
    return n - ym, yp - n
