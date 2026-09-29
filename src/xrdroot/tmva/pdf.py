"""``TMVA::PDF``: a probability density made from a histogram, smoothed and splined.

TMVA's projective likelihood, its Gaussianisation and the classifiers' own
output PDFs all turn a histogram into a density the same way: the histogram
is smoothed ``NSmooth`` times with ``TH1::Smooth``, a spline is drawn through
its bin centres - ``Spline0`` the histogram itself, ``Spline1`` straight
lines, ``Spline2`` TMVA's quadratic through each three points - and the
spline is sampled into a histogram of 10000 bins, which is what is read back,
by linear interpolation between those fine bins. Every step is TMVA's, in
its single precision (the histograms are ``TH1F``), so a density here is the
density TMVA's weight file would hold for the same events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..hist import Histogram
from ..stats import smooth_array
from . import hists
from .log import Logger
from .options import Options
from .xmlfile import Node, number

__all__ = ["PDF", "PDFSettings", "pdf_from_xml", "settings"]

#: ``PDF::fgNbin_PdfHist``: the bins the spline is sampled into.
NBIN_PDF_HIST = 10000
#: ``PDF::fgEpsilon``: the least a density is.
EPSILON = 1.0e-12
#: The interpolations TMVA knows, by the names its options give them.
METHODS = ("Spline0", "Spline1", "Spline2", "Spline3", "Spline5", "KDE")


@dataclass
class PDFSettings:
    """A PDF's options: ``NSmooth``, ``MinNSmooth``, ``MaxNSmooth``, ``NAvEvtPerBin``..."""

    nsmooth: int = 0
    min_nsmooth: int = -1
    max_nsmooth: int = -1
    avg_per_bin: int = 50
    nbins: int = 0
    interpolation: str = "Spline2"
    check_hist: bool = False

    def smoothing(self) -> tuple[int, int]:
        """``ProcessOptions``: the least and most smoothing passes, ``NSmooth`` unless both set."""
        if self.max_nsmooth < 0 or self.min_nsmooth < 0:
            return max(self.nsmooth, 0), max(self.nsmooth, 0)
        return self.min_nsmooth, self.max_nsmooth

    def hist_bins(self, events: int) -> int:
        """``GetHistNBins``: the reference histogram's bins for ``events`` events."""
        factor = 5 if self.interpolation == "KDE" else 1
        if self.nbins > 0:
            return self.nbins * factor
        return events // self.avg_per_bin * factor


def settings(options: Options, suffix: str = "", base: PDFSettings | None = None) -> PDFSettings:
    """The PDF options of ``options`` with ``suffix`` - ``"MVAPdf"``, ``"Sig[2]"`` - over ``base``.

    A suffix ending in ``[i]`` is an element of an array option, as TMVA
    declares ``NSmoothSig[0]``; any option not given takes ``base``'s value.
    """
    base = base or PDFSettings()
    name, _, index = suffix.partition("[")

    def get(option: str, default: Any) -> Any:
        if index:
            return options.array(option + name, int(index.rstrip("]")) + 1, default)[-1]
        if isinstance(default, bool):
            return options.flag(option + name, default)
        if isinstance(default, int):
            return options.integer(option + name, default)
        return options.text_of(option + name, default)

    return PDFSettings(
        get("NSmooth", base.nsmooth),
        get("MinNSmooth", base.min_nsmooth),
        get("MaxNSmooth", base.max_nsmooth),
        get("NAvEvtPerBin", base.avg_per_bin),
        get("Nbins", base.nbins),
        get("PDFInterpol", base.interpolation),
        get("CheckHist", base.check_hist),
    )


def _quadrax(x: Any, x1: Any, x2: Any, x3: Any, y1: Any, y2: Any, y3: Any) -> Any:
    """``TSpline2::Quadrax``: the parabola through three points, in single precision."""
    f = np.float32
    x = f(x)
    xs, ys = (f(x1), f(x2), f(x3)), (f(y1), f(y2), f(y3))
    a, b, c = _linear(xs, ys), _square(xs, ys), _product(xs, ys)
    x1, x2, x3 = xs
    denom = (x2 - x3) * (x3 - x1) * (x1 - x2)
    with np.errstate(all="ignore"):
        value = (-a * x * x + b * x - c) / denom
    return np.where(denom != 0, value, f(0)).astype(np.float64)


def _linear(xs: tuple[Any, Any, Any], ys: tuple[Any, Any, Any]) -> Any:
    """The parabola's coefficient ``a``, as ``Quadrax`` works it out, before the division."""
    (x1, x2, x3), (y1, y2, y3) = xs, ys
    return y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2)


def _square(xs: tuple[Any, Any, Any], ys: tuple[Any, Any, Any]) -> Any:
    """The parabola's coefficient ``b``, before the division."""
    (x1, x2, x3), (y1, y2, y3) = xs, ys
    return y1 * (x2 * x2 - x3 * x3) + y2 * (x3 * x3 - x1 * x1) + y3 * (x1 * x1 - x2 * x2)


def _product(xs: tuple[Any, Any, Any], ys: tuple[Any, Any, Any]) -> Any:
    """The parabola's coefficient ``c``, before the division."""
    (x1, x2, x3), (y1, y2, y3) = xs, ys
    return y1 * (x2 - x3) * x2 * x3 + y2 * (x3 - x1) * x3 * x1 + y3 * (x1 - x2) * x1 * x2


def _bin_of(xs: Any, x: Any) -> Any:
    """The last point at or below each ``x``, clamped to the points: ``TSpline::Eval``'s search."""
    return np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 1)


def spline2(xs: Any, ys: Any, x: Any) -> Any:
    """``TMVA::TSpline2::Eval``: the mean of the two parabolas through the points around ``x``."""
    n = len(xs)
    if n < 3:
        return ys[_bin_of(xs, x)]
    i = _bin_of(xs, x)
    lo = np.clip(i - 1, 0, n - 3)
    first = _quadrax(x, xs[lo], xs[lo + 1], xs[lo + 2], ys[lo], ys[lo + 1], ys[lo + 2])
    hi = np.clip(i, 0, n - 3)
    second = _quadrax(x, xs[hi], xs[hi + 1], xs[hi + 2], ys[hi], ys[hi + 1], ys[hi + 2])
    edge = _quadrax(x, xs[n - 3], xs[n - 2], xs[n - 1], ys[n - 3], ys[n - 2], ys[n - 1])
    start = _quadrax(x, xs[0], xs[1], xs[2], ys[0], ys[1], ys[2])
    middle = (first + second) * 0.5
    return np.where(i == 0, start, np.where(i >= n - 2, edge, middle))


def spline1(xs: Any, ys: Any, x: Any) -> Any:
    """``TMVA::TSpline1::Eval``: the straight line through the two points around ``x``."""
    n = len(xs)
    i = _bin_of(xs, x)
    up = ((x > xs[i]) & (i != n - 1)) | (i == 0)
    other = np.where(up, i + 1, i - 1)
    return ys[i] + (x - xs[i]) * (ys[i] - ys[other]) / (xs[i] - xs[other])


def interpolate(
    contents: Any, centres: Any, low: float, high: float, x: Any, floor: float = EPSILON
) -> Any:
    """``PDF::GetVal``'s reading of a fine, evenly binned histogram: a line between bin centres.

    ``contents`` and ``centres`` are the inner bins only, of an axis from
    ``low`` to ``high``; each ``x`` falls in the bin ``FindBin`` gives,
    clamped to the first and the last.
    """
    n = len(centres)
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        step = np.floor(n * (x - low) / (high - low))
    bins = np.clip(np.nan_to_num(step, nan=n), 0, n - 1).astype(np.int64)
    up = ((x > centres[bins]) & (bins != n - 1)) | (bins == 0)
    other = np.where(up, bins + 1, bins - 1)
    dy = contents[bins] - contents[other]
    value = contents[bins] + (x - centres[bins]) * dy / (centres[bins] - centres[other])
    return np.maximum(value, floor)


class PDF:
    """One density: the original histogram, its smoothed copy, and the fine histogram read back."""

    def __init__(self, name: str, spec: PDFSettings, normalise: bool = True) -> None:
        self.name = name
        self.spec = spec
        self.normalise = normalise
        self.log = Logger(name)

    def build(self, histogram: Histogram) -> PDF:
        """``BuildPDF``: smooth, spline and sample ``histogram``."""
        if self.spec.interpolation in ("Spline3", "Spline5", "KDE"):
            raise self.log.refuse(
                f"PDFInterpol={self.spec.interpolation} is a TMVA interpolation xrdroot does not "
                "have; Spline0, Spline1 and Spline2 are the ones it has"
            )
        name = histogram.name
        self.original = hists.renamed(histogram.copy(), name + "_original")
        self.smoothed = hists.renamed(histogram.copy(), name + "_smoothed")
        low, high = self.spec.smoothing()
        if low < 0 or high < low:
            raise self.log.fatal(
                f"PDF construction called with minnsmooth={low}, maxnsmooth={high}"
            )
        if high > 0 and len(self.smoothed) > 1:
            _smooth(self.smoothed, low, high)
        self._sample()
        return self

    def _sample(self) -> None:
        """``FillSplineToHist``: the spline sampled into the fine histogram, then normalised."""
        axis = self.smoothed.axes[0]
        points = hists.bins(self.smoothed)[1:-1]
        centres = axis.root_centers()[1:-1]
        method = self.spec.interpolation
        if method == "Spline0":
            fine = hists.renamed(self.smoothed.copy(), self.smoothed.name + "_hist_from_spline0")
            fine._core["TNamed"]["fTitle"] = self.smoothed.name + "_hist from_spline0"
            self._fine_width = float(axis.widths()[0])
        else:
            label = "spline1" if method == "Spline1" else "spline2"
            fine = hists.book(
                self.smoothed.name + "_hist_from_" + label,
                self.smoothed.name + "_hist from_" + label,
                NBIN_PDF_HIST,
                axis.low,
                axis.high,
            )
            x = fine.axes[0].root_centers()[1:-1]
            y = (spline1 if method == "Spline1" else spline2)(centres, points, x)
            fallback = points[np.clip(axis.find_bin(x) - 1, 0, len(points) - 1)]
            y = np.where(y <= EPSILON, fallback, y)
            hists.set_bins(fine, np.concatenate(([0.0], np.maximum(y, EPSILON), [0.0])))
            self._fine_width = (axis.high - axis.low) / NBIN_PDF_HIST
        self.fine = fine
        integral = self.integral()
        if integral < 0:
            raise self.log.fatal(f"Integral: {integral:g} <= 0")
        if self.normalise and integral > 0:
            cells = fine._cells()
            cells[:] = cells / integral
        self._contents = hists.bins(fine)[1:-1]
        self._centres = fine.axes[0].root_centers()[1:-1]

    def integral(self) -> float:
        """``GetIntegral``: the fine histogram's sum of weights times its bin width."""
        return float(np.sum(hists.bins(self.fine)[1:-1])) * self._fine_width

    @property
    def xmin(self) -> float:
        return float(self.smoothed.axes[0].low)

    @property
    def xmax(self) -> float:
        return float(self.smoothed.axes[0].high)

    def value(self, x: Any, floor: float = EPSILON) -> Any:
        """``GetVal``: the density at each ``x``, never below ``floor``."""
        x = np.asarray(x, dtype=np.float64)
        if self.spec.interpolation == "Spline0":
            bins = np.clip(self.fine.axes[0].find_bin(x), 1, len(self._contents))
            return np.maximum(self._contents[bins - 1], floor)
        axis = self.fine.axes[0]
        return interpolate(self._contents, self._centres, axis.low, axis.high, x, floor)

    def integral_between(self, low: float, high: float) -> float:
        """``GetIntegral(xmin, xmax)``: the fine bins between two points, part-bins at the ends."""
        axis = self.fine.axes[0]
        first = max(int(axis.find_bin(low)), 1)
        last = min(int(axis.find_bin(high)), axis.nbins)
        if last < first:
            return 0.0
        edges = axis.edges()
        widths = np.float32(np.diff(edges))[first - 1 : last].astype(np.float64)
        widths[0] = np.float32(edges[first] - low)
        if last > first:
            widths[-1] = np.float32(high - edges[last - 1])
        widths = np.where((widths < 0) & (widths > -1.0e-8), 0.0, widths)
        return float(np.dot(self._contents[first - 1 : last], widths))

    def add_xml(self, parent: Node) -> None:
        """``PDF::AddXMLTo``: the settings, and the original histogram it is rebuilt from."""
        low, high = self.spec.smoothing()
        node = parent.add(
            "PDF",
            Name=self.name,
            MinNSmooth=low,
            MaxNSmooth=high,
            InterpolMethod=INTERPOLATIONS.get(self.spec.interpolation, 2),
            KDE_type=1,
            KDE_iter=1,
            KDE_border=1,
            KDE_finefactor=number(1.0),
        )
        axis = self.original.axes[0]
        contents = hists.bins(self.original)[1:-1]
        histogram = node.add(
            "Histogram",
            Name=self.original.name,
            NBins=axis.nbins,
            XMin=number(axis.low),
            XMax=number(axis.high),
            HasEquidistantBins=int(axis.even),
        )
        histogram.block(contents)
        if not axis.even:
            node.add("HistogramBinning", NBins=axis.nbins).block(axis.edges())


#: ``PDF::EInterpolateMethod``, by the names the options give them.
INTERPOLATIONS = {"Spline0": 0, "Spline1": 1, "Spline2": 2, "Spline3": 3, "Spline5": 5, "KDE": 6}


def pdf_from_xml(node: Any, normalise: bool = True) -> PDF:
    """``PDF::ReadXML``: the density rebuilt from its original histogram, as TMVA rebuilds it."""
    methods = {number: name for name, number in INTERPOLATIONS.items()}
    low, high = int(node.get("MinNSmooth", 0)), int(node.get("MaxNSmooth", 0))
    spec = PDFSettings(
        low, low, high, interpolation=methods.get(int(node.get("InterpolMethod", 2)), "Spline2")
    )
    source = node.find("Histogram")
    name = str(source.get("Name"))
    nbins = int(source.get("NBins"))
    contents = [float(token) for token in (source.text or "").split()][:nbins]
    if int(source.get("HasEquidistantBins", 1)):
        made = hists.book(name, name, nbins, float(source.get("XMin")), float(source.get("XMax")))
    else:
        edges = [float(token) for token in (node.find("HistogramBinning").text or "").split()]
        made = hists.book_edges(name, name, edges[: nbins + 1])
    hists.set_bins(made, [0.0, *contents, 0.0], entries=nbins)
    pdf = PDF(str(node.get("Name")), spec, normalise)
    base = name[: -len("_original")] if name.endswith("_original") else name
    return pdf.build(hists.renamed(made, base))


def _smooth(histogram: Histogram, low: int, high: int) -> None:
    """``PDF::SmoothHistogram``: ``Smooth(n)``, or more passes where the errors are larger."""
    cells = histogram._cells()
    if low == high:
        cells[1:-1] = smooth_array(cells[1:-1].astype(np.float64), low)
        return
    for _ in range(_passes(histogram, low, high)):
        cells[1:-1] = smooth_array(cells[1:-1].astype(np.float64), 1)


def _passes(histogram: Histogram, low: int, high: int) -> int:
    """How many single passes ``SmoothHistogram``'s loop over bins makes for a spread of passes.

    For each level from the most passes down, TMVA smooths the whole
    histogram once for every run of bins that wants at least that many -
    ``Smooth(1, "R")`` with no range set being the whole histogram - and once
    for each level at or below the least.
    """
    contents = hists.bins(histogram)[1:-1]
    errors = np.sqrt(np.asarray(histogram.variances(), dtype=np.float64))
    filled = contents > errors
    relative = errors[filled] / contents[filled]
    mean = float(np.mean(relative)) if len(relative) else 0.0
    rms = float(np.sqrt(max(np.mean(relative**2) - mean**2, 0.0))) if len(relative) else 0.0
    wanted = np.full(len(contents), high)
    if rms > 0:
        spread = ((errors / np.where(filled, contents, 1.0) - (mean - rms)) / (2 * rms)) * (
            high - low
        )
        wanted = np.where(filled, spread.astype(np.int64) + low, high)
    wanted = np.clip(wanted, low, high)
    total = 0
    for level in range(high, -1, -1):
        total += 1 if level <= low else _runs(wanted >= level)
    return total


def _runs(marked: Any) -> int:
    """The runs of two or more marked bins that end before the last bin, as TMVA counts them."""
    count, start, stop = 0, -1, -1
    for index, flag in enumerate(marked):
        if flag:
            if start == -1:
                start = index
            else:
                stop = index
        elif stop >= 0:
            count += 1
            start = stop = -1
        else:
            start = -1
    return count
