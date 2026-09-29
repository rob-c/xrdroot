"""How a classifier is judged: ``MethodBase::TestClassification`` and the numbers after it.

A classifier's output over the test sample is histogrammed for signal and
background - 40 bins over its range, clipped to ten standard deviations,
and 10000 for the efficiencies - and from those come what TMVA prints: the
significance (the difference of the means over their combined RMS), the
separation (the overlap of the two output densities) and, through
:mod:`.efficiency`, the signal efficiency at each background efficiency.
The ROC integral printed is the Factory's, from ``TMVA::ROCCurve``: the
area under the curve of every output value, not of a histogram.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..hist import Histogram
from . import hists, libcxxsort
from .pdf import PDF, PDFSettings

__all__ = [
    "ClassifierTest",
    "compute_stat",
    "norm_hist",
    "roc_curve",
    "roc_integral",
    "separation",
    "separation_of_hists",
    "test_classification",
]

#: ``gConfig().fVariablePlotting``: bins of a classifier's output, and of the ROC curve.
NBINS_MVA_OUTPUT, NBINS_ROC = 40, 100
#: ``NBIN_HIST_HIGH``: the fine bins the efficiencies are counted in.
NBINS_HIGH = 10000


def _weighted(values: Any, weights: Any) -> tuple[float, float]:
    """``Tools::Mean`` and ``Tools::RMS``: a weighted mean and a weighted spread."""
    total = float(np.sum(weights))
    if total <= 0:
        return 0.0, 0.0
    mean = float(np.dot(values, weights)) / total
    square = float(np.dot(values * values, weights)) / total
    return mean, float(np.sqrt(max(square - mean * mean, 0.0)))


def compute_stat(values: Any, signal: Any, weights: Any) -> tuple[float, ...]:
    """``Tools::ComputeStat``: the signal and background means and RMS, and the range."""
    values = np.asarray(values, dtype=np.float64)
    mean_s, rms_s = _weighted(values[signal], weights[signal])
    mean_b, rms_b = _weighted(values[~signal], weights[~signal])
    return mean_s, mean_b, rms_s, rms_b, float(values.min()), float(values.max())


def norm_hist(histogram: Histogram, norm: float = 1.0) -> float:
    """``Tools::NormHist``: scaled so that its area - weights times bin width - is ``norm``."""
    histogram.sumw2(True)
    total = float(np.sum(hists.bins(histogram)[1:-1]))
    if total == 0:
        return 1.0
    axis = histogram.axes[0]
    area = total * (axis.high - axis.low) / axis.nbins
    if area > 0:
        histogram.scale(norm / area)
    return area


def separation_of_hists(signal: Histogram, background: Histogram) -> float:
    """``Tools::GetSeparation(TH1*, TH1*)``: half the integral of ``(s - b)^2 / (s + b)``."""
    axis = signal.axes[0]
    width = (axis.high - axis.low) / axis.nbins
    s = hists.bins(signal)[1:-1]
    b = hists.bins(background)[1:-1]
    norm_s, norm_b = float(np.sum(s)) * width, float(np.sum(b)) * width
    if norm_s <= 0 or norm_b <= 0:
        return 0.0
    s, b = s / norm_s, b / norm_b
    both = s + b
    terms = np.where(both > 0, (s - b) ** 2 / np.where(both > 0, both, 1.0), 0.0)
    return float(np.sum(terms)) * 0.5 * width


def separation(pdf_s: PDF, pdf_b: PDF) -> float:
    """``Tools::GetSeparation(PDF, PDF)``: the same, from the densities, in 100 steps."""
    low, high = pdf_s.xmin, pdf_s.xmax
    step = (high - low) / 100
    x = (np.arange(100) + 0.5) * step + low
    s, b = pdf_s.value(x), pdf_b.value(x)
    both = s + b
    terms = np.where(both > 0, (s - b) ** 2 / np.where(both > 0, both, 1.0), 0.0)
    return float(np.sum(terms)) * 0.5 * step


@dataclass
class ClassifierTest:
    """What ``TestClassification`` leaves behind: statistics, histograms and output densities."""

    mean_s: float
    mean_b: float
    rms_s: float
    rms_b: float
    xmin: float
    xmax: float
    positive: bool
    histograms: dict[str, Histogram] = field(default_factory=dict)
    pdf_s: PDF | None = None
    pdf_b: PDF | None = None

    def significance(self) -> float:
        """``GetSignificance``: ``|<S> - <B>| / sqrt(RMS_S^2 + RMS_B^2)``."""
        rms = float(np.sqrt(self.rms_s**2 + self.rms_b**2))
        return abs(self.mean_s - self.mean_b) / rms if rms > 0 else 0.0

    def separation(self) -> float:
        """``GetSeparation()``: of the output densities."""
        if self.pdf_s is None or self.pdf_b is None:
            return 0.0
        return separation(self.pdf_s, self.pdf_b)


#: What a method with output densities has histogrammed beside its output, and the suffix.
EXTRAS = (("proba", "_Proba"), ("rarity", "_Rarity"))


def _book(name: str, nbins: int, low: float, high: float) -> Histogram:
    made = hists.book(name, name, nbins, low, high, kind="D")
    made.sumw2(True)
    return made


def _output_histograms(
    testvar: str, low: float, top: float, asked: list[tuple[str, str]]
) -> dict[str, Histogram]:
    """The histograms of a classifier's test output, coarse and fine, and of what else was asked."""
    specs = [
        ("MVA_S", "_S", NBINS_MVA_OUTPUT, low, top),
        ("MVA_B", "_B", NBINS_MVA_OUTPUT, low, top),
        ("MVA_HIGHBIN_S", "_S_high", NBINS_HIGH, low, top),
        ("MVA_HIGHBIN_B", "_B_high", NBINS_HIGH, low, top),
    ]
    specs += [
        (f"{kind}_{side}", f"{name}_{side}", 40, 0.0, 1.0) for kind, name in asked for side in "SB"
    ]
    return {key: _book(testvar + suffix, nbins, a, b) for key, suffix, nbins, a, b in specs}


def _described(values: Any, signal: Any, weights: Any) -> ClassifierTest:
    """The output's statistics, and its range clipped to ten standard deviations of either class."""
    mean_s, mean_b, rms_s, rms_b, low, high = compute_stat(values, signal, weights)
    low = max(min(mean_s - 10 * rms_s, mean_b - 10 * rms_b), low)
    high = min(max(mean_s + 10 * rms_s, mean_b + 10 * rms_b), high)
    return ClassifierTest(mean_s, mean_b, rms_s, rms_b, low, high, mean_s > mean_b)


def _fill_normalised(
    histograms: dict[str, Histogram], sources: dict[str, Any], signal: Any, weights: Any
) -> None:
    """Each source's signal and background events filled into its two histograms, then normed."""
    for side, mask in (("S", signal), ("B", ~signal)):
        for source, data in sources.items():
            histograms[f"{source}_{side}"].fill(np.asarray(data)[mask], weight=weights[mask])
    for histogram in histograms.values():
        norm_hist(histogram)


def test_classification(
    testvar: str, values: Any, signal: Any, weights: Any, extra: dict[str, Any] | None = None
) -> ClassifierTest:
    """``TestClassification``: a classifier's test output histogrammed and described.

    ``extra`` holds, for a method with output densities, the ``"proba"``
    and ``"rarity"`` of every event, which are histogrammed beside it.
    """
    values = np.asarray(values, dtype=np.float32).astype(np.float64)
    weights = np.asarray(weights, dtype=np.float32).astype(np.float64)
    made = _described(values, signal, weights)
    given = extra or {}
    asked = [(kind, name) for kind, name in EXTRAS if kind in given]
    made.histograms = _output_histograms(testvar, made.xmin, made.xmax + 0.00001, asked)
    sources = {"MVA": values, "MVA_HIGHBIN": values, **{kind: given[kind] for kind, _ in asked}}
    _fill_normalised(made.histograms, sources, signal, weights)
    spec = PDFSettings(nsmooth=0, min_nsmooth=0, max_nsmooth=0)
    made.pdf_s = PDF(" PDF Sig", spec).build(made.histograms["MVA_S"])
    made.pdf_b = PDF(" PDF Bkg", spec).build(made.histograms["MVA_B"])
    return made


def _rates(values: Any, signal: Any, weights: Any) -> tuple[Any, Any]:
    """``ROCCurve::ComputeSensitivity`` and ``ComputeSpecificity``, over every output value."""
    order = libcxxsort.order(np.asarray(values, dtype=np.float32))
    weights = np.asarray(weights, dtype=np.float32).astype(np.float64)[order]
    signal = np.asarray(signal, dtype=bool)[order]
    negatives = np.cumsum(np.where(signal, 0.0, weights))
    positives = np.cumsum(np.where(signal, weights, 0.0)[::-1])[::-1]
    total_b = negatives[-1] if len(negatives) else 0.0
    total_s = positives[0] if len(positives) else 0.0
    tiny = np.finfo(np.float64).tiny
    specificity = negatives / total_b if total_b > tiny else np.zeros_like(negatives)
    sensitivity = positives / total_s if total_s > tiny else np.zeros_like(positives)
    return (
        np.concatenate(([1.0], sensitivity, [0.0])),
        np.concatenate(([0.0], specificity, [1.0])),
    )


def roc_integral(values: Any, signal: Any, weights: Any) -> float:
    """``ROCCurve::GetROCIntegral``: the trapezoidal area under (1 - FNR, specificity)."""
    sensitivity, specificity = _rates(values, signal, weights)
    steps = np.diff(1.0 - sensitivity)
    return float(np.sum(0.5 * steps * (specificity[:-1] + specificity[1:])))


def roc_curve(values: Any, signal: Any, weights: Any) -> tuple[Any, Any]:
    """``ROCCurve::GetROCCurve``: the signal efficiencies and background rejections."""
    return _rates(values, signal, weights)
