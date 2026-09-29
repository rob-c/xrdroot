"""How a regression is judged: ``TestRegression``'s numbers and ``ResultsRegression``'s histograms.

The deviation of each event's regression from its true target gives the
bias (its weighted mean), the RMS about it, and the same again for the
events within two sigma ("truncated"); the mutual information of output
and target comes from their 150 x 100 histogram rebinned by two, as
``Tools::GetMutualInformation`` has it. The histograms are the deviation
against each variable and target, and the squared deviation's
distribution, whole and below its 90% quantile. All of it in TMVA's single
precision where TMVA's arrays are ``Float_t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..hist import Histogram
from . import hists
from .dataset import DataSetInfo, Events

__all__ = ["RegressionStats", "deviation_histograms", "mutual_information", "test_regression"]

f32 = np.float32


@dataclass
class RegressionStats:
    """``TestRegression``'s outputs for one sample."""

    bias: float
    bias_t: float
    dev: float
    dev_t: float
    rms: float
    rms_t: float
    minf: float
    minf_t: float
    corr: float


def mutual_information(counts: Any) -> float:
    """``GetMutualInformation`` of a 2-D histogram's in-range bins, rebinned by two each way."""
    total = float(counts.sum())
    if total == 0:
        return -1.0
    nx, ny = counts.shape
    merged = counts[: nx - nx % 2].reshape(nx // 2, 2, ny)[:, :, : ny - ny % 2]
    merged = merged.reshape(nx // 2, 2, ny // 2, 2).sum(axis=(1, 3))
    p_xy = merged / total
    p_x = merged.sum(axis=1, keepdims=True) / total
    p_y = merged.sum(axis=0, keepdims=True) / total
    good = (p_xy > 0) & (p_x > 0) & (p_y > 0)
    ratio = np.where(good, p_xy / np.where(good, p_x * p_y, 1.0), 1.0)
    return float(np.sum(np.where(good, p_xy * np.log(ratio), 0.0)))


def _counts(r: Any, t: Any, w: Any, low: float, high: float) -> Any:
    """The bins of ``TH2F(150, low, high, 100, low, high)`` filled with ``(r, t)``."""
    counts = np.zeros((150, 100))
    ix = np.floor(150 * (r - low) / (high - low)).astype(np.int64)
    iy = np.floor(100 * (t - low) / (high - low)).astype(np.int64)
    inside = (ix >= 0) & (ix < 150) & (iy >= 0) & (iy < 100) & (r >= low) & (t >= low)
    np.add.at(counts, (ix[inside], iy[inside]), w[inside])
    return counts.astype(f32).astype(np.float64)


def _sum(values: Any) -> float:
    """A sum taken in double one value after another, as TMVA's loops take it."""
    return float(np.cumsum(np.asarray(values, dtype=np.float64))[-1]) if len(values) else 0.0


def _product(*factors: Any) -> Any:
    """A product of ``Float_t`` values, taken in single precision as C++ takes it."""
    found = np.asarray(factors[0], dtype=f32)
    for factor in factors[1:]:
        found = (found * np.asarray(factor, dtype=f32)).astype(f32)
    return found.astype(np.float64)


def _moments(d: Any, w: Any) -> tuple[float, float, float]:
    total = _sum(w)
    bias = _sum(_product(w, d)) / total
    dev = _sum(_product(w, np.abs(d))) / total
    rms = float(np.sqrt(_sum(_product(w, d, d)) / total - bias * bias))
    return bias, dev, rms


def test_regression(output: Any, target: Any, weights: Any) -> RegressionStats:
    """``MethodBase::TestRegression`` of one sample: the first target's outputs and truths."""
    r = np.asarray(output, dtype=f32).astype(np.float64)
    t = np.asarray(target, dtype=f32).astype(np.float64)
    w = np.asarray(weights, dtype=f32).astype(np.float64)
    d = (r - t).astype(f32).astype(np.float64)
    bias, dev, rms = _moments(d, w)
    total = _sum(w)
    m1, m2 = _sum(_product(t, w)) / total, _sum(_product(r, w)) / total
    corr = _sum(_product(t, r)) / total - m1 * m2
    corr /= np.sqrt(
        (_sum(_product(t, t, w)) / total - m1 * m1) * (_sum(_product(r, r, w)) / total - m2 * m2)
    )
    low = float(min(t.min(), r.min())) if len(t) else 0.0
    high = float(max(t.max(), r.max())) if len(t) else 1.0
    keep = (d >= bias - 2 * rms) & (d <= bias + 2 * rms)
    bias_t, dev_t, rms_t = _moments(d[keep], w[keep])
    return RegressionStats(
        bias,
        bias_t,
        dev,
        dev_t,
        rms,
        rms_t,
        mutual_information(_counts(r, t, w, low, high)),
        mutual_information(_counts(r[keep], t[keep], w[keep], low, high)),
        float(corr),
    )


def _deviation(
    name: str, x: Any, y: Any, low: float, high: float, titles: tuple[str, str]
) -> Histogram:
    """``DeviationAsAFunctionOf``: the deviation against ``x``, 50 x 50 bins with margins."""
    xmin, xmax = f32(min(low, float(x.min()))), f32(max(high, float(x.max())))
    ymin, ymax = f32(y.min()), f32(y.max())
    step = f32(abs(xmax - xmin) / f32(49))
    xmin, xmax = f32(xmin - f32(1.01) * step), f32(xmax + f32(1.01) * step)
    step = f32((ymax - ymin) / f32(49))
    ymin, ymax = f32(ymin - f32(1.01) * step), f32(ymax + f32(1.01) * step)
    made = Histogram.book(
        name,
        hists.auto_axis(50, float(xmin), float(xmax), x),
        hists.auto_axis(50, float(ymin), float(ymax), y),
        title=f"{name};{titles[0]};{titles[1]}",
        kind="F",
    )
    made.fill(x, y)
    return made


def _quadratic(name: str, squared: Any, weights: Any, cut: float | None) -> Histogram:
    """``QuadraticDeviation``: the squared deviations in 500 bins, up to 1.1 of the largest."""
    top = f32(cut) if cut is not None else f32(max(float(squared.max()), 0.0))
    keep = np.ones(len(squared), dtype=bool) if cut is None else squared <= cut
    nbins, low, high = hists.auto_axis(500, 0.0, float(f32(top * f32(1.1))), squared[keep])
    made = hists.book(name, f"{name};Quadratic Deviation;Weighted Entries", nbins, low, high)
    made.fill(squared[keep], weight=weights[keep])
    return made


def quantile(histogram: Histogram, probability: float) -> float:
    """``TH1::GetQuantiles`` of one probability, as ROOT 6.40 answers it.

    That is :func:`xrdroot.distribution.quantiles`' answer - within a bin on
    the straight line across it, and on an edge where ROOT has its own rule.
    An empty histogram, which ROOT refuses and leaves the answer unset for,
    answers its low edge.
    """
    from ..distribution import quantiles

    if not float(np.sum(hists.bins(histogram)[1:-1])):
        return float(histogram.axes[0].low)
    return float(quantiles(histogram, [probability])[0])


def deviation_histograms(
    prefix: str, dsi: DataSetInfo, events: Events, output: Any
) -> list[Histogram]:
    """``CreateDeviationHistograms``: every histogram of the regression's deviations."""
    output = np.asarray(output, dtype=f32).reshape(len(events), -1).astype(np.float64)
    targets = np.asarray(events.targets, dtype=f32).astype(np.float64)
    weights = np.asarray(events.weights, dtype=f32).astype(np.float64)
    made = []
    columns = [
        (events.values[:, i], dsi.variables[i], f"var{i}") for i in range(len(dsi.variables))
    ]
    columns += [(targets[:, i], dsi.targets[i], f"tgt{i}") for i in range(len(dsi.targets))]
    for x, info, tag in columns:
        for k, target in enumerate(dsi.targets):
            y = (output[:, k] - targets[:, k]).astype(f32).astype(np.float64)
            title = f"{target.title}_{{regression}} - {target.title}_{{true}}"
            made.append(
                _deviation(
                    f"{prefix}_reg_{tag}_rtgt{k}",
                    np.asarray(x, dtype=np.float64),
                    y,
                    info.minimum,
                    info.maximum,
                    (info.title, title),
                )
            )
    for k in range(len(dsi.targets)):
        squared = ((output[:, k] - targets[:, k]).astype(f32) ** 2).astype(np.float64)
        whole = _quadratic(f"{prefix}_Quadr_Deviation_target_{k}_", squared, weights, None)
        made.append(whole)
        cut = quantile(whole, 0.9)
        made.append(_quadratic(f"{prefix}_Quadr_Dev_best90perc_target_{k}_", squared, weights, cut))
    return made
