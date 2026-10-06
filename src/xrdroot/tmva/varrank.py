"""The method-unspecific ranking of a regression's variables, as ``TransformationHandler`` makes it.

With one target, each variable is ranked four ways: by its correlation
with the target (unweighted), and by the mutual information, the
correlation ratio and the transposed correlation ratio of the target
against it in the 300 x 300 scatter plot ``PlotVariables`` books, rebinned
by two. TMVA books that plot with the ranges of the variables, not of the
target on its x axis - the target's index is taken as a variable's - and
that is kept, since it is what TMVA's numbers come from.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import DataSetInfo, Events
from .regeval import mutual_information

__all__ = ["correlation_ratio", "regression_rankings"]

#: The scatter plots' bins on each axis.
NBINS_2D = 300


def _scatter(x: Any, y: Any, w: Any, xr: tuple[float, float], yr: tuple[float, float]) -> Any:
    """Every bin, the flow bins at the edges, of ``TH2F(300, xr, 300, yr)`` filled."""
    counts = np.zeros((NBINS_2D + 2, NBINS_2D + 2))
    ix = np.clip(np.floor(NBINS_2D * (x - xr[0]) / (xr[1] - xr[0])) + 1, 0, NBINS_2D + 1)
    iy = np.clip(np.floor(NBINS_2D * (y - yr[0]) / (yr[1] - yr[0])) + 1, 0, NBINS_2D + 1)
    np.add.at(counts, (ix.astype(np.int64), iy.astype(np.int64)), w)
    return counts.astype(np.float32).astype(np.float64)


def _rebinned(full: Any) -> Any:
    """``RebinX(2)`` and ``RebinY(2)``: the in-range bins merged in pairs, the flow bins kept."""
    n = (full.shape[0] - 2) // 2
    made = np.zeros((n + 2, n + 2))
    pairs_x = full[1:-1].reshape(n, 2, full.shape[1]).sum(axis=1)
    rows = np.concatenate([full[:1], pairs_x, full[-1:]])
    pairs_y = rows[:, 1:-1].reshape(n + 2, n, 2).sum(axis=2)
    made[:, 1:-1] = pairs_y
    made[:, 0], made[:, -1] = rows[:, 0], rows[:, -1]
    return made


def _transposed(full: Any) -> Any:
    """``Tools::TransposeHist``: the in-range bins transposed, the flow bins left where they are."""
    made = full.copy()
    made[1:-1, 1:-1] = full[1:-1, 1:-1].T
    return made


def correlation_ratio(full: Any, yr: tuple[float, float]) -> float:
    """``Tools::GetCorrelationRatio``: the spread of the y means across x bins, over y's.

    The y distribution it divides by is ``ProjectionY``'s, which takes every x
    bin, the flow bins too, while the means are of the in-range x bins only -
    so the ratio is not bounded by one, as TMVA's is not.
    """
    total = float(full[1:-1, 1:-1].sum())
    if total == 0:
        return -1.0
    merged = _rebinned(full)
    inner = merged[1:-1, 1:-1]
    ny = inner.shape[1]
    centres = yr[0] + (np.arange(ny) + 0.5) * (yr[1] - yr[0]) / ny
    projection = merged[:, 1:-1].sum(axis=0)
    mean = float(projection @ centres / projection.sum())
    rms2 = float(projection @ (centres - mean) ** 2 / projection.sum())
    per_x = inner.sum(axis=1)
    means = np.where(per_x > 0, inner @ centres / np.where(per_x > 0, per_x, 1.0), 0.0)
    return float(np.sum(per_x / total * (means - mean) ** 2) / rms2)


def regression_rankings(
    dsi: DataSetInfo, events: Events, stats: Any
) -> list[tuple[str, list[tuple[str, float]]]]:
    """Every ranking TMVA prints for a single-target regression's variables."""
    values = np.asarray(events.values, dtype=np.float32).astype(np.float64)
    target = np.asarray(events.targets[:, 0], dtype=np.float32).astype(np.float64)
    weights = np.asarray(events.weights, dtype=np.float32).astype(np.float64)
    labels = [info.label for info in dsi.variables]
    n = len(target)
    mean_t = target.sum() / n
    var_t = (target * target).sum() / n - mean_t**2
    correlation = []
    for index in range(values.shape[1]):
        x = values[:, index]
        mean_x = x.sum() / n
        var_x = (x * x).sum() / n - mean_x**2
        covariance = (x * target).sum() / n - mean_x * mean_t
        correlation.append(abs(covariance / np.sqrt(var_x * var_t)))
    low, high = stats[2], stats[3]
    xr = (float(low[0]), float(high[0]))
    scatters = []
    for index in range(values.shape[1]):
        yr = (float(low[index]), float(high[index]))
        scatters.append((_scatter(target, values[:, index], weights, xr, yr), xr, yr))
    return [
        ("|Correlation with target|", list(zip(labels, correlation, strict=False))),
        (
            "Mutual information",
            [
                (label, mutual_information(c[1:-1, 1:-1]))
                for label, (c, _, _) in zip(labels, scatters, strict=False)
            ],
        ),
        (
            "Correlation Ratio",
            [
                (label, correlation_ratio(c, yr))
                for label, (c, _, yr) in zip(labels, scatters, strict=False)
            ],
        ),
        (
            "Correlation Ratio (T)",
            [
                (label, correlation_ratio(_transposed(c), yr))
                for label, (c, _, yr) in zip(labels, scatters, strict=False)
            ],
        ),
    ]
