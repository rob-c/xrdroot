"""How a multiclass classifier is judged: each class against the rest, and each against each.

A method's output for an event is a probability per class. Taking one
class as signal and every other - or one other - as background gives a
ROC curve (``ROCCurve``, from every event's output for that class), its
area and the signal efficiency at a background efficiency, read off the
curve by ``TSpline1``'s straight lines; the confusion matrices are the last
for each pair of classes. ``ResultsMulticlass``'s histograms are each
class's events' output for every class, 40 bins normalised, and the ROC
curves as graphs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..graph import Graph
from . import hists
from .dataset import DataSetInfo, Events
from .evaluation import _rates, norm_hist, roc_integral
from .pdf import spline1

__all__ = [
    "confusion",
    "eff_s_for_eff_b",
    "performance_graphs",
    "response_histograms",
    "roc_summary",
]

#: ``fNbinsMVAoutput``: the bins of each response histogram.
NBINS = 40


def eff_s_for_eff_b(values: Any, signal: Any, weights: Any, level: float) -> float:
    """``ROCCurve::GetEffSForEffB``: the signal efficiency where the background's is ``level``."""
    sensitivity, specificity = _rates(values, signal, weights)
    eff_b = (1.0 - specificity)[::-1]
    eff_s = sensitivity[::-1]
    return float(spline1(eff_b, eff_s, np.array([level]))[0])


def roc_summary(output: Any, events: Events, cls: int) -> tuple[float, float, float, float]:
    """One class against the rest: the ROC area and the efficiencies at 1%, 10% and 30%."""
    values, signal = output[:, cls], events.classes == cls
    weights = events.weights
    area = roc_integral(values, signal, weights)
    effs = [eff_s_for_eff_b(values, signal, weights, level) for level in (0.01, 0.10, 0.30)]
    return (area, *effs)


def confusion(output: Any, events: Events, nclasses: int, level: float) -> Any:
    """``ResultsMulticlass::GetConfusionMatrix``: row against column, row taken as signal."""
    made = np.full((nclasses, nclasses), np.nan)
    for row in range(nclasses):
        for column in range(nclasses):
            if row == column:
                continue
            pick = (events.classes == row) | (events.classes == column)
            made[row, column] = eff_s_for_eff_b(
                output[pick, row], events.classes[pick] == row, events.weights[pick], level
            )
    return made


def response_histograms(prefix: str, dsi: DataSetInfo, events: Events, output: Any) -> list[Any]:
    """``CreateMulticlassHistos``: class ``i``'s events' output for class ``j``, normalised."""
    made = []
    low, high = float(np.float32(-0.0002)), float(np.float32(1.0002))
    names = [cls.name for cls in dsi.classes]
    for i, own in enumerate(names):
        mask = events.classes == i
        for j, other in enumerate(names):
            name = f"{prefix}_{other}_prob_for_{own}"
            histogram = hists.book(name, name, NBINS, low, high)
            histogram.fill(output[mask, j].astype(np.float64), weight=events.weights[mask])
            norm_hist(histogram)
            made.append(histogram)
    return made


def _graph(name: str, title: str, values: Any, signal: Any, weights: Any) -> Graph:
    sensitivity, specificity = _rates(values, signal, weights)
    return Graph.new(name, sensitivity, specificity, title=title)


def performance_graphs(prefix: str, dsi: DataSetInfo, events: Events, output: Any) -> list[Any]:
    """``CreateMulticlassPerformanceHistos``: each class's ROC curve, then each pair's."""
    names = [cls.name for cls in dsi.classes]
    made = [
        _graph(
            f"{prefix}_rejBvsS_{name}",
            f"{prefix}_{name}",
            output[:, i],
            events.classes == i,
            events.weights,
        )
        for i, name in enumerate(names)
    ]
    for i, own in enumerate(names):
        for j, other in enumerate(names):
            if i == j:
                continue
            pick = (events.classes == i) | (events.classes == j)
            made.append(
                _graph(
                    f"{prefix}_1v1rejBvsS_{own}_vs_{other}",
                    f"{prefix}_{own}_vs_{other}",
                    output[pick, i],
                    events.classes[pick] == i,
                    events.weights[pick],
                )
            )
    return made
