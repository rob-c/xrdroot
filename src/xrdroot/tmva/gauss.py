"""``VariableGaussTransform``: each variable through its own cumulative distribution.

``VarTransform=G`` makes every variable Gaussian and ``U`` makes it flat.
TMVA bins each variable so that every bin holds about a two-thousandth of
the weight (and at least ten events' worth), fills that histogram, smooths
it once, turns it into its cumulative distribution and reads that back by
straight lines between bin centres - a ``Spline1`` :class:`~.pdf.PDF` - and
the Gaussian value is ``sqrt(2) * ErfInverse(2 * cumulant - 1)``. Every step
here is that, the binning's single-precision running sums included.
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np

from . import hists
from .dataset import Events, as_float
from .pdf import PDF, PDFSettings
from .transforms import Transform

__all__ = ["Gauss", "Uniform"]

#: The least number of events' weight a bin holds, and the most bins.
NEVMIN, NBINSMAX = 10, 2000
#: The step below the smallest value that the first edge is put at.
EDGE_EPS = np.float32(1.0e-4)
#: ``sqrt(2)`` as TMVA writes it.
SQRT2 = 1.414213562
#: The furthest the cumulant is let go from 0 and 1, and ``ErfInverse``'s argument from +-1.
CUMULANT_EDGE, ERF_EDGE = 10e-10, 0.99999999

_inverse = np.vectorize(statistics.NormalDist().inv_cdf, otypes=[np.float64])


def _edges(values: Any, weights: Any) -> list[float]:
    """The bin edges ``GetCumulativeDist`` works out for one variable's sorted values."""
    order = np.argsort(values, kind="stable")
    ordered = np.asarray(values, dtype=np.float32)[order]
    ordered_weights = np.asarray(weights, dtype=np.float32)[order]
    total = np.float32(np.sum(np.asarray(weights, dtype=np.float32), dtype=np.float32))
    per_bin = max(np.float32(total / NBINSMAX), np.float32(np.min(ordered_weights) * NEVMIN))
    edges = [float(ordered[0] - EDGE_EPS), float(ordered[0])]
    running, last = np.float32(0), ordered[0]
    for value, weight in zip(ordered, ordered_weights):
        running = np.float32(running + weight)
        if running >= per_bin and value > last:
            edges.append(float(value))
            running, last = np.float32(0), value
    if running != 0 and float(ordered[-1]) > edges[-1]:
        edges.append(float(ordered[-1]))
    return edges


def _cumulative(values: Any, weights: Any, index: int) -> PDF:
    """One variable's cumulative distribution, as the ``Spline1`` density TMVA reads it by."""
    name = f"Cumulative_Var{index}_cls0"
    histogram = hists.book_edges(name, name, _edges(values, weights))
    histogram.fill(np.asarray(values, dtype=np.float64), weight=np.asarray(weights))
    histogram.smooth(1)
    contents = hists.bins(histogram)[1:-1].astype(np.float32).astype(np.float64)
    positive = np.where(contents > 0, contents, 0.0)
    cumulative = np.cumsum(positive) / float(np.sum(positive))
    hists.set_bins(histogram, np.concatenate(([0.0], cumulative, [0.0])))
    spec = PDFSettings(nsmooth=0, min_nsmooth=0, max_nsmooth=0, interpolation="Spline1")
    return PDF(f"GaussTransform var{index} cls0", spec, normalise=False).build(histogram)


class Gauss(Transform):
    """``VarTransform=G``: each variable made Gaussian."""

    letter, name = "G", "Gauss"
    flat = False

    def announce(self) -> str:
        return "Preparing the Gaussian transformation..."

    def prepare(self, events: Events) -> None:
        chosen = self._chosen(events)
        self.pdfs = [
            _cumulative(chosen.values[:, index], chosen.weights, index)
            for index in range(chosen.values.shape[1])
        ]

    def apply(self, values: Any) -> Any:
        columns = []
        for index, pdf in enumerate(self.pdfs):
            cumulant = np.clip(pdf.value(values[:, index]), CUMULANT_EDGE, 1.0 - CUMULANT_EDGE)
            if self.flat:
                columns.append(cumulant)
                continue
            argument = np.clip(2.0 * cumulant - 1.0, -ERF_EDGE, ERF_EDGE)
            columns.append(SQRT2 * _inverse((argument + 1.0) / 2.0) / np.sqrt(2.0))
        return as_float(np.column_stack(columns)) if columns else values


class Uniform(Gauss):
    """``VarTransform=U``: each variable made flat."""

    letter, name, flat = "U", "Uniform", True
