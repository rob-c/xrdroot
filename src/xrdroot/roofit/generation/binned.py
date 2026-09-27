"""``RooAbsPdf::generateBinned``: a histogram of events drawn bin by bin.

Each bin expects the density at its centre times its width times the number
of events (``fillDataHist``); ``ExpectedData`` keeps exactly that - an
Asimov dataset - ``Extended`` draws a Poisson number in each bin, and
otherwise a Poisson number is drawn in each and bins picked at random by
accept-reject are raised or lowered until the total is what was asked for,
every draw from RooFit's generator in RooFit's order.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..messages import ERROR, log
from ..rng import generator

__all__ = ["binned"]


def _count(pdf: Any, names: frozenset[str], count: Any, exact: bool) -> float | None:
    if count is not None and float(count) > 0:
        return float(count)
    if not pdf.canBeExtended():
        log(
            pdf,
            ERROR,
            "InputArguments",
            f"RooAbsPdf::generateBinned({pdf.GetName()}) ERROR: No event "
            "count provided and p.d.f does not provide expected number of events",
        )
        return None
    expected = pdf.expected(names)
    return expected if exact else float(round(expected))


def _fixed_total(weights: np.ndarray[Any, Any], counts: list[int], total: float) -> list[int]:
    """Raise or lower random bins, accepted by their weight, until the counts add up to
    ``total``."""
    extra = abs(int(total) - sum(counts))
    step = -1 if sum(counts) > total else 1
    top = float(np.max(weights))
    rng = generator()
    while extra > 0:
        index = rng.Integer(len(weights))
        if rng.Uniform(top) < weights[index]:
            if step == 1:
                counts[index] += 1
                extra -= 1
            elif counts[index] > 0:
                counts[index] -= 1
                extra -= 1
    return counts


def binned(pdf: Any, variables: list[Any], count: Any, options: Any) -> Any:
    from ..data.datahist import RooDataHist

    expected_data = bool(options.get("ExpectedData", 0, False) or options.get("Asimov", 0, False))
    extended = bool(options.get("Extended", 0, False))
    names = frozenset(one.GetName() for one in variables)
    total = _count(pdf, names, count, expected_data or extended)
    if total is None:
        return None
    hist = RooDataHist(options.get("Name") or "genData", "genData", variables)
    ctx = {name: hist.column(name) for name in names}
    weights = np.asarray(pdf.value(ctx, names), dtype=np.float64) * hist.binVolumes()
    values = _contents(weights, total, expected_data, extended)
    hist._weights = values
    hist._sumw2 = (
        values.copy() if not expected_data else np.array([math.sqrt(v) ** 2 for v in values])
    )
    return hist


def _contents(weights: Any, total: Any, expected_data: bool, extended: bool) -> Any:
    """The bins' contents: expected, Poisson-varied, or Poisson-varied to add up to ``total``."""
    if expected_data:
        return weights * total
    counts = [generator().Poisson(w * total) for w in weights]
    if extended:
        return np.array(counts, dtype=np.float64)
    return np.array(_fixed_total(weights, counts, total), dtype=np.float64)
