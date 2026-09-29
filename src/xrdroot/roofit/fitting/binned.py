"""The binned likelihood RooFit computes for a HistFactory channel: Poisson counts, bin by bin.

A ``RooRealSumPdf`` with the ``BinnedLikelihood`` attribute - alone, or the
factor of a product that the rest of constrains - is not normalised in a
likelihood: with its bin-width functions switched off its value at a bin's
centre is the bin's expected count ``mu``, and the likelihood of the bin's
``n`` events is ``mu - n log(mu) + log(n!)``, Kahan-summed as
``RooNLLVarNew::doEvalBinnedL`` sums it, with no extended term of its own.
Its empty bins count too.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...random import libm
from ...stats import log_gamma
from .kahan import Kahan

__all__ = ["binned_part", "binned_terms"]


def binned_part(pdf: Any) -> Any:
    """``RooHelpers::getBinnedL``: the binned sum a channel's likelihood counts, or ``None``."""
    if pdf.getAttribute("BinnedLikelihood") and pdf.InheritsFrom("RooRealSumPdf"):
        return pdf
    if pdf.InheritsFrom("RooProdPdf"):
        for part in pdf.pdfList():
            if part.getAttribute("BinnedLikelihood") and part.InheritsFrom("RooRealSumPdf"):
                return part
            if part.getAttribute("MAIN_MEASUREMENT"):
                return None
    return None


def binned_terms(binned: Any, columns: dict[str, Any], weights: Any) -> tuple[Kahan, float, int]:
    """The bins' Poisson terms and their events: the sum, the events, and how many bins had
    events but no expectation - which RooFit logs as an evaluation error."""
    from ..pdfs.histfactory import binned_likelihood

    with binned_likelihood():
        mu = np.broadcast_to(np.asarray(binned.compute(dict(columns)), dtype=np.float64),
                             weights.shape)  # fmt: skip
    total, events, bad = Kahan(), Kahan(), 0
    for m, n in zip(mu.tolist(), weights.tolist()):
        if m <= 0 and n > 0:
            bad += 1
            continue
        if abs(m) < 1e-10 and abs(n) < 1e-10:
            total.add(0.0)  # Poisson(0|0): nothing, added all the same, as the kernel adds it
            events.add(n)
            continue
        total.add(m - n * float(libm.log(m)) + log_gamma(n + 1.0))
        events.add(n)
    return total, events.total, bad
