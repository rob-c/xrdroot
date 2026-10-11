"""``data.plotOn(frame, Efficiency(cat))``: the fraction of each bin's events in a state.

Two histograms of the frame's variable - the events in the category's state
of index 1, and those in the state of index 0 - give a point per bin at
``n1 / (n1 + n0)``, with ``RooHistError::getBinomialIntervalEff``'s bar: the
exact binomial interval, each end where the binomial probability of the
count or more - or of it or fewer - is that of a normal variable more than
half a sigma out, as ROOT 6.40 finds them (its "one sigma" bar is that
wide); its low edge held at the estimate itself when the interval would not
contain it, as RooHistError holds it, and both ends of the empty bin at
minus and plus one.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...efficiency import beta_quantile, regularized_beta
from .hist import RooHist

__all__ = ["binomial_interval", "from_efficiency"]


def _at_most(k: int, total: int, p: float) -> float:
    """The binomial probability of ``k`` or fewer of ``total``, for ``0 <= k < total``, at
    probability ``p``."""
    return regularized_beta(1.0 - p, total - k, k + 1)


def _ordered(n: int, m: int, sigmas: float) -> tuple[float, float]:
    """The interval for ``n <= m``: central, or with its low edge at the estimate."""
    beta = math.erf(0.5 * sigmas / math.sqrt(2.0))
    alpha = 0.5 * (1.0 - beta)
    total, eff = n + m, n / (n + m)
    below = _at_most(n - 1, total, eff) if n > 0 else 1.0
    if below > alpha + beta:
        return eff, beta_quantile(1.0 - (below - beta), n + 1, m)
    return beta_quantile(alpha, n, m + 1), beta_quantile(1.0 - alpha, n + 1, m)


def binomial_interval(n: int, m: int, sigmas: float = 1.0) -> tuple[float, float]:
    """``getBinomialIntervalEff(n, m)``: the efficiency's interval for ``n`` of ``n + m``."""
    if n == 0 and m == 0:
        return -1.0, 1.0
    if n > m:
        low, high = _ordered(m, n, sigmas)
        return 1.0 - high, 1.0 - low
    return _ordered(n, m, sigmas)


def from_efficiency(name: str, edges: Any, accepted: Any, rejected: Any,
                    x_error: float = 1.0) -> RooHist:  # fmt: skip
    """``RooHist(h1, h2, ..., efficiency=true)``: a point per bin, its efficiency and bar."""
    edges = np.asarray(edges, dtype=np.float64)
    centers, half = 0.5 * (edges[:-1] + edges[1:]), 0.5 * np.diff(edges) * x_error
    ys, lows, highs = [], [], []
    for n1, n0 in zip(accepted, rejected, strict=True):
        n, m = int(n1), int(n0)
        eff = n / (n + m) if n + m else 0.0
        low, high = binomial_interval(n, m)
        ys.append(eff)
        lows.append(eff - low)
        highs.append(high - eff)
    made = RooHist(name, "", centers, ys, half, half, lows, highs)
    made.entries = float(np.sum(accepted) + np.sum(rejected))
    return made
