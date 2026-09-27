"""``RooMultiVarGaussian``: a Gaussian in several variables, with a covariance matrix.

It is what a fit result becomes as a density (``createHessePdf``) - and so
what an error band is sampled from - its events drawn as RooFit draws them:
a standard normal draw for each variable, times the transpose of the
covariance's Cholesky factor, plus the means, drawn again if any value is
outside its variable's range.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..collections import as_list
from ..pdf import RooAbsPdf
from ..real import Context

__all__ = ["RooMultiVarGaussian"]


def _cholesky_upper(cov: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """``TDecompChol``'s ``U``, with ``cov = U^T U``, row by row as ROOT computes it."""
    n = len(cov)
    u = np.zeros((n, n))
    for i in range(n):
        total = cov[i, i]
        for k in range(i):
            total -= u[k, i] * u[k, i]
        u[i, i] = np.sqrt(total)
        for j in range(i + 1, n):
            value = cov[i, j]
            for k in range(i):
                value -= u[k, i] * u[k, j]
            u[i, j] = value / u[i, i]
    return u


class RooMultiVarGaussian(RooAbsPdf):
    """``exp(-(x-mu)^T V^-1 (x-mu) / 2)`` over the variables ``x``."""

    def __init__(self, name: Any, title: Any, xs: Any, mus: Any, cov: Any) -> None:
        super().__init__(name, title)
        self.xs = self._list_proxy("x", as_list(xs))
        self.mus = self._list_proxy("mu", as_list(mus))
        self.cov = np.array(getattr(cov, "values", cov), dtype=np.float64)
        self.inverse = np.linalg.inv(self.cov)
        self.lower = _cholesky_upper(self.cov).T

    def compute(self, ctx: Context) -> Any:
        diffs = [x.compute(ctx) - mu.compute(ctx) for x, mu in zip(self.xs, self.mus)]
        alpha: Any = 0.0
        for i, di in enumerate(diffs):
            for j, dj in enumerate(diffs):
                alpha = alpha + di * self.inverse[i, j] * dj
        return np.exp(-0.5 * alpha)

    def generator_code(self, names: frozenset[str]) -> int:
        return -1 if names == frozenset(x.GetName() for x in self.xs) else 0

    def generate_event(self, code: int, rng: Any) -> dict[str, float]:
        """``generateEvent``: ``L z + mu`` - summed as ``TVectorD *= TMatrixD`` sums - until inside."""
        n = len(self.xs)
        while True:
            z = [rng.Gaus(0.0, 1.0) for _ in range(n)]
            drawn = []
            for i in range(n):
                total = 0.0
                for k in range(n):
                    total += self.lower[i, k] * z[k]
                drawn.append(total + self.mus[i].getVal())
            if all(x.getMin() <= v <= x.getMax() for x, v in zip(self.xs, drawn)):
                return {x.GetName(): v for x, v in zip(self.xs, drawn)}
