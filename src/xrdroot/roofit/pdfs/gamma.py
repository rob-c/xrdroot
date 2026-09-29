"""``RooGamma``: the gamma density of shape ``gamma``, scale ``beta``, from ``mu``.

Its value is ``TMath::GammaDist`` - MathCore's gamma density - and a
likelihood's events take RooFit's kernel: ``-lgamma(gamma)`` less the
scaled distance, plus its logarithm times ``gamma - 1``, through VDT's
``fast_exp`` and ``fast_log``. It integrates over ``x`` by the gamma
distribution function, and draws ``x`` by Marsaglia and Tsang's method,
as RooFit does - for a shape below one, from the shape one above it, times
a uniform draw to the power of its inverse.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...random import libm
from .. import kernels
from ..pdf import RooAbsPdf, check_range
from ..real import Context
from .basic import ref
from .chisquare import _each

__all__ = ["RooGamma"]


def gamma_dist(x: float, gamma: float, mu: float, beta: float) -> float:
    """``TMath::GammaDist``: nothing below ``mu`` or for a shape or scale that is not positive."""
    from ...pyroot.core.distributions import gamma_pdf

    if x < mu or gamma <= 0 or beta <= 0:
        return 0.0
    return float(gamma_pdf(x, gamma, beta, mu))


def _kernel(x: Any, gamma: Any, beta: Any, mu: Any) -> Any:
    """``computeGamma``, event by event."""
    columns = (np.asarray(v, dtype=np.float64) for v in (x, gamma, beta, mu))
    x, gamma, beta, mu = np.broadcast_arrays(*columns)
    lgamma = np.asarray(libm.lgamma(gamma), dtype=np.float64)
    out = np.where(x == mu, (gamma == 1.0) / beta, -lgamma)
    away = x != mu
    inverse = 1.0 / beta
    scaled = (x - mu) * inverse
    logged = kernels.fast_log(np.where(away, scaled, 1.0))
    found = kernels.fast_exp(out - scaled + logged * (gamma - 1)) * inverse
    return np.where(away, found, out)


def _positive(c: float, rng: Any) -> tuple[float, float]:
    """A normal draw ``xgen`` and ``1 + c xgen``, drawn again until that is positive."""
    xgen, v = 0.0, 0.0
    while v <= 0.0:
        xgen = float(rng.gaus(0.0, 1.0))
        v = 1.0 + c * xgen
    return xgen, v


def _draw(gamma: float, beta: float, mu: float, low: float, high: float, rng: Any) -> float:
    """``randomGamma``: Marsaglia and Tsang's squeeze, drawn again until inside the range."""
    while True:
        d = gamma - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        xgen, v = _positive(c, rng)
        v = v * v * v
        u = float(rng.rndm())
        x = d * v * beta + mu
        if u < 1.0 - 0.0331 * (xgen * xgen) * (xgen * xgen) and low < x < high:
            return x
        if math.log(u) < 0.5 * xgen * xgen + d * (1.0 - v + math.log(v)) and low < x < high:
            return x


class RooGamma(RooAbsPdf):
    """The gamma density: ``Gamma(x - mu; gamma, beta)``."""

    def __init__(self, name: Any, title: Any, x: Any, gamma: Any, beta: Any, mu: Any) -> None:
        super().__init__(name, title)
        self.x = self._proxy("x", ref(x))
        self.gamma = self._proxy("gamma", ref(gamma))
        self.beta = self._proxy("beta", ref(beta))
        self.mu = self._proxy("mu", ref(mu))
        check_range(self, [gamma, beta], 0.0)

    def compute(self, ctx: Context) -> Any:
        args = [one.compute(ctx) for one in (self.x, self.gamma, self.beta, self.mu)]
        if kernels.active():
            return _kernel(*args)
        x, gamma, beta, mu = args
        return _each(gamma_dist, x, gamma, mu, beta)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        if self.x.GetName() not in names or not self.x.isFundamental():
            return frozenset()
        return frozenset([self.x.GetName()])

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        from ...pyroot.core.rmath import gamma_cdf

        low, high = self.x.getMin(rng), self.x.getMax(rng)
        shape, scale, mu = (one.compute(ctx) for one in (self.gamma, self.beta, self.mu))
        return _each(lambda a, b, m: gamma_cdf(high, a, b, m) - gamma_cdf(low, a, b, m),
                     shape, scale, mu)  # fmt: skip

    def generator_code(self, names: frozenset[str]) -> int:
        return 1 if self.x.isFundamental() and names == frozenset([self.x.GetName()]) else 0

    def generate_event(self, code: int, rng: Any) -> dict[str, float]:
        """``generateEvent``: a draw in range, directly for a shape of one or more."""
        gamma, beta, mu = self.gamma.getVal(), self.beta.getVal(), self.mu.getVal()
        low, high = self.x.getMin(), self.x.getMax()
        if gamma >= 1:
            return {self.x.GetName(): _draw(gamma, beta, mu, low, high, rng)}
        while True:
            u = float(rng.rndm())
            value = _draw(1 + gamma, beta, mu, 0.0, math.inf, rng) * u ** (1.0 / gamma)
            if low < value < high:
                return {self.x.GetName(): value}
