"""``RooHypatia2``: ROOT's Hypatia line shape - a hyperbolic core with a power-law tail each side.

The shape LHCb fits a peak with when the resolution has heavy tails: for ``zeta = 0`` and
``lambda < 0`` the core is ``(1 + d^2/sigma^2)^(lambda - 1/2)`` and each side beyond
``a * sigma`` (``a2 * sigma`` on the right) is ``A (B -+ d)^-n``, with ``A`` and ``B`` chosen
so the tail meets the core and its slope there - ROOT's own ``RooHypatia2::evaluate``, read from
``roofit/roofitmore/src/RooHypatia2.cxx:242``.

**What is here and what is not.** The ``zeta = 0``, ``lambda < 0`` branch is the one ROOT's own
`testRooHypatia2` writes and it is transcribed whole. ``zeta > 0`` needs the Bessel ratio ROOT
computes with ``besselK`` and its ``LogEval``/``diff_eval`` pair, which this kit does not have, so
it is refused by name rather than evaluated differently. And ``createIntegral`` here is numeric:
ROOT's ``RooHypatia2::analyticalIntegral`` (the closed form for the same three branches) is not
written, so ``forceNumInt`` changes nothing for this class - the values agree because both roads
are the same numeric integral, not because the closed form was reproduced (``W0-220``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from ...random import libm
from ..pdf import RooAbsPdf, check_range
from ..real import Context
from .basic import ref

__all__ = ["RooHypatia2"]


def _power(base: Any, exponent: float) -> Any:
    """``base ** exponent`` through the C library ROOT's ``std::pow`` calls."""
    return libm.power(np.asarray(base, dtype=np.float64), exponent)


class RooHypatia2(RooAbsPdf):
    """``RooHypatia2``: ROOT's Hypatia density over ``x``, with two power-law tails.

    ``Hypatia2::hypatia(x, lambda, zeta, beta, sigma, mu, a, n, a2, n2)``, as the factory writes it.
    """

    def __init__(self, name: Any, title: Any, x: Any, lam: Any, zeta: Any, beta: Any,
                 sigma: Any, mu: Any, a: Any, n: Any, a2: Any, n2: Any) -> None:  # fmt: skip
        super().__init__(name, title)
        self.x = self._proxy("x", ref(x))
        self.lam = self._proxy("lambda", ref(lam))
        self.zeta = self._proxy("zeta", ref(zeta))
        self.beta = self._proxy("beta", ref(beta))
        self.sigma = self._proxy("sigma", ref(sigma))
        self.mu = self._proxy("mu", ref(mu))
        self.a = self._proxy("a", ref(a))
        self.n = self._proxy("n", ref(n))
        self.a2 = self._proxy("a2", ref(a2))
        self.n2 = self._proxy("n2", ref(n2))
        check_range(self, [self.sigma], 0.0)

    def compute(self, ctx: Context) -> Any:
        """``evaluate()``: the core between the two cut points and a power law beyond each."""
        zeta = float(self.zeta.compute(ctx))
        if zeta != 0.0:
            raise UnsupportedFeatureError(
                "RooHypatia2 here evaluates only the zeta = 0 shape: a zeta above zero needs the "
                "ratio of two modified Bessel functions ROOT computes with TMath::BesselK and its "
                "LogEval/diff_eval pair, which this kit does not have yet. Build the shape with "
                "zeta = 0, or fit the peak with a RooCrystalBall instead"
            )
        lam = float(self.lam.compute(ctx))
        if lam >= 0.0:
            raise UnsupportedFeatureError(
                "RooHypatia2 here evaluates only lambda < 0: ROOT itself answers 0 there, logging "
                "'zeta = 0 only supported for lambda < 0', so this kit refuses the shape rather "
                "than returning a density ROOT would not. Use a negative lambda"
            )
        d = np.asarray(self.x.compute(ctx), dtype=np.float64) - float(self.mu.compute(ctx))
        sigma, beta = float(self.sigma.compute(ctx)), float(self.beta.compute(ctx))
        a, n = float(self.a.compute(ctx)), float(self.n.compute(ctx))
        a2, n2 = float(self.a2.compute(ctx)), float(self.n2.compute(ctx))
        asigma, a2sigma = a * sigma, a2 * sigma
        core = libm.exp(beta * d) * _power(1.0 + d * d / (sigma * sigma), lam - 0.5)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            found = np.where(
                d < -asigma,
                self._left(d, lam, beta, sigma, a, n),
                np.where(d > a2sigma, self._right(d, lam, beta, sigma, a2, n2), core),
            )
        return float(found) if np.ndim(found) == 0 else found

    @staticmethod
    def _left(d: Any, lam: float, beta: float, sigma: float, a: float, n: float) -> Any:
        """The left tail, ``A (B - d)^-n``, whose ``A`` and ``B`` meet the core at ``-a sigma``."""
        asigma = a * sigma
        cons1 = libm.exp(-beta * asigma)
        phi = 1.0 + a * a
        k1 = cons1 * _power(phi, lam - 0.5)
        k2 = beta * k1 - cons1 * (lam - 0.5) * _power(phi, lam - 1.5) * 2.0 * a / sigma
        limit = -asigma + n * k1 / k2
        amplitude = k1 * _power(limit + asigma, n)
        return amplitude * _power(limit - d, -n)

    @staticmethod
    def _right(d: Any, lam: float, beta: float, sigma: float, a2: float, n2: float) -> Any:
        """The right tail, ``A (B + d)^-n2``, meeting the core at ``a2 sigma``."""
        a2sigma = a2 * sigma
        cons1 = libm.exp(beta * a2sigma)
        phi = 1.0 + a2 * a2
        k1 = cons1 * _power(phi, lam - 0.5)
        k2 = beta * k1 + cons1 * (lam - 0.5) * _power(phi, lam - 1.5) * 2.0 * a2 / sigma
        limit = -a2sigma - n2 * k1 / k2
        amplitude = k1 * _power(limit + a2sigma, n2)
        return amplitude * _power(limit + d, -n2)
