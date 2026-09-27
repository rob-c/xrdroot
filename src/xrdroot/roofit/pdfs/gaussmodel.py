"""``RooGaussModel``: a Gaussian resolution, and its convolutions with the decay basis functions.

The Gaussian's mean and width are each scaled - ``mean * msf``, ``sigma *
ssf`` - which is how a per-event error enters: ``RooGaussModel(..., dt,
bias, sigma, dterr)`` has a width of ``sigma * dterr`` event by event. Its
convolution with ``exp(-|t|/tau)`` and the rest is RooFit's closed form in
the complex error function (:func:`~xrdroot.roofit.cerf.eval_cerf`); the
exponential case is computed as RooFit's batch kernel
``computeGaussModelExpBasis`` computes it, the others as ``evaluate`` does,
and the integrals over the time are ``analyticalIntegral``'s.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .. import mathfuncs as mf
from ..cerf import eval_cerf
from ..real import Context
from .basic import ref
from .resolution import (
    COS,
    COSH,
    EXP,
    LIN,
    NONE,
    QUAD,
    SIN,
    SINH,
    RooResolutionModel,
    basis_sign,
    basis_type,
)

__all__ = ["RooGaussModel"]

#: ``sqrt(2)``, ``sqrt(2 pi)`` and ``sqrt(pi)``, as RooGaussModel computes them.
ROOT2 = math.sqrt(2.0)
ROOT2PI = math.sqrt(2.0 * math.atan2(0.0, -1.0))
ROOTPI = math.sqrt(math.atan2(0.0, -1.0))

Array = Any


def _both_sides(sign: int, plus: Any, minus: Any) -> Any:
    """The positive-time term where the basis has one, plus the negative-time term where it has one."""
    result: Any = 0.0
    if sign != -1:
        result = result + plus()
    if sign != 1:
        result = result + minus()
    return result


def _exp_like(sign: int, u: Array, c: Array) -> Array:
    return _both_sides(
        sign, lambda: np.real(eval_cerf(0.0, -u, c)), lambda: np.real(eval_cerf(0.0, u, c))
    )


def _hyperbolic(kind: int, sign: int, u: Array, c: Array, y: float) -> Array:
    sgn = 1.0 if kind == COSH else -1.0

    def plus() -> Array:
        return 0.5 * (
            np.real(eval_cerf(0.0, -u, c * (1 - y)))
            + sgn * np.real(eval_cerf(0.0, -u, c * (1 + y)))
        )

    def minus() -> Array:
        return 0.5 * (
            sgn * np.real(eval_cerf(0.0, u, c * (1 - y))) + np.real(eval_cerf(0.0, u, c * (1 + y)))
        )

    return _both_sides(sign, plus, minus)


def _polynomial(kind: int, xprime: Array, u: Array, c: Array) -> Array:
    """The linear and quadratic bases, of positive times only."""
    f0 = np.exp(-xprime + c * c) * mf.erfc(-u + c)
    f1 = np.exp(-u * u)
    x2c2 = xprime - 2 * c * c
    if kind == LIN:
        return x2c2 * f0 + (2 * c / ROOTPI) * f1
    return x2c2 * x2c2 * f0 + (2 * c / ROOTPI) * x2c2 * f1 + 2 * c * c * f0


def _oscillating(kind: int, sign: int, omega_tau: float, u: Array, c: Array) -> Array:
    if kind == SIN:
        if omega_tau == 0.0:
            return np.zeros(np.shape(u))
        return _both_sides(sign, lambda: -np.imag(eval_cerf(-omega_tau, -u, c)),
                           lambda: -np.imag(eval_cerf(omega_tau, u, c)))  # fmt: skip
    return _both_sides(sign, lambda: np.real(eval_cerf(-omega_tau, -u, c)),
                       lambda: np.real(eval_cerf(omega_tau, u, c)))  # fmt: skip


def _decay(kind: int, sign: int, x: Array, mean: Array, sigma: Array, tau: float, p2: float) -> Array:
    """The Gaussian convolved with a basis of lifetime ``tau`` and frequency - or width - ``p2``."""
    omega_tau = (p2 if kind in (SIN, COS) else 0.0) * tau
    xprime = (x - mean) / tau
    c = sigma / (ROOT2 * tau)
    u = xprime / (2 * c)
    if kind == EXP or (kind == COS and omega_tau == 0.0):
        return _exp_like(sign, u, c)
    if kind in (SIN, COS):
        return _oscillating(kind, sign, omega_tau, u, c)
    if kind in (COSH, SINH):
        return _hyperbolic(kind, sign, u, c, tau * p2 / 2)
    return _polynomial(kind, xprime, u, c)


def convolved(x: Array, mean: Array, sigma: Array, p1: float, p2: float, code: int) -> Array:
    """``RooGaussModel::evaluate``: the Gaussian convolved with the basis of ``code``."""
    kind, sign = basis_type(code), basis_sign(code)
    tau = p1 if code else 0.0
    if kind == COSH and p2 == 0:
        kind = EXP
    if kind == NONE or (kind in (EXP, COS) and tau == 0.0):
        xprime = (x - mean) / sigma
        result = np.exp(-0.5 * xprime * xprime) / (sigma * ROOT2PI)
        return result * 2 if code and sign == 0 else result
    if tau == 0.0:
        return np.zeros(np.broadcast(x, mean, sigma).shape)
    return _decay(kind, sign, x, mean, sigma, tau, p2)


class RooGaussModel(RooResolutionModel):
    """``RooGaussModel(name, title, x, mean, sigma[, msSF | meanSF, sigmaSF])``."""

    def __init__(self, name: Any, title: Any, x: Any, mean: Any, sigma: Any, *scales: Any) -> None:
        super().__init__(name, title, x)
        msf, ssf = (scales[0], scales[-1]) if scales else (1.0, 1.0)
        self.mean = self._proxy("mean", ref(mean))
        self.sigma = self._proxy("sigma", ref(sigma))
        self.msf = self._proxy("msf", ref(msf))
        self.ssf = self._proxy("ssf", ref(ssf))
        self._flat_sf_int = False
        self._asymp_int = False

    def advertiseFlatScaleFactorIntegral(self, flag: bool) -> None:
        self._flat_sf_int = bool(flag)

    def advertiseAymptoticIntegral(self, flag: bool) -> None:
        self._asymp_int = bool(flag)

    def _scaled(self, ctx: Context) -> tuple[Array, Array]:
        return self.mean.compute(ctx) * self.msf.compute(ctx), self.sigma.compute(
            ctx
        ) * self.ssf.compute(ctx)

    def compute(self, ctx: Context) -> Any:
        x = np.asarray(self.x.compute(ctx), dtype=np.float64)
        mean, sigma = self._scaled(ctx)
        p1, p2 = self.basis_values(ctx)
        if np.ndim(p1) or np.ndim(p2):
            return np.vectorize(convolved, otypes=[np.float64])(
                x, mean, sigma, p1, p2, self._basis_code
            )
        with np.errstate(all="ignore"):
            return convolved(x, mean, sigma, float(p1), float(p2), self._basis_code)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        mine = frozenset([self.x.GetName()])
        if self._flat_sf_int and self.ssf.isFundamental() and mine | {self.ssf.GetName()} <= names:
            return mine | {self.ssf.GetName()}
        return mine & names

    def integral_code(self, names: frozenset[str]) -> int:
        return 2 if self.ssf.GetName() in names and self.ssf.GetName() != self.x.GetName() else 1

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        from .gaussint import integral

        mean, sigma = self._scaled(ctx)
        ssf_int = 1.0
        if self.ssf.GetName() in names and self.ssf.GetName() != self.x.GetName():
            ssf_int = self.ssf.getMax(rng) - self.ssf.getMin(rng)
        p1, p2 = self.basis_values(ctx)
        bounds = (self.x.getMin(rng), self.x.getMax(rng))
        with np.errstate(all="ignore"):
            found = integral(
                bounds, mean, sigma, float(p1), float(p2), self._basis_code, self._asymp_int
            )
        return found * ssf_int

    def generator_code(self, names: frozenset[str]) -> int:
        return 1 if names == frozenset([self.x.GetName()]) else 0

    def generate_event(self, code: int, rng: Any, bounds: Any = None) -> dict[str, float]:
        """``generateEvent``: a Gaussian draw, drawn again until it is inside ``bounds`` (the range)."""
        low, high = bounds if bounds is not None else (self.x.getMin(), self.x.getMax())
        mean, sigma = (float(np.asarray(one)) for one in self._scaled({}))
        while True:
            value = rng.Gaus(mean, sigma)
            if low < value < high:
                return {self.x.GetName(): value}
