"""``RooGaussModel::analyticalIntegral``: the Gaussian's convolutions integrated over the time.

Every form is RooFit's, case by case: the plain Gaussian (an ``erf``), the
exponential, sine, cosine and hyperbolic bases (differences of the complex
error function, ``evalCerfInt``), and the linear and quadratic bases
(``erf``/``erfc`` combinations).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...random import libm
from .. import mathfuncs as mf
from ..cerf import eval_cerf
from .gaussmodel import ROOT2, ROOTPI
from .resolution import COS, COSH, EXP, LIN, NONE, SIN, SINH, basis_sign, basis_type

__all__ = ["integral"]

Array = Any


def cerf_int(
    sign: float, x: float, tau: float, umin: Array, umax: Array, c: Array, asymptotic: bool = False
) -> tuple[Array, Array]:
    """``evalCerfInt``: the integral of ``evalCerf`` between ``umin`` and ``umax``, as re and im."""
    if asymptotic:
        re, im = np.full(np.shape(c), 2.0), np.zeros(np.shape(c))
    else:
        low, high = eval_cerf(x, umin, c), eval_cerf(x, umax, c)
        re = np.real(low) - np.real(high) + (mf.erf(umin) - mf.erf(umax))
        im = np.imag(low) - np.imag(high)
        re, im = re * sign, im * sign
    re, im = re * 1.0 - im * x, re * x + im * 1.0
    scale = tau / (1.0 + x * x)
    return re * scale, im * scale


def _sides(sign: int, plus: Any, minus: Any) -> Array:
    result: Array = 0.0
    if sign != -1:
        result = result + plus()
    if sign != 1:
        result = result + minus()
    return result


def _gaussian(
    xmin: float, xmax: float, mean: Array, sigma: Array, code: int, asymptotic: bool
) -> Array:
    xscale = ROOT2 * sigma
    xpmin, xpmax = (xmin - mean) / xscale, (xmax - mean) / xscale
    result = 1.0 + 0.0 * xpmin if asymptotic else 0.5 * (mf.erf(xpmax) - mf.erf(xpmin))
    return result * 2 if code and basis_sign(code) == 0 else result


def _parts(xp: tuple[Array, Array], u: tuple[Array, Array], c: Array) -> tuple[Array, ...]:
    """The ``erf``, Gaussian and ``exp * erfc`` terms at both ends, as both polynomials use them."""
    (xpmin, xpmax), (umin, umax) = xp, u
    f0 = mf.erf(-umax) - mf.erf(-umin)
    ea1, ea2 = libm.exp(-umax * umax), libm.exp(-umin * umin)
    tmp1, tmp2 = libm.exp(-xpmax) * mf.erfc(-umax + c), libm.exp(-xpmin) * mf.erfc(-umin + c)
    return f0, ea1, ea2, tmp1, tmp2, libm.exp(c * c)


def _linear(tau: float, xp: tuple[Array, Array], u: tuple[Array, Array], c: Array) -> Array:
    f0, ea1, ea2, tmp1, tmp2, expc2 = _parts(xp, u, c)
    f3 = xp[1] * tmp1 - xp[0] * tmp2
    return -tau * (
        f0 + (2 * c / ROOTPI) * (ea1 - ea2) + (1 - 2 * c * c) * expc2 * (tmp1 - tmp2) + expc2 * f3
    )


def _quadratic_head(f0: Array, f1: Array, f2: Array, c: Array) -> Array:
    return 2 * f0 + (4 * c / ROOTPI) * ((1 - c * c) * f1 + c * f2)


def _quadratic_ends(xp: tuple[Array, Array], tmp: tuple[Array, Array]) -> tuple[Array, Array]:
    """``x exp(-x) erfc`` and ``x^2 exp(-x) erfc`` differenced between the ends."""
    (xpmin, xpmax), (tmp1, tmp2) = xp, tmp
    return xpmax * tmp1 - xpmin * tmp2, xpmax * xpmax * tmp1 - xpmin * xpmin * tmp2


def _quadratic(tau: float, xp: tuple[Array, Array], u: tuple[Array, Array], c: Array) -> Array:
    f0, ea1, ea2, tmp1, tmp2, expc2 = _parts(xp, u, c)
    head = _quadratic_head(f0, ea1 - ea2, u[1] * ea1 - u[0] * ea2, c)
    f4, f5 = _quadratic_ends(xp, (tmp1, tmp2))
    third = (2 * c * c * (2 * c * c - 1) + 2) * expc2 * (tmp1 - tmp2)
    return -tau * (head + third - (4 * c * c - 2) * expc2 * f4 + expc2 * f5)


def _hyperbolic(
    kind: int, sign: int, tau: float, y: float, u: tuple[Array, Array], c: Array, asymptotic: bool
) -> Array:
    (umin, umax), sgn = u, (1.0 if kind == COSH else -1.0)

    def one(side: float, cut: float, lo: Array, hi: Array) -> Array:
        return cerf_int(side, 0.0, tau / (1 - cut), lo, hi, c * (1 - cut), asymptotic)[0]

    return _sides(
        sign,
        lambda: 0.5 * (one(+1, y, -umin, -umax) + sgn * one(+1, -y, -umin, -umax)),
        lambda: 0.5 * (sgn * one(-1, y, umin, umax) + one(-1, -y, umin, umax)),
    )


def _oscillating(
    kind: int, sign: int, tau: float, x: float, u: tuple[Array, Array], c: Array, asymptotic: bool
) -> Array:
    umin, umax = u
    part = 1 if kind == SIN else 0
    factor = -1.0 if kind == SIN else 1.0
    if kind == SIN and x == 0:
        return np.zeros(np.shape(c))
    return _sides(
        sign,
        lambda: factor * cerf_int(+1, -x, tau, -umin, -umax, c, asymptotic)[part],
        lambda: factor * cerf_int(-1, x, tau, umin, umax, c, asymptotic)[part],
    )


def _decay(
    kind: int,
    sign: int,
    bounds: tuple[float, float],
    mean: Array,
    sigma: Array,
    tau: float,
    p2: float,
    asymptotic: bool,
) -> Array:
    """The integral of the Gaussian convolved with a basis of lifetime ``tau``."""
    omega = p2 if kind in (SIN, COS) else 0.0
    c = sigma / (ROOT2 * tau)
    xp = ((bounds[0] - mean) / tau, (bounds[1] - mean) / tau)
    u = (xp[0] / (2 * c), xp[1] / (2 * c))
    if kind == EXP or (kind == COS and omega == 0.0):
        return _oscillating(COS, sign, tau, 0.0, u, c, asymptotic)
    if kind in (SIN, COS):
        return _oscillating(kind, sign, tau, omega * tau, u, c, asymptotic)
    if kind in (COSH, SINH):
        return _hyperbolic(kind, sign, tau, tau * p2 / 2, u, c, asymptotic)
    return (_linear if kind == LIN else _quadratic)(tau, xp, u, c)


def integral(
    bounds: tuple[float, float],
    mean: Array,
    sigma: Array,
    p1: float,
    p2: float,
    code: int,
    asymptotic: bool = False,
) -> Array:
    """The integral over the time, from ``bounds[0]`` to ``bounds[1]``, of convolution ``code``."""
    kind, sign = basis_type(code), basis_sign(code)
    tau = p1 if code else 0.0
    if kind == COSH and p2 == 0:
        kind = EXP
    if kind == NONE or (kind in (EXP, COS) and tau == 0.0):
        return _gaussian(bounds[0], bounds[1], mean, sigma, code, asymptotic)
    if tau == 0.0:
        return np.zeros(np.broadcast(mean, sigma).shape)
    return _decay(kind, sign, bounds, mean, sigma, tau, p2, asymptotic)
