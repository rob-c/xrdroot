"""``RooGaussModel::analyticalIntegral``: the Gaussian's convolutions integrated over the time.

Every form is RooFit's, case by case: the plain Gaussian (an ``erf``), the
exponential, sine, cosine and hyperbolic bases (differences of the complex
error function, ``evalCerfInt``), and the linear and quadratic bases
(``erf``/``erfc`` combinations).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import mathfuncs as mf
from ..cerf import eval_cerf
from .gaussmodel import ROOT2, ROOTPI
from .resolution import COS, COSH, EXP, LIN, NONE, SIN, SINH, basis_sign, basis_type

__all__ = ["integral"]

Array = Any


def cerf_int(sign: float, x: float, tau: float, umin: Array, umax: Array, c: Array,
             asymptotic: bool = False) -> tuple[Array, Array]:  # fmt: skip
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


def _gaussian(xmin: float, xmax: float, mean: Array, sigma: Array, code: int, asymptotic: bool) -> Array:
    xscale = ROOT2 * sigma
    xpmin, xpmax = (xmin - mean) / xscale, (xmax - mean) / xscale
    result = 1.0 + 0.0 * xpmin if asymptotic else 0.5 * (mf.erf(xpmax) - mf.erf(xpmin))
    return result * 2 if code and basis_sign(code) == 0 else result


def _polynomial(kind: int, tau: float, xp: tuple[Array, Array], u: tuple[Array, Array], c: Array) -> Array:
    (xpmin, xpmax), (umin, umax) = xp, u
    f0 = mf.erf(-umax) - mf.erf(-umin)
    ea1, ea2 = np.exp(-umax * umax), np.exp(-umin * umin)
    tmp1, tmp2 = np.exp(-xpmax) * mf.erfc(-umax + c), np.exp(-xpmin) * mf.erfc(-umin + c)
    expc2 = np.exp(c * c)
    if kind == LIN:
        return -tau * (f0 + (2 * c / ROOTPI) * (ea1 - ea2) + (1 - 2 * c * c) * expc2 * (tmp1 - tmp2)
                       + expc2 * (xpmax * tmp1 - xpmin * tmp2))  # fmt: skip
    f2 = umax * ea1 - umin * ea2
    f4, f5 = xpmax * tmp1 - xpmin * tmp2, xpmax * xpmax * tmp1 - xpmin * xpmin * tmp2
    return -tau * (2 * f0 + (4 * c / ROOTPI) * ((1 - c * c) * (ea1 - ea2) + c * f2)
                   + (2 * c * c * (2 * c * c - 1) + 2) * expc2 * (tmp1 - tmp2) - (4 * c * c - 2) * expc2 * f4
                   + expc2 * f5)  # fmt: skip


def _hyperbolic(kind: int, sign: int, tau: float, y: float, u: tuple[Array, Array], c: Array,
                asymptotic: bool) -> Array:  # fmt: skip
    (umin, umax), sgn = u, (1.0 if kind == COSH else -1.0)

    def one(side: float, cut: float, lo: Array, hi: Array) -> Array:
        return cerf_int(side, 0.0, tau / (1 - cut), lo, hi, c * (1 - cut), asymptotic)[0]

    return _sides(sign, lambda: 0.5 * (one(+1, y, -umin, -umax) + sgn * one(+1, -y, -umin, -umax)),
                  lambda: 0.5 * (sgn * one(-1, y, umin, umax) + one(-1, -y, umin, umax)))  # fmt: skip


def _oscillating(kind: int, sign: int, tau: float, x: float, u: tuple[Array, Array], c: Array,
                 asymptotic: bool) -> Array:  # fmt: skip
    umin, umax = u
    part = 1 if kind == SIN else 0
    factor = -1.0 if kind == SIN else 1.0
    if kind == SIN and x == 0:
        return 0.0 * c
    return _sides(sign, lambda: factor * cerf_int(+1, -x, tau, -umin, -umax, c, asymptotic)[part],
                  lambda: factor * cerf_int(-1, x, tau, umin, umax, c, asymptotic)[part])  # fmt: skip


def integral(bounds: tuple[float, float], mean: Array, sigma: Array, p1: float, p2: float, code: int,
             asymptotic: bool = False) -> Array:  # fmt: skip
    """The integral over the time from ``bounds[0]`` to ``bounds[1]`` of the convolution of ``code``."""
    kind, sign = basis_type(code), basis_sign(code)
    tau = p1 if code else 0.0
    if kind == COSH and p2 == 0:
        kind = EXP
    xmin, xmax = bounds
    if kind == NONE or (kind in (EXP, COS) and tau == 0.0):
        return _gaussian(xmin, xmax, mean, sigma, code, asymptotic)
    omega = p2 if kind in (SIN, COS) else 0.0
    if tau == 0.0:
        return 0.0 * sigma
    c = sigma / (ROOT2 * tau)
    xp = ((xmin - mean) / tau, (xmax - mean) / tau)
    u = (xp[0] / (2 * c), xp[1] / (2 * c))
    if kind == EXP or (kind == COS and omega == 0.0):
        return _oscillating(COS, sign, tau, 0.0, u, c, asymptotic)
    if kind in (SIN, COS):
        return _oscillating(kind, sign, tau, omega * tau, u, c, asymptotic)
    if kind in (COSH, SINH):
        return _hyperbolic(kind, sign, tau, tau * p2 / 2, u, c, asymptotic)
    return _polynomial(kind, tau, xp, u, c)
