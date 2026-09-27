"""RooFit's formulas for its densities and their integrals, from ``RooFit/Detail/MathFuncs.h``.

A density's integral in closed form is a formula RooFit chose - an ``erfc``
in the upper tail for a Gaussian, where it is most precise - and a fit
divides by it at every step, so these are those formulas, term for term.
Each takes NumPy arrays or numbers.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..random import libm

__all__ = [
    "erf",
    "erfc",
    "lgamma",
    "gaussian_integral",
    "exponential_integral",
    "polynomial",
    "polynomial_integral",
    "chebychev",
    "chebychev_integral",
    "cb_shape",
    "cb_shape_integral",
    "bifurgauss_integral",
    "approx_erf",
]

Array = Any

_erf = np.vectorize(math.erf, otypes=[np.float64])
_erfc = np.vectorize(math.erfc, otypes=[np.float64])
_lgamma = np.vectorize(math.lgamma, otypes=[np.float64])


def erf(x: Array) -> Array:
    return math.erf(x) if np.ndim(x) == 0 else _erf(x)


def erfc(x: Array) -> Array:
    return math.erfc(x) if np.ndim(x) == 0 else _erfc(x)


def lgamma(x: Array) -> Array:
    return math.lgamma(x) if np.ndim(x) == 0 else _lgamma(x)


def approx_erf(arg: Array) -> Array:
    """``approxErf``: ``erf``, taken as exactly one beyond five."""
    return np.where(arg > 5.0, 1.0, np.where(arg < -5.0, -1.0, erf(arg)))


def gaussian_integral(low: Array, high: Array, mean: Array, sigma: Array) -> Array:
    """``gaussianIntegral``: every case mapped to the upper tail, where ``erfc`` is most precise."""
    scale = 0.5 * math.sqrt(2 * math.pi) * sigma
    xscale = math.sqrt(2.0) * sigma
    smin = (low - mean) / xscale
    smax = (high - mean) / xscale
    ecmin = erfc(np.abs(smin))
    ecmax = erfc(np.abs(smax))
    cond = np.where(
        smin * smax < 0.0,
        2.0 - (ecmin + ecmax),
        np.where(smax <= 0.0, ecmax - ecmin, ecmin - ecmax),
    )
    return scale * cond


def exponential_integral(low: Array, high: Array, c: Array) -> Array:
    with np.errstate(divide="ignore", invalid="ignore"):
        found = (libm.exp(c * high) - libm.exp(c * low)) / c
    return np.where(c == 0.0, high - low, found)


def polynomial(coefs: list[Array], lowest: int, x: Array, pdf_mode: bool) -> Array:
    """``polynomial``: Horner's rule, times ``x`` to the lowest order, plus one for a density."""
    value = coefs[-1] if coefs else 0.0
    for c in reversed(coefs[:-1]):
        value = c + x * value
    value = value * x**lowest if lowest else value
    return value + (1.0 if pdf_mode and lowest > 0 else 0.0)


def polynomial_integral(
    coefs: list[Array], lowest: int, low: Array, high: Array, pdf_mode: bool
) -> Array:
    if not coefs:
        return (high - low) if pdf_mode and lowest > 0 else 0.0
    denom = lowest + len(coefs)
    bottom = top = coefs[-1] / float(denom)
    for c in reversed(coefs[:-1]):
        denom -= 1
        bottom = c / float(denom) + low * bottom
        top = c / float(denom) + high * top
    top = top * high ** (1 + lowest)
    bottom = bottom * low ** (1 + lowest)
    return top - bottom + ((high - low) if pdf_mode and lowest > 0 else 0.0)


def chebychev(coefs: list[Array], x: Array, low: float, high: float) -> Array:
    """``chebychev``: one plus the coefficients times T1, T2... of ``x`` mapped onto [-1, 1]."""
    xp = (x - 0.5 * (high + low)) / (0.5 * (high - low))
    total: Array = 1.0
    last, curr = 1.0, xp
    twox = 2 * xp
    curr, last = twox * curr - last, curr
    for c in coefs:
        total = total + last * c
        curr, last = twox * curr - last, curr
    return total


def chebychev_integral(
    coefs: list[Array], low: float, high: float, full_low: float, full_high: float
) -> Array:
    """``chebychevIntegral``: over [``full_low``, ``full_high``] of the series on [low, high]."""
    half = 0.5 * (high - low)
    mid = 0.5 * (high + low)
    b = (full_high - mid) / half
    a = (full_low - mid) / half
    total: Array = b - a
    if coefs:
        total = 0.5 * (b + a) * (b - a) * coefs[0] + total
    if len(coefs) > 1:
        total = _chebychev_rest(coefs, a, b, total)
    return half * total


def _chebychev_rest(coefs: list[Array], a: float, b: float, total: Array) -> Array:
    bcurr, blast, acurr, alast = b, 1.0, a, 1.0
    acurr, alast = 2 * a * acurr - alast, acurr
    bcurr, blast = 2 * b * bcurr - blast, bcurr
    nminus1 = 1.0
    for c in coefs[1:]:
        term2 = (blast - alast) / nminus1
        acurr, alast = 2 * a * acurr - alast, acurr
        bcurr, blast = 2 * b * bcurr - blast, bcurr
        nminus1 += 1
        term1 = (bcurr - acurr) / (nminus1 + 1.0)
        total = 0.5 * (term1 - term2) * c + total
    return total


def cb_shape(m: Array, m0: Array, sigma: Array, alpha: Array, n: Array) -> Array:
    """``cbShape``: a Gaussian core and a power-law tail beyond ``alpha`` widths."""
    t = (m - m0) / sigma
    t = np.where(alpha < 0, -t, t)
    abs_alpha = np.abs(alpha)
    r = n / abs_alpha
    a = libm.exp(-0.5 * abs_alpha * abs_alpha)
    b = r - abs_alpha
    with np.errstate(all="ignore"):
        tail = a * libm.power(r / (b - t), n)
    return np.where(t >= -abs_alpha, libm.exp(-0.5 * t * t), tail)


def cb_shape_integral(
    low: float, high: float, m0: float, sigma: float, alpha: float, n: float
) -> float:
    """``cbShapeIntegral``, over the core, the tail, or both."""
    sqrt_pi_over2, sqrt2 = 1.2533141373, 1.4142135624
    sig = abs(sigma)
    tmin, tmax = (low - m0) / sig, (high - m0) / sig
    if alpha < 0:
        tmin, tmax = -tmax, -tmin
    abs_alpha = abs(alpha)
    if tmin >= -abs_alpha:
        result = sig * sqrt_pi_over2 * (approx_erf(tmax / sqrt2) - approx_erf(tmin / sqrt2))
    elif tmax <= -abs_alpha:
        result = _cb_tail(tmin, tmax, sig, abs_alpha, n)
    else:
        core = sig * sqrt_pi_over2 * (approx_erf(tmax / sqrt2) - approx_erf(-abs_alpha / sqrt2))
        result = _cb_tail(tmin, None, sig, abs_alpha, n) + core
    return float(result) if result != 0 else 1e-300


def _cb_log_tail(scale: float, lmin: float, lmax: float, n: float) -> float:
    """The tail's part for ``n`` near one, where its power becomes a logarithm."""
    return float(scale * (lmin - lmax + 0.5 * (1.0 - n) * (lmin * lmin - lmax * lmax)))


def _cb_tail(tmin: float, tmax: float | None, sig: float, abs_alpha: float, n: float) -> float:
    """The tail's part: from ``tmin`` to ``tmax``, or to where the core starts for ``None``."""
    r = n / abs_alpha
    a = r * math.exp(-0.5 * abs_alpha * abs_alpha)
    b = r - abs_alpha
    if abs(n - 1.0) < 1.0e-05:
        lmin = math.log(b - tmin)
        lmax = math.log(r) if tmax is None else math.log(b - tmax)
        return _cb_log_tail(a * r ** (n - 1) * sig, lmin, lmax, n)
    upper = 1.0 if tmax is None else (r / (b - tmax)) ** (n - 1.0)
    return float(a * sig / (1.0 - n) * ((r / (b - tmin)) ** (n - 1.0) - upper))


def bifurgauss_integral(low: float, high: float, mean: float, left: float, right: float) -> float:
    xl, xr = math.sqrt(2.0) * left, math.sqrt(2.0) * right
    scale = 0.5 * math.sqrt(2 * math.pi)
    if high < mean:
        return scale * left * (math.erf((high - mean) / xl) - math.erf((low - mean) / xl))
    if low > mean:
        return scale * right * (math.erf((high - mean) / xr) - math.erf((low - mean) / xr))
    return scale * (right * math.erf((high - mean) / xr) - left * math.erf((low - mean) / xl))
