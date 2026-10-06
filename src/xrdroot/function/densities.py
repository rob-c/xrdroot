"""``TMath``'s densities and distribution functions, element by element, as a formula calls them.

``TMath::BetaDist(x, [0], [1])`` in a ``TF1`` is evaluated over every point
drawn at once, so each density of :mod:`.distributions` is taken element by
element here, in ``TMath``'s argument order: ``GammaDist(x, gamma, mu,
beta)`` is the gamma density of shape ``gamma`` and scale ``beta`` from
``mu``. The Laplace densities, the relativistic Breit-Wigner and the
crystal ball's distribution function are the closed forms ROOT's are.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import distributions as dist

__all__ = [
    "beta_dist", "beta_dist_i", "gamma_dist", "log_normal", "student", "student_i", "f_dist",
    "f_dist_i", "laplace_dist", "laplace_dist_i", "breit_wigner_relativistic",
    "crystalball_cdf", "crystalball_cdf_c",
]  # fmt: skip

Array = Any


def _each(function: Any) -> Any:
    """``function`` of numbers, taken element by element over arrays, giving doubles."""
    return np.vectorize(function, otypes=[np.float64])


beta_dist = _each(dist.beta_pdf)
beta_dist_i = _each(dist.beta_cdf)
student = _each(dist.student_pdf)
student_i = _each(dist.student_cdf)
f_dist = _each(dist.f_pdf)
f_dist_i = _each(dist.f_cdf)


def gamma_dist(x: Array, gamma: Array, mu: Array = 0.0, beta: Array = 1.0) -> Array:
    """``TMath::GammaDist(x, gamma, mu, beta)``: shape ``gamma``, scale ``beta``, from ``mu``."""
    return _each(dist.gamma_pdf)(x, gamma, beta, mu)


def log_normal(x: Array, sigma: Array, theta: Array = 0.0, m: Array = 1.0) -> Array:
    """``TMath::LogNormal(x, sigma, theta, m)``: of ``log(x - theta)``, centred on ``log m``."""
    return _each(dist.lognormal_pdf)(x, np.log(m), sigma, theta)


def laplace_dist(x: Array, alpha: Array = 0.0, beta: Array = 1.0) -> Array:
    """``TMath::LaplaceDist``: the double exponential about ``alpha``, of width ``beta``."""
    return np.exp(-np.abs((np.asarray(x) - alpha) / beta)) / (2.0 * beta)


def laplace_dist_i(x: Array, alpha: Array = 0.0, beta: Array = 1.0) -> Array:
    """``TMath::LaplaceDistI``: its distribution function."""
    half = 0.5 * np.exp(-np.abs((np.asarray(x) - alpha) / beta))
    return np.where(np.asarray(x) <= alpha, half, 1.0 - half)


def breit_wigner_relativistic(x: Array, median: Array = 0.0, gamma: Array = 1.0) -> Array:
    """``TMath::BreitWignerRelativistic``: the resonance of mass ``median`` and width ``gamma``."""
    mm, gg, mg = median * median, gamma * gamma, median * gamma
    y = np.sqrt(mm * (mm + gg))
    k = (2.0 * math.sqrt(2.0) / math.pi * mg * y) / np.sqrt(mm + y)
    away = np.asarray(x) * x - mm
    return k / (away * away + mg * mg)


def _crystalball_left(z: Array, a: Array, n: Array) -> tuple[Array, Array]:
    """The crystal ball's integral from minus infinity to ``z``, its tail on the left, and its
    whole integral - both in units of its width."""
    power = _power_tail(-a, a, n)
    gauss = math.sqrt(math.pi / 2.0)
    core = power + gauss * (_erf(z / math.sqrt(2.0)) - _erf(-a / math.sqrt(2.0)))
    whole = power + gauss * (1.0 + _erf(a / math.sqrt(2.0)))
    return np.where(z <= -a, _power_tail(z, a, n), core), whole


def _power_tail(z: Array, a: Array, n: Array) -> Array:
    """The integral of the crystal ball's power-law tail from minus infinity to ``z <= -a``."""
    scale = (n / a) ** n * np.exp(-0.5 * a * a)
    return scale * (n / a - a - z) ** (1.0 - n) / (n - 1.0)


def _erf(values: Array) -> Array:
    return np.vectorize(math.erf, otypes=[np.float64])(values)


def _crystalball(x: Array, alpha: Array, n: Array, sigma: Array, mean: Array) -> Array:
    """The chance of below ``x``: the left integral over the whole, mirrored for ``alpha < 0``."""
    z = (np.asarray(x, dtype=np.float64) - mean) / sigma
    flipped = np.asarray(alpha) < 0
    with np.errstate(all="ignore"):  # n = 1 has no finite integral: answered by zero below
        left, whole = _crystalball_left(np.where(flipped, -z, z), np.abs(alpha), np.asarray(n))
        below = np.where(flipped, 1.0 - left / whole, left / whole)
    return np.where(np.asarray(n) > 1.0, below, 0.0)


def crystalball_cdf(x: Array, alpha: Array, n: Array, sigma: Array, mean: Array = 0.0) -> Array:
    """``ROOT::Math::crystalball_cdf``: zero where ``n <= 1``, which has no finite integral."""
    return _crystalball(x, alpha, n, sigma, mean)


def crystalball_cdf_c(x: Array, alpha: Array, n: Array, sigma: Array, mean: Array = 0.0) -> Array:
    """``ROOT::Math::crystalball_cdf_c``: the chance of above ``x``."""
    return np.where(np.asarray(n) > 1.0, 1.0 - _crystalball(x, alpha, n, sigma, mean), 0.0)
