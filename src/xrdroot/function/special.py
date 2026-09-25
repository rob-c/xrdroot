"""The special functions a fit is written in, the way ROOT's C++ computes them.

``TMath`` and ``ROOT::Math`` define the shapes people fit - the Gaussian, the
Landau, the Crystal Ball, the Breit-Wigner, Chebyshev series - and each has
edge cases of its own: a width of zero, a tail cut off where the double
underflows, a normalisation only defined for some exponents. Each function
here is ROOT's, operation for operation, over whole arrays: the arguments
arrive broadcast against each other and the answer comes back the same shape.

The Landau density is CERNLIB's ``DENLAN`` rational approximation, which is
what ``ROOT::Math::landau_pdf`` evaluates; the others are closed forms.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "gaus",
    "landau",
    "landau_pdf",
    "breit_wigner",
    "breitwigner_pdf",
    "crystalball_function",
    "crystalball_pdf",
    "bigaussian_pdf",
    "chebyshev",
    "gaussian_pdf",
]

Array = Any

#: ``sqrt(2*pi)``, spelled as ``TMath::Gaus`` spells it.
SQRT_TWO_PI = 2.50662827463100024
#: Beyond this many widths ``TMath::Gaus`` answers zero rather than underflow.
GAUS_CUT = 39.0
#: What ``TMath::Gaus`` gives for a width of zero, rather than dividing by it.
GAUS_ZERO_WIDTH = 1.0e30

#: ``DENLAN``'s coefficients, a numerator and a denominator per region of
#: the argument, lowest power first.
LANDAU_P = (
    (0.4259894875, -0.1249762550, 0.03984243700, -0.006298287635, 0.001511162253),
    (0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411, 0.0001283617211),
    (0.1788544503, 0.09359161662, 0.006325387654, 0.00006611667319, -0.000002031049101),
    (0.9874054407, 118.6723273, 849.2794360, -743.7792444, 427.0262186),
    (1.003675074, 167.5702434, 4789.711289, 21217.86767, -22324.94910),
    (1.000827619, 664.9143136, 62972.92665, 475554.6998, -5743609.109),
)
LANDAU_Q = (
    (1.0, -0.3388260629, 0.09594393323, -0.01608042283, 0.003778942063),
    (1.0, 0.7428795082, 0.3153932961, 0.06694219548, 0.008790609714),
    (1.0, 0.6097809921, 0.2560616665, 0.04746722384, 0.006957301675),
    (1.0, 106.8615961, 337.6496214, 2016.712389, 1597.063511),
    (1.0, 156.9424537, 3745.310488, 9834.698876, 66924.28357),
    (1.0, 651.4101098, 56974.73333, 165917.4725, -2815759.939),
)
#: The far left tail's series, and the far right tail's.
LANDAU_A1 = (0.04166666667, -0.01996527778, 0.02709538966)
LANDAU_A2 = (-1.845568670, -4.284640743)
#: Where each of ``DENLAN``'s regions ends, left to right.
LANDAU_EDGES = (-5.5, -1.0, 1.0, 5.0, 12.0, 50.0, 300.0)


def _horner(coefficients: tuple[float, ...], v: Array) -> Array:
    """``c0 + (c1 + (c2 + ...)*v)*v``, nested the way ``DENLAN`` nests it."""
    total = coefficients[-1]
    for c in reversed(coefficients[:-1]):
        total = c + total * v
    return total


def _ratio(region: int, v: Array) -> Array:
    return _horner(LANDAU_P[region], v) / _horner(LANDAU_Q[region], v)


def _landau_left(v: Array) -> list[Array]:
    """The density by the formulas of the two regions left of ``-1``."""
    u_left = np.exp(v + 1.0)
    far_left = (
        0.3989422803
        * (np.exp(-1 / u_left) / np.sqrt(u_left))
        * (1 + (LANDAU_A1[0] + (LANDAU_A1[1] + LANDAU_A1[2] * u_left) * u_left) * u_left)
    )
    far_left = np.where(u_left < 1e-10, 0.0, far_left)
    u_near = np.exp(-v - 1)
    return [far_left, np.exp(-u_near) * np.sqrt(u_near) * _ratio(0, v)]


def _landau_right(v: Array) -> list[Array]:
    """The density by the formulas of the regions from ``-1`` rightwards."""
    inverse = 1 / v
    tails = [inverse * inverse * _ratio(region, inverse) for region in (3, 4, 5)]
    u_far = 1 / (v - v * np.log(v) / (v + 1))
    far_right = u_far * u_far * (1 + (LANDAU_A2[0] + LANDAU_A2[1] * u_far) * u_far)
    return [_ratio(1, v), _ratio(2, v), *tails, far_right]


def landau_pdf(x: Array, xi: Array = 1.0, x0: Array = 0.0) -> Array:
    """``ROOT::Math::landau_pdf``: the Landau density, located at ``x0``, of scale ``xi``."""
    v = (np.asarray(x, np.float64) - x0) / np.where(np.asarray(xi) > 0, xi, 1.0)
    with np.errstate(all="ignore"):
        regions = _landau_left(v) + _landau_right(v)
    chosen = np.select([v < edge for edge in LANDAU_EDGES], regions[:-1], regions[-1])
    return np.where(np.asarray(xi) <= 0, 0.0, chosen / np.where(np.asarray(xi) > 0, xi, 1.0))


def landau(x: Array, mpv: Array = 0.0, sigma: Array = 1.0, norm: Array = 0.0) -> Array:
    """``TMath::Landau``: the density at ``(x - mpv)/sigma``, divided by ``sigma`` when ``norm``.

    A width that is not positive gives zero, as ROOT gives it.
    """
    sigma = np.asarray(sigma, np.float64)
    safe = np.where(sigma > 0, sigma, 1.0)
    density = landau_pdf((np.asarray(x, np.float64) - mpv) / safe)
    density = np.where(np.asarray(norm) != 0, density / safe, density)
    return np.where(sigma <= 0, 0.0, density)


def gaus(x: Array, mean: Array = 0.0, sigma: Array = 1.0, norm: Array = 0.0) -> Array:
    """``TMath::Gaus``: ``exp(-arg*arg/2)`` for ``arg = (x - mean)/sigma``, over
    ``sqrt(2 pi) sigma`` when ``norm``; zero past 39 widths, ``1e30`` for no width."""
    sigma = np.asarray(sigma, np.float64)
    safe = np.where(sigma == 0, 1.0, sigma)
    arg = (np.asarray(x, np.float64) - mean) / safe
    with np.errstate(over="ignore", under="ignore"):
        res = np.exp(-0.5 * arg * arg)
    res = np.where((arg < -GAUS_CUT) | (arg > GAUS_CUT), 0.0, res)
    res = np.where(np.asarray(norm) != 0, res / (SQRT_TWO_PI * safe), res)
    return np.where(sigma == 0, GAUS_ZERO_WIDTH, res)


def gaussian_pdf(x: Array, sigma: Array = 1.0, x0: Array = 0.0) -> Array:
    """``ROOT::Math::gaussian_pdf``: the normal density of width ``sigma`` about ``x0``."""
    tmp = (np.asarray(x, np.float64) - x0) / sigma
    return (1.0 / (math.sqrt(2 * math.pi) * np.abs(sigma))) * np.exp(-tmp * tmp / 2)


def breit_wigner(x: Array, mean: Array = 0.0, gamma: Array = 1.0) -> Array:
    """``TMath::BreitWigner``: ``gamma/((x-mean)^2 + gamma^2/4)/(2 pi)``."""
    d = np.asarray(x, np.float64) - mean
    bw = gamma / (d * d + gamma * gamma / 4)
    return bw / (2 * math.pi)


def breitwigner_pdf(x: Array, gamma: Array, x0: Array = 0.0) -> Array:
    """``ROOT::Math::breitwigner_pdf``: the Cauchy density of full width ``gamma``."""
    half = np.asarray(gamma, np.float64) / 2.0
    d = np.asarray(x, np.float64) - x0
    return half / (math.pi * (d * d + half * half))


def crystalball_function(
    x: Array, alpha: Array, n: Array, sigma: Array, mean: Array = 0.0
) -> Array:
    """``ROOT::Math::crystalball_function``: a Gaussian core with a power-law tail.

    The tail is on the left for a positive ``alpha`` and on the right for a
    negative one, starting ``|alpha|`` widths from the mean; a negative width
    gives zero.
    """
    sigma = np.asarray(sigma, np.float64)
    z = (np.asarray(x, np.float64) - mean) / np.where(sigma == 0, 1.0, sigma)
    z = np.where(np.asarray(alpha) < 0, -z, z)
    abs_alpha = np.abs(alpha)
    with np.errstate(all="ignore"):
        core = np.exp(-0.5 * z * z)
        over = n / abs_alpha
        tail = np.exp(-0.5 * abs_alpha * abs_alpha) * np.power(over / (over - abs_alpha - z), n)
    value = np.where(z > -abs_alpha, core, tail)
    return np.where(sigma < 0.0, 0.0, value)


def crystalball_pdf(x: Array, alpha: Array, n: Array, sigma: Array, mean: Array = 0.0) -> Array:
    """``ROOT::Math::crystalball_pdf``: the Crystal Ball normalised, defined only for ``n > 1``."""
    abs_alpha = np.abs(alpha)
    with np.errstate(all="ignore"):
        c = n / abs_alpha * 1.0 / (n - 1.0) * np.exp(-(np.asarray(alpha) ** 2) / 2.0)
        d = math.sqrt(math.pi / 2.0) * (1.0 + _erf(abs_alpha / math.sqrt(2.0)))
        norm = 1.0 / (sigma * (c + d))
    value = norm * crystalball_function(x, alpha, n, sigma, mean)
    value = np.where(np.asarray(n) <= 1, np.nan, value)
    return np.where(np.asarray(sigma) < 0.0, 0.0, value)


def _erf(values: Array) -> Array:
    return np.vectorize(math.erf, otypes=[np.float64])(values)


def bigaussian_pdf(
    x: Array,
    y: Array,
    sigmax: Array = 1.0,
    sigmay: Array = 1.0,
    rho: Array = 0.0,
    x0: Array = 0.0,
    y0: Array = 0.0,
) -> Array:
    """``ROOT::Math::bigaussian_pdf``: two correlated normals, ``rho`` their correlation."""
    u = (np.asarray(x, np.float64) - x0) / sigmax
    v = (np.asarray(y, np.float64) - y0) / sigmay
    c = 1.0 - rho * rho
    z = u * u - 2.0 * rho * u * v + v * v
    return 1.0 / (2 * math.pi * sigmax * sigmay * np.sqrt(c)) * np.exp(-z / (2.0 * c))


def chebyshev(x: Array, *coefficients: Array) -> Array:
    """``ROOT::Math::ChebyshevN``: ``sum c_i T_i(x)``, by Clenshaw's recurrence as ROOT sums it."""
    x = np.asarray(x, np.float64)
    if len(coefficients) == 1:
        return coefficients[0] + 0.0 * x
    if len(coefficients) == 2:
        return coefficients[0] + coefficients[1] * x
    d1: Array = 0.0
    d2: Array = 0.0
    y2 = 2.0 * x
    for c in reversed(coefficients[1:]):
        d1, d2 = y2 * d1 - d2 + c, d1
    return x * d1 - d2 + coefficients[0]
