"""The Legendre polynomials of ``ROOT::Math``: plain, associated and spherical.

Each is the three-term recurrence C++17's ``<cmath>`` specifies, which is what
ROOT's ``ROOT::Math::legendre``, ``assoc_legendre`` and ``sph_legendre`` call.
"""

from __future__ import annotations

import math

__all__ = ["legendre", "assoc_legendre", "sph_legendre"]


def legendre(l: int, x: float) -> float:
    """``ROOT::Math::legendre``: the Legendre polynomial of degree ``l`` at ``x``."""
    previous, current = 1.0, float(x)
    if int(l) == 0:
        return previous
    for degree in range(1, int(l)):
        following = (2 * degree + 1) * x * current - degree * previous
        previous, current = current, following / (degree + 1)
    return current


def assoc_legendre(l: int, m: int, x: float) -> float:
    """``ROOT::Math::assoc_legendre``: ``P_l^m(x)``, without Condon-Shortley's phase, as C++17's."""
    l, m = int(l), int(m)
    if m > l:
        return math.nan  # undefined, as ROOT's GSL says
    start = math.prod(range(1, 2 * m, 2)) * (1.0 - x * x) ** (m / 2.0)
    if l == m:
        return float(start)
    previous, current = start, x * (2 * m + 1) * start
    for degree in range(m + 1, l):
        following = (2 * degree + 1) * x * current - (degree + m) * previous
        previous, current = current, following / (degree - m + 1)
    return float(current)


def sph_legendre(l: int, m: int, theta: float) -> float:
    """``ROOT::Math::sph_legendre``: the spherical harmonic ``Y_l^m(theta, 0)``, with its phase."""
    l, m = int(l), abs(int(m))
    norm = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m))
    return float((-1) ** m * norm * assoc_legendre(l, m, math.cos(theta)))
