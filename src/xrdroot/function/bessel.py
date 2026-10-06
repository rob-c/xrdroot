"""Bessel functions of real order, as ``ROOT::Math``'s MathMore names them, from their integrals.

``cyl_bessel_j``, ``cyl_bessel_i`` and ``cyl_bessel_k`` are the ordinary
and modified Bessel functions of the first and second kind, and
``sph_bessel`` the spherical one. Each is its integral representation -
over half a period, and over the half line, where the integrand falls
off as ``exp(-x cosh t)`` - summed by Gauss-Legendre, out to where nothing
is left of it. The integrands are smooth, so a few units in the last
place of a double are reached over the orders and arguments plots use;
GSL, which ROOT calls, is not needed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["cyl_bessel_j", "cyl_bessel_i", "cyl_bessel_k", "sph_bessel"]

Array = Any

#: How many Gauss-Legendre nodes each integral is summed at.
NODES = 160
#: Where ``exp(-x cosh t)`` is too small to count: its exponent's least size.
CUTOFF = 60.0


#: The Gauss-Legendre nodes and weights every integral is summed with, on [0, 1].
_POINTS, _FACTORS = np.polynomial.legendre.leggauss(NODES)
_NODES, _WEIGHTS = 0.5 * (_POINTS + 1.0), 0.5 * _FACTORS


def _period(integrand: Any) -> Array:
    """The integral over ``[0, pi]`` of a smooth integrand."""
    return math.pi * (integrand(math.pi * _NODES) * _WEIGHTS).sum(axis=-1)


def _tail(x: Array, integrand: Any, decay: Array | None = None) -> Array:
    """The integral over ``[0, inf)`` of an integrand falling as ``exp(-x cosh t)`` - and as
    ``exp(-decay t)`` too, where that is given and gone sooner - by Gauss-Legendre."""
    top = np.arccosh(np.maximum(CUTOFF / np.maximum(x, 1e-300), 1.0)) + 1.0
    if decay is not None:
        top = np.minimum(top, CUTOFF / np.maximum(decay, 1e-300) + 1.0)
    with np.errstate(all="ignore"):
        values = integrand(top[..., None] * _NODES)
    return top * (values * _WEIGHTS).sum(axis=-1)


def _arguments(nu: Array, x: Array) -> tuple[Array, Array]:
    order, at = np.broadcast_arrays(np.asarray(nu, dtype=np.float64),
                                    np.asarray(x, dtype=np.float64))  # fmt: skip
    return order[..., None], at[..., None]


def cyl_bessel_j(nu: Array, x: Array) -> Array:
    """``J_nu(x)``: Bessel's integral, and its correction where ``nu`` is not whole."""
    order, at = _arguments(nu, x)
    main = _period(lambda tau: np.cos(order * tau - at * np.sin(tau))) / math.pi
    rest = _tail(at[..., 0], lambda t: np.exp(-at * np.sinh(t) - order * t), order[..., 0])
    return main - np.sin(order[..., 0] * math.pi) / math.pi * rest


def cyl_bessel_i(nu: Array, x: Array) -> Array:
    """``I_nu(x)``, the modified Bessel function of the first kind."""
    order, at = _arguments(nu, x)
    main = _period(lambda tau: np.exp(at * np.cos(tau)) * np.cos(order * tau)) / math.pi
    rest = _tail(at[..., 0], lambda t: np.exp(-at * np.cosh(t) - order * t), order[..., 0])
    return main - np.sin(order[..., 0] * math.pi) / math.pi * rest


def cyl_bessel_k(nu: Array, x: Array) -> Array:
    """``K_nu(x)``, the modified Bessel function of the second kind: infinite at zero."""
    order, at = _arguments(nu, x)
    found = _tail(at[..., 0], lambda t: np.exp(-at * np.cosh(t)) * np.cosh(order * t))
    return np.where(at[..., 0] > 0, found, np.inf)


def sph_bessel(n: Array, x: Array) -> Array:
    """``j_n(x)``, the spherical Bessel function: ``J`` of half an order more, scaled."""
    order, at = np.broadcast_arrays(np.asarray(n, dtype=np.float64),
                                    np.asarray(x, dtype=np.float64))  # fmt: skip
    safe = np.where(at > 0, at, 1.0)
    found = np.sqrt(math.pi / (2.0 * safe)) * cyl_bessel_j(order + 0.5, safe)
    return np.where(at > 0, found, np.where(order == 0, 1.0, 0.0))
