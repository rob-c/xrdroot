"""``RooTruthModel::analyticalIntegral``: the basis functions' integrals, from their antiderivatives.

Each basis function of positive times has an antiderivative that is zero
below zero (``x -> max(x, 0)``); the negative-time half is the same
function of ``-x``, counted with the sign the basis's symmetry gives it,
exactly as ``definiteIntegral`` combines them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["definite"]

Array = Any


def _exp(x: Array, tau: Array, dm: Array) -> Array:
    return -tau * np.exp(-np.maximum(x, 0.0) / tau)


def _lin(x: Array, tau: Array, dm: Array) -> Array:
    x = np.maximum(x, 0.0)
    return -(tau + x) * np.exp(-x / tau)


def _quad(x: Array, tau: Array, dm: Array) -> Array:
    x = np.maximum(x, 0.0)
    return -(np.exp(-x / tau) * (2 * tau * tau + x * x + 2 * tau * x)) / tau


def _common(x: Array, tau: Array, dm: Array) -> Array:
    return tau * np.exp(-x / tau) / (dm * dm * tau * tau + 1.0)


def _common_hyperbolic(x: Array, tau: Array, dm: Array) -> Array:
    return 2 * tau * np.exp(-x / tau) / (dm * dm * tau * tau - 4.0)


def _sin(x: Array, tau: Array, dm: Array) -> Array:
    x = np.maximum(x, 0.0)
    fac = _common(x, tau, dm)
    return np.where(fac != 0.0, fac * (-tau * dm * np.cos(dm * x) - np.sin(dm * x)), 0.0)


def _cos(x: Array, tau: Array, dm: Array) -> Array:
    x = np.maximum(x, 0.0)
    fac = _common(x, tau, dm)
    return np.where(fac != 0.0, fac * (tau * dm * np.sin(dm * x) - np.cos(dm * x)), 0.0)


def _sinh(x: Array, tau: Array, dm: Array) -> Array:
    x = np.maximum(x, 0.0)
    fac, arg = _common_hyperbolic(x, tau, dm), 0.5 * dm * x
    return np.where(fac != 0.0, fac * (tau * dm * np.cosh(arg) - 2.0 * np.sinh(arg)), 0.0)


def _cosh(x: Array, tau: Array, dm: Array) -> Array:
    x = np.maximum(x, 0.0)
    fac, arg = _common_hyperbolic(x, tau, dm), 0.5 * dm * x
    return np.where(fac != 0.0, fac * (tau * dm * np.sinh(arg) + 2.0 * np.cosh(arg)), 0.0)


#: Per basis type: its antiderivative for positive times, and whether the basis is even in time.
ANTIDERIVATIVES: dict[int, tuple[Callable[[Array, Array, Array], Array], bool]] = {
    1: (_exp, True), 2: (_sin, False), 3: (_cos, True), 4: (_lin, False),
    5: (_quad, True), 6: (_cosh, True), 7: (_sinh, False),
}  # fmt: skip


def definite(kind: int, xmin: float, xmax: float, tau: Array, dm: Array, sign: int) -> Array:
    """``definiteIntegral``: the integral from ``xmin`` to ``xmax`` of the basis of type ``kind``."""
    antiderivative, symmetric = ANTIDERIVATIVES[kind]
    tau = np.asarray(tau, dtype=np.float64)
    with np.errstate(all="ignore"):
        result: Array = 0.0
        if sign != -1:
            result = result + (antiderivative(xmax, tau, dm) - antiderivative(xmin, tau, dm))
        if sign != 1:
            minus = antiderivative(-xmax, tau, dm) - antiderivative(-xmin, tau, dm)
            result = result + (-minus if symmetric else minus)
    found = np.where(tau == 0.0, 1.0 if symmetric else 0.0, result)
    return found if found.ndim else float(found)
