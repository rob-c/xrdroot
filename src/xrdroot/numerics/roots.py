"""GSL's Brent root solver, as ``ROOT::Math::RootFinder`` with ``kGSL_BRENT`` drives it.

``gsl_root_fsolver_brent``'s steps - inverse quadratic interpolation where
it is safe, bisection where it is not - repeated until
``gsl_root_test_interval`` finds the bracket narrower than ``epsabs + epsrel
min|x|``, or ``maxIter`` iterations are spent, as ``GSLRootFinder::Solve``
repeats them.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from .kronrod import DBL_EPSILON

__all__ = ["brent_root"]


class _Brent:
    """``brent_state_t``: the bracket, the last steps, and the function at the ends."""

    def __init__(self, f: Callable[[float], float], lower: float, upper: float) -> None:
        self.f = f
        f_lower, f_upper = _call(f, lower), _call(f, upper)
        self.a, self.fa = lower, f_lower
        self.b, self.fb = upper, f_upper
        self.c, self.fc = upper, f_upper
        self.d = self.e = upper - lower
        if (f_lower < 0.0 and f_upper < 0.0) or (f_lower > 0.0 and f_upper > 0.0):
            raise ValueError("endpoints do not straddle y=0")

    def _arranged(self) -> tuple[float, float, float, float, float, float, float, float, bool]:
        """The bracket put in order: ``c`` on the far side of the root, ``b`` the better end."""
        a, b, c, d, e = self.a, self.b, self.c, self.d, self.e
        fa, fb, fc = self.fa, self.fb, self.fc
        ac_equal = False
        if _same_sign(fb, fc):
            ac_equal, c, fc, d, e = True, a, fa, b - a, b - a
        if abs(fc) < abs(fb):
            ac_equal = True
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        return a, b, c, d, e, fa, fb, fc, ac_equal

    def iterate(self) -> tuple[float, float, float]:
        """One step: the root estimate and the new bracket."""
        a, b, c, d, e, fa, fb, fc, ac_equal = self._arranged()
        tol, m = 0.5 * DBL_EPSILON * abs(b), 0.5 * (c - b)
        if fb == 0:
            return b, b, b
        if abs(m) <= tol:
            return _ordered(b, c)
        d, e = _step(a, b, c, fa, fb, fc, d, e, m, tol, ac_equal)
        a, fa = b, fb
        b += d if abs(d) > tol else (tol if m > 0 else -tol)
        fb = _call(self.f, b)
        self.a, self.b, self.c, self.d, self.e = a, b, c, d, e
        self.fa, self.fb, self.fc = fa, fb, fc
        return _ordered(b, a if _same_sign(fb, fc) else c)


def _same_sign(u: float, v: float) -> bool:
    return (u < 0 and v < 0) or (u > 0 and v > 0)


def _ordered(b: float, c: float) -> tuple[float, float, float]:
    """The estimate, and the bracket's ends in order."""
    return (b, b, c) if b < c else (b, c, b)


def _interpolated(a: float, b: float, fa: float, fb: float, fc: float, m: float,
                  ac_equal: bool) -> tuple[float, float]:  # fmt: skip
    """The secant - ``a`` and ``c`` the same - or inverse quadratic step: ``p / q``."""
    s = fb / fa
    if ac_equal:
        return 2 * m * s, 1 - s
    q, r = fa / fc, fb / fc
    p = s * (2 * m * q * (q - r) - (b - a) * (r - 1))
    return p, (q - 1) * (r - 1) * (s - 1)


def _step(a: float, b: float, c: float, fa: float, fb: float, fc: float, d: float, e: float,
          m: float, tol: float, ac_equal: bool) -> tuple[float, float]:  # fmt: skip
    """The next step: interpolated when it is safe, else bisection."""
    if abs(e) < tol or abs(fa) <= abs(fb):
        return m, m
    p, q = _interpolated(a, b, fa, fb, fc, m, ac_equal)
    p, q = (p, -q) if p > 0 else (-p, q)
    if 2 * p < min(3 * m * q - abs(tol * q), abs(e * q)):
        return p / q, d
    return m, m


def _call(f: Callable[[float], float], x: float) -> float:
    value = float(f(x))
    if not math.isfinite(value):
        raise ValueError(f"function value is not finite at x = {x}")
    return value


def _converged(lower: float, upper: float, epsabs: float, epsrel: float) -> bool:
    """``gsl_root_test_interval``."""
    same_sign = (lower > 0.0 and upper > 0.0) or (lower < 0.0 and upper < 0.0)
    min_abs = min(abs(lower), abs(upper)) if same_sign else 0.0
    return abs(upper - lower) < epsabs + epsrel * min_abs


def brent_root(f: Callable[[float], float], lower: float, upper: float, max_iter: int = 100,
               epsabs: float = 1e-8, epsrel: float = 1e-10) -> tuple[bool, float, int]:  # fmt: skip
    """``RootFinder::Solve``: whether a root was found, the root, and the iterations taken."""
    try:
        solver = _Brent(f, lower, upper)
    except ValueError:
        return False, 0.5 * (lower + upper), 0
    root = 0.5 * (lower + upper)
    for iteration in range(1, max_iter + 1):
        try:
            root, lower, upper = solver.iterate()
        except ValueError:
            return False, root, iteration
        if _converged(lower, upper, epsabs, epsrel):
            return True, root, iteration
    return False, root, max_iter
