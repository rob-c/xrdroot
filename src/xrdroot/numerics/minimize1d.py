"""MathCore's ``BrentMinimizer1D``: a grid to bracket the minimum, then Brent's method in it.

``BrentMethods::MinimStep`` samples ``npx`` points and narrows the range to
the best one's neighbours; ``MinimBrent`` is golden sections and parabolic
steps inside that, to ``epsrel |x| + epsabs``; ``Minimize`` repeats the pair
up to ten times until Brent converges.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["BrentMinimizer1D"]

#: ``(3 - sqrt(5)) / 2``: the golden section.
GOLDEN = 3.81966011250105097e-01


def minim_step(f: Callable[[float], float], xmin: float, xmax: float,
               npx: int) -> tuple[float, float, float]:  # fmt: skip
    """``MinimStep`` for a minimum: the best grid point and the range round it."""
    if npx < 2:
        return 0.5 * (xmax - xmin), xmin, xmax
    dx = (xmax - xmin) / (npx - 1)
    xxmin, yymin = xmin, f(xmin)
    for i in range(1, npx):
        x = xmin + i * dx
        y = f(x)
        if y < yymin:
            xxmin, yymin = x, y
    xmin, xmax = max(xmin, xxmin - dx), min(xmax, xxmin + dx)
    return min(xxmin, xmax), xmin, xmax


def minim_brent(f: Callable[[float], float], xmin: float, xmax: float, xmiddle: float,
                epsabs: float, epsrel: float,
                itermax: int) -> tuple[float, bool, float, float]:  # fmt: skip
    """``MinimBrent``: the minimum, whether it converged, and the range left."""
    v = w = x = xmiddle
    e, d = 0.0, 0.0
    a, b = xmin, xmax
    fv = fw = fx = f(x)
    for _ in range(itermax):
        m = 0.5 * (a + b)
        tol = epsrel * abs(x) + epsabs
        t2 = 2 * tol
        if abs(x - m) <= (t2 - 0.5 * (b - a)):
            return x, True, xmin, xmax
        if abs(e) > tol:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2 * (q - r)
            if q > 0:
                p = -p
            else:
                q = -q
            r, e = e, d
            if abs(p) >= abs(0.5 * q * r) or p <= q * (a - x) or p >= q * (b - x):
                e = a - x if x >= m else b - x
                d = GOLDEN * e
            else:
                d = p / q
                u = x + d
                if u - a < t2 or b - u < t2:
                    d = abs(tol) if m - x >= 0 else -abs(tol)
        else:
            e = a - x if x >= m else b - x
            d = GOLDEN * e
        u = x + d if abs(d) >= tol else x + (abs(tol) if d >= 0 else -abs(tol))
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, fv, w, fw, x, fx = w, fw, x, fx, u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv, w, fw = w, fw, u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x, False, a, b


class BrentMinimizer1D:
    """``ROOT::Math::BrentMinimizer1D``."""

    def __init__(self) -> None:
        self._f: Callable[[float], float] | None = None
        self._xmin = self._xmax = self._minimum = 0.0
        self._npx = 100
        self._status = -1

    def SetFunction(self, f: Callable[[float], float], xlow: float, xup: float) -> None:
        self._f, self._status = f, -1
        self._xmin, self._xmax = (xup, xlow) if xlow >= xup else (xlow, xup)

    def SetNpx(self, npx: int) -> None:
        self._npx = int(npx)

    def Minimize(self, maxIter: int, absTol: float = 1e-8, relTol: float = 1e-10) -> bool:
        assert self._f is not None
        xmin, xmax = self._xmin, self._xmax
        for _ in range(11):
            x, xmin, xmax = minim_step(self._f, xmin, xmax, self._npx)
            x, ok, xmin, xmax = minim_brent(self._f, xmin, xmax, x, absTol, relTol, maxIter)
            self._minimum = x
            if ok:
                self._status = 0
                return True
        self._status = -2
        return False

    def XMinimum(self) -> float:
        return self._minimum

    def FValMinimum(self) -> float:
        assert self._f is not None
        return float(self._f(self._minimum))

    def Status(self) -> int:
        return self._status
