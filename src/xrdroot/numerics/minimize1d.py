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


class _Brent:
    """``MinimBrent``'s state: the bracket ``[a, b]``, the best three points and last steps."""

    def __init__(self, f: Callable[[float], float], a: float, b: float, x: float) -> None:
        self.f, self.a, self.b = f, a, b
        self.v = self.w = self.x = x
        self.fv = self.fw = self.fx = f(x)
        self.e = self.d = 0.0

    def _golden(self, m: float) -> None:
        self.e = self.a - self.x if self.x >= m else self.b - self.x
        self.d = GOLDEN * self.e

    def _through_three(self) -> tuple[float, float]:
        """The parabola through ``x``, ``w`` and ``v``: its step's numerator and denominator."""
        x, w, v = self.x, self.w, self.v
        r = (x - w) * (self.fx - self.fv)
        q = (x - v) * (self.fx - self.fw)
        p = (x - v) * q - (x - w) * r
        q = 2 * (q - r)
        return (-p, q) if q > 0 else (p, -q)

    def _parabola(self, m: float, tol: float, t2: float) -> None:
        """A parabolic step through the three points - if it is safe - else a golden one."""
        x = self.x
        p, q = self._through_three()
        r, self.e = self.e, self.d
        if abs(p) >= abs(0.5 * q * r) or p <= q * (self.a - x) or p >= q * (self.b - x):
            self._golden(m)
            return
        self.d = p / q
        u = x + self.d
        if u - self.a < t2 or self.b - u < t2:
            self.d = abs(tol) if m - x >= 0 else -abs(tol)

    def step(self, tol: float, t2: float, m: float) -> None:
        if abs(self.e) > tol:
            self._parabola(m, tol, t2)
        else:
            self._golden(m)
        d = self.d
        u = self.x + d if abs(d) >= tol else self.x + (abs(tol) if d >= 0 else -abs(tol))
        self._update(u, self.f(u))

    def _update(self, u: float, fu: float) -> None:
        """The bracket narrowed to the new point, and the best three points kept."""
        if fu <= self.fx:
            self.a, self.b = (self.a, self.x) if u < self.x else (self.x, self.b)
            self.v, self.fv, self.w, self.fw = self.w, self.fw, self.x, self.fx
            self.x, self.fx = u, fu
            return
        self.a, self.b = (u, self.b) if u < self.x else (self.a, u)
        if fu <= self.fw or self.w == self.x:
            self.v, self.fv, self.w, self.fw = self.w, self.fw, u, fu
        elif fu <= self.fv or self.v == self.x or self.v == self.w:
            self.v, self.fv = u, fu


def minim_brent(f: Callable[[float], float], xmin: float, xmax: float, xmiddle: float,
                epsabs: float, epsrel: float,
                itermax: int) -> tuple[float, bool, float, float]:  # fmt: skip
    """``MinimBrent``: the minimum, whether it converged, and the range left."""
    state = _Brent(f, xmin, xmax, xmiddle)
    for _ in range(itermax):
        m = 0.5 * (state.a + state.b)
        tol = epsrel * abs(state.x) + epsabs
        if abs(state.x - m) <= (2 * tol - 0.5 * (state.b - state.a)):
            return state.x, True, xmin, xmax
        state.step(tol, 2 * tol, m)
    return state.x, False, state.a, state.b


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
