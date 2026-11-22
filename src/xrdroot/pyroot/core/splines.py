"""``TSpline5``: ROOT's quintic spline, and the C function ``SaveAs`` writes of it.

ROOT interpolates a table of knots with a quintic polynomial on each interval, choosing the
coefficients so that neighbouring polynomials agree in value, first and second derivative.
The arithmetic here is ``TSpline5::BuildCoeff`` transcribed term for term, in ROOT's order,
because a spline is a table of coefficients rather than a curve: ``GetCoeff`` hands them out
and ``SaveAs`` writes them into a C function a macro runs, so they have to be ROOT's own to
the last digit and not merely fit the knots well.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .refs import store

__all__ = ["TSpline5"]

#: The knots are equidistant, so ``FindX`` finds an interval by dividing: ``fKstep``.
_STEP = True
#: The knots are the abscissas the caller gave, so ``FindX`` searches them: ``!fKstep``.
_NO_STEP = False
#: ``fNpx``: the points ``Draw`` puts between the knots.
_NPX = 100
#: ``SaveAs`` folds a coefficient array five values to a line, under the opening brace.
_FOLD = "\n" + " " * 23


class _Knot:
    """One ``TSplinePoly5``: a knot, and the coefficients of the polynomial that starts there.

    ``y`` is the value at the knot and ``b`` to ``f`` the derivatives, each divided by its
    factorial, so the polynomial is Taylor's - as ROOT reads and writes them.
    """

    __slots__ = ("b", "c", "d", "e", "f", "x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.b = self.c = self.d = self.e = self.f = 0.0

    def value(self, at: float) -> float:
        """``TSplinePoly5::Eval``: the polynomial at ``at``, by Horner's rule."""
        dx = at - self.x
        return self.y + dx * (self.b + dx * (self.c + dx * (self.d + dx * (self.e + dx * self.f))))

    def slope(self, at: float) -> float:
        """``TSplinePoly5::Derivative``: its first derivative at ``at``."""
        dx = at - self.x
        return self.b + dx * (2 * self.c + dx * (3 * self.d + dx * (4 * self.e + dx * 5 * self.f)))


class _Shape:
    """What a ``TSpline5`` constructor makes of its arguments: the knots, and where they run."""

    __slots__ = ("delta", "knots", "kstep", "xmax", "xmin")

    def __init__(self, knots: list[_Knot], delta: float, kstep: bool) -> None:
        self.knots = knots
        self.delta = delta
        self.kstep = kstep
        self.xmin = knots[0].x
        self.xmax = knots[-1].x


def _equidistant(xmin: float, xmax: float, ys: Any, n: int) -> _Shape:
    """``n`` knots spread evenly over ``xmin .. xmax``, carrying the values ``ys``."""
    delta = (xmax - xmin) / (n - 1)
    knots = [_Knot(xmin + i * delta, float(y)) for i, y in enumerate(list(ys)[:n])]
    return _Shape(knots, delta, _STEP)


def _irregular(xs: Any, ys: Any, n: int) -> _Shape:
    """``n`` knots at the abscissas ``xs``, in increasing order, carrying the values ``ys``."""
    given = np.asarray(xs).ravel()[:n]
    knots = [_Knot(float(x), float(y)) for x, y in zip(given, list(ys)[:n], strict=False)]
    return _Shape(knots, -1.0, _NO_STEP)


def _ends(option: Any) -> tuple[int, int, tuple[bool, bool, bool, bool]]:
    """``TSpline5::BoundaryConditions``: the double knots ``option`` adds at each end -
    ``b2``/``e2`` two, ``b1``/``e1`` one - and which end conditions it gives there."""
    if not option:
        return 0, 0, (False, False, False, False)
    text = str(option)
    flags = ("b1" in text, "e1" in text, "b2" in text, "e2" in text)
    cb1, ce1, cb2, ce2 = flags
    return (2 if cb2 else int(cb1)), (2 if ce2 else int(ce1)), flags


def _set_boundaries(knots: list[_Knot], ends: Sequence[float],
                    flags: tuple[bool, bool, bool, bool]) -> None:
    """``TSpline5::SetBoundaries``: what the double and triple knots carry.

    A ``b2``/``e2`` triple knot pins the second derivative there - the value written on
    ``fPoly[2].Y()`` is read back as one - and, given no ``b1``/``e1``, takes the first
    derivative from the finite difference of the neighbouring knots, which ROOT itself calls
    not recommended.
    """
    cb1, ce1, cb2, ce2 = flags
    b1, e1, b2, e2 = ends
    last = len(knots) - 1
    if cb2:
        knots[0].x = knots[1].x = knots[2].x
        knots[0].y = knots[2].y
        knots[2].y = b2
        knots[1].y = b1 if cb1 else (knots[3].y - knots[0].y) / (knots[3].x - knots[2].x)
    elif cb1:
        knots[0].x = knots[1].x
        knots[0].y = knots[1].y
        knots[1].y = b1
    if ce2:
        knots[last].x = knots[last - 1].x = knots[last - 2].x
        knots[last].y = e2
        knots[last - 1].y = e1 if ce1 else ((knots[last - 2].y - knots[last - 3].y)
                                           / (knots[last - 2].x - knots[last - 3].x))
    elif ce1:
        knots[last].x = knots[last - 1].x
        knots[last].y = e1


def _array(name: str, values: Sequence[float]) -> str:
    """``   const double fX[n] = {...};`` as ``SaveAs`` writes it, folded as ROOT folds it."""
    text = "   const double " + name + "[" + str(len(values)) + "] = {"
    chunk = ""
    for i, value in enumerate(values):
        last = i == len(values) - 1
        chunk += f" {value:.17g}" + ("" if last else ",")
        if i % 5 == 4 or last:
            text += chunk
            chunk = "" if last else _FOLD
    return text + " };\n"


def _row(knots: list[_Knot], i: int, widths: Sequence[float], squares: Sequence[float],
         sums: Sequence[float]) -> None:
    """One row of the pentadiagonal matrix of ``TSpline5::BuildCoeff``, in ROOT's own order.

    ``widths`` are the distances ``p, q, r`` between the three knots, ``squares`` their
    squares and ``sums`` the sums ``pq`` and ``qr`` the row's coefficients divide by.
    """
    p, q, r = widths
    p2, q2, r2 = squares
    pq, qr = sums
    if not q:
        knots[i + 1].d = knots[i].e = knots[i - 1].f = 0.0
        return
    _row_curvature(knots, i, (p, q, r), (p2, q2, r2), (pq, qr))
    _row_slope(knots, i, (p, q, r), q2, (pq, qr))


def _row_curvature(knots: list[_Knot], i: int, widths: Sequence[float],
                   squares: Sequence[float], sums: Sequence[float]) -> None:
    """The ``d`` of one row: what each of the three knots does to the second difference."""
    p, q, r = widths
    p2, q2, r2 = squares
    pq, qr = sums
    q3 = q2 * q
    knots[i + 1].d = q3 * 6.0 / (qr * qr)
    knots[i].d += _pull(p, q, r, p2, q2, r2, pq, qr)
    knots[i - 1].d += q3 * 6.0 / (pq * pq)


def _pull(p: float, q: float, r: float, p2: float, q2: float, r2: float, pq: float,
          qr: float) -> float:
    """What the middle knot of a row adds to its ``d``, in ROOT's own terms of the widths."""
    pr = p * r
    pqqr = pq * qr
    return ((q + q) * (pr * 15.0 * pr + (p + r) * q * (pr * 20.0 + q2 * 7.0)
                       + q2 * ((p2 + r2) * 8.0 + pr * 21.0 + q2 + q2)) / (pqqr * pqqr))


def _row_slope(knots: list[_Knot], i: int, widths: Sequence[float], q2: float,
               sums: Sequence[float]) -> None:
    """The ``e`` and ``f`` of one row: what the third and fourth derivatives carry."""
    p, q, r = widths
    pq, qr = sums
    pqqr = pq * qr
    knots[i].e = q2 * (p * qr + pq * 3.0 * (qr + r + r)) / (pqqr * qr)
    knots[i - 1].e += q2 * (r * pq + qr * 3.0 * (pq + p + p)) / (pqqr * pq)
    knots[i - 1].f = q2 * q / pqqr


def _eliminate(knots: list[_Knot], i: int, p: float) -> float:
    """One step of the elimination, and the ``p`` the step after it needs."""
    q = knots[i - 1].d * knots[i - 1].e
    knots[i].d = 1.0 / (knots[i].d - p * knots[i - 2].f - q * knots[i - 1].e)
    knots[i].e -= q * knots[i - 1].f
    knots[i].c = knots[i + 2].c - knots[i + 1].c - p * knots[i - 2].c - q * knots[i - 1].c
    return knots[i - 1].d * knots[i - 1].f


def _integrated_curvature(knots: list[_Knot], i: int, q: float, s: float) -> None:
    """``e`` and ``d`` at knot ``i``: the third derivative integrated once and twice."""
    knots[i].e = s * 5.0
    knots[i].d = (knots[i].c - q * s) * 10


def _integrated_value(knots: list[_Knot], i: int, widths: Sequence[float],
                      spans: Sequence[float], u: float, v: float) -> None:
    """``c`` and ``b`` at knot ``i``, from the two knots either side of it."""
    p, q = widths
    p3, q3 = spans
    knots[i].c = (knots[i].d * (p - q)
                  + (knots[i + 1].b - knots[i].b + (u - knots[i].e) * p3
                     - (v + knots[i].e) * q3) / (p + q))
    knots[i].b = (((p * (knots[i + 1].b - v * q3) + q * (knots[i].b - u * p3)) / (p + q))
                  - p * q * (knots[i].d + knots[i].e * (q - p)))


#: What ``SaveAs`` writes after the coefficients: ROOT's own knot search and polynomial, so
#: that a macro running the file the test wrote runs the search ROOT's own ``Eval`` runs.
_TAIL = (
    "   int klow=0;\n"
    "   if(x<=fXmin) klow=0;\n"
    "   else if(x>=fXmax) klow=fNp-1;\n"
    "   else {\n"
    "     if(fKstep) {\n"
    "       // Equidistant knots, use histogramming\n"
    "       klow = int((x-fXmin)/fDelta);\n"
    "       if (klow > fNp-1) klow = fNp-1;\n"
    "     } else {\n"
    "       int khig=fNp-1, khalf;\n"
    "       // Non equidistant knots, binary search\n"
    "       while(khig-klow>1)\n"
    "         if(x>fX[khalf=(klow+khig)/2]) klow=khalf;\n"
    "         else khig=khalf;\n"
    "     }\n"
    "   }\n"
    "   // Evaluate now\n"
    "   double dx=x-fX[klow];\n"
    "   return (fY[klow]+dx*(fB[klow]+dx*(fC[klow]+dx*(fD[klow]+dx*(fE[klow]+dx*fF[klow])))));\n"
    "}\n"
)


class TSpline5:
    """ROOT's quintic spline: ``Eval``, ``Derivative``, ``GetCoeff`` and ``SaveAs``."""

    def __init__(self, title: str = "", *given: Any) -> None:
        """``TSpline5(title, xmin, xmax, y, n, opt, b1, e1, b2, e2)`` - evenly spread knots -
        or the same with the ``x`` of every knot in place of ``xmin, xmax``.

        The four numbers after ``opt`` are the values a ``b1``/``e1`` (first derivative) or
        ``b2``/``e2`` (second derivative) option puts on the extra knots at the ends.
        """
        self._title = str(title)
        self._npx = _NPX
        even = len(given) >= 4 and np.ndim(given[0]) == 0 and np.ndim(given[1]) == 0
        spread = _equidistant(*given[:4]) if even else _irregular(*given[:3])
        at = 4 if even else 3
        option = given[at] if len(given) > at else None
        ends = [float(value) for value in given[at + 1:]]
        ends += [0.0] * (4 - len(ends))
        beg, end, flags = _ends(option)
        knots = [_Knot() for _ in range(beg)] + spread.knots + [_Knot() for _ in range(end)]
        _set_boundaries(knots, ends[:4], flags)
        self._knots = knots
        self._np = len(knots)
        self._delta = spread.delta
        self._xmin = spread.xmin
        self._xmax = spread.xmax
        self._kstep = spread.kstep
        self._build()

    def GetNp(self) -> int:
        """How many knots the spline has, the ones a boundary option added included."""
        return self._np

    def GetDelta(self) -> float:
        """The distance between evenly spread knots, ``-1`` for a table of abscissas."""
        return float(self._delta)

    def GetXmin(self) -> float:
        return float(self._xmin)

    def GetXmax(self) -> float:
        return float(self._xmax)

    def GetNpx(self) -> int:
        return int(self._npx)

    def SetNpx(self, n: int) -> None:
        self._npx = int(n)

    def FindX(self, x: float) -> int:
        """``FindX``: the knot whose polynomial ``x`` falls in, or the end one outside the
        knots, which extrapolates that polynomial and may be badly wrong."""
        if x <= self._xmin:
            return 0
        if x >= self._xmax:
            return self._np - 1
        if self._kstep:
            return min(int((x - self._xmin) / self._delta), self._np - 1)
        low, high = 0, self._np - 1
        while high - low > 1:
            half = (low + high) // 2
            if x > self._knots[half].x:
                low = half
            else:
                high = half
        return low

    def Eval(self, x: Any) -> float:
        """The spline's value at ``x``, from the polynomial of the knot it falls in."""
        at = float(x)
        return self._knots[self.FindX(at)].value(at)

    def Derivative(self, x: Any) -> float:
        """The spline's first derivative at ``x``."""
        at = float(x)
        return self._knots[self.FindX(at)].slope(at)

    def GetKnot(self, i: int, x: Any = None, y: Any = None) -> None:
        """``GetKnot(i, x, y)``: into ``x`` and ``y``."""
        store(x, self._knots[i].x)
        store(y, self._knots[i].y)

    def GetCoeff(self, i: int, x: Any = None, y: Any = None, b: Any = None, c: Any = None,
                 d: Any = None, e: Any = None, f: Any = None) -> None:
        """``GetCoeff(i, x, y, b, c, d, e, f)``: the coefficients of knot ``i``, into each."""
        knot = self._knots[i]
        for target, value in ((x, knot.x), (y, knot.y), (b, knot.b), (c, knot.c),
                              (d, knot.d), (e, knot.e), (f, knot.f)):
            store(target, value)

    def SetPoint(self, i: int, x: float, y: float) -> None:
        """``SetPoint``: move knot ``i``, leaving the coefficients where ROOT leaves them."""
        if not 0 <= i < self._np:
            return
        self._knots[i].x = float(x)
        self._knots[i].y = float(y)

    def SetPointCoeff(self, i: int, b: float, c: float, d: float, e: float, f: float) -> None:
        """``SetPointCoeff``: the five coefficients of knot ``i``, set as they stand."""
        if not 0 <= i < self._np:
            return
        knot = self._knots[i]
        knot.b, knot.c, knot.d, knot.e, knot.f = (float(v) for v in (b, c, d, e, f))

    def SaveAs(self, filename: str = "", option: Any = "") -> None:
        """Write the spline as a C function of ``x`` that needs no ROOT, as ``SaveAs`` does.

        The function's name is the file's up to its first dot, as ROOT takes it, so a macro
        can read the file back, append the name and run it.
        """
        knots = self._knots
        parts = [
            ("double " + str(filename)).split(".", 1)[0] + "(double x) {\n",
            f"   const int fNp = {self._np}, fKstep = {int(self._kstep)};\n",
            f"   const double fDelta = {self._delta:.17g}, fXmin = {self._xmin:.17g},"
            f" fXmax = {self._xmax:.17g};\n",
            _array("fX", [knot.x for knot in knots]),
            _array("fY", [knot.y for knot in knots]),
            _array("fB", [knot.b for knot in knots]),
            _array("fC", [knot.c for knot in knots]),
            _array("fD", [knot.d for knot in knots]),
            _array("fE", [knot.e for knot in knots]),
            _array("fF", [knot.f for knot in knots]),
            _TAIL,
        ]
        Path(filename).write_text("".join(parts), encoding="ascii")

    def _build(self) -> None:
        """``TSpline5::BuildCoeff``: the five coefficients of every knot, in ROOT's four steps."""
        if self._np <= 2:
            return
        self._pentadiagonal()
        self._differences()
        self._solve()
        self._integrate()

    def _pentadiagonal(self) -> None:
        """The coefficients of the positive definite pentadiagonal matrix, in ``d``, ``e``
        and ``f``."""
        knots = self._knots
        m = self._np - 2
        q = knots[1].x - knots[0].x
        r = knots[2].x - knots[1].x
        q2 = q * q
        r2 = r * r
        qr = q + r
        knots[0].d = knots[0].e = 0.0
        knots[1].d = q * 6.0 * q2 / (qr * qr) if q else 0.0
        if m > 1:
            for i in range(1, m):
                p, q, r = q, r, knots[i + 2].x - knots[i + 1].x
                p2, q2, r2 = q2, r2, r * r
                pq, qr = qr, q + r
                _row(knots, i, (p, q, r), (p2, q2, r2), (pq, qr))
        if r:
            knots[m - 1].d += r * 6.0 * r2 / (qr * qr)

    def _differences(self) -> None:
        """The first and second order divided differences of the values, in ``b`` and ``c``.

        A double or triple knot - the same abscissa twice - carries a derivative instead,
        which ROOT reads as the value written on it, so the two are swapped here as it does.
        """
        knots = self._knots
        for i in range(1, self._np):
            if knots[i].x != knots[i - 1].x:
                knots[i].b = (knots[i].y - knots[i - 1].y) / (knots[i].x - knots[i - 1].x)
            else:
                knots[i].b = knots[i].y
                knots[i].y = knots[i - 1].y
        for i in range(2, self._np):
            if knots[i].x != knots[i - 2].x:
                knots[i].c = (knots[i].b - knots[i - 1].b) / (knots[i].x - knots[i - 2].x)
            else:
                knots[i].c = knots[i].b * 0.5
                knots[i].b = knots[i - 1].b

    def _solve(self) -> None:
        """Solve the system whose right-hand side is ``c(i+2) - c(i+1)``, in place."""
        knots = self._knots
        m = self._np - 2
        if m > 1:
            p = knots[0].c = knots[m - 1].e = knots[0].f = knots[m - 2].f = knots[m - 1].f = 0.0
            knots[1].c = knots[3].c - knots[2].c
            knots[1].d = 1.0 / knots[1].d
            if m > 2:
                for i in range(2, m):
                    p = _eliminate(knots, i, p)
        knots[self._np - 2].c = knots[self._np - 1].c = 0.0
        if self._np > 3:
            for i in range(self._np - 3, 0, -1):
                knots[i].c = ((knots[i].c - knots[i].e * knots[i + 1].c
                               - knots[i].f * knots[i + 2].c) * knots[i].d)

    def _integrate(self) -> None:
        """Integrate the third derivative, which gives every ``c``, ``b`` and the end points."""
        knots = self._knots
        m = self._np - 1
        q = knots[1].x - knots[0].x
        r = knots[2].x - knots[1].x
        b1 = knots[1].b
        q3 = q * q * q
        qr = q + r
        v = t = knots[1].c / qr if qr else 0.0
        knots[0].f = v / q if q else 0.0
        for i in range(1, m):
            p, q = q, r
            r = knots[i + 2].x - knots[i + 1].x if i != m - 1 else 0.0
            p3, q3 = q3, q * q * q
            pq, qr = qr, q + r
            s = t
            t = (knots[i + 1].c - knots[i].c) / qr if qr else 0.0
            u, v = v, t - s
            if not pq:
                knots[i].c = knots[i - 1].c
                knots[i].d = knots[i].e = knots[i].f = 0.0
                continue
            knots[i].f = knots[i - 1].f
            if q:
                knots[i].f = v / q
            _integrated_curvature(knots, i, q, s)
            _integrated_value(knots, i, (p, q), (p3, q3), u, v)
        self._end_points(b1)

    def _end_points(self, b1: float) -> None:
        """The first and last knots, whose ``e`` and ``d`` are zero and whose ``c`` and ``b``
        are fixed by the one polynomial next to them."""
        knots = self._knots
        last = self._np - 1
        p = knots[1].x - knots[0].x
        s = knots[0].f * p * p * p
        knots[0].e = knots[0].d = 0.0
        knots[0].c = knots[1].c - s * 10
        knots[0].b = b1 - (knots[0].c + s) * p
        q = knots[last].x - knots[last - 1].x
        t = knots[last - 1].f * q * q * q
        knots[last].e = knots[last].d = 0.0
        knots[last].c = knots[last - 1].c + t * 10
        knots[last].b += (knots[last].c - t) * q
