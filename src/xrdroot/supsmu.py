"""Friedman's super smoother - R's ``supsmu`` and ``smooth``, as ``TGraphSmooth::BDRsupsmu`` is.

``smooth`` is a running-lines smoother: a window of ``span`` of the points,
its weighted means and (co)variances updated as a point enters and one
leaves, a line through the window at each point - and, asked for them, each
point's cross-validated residual. ``supsmu`` smooths with three spans, a
tenth, a fifth and a half of the points' (0.05, 0.2 and 0.5), smooths their
residuals, picks at each point the span that did best - bent towards the
widest by the ``bass`` tone - smooths the choice, and interpolates between
the three fits by it. The arithmetic is the Fortran's, in its order.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .graphsmooth import added

__all__ = ["smooth", "supsmu"]

#: The three spans, and the Fortran's ``big``, ``sml`` and ``eps``.
SPANS = (0.05, 0.2, 0.5)
BIG, SML, EPS = 1.0e20, 1.0e-7, 1.0e-3


class _Window:
    """The running sums of one window: its weight, means of x and y, variance and covariance."""

    def __init__(self) -> None:
        self.fbw = self.xm = self.ym = self.var = self.cvar = 0.0

    def enter(self, x: float, y: float, wt: float) -> None:
        fbo = self.fbw
        self.fbw += wt
        if self.fbw > 0:
            self.xm = (fbo * self.xm + wt * x) / self.fbw
            self.ym = (fbo * self.ym + wt * y) / self.fbw
        tmp = self.fbw * wt * (x - self.xm) / fbo if fbo > 0.0 else 0.0
        self.var += tmp * (x - self.xm)
        self.cvar += tmp * (y - self.ym)

    def leave(self, x: float, y: float, wt: float) -> None:
        fbo = self.fbw
        self.fbw -= wt
        tmp = fbo * wt * (x - self.xm) / self.fbw if self.fbw > 0.0 else 0.0
        self.var -= tmp * (x - self.xm)
        self.cvar -= tmp * (y - self.ym)
        if self.fbw > 0:
            self.xm = (fbo * self.xm - wt * x) / self.fbw
            self.ym = (fbo * self.ym - wt * y) / self.fbw


def _shifted(x: Any, j: int, periodic: bool, below: bool) -> tuple[int, float]:
    """The point ``j`` of a window that runs off an end - wrapped round, a whole period away,
    when the x are periodic."""
    n = len(x)
    if below and j < 0 and periodic:
        return j + n, float(x[j + n]) - 1.0
    if not below and j >= n and periodic:
        return j - n, float(x[j - n]) + 1.0
    return j, float(x[j])


def _start(x: Any, y: Any, w: Any, ibw: int, periodic: bool) -> _Window:
    window = _Window()
    for i in range(min(2 * ibw + 1, len(x))):
        j, xti = _shifted(x, i - ibw - 1 if periodic else i, periodic, True)
        window.enter(xti, float(y[j]), float(w[j]))
    return window


def _slide(window: _Window, x: Any, y: Any, w: Any, j: int, ibw: int, periodic: bool) -> None:
    """The window moved on to point ``j``: the point ``ibw + 1`` behind out, ``ibw`` ahead in."""
    out, into = j - ibw - 1, j + ibw
    if not periodic and (out < 0 or into >= len(x)):
        return
    out, xto = _shifted(x, out, periodic, True)
    into, xti = _shifted(x, into, periodic, False)
    window.leave(xto, float(y[out]), float(w[out]))
    window.enter(xti, float(y[into]), float(w[into]))


def _residual(window: _Window, x: float, y: float, w: float, fit: float, vsmlsq: float) -> Any:
    """The point's cross-validated residual, or ``None`` when its leverage is one or more."""
    h = 1.0 / window.fbw if window.fbw > 0 else 0.0
    if window.var > vsmlsq:
        h += (x - window.xm) ** 2 / window.var
    a = 1.0 - w * h
    return abs(y - fit) / a if a > 0.0 else None


def _tied_means(x: Any, w: Any, smo: list[float]) -> None:
    """The fit of points of one x made their weighted mean, as the Fortran's last loop does."""
    n, j = len(x), 0
    while j < n:
        j0, sy, fbw = j, smo[j] * w[j], float(w[j])
        while j < n - 1 and x[j + 1] <= x[j]:
            j += 1
            sy += w[j] * smo[j]
            fbw += w[j]
        if j > j0:
            mean = sy / fbw if fbw > 0 else 0.0
            smo[j0 : j + 1] = [mean] * (j + 1 - j0)
        j += 1


def smooth(x: Any, y: Any, w: Any, span: float, iper: int, vsmlsq: float) -> tuple[Any, Any]:
    """The Fortran's ``smooth``: running lines over ``span`` of the points, and - for a positive
    ``iper`` - each point's cross-validated residual; ``|iper|`` 2 for periodic x."""
    n, periodic = len(x), abs(iper) == 2
    ibw = max(int(0.5 * span * n + 0.5), 2)
    window = _start(x, y, w, ibw, periodic)
    smo, acvr = [0.0] * n, [0.0] * n
    for j in range(n):
        _slide(window, x, y, w, j, ibw, periodic)
        a = window.cvar / window.var if window.var > vsmlsq else 0.0
        smo[j] = a * (float(x[j]) - window.xm) + window.ym
        if iper > 0:
            found = _residual(window, float(x[j]), float(y[j]), float(w[j]), smo[j], vsmlsq)
            acvr[j] = found if found is not None else (acvr[j - 1] if j > 0 else 0.0)
    _tied_means(x, w, smo)
    return np.asarray(smo), np.asarray(acvr)


def _scale(x: Any) -> float:
    """The spread of the middle half of the x, widened until it is not nothing."""
    n = len(x)
    i, j = max(n // 4 - 1, 0), max(3 * (n // 4) - 1, 0)  # the Fortran's x(n/4), x(3n/4)
    scale = float(x[j] - x[i])
    while scale <= 0.0:
        j, i = min(j + 1, n - 1), max(i - 1, 0)
        scale = float(x[j] - x[i])
    return scale


def _chosen(fits: list[Any], alpha: float) -> np.ndarray[Any, Any]:
    """At each point the span whose smoothed residual is least, bent by the bass tone."""
    chosen = np.empty(len(fits[0][1]))
    for j in range(len(chosen)):
        resmin, best = BIG, SPANS[0]
        for (_, residual), span in zip(fits, SPANS, strict=True):
            if residual[j] < resmin:
                resmin, best = residual[j], span
        top = fits[2][1][j]
        if 0.0 < alpha <= 10.0 and 0.0 < resmin < top:
            best += (SPANS[2] - best) * max(SML, resmin / top) ** (10.0 - alpha)
        chosen[j] = best
    return chosen


def supsmu(x: Any, y: Any, w: Any, iper: int, span: float, alpha: float) -> np.ndarray[Any, Any]:
    """``BDRsupsmu``: the super smoother's fit at each point - one span's, when given one."""
    x, y, w = (np.asarray(a, dtype=np.float64) for a in (x, y, w))
    if not x[-1] > x[0]:  # one x: every fit the weighted mean
        total = added(w)
        return np.full(len(x), added(w * y) / total if total > 0 else 0.0)
    vsmlsq = (EPS * _scale(x)) ** 2
    jper = _periodicity(x, iper)
    if span > 0.0:
        return smooth(x, y, w, span, jper, vsmlsq)[0]
    fits = []
    for one in SPANS:
        fit, residual = smooth(x, y, w, one, jper, vsmlsq)
        fits.append((fit, smooth(x, residual, w, SPANS[1], -jper, vsmlsq)[0]))
    chosen = np.clip(smooth(x, _chosen(fits, alpha), w, SPANS[1], -jper, vsmlsq)[0], *SPANS[::2])
    return smooth(x, _mixed(fits, chosen), w, SPANS[0], -jper, vsmlsq)[0]


def _periodicity(x: Any, iper: int) -> int:
    """The Fortran's ``jper``: 2, periodic, only for x within [0, 1]; else 1."""
    return 2 if iper == 2 and x[0] >= 0.0 and x[-1] <= 1.0 else 1


def _mixed(fits: list[Any], chosen: Any) -> np.ndarray[Any, Any]:
    """Each point's fit between the middle span's and the nearer other's, as its span lies."""
    mixed = np.empty(len(chosen))
    for j, f in enumerate(chosen - SPANS[1]):
        low = f < 0.0
        f = -f / (SPANS[1] - SPANS[0]) if low else f / (SPANS[2] - SPANS[1])
        mixed[j] = (1.0 - f) * fits[1][0][j] + f * fits[0 if low else 2][0][j]
    return mixed
