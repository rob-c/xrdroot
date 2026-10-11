"""``TGraphSmooth``: R's ``approx``, ``ksmooth``, ``lowess`` and ``supsmu``, as ROOT has them.

ROOT's ``TGraphSmooth`` is a translation of R's C and Fortran - ``approx1``,
``BDRksmooth``, ``clowess`` and Friedman's super smoother - and this is the
same code, statement for statement, over NumPy arrays where an operation is
one on every element and in plain Python where R's loop carries a running
sum, so each value comes out as ROOT's does. :mod:`.supsmu` holds the super
smoother.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["added", "approx1", "collapse_ties", "ksmooth", "lowess", "spread"]

#: How the y of points of one x are merged by ``Approx``: R's ``ties``.
TIES = {"ordered": 0, "mean": 1, "min": 2, "max": 3}


def added(values: Any) -> float:
    """The values added one after another, as C adds them: Python's ``sum`` compensates."""
    total = 0.0
    for value in values:
        total += value
    return total


def spread(low: float, high: float, count: int) -> np.ndarray[Any, Any]:
    """``count`` x evenly from ``low`` to ``high``, as ROOT spreads a smoother's output: the
    first half stepped up from ``low``, the rest down from ``high``, so both ends are exact."""
    delta = (high - low) / (count - 1)
    half = count // 2
    return np.array([low + i * delta if i < half else high - (count - 1 - i) * delta
                     for i in range(count)])  # fmt: skip


def collapse_ties(x: Any, y: Any, ties: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """``Approxin``: the points sorted by x, the y of each run of one x merged - their mean, least
    or greatest - into one point."""
    order = np.argsort(np.asarray(x, dtype=np.float64), kind="stable")
    xs, ys = np.asarray(x, dtype=np.float64)[order], np.asarray(y, dtype=np.float64)[order]
    unique, starts = np.unique(xs, return_index=True)
    merged = []
    for at, start in enumerate(starts):
        stop = starts[at + 1] if at + 1 < len(starts) else len(xs)
        run = ys[start:stop]
        merged.append(run.min() if ties == 2 else run.max() if ties == 3 else added(run) / len(run))
    return unique, np.asarray(merged, dtype=np.float64)


def _bracket(v: float, x: Any) -> tuple[int, int]:
    """The two neighbouring points whose x hold ``v`` between them, found by halving."""
    i, j = 0, len(x) - 1
    while i < j - 1:
        middle = (i + j) // 2
        if v < x[middle]:
            j = middle
        else:
            i = middle
    return i, j


def approx1(v: float, f: float, x: Any, y: Any, kind: int, low: float, high: float) -> float:
    """R's ``approx1``: y at ``v``, linearly (``kind`` 1) or as a step (2) whose height is
    ``1 - f`` of the left point's and ``f`` of the right's; ``low`` and ``high`` outside."""
    if v < x[0]:
        return low
    if v > x[-1]:
        return high
    i, j = _bracket(v, x)
    if v == x[j]:
        return float(y[j])
    if v == x[i]:
        return float(y[i])
    if kind == 1:
        return float(y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i])))
    left = y[i] * (1 - f) if (1 - f) != 0.0 else 0.0
    return float(left + (y[j] * f if f != 0.0 else 0.0))


def ksmooth(x: Any, y: Any, xp: Any, kernel: int, bandwidth: float) -> np.ndarray[Any, Any]:
    """``BDRksmooth``: each output point the kernel-weighted mean of the points near it - a box
    (1) or a normal (2) of ``bandwidth`` in units of the quartiles' half spread - as ROOT's port
    of R's has it, which counts too the points below a window from where the last one began."""
    bw, cutoff = (bandwidth * 0.5, bandwidth * 0.5) if kernel == 1 else (
        bandwidth * 0.3706506, 4 * bandwidth * 0.3706506)  # fmt: skip
    found = np.empty(len(xp))
    imin = 0
    while imin < len(x) and x[imin] < xp[0] - cutoff:
        imin += 1
    for j, x0 in enumerate(xp):
        found[j], imin = _kernel_mean(x, y, x0, imin, (bw, cutoff), kernel)
    return found


def _kernel_mean(x: Any, y: Any, x0: float, imin: int, widths: tuple[float, float],
                 kernel: int) -> tuple[float, int]:  # fmt: skip
    """One output point's weighted mean, and where the next window starts. R's skips a point
    below the window; ROOT's port moves the start to it and counts it all the same."""
    bw, cutoff = widths
    num = den = 0.0
    for i in range(imin, len(x)):
        if x[i] < x0 - cutoff:
            imin = i
        if x[i] > x0 + cutoff:
            break
        u = abs(x[i] - x0) / bw
        w = 1.0 if kernel == 1 else math.exp(-0.5 * u * u)
        num += w * y[i]
        den += w
    return (num / den if den > 0 else 0.0), imin  # ROOT's 0 where R's NA


def _cube(v: float) -> float:
    return v * v * v  # R's fcube, multiplied out: a power rounds otherwise


def _square(v: float) -> float:
    return v * v


def _weights(x: Any, xs: float, nleft: int, h: float, robust: Any) -> tuple[list[float], int]:
    """``lowest``'s tricube weights from ``nleft`` rightwards - every tie picked up - and the
    last point they reach."""
    h9, h1 = 0.999 * h, 0.001 * h
    w: list[float] = []
    j = nleft
    while j < len(x):
        r = abs(x[j] - xs)
        if r <= h9:
            weight = 1.0 if r <= h1 else _cube(1.0 - _cube(r / h))
            w.append(weight * (robust[j] if robust is not None else 1.0))
        elif x[j] > xs:
            break
        else:
            w.append(0.0)
        j += 1
    return w, j - 1


def lowest(x: Any, y: Any, xs: float, nleft: int, nright: int, robust: Any) -> float | None:
    """R's ``lowest``: the locally weighted line through ``x[nleft..]`` at ``xs``, or ``None``
    when every weight is zero."""
    h = max(xs - x[nleft], x[nright] - xs)
    w, last = _weights(x, xs, nleft, h, robust)
    total = added(w)
    if total <= 0.0:
        return None
    w = [one / total for one in w]
    span = range(nleft, last + 1)
    if h > 0.0:
        a = added(wj * x[j] for wj, j in zip(w, span, strict=True))
        c = added(wj * (x[j] - a) * (x[j] - a) for wj, j in zip(w, span, strict=True))
        if math.sqrt(c) > 0.001 * (x[-1] - x[0]):
            b = (xs - a) / c
            w = [wj * (b * (x[j] - a) + 1.0) for wj, j in zip(w, span, strict=True)]
    return float(added(wj * y[j] for wj, j in zip(w, span, strict=True)))


def _fit_pass(x: Any, y: Any, ns: int, delta: float, robust: Any) -> list[float]:
    """One pass of ``clowess``: a local line at each point, those within ``delta`` of the last
    fitted interpolated between, and points of one x given its fit."""
    n = len(x)
    ys = [0.0] * n
    nleft, nright, last, i = 0, ns - 1, -1, 0
    while last < n - 1:
        while nright < n - 1 and x[i] - x[nleft] > x[nright + 1] - x[i]:
            nleft, nright = nleft + 1, nright + 1
        fitted = lowest(x, y, x[i], nleft, nright, robust)
        ys[i] = float(y[i]) if fitted is None else fitted
        _interpolate(x, ys, last, i)
        i, last = _skipped(x, ys, i, delta)
    return ys


def _interpolate(x: Any, ys: list[float], last: int, i: int) -> None:
    """The points skipped between the last fitted and this one, on the line between them."""
    if last < i - 1:
        denom = x[i] - x[last]
        for j in range(last + 1, i):
            alpha = (x[j] - x[last]) / denom
            ys[j] = alpha * ys[i] + (1.0 - alpha) * ys[last]


def _skipped(x: Any, ys: list[float], last: int, delta: float) -> tuple[int, int]:
    """The next point to fit - the last within ``delta`` of this one - and the last fitted, the
    points of this one's x given its fit."""
    cut, i = x[last] + delta, last + 1
    while i < len(x) and x[i] <= cut:
        if x[i] == x[last]:
            ys[i] = ys[last]
            last = i
        i += 1
    return max(last + 1, i - 1), last


def _robustness(residuals: list[float]) -> list[float]:
    """The bisquare weights of the residuals over six times their median. ROOT's port goes on
    when that median is nothing - it has not R's later test for an exact fit - and then every
    point not fitted exactly weighs nothing."""
    n = len(residuals)
    ordered = sorted(abs(r) for r in residuals)
    m1 = n // 2
    cmad = 3.0 * (ordered[m1] + ordered[n - m1 - 1]) if n % 2 == 0 else 6.0 * ordered[m1]
    c9, c1 = 0.999 * cmad, 0.001 * cmad
    found = []
    for r in (abs(one) for one in residuals):
        found.append(1.0 if r <= c1 else _square(1.0 - _square(r / cmad)) if r <= c9 else 0.0)
    return found


def lowess(x: Any, y: Any, span: float, iterations: int, delta: float) -> np.ndarray[Any, Any]:
    """R's ``clowess``: locally weighted lines through ``span`` of the points, refitted
    ``iterations`` times with the points weighted down by how far they lay from the last."""
    x, y = [float(v) for v in x], [float(v) for v in y]
    n = len(x)
    if n < 2:
        return np.asarray(y[:1], dtype=np.float64)
    ns = max(2, min(n, int(span * n + 1e-7)))
    ys = _fit_pass(x, y, ns, delta, None)
    for _ in range(iterations):
        ys = _fit_pass(x, y, ns, delta, _robustness([y[i] - ys[i] for i in range(n)]))
    return np.asarray(ys, dtype=np.float64)
