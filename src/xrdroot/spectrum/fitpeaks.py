"""The peak ``TSpectrumFit`` fits, and its derivatives, over every channel at once.

A peak is a Gaussian ``exp(-p*p)`` of ``p = (i - position) / sigma``, with a
tail ``t * exp(p/b) * erfc(p + 1/2b) / 2`` and a step ``s * erfc(p) / 2``,
over a background ``a0 + a1*i + a2*i*i``; ``erfc`` is Morhac's rational
approximation, not the C library's. ROOT works out each of these for one
channel at a time, summing over the peaks as it goes; here a row of
channels is worked out for every peak at once - a ``(peaks, channels)``
array - and the peaks summed down the columns in their order, so each
channel's number is added up exactly as ROOT adds it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..random import libm

__all__ = [
    "area", "deramp", "derb", "derderi0", "derdersigma", "deri0", "derpa", "ders",
    "dersigma", "dert", "derfc", "erfc", "ourpowl", "peaksum", "shape",
]  # fmt: skip

Array = Any

#: Morhac's coefficients for ``erfc``: ``t * (DA1 + t * (DA2 + t * DA3)) * exp(-x*x)``
#: with ``t = 1 / (1 + DAP * |x|)``.
DA1, DA2, DA3, DAP = 0.1740121, -0.0479399, 0.3739278, 0.47047

#: The square root of pi as ``TSpectrumFit::Area`` writes it, to eight figures.
ODM_PI = 1.7724538

#: Past this an exponent is not taken: ROOT's guard against ``exp`` overflowing.
EXP_LIMIT = 700


def _exp_below(w: Array) -> Array:
    """``exp(-w)`` where ``w < 700``, and 0 past it, as ROOT's ``Erfc`` guards it."""
    with np.errstate(over="ignore", under="ignore"):
        return np.where(w < EXP_LIMIT, libm.exp(-w), 0.0)


def erfc(x: Array) -> Array:
    """``TSpectrumFit::Erfc``: Morhac's approximation to the complementary error function."""
    a = np.abs(x)
    t = 1.0 / (1.0 + DAP * a)
    c = (_exp_below(a * a) * t) * (DA1 + t * (DA2 + t * DA3))
    return np.where(x < 0, 1.0 - c, c)


def derfc(x: Array) -> Array:
    """``TSpectrumFit::Derfc``: the derivative of :func:`erfc`, as ROOT writes it."""
    a = np.abs(x)
    t = 1.0 / (1.0 + DAP * a)
    c = ((((-1.0) * DAP) * _exp_below(a * a)) * t) * t
    return c * (DA1 + t * (2.0 * DA2 + (t * 3.0) * DA3)) - (2.0 * a) * erfc(a)


def peaksum(terms: Array) -> Array:
    """The peaks' terms added channel by channel, first peak first, as ROOT's loop adds them."""
    if terms.shape[0] == 0:
        return np.zeros(terms.shape[1:])
    return np.cumsum(terms, axis=0)[-1]


def _gauss(p: Array) -> Array:
    """``exp(-p*p)`` where ``p*p < 700``, else 0."""
    return _exp_below(p * p)


def _tail_exp(p: Array, b: float) -> Array:
    """``exp(p/b)`` with the exponent held at 700, as every tail term holds it."""
    e = p / b
    with np.errstate(over="ignore"):
        return libm.exp(np.where(e > EXP_LIMIT, float(EXP_LIMIT), e))


def _offsets(i: Array, positions: Array, sigma: float) -> Array:
    """``p = (i - position) / sigma``, a row per peak."""
    return (i[None, :] - np.asarray(positions, dtype=np.float64)[:, None]) / sigma


def _column(values: Array) -> Array:
    return np.asarray(values, dtype=np.float64)[:, None]


def _near(p: Array, values: Array) -> Array:
    """``values`` within three sigma of the peak, and 0 further out, as ROOT cuts them."""
    return np.where(np.abs(p) < 3, values, 0.0)


def shape(i: Array, amp: Array, pos: Array, sigma: float, t: float, s: float, b: float,
          background: tuple[float, float, float]) -> Array:  # fmt: skip
    """``TSpectrumFit::Shape``: the fitted spectrum at the channels ``i``."""
    if sigma > 0.0001:
        p = _offsets(i, pos, sigma)
    else:
        p = np.where(i[None, :] == _column(pos), 0.0, 10.0)
    r = _near(p, _gauss(p))
    if t != 0:
        r = r + ((t * _tail_exp(p, b)) * erfc(p + 1.0 / (2.0 * b))) / 2.0
    else:
        r = r + 0.0
    if s != 0:
        r = r + (s * erfc(p)) / 2.0
    else:
        r = r + 0.0
    a0, a1, a2 = background
    return ((peaksum(_column(amp) * r) + a0) + a1 * i) + (a2 * i) * i


def deramp(i: Array, pos: Array, sigma: float, t: float, s: float, b: float) -> Array:
    """``TSpectrumFit::Deramp``: the derivative by each peak's amplitude, a row per peak."""
    p = _offsets(i, pos, sigma)
    q = _gauss(p)
    if t != 0:
        r = (t * _tail_exp(p, b)) / 2.0
        q = q + np.where(r != 0, r * erfc(p + 1.0 / (2.0 * b)), r)
    else:
        q = q + 0.0
    if s != 0:
        q = q + (s * erfc(p)) / 2.0
    return q


def _tails(p: Array, sigma: float, t: float, s: float, b: float, scale: Array) -> Array:
    """The tail and step terms ``r2 + r3 + r4`` of ``Deri0`` and ``Dersigma``, each times ``scale``.

    ``r2`` and ``r3`` are added to ``r1`` one at a time, which is why this
    hands back the three rather than their sum.
    """
    d = 2.0 * sigma
    zero = np.zeros_like(p)
    r2, r3, r4 = zero, zero, zero
    if t != 0:
        c = p + 1.0 / (2.0 * b)
        e = _tail_exp(p, b)
        r2 = ((((-t) * scale) * e) * erfc(c)) / (d * b)
        r3 = ((((-t) * scale) * e) * derfc(c)) / d
    if s != 0:
        r4 = (((-s) * scale) * derfc(p)) / d
    return np.stack([r2, r3, r4])


def _added(r1: Array, rest: Array) -> Array:
    """``r1 + r2 + r3 + r4``, left to right."""
    return ((r1 + rest[0]) + rest[1]) + rest[2]


def deri0(i: Array, amp: Array, pos: Array, sigma: float, t: float, s: float, b: float) -> Array:
    """``TSpectrumFit::Deri0``: the derivative by each peak's position, a row per peak."""
    p = _offsets(i, pos, sigma)
    r1 = ((2.0 * p) * _gauss(p)) / sigma
    return _column(amp) * _added(r1, _tails(p, sigma, t, s, b, np.ones_like(p)))


def derderi0(i: Array, amp: Array, pos: Array, sigma: float) -> Array:
    """``TSpectrumFit::Derderi0``: the second derivative by each peak's position."""
    p = _offsets(i, pos, sigma)
    r1 = (_gauss(p) * ((4 * p) * p - 2)) / (sigma * sigma)
    return _column(amp) * (r1 + 0.0)


def dersigma(i: Array, amp: Array, pos: Array, sigma: float, t: float, s: float, b: float) -> Array:
    """``TSpectrumFit::Dersigma``: the derivative by the peaks' common sigma."""
    p = _offsets(i, pos, sigma)
    r1 = _near(p, (((2.0 * p) * p) * _gauss(p)) / sigma)
    return peaksum(_column(amp) * _added(r1, _tails(p, sigma, t, s, b, p)))


def derdersigma(i: Array, amp: Array, pos: Array, sigma: float) -> Array:
    """``TSpectrumFit::Derdersigma``: the second derivative by sigma."""
    p = _offsets(i, pos, sigma)
    r1 = _near(p, (((_gauss(p) * p) * p) * ((4.0 * p) * p - 6)) / (sigma * sigma))
    return peaksum(_column(amp) * (r1 + 0.0))


def dert(i: Array, amp: Array, pos: Array, sigma: float, b: float) -> Array:
    """``TSpectrumFit::Dert``: the derivative by the tails' amplitude ``t``."""
    p = _offsets(i, pos, sigma)
    r1 = _tail_exp(p, b) * erfc(p + 1.0 / (2.0 * b))
    return peaksum(_column(amp) * r1) / 2.0


def ders(i: Array, amp: Array, pos: Array, sigma: float) -> Array:
    """``TSpectrumFit::Ders``: the derivative by the steps' amplitude ``s``."""
    return peaksum(_column(amp) * erfc(_offsets(i, pos, sigma))) / 2.0


def derb(i: Array, amp: Array, pos: Array, sigma: float, t: float, b: float) -> Array:
    """``TSpectrumFit::Derb``: the derivative by the tails' slope ``b`` - nothing without tails."""
    r = np.zeros_like(i)
    if t != 0:
        p = _offsets(i, pos, sigma)
        c = p + 1.0 / (2.0 * b)
        e = p / b
        r1 = p * erfc(c) + derfc(c) / 2.0
        with np.errstate(over="ignore"):
            grown = r1 * libm.exp(np.where(e > EXP_LIMIT, float(EXP_LIMIT), e))
        r = peaksum(_column(amp) * np.where(e < -EXP_LIMIT, 0.0, grown))
    return ((-r) * t) / ((2.0 * b) * b)


def ourpowl(a: Array, pw: int) -> Array:
    """``TSpectrumFit::Ourpowl``: ``a`` squared ``pw / 2`` times over, one factor at a time."""
    a2 = a * a
    c = np.ones_like(a)
    for above in (0, 2, 4, 6, 8, 10, 12):
        if pw > above:
            c = c * a2
    return c


def _c_div(a: float, b: float) -> float:
    """``a / b`` as C divides doubles: an infinity or NaN for a zero ``b``, not an exception."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(a) / np.float64(b))


def _tail_area(b: float) -> float:
    """``-(0.5/b)**2``, the exponent of a tail's area."""
    r = _c_div(0.5, b)
    return ((-1.0) * r) * r


def area(a: float, sigma: float, t: float, b: float) -> float:
    """``TSpectrumFit::Area``: a peak's area, its tail's included."""
    r = _tail_area(b) if b != 0 else -0.0
    if abs(r) < EXP_LIMIT:
        return (a * sigma) * (ODM_PI + (t * b) * float(libm.exp(r)))
    return (a * sigma) * ODM_PI


def derpa(sigma: float, t: float, b: float) -> float:
    """``TSpectrumFit::Derpa``: the derivative of a peak's area by its amplitude."""
    r = _tail_area(b)
    if abs(r) < EXP_LIMIT:
        return sigma * (ODM_PI + (t * b) * float(libm.exp(r)))
    return sigma * ODM_PI
