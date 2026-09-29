"""The peaks ``TSpectrum2Fit`` fits, and their derivatives, over every channel at once.

A 2-D peak is ``exp(-(p*p - 2*ro*p*r + r*r) / 2(1 - ro*ro))`` of ``p = (x - x0)
/ sigmax`` and ``r = (y - y0) / sigmay``, with a tail and a step made of
``erfc`` in each direction; beside each peak lie two ridges, a 1-D peak in
x and one in y, of their own amplitudes and positions. As in
:mod:`.fitpeaks`, each function is worked out for every peak and channel at
once and summed down the peaks in ROOT's order - a peak, then its x ridge,
then its y ridge - with ROOT's arithmetic, and its slips kept: ``Derampx``
squares ``p`` before its tail uses it, ``Dersigmax`` asks of a ridge whether
the peak is near, and ``Derby`` is handed the x ridge's tail.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np

from ..random import libm
from .fitpeaks import EXP_LIMIT, derfc, erfc, peaksum

__all__ = ["Peaks2", "shape2"]

Array = Any

#: ``TMath::Sqrt(2.0)``.
S2 = math.sqrt(2.0)

#: ``TSpectrum2Fit``'s pi, to eleven figures.
PI = 3.1415926535


class Peaks2(NamedTuple):
    """Every peak's seven parameters, a column per peak: ROOT's ``7*j`` to ``7*j + 6``."""

    amp: Array
    x0: Array
    y0: Array
    ampx: Array
    ampy: Array
    x1: Array
    y1: Array


def _exp(values: Array) -> Array:
    with np.errstate(over="ignore", under="ignore"):
        return libm.exp(values)


def _below(e: Array) -> Array:
    """``exp(-e)`` where ``e < 700``, else 0."""
    return np.where(e < EXP_LIMIT, _exp(-np.where(e < EXP_LIMIT, e, 0.0)), 0.0)


def gauss2(p: Array, r: Array, ro: float) -> Array:
    """The correlated Gaussian of a 2-D peak."""
    return _below(((p * p - ((2 * ro) * p) * r) + r * r) / (2 * (1 - ro * ro)))


def tailed(p: Array, b: float) -> tuple[Array, Array]:
    """``p / (s2*b)`` and ``erfc(p/s2 + 1/2b)``, the two halves of a tail."""
    return p / (S2 * b), erfc(p / S2 + 1 / (2 * b))


def near(ex: Array, er: Array) -> Array:
    """``exp(ex) * er`` where ``|ex| < 9``, else 0."""
    ok = np.abs(ex) < 9
    return np.where(ok, _exp(np.where(ok, ex, 0.0)) * er, 0.0)


def pair(ex: Array, erx: Array, ey: Array, ery: Array) -> tuple[Array, Array]:
    """``exp(ex) * erx`` and ``exp(ey) * ery``, both 0 unless both exponents are within 9."""
    ok = (np.abs(ex) < 9) & (np.abs(ey) < 9)
    px = np.where(ok, _exp(np.where(ok, ex, 0.0)) * erx, 0.0)
    return px, np.where(ok, _exp(np.where(ok, ey, 0.0)) * ery, 0.0)


def offsets(values: Array, centres: Array, sigma: float) -> Array:
    """``(value - centre) / sigma``, a row per peak."""
    return (values[None, :] - np.asarray(centres, dtype=np.float64)[:, None]) / sigma


def column(values: Array) -> Array:
    return np.asarray(values, dtype=np.float64)[:, None]


def within(*offsets_: Array) -> Array:
    """Where every offset is within three sigma, as ROOT's ``TMath::Abs(p) < 3`` asks."""
    ok = np.abs(offsets_[0]) < 3
    for more in offsets_[1:]:
        ok = ok & (np.abs(more) < 3)
    return ok


def term2(p: Array, r: Array, q: dict[str, float]) -> Array:
    """A 2-D peak of unit amplitude with its tail and step: ``Deramp2`` near the peak."""
    r1 = gauss2(p, r, q["ro"])
    if q["txy"] != 0:
        px, py = pair(*tailed(p, q["bx"]), *tailed(r, q["by"]))
        r1 = r1 + ((0.5 * q["txy"]) * px) * py
    if q["sxy"] != 0:
        r1 = r1 + ((0.5 * q["sxy"]) * erfc(p / S2)) * erfc(r / S2)
    return r1


def ridge(p: Array, t: float, s: float, b: float) -> Array:
    """A 1-D ridge of unit amplitude, with its tail ``t`` of slope ``b`` and its step ``s``."""
    r1 = _below((p * p) / 2)
    if t != 0:
        r1 = r1 + (0.5 * t) * near(*tailed(p, b))
    if s != 0:
        r1 = r1 + (0.5 * s) * erfc(p / S2)
    return r1


def shape2(x: Array, y: Array, peaks: Peaks2, q: dict[str, float]) -> Array:
    """``TSpectrum2Fit::Shape2``: the fitted spectrum at the channels ``(x, y)``."""
    p, r = offsets(x, peaks.x0, q["sigmax"]), offsets(y, peaks.y0, q["sigmay"])
    px, ry = offsets(x, peaks.x1, q["sigmax"]), offsets(y, peaks.y1, q["sigmay"])
    terms = np.empty((3 * len(peaks.amp), len(x)))
    terms[0::3] = np.where(within(p, r), column(peaks.amp) * term2(p, r, q), 0.0)
    along_x = column(peaks.ampx) * ridge(px, q["tx"], q["sx"], q["bx"])
    along_y = column(peaks.ampy) * ridge(ry, q["ty"], q["sy"], q["by"])
    terms[1::3] = np.where(within(px), along_x, 0.0)
    terms[2::3] = np.where(within(ry), along_y, 0.0)
    return ((peaksum(terms) + q["a0"]) + q["ax"] * x) + q["ay"] * y


def deramp2(x: Array, y: Array, peaks: Peaks2, q: dict[str, float]) -> Array:
    """``TSpectrum2Fit::Deramp2``: the derivative by each 2-D peak's amplitude."""
    p, r = offsets(x, peaks.x0, q["sigmax"]), offsets(y, peaks.y0, q["sigmay"])
    return np.where(within(p, r), term2(p, r, q), 0.0)


def derampx(v: Array, centres: Array, sigma: float, t: float, s: float, b: float) -> Array:
    """``TSpectrum2Fit::Derampx``: the derivative by each ridge's amplitude.

    ROOT squares and halves ``p`` in place, so the tail and the step here are
    of ``p*p/2``, not ``p``.
    """
    p = offsets(v, centres, sigma)
    half = (p * p) / 2
    r1 = _below(half)
    if t != 0:
        r1 = r1 + (0.5 * t) * near(*tailed(half, b))
    if s != 0:
        r1 = r1 + (0.5 * s) * erfc(half / S2)
    return np.where(within(p), r1, 0.0)


def _sloped(p: Array, b: float, sigma: float) -> Array:
    """The derivative of a tail's ``erfc(p/s2 + 1/2b)`` by its position, as ROOT writes it."""
    c = p / S2 + 1 / (2 * b)
    return (-erfc(c)) / ((S2 * b) * sigma) - derfc(c) / (S2 * sigma)


def _moved2(p: Array, r: Array, q: dict[str, float], by_x: bool) -> Array:
    """``Deri02`` (``by_x``) or ``Derj02`` before the amplitude: a 2-D peak moved in x or y."""
    ro, sx, sy, bx, by = q["ro"], q["sigmax"], q["sigmay"], q["bx"], q["by"]
    e = (-(ro * r - p)) / sx if by_x else (-(ro * p - r)) / sy
    r1 = gauss2(p, r, ro) * (e / (1 - ro * ro))
    if q["txy"] != 0:
        erx = _sloped(p, bx, sx) if by_x else erfc(p / S2 + 1 / (2 * bx))
        ery = erfc(r / S2 + 1 / (2 * by)) if by_x else _sloped(r, by, sy)
        px, py = pair(p / (S2 * bx), erx, r / (S2 * by), ery)
        r1 = r1 + ((0.5 * q["txy"]) * px) * py
    if q["sxy"] != 0:
        rx = (-derfc(p / S2)) / (S2 * sx) if by_x else erfc(p / S2)
        ry = erfc(r / S2) if by_x else (-derfc(r / S2)) / (S2 * sy)
        r1 = r1 + ((0.5 * q["sxy"]) * rx) * ry
    return r1


def deri02(x: Array, y: Array, peaks: Peaks2, q: dict[str, float], by_x: bool = True) -> Array:
    """``Deri02`` - or, not ``by_x``, ``Derj02``: the derivative by a 2-D peak's position."""
    p, r = offsets(x, peaks.x0, q["sigmax"]), offsets(y, peaks.y0, q["sigmay"])
    return np.where(within(p, r), column(peaks.amp) * _moved2(p, r, q, by_x), 0.0)


def derderi02(x: Array, y: Array, peaks: Peaks2, q: dict[str, float], by_x: bool = True) -> Array:
    """``Derderi02`` - or ``Derderj02``: the second derivative by a 2-D peak's position."""
    ro, sx, sy = q["ro"], q["sigmax"], q["sigmay"]
    p, r = offsets(x, peaks.x0, sx), offsets(y, peaks.y0, sy)
    e = (-(ro * r - p)) / sx if by_x else (-(ro * p - r)) / sy
    e = e / (1 - ro * ro)
    sigma = sx if by_x else sy
    r1 = gauss2(p, r, ro) * (e * e - 1 / (((1 - ro * ro) * sigma) * sigma))
    return np.where(within(p, r), column(peaks.amp) * r1, 0.0)


def deri01(v: Array, amps: Array, centres: Array, sigma: float, t: float, s: float,
           b: float) -> Array:  # fmt: skip
    """``TSpectrum2Fit::Deri01``: the derivative by a ridge's position."""
    p = offsets(v, centres, sigma)
    r1 = (_below((p * p) / 2) * p) / sigma
    if t != 0:
        r1 = r1 + (0.5 * t) * near(p / (S2 * b), _sloped(p, b, sigma))
    if s != 0:
        r1 = r1 + (0.5 * s) * ((-derfc(p / S2)) / (S2 * sigma))
    return np.where(within(p), column(amps) * r1, 0.0)


def derderi01(v: Array, amps: Array, centres: Array, sigma: float) -> Array:
    """``TSpectrum2Fit::Derderi01``: the second derivative by a ridge's position."""
    p = offsets(v, centres, sigma)
    r1 = _below((p * p) / 2) * ((p * p) / (sigma * sigma) - 1 / (sigma * sigma))
    return np.where(within(p), column(amps) * r1, 0.0)
