"""``TSpectrum2Fit``'s derivatives by the parameters its peaks share.

Sigma in x and y, the correlation ``ro``, and the tails' and steps' own
parameters: each a sum over the peaks - and, for most, their ridges - of
the terms :mod:`.fit2peaks` makes, added in ROOT's order.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .fit2peaks import (
    S2,
    Peaks2,
    _below,
    column,
    gauss2,
    near,
    offsets,
    pair,
    tailed,
    within,
)  # fmt: skip
from .fitpeaks import derfc, erfc, peaksum

__all__ = ["derbx", "derdersigmax", "derro", "dersigmax", "dersxy", "dertxy"]

Array = Any


def interleaved(first: Array, second: Array) -> Array:
    """A peak's term, then its ridge's, peak after peak: the order ROOT adds them in."""
    terms = np.empty((2 * first.shape[0], first.shape[1]))
    terms[0::2], terms[1::2] = first, second
    return peaksum(terms)


def _sloped(p: Array, b: float, sigma: float) -> Array:
    """A tail's ``erfc`` differentiated by sigma, ``p`` times over, as ROOT writes it."""
    c = p / S2 + 1 / (2 * b)
    return ((-erfc(c)) * p) / ((S2 * b) * sigma) - (derfc(c) * p) / (S2 * sigma)


def _axes(x: Array, y: Array, peaks: Peaks2, q: dict[str, float], by_x: bool) -> tuple[Any, ...]:
    """The 2-D offsets ``p`` and ``r``, and the ridge's own, in the direction asked for."""
    p, r = offsets(x, peaks.x0, q["sigmax"]), offsets(y, peaks.y0, q["sigmay"])
    if by_x:
        return p, r, offsets(x, peaks.x1, q["sigmax"]), p
    return p, r, offsets(y, peaks.y1, q["sigmay"]), r


def _names(by_x: bool) -> tuple[str, str, str, str, str]:
    """Sigma, the ridge's tail, step and slope, and its amplitude, in x or in y."""
    return ("sigmax", "tx", "sx", "bx", "ampx") if by_x else ("sigmay", "ty", "sy", "by", "ampy")


def _peak_by_sigma(p: Array, r: Array, q: dict[str, float], by_x: bool) -> Array:
    """``Dersigmax``'s (or ``Dersigmay``'s) term for a 2-D peak of unit amplitude."""
    ro, bx, by = q["ro"], q["bx"], q["by"]
    own, other = (p, r) if by_x else (r, p)
    sigma = q["sigmax"] if by_x else q["sigmay"]
    b = (-((ro * p) * r - own * own)) / sigma
    e = (gauss2(p, r, ro) * b) / (1 - ro * ro)
    if q["txy"] != 0:
        erx = _sloped(p, bx, sigma) if by_x else erfc(p / S2 + 1 / (2 * bx))
        ery = erfc(r / S2 + 1 / (2 * by)) if by_x else _sloped(r, by, sigma)
        px, py = pair(p / (S2 * bx), erx, r / (S2 * by), ery)
        e = e + ((0.5 * q["txy"]) * px) * py
    if q["sxy"] != 0:
        moved = ((-derfc(own / S2)) * own) / (S2 * sigma)
        rx, ry = (moved, erfc(other / S2)) if by_x else (erfc(other / S2), moved)
        e = e + ((0.5 * q["sxy"]) * rx) * ry
    return e


def _ridge_by_sigma(u: Array, q: dict[str, float], by_x: bool) -> Array:
    """``Dersigmax``'s (or ``Dersigmay``'s) term for a ridge of unit amplitude."""
    sigma_name, t_name, s_name, b_name, _ = _names(by_x)
    sigma, b = q[sigma_name], q[b_name]
    half = (u * u) / 2
    e = ((2 * half) * _below(half)) / sigma
    if q[t_name] != 0:
        e = e + (0.5 * q[t_name]) * near(u / (S2 * b), _sloped(u, b, sigma))
    if q[s_name] != 0:
        e = e + (0.5 * q[s_name]) * (((-derfc(u / S2)) * u) / (S2 * sigma))
    return e


def dersigmax(x: Array, y: Array, peaks: Peaks2, q: dict[str, float], by_x: bool = True) -> Array:
    """``Dersigmax`` - or, not ``by_x``, ``Dersigmay``: the derivative by sigma.

    Whether a ridge counts is asked of its peak's offset, not its own - ROOT's slip.
    """
    p, r, u, asked = _axes(x, y, peaks, q, by_x)
    amps = getattr(peaks, _names(by_x)[4])
    first = np.where(within(p, r), column(peaks.amp) * _peak_by_sigma(p, r, q, by_x), 0.0)
    second = np.where(within(asked), column(amps) * _ridge_by_sigma(u, q, by_x), 0.0)
    return interleaved(first, second)


def derdersigmax(x: Array, y: Array, peaks: Peaks2, q: dict[str, float],
                 by_x: bool = True) -> Array:  # fmt: skip
    """``Derdersigmax`` - or ``Derdersigmay``: the second derivative by sigma."""
    ro = q["ro"]
    p, r, u, asked = _axes(x, y, peaks, q, by_x)
    own, other = (p, r) if by_x else (r, p)
    sigma = q[_names(by_x)[0]]
    b = (-((ro * p) * r - own * own)) / sigma
    bend = ((3 * own) * own - ((2 * ro) * own) * other) / (sigma * sigma)
    e = (gauss2(p, r, ro) * ((b * b) / (1 - ro * ro) - bend)) / (1 - ro * ro)
    half = (u * u) / 2
    ridge = (_below(half) * ((4 * half) * half - 6 * half)) / (sigma * sigma)
    first = np.where(within(p, r), column(peaks.amp) * e, 0.0)
    second = np.where(within(asked), column(getattr(peaks, _names(by_x)[4])) * ridge, 0.0)
    return interleaved(first, second)


def derro(x: Array, y: Array, peaks: Peaks2, q: dict[str, float]) -> Array:
    """``TSpectrum2Fit::Derro``: the derivative by the peaks' correlation."""
    ro = q["ro"]
    px, qx = offsets(x, peaks.x0, q["sigmax"]), offsets(y, peaks.y0, q["sigmay"])
    rx = (px * px - ((2 * ro) * px) * qx) + qx * qx
    ex = _below(rx / (2 * (1 - ro * ro)))
    tx = (px * qx) / (1 - ro * ro) - (ro * rx) / ((1 - ro * ro) * (1 - ro * ro))
    return peaksum(np.where(within(px, qx), (column(peaks.amp) * ex) * tx, 0.0))


def dertxy(x: Array, y: Array, peaks: Peaks2, q: dict[str, float]) -> Array:
    """``TSpectrum2Fit::Dertxy``: the derivative by the 2-D tails' amplitude, every peak."""
    p, r = offsets(x, peaks.x0, q["sigmax"]), offsets(y, peaks.y0, q["sigmay"])
    px, py = pair(*tailed(p, q["bx"]), *tailed(r, q["by"]))
    return peaksum(((0.5 * column(peaks.amp)) * px) * py)


def dersxy(x: Array, y: Array, peaks: Peaks2, q: dict[str, float]) -> Array:
    """``TSpectrum2Fit::Dersxy``: the derivative by the 2-D steps' amplitude."""
    p, r = offsets(x, peaks.x0, q["sigmax"]), offsets(y, peaks.y0, q["sigmay"])
    return peaksum(((0.5 * column(peaks.amp)) * erfc(p / S2)) * erfc(r / S2))


def dertx(v: Array, amps: Array, centres: Array, sigma: float, b: float) -> Array:
    """``Dertx`` and ``Derty``: the derivative by a ridge tail's amplitude."""
    p = offsets(v, centres, sigma)
    return peaksum((0.5 * column(amps)) * near(*tailed(p, b)))


def dersx(v: Array, amps: Array, centres: Array, sigma: float) -> Array:
    """``Dersx`` and ``Dersy``: the derivative by a ridge step's amplitude."""
    return peaksum((0.5 * column(amps)) * erfc(offsets(v, centres, sigma) / S2))


def _by_slope(u: Array, b: float) -> Array:
    """A tail's ``erfc`` differentiated by its slope ``b``, as ``Derbx`` writes it."""
    c = u / S2 + 1 / (2 * b)
    return ((-erfc(c)) * u) / ((S2 * b) * b) - derfc(c) / ((S2 * b) * b)


def derbx(x: Array, y: Array, peaks: Peaks2, q: dict[str, float], by_x: bool = True) -> Array:
    """``Derbx`` - or ``Derby``: the derivative by the tails' slope in x or y.

    ``Derby`` is handed the x ridge's tail amplitude ``tx`` for its own, as ROOT calls it.
    """
    p, r, u, _ = _axes(x, y, peaks, q, by_x)
    bx, by = q["bx"], q["by"]
    b_name, amp_name = _names(by_x)[3:]
    zero = np.zeros_like(p)
    first, second = zero, zero
    if q["txy"] != 0:
        erx = _by_slope(p, bx) if by_x else erfc(p / S2 + 1 / (2 * bx))
        ery = erfc(r / S2 + 1 / (2 * by)) if by_x else _by_slope(r, by)
        px, py = pair(p / (S2 * bx), erx, r / (S2 * by), ery)
        first = (((0.5 * column(peaks.amp)) * q["txy"]) * px) * py
    if q["tx"] != 0:
        b = q[b_name]
        tail = near(u / (S2 * b), _by_slope(u, b))
        second = ((0.5 * column(getattr(peaks, amp_name))) * q["tx"]) * tail
    return interleaved(first, second)
