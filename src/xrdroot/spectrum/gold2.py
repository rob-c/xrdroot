"""``TSpectrum2::Deconvolution``: Gold's deconvolution in two dimensions, boosted.

The one-dimensional method (:mod:`.gold`) over a plane: ``H'y`` is the
response correlated with the spectrum, ``H'H`` the response's
autocorrelation, and each iteration multiplies every channel by ``H'y``
over the autocorrelation of the current answer. ROOT adds each channel's
terms over ``j2`` and then ``j1``; here each term is a shifted copy of the
whole plane, added in that order, a term of zero - which adds nothing, to
the bit - left out. Unlike the one-dimensional method, every channel is
updated each iteration, and to zero where ``H'y``, the answer or the
autocorrelation is zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..random import libm
from .clipping import Array

__all__ = ["Response2", "correlated", "gold2", "gold2_step"]


class Response2:
    """What ROOT reads off a response plane: its extent to the last non-zero, area and peak."""

    def __init__(self, values: Array) -> None:
        rows, columns = np.nonzero(values)
        #: ``lhx``, ``lhy``: one past the last non-zero channel along each axis, -1 for none.
        self.lengths = (int(rows.max()) + 1, int(columns.max()) + 1) if rows.size else (-1, -1)
        self.area = float(np.add.accumulate(np.concatenate([[0.0], values.ravel()]))[-1])
        peak = float(np.max(values, initial=0.0))
        at = int(np.flatnonzero(values.ravel() == peak)[0]) if peak > 0 else 0
        #: ``positx``, ``posity``: the first channel at the maximum, x outermost.
        self.position = divmod(at, values.shape[1])


def correlated(kernel: Array, plane: Array, offsets: Any) -> Array:
    """``sum kernel[a][b] * plane[i + a + ox][j + b + oy]`` over ``(b, a)`` - ``b`` outermost.

    ``offsets`` is ``(ox, oy)``, where ``kernel[0][0]`` lies; a channel off
    the plane is zero, and adds nothing. (ROOT writes some products the other
    way round, which IEEE multiplication does not notice.)
    """
    sx, sy = plane.shape
    kx, ky = kernel.shape
    ox, oy = offsets
    before = (max(0, -ox), max(0, -oy))
    wide = np.pad(plane, ((before[0], max(0, ox + kx - 1)), (before[1], max(0, oy + ky - 1))))
    total, term = np.zeros((sx, sy)), np.empty((sx, sy))
    for b in range(ky):
        for a in range(kx):
            weight = kernel[a, b]
            if weight == 0:
                continue
            x0, y0 = before[0] + ox + a, before[1] + oy + b
            np.multiply(weight, wide[x0 : x0 + sx, y0 : y0 + sy], out=term)
            np.add(total, term, out=total)
    return total


def autocorrelation(response: Array, lhx: int, lhy: int) -> Array:
    """``H'H``'s kernel: the response correlated with itself at every lag within its extent."""
    head = response[:lhx, :lhy]
    padded = np.zeros((3 * lhx - 2, 3 * lhy - 2))
    padded[lhx - 1 : 2 * lhx - 1, lhy - 1 : 2 * lhy - 1] = head
    full = correlated(head, padded, (0, 0))
    return full[: 2 * lhx - 1, : 2 * lhy - 1]


def gold2_step(x: Array, hty: Array, kernel: Array, reach: tuple[int, int]) -> Array:
    """One iteration: ``x * H'y / H'Hx``, zero where any of the three is."""
    denominator = correlated(kernel, x, (-reach[0], -reach[1]))
    with np.errstate(all="ignore"):
        keep = (x * hty != 0) & (denominator != 0)
        return np.where(keep, x * hty / np.where(denominator != 0, denominator, 1.0), 0.0)


def gold2(
    source: Array, response: Response2, values: Array,
    iterations: int, repetitions: int, boost: float,
) -> Array:  # fmt: skip
    """``source`` deconvolved by the response plane ``values`` (read into ``response``)."""
    lhx, lhy = response.lengths
    hty = correlated(values[:lhx, :lhy], source, (0, 0))
    kernel = autocorrelation(values, lhx, lhy)
    x = np.ones(source.shape)
    for repetition in range(repetitions):
        if repetition:
            x = np.asarray(libm.power(x, boost), dtype=np.float64)
        for _ in range(iterations):
            x = gold2_step(x, hty, kernel, (lhx - 1, lhy - 1))
    return np.roll(response.area * x, response.position, axis=(0, 1))
