"""The deconvolution inside ``TSpectrum::SearchHighRes``: by a Gaussian of the peaks' sigma.

ROOT deconvolves the cleared, extended spectrum by a Gaussian response of
``1000 exp(-(i - 3 sigma)^2 / 2 sigma^2)`` truncated to integers - Gold's
method again, but over the whole of ``H'y`` including the lags before the
first channel, which leaves each channel's ``H'y`` read ``length - 1`` lags
early and the answer shifted back by as much at the end. The arrays here
are ROOT's working-space regions, overlaps and all.
"""

from __future__ import annotations

import numpy as np

from ..random import libm
from .clipping import Array
from .gold import Response
from .sums import gather, ordered_sum, padded

__all__ = ["Deconvolved", "deconvolved", "gaussian_response"]

#: Below this in magnitude ROOT leaves a channel as the last iteration had it.
FLOOR = 0.00001


def gaussian_response(size: int, sigma: float) -> Array:
    """``(Int_t)(1000 exp(-(i - 3 sigma)^2 / (2 sigma sigma)))`` for each of ``size`` channels."""
    distance = np.arange(size, dtype=np.float64) - 3 * sigma
    exponent = distance * distance / (2 * sigma * sigma)
    return np.trunc(1000 * np.asarray(libm.exp(-exponent), dtype=np.float64))


class Deconvolved:
    """Gold's iteration over ``size`` channels by the Gaussian ``response``."""

    def __init__(self, response: Array, values: Array) -> None:
        size = values.shape[0]
        found = Response(response)
        self.length, self.area, self.position = found.length, found.area, found.position
        reach = self.length - 1
        h = response[: self.length]
        lags = np.arange(-reach, reach + 1)
        within = gather(self.length, lags, np.arange(self.length))
        self.kernel = ordered_sum(h[None, :] * padded(h)[within], 1)
        wide = size + 2 * reach
        index = gather(size, np.arange(-reach, size + reach), np.arange(self.length))
        region = ordered_sum(h[None, :] * padded(np.abs(values))[index], 1)
        #: ``H'y`` as ROOT reads it: from the lag ``-reach`` on; and what follows it in the
        #: working space, which a channel below the floor keeps from the start.
        self.hty = region[:size]
        self.kept = np.concatenate([region[size:wide], np.zeros(max(size - 2 * reach, 0))])[:size]
        self.index = gather(size, lags, np.arange(size))
        self.size = size

    def step(self, x: Array, kept: Array) -> Array:
        """One iteration: ``x * H'y / H'Hx`` where both are above the floor, ``kept`` elsewhere."""
        active = (np.abs(self.hty) > FLOOR) & (np.abs(x) > FLOOR)
        denominator = ordered_sum(self.kernel[:, None] * padded(x)[self.index])
        with np.errstate(all="ignore"):
            safe = np.where(denominator != 0, denominator, 1.0)
            ratio = np.where(denominator != 0, self.hty / safe, 0.0)
            return np.where(active, ratio * x, kept)

    def solve(self, iterations: int) -> Array:
        """The last iterate after ``iterations``, from all ones."""
        x, kept = np.ones(self.size), self.kept
        for _ in range(iterations):
            kept = self.step(x, kept)
            x = kept
        return x


def deconvolved(
    values: Array, sigma: float, iterations: int, shift: int, inner: int
) -> tuple[Array, int]:
    """``values`` deconvolved, scaled and shifted back - zero outside the ``inner`` channels.

    The last ``length - 1`` channels, whose number comes with it, are left
    as the iteration left them, as ROOT leaves them.
    """
    size = values.shape[0]
    solver = Deconvolved(gaussian_response(size, sigma), values)
    x = solver.solve(iterations)
    rolled = np.roll(x, solver.position)
    reach = solver.length - 1
    out = x.copy()
    head = np.arange(size - reach)
    inside = (head >= shift) & (head < inner + shift)
    out[: size - reach] = np.where(inside, solver.area * rolled[head + reach], 0.0)
    return out, reach
