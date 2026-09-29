"""``TSpectrum::DeconvolutionRL``: Richardson and Lucy's deconvolution, boosted.

Each iteration multiplies ``x[i]`` by the back-projection of ``y / (H x)`` -
``sum_j h[j - i] * y[j] / (H x)[j]`` - which keeps ``x`` positive and its
sum that of ``y``. ROOT solves only for the channels a whole response fits
in front of (``i <= size - length``), leaves ``y[j]`` itself in the ratio
where it is not positive, and like Gold's method raises the answer to
``boost`` between repetitions and shifts it by the response's peak.
"""

from __future__ import annotations

import numpy as np

from ..random import libm
from .clipping import Array
from .gold import Response
from .sums import gather, ordered_sum, padded

__all__ = ["LucyStep", "lucy"]


class LucyStep:
    """One Richardson-Lucy iteration for a response of ``length`` over ``size`` channels."""

    def __init__(self, response: Array, length: int, size: int) -> None:
        reach = np.arange(length)
        solved = size - length + 1
        #: ``h[k]`` against ``x[j - k]``, ``k`` from ``length - 1`` down, as ROOT's ``(H x)[j]``.
        self.backward = reach[::-1]
        self.convolved = gather(size, -self.backward, np.arange(size))
        np.putmask(self.convolved, self.convolved >= solved, size)
        self.response = response[:length]
        self.projected = gather(size, reach, np.arange(solved))
        self.size, self.solved = size, solved

    def __call__(self, x: Array, y: Array) -> Array:
        ext = padded(x)
        hx = ordered_sum(self.response[self.backward, None] * ext[self.convolved])
        with np.errstate(all="ignore"):
            quotient = np.where(hx > 0, y / np.where(hx > 0, hx, 1.0), 0.0)
        ratio = padded(np.where(y > 0, quotient, y))
        back = ordered_sum(ratio[self.projected] * self.response[:, None])
        out = np.zeros(self.size)
        head = x[: self.solved]
        out[: self.solved] = np.where(head > 0, back * head, 0.0)
        return out


def lucy(
    source: Array, response: Response, values: Array,
    iterations: int, repetitions: int, boost: float,
) -> Array:  # fmt: skip
    """``source`` deconvolved by the response ``values`` (read into ``response``)."""
    size = source.shape[0]
    step = LucyStep(values, response.length, size)
    x = np.where(np.arange(size) <= size - response.length, 1.0, 0.0)
    for repetition in range(repetitions):
        if repetition:
            x = np.asarray(libm.power(x, boost), dtype=np.float64)
        for _ in range(iterations):
            x = step(x, source)
    return np.roll(x, response.position)
