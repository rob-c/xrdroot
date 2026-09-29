"""``TSpectrum::Deconvolution``: Gold's deconvolution of a spectrum by a response, boosted.

Gold's method solves ``y = H x`` for a non-negative ``x`` through the normal
equations ``H'y = H'H x``, multiplying ``x`` each iteration by
``(H'y) / (H'H x)``; ``H'H`` of a convolution is the response's
autocorrelation, so each iteration is one symmetric convolution. After
``numberIterations`` the solution is raised to ``boost`` and iterated again,
``numberRepetitions`` times, which narrows the peaks further. The answer is
shifted by the position of the response's maximum and scaled by its area.
"""

from __future__ import annotations

import numpy as np

from ..random import libm
from .clipping import Array
from .sums import gather, ordered_sum, padded

__all__ = ["Response", "Symmetric", "correlate", "gold", "gold_step", "response_of"]

#: Below this ROOT leaves a channel of ``x`` - or of ``H'y`` - as the last iteration had it.
FLOOR = 0.000001


class Response:
    """What ROOT reads off a response: its length to the last non-zero, area and peak."""

    def __init__(self, values: Array) -> None:
        nonzero = np.flatnonzero(values != 0)
        #: ``lh_gold``: one past the last non-zero channel, or -1 for all zeros.
        self.length = int(nonzero[-1]) + 1 if nonzero.size else -1
        self.area = float(np.add.accumulate(np.concatenate([[0.0], values]))[-1])
        peak = float(np.max(values, initial=0.0))
        #: ``posit``: the first channel at the maximum - 0 if nothing is above zero.
        self.position = int(np.flatnonzero(values == peak)[0]) if peak > 0 else 0


def response_of(values: Array) -> Response | str:
    """The response - or ROOT's message for one that is all zeros."""
    found = Response(values)
    return found if found.length != -1 else "ZERO RESPONSE VECTOR"


def correlate(first: Array, second: Array, lags: int) -> Array:
    """``sum_j first[j] * second[i + j]`` for each ``i < lags``, over ``j`` in order.

    Of the response with itself it is ``H'H``'s kernel; with the spectrum,
    ``H'y`` - ROOT's ``sum_k h[k - i] * y[k]``, the same terms in the same order.
    """
    size = second.shape[0]
    index = gather(size, np.arange(lags), np.arange(first.shape[0]))
    return ordered_sum((first[None, :] * padded(second)[index]).T)


class Symmetric:
    """``H'H x``: ``sum_j kernel[j] * (x[i+j] + x[i-j])`` for each ``i``, ``x[i]`` alone at 0.

    The indices are worked out once, for the thousands of iterations that use them.
    """

    def __init__(self, kernel: Array, length: int, size: int) -> None:
        reach = np.arange(length)
        self.kernel = kernel[:length, None]
        self.ahead = gather(size, reach, np.arange(size))
        self.behind = gather(size, -reach, np.arange(size))
        self.behind[0] = size

    def __call__(self, x: Array) -> Array:
        ext = padded(x)
        return ordered_sum(self.kernel * (ext[self.ahead] + ext[self.behind]))


def gold_step(x: Array, kept: Array, hty: Array, step: Symmetric) -> Array:
    """One Gold iteration: ``x * H'y / H'Hx`` where both are above the floor, ``kept`` elsewhere."""
    active = (hty > FLOOR) & (x > FLOOR)
    denominator = step(x)
    safe = np.where(denominator != 0, denominator, 1.0)
    with np.errstate(all="ignore"):
        ratio = np.where(denominator != 0, hty / safe, 0.0)
        return np.where(active, ratio * x, kept)


def gold(
    source: Array, response: Response, values: Array,
    iterations: int, repetitions: int, boost: float,
) -> Array:  # fmt: skip
    """``source`` deconvolved by the response ``values`` (read into ``response``)."""
    size = source.shape[0]
    kernel = correlate(values, values, size)
    hty = correlate(values, source, size)
    step = Symmetric(kernel, response.length, size)
    x, kept = np.ones(size), hty.copy()
    for repetition in range(repetitions):
        if repetition:
            x = np.asarray(libm.power(x, boost), dtype=np.float64)
        for _ in range(iterations):
            kept = gold_step(x, kept, hty, step)
            x = kept.copy()
    return response.area * np.roll(x, response.position)

