"""``TSpectrum::SmoothMarkov``: a spectrum smoothed as the stationary state of a Markov chain.

The chain steps between neighbouring channels with probabilities set by how
the spectrum rises and falls over ``averWindow`` channels each way: from
each channel ``i`` it goes up with a weight ``sp``, the sum of
``exp((y[i+l] - y[i]) / sqrt(y[i+l] + y[i]))``, and down with ``sm``, the
same looking back from ``i + 1``, all in units of the spectrum's maximum.
Its stationary distribution is then the running product of ``sp / sm``,
which scaled to the spectrum's area is the smoothed spectrum. The weights
are independent, so are worked out at once; the product and the sums are
NumPy's accumulations, which add and multiply in order, as ROOT's loop does.
"""

from __future__ import annotations

import numpy as np

from ..random import libm
from .clipping import Array

__all__ = ["markov_weights", "smooth_markov", "step_weight"]


def step_weight(neighbour: Array, here: Array) -> Array:
    """``exp((a - n) / sqrt(a + n))``, with ``sqrt`` 1 where ``a + n`` is not positive: one term."""
    total = neighbour + here
    root = np.where(total <= 0, 1.0, np.sqrt(np.where(total <= 0, 1.0, total)))
    return np.asarray(libm.exp((neighbour - here) / root), dtype=np.float64)


def markov_weights(scaled: Array, window: int) -> tuple[Array, Array]:
    """``sp`` and ``sm`` for each step ``i -> i + 1`` of the chain over ``scaled``, ROOT's way."""
    last = scaled.shape[0] - 1
    steps = np.arange(last)
    up, down = np.zeros(last), np.zeros(last)
    for reach in range(1, window + 1):
        up = up + step_weight(scaled[np.minimum(steps + reach, last)], scaled[steps])
        down = down + step_weight(scaled[np.maximum(steps - reach + 1, 0)], scaled[steps + 1])
    return up, down


def chain(scaled: Array, window: int, start: float) -> tuple[Array, float]:
    """The unnormalised stationary state from 1 at channel 0, and its sum from ``start``."""
    up, down = markov_weights(scaled, window)
    state = np.multiply.accumulate(np.concatenate([[1.0], up / down]))
    norm = float(np.add.accumulate(np.concatenate([[start], state[1:]]))[-1])
    return state, norm


def smooth_markov(source: Array, window: int) -> Array | str | None:
    """``source`` smoothed over ``window`` channels - ``None`` for an empty spectrum, left as is.

    A refusal is ROOT's message.
    """
    if window <= 0:
        return "Averaging Window must be positive"
    maximum = float(np.max(source, initial=0.0))
    if maximum == 0:
        return None
    area = float(np.add.accumulate(source)[-1])
    state, norm = chain(source / maximum, window, 1.0)
    return (state / norm) * area
