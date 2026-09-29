"""``TSpectrum2::SmoothMarkov``: a two-dimensional spectrum smoothed by a Markov chain.

As in one dimension (:mod:`.markov`), the chain's steps are weighted by how
the spectrum rises and falls over ``averWindow`` channels; the first row
and column are running products as there, and each inner channel mixes
the channel before it along x and the one before it along y by the x and y
weights. That mixing feeds on itself along each row, so it is a loop over
Python floats; the weights, which do not, are worked out at once - read at
the channels ROOT reads them, which for the x steps is the column before
the one stepped along. The sum it is normalised by starts from 0, not from
the first channel's 1, as ROOT's does.
"""

from __future__ import annotations

import numpy as np

from .clipping import Array
from .markov import markov_weights, step_weight

__all__ = ["chain2", "smooth_markov2"]


def _inner_weights(scaled: Array, window: int) -> tuple[Array, Array, Array, Array]:
    """``spx``, ``smx``, ``spy``, ``smy`` for every inner step, each summed in ``l`` order."""
    xmax, ymax = scaled.shape[0] - 1, scaled.shape[1] - 1
    i = np.arange(xmax)[:, None]
    j = np.arange(ymax)[None, :]
    spx = smx = spy = smy = np.zeros((xmax, ymax))
    for reach in range(1, window + 1):
        spx = spx + step_weight(scaled[np.minimum(i + reach, xmax), j], scaled[i, j + 1])
        smx = smx + step_weight(scaled[np.maximum(i - reach + 1, 0), j], scaled[i + 1, j + 1])
        spy = spy + step_weight(scaled[i, np.minimum(j + reach, ymax)], scaled[i + 1, j])
        smy = smy + step_weight(scaled[i, np.maximum(j - reach + 1, 0)], scaled[i + 1, j + 1])
    return spx, smx, spy, smy


def _edge(scaled: Array, window: int) -> list[float]:
    """The chain along one edge from 1: ``w[i + 1] = (sp / sm) * w[i]``."""
    up, down = markov_weights(scaled, window)
    state = [1.0]
    for ratio in (up / down).tolist():
        state.append(ratio * state[-1])
    return state


def chain2(scaled: Array, window: int) -> tuple[Array, float]:
    """The unnormalised stationary state from 1 at ``[0][0]``, and its sum as ROOT adds it."""
    sx, sy = scaled.shape
    state = [[0.0] * sy for _ in range(sx)]
    column, row = _edge(scaled[:, 0], window), _edge(scaled[0, :], window)
    added = column[1:] + row[1:]
    for i in range(sx):
        state[i][0] = column[i]
    state[0] = list(row)
    spx, smx, spy, smy = (weights.tolist() for weights in _inner_weights(scaled, window))
    for i in range(sx - 1):
        above, here = state[i], state[i + 1]
        for j in range(sy - 1):
            value = (spx[i][j] * above[j + 1] + spy[i][j] * here[j]) / (smx[i][j] + smy[i][j])
            here[j + 1] = value
            added.append(value)
    norm = float(np.add.accumulate(np.array([0.0, *added]))[-1])
    return np.array(state), norm


def smooth_markov2(source: Array, window: int) -> Array | str | None:
    """``source`` smoothed over ``window`` channels - ``None`` for an empty spectrum, left as is.

    A refusal is ROOT's message.
    """
    if window <= 0:
        return "Averaging Window must be positive"
    maximum = float(np.max(source, initial=0.0))
    if maximum == 0:
        return None
    area = float(np.add.accumulate(source.ravel())[-1])
    state, norm = chain2(source / maximum, window)
    return area * (state / norm)
