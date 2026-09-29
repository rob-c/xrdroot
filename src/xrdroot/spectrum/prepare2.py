"""The plane ``TSpectrum2::SearchHighRes`` works on: extended, cleared, smoothed - ROOT's way.

ROOT pads the spectrum by ``8 sigma`` channels on every side, each padding
channel the nearest edge channel, takes off a background clipped by
successive filtering - without the ``b > 0`` that ``Background`` asks - and
optionally smooths what is left by a Markov chain. Before smoothing it
copies the plane to where the chain reads it at ``2 * ssizex_ext`` columns
along its working rows, where the chain reads ``2 * ssizey_ext``: the same
place for a square plane, and for any other what those columns hold, which
is reproduced here by the row buffer ROOT's working space is.
"""

from __future__ import annotations

import numpy as np

from .background2 import clipped, successive
from .clipping import Array
from .markov2 import smooth_markov2

__all__ = ["cleared2", "through_rows"]

#: How many of the extended plane's widths each of ROOT's working rows holds.
REGIONS = 16


def _background(values: Array, iterations: int) -> Array:
    """``values`` clipped by successive filtering with windows growing to ``iterations``."""
    values = values.copy()
    sx, sy = values.shape
    for width in range(1, iterations + 1):
        values[width : sx - width, width : sy - width] = clipped(
            values, successive(values, width, width), width, width, False
        )
    return values


def through_rows(values: Array, base: Array) -> tuple[Array, Array]:
    """What the chain reads, and the heights peaks are ranked by, after ROOT's copy.

    ROOT's rows hold ``REGIONS`` regions of ``ssizey_ext`` channels: the
    plane in the second, the ranking heights in the last. It copies each
    row's plane to column ``2 * ssizex_ext`` on, one channel at a time, and
    the chain reads the third region.
    """
    sx, sy = values.shape
    rows = np.zeros((sx, REGIONS * sy))
    rows[:, sy : 2 * sy] = values
    rows[:, (REGIONS - 1) * sy :] = base
    for j in range(sy):
        if 2 * sx + j < REGIONS * sy:
            rows[:, 2 * sx + j] = rows[:, sy + j]
    return rows[:, 2 * sy : 3 * sy].copy(), rows[:, (REGIONS - 1) * sy :].copy()


def cleared2(
    source: Array, shift: int, iterations: int, remove: bool, markov: bool, window: int
) -> tuple[Array, Array] | None:
    """The ranking heights and the plane to deconvolve - ``None`` where the chain has nothing."""
    padded = np.pad(source, shift, mode="edge")
    values = padded - _background(padded, iterations) if remove else padded
    if not markov:
        return values.copy(), values
    read, base = through_rows(values, values)
    smoothed = smooth_markov2(read, window)
    if not isinstance(smoothed, np.ndarray):
        return None
    return base, smoothed
