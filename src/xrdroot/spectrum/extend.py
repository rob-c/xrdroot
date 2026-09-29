"""The spectrum ``TSpectrum::SearchHighRes`` works on: extended, cleared of background, smoothed.

Before deconvolving, ROOT pads the spectrum by ``7 sigma`` channels each
side so that peaks near an end are found - on the left continuing the
slope of the first ``2 sigma`` channels if it falls, never below zero, on
the right the last channel repeated - takes a SNIP background off it, and
optionally smooths it by a Markov chain and takes the background off again.
"""

from __future__ import annotations

import numpy as np

from .background import clip_pass
from .clipping import Array
from .markov import smooth_markov

__all__ = ["cleared", "extended", "left_slope"]

#: ``bw`` of the smoothed clipping before the Markov chain: five channels.
SMOOTHING_HALF = 2


def left_slope(source: Array, sigma: float) -> float:
    """The slope of a line fitted to the first ``2 sigma`` channels, if falling - else 0."""
    count = int(2 * sigma + 0.5)
    if count < 2:
        return 0.0
    m0 = m1 = m2 = l0 = l1 = 0.0
    for channel in range(count):
        a, b = float(channel), float(source[channel])
        m0, m1, m2, l0, l1 = m0 + 1, m1 + a, m2 + a * a, l0 + b, l1 + a * b
    determinant = m0 * m2 - m1 * m1
    slope = (-l0 * m1 + l1 * m0) / determinant if determinant != 0 else 0.0
    return 0.0 if slope > 0 else slope


def extended(source: Array, shift: int, slope: float) -> Array:
    """``source`` with ``shift`` channels before and after it, as ROOT pads it."""
    size = source.shape[0]
    before = source[0] + slope * (np.arange(shift) - shift).astype(np.float64)
    after = np.full(shift, source[size - 1])
    padding = np.where(before < 0, 0.0, before), np.where(after < 0, 0.0, after)
    return np.concatenate([padding[0], source, padding[1]])


def _clipped(values: Array, iterations: int, half: int) -> Array:
    """``values`` clipped with windows growing from 1 to ``iterations``."""
    for width in range(1, iterations + 1):
        values = clip_pass(values, width, 2, half)
    return values


def cleared(
    padded: Array, iterations: int, remove: bool, markov: bool, window: int
) -> tuple[Array, Array] | None:
    """What peaks are looked for in, before and after smoothing: ``padded`` less its background.

    The first is what a peak's height is judged by, the second what is
    deconvolved. ``None`` when the Markov chain has nothing to smooth -
    where ROOT stops and finds no peaks.
    """
    values = padded
    if remove:
        values = padded - _clipped(padded, iterations, SMOOTHING_HALF if markov else 0)
        values = np.where(values < 0, 0.0, values)
    if not markov:
        return values, values
    smoothed = smooth_markov(values, window)
    if not isinstance(smoothed, np.ndarray):
        return None
    return values, smoothed - _clipped(smoothed, iterations, 0) if remove else smoothed
