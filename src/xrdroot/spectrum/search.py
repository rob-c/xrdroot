"""``TSpectrum::SearchHighRes``: peaks found in a spectrum sharpened by deconvolution.

The spectrum is extended, cleared of its background and optionally smoothed
(:mod:`.extend`), deconvolved by a Gaussian of the peaks' ``sigma``
(:mod:`.highres`), and every local maximum of the deconvolved spectrum above
``threshold`` per cent of the highest - both there and in the cleared
spectrum - is a peak, at the centroid of its three channels. Peaks are
kept in order of the cleared spectrum's height at them, as ROOT inserts
each one found.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clipping import Array
from .extend import cleared, extended, left_slope
from .highres import deconvolved

__all__ = ["THRESHOLD", "WIDEST", "Found", "search_high_res"]

#: ``PEAK_WINDOW / 2``: a ``5 sigma`` at least this is too wide for ROOT.
WIDEST = 512

#: ROOT's error for a threshold that is not a percentage strictly between 0 and 100.
THRESHOLD = "Invalid threshold, must be positive and less than 100"


@dataclass
class Found:
    """The peaks found - positions in channels, highest first - and the deconvolved spectrum."""

    positions: list[float] = field(default_factory=list)
    #: ``None`` where ROOT stops before writing ``destVector``.
    deconvolved: Array | None = None
    #: Whether the peaks filled the ``maxPeaks`` places ROOT keeps.
    full: bool = False


def _refusal(
    size: int, sigma: float, threshold: float, remove: bool, markov: bool, window: int
) -> str | None:
    """ROOT's error for arguments it will not search with, or ``None``."""
    checks = (
        (sigma < 1, "Invalid sigma, must be greater than or equal to 1"),
        (threshold <= 0 or threshold >= 100, THRESHOLD),
        (int(5.0 * sigma + 0.5) >= WIDEST, "Too large sigma"),
        (markov and window <= 0, "Averaging window must be positive"),
        (remove and size < 2 * int(7 * sigma + 0.5) + 1, "Too large clipping window"),
    )
    return next((message for failed, message in checks if failed), None)


def _centroid(values: Array, channel: int, shift: int, size: int) -> float:
    """The centroid of the three channels around ``channel``, within the spectrum."""
    moment = total = 0.0
    for near in range(channel - 1, channel + 2):
        moment += float(near - shift) * float(values[near])
        total += float(values[near])
    at = moment / total
    return float(size - 1) if at >= size else 0.0 if at < 0 else at


def _insert(positions: list[float], at: float, height: Array, shift: int, limit: int) -> None:
    """Put the peak at ``at`` among ``positions`` as ROOT does: before the first lower one."""
    count = len(positions)
    tall = height[shift + int(at)]
    place = next((k for k in range(count) if tall > height[shift + int(positions[k])]), None)
    if place is None:
        if count < limit:
            positions.append(at)
        return
    positions.insert(place, at)
    del positions[limit:]


def _candidates(decon: Array, base: Array, limit: int, inner: range, threshold: float) -> Array:
    """The channels ROOT takes as peaks: local maxima above both thresholds, inside the spectrum.

    The highest channels the thresholds are fractions of are looked for below
    ``limit`` only, as ROOT's loop goes.
    """
    channels = np.arange(decon.shape[0])
    inside = (channels >= inner.start) & (channels < inner.stop)
    counted = inside & (channels < limit)
    top = float(np.max(np.where(counted, decon, 0.0), initial=0.0))
    tallest = float(np.max(np.where(counted, base, 0.0), initial=0.0))
    middle = channels[1:-1]
    here = decon[middle]
    peak = (here > decon[middle - 1]) & (here > decon[middle + 1]) & inside[middle]
    peak &= decon[middle] > min(1.0, threshold) / 100 * top
    peak &= base[middle] > threshold * tallest / 100.0
    return middle[peak]


def search_high_res(
    source: Array, sigma: float, threshold: float, remove: bool,
    iterations: int, markov: bool, window: int, limit: int,
) -> Found | str:  # fmt: skip
    """The peaks of ``source``, at most ``limit`` of them - or ROOT's error refusing them."""
    size = source.shape[0]
    refused = _refusal(size, sigma, threshold, remove, markov, window)
    if refused is not None:
        return refused
    shift = int(7 * sigma + 0.5)
    padded = extended(source, shift, left_slope(source, sigma))
    prepared = cleared(padded, shift, remove, markov, window)
    if prepared is None:
        return Found()
    base, values = prepared
    decon, reach = deconvolved(values, sigma, iterations, shift, size)
    positions: list[float] = []
    inner = range(shift, size + shift)
    for channel in _candidates(decon, base, decon.shape[0] - reach, inner, threshold):
        _insert(positions, _centroid(decon, int(channel), shift, size), base, shift, limit)
    return Found(positions, decon[shift : shift + size], len(positions) == limit)
