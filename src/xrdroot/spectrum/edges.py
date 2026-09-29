"""Compton edges: the steps SNIP clips through, put back in the spectrum's own shape.

Where the clipped background leaves the spectrum by a count or more,
``TSpectrum::Background`` with ``compton`` looks for where they meet again,
and if the background is higher after the gap than before it, redraws the
gap as the spectrum's running sum scaled to climb from the one level to the
other - from the left when it climbs, from the right when it falls. A
channel's decision needs the channels before it, so this is a loop, over
Python floats as ROOT's is over doubles.
"""

from __future__ import annotations

import numpy as np

from .clipping import Array

__all__ = ["compton_edges"]


def _gap_end(clipped: list[float], spectrum: list[float], start: int) -> int:
    """Where the background meets the spectrum again after ``start`` - the last channel if never."""
    size = len(clipped)
    end = start + 1
    while end < size:
        met = abs(clipped[end] - spectrum[end]) < 1
        end += 1
        if met:
            break
    return end - 1 if end == size else end


def _excess(spectrum: list[float], channels: range, level: float) -> float:
    """``c = c + b - level`` over ``channels``, from zero: the spectrum above ``level`` there."""
    total = 0.0
    for channel in channels:
        total = total + spectrum[channel] - level
    return total


def _redraw(out: Array, spectrum: list[float], channels: range, low: float, high: float) -> None:
    """The gap over ``channels`` as the running excess above ``low``, scaled to reach ``high``."""
    excess = _excess(spectrum, channels, low)
    if not excess > 1:
        return
    scale = (high - low) / excess
    running = 0.0
    for channel in channels:
        running = running + spectrum[channel] - low
        out[channel] = scale * running + low


def compton_edges(clipped: Array, spectrum: Array) -> Array:
    """``clipped`` with each gap under a Compton edge redrawn from ``spectrum``."""
    below, above = clipped.tolist(), np.asarray(spectrum, dtype=np.float64).tolist()
    out = np.array(clipped, dtype=np.float64)
    channel = 0
    while channel < len(below):
        if abs(below[channel] - above[channel]) >= 1:
            start = max(channel - 1, 0)
            end = _gap_end(below, above, start)
            low, high = below[start], below[end]
            if low <= high:
                _redraw(out, above, range(start, end + 1), low, high)
            else:
                _redraw(out, above, range(end, start - 1, -1), high, low)
            channel = end
        channel += 1
    return out
