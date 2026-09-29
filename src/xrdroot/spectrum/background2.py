"""``TSpectrum2::Background``: the background under a two-dimensional spectrum, by clipping.

Each pass compares every channel with an estimate from the eight channels
``r1`` along x and ``r2`` along y from it. Successive filtering takes the
four corners and the four edges between them, raises each edge to the mean
of its two corners, and estimates the channel by the edges' excess over the
corners plus the corners' mean; one-step filtering takes half the edges
less a quarter of the corners at once. A channel is clipped to the estimate
where that is lower and - in ``TSpectrum2::Background``, not in the peak
search - positive. The window grows to the iterations asked for, or shrinks
from them, along each axis to its own number.
"""

from __future__ import annotations

import numpy as np

from .clipping import Array

__all__ = ["background2", "one_step", "successive"]


def _around(values: Array, r1: int, r2: int) -> tuple[Array, ...]:
    """``p1..p4`` (the corners) and ``s1..s4`` (the edges) of every inner channel, ROOT's names."""
    sx, sy = values.shape

    def at(dx: int, dy: int) -> Array:
        return values[r1 + dx : sx - r1 + dx, r2 + dy : sy - r2 + dy]

    return (at(-r1, -r2), at(-r1, r2), at(r1, -r2), at(r1, r2),
            at(0, -r2), at(-r1, 0), at(r1, 0), at(0, r2))  # fmt: skip


def _raised(edge: Array, first: Array, second: Array) -> Array:
    """``edge`` raised to its corners' mean where that is higher, less that mean."""
    mean = (first + second) / 2.0
    return np.where(mean > edge, mean, edge) - mean


def successive(values: Array, r1: int, r2: int) -> Array:
    """The successive filter's estimate for every inner channel."""
    p1, p2, p3, p4, s1, s2, s3, s4 = _around(values, r1, r2)
    s1, s2 = _raised(s1, p1, p3), _raised(s2, p1, p2)
    s3, s4 = _raised(s3, p3, p4), _raised(s4, p2, p4)
    return (s1 + s4) / 2.0 + (s2 + s3) / 2.0 + (p1 + p2 + p3 + p4) / 4.0


def one_step(values: Array, r1: int, r2: int) -> Array:
    """The one-step filter's estimate for every inner channel."""
    p1, p2, p3, p4, s1, s2, s3, s4 = _around(values, r1, r2)
    return -(p1 + p2 + p3 + p4) / 4 + (s1 + s2 + s3 + s4) / 2


def clipped(values: Array, estimate: Array, r1: int, r2: int, positive: bool) -> Array:
    """The inner channels of ``values`` clipped to ``estimate`` where it is lower (and positive)."""
    sx, sy = values.shape
    inner = values[r1 : sx - r1, r2 : sy - r2]
    lower = (estimate < inner) & (estimate > 0) if positive else estimate < inner
    return np.where(lower, estimate, inner)


def _refusal(sx: int, sy: int, nx: int, ny: int) -> str | None:
    """ROOT's message for arguments it will not clip with, or ``None``."""
    if sx <= 0 or sy <= 0:
        return "Wrong parameters"
    if nx < 1 or ny < 1:
        return "Width of Clipping Window Must Be Positive"
    if sx < 2 * nx + 1 or sy < 2 * ny + 1:
        return "Too Large Clipping Window"
    return None


def background2(
    spectrum: Array, nx: int, ny: int, decreasing: bool, one_step_filter: bool
) -> Array | str:
    """The background of ``spectrum`` - or ROOT's message refusing the arguments."""
    sx, sy = spectrum.shape
    refused = _refusal(sx, sy, nx, ny)
    if refused is not None:
        return refused
    values = np.array(spectrum, dtype=np.float64)
    sampling = max(nx, ny)
    widths = range(sampling, 0, -1) if decreasing else range(1, sampling + 1)
    for width in widths:
        r1, r2 = min(width, nx), min(width, ny)
        filtered = one_step(values, r1, r2) if one_step_filter else successive(values, r1, r2)
        found = clipped(values, filtered, r1, r2, True)
        # One-step filtering copies back only ``width`` in from each edge, as ROOT does.
        c1, c2 = (width, width) if one_step_filter else (r1, r2)
        values[c1 : sx - c1, c2 : sy - c2] = found[c1 - r1 : sx - c1 - r1, c2 - r2 : sy - c2 - r2]
    return values
