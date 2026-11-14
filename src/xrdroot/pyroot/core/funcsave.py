"""``TF2::Save`` and ``TF3::Save``: a function sampled on a grid, and read back between its points.

``Save(xmin, xmax, ymin, ymax, zmin, zmax)`` samples the function at
``Npx + 1`` by ``Npy + 1`` (by ``Npz + 1``) points spanning the ranges
given; an axis given no range - its maximum not above its minimum - is
sampled at the middles of the function's own ``Npx`` bins instead.
``GetSave(x)`` interpolates the samples linearly along each axis, as ROOT's
bilinear and trilinear ``GetSave`` do, and is zero outside them.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

__all__ = ["Samples", "sample", "interpolate"]

#: The points along each axis, and the function's values on their grid.
Samples = tuple[list[np.ndarray[Any, Any]], np.ndarray[Any, Any]]


def _axis(low: float, high: float, own: tuple[float, float], npx: int) -> np.ndarray[Any, Any]:
    """An axis's points: ``npx + 1`` over ``low..high``, or the middles of ``own``'s bins."""
    if high <= low:
        half = (own[1] - own[0]) / npx / 2
        low, high, npx = own[0] + half, own[1] - half, npx - 1
    return np.linspace(low, high, max(npx, 1) + 1)


def sample(evaluate: Callable[[list[np.ndarray[Any, Any]]], Any], ranges: Sequence[float],
           own: Sequence[tuple[float, float]], counts: Sequence[int]) -> Samples:
    """The grid over ``ranges`` (``low, high`` per axis), and the function's values on it."""
    axes = [_axis(float(ranges[2 * d]), float(ranges[2 * d + 1]), own[d], int(counts[d]))
            for d in range(len(counts))]
    mesh = np.meshgrid(*axes, indexing="ij")
    values = np.asarray(evaluate([m.ravel() for m in mesh]), dtype=np.float64)
    return axes, values.reshape(mesh[0].shape)


def _cell(point: float, axis: np.ndarray[Any, Any]) -> tuple[int, float]:
    """The interval of ``axis`` the point is in, and how far along it."""
    intervals = len(axis) - 1
    width = (axis[-1] - axis[0]) / intervals
    at = min(intervals - 1, int((point - axis[0]) / width))
    return at, (point - (axis[0] + at * width)) / width


def interpolate(samples: Samples | None, point: Sequence[float]) -> float:
    """``GetSave``: the samples interpolated linearly at ``point``; zero outside them."""
    if samples is None:
        return 0.0
    axes, values = samples
    if any(not axis[0] <= x <= axis[-1] for x, axis in zip(point, axes, strict=False)):
        return 0.0
    cells = [_cell(float(x), axis) for x, axis in zip(point, axes, strict=False)]
    total = 0.0
    for corner in itertools.product((0, 1), repeat=len(cells)):
        weight = float(np.prod([f if c else 1 - f for (_, f), c in zip(cells, corner,
                                                                         strict=True)]))
        total += weight * float(values[tuple(i + c for (i, _), c in zip(cells, corner,
                                                                         strict=True))])
    return total
