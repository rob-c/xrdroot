"""``fSave``: the values ROOT sampled a function at, for when the function is not there.

A ``TF1`` defined by C++ - a function pointer, a lambda, a convolution -
cannot be written to a file, since the code is not data. What ROOT writes
instead is the function sampled over its range: ``fNpx + 1`` evenly spaced
values, then the two ends they span. Reading such a function back, ROOT's
``TF1::EvalPar`` has no code to call, and ``TF1::GetSave`` interpolates
between the samples instead - a straight line between the two either side,
and zero outside the range, where the samples say nothing.

:func:`interpolate` is ``GetSave`` and :func:`sample` is ``TF1::Save``, which
is what a function defined here by Python code is written as.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError

__all__ = ["interpolate", "sample", "usable", "kept"]

Array = Any


def kept(save: Any) -> bool:
    """Whether there are samples at all: at least two values and the two ends."""
    return len(np.asarray(save if save is not None else ())) >= 4


def usable(save: Any, name: str) -> tuple[Array, float, float]:
    """The samples and the two ends they span, or a refusal saying why there are none.

    A function a histogram's fit saved has its samples at the bin centres
    and says so by repeating its upper end, and interpolating those needs
    the histogram's axis, which the function does not carry.
    """
    values = np.asarray(save if save is not None else (), dtype=np.float64)
    if len(values) < 4 or not values[-1] > values[-2]:
        reason = (
            "holds no saved values"
            if len(values) < 4 or values[-1] != values[-2]
            else "saved its values at the bins of the histogram it was fitted to"
        )
        raise UnsupportedFeatureError(
            f"{name!r} is a function defined by compiled code, which is not in the file, "
            f"and it {reason}, so there is nothing here to evaluate it from"
        )
    return values[:-2], float(values[-2]), float(values[-1])


def interpolate(save: Any, name: str, x: Array) -> Array:
    """``TF1::GetSave``: the straight line between the two samples either side of ``x``."""
    values, low, high = usable(save, name)
    x = np.asarray(x, dtype=np.float64)
    intervals = len(values) - 1
    dx = (high - low) / intervals
    with np.errstate(invalid="ignore"):
        at = np.clip(np.nan_to_num((x - low) / dx).astype(np.int64), 0, intervals - 1)
    below = low + at * dx
    above = below + dx
    y_low, y_high = values[at], values[at + 1]
    line = ((above * y_low - below * y_high) + x * (y_high - y_low)) / dx
    outside = (x < low) | (x > high)
    return np.where(np.isnan(x), x, np.where(outside, 0.0, line))


def sample(evaluate: Callable[[Array], Array], low: float, high: float, npx: int) -> Array:
    """``TF1::Save``: ``npx + 1`` values from ``low`` to ``high``, then the two ends."""
    dx = (high - low) / npx
    points = low + dx * np.arange(npx + 1)
    values = np.asarray(evaluate(points), dtype=np.float64)
    return np.concatenate([values, [low, high]])
