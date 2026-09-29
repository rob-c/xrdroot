"""``TSpectrum::Background``: the background under a spectrum's peaks, by SNIP clipping.

Each pass of the Sensitive Nonlinear Iterative Peak clipping replaces every
channel ``j`` by the smaller of itself and an estimate from channels ``i``
away - their mean, or for a higher filter order the largest of that and
fourth-, sixth- and eighth-order combinations of channels ``i / 2``,
``i / 3`` and ``i / 4`` away - so peaks are cut down while a smooth
background stays. The window ``i`` shrinks from ``numberIterations`` to 1,
or grows to it; with smoothing every channel read is first the mean of the
``smoothWindow`` around it. Compton edges are then restored by the
spectrum's own shape where the clipping cut through a step.
"""

from __future__ import annotations

import numpy as np

from .clipping import Array, combination, shifted_mean, terms
from .edges import compton_edges

__all__ = ["background", "clip_pass"]

#: The filter orders ROOT clips with; with any other it leaves the spectrum as it is.
FILTER_ORDERS = (2, 4, 6, 8)

#: The smoothing windows ``TSpectrum`` has constants for: ``kBackSmoothing3`` to ``15``.
WINDOWS = (3, 5, 7, 9, 11, 13, 15)


def _refusal(size: int, iterations: int, smoothing: bool, window: int) -> str | None:
    """ROOT's message for arguments it will not clip with, or ``None``."""
    if size <= 0:
        return "Wrong Parameters"
    if iterations < 1:
        return "Width of Clipping Window Must Be Positive"
    if size < 2 * iterations + 1:
        return "Too Large Clipping Window"
    if smoothing and window not in WINDOWS:
        return "Incorrect width of smoothing window"
    return None


def _estimate(values: Array, centres: Array, width: int, order: int, half: int) -> Array:
    """The clipping estimate at each centre for window ``width``: ROOT's ``b``, orders and all."""
    smoothed = half > 0
    left = shifted_mean(values, centres - width, half)
    right = shifted_mean(values, centres + width, half)
    estimate = (left + right) / 2
    for higher in (8, 6, 4):
        if order < higher:
            continue
        step, divisor, signs = terms(higher, smoothed)
        reach = width // step
        means = [shifted_mean(values, centres + far * reach, half) for _, _, far in signs]
        other = combination(means, signs, divisor, each=not smoothed)
        estimate = np.where(estimate < other, other, estimate)
    return estimate


def clip_pass(values: Array, width: int, order: int, half: int) -> Array:
    """One pass of clipping at window ``width``: channels ``width`` to ``size - width`` clipped.

    Unsmoothed (``half`` 0) a channel becomes the estimate where that is below
    it; smoothed, it becomes the estimate there and the mean around it
    elsewhere - which is what ROOT's ``if (b < a) av = b`` leaves.
    """
    size = values.shape[0]
    centres = np.arange(width, size - width)
    here = values[centres]
    estimate = _estimate(values, centres, width, order, half)
    kept = shifted_mean(values, centres, half) if half else here
    out = values.copy()
    out[centres] = np.where(estimate < here, estimate, kept)
    return out


def background(
    spectrum: Array, iterations: int, decreasing: bool, order: int,
    smoothing: bool, window: int, compton: bool,
) -> Array | str:  # fmt: skip
    """The background of ``spectrum`` - or ROOT's message refusing the arguments.

    ``order`` is 2, 4, 6 or 8; ``window`` the smoothing window's width.
    """
    size = spectrum.shape[0]
    refused = _refusal(size, iterations, smoothing, window)
    if refused is not None:
        return refused
    half = (window - 1) // 2 if smoothing else 0
    widths = range(iterations, 0, -1) if decreasing else range(1, iterations + 1)
    if order not in FILTER_ORDERS:
        widths = range(0)
    clipped = np.array(spectrum, dtype=np.float64)
    for width in widths:
        clipped = clip_pass(clipped, width, order, half)
    return compton_edges(clipped, spectrum) if compton else clipped
