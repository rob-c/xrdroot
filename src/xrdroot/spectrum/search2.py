"""``TSpectrum2::SearchHighRes``: peaks found in a plane sharpened by deconvolution.

The plane is extended and cleared (:mod:`.prepare2`), deconvolved by a
two-dimensional Gaussian of the peaks' ``sigma`` with Gold's method, and
every channel higher than its eight neighbours and than ``threshold`` per
cent of the highest is a peak, at the centroids of its three channels
along x and along y. Peaks are ranked by the cleared plane's height as each
is found, in ROOT's order - x outermost - until ``maxPeaks`` are kept.
"""

from __future__ import annotations

import numpy as np

from ..random import libm
from .clipping import Array
from .gold2 import Response2, autocorrelation, correlated
from .prepare2 import cleared2
from .search import THRESHOLD, WIDEST, Found

__all__ = ["search_high_res2"]

#: Below this ROOT leaves a channel as the last iteration had it.
FLOOR = 0.000001


def gaussian_plane(sx: int, sy: int, sigma: float) -> Array:
    """``(Int_t)(1000 exp(-((i - 3 sigma)^2 + (j - 3 sigma)^2) / (2 sigma sigma)))``."""
    dx = np.arange(sx, dtype=np.float64)[:, None] - 3 * sigma
    dy = np.arange(sy, dtype=np.float64)[None, :] - 3 * sigma
    exponent = (dx * dx + dy * dy) / (2 * sigma * sigma)
    return np.trunc(1000 * np.asarray(libm.exp(-exponent), dtype=np.float64))


def deconvolved2(values: Array, sigma: float, iterations: int) -> Array:
    """``values`` deconvolved by the Gaussian, scaled by its area and shifted by its peak."""
    sx, sy = values.shape
    plane = gaussian_plane(sx, sy, sigma)
    response = Response2(plane)
    lhx, lhy = response.lengths
    hty = correlated(plane[:lhx, :lhy], np.abs(values), (0, 0))
    kernel = autocorrelation(plane, lhx, lhy)
    x, kept = np.ones((sx, sy)), np.zeros((sx, sy))
    for _ in range(iterations):
        active = (x > FLOOR) & (hty > FLOOR)
        denominator = correlated(kernel, x, (1 - lhx, 1 - lhy))
        with np.errstate(all="ignore"):
            keep = (hty * x != 0) & (denominator != 0)
            ratio = np.where(keep, x * hty / np.where(denominator != 0, denominator, 1.0), 0.0)
        kept = np.where(active, ratio, kept)
        x = kept
    return np.roll(response.area * x, response.position, axis=(0, 1))


def _refusal(sigma: float, threshold: float, markov: bool, window: int) -> str | None:
    """ROOT's error for arguments it will not search with, or ``None``."""
    checks = (
        (sigma < 1, "Invalid sigma, must be greater than or equal to 1"),
        (threshold <= 0 or threshold >= 100, THRESHOLD),
        (int(5.0 * sigma + 0.5) >= WIDEST, "Too large sigma"),
        (markov and window <= 0, "Averaging window must be positive"),
    )
    return next((message for failed, message in checks if failed), None)


def _centroid(line: Array, at: int, shift: int, size: int) -> float:
    """The centroid of ``line``'s three channels around ``at``, within ``size``."""
    moment = total = 0.0
    for near in range(at - 1, at + 2):
        moment += float(near - shift) * float(line[near])
        total += float(line[near])
    found = moment / total
    return 0.0 if found < 0 else float(size - 1) if found >= size else found


def _maxima(plane: Array, shift: int, inner: tuple[int, int], threshold: float) -> Array:
    """The channels higher than their eight neighbours and the threshold, inside, x outermost."""
    sx, sy = plane.shape
    here = plane[1:-1, 1:-1]
    higher = np.ones(here.shape, dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                higher &= here > plane[1 + dx : sx - 1 + dx, 1 + dy : sy - 1 + dy]
    i, j = np.nonzero(higher)
    i, j = i + 1, j + 1
    inside = (i >= shift) & (i < inner[0] + shift) & (j >= shift) & (j < inner[1] + shift)
    top = float(np.max(plane, initial=0.0))
    tall = plane[i, j] > threshold * top / 100.0
    return np.stack([i[inside & tall], j[inside & tall]], axis=1)


def _insert2(
    found: list[tuple[float, float]], peak: tuple[float, float], base: Array, shift: int
) -> None:
    """Put ``peak`` among ``found`` before the first lower one, as ROOT ranks them."""

    def height(at: tuple[float, float]) -> float:
        return float(base[shift + int(at[0] + 0.5), shift + int(at[1] + 0.5)])

    tall = height(peak)
    place = next((k for k, other in enumerate(found) if tall > height(other)), len(found))
    found.insert(place, peak)


def search_high_res2(
    source: Array, sigma: float, threshold: float, remove: bool,
    iterations: int, markov: bool, window: int, limit: int,
) -> tuple[Found, list[float]] | str:  # fmt: skip
    """The peaks of ``source``, x and y, at most ``limit`` - or ROOT's error refusing them."""
    refused = _refusal(sigma, threshold, markov, window)
    if refused is not None:
        return refused
    sx, sy = source.shape
    reach = int(4 * sigma + 0.5)
    shift = 2 * reach
    prepared = cleared2(source, shift, reach, remove, markov, window)
    if prepared is None:
        return Found(), []
    base, values = prepared
    plane = deconvolved2(values, sigma, iterations)
    found: list[tuple[float, float]] = []
    for i, j in _maxima(plane, shift, (sx, sy), threshold).tolist():
        if len(found) < limit:
            peak = (_centroid(plane[:, j], i, shift, sx), _centroid(plane[i, :], j, shift, sy))
            _insert2(found, peak, base, shift)
    decon = plane[shift : shift + sx, shift : shift + sy]
    return Found([x for x, _ in found], decon, len(found) == limit), [y for _, y in found]
