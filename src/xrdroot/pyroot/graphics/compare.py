"""How alike two pictures are: 1 for the same picture, towards 0 for nothing alike.

The tutorial harness compares what ROOT drew with what this draws, and a
pixel-exact comparison says nothing useful about two renderers - one
anti-aliases a line where the other does not - so this is the structural
similarity index (SSIM), which weighs whether the same shapes are in the
same places over whether each pixel is the same. scikit-image's is used
when it is installed; otherwise a NumPy one of the same formula, over
8-by-8 windows of the grey pictures, gives nearly the same number. Two
pictures of different sizes are compared at the smaller's size.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["compare_images", "grey"]

#: SSIM's constants, for pictures of values 0 to 1: ``(0.01 L)^2`` and ``(0.03 L)^2``.
C1, C2 = 0.01**2, 0.03**2
#: The window SSIM is measured over, in pixels.
WINDOW = 8


def grey(picture: Any) -> np.ndarray[Any, Any]:
    """A picture - a file's name or an array - as grey values from 0 to 1."""
    if isinstance(picture, np.ndarray):
        pixels = picture.astype(float)
    else:
        from matplotlib.image import imread

        pixels = np.asarray(imread(str(picture)), dtype=float)
    if pixels.max(initial=0.0) > 1.0:
        pixels = pixels / 255.0
    if pixels.ndim == 3:
        pixels = pixels[..., :3].mean(axis=2)
    return pixels


def _resized(pixels: np.ndarray[Any, Any], shape: tuple[int, int]) -> np.ndarray[Any, Any]:
    """``pixels`` sampled down to ``shape``, the nearest pixel for each."""
    rows = (np.arange(shape[0]) * pixels.shape[0] / shape[0]).astype(int)
    columns = (np.arange(shape[1]) * pixels.shape[1] / shape[1]).astype(int)
    return pixels[np.ix_(rows, columns)]


def _windows(pixels: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """The picture cut into whole windows, one row of values per window."""
    rows, columns = (size // WINDOW * WINDOW for size in pixels.shape)
    cut = pixels[:rows, :columns].reshape(rows // WINDOW, WINDOW, columns // WINDOW, WINDOW)
    return cut.transpose(0, 2, 1, 3).reshape(-1, WINDOW * WINDOW)


def _ssim(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
    """SSIM's mean over windows, by its formula."""
    x, y = _windows(a), _windows(b)
    if not len(x):
        return 1.0 if np.allclose(a, b) else 0.0
    mx, my = x.mean(axis=1), y.mean(axis=1)
    vx, vy = x.var(axis=1), y.var(axis=1)
    cov = ((x - mx[:, None]) * (y - my[:, None])).mean(axis=1)
    index = ((2 * mx * my + C1) * (2 * cov + C2)) / ((mx**2 + my**2 + C1) * (vx + vy + C2))
    return float(index.mean())


def compare_images(a: Any, b: Any) -> float:
    """How alike pictures ``a`` and ``b`` are - files or arrays - from 1 down."""
    first, second = grey(a), grey(b)
    shape = (min(first.shape[0], second.shape[0]), min(first.shape[1], second.shape[1]))
    first, second = _resized(first, shape), _resized(second, shape)
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return _ssim(first, second)
    if min(shape) < 7:
        return _ssim(first, second)
    return float(structural_similarity(first, second, data_range=1.0))
