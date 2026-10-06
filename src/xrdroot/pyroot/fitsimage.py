"""What ``TFITSHDU`` makes of an image unit: a picture, a matrix, a histogram, a row.

Every one refuses a table unit with ROOT's warning and gives ``nullptr``,
which is ``None``; the pixels are the unit's, ``NAXIS1`` fastest.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.messages import Warning
from .linalg.matrices import TMatrixD
from .vectors import TVectorD

__all__ = ["ImageReads"]


def _layer_pixels(unit: Any, where: str, layer: int) -> np.ndarray[Any, Any] | None:
    """One layer's pixels, or ``None`` - warned of - for a unit that is not a picture."""
    sizes = unit._sizes
    if len(sizes) not in (2, 3, 4) or (len(sizes) == 4 and sizes[3] > 1):
        Warning(where, "could not convert image HDU to image because it has %d dimensions.",
                len(sizes))  # fmt: skip
        return None
    if (len(sizes) == 2 and layer > 0) or (len(sizes) > 2 and layer >= sizes[2]):
        Warning(where, "layer out of bounds.")
        return None
    count = sizes[0] * sizes[1]
    return np.asarray(unit._pixels[layer * count:(layer + 1) * count])


def _range(pixels: np.ndarray[Any, Any]) -> tuple[float, float]:
    """The smallest pixel and the largest, as ``TFITSHDU`` finds them: the largest from 0."""
    return float(pixels.min()), max(float(pixels.max()), 0.0)


def _stretched(pixels: np.ndarray[Any, Any], top: float) -> np.ndarray[Any, Any]:
    """Pixels spread over 0 to ``top``, from the smallest to the largest; all ``top`` when
    those are the same."""
    low, high = _range(pixels)
    if high == low:
        return np.full(len(pixels), top)
    return (top / (high - low)) * (pixels - low)


class ImageReads:
    """``ReadAsImage``, ``ReadAsMatrix``, ``ReadAsHistogram``, ``GetArrayRow`` and
    ``GetArrayColumn``."""

    _sizes: list[int]
    _pixels: np.ndarray[Any, Any]
    _kind: str

    def _is_image(self, where: str) -> bool:
        if self._kind != "IMAGE":
            Warning(where, "this is not an image HDU.")
        return self._kind == "IMAGE"

    def ReadAsImage(self, layer: int = 0, pal: Any = None) -> Any:
        from .graphics.images import TImage

        pixels = _layer_pixels(self, "ReadAsImage", int(layer)) if self._is_image(
            "ReadAsImage") else None  # fmt: skip
        if pixels is None:
            return None
        image = TImage.Create()
        image.SetImage(_stretched(pixels, 255.0), self._sizes[0], pal)
        return image

    def ReadAsMatrix(self, layer: int = 0, opt: str = "") -> Any:
        pixels = _layer_pixels(self, "ReadAsMatrix", int(layer)) if self._is_image(
            "ReadAsMatrix") else None  # fmt: skip
        if pixels is None:
            return None
        scaled = str(opt)[:1] in ("S", "s")
        low, high = _range(pixels)
        if scaled and high == low:
            return None  # ROOT fills its buffer with ones and makes no matrix
        values = _stretched(pixels, 1.0) if scaled else pixels
        return TMatrixD(values.reshape(self._sizes[1], self._sizes[0]))

    def ReadAsHistogram(self) -> Any:
        from .core.hists import TH1D, TH2D, TH3D

        if not self._is_image("ReadAsHistogram"):
            return None
        sizes = self._sizes
        if len(sizes) not in (1, 2, 3):
            Warning("ReadAsHistogram", "could not convert image HDU to histogram because it "
                    "has %d dimensions.", len(sizes))  # fmt: skip
            return None
        made = (TH1D, TH2D, TH3D)[len(sizes) - 1]
        axes = [arg for n in sizes for arg in (n, 0, n - 1)]
        histogram = made("", "", *axes)
        grid = np.indices(sizes[::-1]).reshape(len(sizes), -1)[::-1].astype(np.float64)
        weights = np.maximum(self._pixels.astype(np.int64), 0).astype(np.float64)
        histogram._xrd.fill(*grid, weight=weights)
        return histogram

    def _line(self, where: str, index: int, across: bool) -> TVectorD | None:
        if not self._is_image(where):
            return None
        if len(self._sizes) != 2:
            Warning(where, "could not get row from HDU because it has %d dimensions.",
                    len(self._sizes))  # fmt: skip
            return None
        grid = self._pixels.reshape(self._sizes[1], self._sizes[0])
        if index < 0 or index >= grid.shape[0 if across else 1]:
            Warning(where, "index out of bounds.")
            return None
        return TVectorD(grid[index] if across else grid[:, index])

    def GetArrayRow(self, row: int) -> TVectorD | None:
        return self._line("GetArrayRow", int(row), True)

    def GetArrayColumn(self, col: int) -> TVectorD | None:
        return self._line("GetArrayColumn", int(col), False)
