"""``TImage`` and ``TImagePalette``: a picture made from a grid of values and drawn in a pad.

``TImage::Create`` gives a ``TASImage``; ``SetImage(values, width, palette)``
makes its pixels from the values, the bottom row first, coloured by the
palette as libAfterImage colours them (:func:`xrdroot.palette.colorize`).
Drawn, the picture fills the pad it is drawn in. Reading, writing and
editing pictures - ``TImage::Open``, ``WriteImage``, ``Scale``, text drawn
into them - are not here yet, and are refused by name.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...canvas.model import Primitive
from ...errors import UnsupportedFeatureError
from ...palette import CHANNEL_ORDER, colorize
from ..core.objects import TNamed, TObject

__all__ = ["TASImage", "TImage", "TImagePalette"]


class TImagePalette(TObject):
    """``TImagePalette(n)``: ``n`` stops, each a place from 0 to 1 and a 16-bit colour."""

    def __init__(self, count: int = 0) -> None:
        super().__init__()
        self.fNumPoints = int(count)
        self.fPoints = np.zeros(self.fNumPoints)
        self.fColorRed, self.fColorGreen, self.fColorBlue, self.fColorAlpha = (
            np.zeros(self.fNumPoints, np.int64) for _ in range(4))  # fmt: skip

    def _dict(self) -> dict[str, Any]:
        palette = {name: np.asarray(getattr(self, name), np.int64) for name in CHANNEL_ORDER}
        return {"fPoints": np.asarray(self.fPoints, np.float64), **palette}


def grey_palette() -> TImagePalette:
    """The palette ``TFITSHDU::ReadAsImage`` makes: 256 greys, black to white."""
    palette = TImagePalette(256)
    palette.fPoints = np.arange(256) / 255.0
    palette.fColorRed = palette.fColorGreen = palette.fColorBlue = np.arange(256) << 8
    palette.fColorAlpha = np.full(256, 0xFFFF)
    return palette


class TASImage(TNamed):
    """``TASImage``: a picture, as RGBA pixels top row first."""

    def __init__(self, *args: Any) -> None:
        if args:
            raise UnsupportedFeatureError("A TASImage opened from a file or made to a size is "
                                          "not supported yet; TImage::Create and SetImage are.")
        super().__init__("", "")
        self.rgba = np.zeros((0, 0, 4), np.uint8)

    def SetImage(self, values: Any, width: int, palette: Any = None) -> None:
        """``SetImage(values, width, palette)``: pixels from values, the bottom row first."""
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        grid = flat.reshape(-1, int(width))[::-1]
        chosen = palette if palette is not None else grey_palette()
        self.rgba = colorize(grid, chosen._dict())

    def GetWidth(self) -> int:
        return int(self.rgba.shape[1])

    def GetHeight(self) -> int:
        return int(self.rgba.shape[0])

    def IsValid(self) -> bool:
        return bool(self.rgba.size > 0)

    @property
    def _xrd(self) -> Any:
        """What a pad paints of this picture: its pixels, filling the pad."""
        return Primitive("TASImage", {"rgba": self.rgba})

    def Draw(self, option: str = "") -> None:
        from ..core import hooks

        hooks.draw_hook(self, str(option))

    def __getattr__(self, name: str) -> Any:
        if name.startswith(("Open", "Write", "Scale", "Draw", "Crop", "Flip", "Gray", "Paint")):
            raise UnsupportedFeatureError(f"TASImage::{name} is not supported yet: a picture "
                                          "here is made with SetImage and drawn whole.")
        raise AttributeError(name)


class TImage(TASImage):
    """``TImage``: what ``TImage::Create`` makes, which is a ``TASImage``."""

    @staticmethod
    def Create() -> TASImage:
        return TASImage()

    @staticmethod
    def Open(*args: Any) -> TASImage:
        raise UnsupportedFeatureError("TImage::Open is not supported yet: a picture here is "
                                      "made with TImage::Create and SetImage.")
