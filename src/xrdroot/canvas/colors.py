"""The colours one canvas draws with: ROOT's table, then whatever it saved.

A ROOT object never writes a colour, only a number: ``fLineColor = 2`` is
whatever colour 2 was in the session that drew it. The numbers every session
starts with are :mod:`xrdroot.plot.colors`' table, ROOT's own to the bit. A
canvas ROOT saved after drawing also carries the colours it used, as a
``ListOfColors`` of ``TColor`` - a colour a macro made for itself among them,
which no table has - and its palette; those, when there, come first.
"""

from __future__ import annotations

from typing import Any

from ..plot.colors import COLORS, PALETTES, UNKNOWN

__all__ = ["Colors"]

RGB = tuple[float, float, float]


def _rgb(hexed: str) -> RGB:
    """``"#rrggbb"`` as three fractions."""
    return (int(hexed[1:3], 16) / 255, int(hexed[3:5], 16) / 255, int(hexed[5:7], 16) / 255)


class Colors:
    """The colour table one canvas draws with.

    >>> Colors().rgb(2)
    (1.0, 0.0, 0.0)
    """

    __slots__ = ("saved", "palette")

    def __init__(self) -> None:
        #: The colours the canvas saved, by index, over ROOT's own.
        self.saved: dict[int, RGB] = {}
        #: The indices of the palette the canvas was drawn with, if it saved one.
        self.palette: list[int] = []

    def adopt(self, colors: list[Any]) -> None:
        """Take the ``TColor`` objects a canvas saved, each by its number."""
        for color in colors:
            number = int(color.get("fNumber", -1))
            if number >= 0:
                self.saved[number] = (
                    float(color.get("fRed", 0.0)),
                    float(color.get("fGreen", 0.0)),
                    float(color.get("fBlue", 0.0)),
                )

    def rgb(self, index: Any) -> RGB:
        """The colour of ``index``: saved, ROOT's, or black for one nobody defined."""
        number = int(index)
        if number in self.saved:
            return self.saved[number]
        return _rgb(COLORS.get(number, UNKNOWN))

    def rgba(self, index: Any, alpha: float = 1.0) -> tuple[float, float, float, float]:
        """The colour of ``index`` with an opacity, as matplotlib takes one."""
        red, green, blue = self.rgb(index)
        return (red, green, blue, alpha)

    def hexed(self, index: Any) -> str:
        """The colour of ``index`` as ``"#rrggbb"``, which every drawing library takes."""
        return "#" + "".join(f"{round(channel * 255):02x}" for channel in self.rgb(index))

    def colormap(self) -> Any:
        """The palette a colour plot is drawn in: the saved one, or ROOT's ``kBird``."""
        from matplotlib.colors import ListedColormap

        if self.palette:
            return ListedColormap([self.rgb(index) for index in self.palette], "saved")
        return ListedColormap(list(PALETTES["bird"]), "kBird")
