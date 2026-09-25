"""ROOT's colours, by the index a file keeps them under.

A ROOT object never writes a colour, only a number: ``fLineColor = 2`` is
whatever colour 2 was in the session that drew it. The numbers every
session starts with are made by ``TColor::InitializeColors``, and this is
that table - the fifty named colours, the fifty of the "pretty" spectrum
after them, the six colour circles ``kRed`` to ``kCyan`` with their
lighter and darker rings, and the greys of ``kGray``.

The six colour rectangles - ``kOrange``, ``kSpring``, ``kTeal``,
``kAzure``, ``kViolet`` and ``kPink`` - are the one part given only
approximately: each is the base colour ROOT gives it, lightened or darkened
by its offset, rather than ROOT's own table of twenty. A canvas ROOT saved
after drawing carries the colours it used as a ``ListOfColors``, and
those, when there, are used instead of all of this; an index neither
knows is drawn black. This module is private to the canvas: whatever the
rest of the library does with colours, a canvas draws with ROOT's.
"""

from __future__ import annotations

import colorsys
from typing import Any

__all__ = ["BIRD", "Colors"]

RGB = tuple[float, float, float]

#: The colours ``TColor::InitializeColors`` names, 0 to 50.
NAMED: dict[int, RGB] = {
    0: (1.0, 1.0, 1.0),
    1: (0.0, 0.0, 0.0),
    2: (1.0, 0.0, 0.0),
    3: (0.0, 1.0, 0.0),
    4: (0.0, 0.0, 1.0),
    5: (1.0, 1.0, 0.0),
    6: (1.0, 0.0, 1.0),
    7: (0.0, 1.0, 1.0),
    8: (0.35, 0.83, 0.33),
    9: (0.35, 0.33, 0.85),
    10: (0.999, 0.999, 0.999),
    11: (0.754, 0.715, 0.676),
    12: (0.3, 0.3, 0.3),
    13: (0.4, 0.4, 0.4),
    14: (0.5, 0.5, 0.5),
    15: (0.6, 0.6, 0.6),
    16: (0.7, 0.7, 0.7),
    17: (0.8, 0.8, 0.8),
    18: (0.9, 0.9, 0.9),
    19: (0.95, 0.95, 0.95),
    20: (0.8, 0.78, 0.67),
    21: (0.8, 0.78, 0.67),
    22: (0.76, 0.75, 0.66),
    23: (0.73, 0.71, 0.64),
    24: (0.70, 0.65, 0.59),
    25: (0.72, 0.64, 0.61),
    26: (0.68, 0.6, 0.55),
    27: (0.61, 0.56, 0.51),
    28: (0.53, 0.4, 0.34),
    29: (0.69, 0.81, 0.78),
    30: (0.52, 0.76, 0.64),
    31: (0.54, 0.66, 0.63),
    32: (0.51, 0.62, 0.55),
    33: (0.68, 0.74, 0.78),
    34: (0.48, 0.56, 0.6),
    35: (0.46, 0.54, 0.57),
    36: (0.41, 0.51, 0.59),
    37: (0.43, 0.48, 0.52),
    38: (0.49, 0.6, 0.82),
    39: (0.5, 0.5, 0.61),
    40: (0.67, 0.65, 0.75),
    41: (0.83, 0.81, 0.53),
    42: (0.87, 0.73, 0.53),
    43: (0.74, 0.62, 0.51),
    44: (0.78, 0.6, 0.49),
    45: (0.75, 0.51, 0.47),
    46: (0.81, 0.37, 0.38),
    47: (0.67, 0.56, 0.58),
    48: (0.65, 0.47, 0.48),
    49: (0.58, 0.41, 0.44),
    50: (0.83, 0.35, 0.33),
    920: (0.8, 0.8, 0.8),
    921: (0.6, 0.6, 0.6),
    922: (0.4, 0.4, 0.4),
    923: (0.2, 0.2, 0.2),
}

#: The colour circles: which of red, green and blue each one is made of.
CIRCLES: dict[int, RGB] = {
    632: (1, 0, 0),  # kRed
    416: (0, 1, 0),  # kGreen
    600: (0, 0, 1),  # kBlue
    400: (1, 1, 0),  # kYellow
    616: (1, 0, 1),  # kMagenta
    432: (0, 1, 1),  # kCyan
}
#: ``CreateColorsCircle``'s fifteen rings, from ``base - 10`` to ``base + 4``:
#: the level of the channels the colour is made of, and of the others.
RINGS = (
    (255, 204), (255, 153), (204, 153), (255, 102), (204, 102), (153, 102),
    (255, 51), (204, 51), (153, 51), (102, 51),
    (255, 0), (204, 0), (153, 0), (102, 0), (51, 0),
)  # fmt: skip
#: The colour rectangles, by the colour at their base.
RECTANGLES: dict[int, RGB] = {
    800: (1.0, 0.8, 0.0),  # kOrange
    820: (0.8, 1.0, 0.0),  # kSpring
    840: (0.0, 1.0, 0.8),  # kTeal
    860: (0.0, 0.6, 1.0),  # kAzure
    880: (0.8, 0.0, 1.0),  # kViolet
    900: (1.0, 0.0, 0.6),  # kPink
}
#: How far the offsets of a rectangle reach: ``base - 9`` to ``base + 10``.
RECTANGLE_SPAN = (-9, 10)

#: ROOT 6's default palette, ``kBird``: the stops of red, green and blue.
BIRD = (
    (0.2082, 0.1664, 0.5293),
    (0.0592, 0.3599, 0.8684),
    (0.0780, 0.5041, 0.8385),
    (0.0232, 0.6419, 0.7914),
    (0.1802, 0.7178, 0.6425),
    (0.5301, 0.7492, 0.4662),
    (0.8186, 0.7328, 0.3499),
    (0.9956, 0.7862, 0.1968),
    (0.9764, 0.9832, 0.0539),
)
#: What an index nobody defined is drawn as.
UNKNOWN: RGB = (0.0, 0.0, 0.0)


def _pretty() -> dict[int, RGB]:
    """Colours 51 to 100: the spectrum from violet to red, hue alone changing."""
    made = {}
    for step in range(50):
        hue = 280.0 - (step + 1) * 280.0 / 50
        made[51 + step] = colorsys.hls_to_rgb(hue / 360.0, 0.5, 1.0)
    return made


def _circles() -> dict[int, RGB]:
    """The six colour circles, each fifteen rings round its base."""
    made: dict[int, RGB] = {}
    for base, channels in CIRCLES.items():
        for offset, (on, off) in enumerate(RINGS, start=-10):
            made[base + offset] = (
                (on if channels[0] else off) / 255,
                (on if channels[1] else off) / 255,
                (on if channels[2] else off) / 255,
            )
    return made


def _shifted(color: RGB, offset: int) -> RGB:
    """``color`` lightened for a negative offset, darkened for a positive one."""
    if offset < 0:
        towards = -offset / 10.0
        return (
            color[0] + (1 - color[0]) * towards,
            color[1] + (1 - color[1]) * towards,
            color[2] + (1 - color[2]) * towards,
        )
    keep = 1.0 - offset / 12.0
    return (color[0] * keep, color[1] * keep, color[2] * keep)


def _rectangles() -> dict[int, RGB]:
    """The six colour rectangles, approximately: see the module's docstring."""
    low, high = RECTANGLE_SPAN
    return {
        base + offset: _shifted(color, offset)
        for base, color in RECTANGLES.items()
        for offset in range(low, high + 1)
    }


#: Every colour a ROOT session starts with, by index.
DEFAULT: dict[int, RGB] = {**_pretty(), **_circles(), **_rectangles(), **NAMED}


class Colors:
    """The colour table one canvas draws with: ROOT's, then what it saved.

    >>> Colors().rgb(2)
    (1.0, 0.0, 0.0)
    """

    __slots__ = ("table", "palette")

    def __init__(self) -> None:
        self.table: dict[int, RGB] = dict(DEFAULT)
        #: The indices of the palette the canvas was drawn with, if it saved one.
        self.palette: list[int] = []

    def adopt(self, colors: list[Any]) -> None:
        """Take the ``TColor`` objects a canvas saved, each by its number."""
        for color in colors:
            number = int(color.get("fNumber", -1))
            if number >= 0:
                self.table[number] = (
                    float(color.get("fRed", 0.0)),
                    float(color.get("fGreen", 0.0)),
                    float(color.get("fBlue", 0.0)),
                )

    def rgb(self, index: Any) -> RGB:
        """The colour of ``index``, black for one this table does not have."""
        return self.table.get(int(index), UNKNOWN)

    def rgba(self, index: Any, alpha: float = 1.0) -> tuple[float, float, float, float]:
        """The colour of ``index`` with an opacity, as matplotlib takes one."""
        red, green, blue = self.rgb(index)
        return (red, green, blue, alpha)

    def colormap(self) -> Any:
        """The palette a colour plot is drawn in: the saved one, or ``kBird``."""
        from matplotlib.colors import LinearSegmentedColormap, ListedColormap

        if self.palette:
            return ListedColormap([self.rgb(index) for index in self.palette], "saved")
        return LinearSegmentedColormap.from_list("kBird", list(BIRD))
