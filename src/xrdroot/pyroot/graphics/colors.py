"""``TColor``: the colours a session makes for itself, and the palettes it shades with.

Every colour in ROOT is a number into one table. The numbers a session
starts with are :mod:`xrdroot.plot.colors`' - ROOT's own table to the bit;
the ones a macro makes, by ``new TColor(index, r, g, b)``, by
``TColor::GetColor(r, g, b)`` or ``GetColor("#hex")`` or by laying a
gradient with ``CreateGradientColorTable``, go in :data:`MADE`, from
:func:`free_index` up, which is where ROOT's own go too. A pad drawn later
takes them all along, as a canvas ROOT saved carries its ``ListOfColors``.

A palette is a list of those numbers, lowest first: what ``COLZ`` shades a
histogram with. ``SetPalette`` chooses one of ROOT's by number (``kBird``,
57, is the default) or gives one of its own; the gradients here are the
stops ROOT's ``TColor::SetPalette`` lays each of its palettes between.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ...plot.colors import COLORS

__all__ = [
    "PALETTE_NUMBERS",
    "TColor",
    "free_index",
    "gradient",
    "palette_indices",
    "rgb_of",
]

RGB = tuple[float, float, float]

#: The colours made in this session, by index: red, green and blue as fractions.
MADE: dict[int, RGB] = {}
#: The alpha of each colour made with one other than opaque.
ALPHA: dict[int, float] = {}
#: The ``TColor`` objects made, by index, so that ``gROOT->GetColor(n)`` finds them.
OBJECTS: dict[int, TColor] = {}
#: The palettes laid so far, by ROOT's number, so a second ``SetPalette`` reuses them.
LAID: dict[int, list[int]] = {}

#: ROOT's ``EColorPalette``: every palette a macro may ask for by name.
PALETTE_NUMBERS = {
    "kDeepSea": 51, "kGreyScale": 52, "kDarkBodyRadiator": 53, "kBlueYellow": 54,
    "kRainBow": 55, "kInvertedDarkBodyRadiator": 56, "kBird": 57, "kCubehelix": 58,
    "kGreenRedViolet": 59, "kBlueRedYellow": 60, "kOcean": 61, "kColorPrintableOnGrey": 62,
    "kAlpine": 63, "kAquamarine": 64, "kArmy": 65, "kAtlantic": 66, "kAurora": 67,
    "kAvocado": 68, "kBeach": 69, "kBlackBody": 70, "kBlueGreenYellow": 71,
    "kBrownCyan": 72, "kCMYK": 73, "kCandy": 74, "kCherry": 75, "kCoffee": 76,
    "kDarkRainBow": 77, "kDarkTerrain": 78, "kFall": 79, "kFruitPunch": 80, "kFuchsia": 81,
    "kGreyYellow": 82, "kGreenBrownTerrain": 83, "kGreenPink": 84, "kIsland": 85,
    "kLake": 86, "kLightTemperature": 87, "kLightTerrain": 88, "kMint": 89, "kNeon": 90,
    "kPastel": 91, "kPearl": 92, "kPigeon": 93, "kPlum": 94, "kRedBlue": 95, "kRose": 96,
    "kRust": 97, "kSandyTerrain": 98, "kSienna": 99, "kSolar": 100, "kSouthWest": 101,
    "kStarryNight": 102, "kSunset": 103, "kTemperatureMap": 104, "kThermometer": 105,
    "kValentine": 106, "kVisibleSpectrum": 107, "kWaterMelon": 108, "kCool": 109,
    "kCopper": 110, "kGistEarth": 111, "kViridis": 112, "kCividis": 113,
}  # fmt: skip

#: The nine stops every one of ROOT's own palettes is laid between.
STOPS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
#: ``TColor::SetPalette``'s red, green and blue at each stop, in 255ths, by number.
GRADIENTS: dict[int, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = {
    51: ((0, 9, 13, 17, 24, 32, 27, 25, 29), (0, 0, 0, 2, 37, 74, 113, 160, 221),
         (28, 42, 59, 78, 98, 129, 154, 184, 221)),
    52: ((0, 32, 64, 96, 128, 160, 192, 224, 255),) * 3,
    53: ((0, 45, 99, 156, 212, 230, 237, 234, 242), (0, 0, 0, 45, 101, 168, 238, 238, 243),
         (0, 1, 1, 3, 9, 8, 11, 95, 230)),
    54: ((0, 22, 44, 68, 93, 124, 160, 192, 237), (0, 16, 41, 67, 93, 125, 162, 194, 241),
         (97, 100, 99, 99, 93, 68, 44, 26, 74)),
    55: ((0, 5, 15, 35, 102, 196, 208, 199, 110), (0, 48, 124, 192, 206, 226, 97, 16, 0),
         (99, 142, 198, 201, 90, 22, 13, 8, 2)),
    56: ((242, 234, 237, 230, 212, 156, 99, 45, 0), (243, 238, 238, 168, 101, 45, 0, 0, 0),
         (230, 95, 11, 8, 9, 3, 1, 1, 0)),
    57: ((53, 15, 20, 6, 46, 135, 208, 253, 249), (42, 91, 128, 163, 183, 191, 186, 200, 250),
         (134, 221, 213, 201, 163, 118, 89, 50, 13)),
    58: ((0, 24, 2, 54, 176, 236, 202, 194, 255), (0, 29, 92, 129, 117, 120, 176, 236, 255),
         (0, 68, 80, 34, 57, 172, 252, 245, 255)),
    59: ((13, 23, 25, 63, 76, 104, 137, 161, 206), (95, 67, 37, 21, 0, 12, 35, 52, 79),
         (4, 3, 2, 6, 11, 22, 49, 98, 208)),
    60: ((0, 61, 89, 122, 143, 160, 185, 204, 231), (0, 0, 0, 0, 14, 37, 72, 132, 235),
         (0, 140, 224, 144, 4, 5, 6, 9, 13)),
    112: ((26, 51, 43, 33, 28, 35, 74, 144, 246), (9, 24, 55, 87, 118, 150, 180, 200, 222),
          (30, 96, 112, 114, 112, 101, 72, 35, 0)),
}  # fmt: skip
#: The palette ``SetPalette(1)`` gives: ROOT's first, fifty colours from violet to red.
PRETTY = list(range(51, 101))
#: The last colour ROOT makes when it starts - the last of ``kBird``'s, which it
#: lays at once - so that the first a macro makes is numbered as in ROOT.
BUILT_IN = 1178
#: How many colours one of ROOT's own palettes has.
PALETTE_SIZE = 255


def _hex_rgb(hexed: str) -> RGB:
    """``"#rrggbb"`` as three fractions."""
    return tuple(int(hexed[at : at + 2], 16) / 255 for at in (1, 3, 5))  # type: ignore[return-value]


def rgb_of(index: int) -> RGB:
    """The colour of ``index``: one made here, one of ROOT's own, or black."""
    if index in MADE:
        return MADE[index]
    return _hex_rgb(COLORS.get(int(index), "#000000"))


def free_index() -> int:
    """``TColor::GetFreeColorIndex``: the first index above every colour there is."""
    return max(BUILT_IN, max(MADE, default=0)) + 1


def _register(index: int, rgb: RGB, alpha: float = 1.0) -> int:
    MADE[index] = rgb
    if alpha != 1.0:
        ALPHA[index] = alpha
    return index


def gradient(
    stops: Sequence[float],
    red: Sequence[float],
    green: Sequence[float],
    blue: Sequence[float],
    count: int,
    alpha: float = 1.0,
) -> list[int]:
    """``TColor::CreateGradientColorTable``: ``count`` new colours laid between the stops.

    Between each two stops go as many colours as their share of ``count``,
    each a step of the straight line between the two, as ROOT lays them.
    """
    first = free_index()
    made: list[int] = []
    for at in range(1, len(stops)):
        between = math.floor(count * stops[at]) - math.floor(count * stops[at - 1])
        for step in range(between):
            channels = tuple(
                float(c[at - 1] + step * (c[at] - c[at - 1]) / between) for c in (red, green, blue)
            )
            made.append(_register(first + len(made), channels, alpha))  # type: ignore[arg-type]
    return made


def palette_indices(number: int) -> list[int] | None:
    """The colour indices of ROOT's palette ``number``, laid the first time it is asked for.

    ``None`` is the default, ``kBird``, which a pad shades with unless told
    otherwise; ``1`` is ROOT's first palette. A palette not among the
    gradients here is laid as ``kBird``.
    """
    if number == 1:
        return list(PRETTY)
    if number <= 0 or number == PALETTE_NUMBERS["kBird"]:
        return None
    if number not in LAID:
        red, green, blue = (
            [c / 255 for c in channel] for channel in GRADIENTS.get(number, GRADIENTS[57])
        )
        LAID[number] = gradient(STOPS, red, green, blue, PALETTE_SIZE)
    return list(LAID[number])


def _channel(value: Any) -> float:
    """A channel as a fraction, from a fraction or from an integer out of 255."""
    number = float(value)
    return number / 255 if isinstance(value, int) or number > 1 else number


def _closest(rgb: RGB) -> int | None:
    """The index of a colour already defined as ``rgb`` to 255ths, or ``None``."""
    target = tuple(round(c * 255) for c in rgb)
    for index in sorted(set(COLORS) | set(MADE)):
        opaque = ALPHA.get(index, 1.0) == 1.0  # one made see-through is not this colour
        if opaque and tuple(round(c * 255) for c in rgb_of(index)) == target:
            return index
    return None


def _get_color(*args: Any) -> int:
    """``TColor::GetColor``: the index of a colour, made if no colour is it yet.

    A fourth number is its opacity, which a colour already made must share
    to be the one found, as ROOT's search asks.
    """
    if len(args) == 1:
        text = str(args[0])
        if not text.startswith("#") or len(text) != 7:
            raise ValueError(f"a colour by name is '#rrggbb', which {text!r} is not")
        rgb = _hex_rgb(text)
    elif len(args) in (3, 4):
        rgb = (_channel(args[0]), _channel(args[1]), _channel(args[2]))
    else:
        raise TypeError(
            f"TColor::GetColor takes '#rrggbb', or red, green and blue and perhaps an opacity, "
            f"not {len(args)} arguments"
        )
    alpha = min(max(float(args[3]), 0.0), 1.0) if len(args) == 4 else 1.0
    found = _closest(rgb) if alpha == 1.0 else _closest_alpha(rgb, alpha)
    if found is not None:
        return found
    return TColor(free_index(), *rgb, a=alpha).GetNumber()


def _closest_alpha(rgb: RGB, alpha: float) -> int | None:
    """A colour made with this opacity and ``rgb`` to 255ths, or ``None``."""
    target = tuple(round(c * 255) for c in rgb)
    return next((index for index in sorted(MADE) if ALPHA.get(index) == alpha
                 and tuple(round(c * 255) for c in rgb_of(index)) == target), None)  # fmt: skip


class TColor:
    """A colour of the table, and ROOT's static functions over the whole table.

    >>> TColor.GetColor("#ff0000"), TColor.GetColor(255, 0, 0)
    (2, 2)
    """

    def __init__(
        self, index: int = -1, r: float = 0.0, g: float = 0.0, b: float = 0.0,
        name: str = "", a: float = 1.0,
    ) -> None:  # fmt: skip
        number = free_index() if index < 0 else int(index)
        self._number = number
        self._name = name or f"Color{number}"
        _register(number, (float(r), float(g), float(b)), float(a))
        OBJECTS[number] = self

    # -- this colour -----------------------------------------------------------

    def GetNumber(self) -> int:
        return self._number

    def GetName(self) -> str:
        return self._name

    def GetTitle(self) -> str:
        """``GetTitle``: ``#rrggbb``, which ROOT titles a colour with as its RGB is set."""
        return self.AsHexString()

    def GetRed(self) -> float:
        return rgb_of(self._number)[0]

    def GetGreen(self) -> float:
        return rgb_of(self._number)[1]

    def GetBlue(self) -> float:
        return rgb_of(self._number)[2]

    def GetAlpha(self) -> float:
        return ALPHA.get(self._number, 1.0)

    def SetRGB(self, r: float, g: float, b: float) -> None:
        _register(self._number, (float(r), float(g), float(b)), self.GetAlpha())

    def SetAlpha(self, a: float) -> None:
        ALPHA[self._number] = float(a)

    def AsHexString(self) -> str:
        """``#rrggbb``, each channel truncated as ROOT truncates it."""
        return "#" + "".join(f"{int(c * 255):02x}" for c in rgb_of(self._number))

    def __repr__(self) -> str:
        return f"<TColor {self._number} {self.AsHexString()}>"

    # -- the whole table --------------------------------------------------------

    @staticmethod
    def GetColor(*args: Any) -> int:
        """The index of a colour, by ``"#rrggbb"`` or red, green and blue, made if new."""
        return _get_color(*args)

    @staticmethod
    def GetFreeColorIndex() -> int:
        return free_index()

    @staticmethod
    def CreateGradientColorTable(
        number: int,
        stops: Sequence[float],
        red: Sequence[float],
        green: Sequence[float],
        blue: Sequence[float],
        ncolors: int,
        alpha: float = 1.0,
        setPalette: bool = True,
    ) -> int:
        """Lay ``ncolors`` colours between ``number`` stops; the first one's index.

        Unless told not to, the colours become the palette, as ROOT makes them.
        """
        made = gradient(stops[:number], red[:number], green[:number], blue[:number], ncolors, alpha)
        if setPalette:
            from .style import gStyle

            gStyle.SetPalette(len(made), made)
        return made[0]

    @staticmethod
    def GetColorBright(n: int) -> int:
        """A lighter version of colour ``n``, made the first time it is asked for."""
        return _get_color(*(min(1.0, c * 1.2 + 0.1) for c in rgb_of(n)))

    @staticmethod
    def GetColorDark(n: int) -> int:
        """A darker version of colour ``n``, made the first time it is asked for."""
        return _get_color(*(c * 0.7 for c in rgb_of(n)))

    @staticmethod
    def GetColorTransparent(n: int, a: float) -> int:
        """Colour ``n`` at opacity ``a``, as a colour of its own."""
        made = TColor(free_index(), *rgb_of(n), a=a)
        return made.GetNumber()

    @staticmethod
    def SetPalette(ncolors: int, colors: Any = None, alpha: float = 1.0) -> None:
        from .style import gStyle

        gStyle.SetPalette(ncolors, colors, alpha)

    @staticmethod
    def GetPalette() -> list[int]:
        from .style import gStyle

        return list(gStyle.palette())

    @staticmethod
    def GetColorPalette(i: int) -> int:
        """``GetColorPalette(i)``: the palette's colour ``i``, as ``gStyle`` has it."""
        from .style import gStyle

        return int(gStyle.GetColorPalette(i))

    @staticmethod
    def GetNumberOfColors() -> int:
        from .style import gStyle

        return int(gStyle.GetNumberOfColors())

def color_object(index: Any) -> TColor | None:
    """``gROOT->GetColor(n)``: colour ``n`` as a ``TColor`` - one made, or one of ROOT's own -
    or ``None`` for an index no colour has."""
    number = int(index)
    if number not in OBJECTS and (number in MADE or number in COLORS):
        seen = TColor.__new__(TColor)
        seen._number, seen._name = number, f"Color{number}"
        OBJECTS[number] = seen
    return OBJECTS.get(number)


