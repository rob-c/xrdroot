"""ROOT's colours: the numbered table every object's attributes point into.

A histogram does not say it is red. It says its ``fLineColor`` is ``2``, or
``633`` - ``kRed+1`` - and leaves the rest to the ``TColor`` table ROOT builds
when it starts: the fifty-odd colours of its first release, the pretty
spectrum after them, the Petroff sets, and the colour wheel of ``kRed``,
``kAzure`` and the rest, each with its lighter and darker neighbours. This is
that table, generated from ``TColor::InitializeColors`` and
``TColor::CreateColorWheel`` to the same rounding ROOT's ``AsHexString``
does, so a colour chosen in a ROOT macro comes out here as the same colour.

The palettes 2-D pictures are shaded with are made the way
``TColor::CreateGradientColorTable`` makes them: 255 colours laid between
nine stops. ``kBird`` has been ROOT's default since 6.04; ``kViridis`` is
matplotlib's, as ROOT has it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

__all__ = [
    "COLORS",
    "ECOLOR",
    "PALETTES",
    "PETROFF",
    "UNKNOWN",
    "color",
    "palette",
    "palette_color",
]

#: What ``TColor::InitializeColors`` and ``CreateColorWheel`` make, by index,
#: as ``AsHexString`` writes them: truncated, not rounded, from floats.
COLORS = {
    0: "#ffffff", 1: "#000000", 2: "#ff0000", 3: "#00ff00", 4: "#0000ff", 5: "#ffff00",
    6: "#ff00ff", 7: "#00ffff", 8: "#59d354", 9: "#5954d8", 10: "#fefefe", 11: "#c0b6ac",
    12: "#4c4c4c", 13: "#666666", 14: "#7f7f7f", 15: "#999999", 16: "#b2b2b2", 17: "#cccccc",
    18: "#e5e5e5", 19: "#f2f2f2", 20: "#ccc6aa", 21: "#ccc6aa", 22: "#c1bfa8", 23: "#bab5a3",
    24: "#b2a596", 25: "#b7a39b", 26: "#ad998c", 27: "#9b8e82", 28: "#876656", 29: "#afcec6",
    30: "#84c1a3", 31: "#89a8a0", 32: "#829e8c", 33: "#adbcc6", 34: "#7a8e99", 35: "#758991",
    36: "#688296", 37: "#6d7a84", 38: "#7c99d1", 39: "#7f7f9b", 40: "#aaa5bf", 41: "#d3ce87",
    42: "#ddba87", 43: "#bc9e82", 44: "#c6997c", 45: "#bf8277", 46: "#ce5e60", 47: "#aa8e93",
    48: "#a5777a", 49: "#936870", 50: "#d35954", 51: "#9200ff", 52: "#7a00ff", 53: "#6200ff",
    54: "#4a00ff", 55: "#3300ff", 56: "#1b00ff", 57: "#0300ff", 58: "#0014ff", 59: "#002cff",
    60: "#0044ff", 61: "#005bff", 62: "#0073ff", 63: "#008bff", 64: "#00a3ff", 65: "#00bbff",
    66: "#00d2ff", 67: "#00eaff", 68: "#00fffb", 69: "#00ffe3", 70: "#00ffcc", 71: "#00ffb4",
    72: "#00ff9c", 73: "#00ff84", 74: "#00ff6c", 75: "#00ff55", 76: "#00ff3d", 77: "#00ff25",
    78: "#00ff0d", 79: "#0aff00", 80: "#22ff00", 81: "#39ff00", 82: "#51ff00", 83: "#69ff00",
    84: "#81ff00", 85: "#99ff00", 86: "#b0ff00", 87: "#c8ff00", 88: "#e0ff00", 89: "#f8ff00",
    90: "#ffee00", 91: "#ffd600", 92: "#ffbe00", 93: "#ffa600", 94: "#ff8e00", 95: "#ff7700",
    96: "#ff5f00", 97: "#ff4700", 98: "#ff2f00", 99: "#ff1700", 100: "#6f2da8", 101: "#a52a2a",
    102: "#b2beb5", 103: "#5790fc", 104: "#f89c20", 105: "#e42536", 106: "#964a8b", 107: "#9c9ca1",
    108: "#7a21dd", 109: "#1845fb", 110: "#ff5e02", 111: "#c91f16", 112: "#c849a9", 113: "#adad7d",
    114: "#86c8dd", 115: "#578dff", 116: "#656364", 117: "#3f90da", 118: "#ffa90e", 119: "#bd1f01",
    120: "#94a4a2", 121: "#832db6", 122: "#a96b59", 123: "#e76300", 124: "#b9ac70", 125: "#717581",
    126: "#92dadd", 390: "#ffffcc", 391: "#ffff99", 392: "#cccc99", 393: "#ffff66", 394: "#cccc66",
    395: "#999966", 396: "#ffff33", 397: "#cccc33", 398: "#999933", 399: "#666633", 400: "#ffff00",
    401: "#cccc00", 402: "#999900", 403: "#666600", 404: "#333300", 406: "#ccffcc", 407: "#99ff99",
    408: "#99cc99", 409: "#66ff66", 410: "#66cc66", 411: "#669966", 412: "#33ff33", 413: "#33cc33",
    414: "#339933", 415: "#336633", 416: "#00ff00", 417: "#00cc00", 418: "#009900", 419: "#006600",
    420: "#003300", 422: "#ccffff", 423: "#99ffff", 424: "#99cccc", 425: "#66ffff", 426: "#66cccc",
    427: "#669999", 428: "#33ffff", 429: "#33cccc", 430: "#339999", 431: "#336666", 432: "#00ffff",
    433: "#00cccc", 434: "#009999", 435: "#006666", 436: "#003333", 590: "#ccccff", 591: "#9999ff",
    592: "#9999cc", 593: "#6666ff", 594: "#6666cc", 595: "#666699", 596: "#3333ff", 597: "#3333cc",
    598: "#333399", 599: "#333366", 600: "#0000ff", 601: "#0000cc", 602: "#000099", 603: "#000066",
    604: "#000033", 606: "#ffccff", 607: "#ff99ff", 608: "#cc99cc", 609: "#ff66ff", 610: "#cc66cc",
    611: "#996699", 612: "#ff33ff", 613: "#cc33cc", 614: "#993399", 615: "#663366", 616: "#ff00ff",
    617: "#cc00cc", 618: "#990099", 619: "#660066", 620: "#330033", 622: "#ffcccc", 623: "#ff9999",
    624: "#cc9999", 625: "#ff6666", 626: "#cc6666", 627: "#996666", 628: "#ff3333", 629: "#cc3333",
    630: "#993333", 631: "#663333", 632: "#ff0000", 633: "#cc0000", 634: "#990000", 635: "#660000",
    636: "#330000", 791: "#ffcc99", 792: "#cc9966", 793: "#996633", 794: "#996600", 795: "#cc9933",
    796: "#ffcc66", 797: "#ff9900", 798: "#ffcc33", 799: "#cc9900", 800: "#ffcc00", 801: "#ff9933",
    802: "#cc6600", 803: "#663300", 804: "#993300", 805: "#cc6633", 806: "#ff9966", 807: "#ff6600",
    808: "#ff6633", 809: "#cc3300", 810: "#ff3300", 811: "#99ff33", 812: "#66cc00", 813: "#336600",
    814: "#339900", 815: "#66cc33", 816: "#99ff66", 817: "#66ff00", 818: "#66ff33", 819: "#33cc00",
    820: "#33ff00", 821: "#ccff99", 822: "#99cc66", 823: "#669933", 824: "#669900", 825: "#99cc33",
    826: "#ccff66", 827: "#99ff00", 828: "#ccff33", 829: "#99cc00", 830: "#ccff00", 831: "#99ffcc",
    832: "#66cc99", 833: "#339966", 834: "#009966", 835: "#33cc99", 836: "#66ffcc", 837: "#00ff66",
    838: "#33ffcc", 839: "#00cc99", 840: "#00ffcc", 841: "#33ff99", 842: "#00cc66", 843: "#006633",
    844: "#009933", 845: "#33cc66", 846: "#66ff99", 847: "#00ff99", 848: "#33ff66", 849: "#00cc33",
    850: "#00ff33", 851: "#99ccff", 852: "#6699cc", 853: "#336699", 854: "#003399", 855: "#3366cc",
    856: "#6699ff", 857: "#0066ff", 858: "#3366ff", 859: "#0033cc", 860: "#0033ff", 861: "#3399ff",
    862: "#0066cc", 863: "#003366", 864: "#006699", 865: "#3399cc", 866: "#66ccff", 867: "#0099ff",
    868: "#33ccff", 869: "#0099cc", 870: "#00ccff", 871: "#cc99ff", 872: "#9966cc", 873: "#663399",
    874: "#660099", 875: "#9933cc", 876: "#cc66ff", 877: "#9900ff", 878: "#cc33ff", 879: "#9900cc",
    880: "#cc00ff", 881: "#9933ff", 882: "#6600cc", 883: "#330066", 884: "#330099", 885: "#6633cc",
    886: "#9966ff", 887: "#6600ff", 888: "#6633ff", 889: "#3300cc", 890: "#3300ff", 891: "#ff3399",
    892: "#cc0066", 893: "#660033", 894: "#990033", 895: "#cc3366", 896: "#ff6699", 897: "#ff0066",
    898: "#ff3366", 899: "#cc0033", 900: "#ff0033", 901: "#ff99cc", 902: "#cc6699", 903: "#993366",
    904: "#990066", 905: "#cc3399", 906: "#ff66cc", 907: "#ff0099", 908: "#cc0099", 909: "#ff33cc",
    910: "#ff0099", 920: "#cccccc", 921: "#999999", 922: "#666666", 923: "#333333",
}  # fmt: skip

#: ROOT's ``EColor``, the names a macro writes colours by, ``kRed+1`` and all.
ECOLOR = {
    "kWhite": 0, "kBlack": 1, "kGray": 920, "kRed": 632, "kGreen": 416, "kBlue": 600,
    "kYellow": 400, "kMagenta": 616, "kCyan": 432, "kOrange": 800, "kSpring": 820,
    "kTeal": 840, "kAzure": 860, "kViolet": 880, "kPink": 900, "kGrape": 100, "kBrown": 101,
    "kAsh": 102, "kP6Blue": 103, "kP6Yellow": 104, "kP6Red": 105, "kP6Grape": 106,
    "kP6Gray": 107, "kP6Violet": 108, "kP8Blue": 109, "kP8Orange": 110, "kP8Red": 111,
    "kP8Pink": 112, "kP8Green": 113, "kP8Cyan": 114, "kP8Azure": 115, "kP8Gray": 116,
    "kP10Blue": 117, "kP10Yellow": 118, "kP10Red": 119, "kP10Gray": 120, "kP10Violet": 121,
    "kP10Brown": 122, "kP10Orange": 123, "kP10Green": 124, "kP10Ash": 125, "kP10Cyan": 126,
}  # fmt: skip

#: The ten Petroff colours ROOT recommends for telling many things apart,
#: in the order ``kP10Blue`` to ``kP10Cyan``: what several things drawn
#: together are coloured by when none of them says otherwise.
PETROFF = tuple(COLORS[index] for index in range(117, 127))

#: What an index outside the table - a colour a macro made for itself, which
#: is not written in the file - is drawn as: ROOT's own foreground, black.
UNKNOWN = "#000000"

#: ``kRed+1``, ``kAzure - 3``, ``kP10Blue``: an ``EColor`` name and an offset.
_NAMED = re.compile(r"\s*(k[A-Za-z0-9]+)\s*(?:([+-])\s*(\d+))?\s*")

#: The stops every one of ROOT's 255-colour palettes is laid between.
_STOPS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)

#: ``TColor::SetPalette``'s red, green and blue at each stop, by palette.
_GRADIENTS = {
    "bird": (
        (0.2082, 0.0592, 0.0780, 0.0232, 0.1802, 0.5301, 0.8186, 0.9956, 0.9764),
        (0.1664, 0.3599, 0.5041, 0.6419, 0.7178, 0.7492, 0.7328, 0.7862, 0.9832),
        (0.5293, 0.8684, 0.8385, 0.7914, 0.6425, 0.4662, 0.3499, 0.1968, 0.0539),
    ),
    "viridis": (
        tuple(v / 255 for v in (26, 51, 43, 33, 28, 35, 74, 144, 246)),
        tuple(v / 255 for v in (9, 24, 55, 87, 118, 150, 180, 200, 222)),
        tuple(v / 255 for v in (30, 96, 112, 114, 112, 101, 72, 35, 0)),
    ),
}

#: ROOT's palette numbers, for the ones a caller may know by number.
_PALETTE_NUMBERS = {57: "bird", 112: "viridis"}


def _hex(red: float, green: float, blue: float) -> str:
    """``AsHexString``: each channel times 255, truncated as ``Int_t`` truncates."""
    return "#" + "".join(f"{int(channel * 255):02x}" for channel in (red, green, blue))


def _between(low: float, high: float, step: int, count: int) -> float:
    return low + step * (high - low) / count


def _gradient(red: Sequence[float], green: Sequence[float], blue: Sequence[float]) -> list[str]:
    """``TColor::CreateGradientColorTable`` for 255 colours over ``_STOPS``."""
    made: list[str] = []
    for at in range(1, len(_STOPS)):
        count = math.floor(255 * _STOPS[at]) - math.floor(255 * _STOPS[at - 1])
        for step in range(count):
            made.append(
                _hex(
                    _between(red[at - 1], red[at], step, count),
                    _between(green[at - 1], green[at], step, count),
                    _between(blue[at - 1], blue[at], step, count),
                )
            )
    return made


#: The palettes by name, 255 colours each, lowest first.
PALETTES = {name: tuple(_gradient(*channels)) for name, channels in _GRADIENTS.items()}


def palette(name: Any = "bird") -> tuple[str, ...] | None:
    """One of ROOT's palettes by name or number, or ``None`` for one it does not have.

        >>> palette("bird")[0], palette(57)[-1]
        ('#352a86', '#f9f90e')

    ``None`` means the name is left for the drawing library to look up as
    one of its own colour maps - ``"magma"`` is matplotlib's and plotly's.
    """
    key = _PALETTE_NUMBERS.get(name, name) if isinstance(name, int) else str(name).lower()
    return PALETTES.get(str(key).removeprefix("k").lower())


def palette_color(index: int, count: int, name: Any = "bird") -> str:
    """``TPad::NextPaletteColor``: the ``index``-th of ``count`` spread across a palette.

    What ROOT's ``PLC``, ``PMC`` and ``PFC`` options colour each of several
    things drawn together by: evenly from the first colour to the last.
    """
    colors = palette(name) or PALETTES["bird"]
    if count <= 1:
        return colors[0]
    return colors[round(index * (len(colors) - 1) / (count - 1))]


def _named(text: str) -> str | None:
    """``"kRed+1"`` as its colour, or ``None`` for text that is not an ``EColor``."""
    found = _NAMED.fullmatch(text)
    if found is None or found.group(1) not in ECOLOR:
        return None
    offset = int(found.group(3) or 0) * (-1 if found.group(2) == "-" else 1)
    return COLORS.get(ECOLOR[found.group(1)] + offset, UNKNOWN)


def color(value: Any) -> str | None:
    """Whatever names a colour, as the colour to draw with.

        >>> color(2), color("kAzure+1"), color("crimson"), color(None)
        ('#ff0000', '#3399ff', 'crimson', None)

    A number is a ROOT colour index; ``kRed+1`` and the rest of ``EColor``
    are what a ROOT macro writes; anything else - ``"crimson"``,
    ``"#1f77b4"`` - is left for the drawing library, which knows its own.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return COLORS.get(int(value), UNKNOWN)
    text = str(value)
    return _named(text) or text
