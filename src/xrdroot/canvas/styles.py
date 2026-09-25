"""ROOT's attribute numbers - lines, markers, fills, fonts - as matplotlib's.

Like colours, every attribute a ROOT object keeps is a small number with a
meaning fixed by ``TAttLine``, ``TAttMarker``, ``TAttFill`` and
``TAttText``: line style 2 is dashed, marker 20 a filled circle, fill 3004
a hatch, font 42 Helvetica sized as a fraction of the pad. These are those
meanings, translated into the terms matplotlib draws in, with a canvas
drawn at :data:`DPI` so that one of ROOT's pixels is one of the figure's.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DPI",
    "align",
    "dashes",
    "fill",
    "font",
    "marker",
    "marker_size",
    "points",
]

#: The resolution a canvas is drawn at: one ROOT pixel is one figure pixel.
DPI = 100
#: Points in a pixel at that resolution.
POINTS_PER_PIXEL = 72.0 / DPI

#: ``TStyle``'s line styles 1 to 10, as lengths of dash and gap in pixels.
LINE_STYLES: dict[int, tuple[float, ...]] = {
    2: (12, 12),
    3: (4, 8),
    4: (12, 16, 4, 16),
    5: (20, 12, 4, 12),
    6: (20, 12, 4, 12, 4, 12, 4, 12),
    7: (20, 20),
    8: (20, 12, 4, 12, 4, 12),
    9: (80, 20),
    10: (80, 40, 4, 40),
}
#: How much shorter the dashes are drawn than ROOT's pixel lengths: ROOT's
#: are drawn at half size on a screen of its own resolution.
DASH_SCALE = 0.5

#: ``TAttMarker``'s styles, as matplotlib's markers and whether they are filled.
MARKERS: dict[int, tuple[str, bool]] = {
    1: (".", True),
    2: ("+", True),
    3: ("*", True),
    4: ("o", False),
    5: ("x", True),
    6: (".", True),
    7: (".", True),
    8: ("o", True),
    20: ("o", True),
    21: ("s", True),
    22: ("^", True),
    23: ("v", True),
    24: ("o", False),
    25: ("s", False),
    26: ("^", False),
    27: ("D", False),
    28: ("P", False),
    29: ("*", True),
    30: ("*", False),
    31: ("*", True),
    32: ("v", False),
    33: ("D", True),
    34: ("P", True),
}
#: The markers that are a dot whatever the size they are asked for.
DOTS = {1: 1.0, 6: 2.0, 7: 3.0}
#: How wide a marker of size 1 is, in pixels.
MARKER_PIXELS = 8.0

#: The fill styles that hatch, as matplotlib's hatches.
HATCHES: dict[int, str] = {
    3001: "..",
    3002: ".",
    3003: "...",
    3004: "//",
    3005: "\\\\",
    3006: "||",
    3007: "--",
    3010: "++",
    3013: "xx",
    3016: "//",
    3017: "\\\\",
    3021: "++",
    3144: "\\\\",
    3244: "//",
    3354: "\\\\",
    3344: "//",
}

#: ``TAttText``'s font families: the face each family code is, by its ten.
FAMILIES: dict[int, tuple[str, str, str]] = {
    1: ("serif", "italic", "normal"),
    2: ("serif", "normal", "bold"),
    3: ("serif", "italic", "bold"),
    4: ("sans-serif", "normal", "normal"),
    5: ("sans-serif", "italic", "normal"),
    6: ("sans-serif", "normal", "bold"),
    7: ("sans-serif", "italic", "bold"),
    8: ("monospace", "normal", "normal"),
    9: ("monospace", "italic", "normal"),
    10: ("monospace", "normal", "bold"),
    11: ("monospace", "italic", "bold"),
    12: ("serif", "normal", "normal"),
    13: ("serif", "normal", "normal"),
    14: ("serif", "normal", "normal"),
    15: ("serif", "italic", "normal"),
}
#: The face a font code nobody defined is drawn in: ROOT's Helvetica, 42.
DEFAULT_FAMILY = FAMILIES[4]
#: The mathtext face each family of font is drawn with.
MATH = {"serif": "stix", "sans-serif": "dejavusans", "monospace": "dejavusans"}

#: ``TAttText``'s alignment, by its tens and by its units.
HORIZONTAL = {1: "left", 2: "center", 3: "right"}
VERTICAL = {1: "bottom", 2: "center", 3: "top"}


def points(pixels: float) -> float:
    """``pixels`` of ROOT's in the points matplotlib measures lines and text in."""
    return float(pixels) * POINTS_PER_PIXEL


def dashes(style: Any, width: float = 1.0) -> Any:
    """A matplotlib line style for ROOT's ``fLineStyle``, solid for one unknown."""
    lengths = LINE_STYLES.get(int(style))
    if lengths is None:
        return "solid"
    scale = DASH_SCALE / max(float(width), 1.0)
    return (0, tuple(length * scale for length in lengths))


def marker(style: Any) -> tuple[str, bool]:
    """The matplotlib marker for ``fMarkerStyle``, and whether it is filled."""
    return MARKERS.get(int(style), ("o", True))


def marker_size(style: Any, size: Any) -> float:
    """How big a marker is, in points: fixed for a dot, else by ``fMarkerSize``."""
    dot = DOTS.get(int(style))
    if dot is not None:
        return points(dot)
    return points(MARKER_PIXELS * float(size))


def fill(style: Any) -> tuple[bool, str | None, float]:
    """Whether ``fFillStyle`` fills at all, the hatch it fills with, and how opaquely.

    0 is hollow, 1001 solid, the 3000s hatched and the 4000s a solid fill
    whose last two digits are its opacity, 4000 clear and 4100 opaque.
    """
    code = int(style)
    if code == 0 or code == 4000:
        return False, None, 0.0
    if 4000 < code <= 4100:
        return True, None, (code - 4000) / 100.0
    if code in HATCHES or 3000 <= code < 4000:
        return True, HATCHES.get(code, "//"), 1.0
    return True, None, 1.0


def font(code: Any) -> tuple[str, str, str, bool]:
    """The family, style and weight of ``fTextFont``, and whether its size is in pixels.

    The code is ten times the family and then a precision, and precision 3
    alone means the size is pixels; every other is a fraction of the pad.
    """
    number = int(code)
    family, style, weight = FAMILIES.get(number // 10, DEFAULT_FAMILY)
    return family, style, weight, number % 10 == 3


def align(code: Any) -> tuple[str, str]:
    """The horizontal and vertical alignment of ``fTextAlign``: 11 is bottom left."""
    number = int(code)
    return HORIZONTAL.get(number // 10, "left"), VERTICAL.get(number % 10, "bottom")
