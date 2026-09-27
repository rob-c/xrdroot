"""ROOT's named constants where RooFit takes a string for a number: ``"kRed+1"``, ``"kDashed"``.

``LineColor("kAzure+2")`` asks ROOT's interpreter for the value, and this is
the part of that interpreter a colour or a style needs: a name from ROOT's
enumerations, plus or minus a number.
"""

from __future__ import annotations

import re

__all__ = ["CONSTANTS", "named_constant"]

#: The names a string may hold: ``EColor``, ``ELineStyle``, the marker styles.
CONSTANTS = {
    "kWhite": 0,
    "kBlack": 1,
    "kGray": 920,
    "kRed": 632,
    "kGreen": 416,
    "kBlue": 600,
    "kYellow": 400,
    "kMagenta": 616,
    "kCyan": 432,
    "kOrange": 800,
    "kSpring": 820,
    "kTeal": 840,
    "kAzure": 860,
    "kViolet": 880,
    "kPink": 900,
    "kSolid": 1,
    "kDashed": 2,
    "kDotted": 3,
    "kDashDotted": 4,
    "kDot": 1,
    "kPlus": 2,
    "kStar": 3,
    "kCircle": 4,
    "kMultiply": 5,
    "kFullDotSmall": 6,
    "kFullDotMedium": 7,
    "kFullDotLarge": 8,
    "kFullCircle": 20,
    "kFullSquare": 21,
    "kFullTriangleUp": 22,
    "kFullTriangleDown": 23,
    "kOpenCircle": 24,
    "kOpenSquare": 25,
    "kOpenTriangleUp": 26,
    "kOpenDiamond": 27,
    "kOpenCross": 28,
    "kFullStar": 29,
    "kOpenStar": 30,
}
#: ``kRed+1``, ``kAzure - 9``, ``42``.
EXPRESSION = re.compile(r"^\s*(?:(k\w+)|(-?\d+))\s*(?:([+-])\s*(\d+))?\s*$")


def named_constant(text: str) -> int:
    """The value of ``text``: a constant, a number, or either plus or minus a number."""
    found = EXPRESSION.match(text)
    if found is None or (found.group(1) and found.group(1) not in CONSTANTS):
        raise ValueError(f'"{text}" is not a valid color name or style that RooFit can read')
    base = CONSTANTS[found.group(1)] if found.group(1) else int(found.group(2))
    offset = int(found.group(4) or 0)
    return base + offset if found.group(3) != "-" else base - offset
