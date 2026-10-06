"""The attribute groups primitives are drawn with: lines, fills, borders, text, fonts, markers.

Each is ROOT 7's ``RAttr...``, with the members and the defaults ROOT
gives them, and the enumerations a macro names them by:
``RAttrLine::kDashed``, ``RAttrFill::k3001``, ``RAttrText::kLeftTop``,
``RAttrFont::kArialOblique``. A font is set from one of those numbers as
the family, style and weight it stands for.
"""

from __future__ import annotations

from typing import Any

from .attrs import Attrs, RColor, frozen
from .lengths import RPadLength, length

__all__ = ["RAttrLine", "RAttrFill", "RAttrBorder", "RAttrFont", "RAttrText", "RAttrMarker",
           "RAttrMargins"]  # fmt: skip


def _black() -> RColor:
    return RColor("black")


class RAttrLine(Attrs):
    """``RAttrLine``: a line's colour, width, style and dash pattern."""

    MEMBERS = frozen({"color": _black, "width": lambda: 1.0, "style": lambda: 1,
                      "pattern": lambda: ""})  # fmt: skip
    kNone, kSolid, kDashed, kDotted, kDashDotted = 0, 1, 2, 3, 4

    def __init__(self, color: Any = None, width: Any = 1.0, style: Any = 1) -> None:
        super().__init__()
        self.color = _black() if color is None else color
        self.width, self.style = float(width), int(style)


for _style in range(1, 11):
    setattr(RAttrLine, f"kStyle{_style}", _style)


class RAttrFill(Attrs):
    """``RAttrFill``: a fill's colour and style - hollow, solid, or one of the hatchings."""

    MEMBERS = frozen({"color": _black, "style": lambda: 0})
    kHollow, kNone, kSolid = 0, 0, 1001


for _hatch in range(3001, 3026):
    setattr(RAttrFill, f"k{_hatch}", _hatch)


class RAttrBorder(RAttrLine):
    """``RAttrBorder``: a box's outline, and how round its corners are."""

    MEMBERS = frozen({"rx": lambda: 0, "ry": lambda: 0})


#: Each of ``RAttrFont``'s fonts, as the family, style and weight it is.
FONTS = {
    1: ("Times New Roman", "", ""), 2: ("Times New Roman", "italic", ""),
    3: ("Times New Roman", "", "bold"), 4: ("Times New Roman", "italic", "bold"),
    5: ("Arial", "", ""), 6: ("Arial", "oblique", ""), 7: ("Arial", "", "bold"),
    8: ("Arial", "oblique", "bold"), 9: ("Courier New", "", ""),
    10: ("Courier New", "oblique", ""), 11: ("Courier New", "", "bold"),
    12: ("Courier New", "oblique", "bold"), 14: ("Verdana", "", ""),
    15: ("Verdana", "italic", ""), 16: ("Verdana", "", "bold"),
    17: ("Verdana", "italic", "bold"),
}  # fmt: skip


class RAttrFont(Attrs):
    """``RAttrFont``: a font's family, style and weight - set whole from ROOT's numbered ones."""

    MEMBERS = frozen({"family": lambda: "", "style": lambda: "", "weight": lambda: ""})
    kTimes, kTimesItalic, kTimesBold, kTimesBoldItalic = 1, 2, 3, 4
    kArial, kArialOblique, kArialBold, kArialBoldOblique = 5, 6, 7, 8
    kCourier, kCourierOblique, kCourierBold, kCourierBoldOblique = 9, 10, 11, 12
    kVerdana, kVerdanaItalic, kVerdanaBold, kVerdanaBoldItalic = 14, 15, 16, 17

    def _assign(self, value: Any) -> None:
        found = FONTS.get(int(value))
        if found is None:
            raise ValueError(f"RAttrFont has no font {value}; its fonts are numbered "
                             f"{', '.join(map(str, FONTS))}")  # fmt: skip
        self.family, self.style, self.weight = found

    def GetFullName(self) -> str:
        """The family, then the style and the weight it is drawn in, as ROOT names a font."""
        return " ".join(part for part in (self.family, self.style, self.weight) if part)


class RAttrText(Attrs):
    """``RAttrText``: text's size, angle, alignment, colour and font."""

    MEMBERS = frozen({"size": lambda: 12.0, "angle": lambda: 0.0, "align": lambda: 22,
                      "color": _black, "font": RAttrFont})  # fmt: skip
    kLeftBottom, kLeftCenter, kLeftTop = 11, 12, 13
    kCenterBottom, kCenter, kCenterTop = 21, 22, 23
    kRightBottom, kRightCenter, kRightTop = 31, 32, 33


class RAttrMarker(Attrs):
    """``RAttrMarker``: a marker's colour, size and style."""

    MEMBERS = frozen({"color": _black, "size": lambda: 0.01, "style": lambda: 1})


class RAttrMargins(Attrs):
    """``RAttrMargins``: the space left at each side - all four set at once by one length."""

    MEMBERS = frozen(dict.fromkeys(("left", "right", "top", "bottom"), RPadLength))

    def _assign(self, value: Any) -> None:
        for side in ("left", "right", "top", "bottom"):
            setattr(self, side, length(value))
