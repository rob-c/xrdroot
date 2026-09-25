"""Paves: ``TPave``, ``TPaveText``, ``TPaveLabel`` and ``TPaveStats``.

A pave is a box with a border and a shadow. Its corners are in the pad's
axes' units unless its option says ``NDC`` - ``"brNDC"`` - when they are
fractions of the pad; either way ROOT keeps both, ``fX1`` and ``fX1NDC``,
and so does this. A ``TPaveText`` stacks lines in it, each a ``TLatex``
whose size, font, colour and alignment are left at zero to take the
pave's.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .drawn import Drawn
from .shapes import TLine
from .text import TLatex

__all__ = ["TPave", "TPaveLabel", "TPaveStats", "TPaveText"]

#: A pave's corners, in the axes' units and in the pad's fractions.
PAVE_FIELDS = {
    **dict.fromkeys(("X1", "Y1", "X2", "Y2"), float),
    **{f"{name}NDC": float for name in ("X1", "Y1", "X2", "Y2")},
    "BorderSize": int, "ShadowColor": int, "CornerRadius": float, "Name": str,
}  # fmt: skip


class TPave(Drawn):
    """A box with a border and a shadow on the sides its option names."""

    classname = "TPave"
    groups: ClassVar[tuple[str, ...]] = ("line", "fill")
    fields: ClassVar[dict[str, type]] = PAVE_FIELDS

    def __init__(
        self, x1: float = 0.0, y1: float = 0.0, x2: float = 0.0, y2: float = 0.0,
        bordersize: int = 4, option: str = "br",
    ) -> None:  # fmt: skip
        option = "brNDC" if option.upper() == "NDC" else str(option)
        super().__init__(
            fX1=float(x1), fY1=float(y1), fX2=float(x2), fY2=float(y2),
            fBorderSize=int(bordersize), fOption=option, fCornerRadius=0.0,
        )  # fmt: skip
        self.members["fShadowColor"] = self.members["fLineColor"]
        ndc = "NDC" in option.upper()
        for name in ("X1", "Y1", "X2", "Y2"):
            self.members[f"f{name}NDC"] = self.members[f"f{name}"] if ndc else 0.0

    def SetOption(self, option: str = "br") -> None:
        self.members["fOption"] = str(option)

    def GetOption(self) -> str:
        return str(self.members["fOption"])

    def is_ndc(self) -> bool:
        """Whether its corners are fractions of the pad."""
        return "NDC" in self.GetOption().upper()

    def SetX1NDC(self, x1: float) -> None:
        self._corner("X1", x1)

    def SetY1NDC(self, y1: float) -> None:
        self._corner("Y1", y1)

    def SetX2NDC(self, x2: float) -> None:
        self._corner("X2", x2)

    def SetY2NDC(self, y2: float) -> None:
        self._corner("Y2", y2)

    def _corner(self, name: str, value: float) -> None:
        """A corner moved in the pad's fractions, which a pave placed so keeps as its place."""
        self.members[f"f{name}NDC"] = float(value)
        if self.is_ndc():
            self.members[f"f{name}"] = float(value)


class TPaveText(TPave):
    """A pave of lines of text, stacked from the top."""

    classname = "TPaveText"
    groups: ClassVar[tuple[str, ...]] = ("line", "fill", "text")
    defaults: ClassVar[dict[str, Any]] = {"fTextAlign": 22, "fTextSize": 0.0, "fMargin": 0.05}

    def __init__(
        self, x1: float = 0.0, y1: float = 0.0, x2: float = 0.0, y2: float = 0.0,
        option: str = "br",
    ) -> None:  # fmt: skip
        super().__init__(x1, y1, x2, y2, 4, option)
        self.members["fLines"] = []
        self.members["fLabel"] = ""

    def AddText(self, *args: Any) -> TLatex:
        """A line of text, or ``(x, y, text)`` a line placed within the pave's own fractions."""
        x, y, text = (0.0, 0.0, args[0]) if len(args) == 1 else args
        line = TLatex(x, y, str(text))
        line.members.update(fTextAlign=0, fTextColor=0, fTextFont=0, fTextSize=0.0)
        self.members["fLines"].append(line)
        return line

    def AddLine(self, x1: float = 0.0, y1: float = 0.0, x2: float = 0.0, y2: float = 0.0) -> TLine:
        line = TLine(x1, y1, x2, y2)
        self.members["fLines"].append(line)
        return line

    def GetListOfLines(self) -> list[Any]:
        return list(self.members["fLines"])

    def GetSize(self) -> int:
        return len(self.members["fLines"])

    def GetLine(self, number: int) -> Any:
        lines = self.members["fLines"]
        return lines[number] if 0 <= number < len(lines) else None

    def GetLineWith(self, text: str) -> Any:
        return next((one for one in self.members["fLines"] if text in one.GetTitle()), None)

    def DeleteText(self) -> None:
        self.members["fLines"] = []

    def Clear(self, option: str = "") -> None:
        del option
        self.DeleteText()

    def SetLabel(self, label: str) -> None:
        self.members["fLabel"] = str(label)

    def SetMargin(self, margin: float = 0.05) -> None:
        self.members["fMargin"] = float(margin)

    def SetAllWith(self, text: str, option: str, value: float) -> None:
        """Every line holding ``text`` given a size (``"size"``) or a colour (``"color"``)."""
        member = {"size": "fTextSize", "color": "fTextColor", "font": "fTextFont"}
        for line in self.members["fLines"]:
            if text in line.GetTitle() and option.lower() in member:
                line.members[member[option.lower()]] = value


class TPaveLabel(TPave):
    """A pave of one label, in the middle of it."""

    classname = "TPaveLabel"
    groups: ClassVar[tuple[str, ...]] = ("line", "fill", "text")
    defaults: ClassVar[dict[str, Any]] = {"fTextAlign": 22, "fTextSize": 0.99}

    def __init__(
        self, x1: float = 0.0, y1: float = 0.0, x2: float = 0.0, y2: float = 0.0,
        label: str = "", option: str = "br",
    ) -> None:  # fmt: skip
        super().__init__(x1, y1, x2, y2, 3, option)
        self.members["fLabel"] = str(label)

    def SetLabel(self, label: str) -> None:
        self.members["fLabel"] = str(label)

    def GetLabel(self) -> str:
        return str(self.members["fLabel"])

    def DrawPaveLabel(
        self, x1: float, y1: float, x2: float, y2: float, label: str, option: str = ""
    ) -> Any:
        made = TPaveLabel(x1, y1, x2, y2, label, option or self.GetOption())
        for name, value in self.members.items():
            if name.startswith(("fText", "fFill", "fLine", "fBorder")):
                made.members[name] = value
        made.Draw()
        return made


class TPaveStats(TPaveText):
    """A histogram's stats box: what ``fOptStat`` and ``fOptFit`` ask to be said of it."""

    classname = "TPaveStats"
    fields: ClassVar[dict[str, type]] = {
        **PAVE_FIELDS, "OptStat": int, "OptFit": int, "StatFormat": str, "FitFormat": str,
    }  # fmt: skip

    def __init__(
        self, x1: float = 0.0, y1: float = 0.0, x2: float = 0.0, y2: float = 0.0,
        option: str = "brNDC",
    ) -> None:  # fmt: skip
        super().__init__(x1, y1, x2, y2, option)
        self.members.update(fName="stats", fOptStat=-1, fOptFit=-1)
        self.members.update(fStatFormat="6.4g", fFitFormat="5.4g", fTextAlign=12)
        #: What it describes.
        self.parent: Any = None

    def SetOptStat(self, stat: int = 1) -> None:
        self.members["fOptStat"] = int(stat)

    def SetOptFit(self, fit: int = 1) -> None:
        self.members["fOptFit"] = int(fit)

    def SetParent(self, parent: Any) -> None:
        self.parent = parent

    def GetParent(self) -> Any:
        return self.parent
