"""``TLegend``: a pave of entries, each a symbol drawn like what it stands for and a label.

An entry's option says what its symbol is: ``l`` a line, ``p`` a marker,
``f`` a filled box, ``e`` an error bar, ``h`` a header across the row.
The symbol is drawn in the style of the object the entry stands for as it
is when the pad is drawn, so styling a histogram after adding it restyles
its entry, as in ROOT.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .drawn import Drawn
from .paves import PAVE_FIELDS, TPave
from .style import gStyle

__all__ = ["TLegend", "TLegendEntry"]

#: Where a legend made with no corners goes, as ``TLegend()`` places one: its size.
DEFAULT_SIZE = (0.3, 0.15)


class TLegendEntry(Drawn):
    """One entry of a legend: what it stands for, its label and its option."""

    classname = "TLegendEntry"
    groups = ("text", "line", "fill", "marker")
    defaults: ClassVar[dict[str, Any]] = {"fTextAlign": 0, "fTextColor": 0, "fTextFont": 0,
                                          "fTextSize": 0.0}  # fmt: skip

    def __init__(self, obj: Any = None, label: str = "", option: str = "") -> None:
        super().__init__(fObject=obj, fLabel=str(label), fOption=str(option))

    def GetObject(self) -> Any:
        return self.members["fObject"]

    def SetObject(self, obj: Any) -> None:
        self.members["fObject"] = obj

    def GetLabel(self) -> str:
        return str(self.members["fLabel"])

    def SetLabel(self, label: str = "") -> None:
        self.members["fLabel"] = str(label)

    def GetOption(self) -> str:
        return str(self.members["fOption"])

    def SetOption(self, option: str = "lpf") -> None:
        self.members["fOption"] = str(option)


def _title_of(obj: Any) -> str:
    """What an entry is labelled when it is not told: the object's title."""
    getter = getattr(obj, "GetTitle", None)
    if callable(getter):
        return str(getter())
    return str(getattr(getattr(obj, "_xrd", obj), "title", "") or "")


class TLegend(TPave):
    """A legend: rows and columns of entries in a pave, placed in NDC."""

    classname = "TLegend"
    groups = ("line", "fill", "text")
    fields: ClassVar[dict[str, type]] = {
        **PAVE_FIELDS, "Margin": float, "NColumns": int, "EntrySeparation": float,
        "ColumnSeparation": float,
    }  # fmt: skip

    def __init__(self, *args: Any) -> None:
        corners, rest = _placed(args)
        header, option = (list(rest) + ["", "brNDC"][len(rest) :])[:2]
        super().__init__(*corners, gStyle.GetLegendBorderSize(), str(option))
        self.members.update(
            fFillColor=gStyle.GetLegendFillColor(), fFillStyle=1001,
            fTextFont=gStyle.GetLegendFont(), fTextSize=gStyle.GetLegendTextSize(),
            fTextAlign=12, fMargin=0.25, fNColumns=1, fEntrySeparation=0.1,
            fColumnSeparation=0.0, fPrimitives=[],
        )  # fmt: skip
        if header:
            self.SetHeader(str(header))

    def AddEntry(self, obj: Any, label: str = "", option: str = "lpf") -> TLegendEntry:
        """An entry for ``obj`` - or the object of that name in the current pad - labelled."""
        if isinstance(obj, str):
            from .pads import find_anywhere

            obj = find_anywhere(obj)
        entry = TLegendEntry(obj, label or (_title_of(obj) if obj is not None else ""), option)
        self.members["fPrimitives"].append(entry)
        return entry

    def SetHeader(self, header: str = "", option: str = "") -> None:
        """A header across the top row; ``"C"`` centres it."""
        entries = self.members["fPrimitives"]
        if entries and "h" in entries[0].GetOption():
            entries.pop(0)
        entry = TLegendEntry(None, str(header), "h")
        if "C" in option.upper():
            entry.members["fTextAlign"] = 22
        entries.insert(0, entry)

    def GetHeader(self) -> str:
        entries = self.members["fPrimitives"]
        return entries[0].GetLabel() if entries and "h" in entries[0].GetOption() else ""

    def GetListOfPrimitives(self) -> list[TLegendEntry]:
        return list(self.members["fPrimitives"])

    def GetNRows(self) -> int:
        columns = max(int(self.members["fNColumns"]), 1)
        return -(-len(self.members["fPrimitives"]) // columns)

    def Clear(self, option: str = "") -> None:
        del option
        self.members["fPrimitives"] = []

    def DeleteEntry(self) -> None:
        if self.members["fPrimitives"]:
            self.members["fPrimitives"].pop()


def _placed(args: tuple[Any, ...]) -> tuple[tuple[float, ...], tuple[Any, ...]]:
    """The corners of a legend and what follows them, from however it was made.

    ``TLegend(x1, y1, x2, y2, ...)`` is placed where it says; ``TLegend(w, h,
    ...)`` and ``TLegend()`` go in the top right of the frame, that size.
    """
    numbers = [a for a in args[:4] if isinstance(a, (int, float)) and not isinstance(a, bool)]
    if len(numbers) >= 4:
        return tuple(float(a) for a in numbers[:4]), args[4:]
    width, height = (float(numbers[0]), float(numbers[1])) if len(numbers) >= 2 else DEFAULT_SIZE
    right = 1.0 - gStyle.GetPadRightMargin() - 0.02
    top = 1.0 - gStyle.GetPadTopMargin() - 0.02
    return (right - width, top - height, right, top), args[len(numbers) :]
