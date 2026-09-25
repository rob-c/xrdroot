"""``TPad``: a rectangle of a canvas, what is drawn in it, and ``gPad``.

A pad is kept as a saved one is read: its members by ROOT's names - where
it sits (``fXlowNDC``...), its margins, ``fLogy``, ``fGridx``, its fill -
and its primitives, each with the option it was drawn with. Drawing one
hands it to :mod:`xrdroot.canvas` as a pad ROOT had written, so a live
pad and a saved one are drawn by the same code.

``gPad`` is whichever pad is current: the canvas made last, or the pad
``cd`` went to.
"""

from __future__ import annotations

from typing import Any

from .drawn import Drawn
from .style import gStyle

__all__ = ["CANVASES", "TPad", "current", "find_anywhere", "gPad", "set_current"]

#: The canvases made and not closed, oldest first: ``gROOT->GetListOfCanvases()``.
CANVASES: list[Any] = []
_CURRENT: list[Any] = [None]


def current() -> Any:
    """The current pad, or ``None`` before any canvas is made."""
    return _CURRENT[0]


def set_current(pad: Any) -> None:
    _CURRENT[0] = pad


def _members(name: str, title: str, place: tuple[float, float, float, float]) -> dict[str, Any]:
    """A pad's members, as ``gStyle`` makes a new one."""
    xlow, ylow, xup, yup = place
    return {
        "fName": name, "fTitle": title, "fBits": 0x03000000,
        "fXlowNDC": xlow, "fYlowNDC": ylow, "fWNDC": xup - xlow, "fHNDC": yup - ylow,
        "fLeftMargin": gStyle.GetPadLeftMargin(), "fRightMargin": gStyle.GetPadRightMargin(),
        "fBottomMargin": gStyle.GetPadBottomMargin(), "fTopMargin": gStyle.GetPadTopMargin(),
        "fX1": 0.0, "fY1": 0.0, "fX2": 1.0, "fY2": 1.0,
        "fLogx": gStyle.GetOptLogx(), "fLogy": gStyle.GetOptLogy(), "fLogz": gStyle.GetOptLogz(),
        "fGridx": gStyle.GetPadGridX(), "fGridy": gStyle.GetPadGridY(),
        "fTickx": gStyle.GetPadTickX(), "fTicky": gStyle.GetPadTickY(),
        "fFillColor": gStyle.GetPadColor(), "fFillStyle": 1001,
        "fLineColor": 1, "fLineStyle": 1, "fLineWidth": 1,
        "fBorderMode": gStyle.GetPadBorderMode(), "fBorderSize": gStyle.GetPadBorderSize(),
        "fFrameFillColor": gStyle.GetFrameFillColor(),
        "fFrameFillStyle": gStyle.GetFrameFillStyle(),
        "fFrameLineColor": gStyle.GetFrameLineColor(),
        "fFrameLineWidth": gStyle.GetFrameLineWidth(),
        "fFrameLineStyle": gStyle.GetFrameLineStyle(),
        "fFrameBorderMode": gStyle.GetFrameBorderMode(),
        "fFrameBorderSize": gStyle.GetFrameBorderSize(),
        "fTheta": 30.0, "fPhi": 30.0,
    }  # fmt: skip


#: The pad's members a macro sets and gets by name: ``SetLeftMargin`` is ``fLeftMargin``.
PAD_FIELDS = {
    "LeftMargin": float, "RightMargin": float, "BottomMargin": float, "TopMargin": float,
    "Logx": int, "Logy": int, "Logz": int, "Gridx": int, "Gridy": int,
    "Tickx": int, "Ticky": int, "BorderMode": int, "BorderSize": int,
    "FrameFillColor": int, "FrameFillStyle": int, "FrameLineColor": int,
    "FrameLineWidth": int, "FrameLineStyle": int, "FrameBorderMode": int,
    "FrameBorderSize": int, "Theta": float, "Phi": float,
    "XlowNDC": float, "YlowNDC": float, "WNDC": float, "HNDC": float,
    "X1": float, "Y1": float, "X2": float, "Y2": float,
}  # fmt: skip
#: The setters ROOT gives a default argument of one: ``SetLogy()`` is ``SetLogy(1)``.
SWITCHES = ("Logx", "Logy", "Logz", "Gridx", "Gridy", "Tickx", "Ticky")


class TPad(Drawn):
    """A pad: a rectangle of its canvas, with margins round a frame, and what it draws."""

    classname = "TPad"
    groups = ("line", "fill")
    fields = PAD_FIELDS

    def __init__(
        self, name: str = "", title: str = "", xlow: float = 0.0, ylow: float = 0.0,
        xup: float = 1.0, yup: float = 1.0, color: int = -1, bordersize: int = -1,
        bordermode: int = -2,
    ) -> None:  # fmt: skip
        super().__init__()
        self.members.update(_members(str(name), str(title), (xlow, ylow, xup, yup)))
        for member, value, unset in (
            ("fFillColor", color, -1), ("fBorderSize", bordersize, -1),
            ("fBorderMode", bordermode, -2),
        ):  # fmt: skip
            if value != unset:
                self.members[member] = int(value)
        #: What the pad draws, in order, each with the option it was drawn with.
        self.primitives: list[tuple[Any, str]] = []
        #: The pad this one is drawn in, and its number there.
        self.mother: TPad | None = None
        self.number = 0
        #: The stats boxes and title made when it was last drawn, by what each is of.
        self.made: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("Set") and name[3:] in SWITCHES:
            member = f"f{name[3:]}"
            return lambda value=1: self.members.__setitem__(member, int(value))
        return super().__getattr__(name)

    # -- where it is, and whose ---------------------------------------------------

    def cd(self, subpadnumber: int = 0) -> TPad:
        """Make this pad current, or the ``n``-th pad ``Divide`` made in it."""
        if subpadnumber:
            found = self.GetPad(subpadnumber)
            if found is None:
                return self  # ROOT leaves gPad where it was for a pad that is not there
            set_current(found)
            return found
        set_current(self)
        return self

    def GetPad(self, subpadnumber: int) -> TPad | None:
        return next((pad for pad in self.pads() if pad.number == subpadnumber), None)

    def pads(self) -> list[TPad]:
        return [obj for obj, _ in self.primitives if isinstance(obj, TPad)]

    def GetMother(self) -> TPad:
        return self.mother or self

    def GetCanvas(self) -> Any:
        pad = self
        while pad.mother is not None:
            pad = pad.mother
        return pad

    def GetNumber(self) -> int:
        return self.number

    def SetPad(self, xlow: float, ylow: float, xup: float, yup: float) -> None:
        self.members.update(fXlowNDC=xlow, fYlowNDC=ylow, fWNDC=xup - xlow, fHNDC=yup - ylow)

    def SetMargin(self, left: float, right: float, bottom: float, top: float) -> None:
        self.members.update(fLeftMargin=left, fRightMargin=right)
        self.members.update(fBottomMargin=bottom, fTopMargin=top)

    def SetGrid(self, valuex: int = 1, valuey: int = 1) -> None:
        self.members.update(fGridx=int(valuex), fGridy=int(valuey))

    def SetTicks(self, valuex: int = 1, valuey: int = 1) -> None:
        self.members.update(fTickx=int(valuex), fTicky=int(valuey))

    def Range(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """The pad's own coordinates, for what is drawn in them with no frame."""
        self.members.update(fX1=float(x1), fY1=float(y1), fX2=float(x2), fY2=float(y2))

    def GetRange(self, *cells: Any) -> tuple[float, float, float, float]:
        ends = (self.members[name] for name in ("fX1", "fY1", "fX2", "fY2"))
        values = tuple(float(v) for v in ends)
        for cell, value in zip(cells, values):
            cell.value = value
        return values  # type: ignore[return-value]

    def SetEditable(self, mode: bool = True) -> None:
        self.members["fEditable"] = bool(mode)

    def IsEditable(self) -> bool:
        return bool(self.members.get("fEditable", True))

    def SetFixedAspectRatio(self, fixed: bool = True) -> None:
        self.members["fFixedAspectRatio"] = bool(fixed)

    # -- what it holds -------------------------------------------------------------

    def Divide(
        self, nx: int = 1, ny: int = 1, xmargin: float = 0.01, ymargin: float = 0.01,
        color: int = 0,
    ) -> None:  # fmt: skip
        """``nx`` by ``ny`` pads in this one, numbered from the top left, row by row."""
        before = current()
        self.Clear()
        set_current(before)
        dx, dy = 1.0 / max(int(nx), 1), 1.0 / max(int(ny), 1)
        for row in range(int(ny)):
            for column in range(int(nx)):
                x1, y2 = column * dx + xmargin, 1.0 - row * dy - ymargin
                x2, y1 = x1 + dx - 2 * xmargin, max(y2 - dy + 2 * ymargin, 0.0)
                number = row * int(nx) + column + 1
                name = f"{self.GetName()}_{number}"
                pad = TPad(name, name, x1, y1, x2, y2, color or self.members["fFillColor"])
                self._adopt(pad, number)

    def _adopt(self, pad: TPad, number: int = 0) -> None:
        pad.mother, pad.number = self, number
        self.primitives.append((pad, ""))

    def Draw(self, option: str = "") -> None:
        """Draw this pad in the current one; a canvas is drawn already."""
        parent = current()
        if parent is None or parent is self or self.mother is not None:
            return
        parent._adopt(self)

    def Clear(self, option: str = "") -> None:
        """Take everything out of the pad, pads and all, and make it current."""
        del option
        self.primitives = []
        self.made = {}
        set_current(self)

    def add(self, obj: Any, option: str) -> None:
        """Put ``obj`` in this pad, drawn with ``option``."""
        self.primitives.append((obj, str(option)))

    def GetListOfPrimitives(self) -> list[Any]:
        return [obj for obj, _ in self.primitives] + list(self.made.values())

    def FindObject(self, name: Any) -> Any:
        """The object of that name - or that object - in this pad or any pad in it."""
        for obj in self.GetListOfPrimitives():
            if obj is name or _name_of(obj) == name:
                return obj
            if isinstance(obj, TPad):
                found = obj.FindObject(name)
                if found is not None:
                    return found
        return None

    def GetPrimitive(self, name: str) -> Any:
        return next((obj for obj in self.GetListOfPrimitives() if _name_of(obj) == name), None)

    def Modified(self, flag: bool = True) -> None:
        self.members["fModified"] = bool(flag)

    def IsModified(self) -> bool:
        return bool(self.members.get("fModified", False))

    def Update(self) -> None:
        """Work out what drawing the pad makes - its frame, stats boxes and titles - now."""
        from .snapshot import prepare

        prepare(self)
        self.Modified(False)

    def Paint(self, option: str = "") -> None:
        self.Update()

    def RedrawAxis(self, option: str = "") -> None:
        """Nothing: the axes are drawn over everything in the pad anyway."""

    def Flush(self) -> None:
        """Nothing: nothing is shown until the pad is saved."""

    def Close(self, option: str = "") -> None:
        if current() is not None and current().GetCanvas() is self.GetCanvas():
            set_current(None)

    # -- its frame -------------------------------------------------------------------

    def _frame(self) -> tuple[float, float, float, float]:
        from .snapshot import frame_of

        return frame_of(self)

    def GetUxmin(self) -> float:
        return self._frame()[0]

    def GetUymin(self) -> float:
        return self._frame()[1]

    def GetUxmax(self) -> float:
        return self._frame()[2]

    def GetUymax(self) -> float:
        return self._frame()[3]

    def GetFrame(self) -> Any:
        from .snapshot import frame_box

        return frame_box(self)

    def DrawFrame(self, xmin: float, ymin: float, xmax: float, ymax: float, title: str = "") -> Any:
        """An empty histogram framing ``xmin`` to ``xmax`` and ``ymin`` to ``ymax``, drawn."""
        from .frames import frame_histogram

        self.cd()
        made = frame_histogram(xmin, ymin, xmax, ymax, title)
        made.Draw(" ")
        return made

    def BuildLegend(
        self, x1: float = 0.3, y1: float = 0.21, x2: float = 0.3, y2: float = 0.21,
        title: str = "", option: str = "",
    ) -> Any:  # fmt: skip
        """A legend of everything the pad draws that has a title, drawn in it."""
        from .legend import TLegend

        corners = (0.5, 0.67, 0.88, 0.88) if (x1, x2) == (x2, x1) else (x1, y1, x2, y2)
        legend = TLegend(*corners, title)
        for obj, drawn in self.primitives:
            if hasattr(obj, "_xrd") and not isinstance(obj, TPad):
                legend.AddEntry(obj, "", option or _entry_option(drawn))
        self.cd()
        legend.Draw()
        return legend

    # -- pictures -------------------------------------------------------------------

    def SaveAs(self, filename: str = "", option: str = "") -> None:
        from .output import save

        save(self, filename or f"{self.GetName()}.png", option)

    def Print(self, filename: str = "", option: str = "") -> None:
        from .output import save

        save(self, filename or f"{self.GetName()}.ps", option)

    def ls(self, option: str = "") -> None:
        print(f"{self.classname} {self.GetName()}: {self.GetTitle()}")
        for obj, drawn in self.primitives:
            print(f"  {type(obj).__name__} {_name_of(obj)!r} {drawn!r}")

    def __repr__(self) -> str:
        return f"<{self.classname} {self.GetName()!r} of {len(self.primitives)} primitives>"


def _entry_option(option: str) -> str:
    """What ``BuildLegend`` shows of something drawn with ``option``."""
    upper = option.upper()
    return "lp" if "P" in upper and "HIST" not in upper else "lf" if "HIST" in upper else "l"


def _name_of(obj: Any) -> str:
    getter = getattr(obj, "GetName", None)
    if callable(getter):
        return str(getter())
    return str(getattr(getattr(obj, "_xrd", obj), "name", "") or "")


def find_anywhere(name: str) -> Any:
    """The object called ``name`` in the current pad, or in any canvas, or ``None``."""
    pads = ([current()] if current() is not None else []) + list(CANVASES)
    return next((found for pad in pads if (found := pad.FindObject(name)) is not None), None)


class _Current:
    """``gPad``: whichever pad is current, whenever it is asked; false when there is none."""

    def __getattr__(self, name: str) -> Any:
        pad = current()
        if pad is None:
            raise AttributeError(f"gPad is null - no canvas has been made - so it has no {name}")
        return getattr(pad, name)

    def __bool__(self) -> bool:
        return current() is not None

    def __eq__(self, other: object) -> bool:
        return current() is other or (other is None and current() is None)

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return repr(current())


gPad: Any = _Current()
