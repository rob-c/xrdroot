"""Lines, arrows, boxes, ellipses, markers and polylines, as a macro makes and draws them.

Each is its members and the ``Set``/``Get`` of each (:mod:`.drawn`); the
``DrawLine``, ``DrawBox`` and the rest draw a copy with this one's
attributes at a new place, and hand the copy back, as ROOT's do.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from .drawn import NDC_BIT, Drawn

__all__ = [
    "TArc",
    "TArrow",
    "TBox",
    "TCrown",
    "TEllipse",
    "TLine",
    "TMarker",
    "TPolyLine",
    "TPolyMarker",
    "TWbox",
]

#: The ends of a line or the corners of a box.
CORNERS = {"X1": float, "Y1": float, "X2": float, "Y2": float}


def _copy(obj: Drawn, option: str = "", **members: Any) -> Any:
    """A copy of ``obj`` at a new place, drawn, as every ``DrawLine``-like call makes."""
    made = obj.Clone()
    made.members.update(members)
    made.Draw(option)
    return made


class TLine(Drawn):
    """A straight line, ``(x1, y1)`` to ``(x2, y2)``."""

    classname = "TLine"
    groups = ("line",)
    fields = CORNERS

    def __init__(self, x1: float = 0.0, y1: float = 0.0, x2: float = 0.0, y2: float = 0.0) -> None:
        super().__init__(fX1=float(x1), fY1=float(y1), fX2=float(x2), fY2=float(y2))

    def DrawLine(self, x1: float, y1: float, x2: float, y2: float) -> Any:
        return _copy(self, "", fX1=x1, fY1=y1, fX2=x2, fY2=y2)

    def DrawLineNDC(self, x1: float, y1: float, x2: float, y2: float) -> Any:
        made = _copy(self, "", fX1=x1, fY1=y1, fX2=x2, fY2=y2)
        made.SetBit(NDC_BIT)
        return made


class TArrow(TLine):
    """A line with a head, or two: ``">"``, ``"<|>"``, ``"->-"``..."""

    classname = "TArrow"
    groups = ("line", "fill")
    fields: ClassVar[dict[str, type]] = {**CORNERS, "ArrowSize": float, "Angle": float}

    def __init__(
        self, x1: float = 0.0, y1: float = 0.0, x2: float = 0.0, y2: float = 0.0,
        arrowsize: float = 0.05, option: str = ">",
    ) -> None:  # fmt: skip
        super().__init__(x1, y1, x2, y2)
        self.members.update(fArrowSize=float(arrowsize), fOption=str(option), fAngle=60.0)

    def SetOption(self, option: str = ">") -> None:
        self.members["fOption"] = str(option)

    def GetOption(self) -> str:
        return str(self.members["fOption"])

    def DrawArrow(
        self, x1: float, y1: float, x2: float, y2: float, size: float = 0.0, option: str = ""
    ) -> Any:
        return _copy(
            self, "", fX1=x1, fY1=y1, fX2=x2, fY2=y2,
            fArrowSize=size or self.members["fArrowSize"], fOption=option or self.GetOption(),
        )  # fmt: skip


class TBox(Drawn):
    """A rectangle, corner to corner, filled and outlined as its attributes say."""

    classname = "TBox"
    groups = ("line", "fill")
    fields = CORNERS

    def __init__(self, x1: float = 0.0, y1: float = 0.0, x2: float = 0.0, y2: float = 0.0) -> None:
        super().__init__(fX1=float(x1), fY1=float(y1), fX2=float(x2), fY2=float(y2))

    def DrawBox(self, x1: float, y1: float, x2: float, y2: float) -> Any:
        return _copy(self, "", fX1=x1, fY1=y1, fX2=x2, fY2=y2)


class TWbox(TBox):
    """A box with a raised or sunken border."""

    classname = "TWbox"
    fields: ClassVar[dict[str, type]] = {**CORNERS, "BorderSize": int, "BorderMode": int}

    def __init__(
        self, x1: float = 0.0, y1: float = 0.0, x2: float = 0.0, y2: float = 0.0,
        color: int = 18, bordersize: int = 5, bordermode: int = 1,
    ) -> None:  # fmt: skip
        super().__init__(x1, y1, x2, y2)
        self.members.update(fFillColor=int(color), fFillStyle=1001)
        self.members.update(fBorderSize=int(bordersize), fBorderMode=int(bordermode))


#: An ellipse's members.
ROUND = {"X1": float, "Y1": float, "R1": float, "R2": float,
         "Phimin": float, "Phimax": float, "Theta": float}  # fmt: skip


class TEllipse(Drawn):
    """An ellipse, or the slice of one between two angles, tilted by ``theta``."""

    classname = "TEllipse"
    groups = ("line", "fill")
    fields = ROUND

    def __init__(
        self, x1: float = 0.0, y1: float = 0.0, r1: float = 0.0, r2: float = 0.0,
        phimin: float = 0.0, phimax: float = 360.0, theta: float = 0.0,
    ) -> None:  # fmt: skip
        super().__init__(
            fX1=float(x1), fY1=float(y1), fR1=float(r1), fR2=float(r2 or r1),
            fPhimin=float(phimin), fPhimax=float(phimax), fTheta=float(theta),
        )  # fmt: skip

    def DrawEllipse(
        self, x1: float, y1: float, r1: float, r2: float,
        phimin: float = 0.0, phimax: float = 360.0, theta: float = 0.0, option: str = "",
    ) -> Any:  # fmt: skip
        return _copy(
            self, option, fX1=x1, fY1=y1, fR1=r1, fR2=r2 or r1,
            fPhimin=phimin, fPhimax=phimax, fTheta=theta,
        )  # fmt: skip


class TArc(TEllipse):
    """A circle, or an arc of one."""

    classname = "TArc"

    def __init__(
        self, x1: float = 0.0, y1: float = 0.0, radius: float = 0.0,
        phimin: float = 0.0, phimax: float = 360.0,
    ) -> None:  # fmt: skip
        super().__init__(x1, y1, radius, radius, phimin, phimax)

    def DrawArc(
        self, x1: float, y1: float, radius: float,
        phimin: float = 0.0, phimax: float = 360.0, option: str = "",
    ) -> Any:  # fmt: skip
        return _copy(self, option, fX1=x1, fY1=y1, fR1=radius, fR2=radius,
                     fPhimin=phimin, fPhimax=phimax)  # fmt: skip


class TCrown(TEllipse):
    """The ring between two circles, or a slice of it."""

    classname = "TCrown"

    def __init__(
        self, x1: float = 0.0, y1: float = 0.0, radin: float = 0.0, radout: float = 0.0,
        phimin: float = 0.0, phimax: float = 360.0,
    ) -> None:  # fmt: skip
        super().__init__(x1, y1, radin, radout, phimin, phimax)
        self.members["fR2"] = float(radout)


class TMarker(Drawn):
    """One marker at ``(x, y)``."""

    classname = "TMarker"
    groups = ("marker",)
    fields: ClassVar[dict[str, type]] = {"X": float, "Y": float}

    def __init__(self, x: float = 0.0, y: float = 0.0, marker: int = 1) -> None:
        super().__init__(fX=float(x), fY=float(y))
        self.members["fMarkerStyle"] = int(marker)

    def DrawMarker(self, x: float, y: float) -> Any:
        return _copy(self, "", fX=x, fY=y)


class TPolyLine(Drawn):
    """Points joined by lines, or filled as an area when drawn ``"f"``."""

    classname = "TPolyLine"
    groups = ("line", "fill")

    def __init__(self, n: int = 0, x: Any = None, y: Any = None, option: str = "") -> None:
        xs = np.zeros(int(n)) if x is None else np.asarray(x, dtype=float)[: int(n)].copy()
        ys = np.zeros(int(n)) if y is None else np.asarray(y, dtype=float)[: int(n)].copy()
        super().__init__(fN=int(n), fX=xs, fY=ys, fOption=str(option), fLastPoint=int(n) - 1)

    def SetPoint(self, i: int, x: float, y: float) -> None:
        """Point ``i``, the polyline grown to hold it if it is past the end."""
        index = int(i)
        if index >= len(self.members["fX"]):
            grow = index + 1 - len(self.members["fX"])
            self.members["fX"] = np.append(self.members["fX"], np.zeros(grow))
            self.members["fY"] = np.append(self.members["fY"], np.zeros(grow))
        self.members["fX"][index], self.members["fY"][index] = float(x), float(y)
        self.members["fN"] = max(self.members["fN"], index + 1)
        self.members["fLastPoint"] = max(self.members["fLastPoint"], index)

    def SetNextPoint(self, x: float, y: float) -> int:
        self.SetPoint(self.members["fLastPoint"] + 1, x, y)
        return int(self.members["fLastPoint"])

    def SetPolyLine(self, n: int, x: Any = None, y: Any = None, option: str = "") -> None:
        TPolyLine.__init__(self, n, x, y, option or self.members["fOption"])

    def GetN(self) -> int:
        return int(self.members["fN"])

    def GetX(self) -> Any:
        return self.members["fX"]

    def GetY(self) -> Any:
        return self.members["fY"]

    def DrawPolyLine(self, n: int, x: Any, y: Any, option: str = "") -> Any:
        return _copy(self, option, fN=int(n), fX=np.asarray(x, float)[:n].copy(),
                     fY=np.asarray(y, float)[:n].copy(), fOption=option)  # fmt: skip


class TPolyMarker(TPolyLine):
    """A marker at each of its points."""

    classname = "TPolyMarker"
    groups = ("marker",)
