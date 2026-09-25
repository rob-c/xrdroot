"""``TCanvas``: the pad a window would be, in batch mode always, and its size in pixels.

A canvas is made at a window size - 700 by 500 unless told, as ``gStyle``
says - and, as ROOT does in batch mode, its drawing area is that window
less its decoration: four pixels narrower and 28 shorter, so a canvas of
700 by 500 saves as a picture of 696 by 472. A negative width asks for the
drawing area itself to be that size. Nothing is shown; a canvas is drawn
when it is saved.
"""

from __future__ import annotations

from typing import Any

from .pads import CANVASES, TPad, set_current
from .style import gStyle

__all__ = ["DECORATION", "TCanvas", "default_canvas"]

#: What a window's frame takes from the drawing area of a canvas: across, and down.
DECORATION = (4, 28)


def _size(args: tuple[Any, ...]) -> tuple[int, int, int, int, bool]:
    """Where and how big: ``(form)``, ``(w, h)`` or ``(x, y, w, h)``, and whether it is decorated."""
    numbers = [int(a) for a in args if isinstance(a, (int, float)) and not isinstance(a, bool)]
    if len(numbers) >= 4:
        x, y, w, h = numbers[:4]
    elif len(numbers) >= 2:
        x, y, (w, h) = gStyle.GetCanvasDefX(), gStyle.GetCanvasDefY(), numbers[:2]
    else:
        form = numbers[0] if numbers else 1
        x = y = 10 * form
        w, h = (gStyle.GetCanvasDefW(), gStyle.GetCanvasDefH()) if form <= 1 else (500, 500)
    return x, y, abs(w), abs(h), w > 0


class TCanvas(TPad):
    """A canvas: the top pad of a picture, which is saved at its size in pixels.

    >>> c = TCanvas("c", "c", 800, 600)
    >>> c.GetWw(), c.GetWh()
    (796, 572)
    """

    classname = "TCanvas"

    def __init__(self, name: str = "c1", title: str = "", *size: Any) -> None:
        if not isinstance(title, str):
            title, size = str(name), (title, *size)
        super().__init__(str(name), str(title) or str(name), 0.0, 0.0, 1.0, 1.0)
        x, y, w, h, decorated = _size(size)
        across, down = DECORATION if decorated else (0, 0)
        self.members.update(
            fFillColor=gStyle.GetCanvasColor(), fBorderMode=gStyle.GetCanvasBorderMode(),
            fBorderSize=gStyle.GetCanvasBorderSize(), fWindowTopX=x, fWindowTopY=y,
            fWindowWidth=w, fWindowHeight=h, fCw=max(w - across, 1), fCh=max(h - down, 1),
        )  # fmt: skip
        for old in [c for c in CANVASES if c.GetName() == self.GetName()]:
            CANVASES.remove(old)  # ROOT deletes a canvas of the same name
        CANVASES.append(self)
        set_current(self)

    def GetWw(self) -> int:
        return int(self.members["fCw"])

    def GetWh(self) -> int:
        return int(self.members["fCh"])

    def GetWindowWidth(self) -> int:
        return int(self.members["fWindowWidth"])

    def GetWindowHeight(self) -> int:
        return int(self.members["fWindowHeight"])

    def GetWindowTopX(self) -> int:
        return int(self.members["fWindowTopX"])

    def GetWindowTopY(self) -> int:
        return int(self.members["fWindowTopY"])

    def SetCanvasSize(self, ww: int, wh: int) -> bool:
        """The drawing area's own size, in pixels."""
        self.members.update(fCw=int(ww), fCh=int(wh))
        return True

    def SetWindowSize(self, ww: int, wh: int) -> None:
        """The window's size, and so the drawing area's, less the decoration."""
        self.members.update(fWindowWidth=int(ww), fWindowHeight=int(wh))
        self.members.update(fCw=int(ww) - DECORATION[0], fCh=int(wh) - DECORATION[1])

    def SetWindowPosition(self, x: int, y: int) -> None:
        self.members.update(fWindowTopX=int(x), fWindowTopY=int(y))

    def Close(self, option: str = "") -> None:
        super().Close(option)
        if self in CANVASES:
            CANVASES.remove(self)

    def IsBatch(self) -> bool:
        return True

    def SetBatch(self, batch: bool = True) -> None:
        """Nothing: every canvas here is a batch canvas."""

    def Show(self) -> None:
        """Nothing: a batch canvas has no window to show."""

    def Iconify(self) -> None:
        """Nothing: a batch canvas has no window to shrink."""

    def SetRealAspectRatio(self, axis: int = 1) -> bool:
        return True

    def __repr__(self) -> str:
        return f"<TCanvas {self.GetName()!r} {self.GetWw()}x{self.GetWh()}>"


def default_canvas() -> TCanvas:
    """``gROOT->MakeDefCanvas()``: ``c1``, or ``c1_n2`` and on when there is one already."""
    names = {c.GetName() for c in CANVASES}
    name, number = "c1", 1
    while name in names:
        number += 1
        name = f"c1_n{number}"
    return TCanvas(name, name, gStyle.GetCanvasDefX(), gStyle.GetCanvasDefY(),
                   gStyle.GetCanvasDefW(), gStyle.GetCanvasDefH())  # fmt: skip
