"""ROOT 7's canvas and its pads: the primitives drawn on each, in order, and the frame.

``RCanvas::Create(title)`` makes a canvas; ``canvas->Draw<RLine>(p1, p2)``
makes a primitive, puts it on the pad and hands it back to be set up;
``Divide(nx, ny)`` makes sub-pads, ``pads[x][y]``; ``AddFrame()`` the
frame with its axes. ROOT shows a canvas in a web browser, through a
window it serves; run in batch, as here, there is no window, so
``Show`` and ``Update`` change nothing anyone sees and an update is
never confirmed, as in ROOT's batch mode, and ``SaveAs`` - which ROOT
renders through a headless browser - reports it made no image.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

from ..core.messages import message
from ..core.objects import typed
from . import drawables
from .attrs import frozen
from .axes import RAttrAxis
from .drawables import Primitive, TObjectDrawable
from .kinds import RAttrBorder, RAttrFill, RAttrMargins
from .lengths import RPadPos, position

__all__ = ["RFrame", "RPad", "RCanvas"]


class RFrame(Primitive):
    """``RFrame``: the pad's frame - its border, fill and margins, and its axes ``x``, ``y``."""

    MEMBERS = frozen({"border": RAttrBorder, "fill": RAttrFill, "margins": RAttrMargins,
                      "drawAxes": lambda: False, "gridX": lambda: False, "gridY": lambda: False,
                      "x": RAttrAxis, "y": RAttrAxis, "z": RAttrAxis})  # fmt: skip
    CSS_TYPE = "frame"


def _made(kind: Any, args: tuple[Any, ...]) -> Primitive:
    """The primitive ``Draw<kind>(args...)`` makes: by class, by name, or a ROOT 6 object's."""
    if kind is None:
        return TObjectDrawable(*args)
    if isinstance(kind, type):
        return kind(*args)  # type: ignore[no-any-return]
    name = str(kind).rpartition("::")[2]
    found = getattr(drawables, name, None)
    if not (isinstance(found, type) and issubclass(found, Primitive)) or name == "Primitive":
        raise TypeError(f"{kind} is not a drawable a ROOT 7 pad holds; they are "
                        f"{', '.join(n for n in drawables.__all__ if n != 'Primitive')}")
    return found(*args)


class _Pad:
    """What a canvas and a pad share: the primitives on it, in the order they were drawn."""

    def __init__(self) -> None:
        self._primitives: list[Any] = []
        self._style: Any = None

    @typed
    def Draw(self, kind: Any, *args: Any) -> Any:
        """``Draw<T>(args...)``: a new primitive on this pad, to be set up by the caller."""
        made = _made(kind, args)
        self._primitives.append(made)
        return made

    Add = Draw

    def AddFrame(self) -> RFrame:
        """``AddFrame()``: the pad's frame, made first if it has none, behind all the rest."""
        found = self.GetFrame()
        if found is None:
            found = RFrame()
            self._primitives.insert(0, found)
        return found

    def GetFrame(self) -> RFrame | None:
        return next((p for p in self._primitives if isinstance(p, RFrame)), None)

    def GetPrimitives(self) -> list[Any]:
        return list(self._primitives)

    def NumPrimitives(self) -> int:
        return len(self._primitives)

    def FindPrimitive(self, id: Any) -> Any:
        return next((p for p in self._primitives if getattr(p, "id", None) == str(id)), None)

    def Wipe(self) -> None:
        self._primitives.clear()

    def Divide(self, nx: Any, ny: Any, padding: Any = None) -> list[list[RPad]]:
        """``Divide(nx, ny)``: ``nx`` columns of ``ny`` pads each, ``pads[x][y]``."""
        columns, rows = int(nx), int(ny)
        pads = [[RPad(self, RPadPos(x / columns, y / rows), RPadPos(1 / columns, 1 / rows))
                 for y in range(rows)] for x in range(columns)]  # fmt: skip
        self._primitives.extend(pad for column in pads for pad in column)
        return pads

    def UseStyle(self, style: Any) -> None:
        """``UseStyle``: the style this pad's primitives are drawn with."""
        self._style = style

    def GetStyle(self) -> Any:
        return self._style


class RPad(_Pad):
    """``RPad``: a pad within a canvas or another pad, at a place and of a size on it."""

    def __init__(self, parent: Any = None, pos: Any = None, size: Any = None) -> None:
        super().__init__()
        self._parent = parent
        self.pos = RPadPos() if pos is None else position(pos)
        self.size = RPadPos(1, 1) if size is None else position(size)

    def GetParent(self) -> Any:
        return self._parent


class RCanvas(_Pad):
    """``RCanvas``: a canvas - a pad with a title and a size, shown in a browser's window."""

    #: The canvases made and not yet removed, as ``RCanvas::GetCanvases`` lists them.
    _made: ClassVar[list[RCanvas]] = []

    def __init__(self, title: Any = "") -> None:
        super().__init__()
        self._title = str(title)
        self._size = [800, 600]
        self._modified = False
        self._kept: list[Any] = []
        RCanvas._made.append(self)

    @staticmethod
    def Create(title: Any = "") -> RCanvas:
        return RCanvas(title)

    @staticmethod
    def GetCanvases() -> list[RCanvas]:
        return list(RCanvas._made)

    def __repr__(self) -> str:
        return f"<RCanvas {self._title!r} with {len(self._primitives)} primitives>"

    def GetTitle(self) -> str:
        return self._title

    def SetTitle(self, title: Any) -> None:
        self._title = str(title)

    def SetSize(self, width: Any, height: Any) -> None:
        self._size = [int(width), int(height)]

    def GetSize(self) -> list[int]:
        return list(self._size)

    def Show(self, where: Any = "") -> None:
        """``Show``: in batch there is no window to show the canvas in, as in ROOT's."""

    def Hide(self) -> None:
        """``Hide``: no window shows the canvas."""

    def IsShown(self) -> bool:
        return False

    def Modified(self) -> None:
        self._modified = True

    def IsModified(self) -> bool:
        return self._modified

    def Update(self, asynchronous: Any = False, done: Any = None) -> None:
        """``Update``: no window shows the canvas, so none is brought up to date - and, as in
        ROOT's batch mode, the callback that would say one was is never called."""
        self._modified = False

    def Run(self, seconds: Any = 0.0) -> None:
        """``Run(tm)``: the canvas's window served for ``tm`` seconds - here, the time passed."""
        time.sleep(max(float(seconds), 0.0))

    def ClearOnClose(self, kept: Any) -> None:
        """``ClearOnClose(obj)``: ``obj`` kept as long as the canvas is."""
        self._kept.append(kept)

    def Remove(self) -> None:
        """``Remove``: the canvas no longer one of ``GetCanvases()``."""
        if self in RCanvas._made:
            RCanvas._made.remove(self)

    def SaveAs(self, filename: Any) -> bool:
        """``SaveAs``: ROOT 7 makes a canvas's image in a headless web browser, which xrdroot
        does not drive; as ROOT does with no browser to hand, no file is made and the answer
        is ``false``, with a warning saying why."""
        message("Warning", "RCanvas::SaveAs", "the image %s of a ROOT 7 canvas is made by a "
                "web browser, which xrdroot does not drive; no file was written", str(filename))
        return False
