"""``gVirtualX``: the graphics system a canvas draws through, batch here.

ROOT's ``gVirtualX`` is the window system - X11, Cocoa, Qt - and on a machine
with no window it is a batch implementation that collects the primitives and
draws nothing. Tutorials ask it what it is (``gVirtualX.InheritsFrom("TGCocoa")``)
and set attributes on it; both are answered here as the batch one answers them.
"""

from __future__ import annotations

from typing import Any

__all__ = ["TVirtualX", "gVirtualX"]


class TVirtualX:
    """``TVirtualX``: the window system's interface, as much of it as a batch run needs."""

    #: The class ROOT's batch system reports itself as.
    _class_name = "TVirtualX"

    def InheritsFrom(self, name: Any) -> bool:
        """``InheritsFrom``: whether this is, or derives from, the class named."""
        return str(name) == self._class_name or str(name) == "TVirtualX"

    def IsBatch(self) -> bool:
        return True

    def GetName(self) -> str:
        return self._class_name

    def GetTitle(self) -> str:
        return "ROOT's window system, in batch"

    def SetDoubleBuffer(self, mode: Any = 1) -> None:
        """``SetDoubleBuffer``: nothing to double buffer where nothing is shown."""

    def SetCursor(self, *arguments: Any) -> None:
        """``SetCursor``: there is no cursor to set."""

    def SetTitle(self, *arguments: Any) -> None:
        """``SetTitle``: there is no window to name."""

    def MapWindow(self, *arguments: Any) -> None:
        """``MapWindow``: there is no window to map."""

    def Flush(self, *arguments: Any) -> None:
        """``Flush``: nothing was drawn, so nothing to flush."""

    def __repr__(self) -> str:
        return "<TVirtualX in batch>"


#: ``gVirtualX``: the window system in use - batch, since there is no window.
gVirtualX = TVirtualX()
