"""``TPostScript``, ``TPDF`` and ``TSVG``: a pad written to a file by a device.

ROOT's file devices are pages a pad paints itself into: opening one takes
over the painting, so a macro opens a ``TPostScript``, draws, and closes it,
and the file holds a page for every pad drawn while it was open. Here the
painting is not per command - :mod:`xrdroot.canvas` draws a whole pad at
once, as ``SaveAs`` does - so a device is the page it writes when it is
closed: the pad it was opened on, or, opened before any pad was drawn, the
pad current when it closes. That is what ROOT's own ``SaveAs("x.ps")``
gives for one pad, in the format the device's class names.

The second argument is ROOT's orientiation code (``-111`` portrait, ``111``
landscape, and the other pairs it documents); it chose the paper a page is
printed on, and a page that is a picture of the pad has none, so it is kept
as it was given and otherwise unused. A ``TSVG``'s third argument, ROOT's
``compact``, likewise only shortens the XML a browser reads.

A device left open when the session ends is closed then, as ROOT closes one
when it is destroyed, so a macro that forgets its ``Close`` still leaves a
readable file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["OPEN", "TPDF", "TPostScript", "TSVG", "TVirtualPS", "close_devices"]

#: The devices not closed yet, which closing the session closes.
OPEN: list[TVirtualPS] = []


class TVirtualPS:
    """A page written as the format its class names, from the pad it was opened on.

    What a subclass states is its :attr:`suffix`, which a name given without
    one gets, and the format ``SaveAs`` takes from it.
    """

    classname = "TVirtualPS"
    #: The suffix a file name given without one is given.
    suffix = ""

    def __init__(self, filename: Any = "", orientation: Any = -111, *rest: Any) -> None:
        name = str(filename) or f"{self.classname}.{self.suffix.lstrip('.')}"
        if not Path(name).suffix:
            name += self.suffix
        self.fName = name
        self.fOrientation = int(orientation)
        self.fClosed = False
        self._pad: Any = _current_pad()
        OPEN.append(self)

    def Close(self, option: str = "") -> None:
        """Write the page, once: a device closed twice writes no second one."""
        if self.fClosed:
            return
        self.fClosed = True
        OPEN.remove(self)
        pad = self._pad if self._pad is not None else _current_pad()
        if pad is not None:
            pad.SaveAs(self.fName)

    def GetName(self) -> str:
        return self.fName


class TPostScript(TVirtualPS):
    classname = "TPostScript"
    suffix = ".ps"


class TPDF(TVirtualPS):
    classname = "TPDF"
    suffix = ".pdf"


class TSVG(TVirtualPS):
    classname = "TSVG"
    suffix = ".svg"


def _current_pad() -> Any:
    """The pad drawn on now, or ``None`` before any exists - not ``gPad``, which only
    refuses a name when there is none."""
    from .pads import current

    return current()


def close_devices() -> None:
    """Close every device left open, as ROOT closes one when it is destroyed."""
    for device in list(OPEN):
        device.Close()
