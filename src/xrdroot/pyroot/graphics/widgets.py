"""``TButton``, as a picture shows it; ``TBrowser`` in batch; ``TControlBar``/``TSlider`` refused.

A ``TButton`` is a small raised pad with its title in the middle and a
line of C++ it would run when clicked. In a picture it is only the pad and
the title, drawn as ROOT draws them - and a picture is all there is here,
as in ROOT's batch mode - so the button is kept, its method is kept, and
nothing ever clicks it. The widgets that are nothing but interaction - a
bar of buttons in a window of its own (``TControlBar``), a slider dragged
along a pad (``TSlider``) - are refused, saying why, since a script that
builds one is waiting on a person to use it. ``TBrowser`` is neither: it is a
window too, but ROOT's own batch run answers ``new TBrowser`` with a warning and
carries on, and the geometry export macros of ROOT's tests open one as their
first line.
"""

from __future__ import annotations

from typing import Any

from ...errors import UnsupportedFeatureError
from ..core.messages import Warning
from .pads import TPad
from .text import TLatex

__all__ = ["TButton", "TControlBar", "TSlider", "TGroupButton", "TBrowser"]

#: ``TAttText(22, 0, 1, 61, 0.65)``: how a button's title is written.
TITLE_TEXT = {"align": 22, "color": 1, "font": 61, "size": 0.65}


class TButton(TPad):
    """``TButton(title, method, x1, y1, x2, y2)``: a raised pad, its title, and its method."""

    classname = "TButton"

    def __init__(self, title: str = "", method: str = "", x1: float = 0.0, y1: float = 0.0,
                 x2: float = 1.0, y2: float = 1.0) -> None:  # fmt: skip
        super().__init__("button", str(title), x1, y1, x2, y2, 18, 2, 1)
        self._method = str(method)
        self._text: Any = None
        if title:
            self._text = TLatex(0.5, 0.5, str(title))
            self._text.SetTextAlign(TITLE_TEXT["align"])
            self._text.SetTextColor(TITLE_TEXT["color"])
            self._text.SetTextFont(TITLE_TEXT["font"])
            self._text.SetTextSize(TITLE_TEXT["size"])
            self.add(self._text, "")

    def GetMethod(self) -> str:
        return self._method

    def SetMethod(self, method: str) -> None:
        self._method = str(method)

    def __getattr__(self, name: str) -> Any:
        """The title's text attributes - ``SetTextSize`` and the rest - are the button's."""
        if name.startswith(("SetText", "GetText")) and self.__dict__.get("_text") is not None:
            return getattr(self._text, name)
        return super().__getattr__(name)


def _interactive(what: str, does: str) -> UnsupportedFeatureError:
    return UnsupportedFeatureError(
        f"{what} {does}, which needs someone at a screen; xrdroot.pyroot draws pictures, as "
        f"ROOT does in batch, and has no window for it to be used in"
    )


class TControlBar:
    """``TControlBar``: refused - it is a window of buttons for someone to press."""

    def __init__(self, *args: Any) -> None:
        raise _interactive("TControlBar", "is a window of buttons waiting to be pressed")


class TSlider:
    """``TSlider``: refused - it is a slider for someone to drag."""

    def __init__(self, *args: Any) -> None:
        raise _interactive("TSlider", "is a slider waiting to be dragged along a pad")


class TGroupButton(TButton):
    """``TGroupButton``: one of a group of buttons, drawn as any button is."""

    classname = "TGroupButton"


class TBrowser:
    """``TBrowser``: ROOT's browser window, which a batch run says it cannot open.

    ``TBrowser``'s constructor, given no window system, prints ROOT's own warning
    and stops there, leaving the browser half-built behind it; nothing afterwards
    touches it. That is the whole of what the geometry export macros of
    roottest/root/io/geo ask of it - ``new TBrowser`` is the line each one opens
    with - so refusing it would refuse the macro, while a browser with no window
    to browse in has nothing else to offer.
    """

    classname = "TBrowser"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        Warning("TBrowser::TBrowser", "The ROOT browser cannot run in batch mode")

    def __repr__(self) -> str:
        return "<TBrowser in batch>"
