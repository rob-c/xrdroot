"""``DrawFrame``: an empty histogram that is nothing but a frame of the range asked for.

ROOT's ``TPad::DrawFrame`` books a ``TH1F`` called ``hframe`` of a thousand
empty bins over the x range, fixes its minimum and maximum to the y range
and draws it with no stats box. That is what this makes: through the core
module's ``TH1F`` when it has one, so that the frame is a histogram like
any other, and otherwise as a small stand-in of its own that holds the
histogram and answers the axis titles a macro sets on a frame.
"""

from __future__ import annotations

import importlib
from typing import Any

from ...canvas.statbox import NO_STATS
from ...hist import Histogram
from ..core import draw_hook

__all__ = ["FrameHistogram", "frame_histogram"]

#: How many bins ROOT's ``hframe`` has.
BINS = 1000


class _Axis:
    """An axis of the frame, as far as a frame's axis is styled: its title."""

    def __init__(self, row: dict[str, Any]) -> None:
        self._row = row

    def SetTitle(self, title: str = "") -> None:
        self._row["TNamed"]["fTitle"] = str(title)

    def GetTitle(self) -> str:
        return str(self._row["TNamed"]["fTitle"])


class FrameHistogram:
    """``hframe``, when there is no core ``TH1F`` to make it with."""

    def __init__(self, histogram: Histogram) -> None:
        self._xrd = histogram

    def GetName(self) -> str:
        return self._xrd.name

    def GetTitle(self) -> str:
        return self._xrd.title

    def GetXaxis(self) -> _Axis:
        return _Axis(self._xrd._core["fXaxis"])

    def GetYaxis(self) -> _Axis:
        return _Axis(self._xrd._core["fYaxis"])

    def SetTitle(self, title: str) -> None:
        main, *axes = str(title).split(";")
        self._xrd._core["TNamed"]["fTitle"] = main
        for axis, name in zip(axes, ("fXaxis", "fYaxis")):
            self._xrd._core[name]["TNamed"]["fTitle"] = axis

    def Draw(self, option: str = "") -> None:
        draw_hook(self, option)


def frame_histogram(xmin: float, ymin: float, xmax: float, ymax: float, title: str) -> Any:
    """``hframe``: empty, over ``xmin`` to ``xmax``, its minimum and maximum the y range."""
    core = importlib.import_module("xrdroot.pyroot.core")
    maker = getattr(core, "TH1F", None)
    if maker is not None:
        made = maker("hframe", title, BINS, xmin, xmax)
        made.SetMinimum(ymin)
        made.SetMaximum(ymax)
        made.SetStats(0)
        return made
    frame = FrameHistogram(Histogram.book("hframe", (BINS, float(xmin), float(xmax))))
    frame.SetTitle(title)
    held = frame._xrd._core
    held.update(fMinimum=float(ymin), fMaximum=float(ymax))
    held["TNamed"]["fBits"] = int(held["TNamed"].get("fBits", 0)) | NO_STATS
    return frame
