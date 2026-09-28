"""What ROOT adds to a pad when it paints it: the frame, the title, and stats boxes.

``THistPainter`` makes these from ``gStyle`` the first time a pad is
painted and keeps them, so a macro can ``Update`` and then find the stats
box to move it. :func:`decorate` does the same: the ``TFrame`` first, as
its pad's frame attributes say; after the histogram or graph whose frame
it is, a ``title`` pave where ``gStyle``'s title goes; after each
histogram shown with its statistics - and each fitted graph, with
``SetOptFit`` - a ``TPaveStats`` of the lines ``THistPainter::PaintStat``
writes, at ``gStyle``'s corner, as big as its lines need.
"""

from __future__ import annotations

from typing import Any

from ...canvas.frame import NO_TITLE
from ...canvas.model import lookup
from ...canvas.statbox import fit_rows, shows_stats, stats_rows
from ...graph import Graph
from ...hist import Histogram
from .paves import TPaveStats, TPaveText
from .shapes import TWbox
from .style import gStyle

__all__ = ["TFrame", "decorate", "frame_of_pad"]

#: How much wider a stats box that describes a fit is, as ``PaintStat`` makes it.
FIT_WIDTH = 1.8


class TFrame(TWbox):
    """A pad's frame: the box its axes are drawn round, filled and outlined."""

    classname = "TFrame"


#: A frame's own members, and the pad's they are kept as.
FRAME_MEMBERS = {
    "fFillColor": "fFrameFillColor", "fFillStyle": "fFrameFillStyle",
    "fLineColor": "fFrameLineColor", "fLineWidth": "fFrameLineWidth",
    "fLineStyle": "fFrameLineStyle", "fBorderMode": "fFrameBorderMode",
    "fBorderSize": "fFrameBorderSize",
}  # fmt: skip


def frame_of_pad(pad: Any) -> TFrame:
    """The pad's ``TFrame``, made from its frame attributes the first time it is asked for."""
    if "TFrame" not in pad.made:
        frame = TFrame()
        for own, kept in FRAME_MEMBERS.items():
            frame.members[own] = pad.members[kept]
        pad.made["TFrame"] = frame
    found: TFrame = pad.made["TFrame"]
    return found


def decorate(pad: Any, drawn: list[tuple[Any, str]], found: Any) -> list[tuple[Any, str]]:
    """``drawn`` with the frame, title and stats boxes painting the pad adds, in their places."""
    from .snapshot import primitive

    if found is None:
        return drawn
    made: list[tuple[Any, str]] = [(primitive(frame_of_pad(pad)), "")]
    for obj, option in drawn:
        made.append((obj, option))
        if obj is found[0]:
            made += [(primitive(one), "") for one in _title(pad, obj)]
        made += [(primitive(one), "") for one in _stats(pad, obj, option)]
    return made


def _title(pad: Any, obj: Any) -> list[TPaveText]:
    """The ``title`` pave ``gStyle`` puts at the top of a pad, if it asks for one."""
    title = str(getattr(obj, "title", "") or "").split(";")[0]
    if not gStyle.GetOptTitle() or not title or int(lookup(obj, "fBits", 0) or 0) & NO_TITLE:
        pad.made.pop("title", None)
        return []
    height, width = _title_size(pad, title)
    x1, y1 = _aligned(gStyle.GetTitleX(), gStyle.GetTitleY(), width, height)
    pave = TPaveText(x1, y1, x1 + width, y1 + height, "blNDC")
    pave.members.update(
        fName="title", fBorderSize=gStyle.GetTitleBorderSize(),
        fFillColor=gStyle.GetTitleFillColor(), fFillStyle=gStyle.GetTitleStyle(),
        fTextFont=gStyle.GetTitleFont(""), fTextSize=0.0, fTextColor=gStyle.GetTitleTextColor(),
        fTextAlign=22,
    )  # fmt: skip
    pave.AddText(title)
    pad.made["title"] = pave
    return [pave]


def _title_size(pad: Any, title: str) -> tuple[float, float]:
    """``THistPainter::PaintTitle``'s box: ``gStyle``'s, or as tall and wide as the title in
    ``gStyle``'s text font."""
    from ...canvas.latex import formula_form

    height = gStyle.GetTitleH()
    if height <= 0:
        size = gStyle.GetTitleFontSize()
        if gStyle.GetTitleFont("") % 10 == 3:
            size /= max(_pad_pixels(pad))
        height = 1.1 * size
    height = height if height > 0 else 0.05
    width = gStyle.GetTitleW()
    if width > 0:
        return height, width
    wide, high = _pad_pixels(pad)
    whole = (round(wide), round(high))
    form = formula_form(title, height, int(gStyle.GetTextFont()), whole, min(wide, high))
    height = max(height, 1.2 * form.height / high)
    return height, min(0.7, 0.02 + form.width / wide)


def _pad_pixels(pad: Any) -> tuple[float, float]:
    """How wide and tall a pad is in its canvas's pixels."""
    canvas = pad.GetCanvas()
    wide = float(canvas.members.get("fCw", 700)) * _fraction(pad, "fWNDC")
    high = float(canvas.members.get("fCh", 500)) * _fraction(pad, "fHNDC")
    return wide, high


def _fraction(pad: Any, member: str) -> float:
    """How much of its canvas a pad is, across or down, however deep it is."""
    share = 1.0
    while pad.mother is not None:
        share *= float(pad.members[member])
        pad = pad.mother
    return share


def _aligned(x: float, y: float, width: float, height: float) -> tuple[float, float]:
    """The bottom left corner of a box ``TitleAlign`` puts at ``(x, y)``."""
    align = int(gStyle.GetTitleAlign())
    across = {1: 0.0, 2: 0.5, 3: 1.0}.get(align // 10, 0.0)
    down = {1: 0.0, 2: 0.5, 3: 1.0}.get(align % 10, 1.0)
    return x - across * width, y - down * height


def _options(pave: Any) -> tuple[int, int]:
    """The ``fOptStat`` and ``fOptFit`` a box says: its own, or ``gStyle``'s."""
    stat = int(pave.members["fOptStat"]) if pave is not None else -1
    fit = int(pave.members["fOptFit"]) if pave is not None else -1
    stat = gStyle.GetOptStat() if stat < 0 else stat
    fit = gStyle.GetOptFit() if fit < 0 else fit
    return (1111 if stat == 1 else stat), (111 if fit == 1 else fit)


def _stats(pad: Any, obj: Any, option: str) -> list[TPaveStats]:
    """The stats box of ``obj``, if it has one drawn with ``option``."""
    key = f"stats{id(obj)}"
    stat, fit = _options(pad.made.get(key))
    rows = _rows(obj, option, stat, fit)
    if not rows:
        pad.made.pop(key, None)
        return []
    fitted = bool(fit_rows(obj, fit))
    box = pad.made.get(key) or _new_box(len(rows), fitted)
    box.SetParent(obj)
    box.DeleteText()
    for name, value in rows:
        box.AddText(f"{name} = {value}" if value else name)
    pad.made[key] = box
    return [box]


def _rows(obj: Any, option: str, stat: int, fit: int) -> list[tuple[str, str]]:
    """The lines of a stats box: a histogram's statistics and fit, or a graph's fit."""
    if isinstance(obj, Histogram) and shows_stats(obj, option):
        return stats_rows(obj, stat) + fit_rows(obj, fit)
    if isinstance(obj, Graph) and "SAME" not in option.upper():
        return fit_rows(obj, fit)
    return []


def _new_box(lines: int, fitted: bool) -> TPaveStats:
    """A stats box at ``gStyle``'s corner, as tall as its lines and wider for a fit."""
    width = gStyle.GetStatW() * (FIT_WIDTH if fitted else 1.0)
    x2, y2 = gStyle.GetStatX(), gStyle.GetStatY()
    height = gStyle.GetStatFontSize() * lines
    if height <= 0 or gStyle.GetStatFont() % 10 == 3:
        height = 0.25 * lines * gStyle.GetStatH()
    box = TPaveStats(x2 - width, y2 - height, x2, y2, "brNDC")
    box.members.update(
        fBorderSize=gStyle.GetStatBorderSize(), fFillColor=gStyle.GetStatColor(),
        fFillStyle=gStyle.GetStatStyle(), fTextFont=gStyle.GetStatFont(),
        fTextSize=gStyle.GetStatFontSize(), fTextColor=gStyle.GetStatTextColor(),
    )  # fmt: skip
    return box
