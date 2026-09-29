"""A live pad as the :class:`xrdroot.canvas.Canvas` a saved one is read as.

This is what makes a pad drawn here and a pad ROOT saved the same thing to
draw: each pad becomes a :class:`~xrdroot.canvas.Pad` of its members, its
primitives the objects they stand for - a histogram its ``_xrd``, a line
or a legend a :class:`~xrdroot.canvas.Primitive` of its members - each
with the option it was drawn with. What ROOT makes when it paints a pad,
the ``TFrame``, the ``title`` pave and each histogram's stats box, is made
by :func:`prepare` as ROOT would make it from ``gStyle``, and set among
the primitives where ROOT puts it, so the pad reads as one ROOT drew
before it saved it. The colours made in the session, and the palette if it
is not ``kBird``, go with the canvas as the colour tables a saved canvas
carries.
"""

from __future__ import annotations

from typing import Any

from ...buffer import Listed
from ...canvas import Canvas, Pad, Primitive
from ...canvas.frame import extent, owner
from ...efficiency import Efficiency
from ...function import Function
from ...graph import Graph
from ...hist import Histogram
from ...stacks import MultiGraph, Stack
from . import colors
from .drawn import Drawn
from .pads import TPad
from .style import gStyle

__all__ = ["DATA", "data_of", "frame_box", "frame_of", "model", "prepare"]

#: The classes a pad draws as data, as :mod:`xrdroot.canvas` does.
DATA = (Histogram, Graph, MultiGraph, Stack, Function, Efficiency)
#: Members that hold drawing classes of their own, which become primitives too.
HELD = ("fLines", "fPrimitives")


def data_of(obj: Any) -> Any:
    """What ``obj`` stands for: a wrapper's ``_xrd``, or itself."""
    return getattr(obj, "_xrd", obj)


def is_data(obj: Any) -> bool:
    return isinstance(data_of(obj), DATA)


def primitive(obj: Drawn) -> Primitive:
    """A drawing class as the :class:`~xrdroot.canvas.Primitive` ROOT would have saved."""
    members = dict(obj.members)
    for name in HELD:
        if isinstance(members.get(name), list):
            members[name] = [_converted(one) for one in members[name]]
    if "fObject" in members:
        members["fObject"] = _converted(members["fObject"])
    if obj.classname == "TPaveLabel":
        members["fTextSize"] = 0.0  # ROOT's is of the box, not of the pad: sized to fit
    return Primitive("TLatex" if obj.classname == "TMathText" else obj.classname, members)


def _converted(obj: Any) -> Any:
    if isinstance(obj, TPad):
        return model(obj)
    if isinstance(obj, Drawn):
        return primitive(obj)
    return data_of(obj)


def _lays_out(obj: Any) -> bool:
    """Is ``obj`` a helper that lays out pads, painting nothing itself?"""
    return callable(getattr(type(obj), "paint_pad", None))


def _painted(pad: TPad) -> list[tuple[Any, str]]:
    """Each primitive converted, a helper that lays out pads asked to place what it made.

    A helper may put what it paints into the pad beside itself, so the pad's
    primitives are walked as they were before any of them was asked.
    """
    from .texec import run_hung

    run_hung(pad)
    for obj, _ in list(pad.primitives):
        if _lays_out(obj):
            obj.paint_pad()
    from .cutg import cut_drawn

    cut = [cut_drawn(obj, option) for obj, option in pad.primitives if not _lays_out(obj)]
    return [(_converted(obj), option) for obj, option in cut]


def _drawn(pad: TPad) -> list[tuple[Any, str]]:
    """The pad's primitives as :mod:`xrdroot.canvas` takes them, before what painting adds.

    A helper that lays out pads - a ``TRatioPlot`` - paints nothing of its own:
    it is asked to place what it made afresh (``paint_pad``), and then passed over.
    """
    made = _painted(pad)
    if made and owner(_bare(pad, made)) is None:
        at = next((i for i, (obj, _) in enumerate(made) if isinstance(obj, Graph)), None)
        if at is not None and "SAME" not in made[at][1].upper():
            made[at] = (made[at][0], "A" + made[at][1])  # a graph alone draws its axes
    return made


def _bare(pad: TPad, drawn: list[tuple[Any, str]]) -> Pad:
    members = dict(pad.members)
    members["fPrimitives"] = Listed([obj for obj, _ in drawn], [option for _, option in drawn])
    return Pad("TPad", members)


def frame_of(pad: TPad) -> tuple[float, float, float, float]:
    """``fUxmin, fUymin, fUxmax, fUymax``: the frame's extent, logarithmic ends as powers."""
    bare = _bare(pad, _drawn(pad))
    found = owner(bare)
    if found is None:
        return tuple(float(pad.members[n]) for n in ("fX1", "fY1", "fX2", "fY2"))  # type: ignore[return-value]
    xmin, ymin, xmax, ymax = extent(*found, bare)
    (x1, x2), (y1, y2) = (
        _logged(xmin, xmax, bool(pad.members["fLogx"])),
        _logged(ymin, ymax, bool(pad.members["fLogy"])),
    )
    return x1, y1, x2, y2


def _logged(low: float, high: float, log: bool) -> tuple[float, float]:
    import math

    if not log:
        return low, high
    low = low if low > 0 else high * 1e-3
    return math.log10(low), math.log10(high)


def frame_box(pad: TPad) -> Any:
    """``GetFrame()``: the pad's frame, whose fill and line are the frame's from then on."""
    from .decorations import frame_of_pad

    return frame_of_pad(pad)


def prepare(pad: TPad) -> list[tuple[Any, str]]:
    """What painting ``pad`` draws, in order: its primitives, and its frame, title and stats."""
    from .decorations import decorate

    drawn = _drawn(pad)
    found = owner(_bare(pad, drawn))
    return decorate(pad, drawn, found)


def model(pad: TPad) -> Pad:
    """A live pad, and every pad in it, as :mod:`xrdroot.canvas` draws a saved one."""
    drawn = prepare(pad)
    members = dict(pad.members)
    tables = _colour_tables() if pad.mother is None else []
    members["fPrimitives"] = Listed(
        [obj for obj, _ in drawn] + tables, [option for _, option in drawn] + [""] * len(tables)
    )
    if pad.mother is not None or not hasattr(pad, "GetWw"):
        return Pad("TPad", members)
    return Canvas("TCanvas", {"TPad": members, "fCw": pad.GetWw(), "fCh": pad.GetWh()})


def _colour_tables() -> list[list[Primitive]]:
    """The colours made in the session and the palette, as a saved canvas carries them."""
    made = [
        Primitive("TColor", {"fNumber": n, "fRed": r, "fGreen": g, "fBlue": b})
        for n, (r, g, b) in sorted(colors.MADE.items())
    ]
    palette = gStyle.custom_palette()
    if palette is None:
        return [made] if made else []
    made = made or [Primitive("TColor", {"fNumber": 0, "fRed": 1.0, "fGreen": 1.0, "fBlue": 1.0})]
    return [made, [Primitive("TColor", {"fNumber": n}) for n in palette]]
