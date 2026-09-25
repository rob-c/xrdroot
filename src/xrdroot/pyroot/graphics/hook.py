"""What ``Draw`` does: put the object on the current pad, as ROOT's ``AppendPad`` does.

With no canvas yet, drawing makes ROOT's default one - ``c1``, 700 by 500 -
as ``gROOT->MakeDefCanvas`` does. Data drawn without ``SAME`` clears the
pad first, as ``TH1::Draw`` does: a histogram, a function, a stack or an
efficiency always, and a graph or multigraph when it draws its axes, with
``A``. Everything else - text, lines, legends, pads - is added over what
is there.
"""

from __future__ import annotations

from typing import Any

from ...graph import Graph
from ...stacks import MultiGraph
from ..core import set_draw_hook
from .canvas import default_canvas
from .pads import TPad, current
from .snapshot import data_of, is_data

__all__ = ["draw", "install"]


def _clears(obj: Any, option: str) -> bool:
    """Whether drawing ``obj`` with ``option`` starts the pad afresh."""
    upper = option.upper()
    if not is_data(obj) or "SAME" in upper:
        return False
    if isinstance(data_of(obj), (Graph, MultiGraph)):
        return "A" in upper
    return True


def draw(obj: Any, option: str = "") -> None:
    """Add ``obj`` to the current pad, drawn with ``option``, making a canvas if there is none."""
    pad = current() or default_canvas()
    if isinstance(obj, TPad):
        obj.Draw(option)
        return
    if _clears(obj, option):
        pad.Clear()
    pad.add(obj, option)
    pad.Modified()


def install() -> None:
    """Make :func:`draw` what ``TObject::Draw`` does."""
    set_draw_hook(draw)
