"""``paramOn``: a box of a density's parameters and their errors, on a frame.

``pdf.paramOn(frame, Layout(0.55))`` adds a ``TPaveText`` - borderless,
unfilled, text size 0.04 - of one line per free parameter, each as
``RooRealVar::format(2, "NELU")`` writes it, from 0.06 of the pad down per
line from its top. What the pave is, is the graphics layer's: it is made by
the function :func:`set_pave` installs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..cmdargs import commands
from ..collections import as_list
from ..formatting import format_command, format_var

__all__ = ["Pave", "param_on", "set_pave"]


class Pave:
    """A pave's corners, option and lines: what a frame keeps until the graphics layer makes one."""

    def __init__(self, x1: float, y1: float, x2: float, y2: float, option: str) -> None:
        self.corners = (x1, y1, x2, y2)
        self.option = option
        self.lines: list[str] = []
        self.name = ""
        self.attributes: dict[str, Any] = {}

    def AddText(self, text: str) -> None:
        self.lines.append(str(text))

    def SetName(self, name: str) -> None:
        self.name = str(name)

    def GetName(self) -> str:
        return self.name

    def ClassName(self) -> str:
        return "TPaveText"

    def __getattr__(self, name: str) -> Any:
        if name.startswith("Set"):
            return lambda value: self.__dict__["attributes"].__setitem__(name[3:], value)
        raise AttributeError(name)


#: What makes a pave: the pyroot layer's ``TPaveText``, or the stand-in above.
PAVE: list[Callable[..., Any]] = [Pave]


def set_pave(fn: Callable[..., Any]) -> None:
    PAVE[0] = fn


def param_on(pdf: Any, frame: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    options = commands([a for a in args if hasattr(a, "name")], kwargs)
    label = str(options.get("Label", 0, "") or "")
    xmin = float(options.get("Layout", 0, 0.65))
    xmax = float(options.get("Layout", 1, 0.9))
    ymax = int(float(options.get("Layout", 2, 0.9)) * 10000) / 10000.0
    show_constants = bool(options.get("ShowConstants", 0, False))
    params = pdf.getParameters(frame.norm_vars or [])
    if "Parameters" in options:
        wanted = {one.GetName() for one in as_list(options.get("Parameters"))}
        params = [one for one in params if one.GetName() in wanted]
    shown = [p for p in params if show_constants or not p.isConstant()]
    lines = label.split("\n") if label else []
    ymin = ymax - 0.06 * len(shown) - 0.06 * len(lines)
    box = PAVE[0](xmin, ymax, xmax, ymin, "BRNDC")
    box.SetName(f"{pdf.GetName()}_paramBox")
    box.SetFillColor(0)
    box.SetBorderSize(0)
    box.SetTextAlign(12)
    box.SetTextSize(0.04)
    box.SetFillStyle(0)
    command = options.every("Format")
    for param in shown:
        box.AddText(format_command(param, command[-1]) if command else format_var(param, 2, "NELU"))
    for line in lines:
        box.AddText(line)
    frame.addObject(box)
    return frame
