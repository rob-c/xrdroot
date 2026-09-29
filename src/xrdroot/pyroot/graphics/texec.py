"""``TExec``: a line of C++ run whenever the pad it is drawn in is painted.

ROOT paints a pad's primitives in order, and a ``TExec`` among them runs
its command when its turn comes - ``Pal1()`` setting the palette the
surface drawn after it is coloured with, ``drawtext()`` writing a label at
each point of a graph with ``TLatex::PaintText``. Here the command runs,
through :mod:`xrdroot.cint`, each time the pad is painted, and what it
paints - text painted with ``PaintText`` or ``PaintLatex`` - is put in the
pad where the ``TExec`` is, as ROOT's painting leaves it on the picture.
"""

from __future__ import annotations

from typing import Any

from ..core.objects import TNamed

__all__ = ["TExec"]

#: What a ``TExec``'s command is painting, while it runs; nothing when none is.
PAINTING: list[list[tuple[Any, str]]] = []


def painted(obj: Any, option: str = "") -> bool:
    """Take ``obj`` as painted by the command running, if one is: whether it was taken."""
    if not PAINTING:
        return False
    PAINTING[-1].append((obj, option))
    return True


class TExec(TNamed):
    """``TExec(name, command)``: ``command`` run each time the pad it is in is painted."""

    def __init__(self, name: Any = "", command: Any = "") -> None:
        super().__init__(str(name), str(command))
        self._pad: Any = None
        self._made: list[tuple[Any, str]] = []

    def SetAction(self, command: str) -> None:
        self.SetTitle(command)

    def GetAction(self) -> str:
        return self.GetTitle()

    def Exec(self, command: str = "") -> list[tuple[Any, str]]:
        """Run ``command`` - or the ``TExec``'s own - and hand back what it painted."""
        from ..core.troot import gROOT

        PAINTING.append([])
        try:
            gROOT.ProcessLine(str(command) or self.GetTitle())
        finally:
            made = PAINTING.pop()
        return made

    def Draw(self, option: str = "") -> None:
        """``Draw``: into the current pad, to run when it is painted."""
        from .canvas import default_canvas
        from .pads import current

        self._pad = current() or default_canvas()
        self._pad.add(self, str(option))

    def paint_pad(self) -> None:
        """Run the command, what it paints put in the pad in its place."""
        from .pads import current, set_current
        from .polar import replace_made

        before = current()
        set_current(self._pad)
        try:
            made = self.Exec()
        finally:
            set_current(before)
        replace_made(self._pad, self, made)


def _hung(obj: Any) -> list[TExec]:
    """The ``TExec``s hung on a histogram's or a graph's list of functions."""
    extras = getattr(obj, "__dict__", {}).get("_extras", [])
    return [one for one in extras if isinstance(one, TExec)]


def run_hung(pad: Any) -> None:
    """Run the ``TExec``s hung on what the pad draws, as painting each runs its functions',
    what they paint put just after it - in place of what they painted last time."""
    from .pads import current, set_current

    for obj, _ in list(pad.primitives):
        for execute in _hung(obj):
            old = {id(made) for made, _ in execute._made}
            pad.primitives = [(o, opt) for o, opt in pad.primitives if id(o) not in old]
            before = current()
            set_current(pad)
            try:
                execute._made = execute.Exec()
            finally:
                set_current(before)
            at = next(i for i, (o, _) in enumerate(pad.primitives) if o is obj) + 1
            pad.primitives[at:at] = execute._made
