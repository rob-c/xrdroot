"""``TGraphPolar``: a graph of angles and radii, drawn in its pad's polargram.

It is a ``TGraphErrors`` - its ``x`` the angles, its ``y`` the radii - whose
drawing is ``TGraphPainter::PaintGraphPolar``'s: the polargram found in the
pad, or made from the points and drawn; the points taken into its unit
circle; error bars along the radius and arcs round it (``E``); the line cut
where it leaves the circle (``L``, ``C``), filled (``F``), and marked
(``P``). What it paints is made afresh each time its pad is painted.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...canvas import polargram as grid
from ..core.graphs import TGraphErrors
from .polar import TGraphPolargram, replace_made
from .shapes import TLine, TPolyLine, TPolyMarker

__all__ = ["TGraphPolar"]


def _styled(piece: Any, graph: Any, groups: tuple[str, ...]) -> Any:
    """``piece`` in ``graph``'s line, fill or marker attributes."""
    names = {"line": ("LineColor", "LineStyle", "LineWidth"), "fill": ("FillColor", "FillStyle"),
             "marker": ("MarkerColor", "MarkerStyle", "MarkerSize")}  # fmt: skip
    for group in groups:
        for name in names[group]:
            getattr(piece, f"Set{name}")(getattr(graph, f"Get{name}")())
    return piece


class TGraphPolar(TGraphErrors):
    """``TGraphPolar(n, theta, r[, etheta, er])``: points at angles ``theta``, radii ``r``."""

    CLASS_TITLE = "Polar plot"

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self._polargram: Any = None
        self._pad: Any = None
        self._made: list[tuple[Any, str]] = []

    def GetPolargram(self) -> Any:
        """The polargram it was painted in - none until its pad has been painted once."""
        return self._polargram

    def SetPolargram(self, polargram: Any) -> None:
        self._polargram = polargram

    def CreatePolargram(self, option: str = "") -> Any:
        """A polargram round the points' reach, as ``TGraphPolar::CreatePolargram`` makes one."""
        if not self.GetN():
            return None
        x, y = self.GetX(), self.GetY()
        ranges = grid.polar_range(x, y, self.GetEX(), self.GetEY())
        rmin, rmax, tmin, tmax = ranges
        return TGraphPolargram("Polargram", rmin, rmax, tmin, tmax, option)

    def SetMinRadial(self, minimum: float) -> None:
        if self._polargram is not None:
            self._polargram.SetRangeRadial(minimum, self._polargram.GetRMax())

    def SetMaxRadial(self, maximum: float) -> None:
        if self._polargram is not None:
            self._polargram.SetRangeRadial(self._polargram.GetRMin(), maximum)

    def SetMinPolar(self, minimum: float) -> None:
        if self._polargram is not None:
            self._polargram.ChangeRangePolar(minimum, self._polargram.GetTMax())

    def SetMaxPolar(self, maximum: float) -> None:
        if self._polargram is not None:
            self._polargram.ChangeRangePolar(self._polargram.GetTMin(), maximum)

    def GetXpol(self) -> np.ndarray[Any, Any]:
        return self._unit_circle()[0]

    def GetYpol(self) -> np.ndarray[Any, Any]:
        return self._unit_circle()[1]

    def _unit_circle(self) -> tuple[Any, Any]:
        gram = self._polargram or self.CreatePolargram()
        ranges = (gram.GetRMin(), gram.GetRMax(), gram.GetTMin(), gram.GetTMax())
        xs, ys = grid.to_unit_circle(self.GetX(), self.GetY(), ranges, gram.scale())
        return xs, ys

    def Draw(self, option: str = "") -> None:
        """``Draw``: into the current pad, over what is there, as ``AppendPad`` puts it."""
        from .canvas import default_canvas
        from .pads import current

        self._pad = current() or default_canvas()
        self._pad.add(self, str(option))

    # -- painting ------------------------------------------------------------------------

    def _found_polargram(self) -> Any:
        """The pad's polargram: its own if still there, else the pad's first, else one made."""
        pad = self._pad
        drawn = [obj for obj, _ in pad.primitives if isinstance(obj, TGraphPolargram)]
        if self._polargram is not None and self._polargram in drawn:
            return self._polargram
        if drawn:
            return drawn[0]
        made = self.CreatePolargram("")
        made.draw_in(pad, "".join(c for c in "NO" if c in self._option()))
        return made

    def _option(self) -> str:
        option = next((opt for obj, opt in self._pad.primitives if obj is self), "")
        return str(option).upper().replace("SAME", "")

    def _errors(self, gram: Any, ranges: tuple[float, ...]) -> list[tuple[Any, str]]:
        """``E``: a bar along the radius at each point, and an arc round it."""
        x, y, ex, ey = self.GetX(), self.GetY(), self.GetEX(), self.GetEY()
        c = gram.scale()
        low = grid.to_unit_circle(x, y - ey, ranges, c)
        high = grid.to_unit_circle(x, y + ey, ranges, c)
        made: list[tuple[Any, str]] = []
        for x1, y1, x2, y2 in zip(*low, *high, strict=False):
            if (x1, y1) != (x2, y2):
                made.append((_styled(TLine(x1, y1, x2, y2), self, ("line",)), ""))
        rmin, rmax, tmin, tmax = ranges
        per_turn = (tmax - tmin) / (2 * np.pi)
        for theta, r, e in zip(x, y, ex, strict=False):
            phis = [np.degrees(c * (theta + s * e - tmin) / per_turn) for s in (-1, 1)]
            if phis[0] != phis[1]:
                xs, ys = grid.circle((r - rmin) / (rmax - rmin), *phis)
                made.append((_styled(TPolyLine(len(xs), xs, ys), self, ("line",)), ""))
        return made

    def _run(self, xs: list[float], ys: list[float], option: str) -> list[tuple[Any, str]]:
        """One unbroken run of the line: filled, joined and marked as the option says."""
        made: list[tuple[Any, str]] = []
        if "F" in option:
            made.append((_styled(TPolyLine(len(xs), xs, ys, "f"), self, ("fill",)), "f"))
        if "L" in option or "C" in option:
            made.append((_styled(TPolyLine(len(xs), xs, ys), self, ("line",)), ""))
        if "P" in option:
            made.append((_styled(TPolyMarker(len(xs), xs, ys), self, ("marker",)), ""))
        return made

    def paint_pad(self) -> None:
        """``PaintGraphPolar``: the polargram found or made, then the points in its circle."""
        gram = self._found_polargram()
        self._polargram = gram
        option = self._option()
        ranges = (gram.GetRMin(), gram.GetRMax(), gram.GetTMin(), gram.GetTMax())
        made = self._errors(gram, ranges) if "E" in option else []
        xs, ys = grid.to_unit_circle(self.GetX(), self.GetY(), ranges, gram.scale())
        for run_x, run_y in grid.inside_runs(xs, ys):
            made += self._run(run_x, run_y, option)
        replace_made(self._pad, self, made)
