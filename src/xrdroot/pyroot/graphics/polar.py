"""``TGraphPolar`` and ``TGraphPolargram``: points at an angle and a radius, on a polar grid.

A polar graph is drawn as ROOT draws one: when its pad is painted it finds
the pad's polargram - or makes one from its points' reach
(``CreatePolargram``) and draws it - and paints its points in the
polargram's unit circle, cut where a line leaves it. The polargram paints
its circles, spokes and labels as ``TGraphPolargram::Paint`` does, and sets
the pad's range to ``-1.25..1.25``. Both are pads' layout helpers: what
they paint is made afresh - lines, polylines, markers, text, an axis - each
time the pad is, from their numbers as they are then, which is why a script
can change the polargram after ``Update`` and see it.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from ...canvas import polargram as grid
from ...limits import optimize
from .drawn import Drawn
from .shapes import TLine, TPolyLine
from .text import TGaxis, TLatex

__all__ = ["TGraphPolargram"]

#: Each unit of angle a polargram labels in, and how many of it make half a turn.
HALF_TURN = {"radian": math.pi, "degree": 180.0, "grad": 100.0}


def replace_made(pad: Any, owner: Any, made: list[tuple[Any, str]]) -> None:
    """Put what ``owner`` paints just after it in ``pad``, in place of what it painted last."""
    old = {id(obj) for obj, _ in getattr(owner, "_made", [])}
    pad.primitives = [(obj, option) for obj, option in pad.primitives if id(obj) not in old]
    at = next(i for i, (obj, _) in enumerate(pad.primitives) if obj is owner) + 1
    pad.primitives[at:at] = made
    owner._made = made


class TGraphPolargram(Drawn):
    """``TGraphPolargram(name, rmin, rmax, tmin, tmax[, option])``: a polar grid."""

    classname = "TGraphPolargram"
    groups: ClassVar[tuple[str, ...]] = ("line", "text")

    def __init__(self, name: str = "", rmin: float = 0.0, rmax: float = 1.0, tmin: float = 0.0,
                 tmax: float = 0.0, option: str = "") -> None:  # fmt: skip
        super().__init__(fName=str(name), fTitle="Polargram")
        self._range = [float(rmin), float(rmax), float(tmin), float(tmax)]
        self._divisions = {"radial": 508, "polar": 508}
        self._labels = {"polar": [1, 62, 0.04], "radial": [1, 62, 0.035]}
        self._offsets = {"polar": 0.04, "radial": 0.025}
        self._unit, self._angle, self._option = "radian", 0.0, ""
        self._made: list[tuple[Any, str]] = []
        units = [("R", "radian", 2 * math.pi), ("D", "degree", 360.0), ("G", "grad", 200.0)]
        chosen = next((u for u in units if u[0] in str(option).upper()), None)
        if chosen is not None:
            self._unit, self._range[2], self._range[3] = chosen[1], 0.0, chosen[2]
        self.SetLineStyle(3)

    # -- its numbers ----------------------------------------------------------------------

    def GetRMin(self) -> float:
        return self._range[0]

    def GetRMax(self) -> float:
        return self._range[1]

    def GetTMin(self) -> float:
        return self._range[2]

    def GetTMax(self) -> float:
        return self._range[3]

    def SetRangeRadial(self, rmin: float, rmax: float) -> None:
        if rmin < rmax:
            self._range[:2] = [float(rmin), float(rmax)]

    def ChangeRangePolar(self, tmin: float, tmax: float) -> None:
        if tmin < tmax:
            self._range[2:] = [float(tmin), float(tmax)]

    def SetRangePolar(self, tmin: float, tmax: float) -> None:
        """``SetRangePolar``: the angles' range, in no unit ROOT knows - labelled by number."""
        self._unit = ""
        self.ChangeRangePolar(tmin, tmax)

    def SetToRadian(self) -> None:
        self._unit = "radian"
        self.ChangeRangePolar(0.0, 2 * math.pi)

    def SetToDegree(self) -> None:
        self._unit = "degree"
        self.ChangeRangePolar(0.0, 360.0)

    def SetToGrad(self) -> None:
        self._unit = "grad"
        self.ChangeRangePolar(0.0, 200.0)

    def SetTwoPi(self) -> None:
        self.SetRangePolar(0.0, 2 * math.pi)

    def IsRadian(self) -> bool:
        return self._unit == "radian"

    def IsDegree(self) -> bool:
        return self._unit == "degree"

    def IsGrad(self) -> bool:
        return self._unit == "grad"

    def scale(self) -> float:
        """The painter's ``c``: what turns an angle in the polargram's unit into radians."""
        return {"degree": 180 / math.pi, "grad": 100 / math.pi}.get(self._unit, 1.0)

    def SetNdivPolar(self, ndiv: int = 508) -> None:
        if int(ndiv) > 0:
            self._divisions["polar"] = int(ndiv)

    def SetNdivRadial(self, ndiv: int = 508) -> None:
        self._divisions["radial"] = int(ndiv)

    def GetNdivPolar(self) -> int:
        return self._divisions["polar"]

    def GetNdivRadial(self) -> int:
        return self._divisions["radial"]

    def SetAxisAngle(self, angle: float = 0.0) -> None:
        self._angle = math.radians(float(angle))

    def GetAngle(self) -> float:
        return self._angle

    # -- its labels ----------------------------------------------------------------------

    def SetPolarLabelColor(self, color: int = 1) -> None:
        self._labels["polar"][0] = int(color)

    def SetPolarLabelFont(self, font: int = 62) -> None:
        self._labels["polar"][1] = int(font)

    def SetPolarLabelSize(self, size: float = 0.04) -> None:
        self._labels["polar"][2] = float(size)

    def SetRadialLabelColor(self, color: int = 1) -> None:
        self._labels["radial"][0] = int(color)

    def SetRadialLabelFont(self, font: int = 62) -> None:
        self._labels["radial"][1] = int(font)

    def SetRadialLabelSize(self, size: float = 0.035) -> None:
        self._labels["radial"][2] = float(size)

    def GetPolarLabelSize(self) -> float:
        return float(self._labels["polar"][2])

    def GetRadialLabelSize(self) -> float:
        return float(self._labels["radial"][2])

    def SetPolarOffset(self, offset: float = 0.04) -> None:
        self._offsets["polar"] = float(offset)

    def SetRadialOffset(self, offset: float = 0.025) -> None:
        self._offsets["radial"] = float(offset)

    # -- drawing -------------------------------------------------------------------------

    def Draw(self, option: str = "") -> None:
        """``Draw``: into the current pad, over what is there."""
        from .canvas import default_canvas
        from .pads import current

        self.draw_in(current() or default_canvas(), str(option))

    def draw_in(self, pad: Any, option: str) -> None:
        """``Paint`` then ``AppendPad``, into ``pad``: what ``Draw`` does into the current one."""
        self._pad, self._option = pad, option
        pad.add(self, option)
        self.paint_pad()

    def _line(self, x1: float, y1: float, x2: float, y2: float, style: int) -> tuple[Any, str]:
        line = TLine(x1, y1, x2, y2)
        line.SetLineColor(self.GetLineColor())
        line.SetLineWidth(self.GetLineWidth())
        line.SetLineStyle(style)
        return line, ""

    def _circle(self, r: float, style: int) -> tuple[Any, str]:
        xs, ys = grid.circle(r)
        polyline = TPolyLine(len(xs), xs, ys)
        polyline.SetLineColor(self.GetLineColor())
        polyline.SetLineWidth(self.GetLineWidth())
        polyline.SetLineStyle(style)
        return polyline, ""

    def _label(self, x: float, y: float, text: str, align: int, angle: float) -> tuple[Any, str]:
        color, font, size = self._labels["polar"]
        latex = TLatex(x, y, text)
        latex.SetTextColor(int(color))
        latex.SetTextFont(int(font))
        latex.SetTextSize(float(size))
        latex.SetTextAlign(align)
        latex.SetTextAngle(angle)
        return latex, ""

    def paint_pad(self) -> None:
        """``Paint``: the circles and axis, the spokes and their labels, afresh."""
        pad = self._pad
        pad.members.update(fX1=-1.25, fY1=-1.25, fX2=1.25, fY2=1.25)
        option = self._option.upper()
        polar, radial = "P" in option, "R" in option
        polar, radial = (True, True) if not polar and not radial else (polar, radial)
        spokes = self._polar("N" not in option, "O" in option) if polar else []
        made = self._radial(radial) + spokes
        replace_made(pad, self, made)

    def _axis(self) -> tuple[Any, str]:
        """``PaintRadialDivisions``' axis: along the axis angle, graduated ``rmin`` to ``rmax``."""
        ndiv = self._divisions["radial"]
        axis = TGaxis(0.0, 0.0, math.cos(self._angle), math.sin(self._angle), self._range[0],
                      self._range[1], abs(ndiv), "SDH" + ("N" if ndiv < 0 else ""))  # fmt: skip
        color, font, size = self._labels["radial"]
        axis.SetLabelColor(int(color))
        axis.SetLabelFont(int(font))
        axis.SetLabelSize(float(size))
        axis.SetLabelOffset(self._offsets["radial"])
        return axis, ""

    def _optimized_circles(self, major: int, minor: int) -> list[tuple[Any, str]]:
        """The circles at round radii ``Optimize`` finds, the minor ones dashed between."""
        rmin, rmax = self._range[:2]
        low, high, count, width = optimize(rmin, rmax, major)
        first, last = (low - rmin) / (rmax - rmin), (high - rmin) / (rmax - rmin)
        dist = (last - first) / count
        between = optimize(low, low + width, minor)[2]
        made, radius = [], first
        for i in range(1, count + 3):
            made.append(self._circle(radius, 1))
            minors = [radius + j * dist / between for j in range(1, between + 1)]
            made += [self._circle(r, 2) for r in minors if r <= 1]
            radius = first + (i - 1) * dist
        return made

    def _even_circles(self, major: int, minor: int) -> list[tuple[Any, str]]:
        """With the divisions not to be optimised, circles evenly out to the edge."""
        made = []
        for i in range(1, major + 1):
            made.append(self._circle(i / major, 1))
            made += [self._circle(i / major - j / (major * minor), 2) for j in range(1, minor)]
        return made

    def _radial(self, axis: bool) -> list[tuple[Any, str]]:
        """``PaintRadialDivisions``: the axis, the edge of the circle, the circles within."""
        ndiv = self._divisions["radial"]
        major, minor = abs(ndiv) % 100, abs(ndiv) // 100
        made = [self._axis()] if axis else []
        made.append(self._circle(1.0, self.GetLineStyle()))
        rings = self._optimized_circles if ndiv > 0 else self._even_circles
        self.SetLineStyle(1)  # as ROOT's painting leaves the polargram's own line
        return made + rings(major, minor)

    def _spoke_label(self, i: int, major: int, ortho: bool) -> tuple[Any, str]:
        """The label at the end of spoke ``i``: along it (``O``), or upright beside it."""
        theta = i * 2 * math.pi / major
        tmin, tmax = self._range[2:]
        text = (grid.radian_label(i, major) if self.IsRadian()
                else grid.number_label(tmin + i * (tmax - tmin) / major))  # fmt: skip
        reach = 1 + self._offsets["polar"]
        x, y = reach * math.cos(theta), reach * math.sin(theta)
        if ortho:
            align, angle = grid.find_align(theta, True), grid.find_text_angle(theta)
            return self._label(x, y, text, align, angle)
        lift = 0.04 if 3 * math.pi / 12 <= theta < 2 * math.pi / 3 else 0.01
        return self._label(x, y + lift, text, grid.find_align(theta), 0.0)

    def _polar(self, labels: bool, ortho: bool) -> list[tuple[Any, str]]:
        """``PaintPolarDivisions``: a spoke and its label at each division, dashed ones between."""
        ndiv = self._divisions["polar"]
        major, minor = ndiv % 100, ndiv // 100
        made = []
        for i in range(major):
            theta = i * 2 * math.pi / major
            made += [self._spoke_label(i, major, ortho)] if labels else []
            made.append(self._line(0.0, 0.0, math.cos(theta), math.sin(theta), 1))
            between = [theta + j * 2 * math.pi / (major * minor) for j in range(1, minor)]
            made += [self._line(0.0, 0.0, math.cos(t), math.sin(t), 2) for t in between]
        return made
