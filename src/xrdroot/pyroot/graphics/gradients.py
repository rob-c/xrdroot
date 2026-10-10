"""``TColorGradient``, ``TLinearGradient`` and ``TRadialGradient``: a colour that shades.

A gradient is a colour of the table like any other - ``SetFillColor`` takes
its index - whose colour runs between stops: each stop a colour of the table
and where, from 0 to 1, it sits. A linear one runs from a start point to an
end point, a radial one out from a centre to a radius, both in the unit
square of the thing filled (``kObjectBoundingMode``) or of the pad
(``kPadMode``). ``TColor::GetLinearGradient(angle, colors)`` lays the stops
evenly along a line at ``angle`` degrees across the object, 0 left to right
and 90 bottom to top; ``GetRadialGradient(r, colors)`` out from its middle.
What a pad draws with one is :mod:`xrdroot.canvas.gradient`'s business.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .colors import ALPHA, TColor, free_index, rgb_of

__all__ = ["TColorGradient", "TLinearGradient", "TRadialGradient", "GRADIENTS"]

#: ``TColorGradient::ECoordinateMode``: the unit square is the pad's, or the object's box.
kPadMode, kObjectBoundingMode = 0, 1
#: Every gradient made in the session, by its colour index: what a canvas is given.
GRADIENTS: dict[int, TColorGradient] = {}
#: One stop's colour: red, green, blue and opacity.
RGBA = tuple[float, float, float, float]


class Point:
    """``TColorGradient::Point``: ``fX`` and ``fY``, as ``{0.3, 0.3}`` is given."""

    def __init__(self, x: Any = 0.0, y: Any = 0.0) -> None:
        if not isinstance(x, (int, float)):  # a pair, as a braced list is handed over
            x, y = x
        self.fX, self.fY = float(x), float(y)

    def __iter__(self) -> Any:
        return iter((self.fX, self.fY))

    def __repr__(self) -> str:
        return f"Point({self.fX}, {self.fY})"


def _indices(colors: list[Any]) -> bool:
    """Whether the stops are the table's indices, rather than ``r, g, b, a`` four at a time."""
    return bool(colors) and all(isinstance(c, (int, np.integer)) for c in colors)


def _rgba(colors: Any, npoints: int) -> list[RGBA]:
    """The first ``npoints`` stops' colours, from indices or from the numbers themselves."""
    listed = list(colors)
    if _indices(listed):
        return [(*rgb_of(int(c)), ALPHA.get(int(c), 1.0)) for c in listed[:npoints]]
    flat = [float(c) for c in listed[: 4 * npoints]]
    return [(flat[at], flat[at + 1], flat[at + 2], flat[at + 3]) for at in range(0, len(flat), 4)]


class TColorGradient(TColor):
    """``TColorGradient(index, n, positions, colors[, mode])``: a colour that runs between stops."""

    Point = Point
    kPadMode, kObjectBoundingMode = kPadMode, kObjectBoundingMode
    KIND: ClassVar[str] = ""

    def __init__(
        self, index: int = -1, npoints: int = 0, positions: Any = (), colors: Any = (),
        mode: int = kObjectBoundingMode,
    ) -> None:  # fmt: skip
        self._positions = [float(p) for p in list(positions)[: int(npoints)]]
        self._stops = _rgba(colors, int(npoints))
        self._mode = int(mode)
        first = self._stops[0] if self._stops else (0.0, 0.0, 0.0, 1.0)
        super().__init__(index, *first[:3], a=first[3])
        GRADIENTS[self.GetNumber()] = self

    def ResetColor(self, npoints: int, positions: Any, colors: Any) -> None:
        self._positions = [float(p) for p in list(positions)[: int(npoints)]]
        self._stops = _rgba(colors, int(npoints))

    def SetColorAlpha(self, index: int, alpha: float) -> None:
        r, g, b, _ = self._stops[int(index)]
        self._stops[int(index)] = (r, g, b, float(alpha))

    def GetColorAlpha(self, index: int) -> float:
        return self._stops[int(index)][3]

    def SetCoordinateMode(self, mode: int) -> None:
        self._mode = int(mode)

    def GetCoordinateMode(self) -> int:
        return self._mode

    def GetNumberOfSteps(self) -> int:
        return len(self._positions)

    def GetColorPositions(self) -> list[float]:
        return list(self._positions)

    def GetColors(self) -> list[float]:
        """The stops' colours flat, ``r, g, b, a`` each, as ROOT keeps them."""
        return [channel for stop in self._stops for channel in stop]

    def members(self) -> dict[str, Any]:
        """The gradient as a canvas is given it, by ROOT's member names."""
        r, g, b = rgb_of(self.GetNumber())
        return {"fNumber": self.GetNumber(), "fRed": r, "fGreen": g, "fBlue": b,
                "fColorPositions": list(self._positions), "fColors": self.GetColors(),
                "fCoordinateMode": self._mode}  # fmt: skip


class TLinearGradient(TColorGradient):
    """``TLinearGradient``: the stops laid from ``fStart`` to ``fEnd`` in the unit square."""

    KIND = "linear"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._start, self._end = Point(0.0, 0.0), Point(1.0, 0.0)

    def SetStartEnd(self, p1: Any, p2: Any) -> None:
        self._start, self._end = Point(p1), Point(p2)

    def GetStart(self) -> Point:
        return self._start

    def GetEnd(self) -> Point:
        return self._end

    def members(self) -> dict[str, Any]:
        found = super().members()
        found.update(fStart=vars(self._start), fEnd=vars(self._end))
        return found


class TRadialGradient(TColorGradient):
    """``TRadialGradient``: the stops laid out from a centre to a radius - or, extended, from
    one circle to another."""

    KIND = "radial"
    kSimple, kExtended = 0, 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._start, self._r1, self._end, self._r2 = Point(0.5, 0.5), 0.0, Point(0.5, 0.5), 0.5
        self._type = self.kSimple

    def SetRadialGradient(self, center: Any, radius: float) -> None:
        """``SetRadialGradient(center, radius)``: a simple gradient, out from the centre."""
        self._end, self._r2, self._type = Point(center), float(radius), self.kSimple

    def SetStartEndR1R2(self, p1: Any, r1: float, p2: Any, r2: float) -> None:
        """``SetStartEndR1R2``: an extended one, from circle ``p1, r1`` to circle ``p2, r2``."""
        self._start, self._r1, self._end, self._r2 = Point(p1), float(r1), Point(p2), float(r2)
        self._type = self.kExtended

    def GetCenter(self) -> Point:
        return self._end

    def GetRadius(self) -> float:
        return self._r2

    def GetStart(self) -> Point:
        return self._start

    def GetR1(self) -> float:
        return self._r1

    def GetEnd(self) -> Point:
        return self._end

    def GetR2(self) -> float:
        return self._r2

    def GetGradientType(self) -> int:
        return self._type

    def members(self) -> dict[str, Any]:
        found = super().members()
        found.update(fStart=vars(self._start), fR1=self._r1, fEnd=vars(self._end),
                     fR2=self._r2, fType=self._type)  # fmt: skip
        return found


def _even(count: int) -> list[float]:
    """Stops laid evenly from 0 to 1, as ROOT lays them when no positions are given."""
    return [at / (count - 1) for at in range(count)] if count > 1 else [0.0]


def _stops(method: str, colors: Any, positions: Any) -> tuple[list[int], list[float]] | None:
    """The colours and positions a gradient is made of, or ``None`` after ROOT's refusal."""
    from ..core.messages import message

    listed = [int(c) for c in colors]
    if len(listed) < 2:
        message("Error", f"TColor::{method}", "number of colors should be at least 2")
        return None
    return listed, [float(p) for p in positions] or _even(len(listed))


def linear_gradient(angle: float, colors: Any, positions: Any = ()) -> int:
    """``TColor::GetLinearGradient``: a new colour shading along ``angle`` degrees, 0 left to
    right and 90 bottom to top, across whatever it fills; ``-1`` for fewer than two colours."""
    found = _stops("GetLinearGradient", colors, positions)
    if found is None:
        return -1
    stops, where = found
    made = TLinearGradient(free_index(), len(stops), where, stops)
    c, s = math.cos(math.radians(float(angle))), math.sin(math.radians(float(angle)))
    made.SetStartEnd((0.5 - 0.5 * c, 0.5 - 0.5 * s), (0.5 + 0.5 * c, 0.5 + 0.5 * s))
    return made.GetNumber()


def radial_gradient(radius: float, colors: Any, positions: Any = ()) -> int:
    """``TColor::GetRadialGradient``: a new colour shading out from the middle of whatever it
    fills to ``radius`` of its box; ``-1`` for fewer than two colours."""
    found = _stops("GetRadialGradient", colors, positions)
    if found is None:
        return -1
    stops, where = found
    made = TRadialGradient(free_index(), len(stops), where, stops)
    made.SetRadialGradient((0.5, 0.5), float(radius))
    return made.GetNumber()
