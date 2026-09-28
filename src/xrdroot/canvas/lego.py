"""``TPainter3dAlgorithms``' moving screen: the faces of a lego or surface, hidden lines removed.

ROOT draws ``LEGO`` and ``SURF`` in cartesian coordinates from the front to
the back, and hides what is behind with a "moving screen": the screen's
width is cut into 2000 slices, and each keeps the highest and lowest point
drawn in it so far. Each edge of a face is drawn only where it is above
the highest or below the lowest (``FindVisibleDraw``), and once the face is
drawn its edges widen the screen (``ModifyScreen``). The box's back walls
are drawn after the data, so they show only where no face hides them, with
dotted lines at the levels of the z axis's primary divisions
(``DrawFaceMove1``); the box's front edges last of all.

What is drawn is kept as segments in the pad's coordinates, each with its
colour, width and style.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from ..limits import optimize
from .view3d import View3D

__all__ = ["MovingScreen", "Segment"]

#: ``NumOfSlices``: how many slices the screen is cut into.
SLICES = 2000
#: The segments too short to draw, as fractions of their edge.
EPS = 1e-9
#: How close a line must come to the screen's edge to be above or below it.
TOUCH = 1e-6
#: What an untouched slice's edges are.
VERY_BIG = 9e99

Point = tuple[float, float, float]


class Segment(NamedTuple):
    """One visible piece of a line, in the pad's coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    color: int
    width: int
    style: int


class _Scan:
    """``FindVisibleDraw``'s walk along a sloped line: where it goes in and out of sight."""

    def __init__(self) -> None:
        self.parts: list[list[float]] = []
        self.seen = -1

    def _open(self, at: float) -> None:
        self.seen = 1
        self.parts.append([at, math.nan])

    def _close(self, at: float) -> None:
        self.parts[-1][1] = at

    @staticmethod
    def _case(above: float, below: float) -> int:
        """0 above what the slice has drawn, 2 below it, 1 hidden in it."""
        return 0 if above > TOUCH else (2 if below < -TOUCH else 1)

    def step(self, at: float, dt: float, gaps: tuple[float, float, float, float]) -> None:
        """One slice: whether the line is seen at its left end, and where it crosses the screen's edges."""
        up1, down1, up2, down2 = gaps
        left = self._case(up1, down1)
        if left != 1 and self.seen <= 0:
            self._open(at)
        if left == 1 and self.seen >= 0:
            self.seen = -1
            self._close(at)
        crossing_up = at + dt * (up1 / (up1 - up2)) if up1 != up2 else at
        crossing_down = at + dt * (down1 / (down1 - down2)) if down1 != down2 else at
        self._cross(left * 3 + self._case(up2, down2), crossing_up, crossing_down)

    def _cross(self, case: int, crossing_up: float, crossing_down: float) -> None:
        """What happens within the slice, by the cases at its two ends."""
        if case in (1, 7):
            self.seen = -1
            self._close(crossing_up if case == 1 else crossing_down)
        elif case in (2, 6):
            self._close(crossing_up if case == 2 else crossing_down)
            self.parts.append([crossing_down if case == 2 else crossing_up, math.nan])
        elif case in (3, 5):
            self._open(crossing_up if case == 3 else crossing_down)

    def finished(self) -> list[tuple[float, float]]:
        if self.seen > 0:
            self._close(1.0)
        return [(start, end) for start, end in self.parts]


class MovingScreen:
    """The moving screen over one view, and every segment drawn through it."""

    def __init__(self, view: View3D) -> None:
        self.view = view
        self.segments: list[Segment] = []
        self.levels: list[float] = []
        self.x0, self.dx = -1.1, 2.2 / SLICES
        self.up = [-VERY_BIG] * (2 * SLICES)
        self.down = [VERY_BIG] * (2 * SLICES)

    def reset(self, low: float = -1.1, high: float = 1.1) -> None:
        """``InitMoveScreen``: nothing drawn yet."""
        self.x0, self.dx = low, (high - low) / SLICES
        self.up = [-VERY_BIG] * (2 * SLICES)
        self.down = [VERY_BIG] * (2 * SLICES)

    def grid_levels(self, ndivz: int) -> None:
        """``DefineGridLevels``: the z axis's primary divisions, where the back walls are lined."""
        low, high = self.view.rmin[2], self.view.rmax[2]
        if ndivz > 0:
            start, _end, count, width = optimize(low, high, ndivz)
        else:
            count = abs(ndivz)
            start, width = low, (high - low) / count
        self.levels = [start + i * width for i in range(count + 1)]

    def _slices(self, a: Point, b: Point) -> tuple[float, float, float, float, bool]:
        """An edge's ends across and up the screen, left to right, and whether it was turned round."""
        x1, y1, _z1 = self.view.screen(a)
        x2, y2, _z2 = self.view.screen(b)
        if x1 >= x2:
            return x2, y2, x1, y1, True
        return x1, y1, x2, y2, False

    def visible(self, a: Point, b: Point) -> list[tuple[float, float]]:
        """``FindVisibleDraw``: the parts of the edge ``a`` to ``b`` the screen does not hide."""
        x1, y1, x2, y2, back = self._slices(a, b)
        i1 = int((x1 - self.x0) / self.dx) + 15
        i2 = int((x2 - self.x0) / self.dx) + 15
        if i1 != i2:
            parts = self._sloped(y1, y2, i1, i2)
        else:
            parts, back = self._upright(y1, y2, i1, back)
        return [(1 - t0, 1 - t1) for t0, t1 in parts] if back else parts

    def _upright(self, y1: float, y2: float, i1: int, back: bool) -> tuple[list[tuple[float, float]], bool]:
        """A line within one slice: seen where it reaches above or below what is drawn there."""
        if y2 <= y1:
            if y2 == y1:
                return [], back
            back = not back
            y1, y2 = y2, y1
        top, bottom = self.up[2 * i1 - 2], self.down[2 * i1 - 2]
        if i1 != 1:
            top, bottom = max(top, self.up[2 * i1 - 3]), min(bottom, self.down[2 * i1 - 3])
        if not (y1 < top and y2 > bottom):
            return [(0.0, 1.0)], back
        if y1 >= bottom and y2 <= top:
            return [], back
        parts = []
        if bottom > y1:
            parts.append((0.0, (bottom - y1) / (y2 - y1)))
        if top < y2:
            parts.append(((top - y1) / (y2 - y1), 1.0))
        return parts, back

    def _sloped(self, y1: float, y2: float, i1: int, i2: int) -> list[tuple[float, float]]:
        """A line across slices ``i1`` to ``i2``: in and out of sight at each slice's ends."""
        state = _Scan()
        span = float(i2 - i1)
        dy, dt = (y2 - y1) / span, 1 / span
        for i in range(i1, i2):
            yy1 = y1 + dy * (i - i1)
            state.step(dt * (i - i1), dt, (yy1 - self.up[2 * i - 2], yy1 - self.down[2 * i - 2],
                                            yy1 + dy - self.up[2 * i - 1], yy1 + dy - self.down[2 * i - 1]))
            if len(state.parts) + 1 >= 100:
                break
        return state.finished()

    def cover(self, a: Point, b: Point) -> None:
        """``ModifyScreen``: the screen widened to hide what is behind the edge ``a`` to ``b``."""
        x1, y1, x2, y2, _back = self._slices(a, b)
        i1 = int((x1 - self.x0) / self.dx) + 15
        i2 = int((x2 - self.x0) / self.dx) + 15
        if i1 == i2:
            return
        dy = (y2 - y1) / (i2 - i1)
        for i in range(i1, i2):
            yy1 = y1 + dy * (i - i1)
            yy2 = yy1 + dy
            self.down[2 * i - 2] = min(self.down[2 * i - 2], yy1)
            self.down[2 * i - 1] = min(self.down[2 * i - 1], yy2)
            self.up[2 * i - 2] = max(self.up[2 * i - 2], yy1)
            self.up[2 * i - 1] = max(self.up[2 * i - 1], yy2)
