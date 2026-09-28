"""``TPainter3dAlgorithms``' raster screen: what hides what among a histogram's boxes.

ROOT draws a ``TH3`` drawn ``LEGO`` or ``BOX`` (``PaintH3BoxRaster``) from the
front to the back, and hides what is behind with a screen of 1000 by 800
cells over the view's square from -1.1 to 1.1: a line is drawn only where
the cells it passes are clear (``FindVisibleLine``), and each face drawn
fills the cells it covers, its border too (``FillPolygonBorder``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["RasterScreen"]

#: ``InitRaster``'s screen: how many cells across and up, over the view from -1.1 to 1.1.
CELLS = (1000, 800)
LOW, HIGH = -1.1, 1.1
#: The most visible pieces ``FindVisibleLine`` keeps of one line.
MOST = 100

Parts = list[list[float]]


class RasterScreen:
    """The cells a lego of boxes has covered so far, and the lines they hide."""

    def __init__(self) -> None:
        self.nx, self.ny = CELLS
        self.cells = np.zeros((self.ny, self.nx), dtype=bool)

    def cell(self, x: float, y: float) -> tuple[int, int]:
        """The cell a point of the view is in, truncated towards zero as ROOT's ``Int_t`` is."""
        span = HIGH - LOW
        return int(self.nx * ((x - LOW) / span) - 0.01), int(self.ny * ((y - LOW) / span) - 0.01)

    def _hidden(self, ix: int, iy: int) -> bool:
        """Whether a cell hides what is behind it: covered, or off the screen."""
        return not (0 <= ix < self.nx and 0 <= iy < self.ny) or bool(self.cells[iy, ix])

    def visible(self, p1: Any, p2: Any) -> list[tuple[float, float]]:
        """``FindVisibleLine``: the parts of the line ``p1`` to ``p2`` in sight, as fractions."""
        x1, y1 = self.cell(p1[0], p1[1])
        x2, y2 = self.cell(p2[0], p2[1])
        inverted = y1 > y2
        if inverted:
            x1, x2, y1, y2 = x2, x1, y2, y1
        if y1 >= self.ny or y2 < 0 or min(x1, x2) >= self.nx or max(x1, x2) < 0:
            return []
        if y2 - y1 > abs(x2 - x1):
            parts, dt = self._steep((x1, y1), (x2, y2))
        else:
            parts, dt = self._shallow((x1, y1), (x2, y2))
        return _finished(parts, dt, inverted)

    def _shallow(self, start: tuple[int, int], end: tuple[int, int]) -> tuple[Parts, float]:
        """A line more across than up: a cell a column, stepping up as Bresenham's walk does."""
        (x1, y1), (x2, y2) = start, end
        step = 1 if x2 >= x1 else -1
        dx, dy = abs(x2 - x1), y2 - y1
        dt = 1.0 / (dx + 1.0)
        now, test, iy = -dt, float(-(dx + 2 * dy)), y1
        walk = _Walk()
        for ix in range(x1, x2 + step, step):
            now += dt
            test += 2 * dy
            if test >= 0:
                iy, test = iy + 1, test - 2 * dx
            if walk.at(now, self._hidden(ix, iy)):
                return walk.parts, dt
        walk.end(now + dt + dt * 0.5)
        return walk.parts, dt

    def _steep(self, start: tuple[int, int], end: tuple[int, int]) -> tuple[Parts, float]:
        """A line more up than across: a cell a row, the rows past the screen's top not walked."""
        (x1, y1), (x2, y2) = start, end
        step = 1 if x2 >= x1 else -1
        dx, dy = abs(x2 - x1), y2 - y1
        dt = 1.0 / (dy + 1.0)
        now, test, ix = -dt, float(-(dy + 2 * dx)), x1
        walk = _Walk()
        for iy in range(y1, min(y2, self.ny - 1) + 1):
            now += dt
            test += 2 * dx
            if test >= 0:
                ix, test = ix + step, test - 2 * dy
            if walk.at(now, iy < 0 or self._hidden(ix, iy)):
                return walk.parts, dt
        walk.end(now + dt)
        return walk.parts, dt

    def fill(self, points: list[Any]) -> None:
        """``FillPolygonBorder``: the cells a convex face covers, its border's too, now covered."""
        corners = [self.cell(x, y) for x, y in points]
        spans: dict[int, list[int]] = {}
        for (xa, ya), (xb, yb) in zip(corners, corners[1:] + corners[:1]):
            for x, y in _edge_cells(xa, ya, xb, yb):
                low, high = spans.get(y, [x, x])
                spans[y] = [min(low, x), max(high, x)]
        for y, (low, high) in spans.items():
            if 0 <= y < self.ny:
                self.cells[y, max(low, 0) : min(high, self.nx - 1) + 1] = True


class _Walk:
    """The pieces of a line in sight, as a walk along it finds them going in and out of view."""

    def __init__(self) -> None:
        self.parts: Parts = []
        self.seen = False

    def at(self, now: float, hidden: bool) -> bool:
        """One cell of the walk; whether as many pieces as are kept have been found."""
        if not hidden:
            if not self.seen:
                self.seen = True
                self.parts.append([now, now])
            return False
        if self.seen:
            self.seen = False
            self.parts[-1][1] = now
            return len(self.parts) == MOST
        return False

    def end(self, at: float) -> None:
        """The last piece, if the line was in sight at its end, runs on to ``at``."""
        if self.seen:
            self.parts[-1][1] = at


def _finished(parts: Parts, dt: float, inverted: bool) -> list[tuple[float, float]]:
    """The pieces snapped to the line's ends within a cell of them, and turned back if turned."""
    if not parts:
        return []
    near = dt * 1.1
    if parts[0][0] <= near:
        parts[0][0] = 0.0
    if parts[-1][1] >= 1 - near:
        parts[-1][1] = 1.0
    if inverted:
        return [(1 - end, 1 - start) for start, end in parts]
    return [(start, end) for start, end in parts]


def _edge_cells(xa: int, ya: int, xb: int, yb: int) -> list[tuple[int, int]]:
    """The cells of an edge, one a row where it is steep and all of each run where it is not."""
    count = max(abs(xb - xa), abs(yb - ya))
    if count == 0:
        return [(xa, ya)]
    return [(xa + round((xb - xa) * k / count), ya + round((yb - ya) * k / count))
            for k in range(count + 1)]  # fmt: skip
