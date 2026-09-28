"""The faces a lego or surface plot is drawn with, through the moving screen.

``DrawFaceMove1`` draws a face's level lines - where it crosses the z
axis's primary divisions, dotted - then its edges, each only where the
screen shows it, and then widens the screen by the face; ``DrawFaceMove2``
the same without level lines. The box's back walls (``BackBox``) are two
faces drawn that way; its front edges (``FrontBox``) two polylines drawn
over everything. :func:`lego_cells` and :func:`surface_cells` are
``LegoCartesian`` and ``SurfaceCartesian``: every bin's faces, from the
front to the back.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Any

from .lego import EPS, MovingScreen, Point, Segment
from .view3d import RAD, View3D

__all__ = ["back_box", "box_corners", "draw_face", "front_box", "lego_cells", "surface_cells"]

#: The back walls' corners, as ``BackBox`` numbers them.
BACK = ((1, 4, 8, 5), (4, 3, 7, 8))
#: The front edges' corners, as ``FrontBox`` numbers them.
FRONT = ((1, 2, 6, 5), (2, 3, 7, 6))
#: What the box is drawn in: black, a pixel wide, solid (its levels dotted).
BOX = (1, 1, 1)
#: The style level lines are drawn in.
DOTTED = 3


def _segments(screen: MovingScreen, a: Point, b: Point, look: tuple[int, int, int]) -> None:
    """The visible parts of the line ``a`` to ``b``, added to the screen's segments."""
    parts = screen.visible(a, b)
    if not parts:
        return
    (ax, ay, _), (bx, by, _) = screen.view.to_ndc(a), screen.view.to_ndc(b)
    color, width, style = look
    for t0, t1 in parts:
        if abs(t0 - t1) > EPS:
            screen.segments.append(Segment(ax + (bx - ax) * t0, ay + (by - ay) * t0,
                                           ax + (bx - ax) * t1, ay + (by - ay) * t1,
                                           color, width, style))  # fmt: skip


def level_lines(screen: MovingScreen, points: list[Point],
                values: list[float]) -> list[tuple[Point, Point]]:  # fmt: skip
    """``FindLevelLines``: where a face crosses each level, as the ends of a line across it."""
    if not screen.levels:
        return []
    low, high = min(values), max(values)
    if low >= screen.levels[-1] or high <= screen.levels[0]:
        return []
    lines: list[tuple[Point, Point]] = []
    for level in screen.levels:
        if low >= level:
            continue
        if high < level or len(lines) >= 200:
            break
        # low < level <= high: the face's edges cross the level at least twice
        found = _crossings(points, values, level)
        lines.append((found[0], found[1]))
    return lines


def _crossings(points: list[Point], values: list[float], level: float) -> list[Point]:
    """The first two points where the face's edges cross ``level``.

    A level strictly inside a face's range is crossed by at least two of its
    edges, going round it; ``FindLevelLines`` keeps the first two.
    """
    count = len(points)

    def crossing(i: int) -> Point | None:
        j = (i + 1) % count
        d1 = (values[i] - level) or 1e-99
        d2 = (values[j] - level) or 1e-99
        if d1 * d2 > 0:
            return None
        span = values[j] - values[i]
        return _between(points[i], points[j], d1 / span, d2 / span)

    found = [point for point in map(crossing, range(count)) if point is not None]
    return found[:2]


def _between(p: Point, q: Point, d1: float, d2: float) -> Point:
    """The point ``d2`` of ``p`` less ``d1`` of ``q``: where an edge crosses a level."""
    return (d2 * p[0] - d1 * q[0], d2 * p[1] - d1 * q[1], d2 * p[2] - d1 * q[2])


def draw_face(screen: MovingScreen, points: list[Point], values: list[float] | None,
              look: tuple[int, int, int],
              levels: tuple[int, int, int] | None = None) -> None:  # fmt: skip
    """``DrawFaceMove1`` (with ``values``, its level lines) or ``DrawFaceMove2``.

    The face's visible edges are drawn, then the screen is widened by it.
    """
    if values is not None and levels is not None:
        for a, b in level_lines(screen, points, values):
            _segments(screen, a, b, levels)
    count = len(points)
    for i in range(count):
        _segments(screen, points[i], points[(i + 1) % count], look)
    for i in range(count):
        screen.cover(points[i], points[(i + 1) % count])


def box_corners(view: View3D, angle: float = 90.0) -> list[Point]:
    """The box's corners as ``BackBox``, ``FrontBox`` and ``PaintLegoAxis`` take them.

    Each is turned by ``angle``, the angle between the x and y axes, which
    ROOT keeps at 90 degrees for a cartesian plot.
    """
    cosa, sina = math.cos(RAD * angle), math.sin(RAD * angle)
    return [(x + y * cosa, y * sina, z) for x, y, z in view.corners().vertices]


def _box_corners(screen: MovingScreen) -> list[Point]:
    """The corners of the box the screen looks at, as :func:`box_corners` turns them."""
    return box_corners(screen.view)


def back_box(screen: MovingScreen) -> None:
    """``BackBox``: the two back walls, their levels dotted, where nothing in front hides them."""
    corners = _box_corners(screen)
    for face in BACK:
        points = [corners[k - 1] for k in face]
        draw_face(screen, points, [p[2] for p in points], BOX, (1, 1, DOTTED))


def front_box(screen: MovingScreen) -> list[list[tuple[float, float]]]:
    """``FrontBox``: the front edges, as two polylines in the pad's coordinates, drawn over all."""
    corners = [screen.view.to_ndc(p) for p in _box_corners(screen)]
    return [[(corners[k - 1][0], corners[k - 1][1]) for k in face] for face in FRONT]


def _order(view: Any, nx: int, ny: int) -> tuple[range, range]:
    """The cells from the front to the back: the order ``LegoCartesian`` walks them for ``FB``."""
    incrx = -1 if view.tnorm[8] < 0 else 1
    incry = -1 if view.tnorm[9] < 0 else 1
    incrx, incry = -incrx, -incry
    xs = range(1, nx + 1) if incrx == 1 else range(nx, 0, -1)
    ys = range(1, ny + 1) if incry == 1 else range(ny, 0, -1)
    return xs, ys


def _sides(view: Any) -> tuple[bool, ...]:
    """Which sides of a lego's block face the eye: -y, +x, +y, -x, the bottom and the top."""
    y, x, z = view.normal(0, 1, 0), view.normal(1, 0, 0), view.normal(0, 0, 1)
    return (y < 0, x > 0, y > 0, x < 0, z < 0, z > 0)


def lego_cells(screen: MovingScreen, nx: int, ny: int, cell: Callable[[int, int], Any],
               look: tuple[int, int, int]) -> None:  # fmt: skip
    """``LegoCartesian``: each bin a block from ``zmin`` to its content, its seen faces drawn."""
    xs, ys = _order(screen.view, nx, ny)
    seen = _sides(screen.view)
    for iy in ys:
        for ix in xs:
            corners, (bottom, top) = cell(ix, iy)
            low = [(x, y, bottom) for x, y in corners]
            high = [(x, y, top) for x, y in corners]
            for side in _block_faces(low, high, seen, bottom != top):
                draw_face(screen, side, None, look)


def _block_faces(low: list[Point], high: list[Point], seen: tuple[bool, ...],
                 sided: bool) -> Iterator[list[Point]]:  # fmt: skip
    """A block's faces in ``LegoCartesian``'s order: sides if it has height, bottom, top."""
    for i in range(4 if sided else 0):
        if seen[i]:
            j = (i + 1) % 4
            yield [low[i], low[j], high[j], high[i]]
    if seen[4]:
        yield [low[3], low[2], low[1], low[0]]
    if seen[5]:
        yield high


def surface_cells(screen: MovingScreen, nx: int, ny: int,
                  cell: Callable[[int, int], list[Point]],
                  look: tuple[int, int, int]) -> None:  # fmt: skip
    """``SurfaceCartesian``: each square between four bins' middles, level lines and all.

    The squares are drawn from the front to the back.
    """
    xs, ys = _order(screen.view, nx, ny)
    color, width, _style = look
    for iy in ys:
        for ix in xs:
            points = cell(ix, iy)
            draw_face(screen, points, [p[2] for p in points], look, (color, width, DOTTED))
