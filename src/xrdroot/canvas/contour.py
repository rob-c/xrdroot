"""``THistPainter::PaintContour``'s lines: ``CONT1``, ``CONT2`` and ``CONT3``, as ROOT finds them.

ROOT cuts the grid of bin centres into cells, four neighbouring centres
each, and walks round each cell twice from its lowest corner - once each
way - noting where each edge crosses a level (``PaintContourLine``); the
crossings are then paired level by level into segments, and each segment
is painted on its own, clipped to the frame. The levels are
``TH1::SetContour``'s: ``gStyle``'s twenty, evenly from the lowest content
to the highest. ``CONT1`` draws each level in its colour of the palette,
``CONT2`` in one of the five line styles by that colour, and ``CONT3``
every level in the histogram's own line.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Any, NamedTuple

import numpy as np

from .raster import add_line, frame_clip
from .scene import Scene

__all__ = ["contour_segments", "paint_contour_lines"]

#: ``kMAXCONTOUR``: the most crossings a cell's edge may have noted.
MAX_CONTOUR = 404
#: How many times a cell's crossings may be turned about to pair them, before it is let go.
REORDERS = 100


class Crossing(NamedTuple):
    """Where an edge of a cell crosses a level, and which level."""

    x: float
    y: float
    level: int


def levels_of(values: np.ndarray[Any, Any], count: int) -> list[float]:
    """``TH1::SetContour``: ``count`` levels, evenly from the lowest content, the highest above."""
    low, high = float(np.min(values)), float(np.max(values))
    if low == high and low != 0:
        low, high = low - 0.01 * abs(low), high + 0.01 * abs(high)
    step = (high - low) / count
    return [low + step * index for index in range(count)]


def _edge(ends: tuple[Any, Any], levels: list[float], into: list[Crossing], at: int) -> int:
    """``PaintContourLine``: the levels an edge rises through, every other place from ``at``."""
    (e1, c1, x1, y1), (e2, c2, x2, y2) = ends
    upright = x1 == x2
    length = y2 - y1 if upright else x2 - x1
    rise = e2 - e1
    level, place, count = c1 + 1, 0, 0
    while level <= c2 and place <= MAX_CONTOUR // 2 - 3:
        along = length * ((levels[level] - e1) / rise)
        point = (x1, y1 + along) if upright else (x1 + along, y1)
        into[at + place] = Crossing(point[0], point[1], level)
        count, place, level = count + 1, place + 2, level + 1
    return count


def _lowest(z: list[float]) -> int:
    """The cell's lowest corner, counting from 1 as ROOT does."""
    n = 0 if z[0] <= z[1] else 1
    m = 2 if z[2] <= z[3] else 3
    return (m if z[n] > z[m] else n) + 1


def _walk(corners: list[Any], levels: list[float], into: list[Crossing], back: bool) -> int:
    """Round the cell from its lowest corner, one way or the other: how far ``into`` is filled."""
    n = _lowest([corner[0] for corner in corners])
    filled = 2 if back else 1
    for _ in range(4):
        m = (4 if n == 1 else n - 1) if back else n % 4 + 1
        filled += 2 * _edge((corners[n - 1], corners[m - 1]), levels, into, filled - 1)
        n = m
    return filled


def _paired(into: list[Crossing], filled: int) -> bool:
    """The crossings turned about until each pair is of one level; whether that was done."""
    turned = 0
    for ix in range(1, filled - 4, 2):
        while into[ix - 1].level != into[ix].level:
            kept = into[ix]
            for jx in range(ix, filled - 4, 2):
                into[jx] = into[jx + 2]
            into[filled - 3] = kept
            if turned > REORDERS:
                break
            turned += 1
    return turned <= REORDERS


def _cell(corners: list[Any], levels: list[float], into: list[Crossing]) -> list[Any]:
    """One cell's segments, each a pair of crossings of one level."""
    _walk(corners, levels, into, back=False)
    filled = _walk(corners, levels, into, back=True)
    if not _paired(into, filled):
        return []
    return [(into[ix - 1], into[ix]) for ix in range(1, filled - 1, 2)]


def contour_segments(xs: Any, ys: Any, values: np.ndarray[Any, Any],
                     levels: list[float]) -> list[Any]:  # fmt: skip
    """Every segment ROOT paints: the cells of neighbouring centres, the lowest ``y`` first."""
    into = [Crossing(0.0, 0.0, 0)] * (2 * MAX_CONTOUR)
    made: list[Any] = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            z = [float(values[i, j]), float(values[i + 1, j]),
                 float(values[i + 1, j + 1]), float(values[i, j + 1])]  # fmt: skip
            ranks = [bisect_right(levels, one) - 1 for one in z]
            if len(set(ranks)) == 1:
                continue
            across = (xs[i], xs[i + 1], xs[i + 1], xs[i])
            up = (ys[j], ys[j], ys[j + 1], ys[j + 1])
            corners = [(z[k], ranks[k], float(across[k]), float(up[k])) for k in range(4)]
            made += _cell(corners, levels, into)
    return made


def _look_of(scene: Scene, layer: Any, level: int, count: int) -> tuple[Any, int, int]:
    """A level's colour, width and style: the palette's for ``CONT1``, a style for ``CONT2``."""
    look = layer.look
    color, width, style = look.color, round(look.width), int(look.line_style or 1)
    if layer.mode in (11, 12):
        colours = scene.colors.colormap()
        shade = int((level + 0.99) * colours.N / count)
        if layer.mode == 11:
            color = colours(shade)[:3]
        else:
            style = shade % 5 or 5
    return color, width, style


def paint_contour_lines(scene: Scene, layer: Any) -> None:
    """``CONT1``, ``CONT2`` or ``CONT3``: each level's segments, clipped to the frame."""
    levels = levels_of(np.asarray(layer.values, float), int(layer.levels))
    from .datapaint import pixels_of

    grouped: dict[tuple[Any, int, int], list[Any]] = {}
    for start, end in contour_segments(layer.x, layer.y, np.asarray(layer.values, float), levels):
        pixels = pixels_of(scene, [start.x, end.x], [start.y, end.y])
        key = _look_of(scene, layer, start.level, len(levels))
        grouped.setdefault(key, []).append(pixels)
    for (color, width, style), lines in grouped.items():
        add_line(scene, lines, color, width, style, frame_clip(scene), many=True)
