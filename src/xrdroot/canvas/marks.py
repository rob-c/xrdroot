"""ROOT's markers and lines, in the canvas's pixels, as ``TImageDump`` draws them.

``TImageDump::DrawPolyMarker`` draws each marker at the whole pixel its
point rounds to, ``8 * fMarkerSize`` pixels across (less for a marker with
a thick outline): a dot is one pixel, 6 and 7 a small and a larger cross of
pixels, 20 (and 8) a filled disc of radius ``int(size / 2)``, 21 a filled
square, 22 and 23 triangles, and so on - each a polygon filled or outlined,
or a few lines. :func:`marker_path` is that shape as a path about the
point's pixel, and :func:`draw_markers` puts one at every point.

A line is drawn between the pixels its ends round to, ``fLineWidth``
pixels wide: straight across or down a line of that many pixels, and
otherwise a square brush that thick moved along it (``TASImage::DrawWideLine``).
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np

__all__ = ["draw_markers", "marker_path", "marker_pixels"]

#: ``kBASEMARKER``: how many pixels across a marker of size 1 is.
BASE = 8
#: The markers drawn as a single pixel: ``TImageDump``'s 1, and 9 to 19.
DOT_STYLES = frozenset({1, *range(9, 20)})

#: Each marker's outline, as fractions of its size about its middle, ``y`` down,
#: and whether it is filled: ``TImageDump::DrawPolyMarker``'s points.
SHAPES: dict[int, tuple[tuple[tuple[float, float], ...], bool]] = {
    21: (((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)), True),
    25: (((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)), False),
    22: (((0.0, -0.5), (0.5, 0.5), (-0.5, 0.5)), True),
    26: (((0.0, -0.5), (0.5, 0.5), (-0.5, 0.5), (0.0, -0.5)), False),
    23: (((-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)), True),
    32: (((-0.5, -0.5), (0.5, -0.5), (0.0, 0.5), (-0.5, -0.5)), False),
    33: (((0.0, -0.5), (1 / 3, 0.0), (0.0, 0.5), (-1 / 3, 0.0)), True),
    27: (((0.0, -0.5), (1 / 3, 0.0), (0.0, 0.5), (-1 / 3, 0.0), (0.0, -0.5)), False),
    34: (((-1 / 6, -1 / 6), (-1 / 6, -0.5), (1 / 6, -0.5), (1 / 6, -1 / 6), (0.5, -1 / 6),
          (0.5, 1 / 6), (1 / 6, 1 / 6), (1 / 6, 0.5), (-1 / 6, 0.5), (-1 / 6, 1 / 6),
          (-0.5, 1 / 6), (-0.5, -1 / 6)), True),
    29: (((0.0, 0.5), (0.112255, 0.15451), (0.47552, 0.15451), (0.181635, -0.05902),
          (0.29389, -0.40451), (0.0, -0.19098), (-0.29389, -0.40451), (-0.181635, -0.05902),
          (-0.47552, 0.15451), (-0.112255, 0.15451)), True),
}  # fmt: skip
#: The outlined versions of filled shapes, where ROOT draws the same points as a line.
SHAPES[28] = (SHAPES[34][0] + SHAPES[34][0][:1], False)
SHAPES[30] = (SHAPES[29][0] + SHAPES[29][0][:1], False)
#: The markers made of lines: plus, cross and asterisk, as pairs of ends.
STROKES: dict[int, tuple[tuple[float, float, float, float], ...]] = {
    2: ((-0.5, 0.0, 0.5, 0.0), (0.0, -0.5, 0.0, 0.5)),
    5: ((-0.3535, -0.3535, 0.3535, 0.3535), (-0.3535, 0.3535, 0.3535, -0.3535)),
    3: ((-0.5, 0.0, 0.5, 0.0), (0.0, -0.5, 0.0, 0.5),
        (-0.3535, -0.3535, 0.3535, 0.3535), (-0.3535, 0.3535, 0.3535, -0.3535)),
}  # fmt: skip
STROKES[31] = STROKES[3]


def base_style(style: int) -> int:
    """``TAttMarker::GetMarkerStyleBase``, then ``TImageDump``'s own folding of the styles."""
    style = abs(int(style))
    base = style % 1000 if style >= 1000 else style
    return {4: 24, 8: 20}.get(base, 1 if base in DOT_STYLES else base)


def line_width(style: int) -> int:
    """``TAttMarker::GetMarkerLineWidth``: the outline's width, set by styles over 1000."""
    style = abs(int(style))
    return max(style // 1000, 1) if style >= 1000 else 1


def marker_pixels(style: int, size: float) -> float:
    """How many pixels across ``TImageDump`` draws a marker."""
    base = base_style(style)
    pixels = (float(size) - math.floor(line_width(style) / 2) / 4) * BASE
    return pixels * {6: 0.2, 7: 0.3}.get(base, 1.0)


def _dots(base: int) -> list[tuple[int, int]]:
    """The pixels of the dot markers, about the point: 1 alone, 6 a plus of five, 7 nine."""
    plus = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
    corners = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    return {6: plus, 7: plus + corners}.get(base, [(0, 0)])


@functools.lru_cache(maxsize=256)
def marker_path(style: int, size: float) -> tuple[Any, bool, int]:
    """A marker as a path in pixels about the middle of its point's pixel, ``y`` down.

    Returns the path, whether it is filled, and its outline's width.
    """
    from matplotlib.path import Path

    base, m = base_style(style), marker_pixels(style, size)
    if base in (1, 6, 7):
        squares = [Path.unit_rectangle().transformed(_shift(dx - 0.5, dy - 0.5)) for dx, dy in _dots(base)]
        return Path.make_compound_path(*squares), True, 0
    if base in (20, 24):
        radius = int(m / 2)
        circle = Path.circle((0.0, 0.0), radius + (0.5 if base == 20 else 0.0))
        return circle, base == 20, line_width(style)
    if base in STROKES:
        lines = [Path([(math.floor(x1 * m), math.floor(y1 * m)), (math.floor(x2 * m), math.floor(y2 * m))])
                 for x1, y1, x2, y2 in STROKES[base]]  # fmt: skip
        return Path.make_compound_path(*lines), False, line_width(style)
    points, filled = SHAPES.get(base, SHAPES[21])
    corners = [(math.floor(x * m), math.floor(y * m)) for x, y in points]
    return Path(corners, closed=filled), filled, line_width(style)


def _shift(dx: float, dy: float) -> Any:
    from matplotlib.transforms import Affine2D

    return Affine2D().translate(dx, dy)


def draw_markers(scene: Any, pixels: np.ndarray[Any, Any], style: int, size: float, color: Any) -> None:
    """A marker at each of ``pixels`` - the canvas's, ``y`` down, rounded - as ``TImageDump`` draws them."""
    from matplotlib.collections import PathCollection
    from matplotlib.transforms import Affine2D

    if not len(pixels):
        return
    path, filled, width = marker_path(int(style), float(size))
    scale = Affine2D().scale(1 / 100, 1 / 100) + scene.figure.dpi_scale_trans
    points = np.rint(np.asarray(pixels, dtype=float)) + 0.5
    collection = PathCollection(
        [path.transformed(Affine2D().scale(1, -1))], offsets=points, transform=scale,
        facecolors=[color] if filled else ["none"], edgecolors=["none"] if filled else [color],
        linewidths=[0.72 * max(width, 1)], zorder=scene.layer(), clip_on=True,
    )  # fmt: skip
    if hasattr(collection, "set_offset_transform"):
        collection.set_offset_transform(scene.display)
    else:  # pragma: no cover - matplotlib before 3.6 keeps it unexposed
        collection._transOffset = scene.display
    collection.set_clip_box(scene.ax.bbox)
    collection.set_gid(MARKER_GID)
    scene.ax.add_collection(collection, autolim=False)


#: What marks a marker's collection, which is placed at pixels' middles and never snapped.
MARKER_GID = "root-markers"
