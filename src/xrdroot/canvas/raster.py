"""Lines as ``TASImage`` sets their pixels: which pixels a ROOT line is made of.

A line of a picture ROOT saves is drawn by ``TASImage``: straight across or
down, ``DrawHLine`` and ``DrawVLine`` fill every pixel from one end to the
other, as many rows (or columns) as the line is thick, starting half its
thickness above (or left of) it; any other one-pixel line is Bresenham's,
from its first pixel up to but not including its last (``DrawLineInternal``),
and a thicker one a square brush as wide as the line stamped along it
(``DrawWideLine``). A dashed line (``DrawDashLine``) is dashed afresh along
each segment, the dashes shortened by the cosine of its slope.

:func:`segment_pixels` and :func:`polyline_pixels` are those rules, over
whole pixels; :class:`PixelLine` is a matplotlib artist that fills exactly
those pixels when it is rasterised, and strokes the line as any other in a
PDF or SVG.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

__all__ = ["PixelLine", "polyline_pixels", "segment_pixels"]

#: Pixels as ``TASImage`` sets them: their columns, and their rows.
Pixels = tuple[list[int], list[int]]


def _half(thick: int) -> int:
    """How far above (or left of) a straight line ``DrawHLine`` (or ``DrawVLine``) starts."""
    return thick // 2 if thick > 1 else 0


def _span(a: int, b: int) -> list[int]:
    """Every whole pixel from the lower of ``a`` and ``b`` to the higher, both included."""
    return list(range(min(a, b), max(a, b) + 1))


def _rows(xs: Sequence[int], top: int, thick: int) -> Pixels:
    """``DrawHLine``'s pixels: the run ``xs`` once in each of ``thick`` rows from ``top`` down."""
    return [x for _ in range(thick) for x in xs], [top + w for w in range(thick) for _ in xs]


def _columns(ys: Sequence[int], left: int, thick: int) -> Pixels:
    """``DrawVLine``'s pixels: each of ``ys`` across ``thick`` columns from ``left``."""
    return [left + w for _ in ys for w in range(thick)], [y for y in ys for _ in range(thick)]


def _straight(x1: int, y1: int, x2: int, y2: int, thick: int) -> Pixels:
    """``DrawHLine`` or ``DrawVLine``: every pixel between the ends, ``thick`` of them across."""
    half = _half(thick)
    if y1 == y2:
        return _rows(_span(x1, x2), y1 - half, thick)
    return _columns(_span(y1, y2), x1 - half, thick)


def _bresenham(x1: int, y1: int, x2: int, y2: int, last: bool = False) -> Pixels:
    """``DrawLineInternal``'s one-pixel line: its first pixel on, its last off unless ``last``."""
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    steep = dy > dx
    if steep:
        x1, y1, x2, y2, dx, dy = y1, x1, y2, x2, dy, dx
    if x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    step = 1 if y2 > y1 else -1
    d, xs, ys, y = 2 * dy - dx, [x1], [y1], y1
    for x in range(x1, x2 + (1 if last else 0)):
        xs.append(x)
        ys.append(y)
        if d >= 0:
            y += step
            d += 2 * (dy - dx)
        else:
            d += 2 * dy
    return (ys, xs) if steep else (xs, ys)


def _brushed(x1: int, y1: int, x2: int, y2: int, thick: int) -> Pixels:
    """``DrawWideLine``: a ``thick`` square brush, centred on each pixel of the line.

    A line that goes nowhere - a dash rounded to no length - libAfterImage's
    ``asim_line_to`` does not draw at all.
    """
    if (x1, y1) == (x2, y2):
        return [], []
    xs, ys = _bresenham(x1, y1, x2, y2, last=True)
    offsets = range(-(thick // 2), thick - thick // 2)
    return ([x + dx for x in xs for dx in offsets for _ in offsets],
            [y + dy for y in ys for _ in offsets for dy in offsets])  # fmt: skip


def segment_pixels(x1: int, y1: int, x2: int, y2: int, thick: int = 1) -> Pixels:
    """The pixels of one line from ``(x1, y1)`` to ``(x2, y2)``, ``thick`` wide."""
    thick = max(int(thick), 1)
    if x1 == x2 and y1 == y2:
        return [], []
    if x1 == x2 or y1 == y2:
        return _straight(x1, y1, x2, y2, thick)
    if thick > 1:
        return _brushed(x1, y1, x2, y2, thick)
    return _bresenham(x1, y1, x2, y2)


def _dash_states(dashes: Sequence[int]) -> Iterator[bool]:
    """Pixel by pixel along a dashed line, whether it is in a dash (not a gap).

    As ``DrawDashHLine`` counts: a dash or gap ends once its length of
    pixels is counted, and after the last gap the first dash begins again.
    """
    dash, count = 0, 0
    while True:
        yield dash % 2 == 0
        count += 1
        if count >= dashes[dash]:
            dash, count = dash + 1, 0
        if dash >= len(dashes):
            dash, count = 0, 0


def _dashed_straight(x1: int, y1: int, x2: int, y2: int, thick: int,
                     dashes: tuple[int, ...]) -> Pixels:  # fmt: skip
    """``DrawDashHLine`` and ``DrawDashVLine``: along the line, on and off by the dash lengths."""
    across = y1 == y2
    ends = sorted((x1, x2)) if across else sorted((y1, y2))
    half = _half(thick)
    along = zip(range(ends[0], ends[1] + 1), _dash_states(dashes), strict=False)
    kept = [at for at, on in along if on]
    if across:
        top = y1 - half
        rows = [top + w for _ in kept for w in range(thick)]
        return [at for at in kept for _ in range(thick)], rows
    return _columns(kept, x1 - half, thick)


def _dashed_slanted(x1: int, y1: int, x2: int, y2: int, dashes: tuple[int, ...]) -> Pixels:
    """``DrawDashZLine``: Bresenham's pixels, on and off by dashes shortened by the slope."""
    shrink = math.cos(math.atan2(abs(y2 - y1), abs(x2 - x1)))
    shortened = [max(round(length * shrink), 0) for length in dashes]
    xs, ys = _bresenham(x1, y1, x2, y2)
    along = zip(xs[1:], ys[1:], _dash_states(shortened), strict=False)
    kept = [(x, y) for x, y, on in along if on]
    return [xs[0]] + [x for x, _ in kept], [ys[0]] + [y for _, y in kept]


def _dash_steps(angle: float, dashes: tuple[int, ...]) -> list[tuple[float, float]]:
    """``DrawDashZTLine``'s steps along the slope: dashes at half their length, gaps at twice."""
    steps = [(length * math.cos(angle), length * math.sin(angle)) for length in dashes]
    return [(dx / 2, dy / 2) if index % 2 == 0 else (dx * 2, dy * 2)
            for index, (dx, dy) in enumerate(steps)]  # fmt: skip


def _left_to_right(x1: int, y1: int, x2: int, y2: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """A line's ends, the left one first, as ``DrawDashZTLine`` walks it."""
    return ((x1, y1), (x2, y2)) if x1 <= x2 else ((x2, y2), (x1, y1))


def _short_of(fx: float, fy: float, end: tuple[int, int], sign: int) -> bool:
    """Whether ``DrawDashZTLine``'s walk is still short of the line's end, across and up or down."""
    return fx < end[0] and (fy < end[1] if sign > 0 else fy > end[1])


def _dashed_thick(x1: int, y1: int, x2: int, y2: int, thick: int,
                  dashes: tuple[int, ...]) -> Pixels:  # fmt: skip
    """``DrawDashZTLine``: wide dashes stamped along the slope, a step at a time."""
    steps = _dash_steps(math.atan2(abs(y2 - y1), abs(x2 - x1)), dashes)
    (x, y), end = _left_to_right(x1, y1, x2, y2)
    sign = 1 if end[1] > y else -1
    fx, fy, x0, y0 = float(x), float(y), float(x), float(y)
    xs: list[int] = []
    ys: list[int] = []
    dash = 0
    while _short_of(fx, fy, end, sign):
        fx, fy = fx + steps[dash][0], fy + sign * steps[dash][1]
        if dash % 2 == 0:
            more = _brushed(round(x0), round(y0), round(fx), round(fy), thick)
            xs, ys = xs + more[0], ys + more[1]
        else:
            x0, y0 = fx, fy
        dash = (dash + 1) % len(dashes)
    return xs, ys


def dashed_pixels(x1: int, y1: int, x2: int, y2: int, thick: int,
                  dashes: tuple[int, ...]) -> Pixels:  # fmt: skip
    """``DrawDashLine``: one dashed segment, as straight, thin or thick."""
    thick = max(int(thick), 1)
    if x1 == x2 or y1 == y2:
        return _dashed_straight(x1, y1, x2, y2, thick, dashes)
    if thick < 2:
        return _dashed_slanted(x1, y1, x2, y2, dashes)
    return _dashed_thick(x1, y1, x2, y2, thick, dashes)


def polyline_pixels(points: np.ndarray[Any, Any], thick: int = 1,
                    dashes: tuple[int, ...] = ()) -> np.ndarray[Any, Any]:  # fmt: skip
    """Every pixel of a line through whole-pixel ``points``, each segment as ROOT draws one."""
    xs: list[int] = []
    ys: list[int] = []
    whole = np.asarray(points, dtype=np.int64)
    for (x1, y1), (x2, y2) in zip(whole[:-1].tolist(), whole[1:].tolist(), strict=False):
        if dashes and len(dashes) % 2 == 0:
            more = dashed_pixels(x1, y1, x2, y2, thick, dashes)
        else:
            more = segment_pixels(x1, y1, x2, y2, thick)
        xs += more[0]
        ys += more[1]
    if not xs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.column_stack([xs, ys]), axis=0)


def root_dashes(style: Any) -> tuple[int, ...]:
    """``TImageDump``'s dashes for ``fLineStyle``: a quarter of each of ``gStyle``'s lengths.

    A solid line has none.
    """
    from .styles import LINE_STYLES

    return tuple(int(length) // 4 for length in LINE_STYLES.get(int(style), ()))


def add_line(scene: Any, points: Any, color: Any, thick: Any = 1, style: Any = 1,
             clip: Any = None, many: bool = False) -> PixelLine:  # fmt: skip
    """A :class:`PixelLine` through canvas ``points`` (or, ``many``, each of a list), on top."""
    line = PixelLine(scene, points, color, int(thick), root_dashes(style), clip, many)
    line.set_zorder(scene.layer())
    scene.ax.add_artist(line)
    line.set_clip_on(False)
    return line


def frame_clip(scene: Any) -> tuple[int, int, int, int]:
    """The frame's pixels, inclusive: what a graph or histogram is clipped to."""
    left, right, bottom, top = scene.pad.margins
    x0, y1 = scene.pixel(left, bottom)
    x1, y0 = scene.pixel(1 - right, 1 - top)
    return round(x0), round(y0), round(x1), round(y1)


def _runs(pixels: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Pixels as runs along their rows.

    Each run is the ``x`` of its first pixel, one past its last, and its row.
    """
    order = np.lexsort((pixels[:, 0], pixels[:, 1]))
    xs, ys = pixels[order, 0], pixels[order, 1]
    starts = np.flatnonzero(np.r_[True, (np.diff(ys) != 0) | (np.diff(xs) != 1)])
    ends = np.r_[starts[1:], len(xs)] - 1
    return np.column_stack([xs[starts], xs[ends] + 1, ys[starts]])


def _rectangles(runs: np.ndarray[Any, Any], height: float, scale: float) -> Any:
    """Runs of pixels as one path of rectangles in a raster's display, ``y`` up."""
    from matplotlib.path import Path

    x0, x1, row = (runs[:, i].astype(float) * scale for i in range(3))
    top, bottom = height - row, height - row - scale
    round_it = ((x0, bottom), (x1, bottom), (x1, top), (x0, top), (x0, bottom))
    corners = np.stack([np.column_stack(pair) for pair in round_it], axis=1)
    codes = np.tile([Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY], len(runs))
    return Path(corners.reshape(-1, 2), codes)


def _artist_base() -> Any:
    """matplotlib's ``Artist``, imported only once a line is drawn, as the rest of matplotlib is."""
    from matplotlib.artist import Artist

    return Artist


class PixelLine(_artist_base()):  # type: ignore[misc]
    """ROOT lines - polylines of whole pixels - drawn pixel for pixel in a raster.

    ``points`` are the canvas's pixels, ``y`` down, as ``TImageDump`` rounds
    them: one polyline, or with ``many`` a list of them; ``clip`` is the box
    of pixels, inclusive, that is drawn in. Vector output strokes the same
    lines ``thick`` pixels wide.
    """

    def __init__(self, scene: Any, points: Any, color: Any, thick: int = 1,
                 dashes: tuple[int, ...] = (), clip: tuple[int, int, int, int] | None = None,
                 many: bool = False) -> None:  # fmt: skip
        super().__init__()
        lines = points if many else [points]
        self.lines = [np.rint(np.asarray(line, dtype=float)).astype(np.int64) for line in lines]
        self.lines = [line for line in self.lines if len(line) >= 2]
        self.color = color
        self.thick = max(int(thick), 1)
        self.dashes = tuple(dashes)
        self.pixel_clip = clip
        self.canvas_size = scene.canvas
        self.to_figure = scene.display

    @property
    def points(self) -> np.ndarray[Any, Any]:
        """The first polyline's points."""
        return self.lines[0] if self.lines else np.zeros((0, 2), dtype=np.int64)

    def pixels(self) -> np.ndarray[Any, Any]:
        """Every pixel the lines set, inside their clip."""
        parts = [polyline_pixels(line, self.thick, self.dashes) for line in self.lines]
        none = np.zeros((0, 2), dtype=np.int64)
        found = np.unique(np.concatenate(parts), axis=0) if parts else none
        if self.pixel_clip is not None and len(found):
            x0, y0, x1, y1 = self.pixel_clip
            across, down = found[:, 0], found[:, 1]
            found = found[(across >= x0) & (across <= x1) & (down >= y0) & (down <= y1)]
        return found

    def draw(self, renderer: Any) -> None:
        """Pixel for pixel in a raster (Agg), and stroked in anything else."""
        if not self.get_visible() or not self.lines:
            return
        from matplotlib.backends.backend_agg import RendererAgg

        if isinstance(renderer, RendererAgg):
            self._draw_pixels(renderer)
        else:
            self._draw_stroke(renderer)

    def _draw_pixels(self, renderer: Any) -> None:
        """The pixels filled as rectangles, unantialiased, so no pixel is shaded but a set one."""
        from matplotlib.transforms import IdentityTransform

        found = self.pixels()
        if not len(found):
            return
        from matplotlib.colors import to_rgba

        scale = renderer.width / self.canvas_size[0]
        gc = renderer.new_gc()
        gc.set_foreground(self.color)
        gc.set_antialiased(False)
        gc.set_linewidth(0.0)
        path = _rectangles(_runs(found), renderer.height, scale)
        renderer.draw_path(gc, path, IdentityTransform(), to_rgba(self.color))
        gc.restore()

    def _draw_stroke(self, renderer: Any) -> None:
        """The lines stroked through the middles of their pixels, as wide and dashed as ROOT's."""
        from matplotlib.path import Path

        gc = renderer.new_gc()
        gc.set_foreground(self.color)
        gc.set_linewidth(0.72 * self.thick)
        if self.dashes:
            gc.set_dashes(0, [0.72 * length for length in self.dashes])
        for line in self.lines:
            renderer.draw_path(gc, Path(line.astype(float) + 0.5), self.to_figure, None)
        gc.restore()
