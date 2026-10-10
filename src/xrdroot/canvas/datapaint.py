"""A picture's layers painted on a pad as ROOT's painters put them in pixels.

:mod:`xrdroot.plot` says what a histogram or a graph is drawn as - steps,
points with bars, a line, a band - and this draws each the way
``THistPainter`` and ``TGraphPainter`` draw it through ``TImageDump``: lines
``fLineWidth`` pixels wide, dashed as ``gStyle``'s line styles are, clipped
to the frame; a histogram's outline without the drops at the frame's edges
and never below its bottom; a graph's error bars from the edge of its
marker (``TGraphPainter::PaintGraphAsymmErrors``), with ends two pixels
either side (``gStyle``'s ``EndErrorSize``) where they are not clipped;
markers ROOT's own shapes (:mod:`.marks`). Everything is placed in the
canvas's pixels, which is where ROOT rounds it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..plot.model import Area, Band, Bars, Boxes, Curve, Look, Points, Steps
from . import styles
from .marks import draw_markers
from .raster import add_line, frame_clip
from .scene import Scene

__all__ = ["PAINTED", "pixels_of"]

#: ``gStyle``'s ``fEndErrorSize``: the pixels an error bar's end reaches either side.
END_ERROR = 2
#: ``TGraphPainter``'s ``cxx``: how much of a marker, from style 20 on, an error bar
#: leaves out across.
GAP_X = (1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.6, 1.0, 0.5, 0.5, 1.0, 0.5, 0.6, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0)  # fmt: skip
#: ``TGraphPainter``'s ``cyy``: the same, up.
GAP_Y = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0)  # fmt: skip
#: The fill style an area, band or box is drawn in when it has none: solid.
SOLID = 1001


def pixels_of(scene: Scene, xs: Any, ys: Any) -> np.ndarray[Any, Any]:
    """Points in the axes' units as the canvas's pixels, ``y`` down, unrounded."""
    points = np.column_stack([np.asarray(xs, float), np.asarray(ys, float)])
    shown = scene.ax.transData.transform(points)
    shown[:, 1] = scene.canvas[1] - shown[:, 1]
    return shown


def _clipped(scene: Scene, pixels: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Points far outside the pad brought in along their lines.

    Without it a line to a point millions of pixels away would be rasterised
    all the way there.
    """
    limit = 4.0 * max(scene.canvas)
    return np.clip(pixels, -limit, limit)


def polyline(scene: Scene, pixels: np.ndarray[Any, Any], look: Look, clip: bool = True) -> None:
    """A line through ``pixels``, each rounded to the whole pixel ``TImageDump`` puts it at."""
    finite = pixels[np.all(np.isfinite(pixels), axis=1)]
    if len(finite) < 2 or not look.color or look.width <= 0:
        return
    add_line(scene, _clipped(scene, finite), look.color, round(look.width), look.line_style,
             frame_clip(scene) if clip else None)  # fmt: skip


def _fill(scene: Scene, pixels: np.ndarray[Any, Any], look: Look) -> None:
    """A filled area through ``pixels``, clipped to the frame, hatched as its fill style says."""
    from matplotlib.patches import Polygon

    fills, hatch, alpha = styles.fill(look.fill_style or SOLID)
    if not fills or look.fill is None:
        return
    face = look.fill if hatch is None else "none"
    polygon = Polygon(np.rint(pixels), closed=True, transform=scene.display,
                      zorder=scene.layer(), facecolor=face, edgecolor="none", linewidth=0.0,
                      alpha=look.alpha * alpha)  # fmt: skip
    if hatch is not None:
        polygon.set_hatch(hatch)
        polygon.set_edgecolor(look.fill)
    scene.ax.add_artist(polygon)
    polygon.set_clip_path(scene.ax.patch)
    if scene.shading is not None:
        from .gradient import shade

        shade(scene, polygon, scene.shading)


# -- a histogram's outline ---------------------------------------------------------------------


def _outline(scene: Scene, layer: Steps) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """``PaintGrapHist``'s ``H``: the steps, with drops to the baseline only inside the frame."""
    (xmin, xmax), (ymin, ymax) = scene.ax.get_xlim(), scene.ax.get_ylim()
    base = min(max(0.0, ymin), ymax) if layer.baseline is None else float(np.min(layer.baseline))
    edges = np.asarray(layer.edges, float)
    values = np.maximum(np.asarray(layer.values, float), ymin)
    xs = np.repeat(edges, 2)[1:-1]
    ys = np.repeat(values, 2)
    if edges[0] > xmin:
        xs, ys = np.concatenate([[edges[0]], xs]), np.concatenate([[base], ys])
    if edges[-1] < xmax:
        xs, ys = np.concatenate([xs, [edges[-1]]]), np.concatenate([ys, [base]])
    return xs, ys


def paint_steps(scene: Scene, layer: Steps) -> None:
    """A histogram's fill and outline; a stacked one's fill reaches down only to the one below.

    ``THStack`` paints the sum of all its histograms first and each lower
    sum over it, so each fill shows from the one below it to its own top:
    the area between the two steps, drawn here in the order they are added.
    """
    xs, ys = _outline(scene, layer)
    if layer.look.fill is not None:
        (_, _), (ymin, ymax) = scene.ax.get_xlim(), scene.ax.get_ylim()
        edges = np.asarray(layer.edges, float)
        steps_x = np.repeat(edges, 2)[1:-1]
        low = np.full(len(layer.values), min(max(0.0, ymin), ymax))
        if layer.baseline is not None:
            low = np.maximum(np.asarray(layer.baseline, float) * np.ones(len(low)), ymin)
        area_x = np.concatenate([steps_x, steps_x[::-1]])
        tops = np.repeat(np.asarray(layer.values, float), 2)
        area_y = np.concatenate([tops, np.repeat(low, 2)[::-1]])
        _fill(scene, pixels_of(scene, area_x, area_y), layer.look)
    polyline(scene, pixels_of(scene, xs, ys), layer.look)


def paint_curve(scene: Scene, layer: Curve) -> None:
    """A graph's or function's line through its points, clipped to the frame."""
    polyline(scene, pixels_of(scene, layer.x, layer.y), layer.look)


def _solid(look: Look) -> Look:
    """``look``, filled solid when it has no fill style of its own."""
    return look._replace(fill_style=look.fill_style or SOLID)


def paint_area(scene: Scene, layer: Area) -> None:
    """A graph's ``F``: the polygon through its points, filled in its fill attributes."""
    _fill(scene, pixels_of(scene, layer.x, layer.y), _solid(layer.look))


def paint_band(scene: Scene, layer: Band) -> None:
    """A band of errors: out along its tops and back along its bottoms, filled."""
    xs = np.concatenate([np.asarray(layer.x, float), np.asarray(layer.x, float)[::-1]])
    ys = np.concatenate([np.asarray(layer.high, float), np.asarray(layer.low, float)[::-1]])
    _fill(scene, pixels_of(scene, xs, ys), _solid(layer.look))


def _rectangles(scene: Scene, x0: Any, x1: Any, y0: Any, y1: Any, look: Look) -> None:
    """Each rectangle filled if ``look`` has a fill, or else outlined as a closed line."""
    columns = (np.asarray(x0, float), np.asarray(x1, float),
               np.asarray(y0, float), np.asarray(y1, float))  # fmt: skip
    for a, b, c, d in zip(*columns, strict=False):
        corners = pixels_of(scene, [a, b, b, a], [c, c, d, d])
        if look.fill is not None:
            _fill(scene, corners, _solid(look))
        else:
            polyline(scene, np.vstack([corners, corners[:1]]), look)


def paint_boxes(scene: Scene, layer: Boxes) -> None:
    """Boxes of errors: a rectangle round each point's, filled or outlined."""
    _rectangles(scene, layer.x0, layer.x1, layer.y0, layer.y1, layer.look)


def paint_bars(scene: Scene, layer: Bars) -> None:
    """Bars: a rectangle from zero up to each value, between its left and right."""
    base = np.zeros_like(np.asarray(layer.values, float))
    _rectangles(scene, layer.left, layer.right, base, layer.values, layer.look)


# -- points and their error bars -------------------------------------------------------------


def _gaps(look: Look) -> tuple[float, float]:
    """How many pixels of a bar the marker covers, across and up, as ``cxx`` and ``cyy`` say."""
    style = abs(int(look.marker_style)) % 1000 if look.marker is not None else 1
    half = int(0.5 * float(look.marker_size) * 8)
    if 20 <= style <= 49:
        return half * GAP_X[style - 20], half * GAP_Y[style - 20]
    return 0.0, 0.0


def _bar(start: float, end: float, limit: float, beyond: bool) -> tuple[float, bool] | None:
    """One arm of an error bar, from the marker's edge to its end or the frame.

    Returns where the arm stops and whether the frame cut it short, or
    nothing when the marker covers all of it.
    """
    clipped = end < limit if beyond else end > limit
    stop = limit if clipped else end
    if (stop > start) if beyond else (stop < start):
        return None
    return stop, clipped


def _arms(centre: tuple[float, float], ends: tuple[float, float, float, float],
          frame: tuple[float, ...], gaps: tuple[float, float],
          caps: bool) -> list[tuple[float, float, float, float]]:  # fmt: skip
    """The segments of one point's error bars, in pixels: up, down, left, right, and their ends."""
    px, py = centre
    up, down, left, right = ends
    top, bottom, first, last = frame
    segments = []
    for start, end, limit, beyond, vertical in (
        (py - gaps[1], up, top, True, True), (py + gaps[1], down, bottom, False, True),
        (px - gaps[0], left, first, True, False), (px + gaps[0], right, last, False, False),
    ):  # fmt: skip
        if end == (py if vertical else px):
            continue
        found = _bar(start, end, limit, beyond)
        if found is None:
            continue
        stop, clipped = found
        segments.append((px, start, px, stop) if vertical else (start, py, stop, py))
        if caps and not clipped:
            e = END_ERROR
            cap = (px - e, stop, px + e, stop) if vertical else (stop, py - e, stop, py + e)
            segments.append(cap)
    return segments


def paint_points(scene: Scene, layer: Points) -> None:
    """``TGraphPainter::PaintGraphAsymmErrors``: each point in the frame's bars, then its marker."""
    (xmin, xmax), (ymin, ymax) = scene.ax.get_xlim(), scene.ax.get_ylim()
    x, y = np.asarray(layer.x, float), np.asarray(layer.y, float)
    inside = ((x >= min(xmin, xmax)) & (x <= max(xmin, xmax))
              & (y >= min(ymin, ymax)) & (y <= max(ymin, ymax)))  # fmt: skip
    centre = pixels_of(scene, x, y)
    ups = pixels_of(scene, x, y + np.asarray(layer.yhigh, float))[:, 1]
    downs = pixels_of(scene, x, y - np.asarray(layer.ylow, float))[:, 1]
    lefts = pixels_of(scene, x - np.asarray(layer.xlow, float), y)[:, 0]
    rights = pixels_of(scene, x + np.asarray(layer.xhigh, float), y)[:, 0]
    corner, far = pixels_of(scene, [xmin, xmax], [ymin, ymax])
    frame = (far[1], corner[1], corner[0], far[0])
    gaps = _gaps(layer.look)
    segments = []
    for index in np.flatnonzero(inside):
        ends = (ups[index], downs[index], lefts[index], rights[index])
        segments += _arms(tuple(centre[index]), ends, frame, gaps, layer.caps)
    look = layer.look
    if segments and look.color:
        clip = frame_clip(scene)
        for x1, y1, x2, y2 in segments:
            add_line(scene, [(x1, y1), (x2, y2)], look.color, round(look.width), 1, clip)
    if look.marker is not None:
        draw_markers(scene, centre[inside], look.marker_style, look.marker_size, look.marker_color)


#: The layers painted here, by their kind; the rest are drawn by :mod:`xrdroot.plot`.
PAINTED = {
    Steps: paint_steps, Curve: paint_curve, Band: paint_band, Boxes: paint_boxes,
    Bars: paint_bars, Area: paint_area, Points: paint_points,
}  # fmt: skip
