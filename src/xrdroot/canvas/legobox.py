"""``THistPainter::PaintH3BoxRaster``: a three-dimensional histogram as a box per bin.

A ``TH3`` drawn ``LEGO`` (or ``BOX``) is a box in each bin, centred on it,
as big as the cube root of its content's share of the highest - so the
fullest bin's box half fills its bin - seen from the pad's ``fTheta`` and
``fPhi`` through ``TView3D``. The bins are drawn from the front to the
back, each box's faces that face the eye outlined where nothing drawn
before hides them, then covered (:mod:`.raster3d`); the box round them all
has its back walls lined at the z axis's divisions where the boxes leave
them in sight, and its front edges drawn over everything, with its axes.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..limits import optimize
from .frame import shown_bins
from .lego import Segment
from .legofaces import BACK, FRONT, box_corners, level_lines
from .legoplot import PHI, THETA, _axes, _draw_segments, _Pad
from .model import lookup
from .raster import add_line
from .raster3d import RasterScreen
from .scene import Scene
from .view3d import View3D

__all__ = ["paint_h3_boxes"]

#: A box's corners, from its centre in half-widths, as ``PaintH3BoxRaster`` numbers them.
CUBE = ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))  # fmt: skip
#: Its faces, by their corners, and each face's normal: bottom, top, then -y, +x, +y, -x.
FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
NORMALS = ((0, 0, -1), (0, 0, 1), (0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0))
#: What the box round the bins is drawn in: black, a pixel wide; its levels dotted.
BLACK, DOTTED = (1, 1, 1), (1, 1, 3)


def _edges(h: Any) -> tuple[list[Any], np.ndarray[Any, Any]]:
    """Each axis's shown edges, and the shown bins' contents."""
    spans = [shown_bins(h, axis) for axis in range(3)]
    edges = [h.axes[axis].edges()[low : high + 1] for axis, (low, high) in enumerate(spans)]
    values = h.values()[tuple(slice(low, high) for low, high in spans)]
    return edges, values


def _order(view: View3D, counts: tuple[int, ...]) -> list[range]:
    """The bins from the front to the back, along each axis: ``PaintH3BoxRaster``'s order."""
    return [range(count) if view.tnorm[8 + axis] < 0 else range(count - 1, -1, -1)
            for axis, count in enumerate(counts)]  # fmt: skip


def _draw(screen: RasterScreen, segments: list[Segment], ends: tuple[Any, Any],
          look: tuple[int, int, int]) -> None:  # fmt: skip
    """The parts of the line between ``ends`` the screen leaves in sight, as segments."""
    (ax, ay), (bx, by) = ends
    for t0, t1 in screen.visible((ax, ay), (bx, by)):
        segments.append(Segment(ax + (bx - ax) * t0, ay + (by - ay) * t0,
                                ax + (bx - ax) * t1, ay + (by - ay) * t1, *look))  # fmt: skip


def _box(view: View3D, screen: RasterScreen, corners: list[Any], draw: Any) -> None:
    """One bin's box: each face towards the eye outlined where in sight, then covered."""
    for face, normal in zip(FACES, NORMALS):
        if view.normal(*normal) <= 0:
            continue
        points = [corners[k] for k in face]
        for i in range(4):
            draw((points[i], points[(i + 1) % 4]))
        screen.fill(points)


#: ``TH1``'s ``fMinimum`` and ``fMaximum`` when never set.
UNSET = -1111.0


def _extremes(h: Any, values: np.ndarray[Any, Any]) -> tuple[float, float]:
    """``GetMinimum`` and ``GetMaximum``: the histogram's own, if set, else its bins'."""
    lowest = float(lookup(h, "fMinimum", UNSET))
    highest = float(lookup(h, "fMaximum", UNSET))
    return (float(np.min(values)) if lowest == UNSET else lowest,
            float(np.max(values)) if highest == UNSET else highest)  # fmt: skip


class _Bins:
    """The shown bins: their middles, widths and contents, and the range boxes are sized over."""

    def __init__(self, h: Any) -> None:
        edges, self.values = _edges(h)
        self.middles = [0.5 * (e[1:] + e[:-1]) for e in edges]
        self.widths = [np.diff(e) for e in edges]
        lowest, highest = _extremes(h, self.values)
        self.low = max(lowest, 0.0)
        self.high = max(abs(highest), abs(lowest))

    def scale(self, where: tuple[int, int, int]) -> float:
        """How big a bin's box is: the cube root of its share of the range, halved; 0 for none."""
        w = min(abs(float(self.values[where])), self.high)
        if w < self.low:
            return 0.0
        return float(((w - self.low) / (self.high - self.low)) ** (1.0 / 3.0) / 2.0)

    def corners(self, view: View3D, where: tuple[int, int, int], scale: float) -> list[Any]:
        """A bin's box's eight corners, as the view lays them flat."""
        sides = [float(self.widths[a][where[a]]) * scale for a in range(3)]
        middle = [float(self.middles[a][where[a]]) for a in range(3)]
        return [view.to_ndc([c[a] * sides[a] + middle[a] for a in range(3)])[:2] for c in CUBE]


def _boxes(view: View3D, screen: RasterScreen, h: Any, look: tuple[int, int, int]) -> list[Segment]:
    """Every bin's box, front to back, as big as the cube root of its share of the highest."""
    bins = _Bins(h)
    segments: list[Segment] = []
    if bins.high <= bins.low:
        return segments
    xs, ys, zs = _order(view, bins.values.shape)
    for where in itertools.product(xs, ys, zs):
        scale = bins.scale(where)
        if scale:
            corners = bins.corners(view, where, scale)
            _box(view, screen, corners, lambda ends: _draw(screen, segments, ends, look))
    return segments


def _back_box(view: View3D, screen: RasterScreen, ndivz: int) -> list[Segment]:
    """``BackBox`` drawn by ``DrawFaceRaster1``: the back walls' levels dotted, then their edges."""
    low, high = view.rmin[2], view.rmax[2]
    start, _end, count, width = optimize(low, high, ndivz)
    # level_lines reads a screen's levels, and nothing else of it
    levels: Any = SimpleNamespace(levels=[start + i * width for i in range(count + 1)])
    corners = box_corners(view)
    segments: list[Segment] = []
    for face in BACK:
        points = [corners[k - 1] for k in face]
        for a, b in level_lines(levels, points, [p[2] for p in points]):
            _draw(screen, segments, (view.to_ndc(a)[:2], view.to_ndc(b)[:2]), DOTTED)
        flat = [view.to_ndc(p)[:2] for p in points]
        for i in range(4):
            _draw(screen, segments, (flat[i], flat[(i + 1) % 4]), BLACK)
        screen.fill(flat)
    return segments


def paint_h3_boxes(scene: Scene, h: Any, option: str) -> None:
    """A three-dimensional histogram as its boxes, in the box round them, with its axes."""
    scene.ax.set_axis_off()  # no frame: the box is drawn instead
    edges, _values = _edges(h)
    rmin = tuple(float(e[0]) for e in edges)
    rmax = tuple(float(e[-1]) for e in edges)
    theta = float(scene.pad.get("fTheta", THETA))
    phi = float(scene.pad.get("fPhi", PHI))
    view = View3D(rmin, rmax, -90 - phi, 90 - theta)  # type: ignore[arg-type]
    pad = _Pad(scene, view.pad_range(scene.pad.margins))
    screen = RasterScreen()
    look = (int(lookup(h, "fLineColor", 1)), int(lookup(h, "fLineWidth", 1)),
            int(lookup(h, "fLineStyle", 1)))  # fmt: skip
    segments = [] if "ISO" in option.upper() else _boxes(view, screen, h, look)
    zaxis = lookup(h, "fZaxis") or {}
    segments += _back_box(view, screen, int(lookup(zaxis, "fNdivisions", 510)) % 100)
    _draw_segments(pad, segments)
    corners = [view.to_ndc(p) for p in box_corners(view)]
    fronts = [[pad.pixel(corners[k - 1][0], corners[k - 1][1]) for k in face] for face in FRONT]
    add_line(scene, fronts, scene.colors.rgb(1), many=True)
    _axes(pad, view, h)
