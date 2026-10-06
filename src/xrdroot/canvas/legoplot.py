"""``THistPainter::PaintLego`` and ``PaintSurface``: a two-dimensional histogram in three.

``LEGO`` draws each bin as a block standing on the z axis's bottom, and
``SURF`` a mesh through the bins' middles; both are seen from the pad's
``fTheta`` and ``fPhi`` (:mod:`.view3d`), drawn from the front to the back
with hidden lines removed (:mod:`.lego`), in the histogram's line colour,
width and style. The box round them has its back walls lined at the z
axis's divisions and its front edges drawn over everything; its three
axes run along the box's edges nearest the eye (``PaintLegoAxis``), each a
``TGaxis`` (:mod:`.axis`). There is no frame: the pad's range is the box's
shadow, fitted into the pad's margins.

The z range is ``TableInit``'s - the lowest and highest bin, five percent
of the span above, and below unless that passes zero - with another five
percent on top, as ``PaintLego`` adds.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .axis import paint_axis
from .dressing import axis_of, draw_painted
from .frame import shown_bins
from .lego import MovingScreen, Segment
from .legofaces import back_box, box_corners, front_box, lego_cells, surface_cells
from .model import lookup
from .raster import add_line
from .scene import Scene
from .text import nint
from .view3d import View3D

__all__ = ["paint_three_d", "three_d_kind"]

#: ``gStyle``'s ``fHistTopMargin``: the room above the highest bin.
TOP_MARGIN = 0.05
#: The pad's view angles when never set: ``fTheta`` and ``fPhi``.
THETA, PHI = 30.0, 30.0
#: ``PaintLegoAxis``'s nearness, in NDC, below which an axis's ends are one point and it is
#: left out: the x and y axes', and the z axis's.
SHORT_AXIS = 0.001
SHORT_Z_AXIS = 0.1


def three_d_kind(option: str) -> str:
    """``"LEGO"`` or ``"SURF"`` if ``option`` asks for one of them, else nothing."""
    upper = option.upper()
    return next((kind for kind in ("LEGO", "SURF") if kind in upper), "")


def _z_range(values: np.ndarray[Any, Any], h: Any) -> tuple[float, float]:
    """``TableInit``'s z range, and ``PaintLego``'s extra margin on top."""
    zmin, zmax = float(values.min()), float(values.max())
    stored_max = float(lookup(h, "fMaximum", -1111))
    stored_min = float(lookup(h, "fMinimum", -1111))
    zmax = stored_max if stored_max != -1111 else zmax + TOP_MARGIN * (zmax - zmin)
    if stored_min != -1111:
        zmin = stored_min
    else:
        dz = TOP_MARGIN * (zmax - zmin)
        zmin = 0.0 if zmin >= 0 and zmin - dz <= 0 else zmin - dz
    if zmin == 0 and zmax == 0:
        zmin, zmax = -1.0, 1.0
    if zmin >= zmax:
        delta = abs(zmin) or 1.0
        zmin, zmax = zmin - 0.5 * delta, zmax + 0.5 * delta
    return zmin, zmin + (zmax - zmin) * (1 + TOP_MARGIN)


class _Table:
    """The histogram's shown bins, their edges and contents, and the box they fill."""

    def __init__(self, h: Any) -> None:
        (x0, x1), (y0, y1) = shown_bins(h, 0), shown_bins(h, 1)
        self.xedges = h.axes[0].edges()[x0 : x1 + 1]
        self.yedges = h.axes[1].edges()[y0 : y1 + 1]
        self.values = h.values()[x0:x1, y0:y1]
        self.zmin, self.ztop = _z_range(self.values, h)
        zmax = self.zmin + (self.ztop - self.zmin) / (1 + TOP_MARGIN)
        self.zmax = zmax

    def box(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """The box's lowest and highest corners: the shown bins' edges, and the z range."""
        return (float(self.xedges[0]), float(self.yedges[0]), self.zmin), (
            float(self.xedges[-1]), float(self.yedges[-1]), self.ztop)  # fmt: skip

    def clamp(self, value: float) -> float:
        """A bin's content kept within the z range, as ``TableInit`` draws it."""
        return min(max(float(value), self.zmin), self.zmax)

    def block(self, ix: int, iy: int) -> tuple[list[tuple[float, float]], tuple[float, float]]:
        """``LegoFunction``: a bin's corners and its block's bottom and top."""
        x1, x2 = float(self.xedges[ix - 1]), float(self.xedges[ix])
        y1, y2 = float(self.yedges[iy - 1]), float(self.yedges[iy])
        top = self.clamp(self.values[ix - 1, iy - 1])
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], (self.zmin, top)

    def square(self, ix: int, iy: int) -> list[tuple[float, float, float]]:
        """``SurfaceFunction``: the middles of four neighbouring bins, at their contents."""
        xs = 0.5 * (self.xedges[:-1] + self.xedges[1:])
        ys = 0.5 * (self.yedges[:-1] + self.yedges[1:])
        corners = ((0, 0), (1, 0), (1, 1), (0, 1))
        return [(float(xs[ix - 1 + a]), float(ys[iy - 1 + b]),
                 self.clamp(self.values[ix - 1 + a, iy - 1 + b]))
                for a, b in corners]  # fmt: skip


class _Pad:
    """The pad's range once ``PadRange`` has set it: pad coordinates into its NDC and pixels."""

    def __init__(self, scene: Scene, pad_range: tuple[float, float, float, float]) -> None:
        self.scene = scene
        self.x1, self.y1, self.x2, self.y2 = pad_range

    def ndc(self, x: float, y: float) -> tuple[float, float]:
        """A point of the pad's range in the pad's NDC."""
        return (x - self.x1) / (self.x2 - self.x1), (y - self.y1) / (self.y2 - self.y1)

    def pixel(self, x: float, y: float) -> tuple[float, float]:
        """A point of the pad's range as the canvas's pixel, unrounded."""
        return self.scene.pixel(*self.ndc(x, y))


def _draw_segments(pad: _Pad, segments: list[Segment]) -> None:
    """The visible pieces, one pixel line per colour, width and style."""
    grouped: dict[tuple[int, int, int], list[Any]] = {}
    for one in segments:
        grouped.setdefault((one.color, one.width, one.style), []).append(
            [pad.pixel(one.x1, one.y1), pad.pixel(one.x2, one.y2)])
    for (color, width, style), lines in grouped.items():
        add_line(pad.scene, lines, pad.scene.colors.rgb(color), width, style, many=True)


#: An axis's two ends in the pad's coordinates.
Ends = list[tuple[float, float]]


def _axis_ends(view: View3D) -> dict[str, Ends]:
    """``PaintLegoAxis``'s ends of the x, y and z axes: the box's corners ``AxisVertex`` picks.

    A y axis that is all but upright is made exactly so.
    """
    corners = box_corners(view)
    ends = view.corners()
    tips = {name: [view.to_ndc(corners[k - 1])[:2] for k in pair]
            for name, pair in zip("xyz", (ends.x, ends.y, ends.z), strict=False)}  # fmt: skip
    if abs(tips["y"][0][0] - tips["y"][1][0]) < SHORT_AXIS:
        tips["y"][1] = (tips["y"][0][0], tips["y"][1][1])
    return tips


def _axis_option(name: str, a: tuple[float, float], b: tuple[float, float]) -> str | None:
    """The ``TGaxis`` option an axis is drawn with - its ticks' side - or none if it is too short.

    z's ticks go the way it runs up; x's and y's the way they run left.
    """
    if name == "z":
        sign, far = ("+" if b[1] > a[1] else "-"), SHORT_Z_AXIS
    else:
        sign, far = ("+" if a[0] > b[0] else "-"), SHORT_AXIS
    if abs(a[0] - b[0]) < far and abs(a[1] - b[1]) <= far:
        return None
    return "SDH=" + sign


def _axes(pad: _Pad, view: View3D, h: Any) -> None:
    """``PaintLegoAxis``: x and y along the box's front bottom edges, z up its side."""
    tips = _axis_ends(view)
    for index, name in enumerate("xyz"):
        (a, b) = tips[name]
        chopt = _axis_option(name, a, b)
        if chopt is not None:
            _one_axis(pad, h, index, (a, b), chopt, (view.rmin[index], view.rmax[index]))


def _one_axis(pad: _Pad, h: Any, index: int, ends: Any, chopt: str,
              span: tuple[float, float]) -> None:  # fmt: skip
    """One of the box's axes as a ``TGaxis`` from the histogram's axis.

    As ``PaintLegoAxis`` sets it: its divisions never optimised when they
    are negative, x's and y's labels pushed out past their ticks, and y's
    title offset one and a half when it has none.
    """
    members = lookup(h, f"f{'XYZ'[index]}axis") or {}
    (x0, y0), (x1, y1) = (pad.ndc(*end) for end in ends)
    axis = axis_of(members, x0=x0, y0=y0, x1=x1, y1=y1, wmin=span[0], wmax=span[1], chopt=chopt,
                   pad=pad.scene.pixels)  # fmt: skip
    ndiv = axis.ndiv
    axis = replace(axis, ndiv=abs(ndiv), chopt=chopt + ("N" if ndiv < 0 else ""))
    if index < 2:
        axis = replace(axis, label_offset=axis.label_offset + axis.tick_size)
    if index == 1 and axis.title_offset == 0:
        axis = replace(axis, title_offset=1.5)

    def pixel(u: float, v: float) -> tuple[int, int]:
        """A point of the pad's NDC as the whole pixel ``TGaxis`` rounds it to."""
        px, py = pad.scene.pixel(u, v)
        return nint(px), nint(py)

    draw_painted(pad.scene, paint_axis(axis, pixel), axis)


def paint_three_d(scene: Scene, h: Any, option: str) -> None:
    """A two-dimensional histogram as ``LEGO`` or ``SURF``, with its box and axes."""
    scene.ax.set_axis_off()  # no frame: the box is drawn instead
    table = _Table(h)
    theta = float(scene.pad.get("fTheta", THETA))
    phi = float(scene.pad.get("fPhi", PHI))
    view = View3D(*table.box(), -90 - phi, 90 - theta)
    pad = _Pad(scene, view.pad_range(scene.pad.margins))
    screen = MovingScreen(view)
    zaxis = lookup(h, "fZaxis") or {}
    screen.grid_levels(int(lookup(zaxis, "fNdivisions", 510)) % 100)
    look = (int(lookup(h, "fLineColor", 1)), int(lookup(h, "fLineWidth", 1)),
            int(lookup(h, "fLineStyle", 1)))  # fmt: skip
    nx, ny = table.values.shape
    if three_d_kind(option) == "LEGO":
        lego_cells(screen, nx, ny, table.block, look)
    else:
        surface_cells(screen, nx - 1, ny - 1, table.square, look)
    back_box(screen)
    _draw_segments(pad, screen.segments)
    front = [[pad.pixel(x, y) for x, y in line] for line in front_box(screen)]
    add_line(scene, front, scene.colors.rgb(1), many=True)
    _axes(pad, view, h)

