"""``THistPainter::PaintAxis``: the axes round a frame, each painted by a ``TGaxis``.

The x axis runs along the frame's bottom and the y axis up its left side,
each with the attributes of its ``TAxis`` - ``fNdivisions``, the label and
title fonts, sizes and offsets, ``fTickLength`` - and the options
``THistPainter`` gives it: ``S`` for ticks of the axis's own length, ``N``
for divisions never optimised, ``G`` for a logarithmic scale, ``W`` for a
grid across the frame. ``fTickx`` and ``fTicky`` add the axes' twins on the
top and right, ticked inward and, unless 2, unlabelled.

What each axis paints comes from :func:`~.axis.paint_axis`; this draws it:
segments in NDC, a pixel wide, in the axis's colour, the grid dotted as
``gStyle``'s grid is, and every label through :func:`~.latex.paint_latex`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .axis import Axis, Painted, paint_axis
from .latex import paint_latex
from .model import lookup
from .raster import add_line
from .scene import Scene
from .text import nint

__all__ = ["axis_of", "dress_axes", "draw_painted"]

#: ``TAxis``'s bits in ``fBits``, by the name :class:`~.axis.Axis` knows each by.
BITS = {
    "decimals": 1 << 7, "tickplus": 1 << 9, "tickminus": 1 << 10, "centertitle": 1 << 12,
    "centerlabels": 1 << 14, "rotatetitle": 1 << 15, "morelog": 1 << 16, "noexponent": 1 << 17,
}  # fmt: skip
#: ``gStyle``'s grid: dotted (style 3), a pixel wide, in the axis's colour.
GRID_STYLE, GRID_WIDTH = 3, 1


def axis_of(members: Any, **placed: Any) -> Axis:
    """An :class:`~.axis.Axis` with the attributes of a ``TAxis``, ``TGaxis::ImportAxisAttributes``'s."""
    bits = int(lookup(members, "fBits", 0) or 0)
    return Axis(
        ndiv=int(lookup(members, "fNdivisions", 510)),
        label_font=int(lookup(members, "fLabelFont", 42) or 42),
        label_size=float(lookup(members, "fLabelSize", 0.035)),
        label_color=int(lookup(members, "fLabelColor", 1)),
        label_offset=float(lookup(members, "fLabelOffset", 0.005)),
        tick_size=float(lookup(members, "fTickLength", 0.03)),
        title=str(lookup(members, "fTitle", "") or ""),
        title_offset=float(lookup(members, "fTitleOffset", 1.0)),
        title_size=float(lookup(members, "fTitleSize", 0.035)),
        title_font=int(lookup(members, "fTitleFont", 42) or 42),
        title_color=int(lookup(members, "fTitleColor", 1)),
        line_color=int(lookup(members, "fAxisColor", 1)),
        bits=frozenset(name for name, bit in BITS.items() if bits & bit),
        **placed,
    )


def draw_painted(scene: Scene, painted: Painted, axis: Axis) -> None:
    """An axis's segments, its grid, and its labels and title, drawn in the pad."""
    color = scene.colors.rgb(axis.line_color)
    for segments, style in ((painted.lines, 1), (painted.grid, GRID_STYLE)):
        for x1, y1, x2, y2 in segments:
            add_line(scene, [scene.pixel(x1, y1), scene.pixel(x2, y2)], color, GRID_WIDTH, style)
    for label in painted.labels:
        attributes = {"font": label.font, "size": label.size, "color": label.color,
                      "align": label.align, "angle": label.angle, "line": 2}  # fmt: skip
        paint_latex(scene, label.text, scene.pixel(label.u, label.v), attributes)


def _divisions(ndiv: int, scene: Scene) -> int:
    """``fNdivisions`` over 1000, whose primaries scale with the pad's width, as ``PaintAxis`` has it."""
    if ndiv <= 1000:
        return ndiv
    primary = max(1, ndiv % 100)
    return 100 * (ndiv // 100) + int(primary * scene.box[2])


def _chopt(ndiv: int, grid: bool, log: bool) -> str:
    return "SDH" + ("N" if ndiv < 0 else "") + ("W" if grid else "") + ("G" if log else "")


def dress_axes(scene: Scene, source: Any) -> None:
    """The frame's axes, and their twins on the far sides when the pad asks for them."""
    pad = scene.pad
    left, right, bottom, top = pad.margins
    x0, x1, y0, y1 = left, 1 - right, bottom, 1 - top
    (xmin, xmax), (ymin, ymax) = scene.ax.get_xlim(), scene.ax.get_ylim()
    tickx, ticky = pad.ticks
    gridx, gridy = pad.grid

    def pixel(u: float, v: float) -> tuple[int, int]:
        px, py = scene.pixel(u, v)
        return nint(px), nint(py)

    xaxis = axis_of(lookup(source, "fXaxis") or {}, x0=x0, y0=y0, x1=x1, y1=y0, wmin=xmin, wmax=xmax,
                    grid_length=y1 - y0, pad=scene.pixels)  # fmt: skip
    ndiv = _divisions(xaxis.ndiv, scene)
    xaxis = replace(xaxis, ndiv=abs(ndiv), chopt=_chopt(ndiv, gridx, pad.logx))
    _paint(scene, xaxis, pixel)
    if tickx:
        twin = xaxis.chopt.replace("W", "z") + "-" + ("U" if tickx < 2 else "")
        _paint(scene, replace(xaxis, y0=y1, y1=y1, chopt=twin, title=""), pixel)
    yaxis = axis_of(lookup(source, "fYaxis") or {}, x0=x0, y0=y0, x1=x0, y1=y1, wmin=ymin, wmax=ymax,
                    grid_length=x1 - x0, pad=scene.pixels)  # fmt: skip
    ndiv = yaxis.ndiv
    yaxis = replace(yaxis, ndiv=abs(ndiv), chopt=_chopt(ndiv, gridy, pad.logy))
    _paint(scene, yaxis, pixel)
    if ticky:
        twin = yaxis.chopt.replace("W", "z") + ("U" if ticky < 2 else "+L")
        size = -yaxis.tick_size if ticky < 2 else yaxis.tick_size
        _paint(scene, replace(yaxis, x0=x1, x1=x1, chopt=twin, title="", tick_size=size), pixel)


def _paint(scene: Scene, axis: Axis, pixel: Any) -> None:
    draw_painted(scene, paint_axis(axis, pixel), axis)
