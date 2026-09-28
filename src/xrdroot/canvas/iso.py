"""``THistPainter::PaintH3Iso``: a three-dimensional histogram's isosurface, shaded by light.

ROOT draws a ``TH3`` drawn ``ISO`` as the surface where its contents cross
their mean over all bins (``IsoSurface``): each cell of eight neighbouring
bin centres is cut into triangles where the surface passes, each corner
lit by ``Luminosity`` - an ambient light of 1 and a light of 10 from the
screen's (1, 1, 1), on a surface of ambient 0.15, diffuse 0.15 and
specular 0.8 - through the contents' gradient there, and each triangle is
filled in bands of 28 shades of the histogram's fill colour
(``FillPolygon``), lightness 0.4 to 0.9, from the back to the front.

ROOT cuts a cell with its own marching cubes; this cuts each into six
tetrahedra, which finds the same surface through slightly other triangles.
"""

from __future__ import annotations

import colorsys
import itertools
import math
from typing import Any

import numpy as np

from .view3d import View3D

__all__ = ["iso_polygons"]

#: ``PaintH3Iso``'s lights and surface: the ambient light, the light at (1, 1, 1) of the
#: screen, and how much of each the surface gives back - ambient, diffuse, specular.
AMBIENT, LIGHT = 1.0, 10.0
QA, QD, QS = 0.15, 0.15, 0.8
#: How many shades the surface is drawn in.
SHADES = 28
#: A cell's corners, as offsets from its first bin, and its six tetrahedra about its diagonal.
CORNERS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
TETRAHEDRA = ((0, 5, 1, 6), (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6))


def gradients(values: np.ndarray[Any, Any], centres: list[Any]) -> list[np.ndarray[Any, Any]]:
    """The contents' gradient at each bin: central differences inside, one-sided at the ends."""
    return [np.gradient(values, centre, axis=axis, edge_order=1)
            for axis, centre in enumerate(centres)]  # fmt: skip


def _cut(a: Any, b: Any, level: float) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Where the surface crosses the edge between two corners: its place and its gradient."""
    (fa, pa, ga), (fb, pb, gb) = a, b
    t = (level - fa) / (fb - fa)
    return pa + t * (pb - pa), ga + t * (gb - ga)


def _quad(inside: list[Any], outside: list[Any], level: float) -> list[list[Any]]:
    """Two corners in and two out: the surface crosses four edges, a quad of two triangles."""
    (i1, i2), (o1, o2) = inside, outside
    quad = [_cut(i1, o1, level), _cut(i1, o2, level), _cut(i2, o2, level), _cut(i2, o1, level)]
    return [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]


def _sides(corners: list[Any], level: float) -> tuple[list[Any], list[Any]]:
    """The corners at or above ``level``, and those below it."""
    inside: list[Any] = []
    outside: list[Any] = []
    for corner in corners:
        (inside if corner[0] >= level else outside).append(corner)
    return inside, outside


def _tetrahedron(corners: list[Any], level: float) -> list[list[Any]]:
    """The triangles the surface makes in one tetrahedron: none, one, or two for a quad."""
    inside, outside = _sides(corners, level)
    if len(inside) in (0, 4):
        return []
    if len(inside) == 2:
        return _quad(inside, outside, level)
    alone, others = (inside[0], outside) if len(inside) == 1 else (outside[0], inside)
    return [[_cut(alone, other, level) for other in others]]


def _corner(values: Any, centres: list[Any], grads: list[Any], k: tuple[int, ...]) -> Any:
    """A cell's corner: the content there, the bin's centre, and the contents' gradient."""
    place = np.array([centres[d][k[d]] for d in range(3)])
    return float(values[k]), place, np.array([grads[d][k] for d in range(3)])


def isosurface(values: np.ndarray[Any, Any], centres: list[Any], level: float) -> list[list[Any]]:
    """Every triangle of the surface where the contents cross ``level``, with its gradients."""
    grads = gradients(values, centres)
    triangles: list[list[Any]] = []
    for first in itertools.product(*(range(n - 1) for n in values.shape)):
        cell = [tuple(first[d] + step[d] for d in range(3)) for step in CORNERS]
        corners = [_corner(values, centres, grads, k) for k in cell]
        for tet in TETRAHEDRA:
            triangles += _tetrahedron([corners[k] for k in tet], level)
    return triangles


def luminosity(view: View3D, normal: np.ndarray[Any, Any]) -> float:
    """``Luminosity``: how bright a surface of ``normal`` is, lit as ``PaintH3Iso`` lights it."""
    n = view.normal_to_ndc(normal)  # NormalWCtoNDC
    size = math.sqrt(float(n @ n))
    if size == 0:
        return AMBIENT * QA
    n = n / (-size if n[2] < 0 else size)
    light = np.ones(3) / math.sqrt(3.0)
    cosn = float(light @ n)
    total = AMBIENT * QA
    if cosn >= 0:
        cosr = (n[1] * (n[2] * light[1] - n[1] * light[2])
                - n[0] * (n[0] * light[2] - n[2] * light[0]))  # fmt: skip
        cosr = max(float(cosr + n[2] * cosn), 0.0)
        total += LIGHT * (QD * cosn + QS * cosr)
    return total


def shades(rgb: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    """``PaintH3Iso``'s 28 shades of the fill colour: its hue and saturation, lightness 0.4 up."""
    hue, _light, saturation = colorsys.rgb_to_hls(*rgb)
    step = 0.5 / SHADES
    return [colorsys.hls_to_rgb(hue, 0.4 + index * step, saturation) for index in range(SHADES)]


def _band(points: list[Any], values: list[float], low: float, high: float) -> list[Any]:
    """``FindPartEdge`` round a polygon: the part of it whose shading is between two levels."""
    kept: list[Any] = []
    count = len(points)
    for i in range(count):
        j = (i + 1) % count
        (p1, f1), (p2, f2) = (points[i], values[i]), (points[j], values[j])
        if low <= f1 <= high:
            kept.append(p1)
        for level in sorted((low, high), reverse=f1 > f2):
            if min(f1, f2) < level < max(f1, f2):
                t = (level - f1) / (f2 - f1)
                kept.append(p1 + t * (p2 - p1))
    return kept


def _bounds(lights: list[float], levels: list[float]) -> list[float]:
    """``FillPolygon``'s bands: below the first level, between each two, and above the last."""
    low = min(levels[0] - 1, min(lights) - 1)
    high = max(levels[-1] + 1, max(lights) + 1)
    return [low, *levels, high]


def _shade(band: int, colours: list[Any], background: Any) -> Any:
    """A band's colour: the pad's below the first level, then ``fColorLevel``'s shades."""
    return background if band == 0 else colours[max(band - 2, 0)]


def iso_polygons(view: View3D, triangles: list[list[Any]], rgb: Any,
                 background: Any) -> list[tuple[Any, Any]]:  # fmt: skip
    """Each triangle's bands, back to front, as flat polygons of the view and their shades."""
    fmin, fmax = AMBIENT * QA, AMBIENT * QA + (LIGHT + 0.1) * (QD + QS)
    levels = [fmin + index * (fmax - fmin) / SHADES for index in range(SHADES + 1)]
    colours = shades(rgb)
    made = []
    for points, lights in _back_to_front(view, triangles):
        bounds = _bounds(lights, levels)
        for band in range(len(bounds) - 1):
            part = _band(points, lights, bounds[band], bounds[band + 1])
            if len(part) >= 3:
                made.append(([(p[0], p[1]) for p in part], _shade(band, colours, background)))
    return made


def _back_to_front(view: View3D, triangles: list[list[Any]]) -> list[tuple[Any, Any]]:
    """The triangles laid flat and lit, the furthest first, so the nearer paint over them."""
    flat = []
    for triangle in triangles:
        points = [np.array(view.to_ndc(point)) for point, _gradient in triangle]
        lights = [luminosity(view, gradient) for _point, gradient in triangle]
        flat.append((float(np.mean([p[2] for p in points])), points, lights))
    return [(points, lights) for _depth, points, lights in sorted(flat, key=lambda item: item[0])]
