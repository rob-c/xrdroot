"""3-D drawn in a pad, as ROOT paints it there: wireframes and points through the pad's view.

With no OpenGL - which is always, here, as it is for ROOT in batch - a
volume drawn in a pad is painted through the pad's own 3-D viewer: every
edge of every visible shape, in its volume's line colour, style and width,
projected by the pad's ``TView3D`` (:mod:`.view`), nothing hidden behind
anything. ``TPolyLine3D`` and ``TPolyMarker3D`` are painted through the
same view. The pad's range is ``-1..1`` both ways, as ``TView3D`` sets it;
a view with no range of its own (``SetAutoRange``) takes the box round
everything 3-D in the pad, as ROOT's first, ranging, paint finds it. Each
line is of whole pixels, as ``TImageDump`` draws a ``TLine``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .scene import extent
from .view import make_view

__all__ = ["GEOMETRY", "paint_geometry", "paint_polyline3d", "paint_polymarker3d", "points_of"]


def points_of(prim: Any) -> np.ndarray[Any, Any]:
    """Every 3-D point a primitive paints (N by 3), or none if it is not 3-D."""
    classname = getattr(prim, "classname", "")
    if classname == "TGeoVolume":
        low, high = extent(prim.get("solids", []))
        return np.array([low, high]) if prim.get("solids") else np.zeros((0, 3))
    if classname in ("TPolyLine3D", "TPolyMarker3D"):
        count = int(prim.get("fN", 0))
        return np.asarray(prim.get("fP", []), dtype=np.float64)[: 3 * count].reshape(-1, 3)
    return np.zeros((0, 3))


def _view(scene: Any) -> Any:
    """The pad's view, its range - if it has none - the box round all its 3-D."""
    found = scene.pad.get("fView")
    spec = found.spec() if found is not None else {
        "longitude": -90.0 - float(scene.pad.get("fPhi", 30.0)),
        "latitude": 90.0 - float(scene.pad.get("fTheta", 30.0)),
    }
    points = [points_of(obj) for obj, _ in scene.pad.primitives]
    every = np.concatenate(points) if points else np.zeros((0, 3))
    low, high = (every.min(axis=0), every.max(axis=0)) if len(every) else ([-1.0] * 3, [1.0] * 3)
    width, height = scene.pixels
    return make_view(spec, low, high, height / width if width else 1.0)


def _pixels(scene: Any, flat: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Points in the pad's ``-1..1`` as the canvas's pixels, ``y`` down."""
    x, y, w, h = scene.box
    width, height = scene.canvas
    u, v = 0.5 * (flat[..., 0] + 1.0), 0.5 * (flat[..., 1] + 1.0)
    return np.stack([(x + u * w) * width, (1 - y - v * h) * height], axis=-1)


def paint_geometry(scene: Any, prim: Any, _option: str) -> None:
    """Every solid in ``prim``'s ``solids``, each edge a line across the pad."""
    from ..canvas.raster import add_line

    view = _view(scene)
    for solid in prim.get("solids", []):
        drawn = view.project(solid.mesh.points)[solid.mesh.edges()]
        if len(drawn) and solid.line_width > 0:
            add_line(scene, list(_pixels(scene, drawn)), scene.colors.rgb(solid.line_color),
                     solid.line_width, solid.line_style, many=True)  # fmt: skip


def paint_polyline3d(scene: Any, prim: Any, _option: str) -> None:
    """A ``TPolyLine3D``: its points joined, through the pad's view."""
    from ..canvas.model import lookup
    from ..canvas.raster import add_line

    flat = _view(scene).project(points_of(prim))
    width = int(lookup(prim, "fLineWidth", 1) or 0)
    if len(flat) > 1 and width > 0:
        add_line(scene, _pixels(scene, flat), scene.colors.rgb(lookup(prim, "fLineColor", 1)),
                 width, lookup(prim, "fLineStyle", 1))  # fmt: skip


def paint_polymarker3d(scene: Any, prim: Any, _option: str) -> None:
    """A ``TPolyMarker3D``: a marker at each point, through the pad's view, as ROOT marks one."""
    from ..canvas.marks import draw_markers
    from ..canvas.model import lookup

    pixels = _pixels(scene, _view(scene).project(points_of(prim)))
    draw_markers(scene, pixels, int(lookup(prim, "fMarkerStyle", 1)),
                 float(lookup(prim, "fMarkerSize", 1.0)),
                 scene.colors.rgb(lookup(prim, "fMarkerColor", 1)))  # fmt: skip


#: The 3-D classes a pad paints, and how.
GEOMETRY = {
    "TGeoVolume": paint_geometry,
    "TPolyLine3D": paint_polyline3d,
    "TPolyMarker3D": paint_polymarker3d,
}
