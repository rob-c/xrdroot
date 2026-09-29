"""A geometry painted in a pad, as ``TGeoPainter`` paints one there: a wireframe in perspective.

With no OpenGL - which is always, here, as it is for ROOT in batch - a
volume drawn in a pad is painted through the pad's own 3-D viewer: every
edge of every visible shape, in its volume's line colour, style and width,
projected by the pad's ``TView3D`` (:mod:`.view`). The pad's range is
``-1..1`` both ways, as ``TView3D`` sets it, so the picture fills the pad
whatever the geometry's size, and nothing is hidden behind anything. Each
edge is a line of whole pixels, as ``TImageDump`` draws a ``TLine``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .scene import segments, view_of

__all__ = ["GEOMETRY", "paint_geometry"]


def _pixels(scene: Any, drawn: np.ndarray[Any, Any]) -> list[np.ndarray[Any, Any]]:
    """Segments in the pad's ``-1..1`` as the canvas's pixels, ``y`` down: one line each."""
    x, y, w, h = scene.box
    width, height = scene.canvas
    u, v = 0.5 * (drawn[..., 0] + 1.0), 0.5 * (drawn[..., 1] + 1.0)
    across = (x + u * w) * width
    down = (1 - y - v * h) * height
    return list(np.stack([across, down], axis=-1))


def paint_geometry(scene: Any, prim: Any, _option: str) -> None:
    """Every solid in ``prim``'s ``solids`` as line segments across the pad."""
    from ..canvas.raster import add_line

    solids = prim.get("solids", [])
    width, height = scene.pixels
    pad = scene.pad
    view = view_of(solids, float(pad.get("fTheta", 30.0)), float(pad.get("fPhi", 30.0)),
                   height / width if width else 1.0)  # fmt: skip
    for solid, drawn in zip(solids, segments(solids, view)):
        if len(drawn) and solid.line_width > 0:
            add_line(scene, _pixels(scene, drawn), scene.colors.rgb(solid.line_color),
                     solid.line_width, solid.line_style, many=True)  # fmt: skip


#: The class a painted geometry comes to the canvas as, and its painter.
GEOMETRY = {"TGeoVolume": paint_geometry}
