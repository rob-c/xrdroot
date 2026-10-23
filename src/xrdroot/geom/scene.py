"""What a geometry draws: each visible volume's surface where it sits, and how it looks.

``TGeoPainter`` walks the tree of placed volumes from the one drawn, down
to the visible depth, and paints each visible one's shape through the
placements above it. What it paints is a :class:`Solid` here - the shape's
:class:`~.mesh.Mesh` in world coordinates, with the volume's line colour,
style and width and its fill and transparency - and a list of those is all
any of the three ways of drawing takes: the pad's wireframe
(:mod:`xrdroot.canvas.geompaint`), plotly's figure and pyvista's scene.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .mesh import Mesh

__all__ = ["Solid", "extent", "points_of"]


@dataclass
class Solid:
    """A shape placed in the world, with its volume's look."""

    mesh: Mesh
    line_color: int = 1
    line_width: int = 1
    line_style: int = 1
    fill_color: int = 19
    transparency: int = 0
    name: str = ""


def extent(solids: Sequence[Solid]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """The box round every point painted: ``Paint("range")``'s ``fRmin`` and ``fRmax``."""
    points = [solid.mesh.points for solid in solids if len(solid.mesh.points)]
    if not points:
        return np.full(3, -1.0), np.full(3, 1.0)
    every = np.concatenate(points)
    return every.min(axis=0), every.max(axis=0)


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

