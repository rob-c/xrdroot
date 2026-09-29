"""What a geometry draws: each visible volume's surface where it sits, and how it looks.

``TGeoPainter`` walks the tree of placed volumes from the one drawn, down
to the visible depth, and paints each visible one's shape through the
placements above it. What it paints is a :class:`Solid` here - the shape's
:class:`~.mesh.Mesh` in world coordinates, with the volume's line colour,
style and width and its fill and transparency - and a list of those is all
any of the three ways of drawing takes: the pad's wireframe
(:mod:`.paint`), plotly's figure and pyvista's scene.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .mesh import Mesh

__all__ = ["Solid", "extent"]


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

