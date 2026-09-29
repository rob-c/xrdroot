"""What a geometry draws: each visible volume's surface where it sits, and how it looks.

``TGeoPainter`` walks the tree of placed volumes from the one drawn, down
to the visible depth, and paints each visible one's shape through the
placements above it. What it paints is a :class:`Solid` here - the shape's
:class:`~.mesh.Mesh` in world coordinates, with the volume's line colour,
style and width and its fill and transparency - and a list of those is all
any of the three ways of drawing takes: the pad's wireframe
(:func:`segments`), plotly's figure and pyvista's scene.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .mesh import Mesh
from .view import PerspectiveView

__all__ = ["Solid", "extent", "segments"]


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


def segments(solids: Sequence[Solid], view: PerspectiveView) -> list[np.ndarray[Any, Any]]:
    """Each solid's edges on the pad (K by 2 by 2), in the ``-1..1`` range ``view`` projects to."""
    drawn = []
    for solid in solids:
        flat = view.project(solid.mesh.points)
        drawn.append(flat[solid.mesh.edges()])
    return drawn


def view_of(solids: Sequence[Solid], theta: float, phi: float, aspect: float) -> PerspectiveView:
    """The perspective a pad of these angles and shape sees these solids in."""
    low, high = extent(solids)
    return PerspectiveView.from_pad(low, high, theta, phi, aspect)
