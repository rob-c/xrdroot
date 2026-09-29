"""A shape's surface as points and faces: what is drawn, whatever draws it.

ROOT hands its painters a ``TBuffer3D`` for each shape - its points, the
segments joining them and the polygons those bound - tessellated with
``gGeoManager->GetNsegments()`` steps round a circle. A :class:`Mesh` is
that: points (N by 3) and faces, each a loop of point indices. Its edges
are what a wireframe draws, its faces what a surface draws, split into
triangles for a renderer that wants those.

Every solid of revolution - a tube, a cone, a polycone, a sphere, a torus -
is made by :func:`revolve` from its profile: the outer and inner walls'
``(rho, z)`` at each section, turned through its range of phi. Every
eight-cornered solid - a box, a trapezoid, a parallelepiped, ``TGeoArb8`` -
is :func:`hexahedron`, and a prism is :func:`extrusion`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["Mesh", "revolve", "hexahedron", "extrusion", "merged"]

#: The faces of an eight-cornered solid, corners 0-3 at -dz and 4-7 at +dz.
HEXAHEDRON_FACES = ((0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3),
                    (3, 7, 4, 0))  # fmt: skip


class Mesh:
    """Points and the faces through them, each face a loop of indices into the points."""

    def __init__(self, points: Any, faces: Sequence[Sequence[int]]):
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self.faces = [tuple(int(i) for i in face) for face in faces]

    def edges(self) -> np.ndarray[Any, Any]:
        """Every edge of every face once, as index pairs (M by 2), none of zero length."""
        found = set()
        for face in self.faces:
            for a, b in zip(face, face[1:] + face[:1]):
                if a != b:
                    found.add((min(a, b), max(a, b)))
        return np.array(sorted(found), dtype=np.int64).reshape(-1, 2)

    def triangles(self) -> np.ndarray[Any, Any]:
        """Each face fanned into triangles from its first corner (K by 3)."""
        made = [(face[0], b, c) for face in self.faces for b, c in zip(face[1:], face[2:])]
        kept = [t for t in made if len(set(t)) == 3]
        return np.array(kept, dtype=np.int64).reshape(-1, 3)

    def transformed(self, matrix: Any) -> Mesh:
        """The same surface with its points taken through ``matrix``."""
        return Mesh(matrix.to_master(self.points), self.faces)

    def extent(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """The lowest and highest point along each axis."""
        return self.points.min(axis=0), self.points.max(axis=0)


def merged(meshes: Sequence[Mesh]) -> Mesh:
    """Several meshes as one, their faces renumbered into the points laid end to end."""
    points, faces, offset = [], [], 0
    for mesh in meshes:
        points.append(mesh.points)
        faces += [tuple(i + offset for i in face) for face in mesh.faces]
        offset += len(mesh.points)
    return Mesh(np.concatenate(points) if points else np.zeros((0, 3)), faces)


def _ring(rho: float, z: float, phis: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.column_stack([rho * np.cos(phis), rho * np.sin(phis), np.full(len(phis), z)])


def _phis(phi1: float, dphi: float, steps: int) -> tuple[np.ndarray[Any, Any], bool]:
    """A ring's angles, and whether it closes on itself: a full turn, the last not repeated."""
    whole = dphi >= 360.0
    count = steps if whole else steps + 1
    return np.radians(phi1 + dphi * np.arange(count) / steps), whole


class _Rings:
    """The rings of a solid of revolution: ``n`` points round each section of each wall."""

    def __init__(self, sections: int, n: int, whole: bool, closed: bool) -> None:
        self.sections, self.n = sections, n
        self.pairs = [(k, (k + 1) % sections) for k in range(sections if closed else sections - 1)]
        self.round = [(j, (j + 1) % n) for j in range(n if whole else n - 1)]

    def ring(self, wall: int, k: int) -> list[int]:
        """The point indices of wall ``wall`` (0 outer, 1 inner) at section ``k``."""
        start = (wall * self.sections + k) * self.n
        return list(range(start, start + self.n))

    def band(self, a: list[int], b: list[int]) -> list[tuple[int, ...]]:
        """The quadrilaterals between two rings, step by step round them."""
        return [(a[j], a[i], b[i], b[j]) for j, i in self.round]

    def walls(self) -> list[tuple[int, ...]]:
        return [face for wall in (0, 1) for k, m in self.pairs
                for face in self.band(self.ring(wall, k), self.ring(wall, m))]  # fmt: skip

    def caps(self) -> list[tuple[int, ...]]:
        """The ends of the profile, each joining the outer wall to the inner."""
        return [face for k in (0, self.sections - 1)
                for face in self.band(self.ring(0, k), self.ring(1, k))]  # fmt: skip

    def sides(self) -> list[tuple[int, ...]]:
        """The flat faces at the two ends of the range of phi, each round the whole profile."""
        order = list(range(self.sections))
        return [tuple([self.ring(0, k)[j] for k in order]
                      + [self.ring(1, k)[j] for k in reversed(order)])
                for j in (0, self.n - 1)]  # fmt: skip


def revolve(
    outer: Sequence[tuple[float, float]],
    inner: Sequence[tuple[float, float]],
    phi1: float = 0.0,
    dphi: float = 360.0,
    steps: int = 20,
    closed: bool = False,
) -> Mesh:
    """A solid of revolution: ``outer`` and ``inner`` walls, each ``(rho, z)`` point by point.

    The walls are joined at their ends by caps - unless the profile is
    ``closed`` (a torus's circle) - and, over less than a full turn, by the
    two flat faces at ``phi1`` and ``phi1 + dphi``.
    """
    phis, whole = _phis(phi1, dphi, steps)
    points = np.concatenate([_ring(rho, z, phis) for wall in (outer, inner) for rho, z in wall])
    rings = _Rings(len(outer), len(phis), whole, closed)
    faces = rings.walls() + ([] if closed else rings.caps()) + ([] if whole else rings.sides())
    return Mesh(points, faces)


def hexahedron(corners: Any) -> Mesh:
    """An eight-cornered solid: corners 0-3 round its -dz face, 4-7 round its +dz face."""
    return Mesh(corners, HEXAHEDRON_FACES)


def extrusion(polygon: Sequence[tuple[float, float]], sections: Sequence[Sequence[float]]) -> Mesh:
    """A prism: ``polygon`` placed at each section's ``(z, x0, y0, scale)``, walls between."""
    count = len(polygon)
    xy = np.asarray(polygon, dtype=np.float64)
    points = np.concatenate([
        np.column_stack([xy * scale + (x0, y0), np.full(count, z)])
        for z, x0, y0, scale in sections
    ])  # fmt: skip
    faces: list[tuple[int, ...]] = [tuple(range(count)[::-1])]
    for k in range(len(sections) - 1):
        a, b = k * count, (k + 1) * count
        faces += [(a + j, a + (j + 1) % count, b + (j + 1) % count, b + j) for j in range(count)]
    last = (len(sections) - 1) * count
    faces.append(tuple(last + j for j in range(count)))
    return Mesh(points, faces)


def circle_profile(centre: float, radius: float, steps: int) -> list[tuple[float, float]]:
    """A circle of ``radius`` round ``(centre, 0)`` in the ``(rho, z)`` plane: a torus's section."""
    turns = [2 * math.pi * k / steps for k in range(steps)]
    return [(centre + radius * math.cos(t), radius * math.sin(t)) for t in turns]
