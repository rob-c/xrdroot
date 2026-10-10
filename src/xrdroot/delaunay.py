"""``ROOT::Math::Delaunay2D``: the triangles through scattered points, and a surface over them.

``TGraph2D::Interpolate`` - and every histogram a ``TGraph2D`` draws - is a
plane laid over each triangle of the points' Delaunay triangulation: the
height of a point inside a triangle is its corners' heights weighed by how
near it is to each (its barycentric coordinates), and a point outside every
triangle - beyond the points' convex hull - is ``fZout``, zero.

ROOT triangulates the points not where they are but squeezed into a unit
square, each axis shifted by its middle and divided by its span, and as
the triangulation of a set of points depends on the axes' scales it is
made here in the same square. The triangulation itself is Bowyer and
Watson's: each point in turn takes out the triangles whose circumcircles
hold it and fills the hole with triangles to it, starting from one great
triangle round them all whose corners are taken away at the end.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import UnsupportedFeatureError

__all__ = ["Delaunay"]

#: How far out, in the unit square's units, the first great triangle's corners are -
#: and how much farther each time its corners turn out to have cut into the hull.
SUPER, WIDER = 1.0e3, 100.0
#: How many times a great triangle is widened before the points are refused.
ATTEMPTS = 3
#: How far outside a triangle, in its barycentric coordinates, a point on its edge may
#: round to and still be inside it.
EDGE = 1e-12
#: How many points are looked for among the triangles at once.
CHUNK = 256

Array = np.ndarray[Any, Any]


def _orient(points: Array, tri: Array) -> Array:
    """Twice each triangle's signed area: positive when its corners go round anticlockwise."""
    a, b, c = points[tri[:, 0]], points[tri[:, 1]], points[tri[:, 2]]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def _inside_circles(points: Array, tri: Array, p: Array) -> Array:
    """Which anticlockwise triangles' circumcircles hold ``p``: the in-circle determinant."""
    a, b, c = (points[tri[:, k]] - p for k in range(3))
    la, lb, lc = (np.einsum("ij,ij->i", v, v) for v in (a, b, c))
    det = (a[:, 0] * (b[:, 1] * lc - lb * c[:, 1]) - a[:, 1] * (b[:, 0] * lc - lb * c[:, 0])
           + la * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]))  # fmt: skip
    return det > 0


def _hole_edges(bad: Array) -> Array:
    """The edges round the triangles taken out: those only one of them has."""
    edges = np.concatenate([bad[:, [0, 1]], bad[:, [1, 2]], bad[:, [2, 0]]])
    keys = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return edges[counts[inverse.reshape(-1)] == 1]


def _bowyer_watson(points: Array, count: int) -> Array:
    """The triangles through the first ``count`` points, the last three the great triangle's."""
    tri = np.array([[count, count + 1, count + 2]])
    for index in range(count):
        bad = _inside_circles(points, tri, points[index])
        edges = _hole_edges(tri[bad])
        made = np.column_stack([edges, np.full(len(edges), index)])
        tri = np.concatenate([tri[~bad], made])
    return tri[np.all(tri < count, axis=1)]


def _hull_size(points: Array) -> int:
    """How many points lie on the convex hull, its corners and along its edges: Andrew's chain."""
    ordered = sorted({(float(x), float(y)) for x, y in points})

    def half(sequence: list[tuple[float, float]]) -> list[tuple[float, float]]:
        kept: list[tuple[float, float]] = []
        for p in sequence:
            while len(kept) >= 2 and _turn(kept[-2], kept[-1], p) < 0:
                kept.pop()
            kept.append(p)
        return kept

    return len(half(ordered)) + len(half(ordered[::-1])) - 2


def _turn(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _triangulated(unit: Array) -> Array:
    """The triangulation, widened until every triangle of the hull is there: ``2n - 2 - h``."""
    count = len(unit)
    expected = 2 * count - 2 - _hull_size(unit)
    reach = SUPER
    for _ in range(ATTEMPTS):
        great = np.array([[-3.0 * reach, -3.0 * reach], [3.0 * reach, -3.0 * reach],
                          [0.0, 3.0 * reach]])  # fmt: skip
        tri = _bowyer_watson(np.concatenate([unit, great]), count)
        if len(tri) == expected:
            return tri
        reach *= WIDER
    raise UnsupportedFeatureError(
        f"the {count} points could not be triangulated: so many of them lie along the edge of "
        f"their hull that no triangle round them leaves it whole"
    )


class Delaunay:
    """The triangles through a ``TGraph2D``'s points, and the heights over them.

        >>> d = Delaunay([0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 2])
        >>> float(d.interpolate(0.25, 0.5)), float(d.interpolate(2, 2))
        (0.75, 0.0)
    """

    def __init__(self, x: Any, y: Any, z: Any, zout: float = 0.0) -> None:
        self.x, self.y, self.z = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (x, y, z))
        self.zout = float(zout)
        self._offsets = [-(float(v.max()) + float(v.min())) / 2 if len(v) else 0.0
                         for v in (self.x, self.y)]  # fmt: skip
        self._scales = [_scale(v) for v in (self.x, self.y)]
        self._triangles: Array | None = None

    def _unit(self, x: Any, y: Any) -> Array:
        """Points in the unit square ROOT triangulates in."""
        (ox, oy), (sx, sy) = self._offsets, self._scales
        return np.column_stack([(np.asarray(x, np.float64) + ox) * sx,
                                (np.asarray(y, np.float64) + oy) * sy])  # fmt: skip

    @property
    def triangles(self) -> Array:
        """``(n, 3)`` indices of each triangle's corners, anticlockwise."""
        if self._triangles is None:
            self._triangles = self._made()
        return self._triangles

    def _made(self) -> Array:
        unit = self._unit(self.x, self.y)
        _, first = np.unique(unit, axis=0, return_index=True)
        kept = np.sort(first)
        if len(kept) < 3 or _hull_size(unit[kept]) < 3:
            return np.zeros((0, 3), dtype=np.int64)
        tri = kept[_triangulated(unit[kept])]
        flipped = _orient(unit, tri) < 0
        tri[flipped] = tri[flipped][:, [0, 2, 1]]
        return tri

    def interpolate(self, x: Any, y: Any) -> Any:
        """``ComputeZ``: the height of the plane over the triangle each point is in, or ``zout``."""
        scalar = np.ndim(x) == 0 and np.ndim(y) == 0
        query = self._unit(np.reshape(x, -1), np.reshape(y, -1))
        found = np.full(len(query), self.zout)
        for start in range(0, len(query), CHUNK):
            found[start : start + CHUNK] = self._heights(query[start : start + CHUNK])
        return found[0] if scalar else found

    def _heights(self, query: Array) -> Array:
        """Each point's height: the first triangle it is in, by its barycentric coordinates."""
        tri = self.triangles
        heights = np.full(len(query), self.zout)
        if not len(tri) or not len(query):
            return heights
        unit = self._unit(self.x, self.y)
        a, b, c = unit[tri[:, 0]], unit[tri[:, 1]], unit[tri[:, 2]]
        det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        dx = query[:, None, 0] - c[None, :, 0]
        dy = query[:, None, 1] - c[None, :, 1]
        l1 = ((b[:, 1] - c[:, 1]) * dx + (c[:, 0] - b[:, 0]) * dy) / det
        l2 = ((c[:, 1] - a[:, 1]) * dx + (a[:, 0] - c[:, 0]) * dy) / det
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -EDGE) & (l2 >= -EDGE) & (l3 >= -EDGE)
        hit = inside.any(axis=1)
        which = np.argmax(inside, axis=1)[hit]
        rows = np.flatnonzero(hit)
        z = self.z[tri[which]]
        heights[hit] = l1[rows, which] * z[:, 0] + l2[rows, which] * z[:, 1] + l3[rows, which] * z[:, 2]
        return heights


def _scale(values: Array) -> float:
    """``1 / span`` of an axis, or one for an axis of a single value."""
    span = float(values.max() - values.min()) if len(values) else 0.0
    return 1.0 / span if span > 0 else 1.0
