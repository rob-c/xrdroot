"""The Delaunay surface ``TGraph2D::Interpolate`` lays over scattered points.

Inside the points' hull a height is the plane over the triangle it is in,
so a plane is given back exactly wherever it is asked for; outside, it is
``fZout``. The triangles are made in ROOT's unit square, and a set of
points with no area - fewer than three, or all on a line - has none.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot import delaunay
from xrdroot.delaunay import Delaunay
from xrdroot.errors import UnsupportedFeatureError


def _scattered(count, seed=3):
    rng = np.random.default_rng(seed)
    return rng.uniform(-3, 5, count), rng.uniform(0, 1, count)


def test_a_plane_is_given_back_exactly_inside_the_hull_and_zout_outside():
    x, y = _scattered(80)
    surface = Delaunay(x, y, 2 * x - 3 * y + 1, zout=-7.0)
    qx, qy = np.array([0.1, 1.5, 4.0, 40.0]), np.array([0.5, 0.4, 0.55, 0.5])
    found = surface.interpolate(qx, qy)
    assert found[:3] == pytest.approx(2 * qx[:3] - 3 * qy[:3] + 1, rel=1e-12)
    assert found[3] == -7.0
    assert surface.interpolate(0.1, 0.5) == pytest.approx(2 * 0.1 - 3 * 0.5 + 1)


def test_the_triangles_are_every_one_of_the_hull_each_turning_anticlockwise():
    x, y = _scattered(300, seed=5)
    surface = Delaunay(x, y, np.zeros(300))
    tri = surface.triangles
    unit = surface._unit(x, y)
    assert len(tri) == 2 * 300 - 2 - delaunay._hull_size(unit)
    assert np.all(delaunay._orient(unit, tri) > 0)
    for corners in tri[:40]:  # no point inside any triangle's circumcircle
        assert not delaunay._inside_circles(unit, corners[None, :], unit[0])[0] or 0 in corners


def test_points_with_no_area_make_no_triangles_and_repeated_points_count_once():
    assert len(Delaunay([0, 1], [0, 1], [1, 2]).triangles) == 0
    assert len(Delaunay([0, 1, 2, 3], [0, 1, 2, 3], [0] * 4).triangles) == 0
    square = Delaunay([0, 1, 0, 1, 1], [0, 0, 1, 1, 1], [0, 1, 1, 2, 2])
    assert len(square.triangles) == 2
    assert Delaunay([], [], [])._scales == [1.0, 1.0]
    assert Delaunay([1, 1, 1], [0, 1, 2], [0, 0, 0])._scales[0] == 1.0


def test_many_points_are_looked_for_a_chunk_at_a_time():
    x, y = _scattered(20)
    surface = Delaunay(x, y, x + y)
    qx = np.linspace(-2, 4, delaunay.CHUNK * 2 + 7)
    assert surface.interpolate(qx, np.full(len(qx), 0.5)) == pytest.approx(qx + 0.5, abs=1e-12)
    assert len(Delaunay([0, 1], [0, 1], [0, 1]).interpolate([0.5], [0.5])) == 1


def test_a_great_triangle_that_keeps_cutting_into_the_hull_is_refused(monkeypatch):
    monkeypatch.setattr(delaunay, "SUPER", 1e-6)
    monkeypatch.setattr(delaunay, "WIDER", 1.0)
    x, y = _scattered(30)
    with pytest.raises(UnsupportedFeatureError, match="could not be triangulated"):
        Delaunay(x, y, x).triangles
