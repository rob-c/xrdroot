"""``xrdroot.geom.shapes`` and ``.mesh``: each solid's surface, where ROOT puts its corners.

The boxes are ``TGeoBBox::GetDX`` and the rest as ROOT 6.40 printed them
for the same solids; the counts are those of ``TGeoTube::SetSegsAndPols``
- four circles of ``n`` points, ``n`` lines along z inside and out, ``n``
spokes at each end.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.geom import Matrix, Mesh
from xrdroot.geom import shapes as build
from xrdroot.geom.mesh import merged

#: How far a point worked out through sines and cosines is from the exact.
ROUNDING = 1e-9


def half(mesh: Mesh) -> list[float]:
    low, high = mesh.extent()
    return list(0.5 * (high - low))


def test_a_box_has_eight_corners_six_faces_and_twelve_edges() -> None:
    mesh = build.box(1.0, 2.0, 3.0, origin=(10.0, 0.0, 0.0))
    assert (len(mesh.points), len(mesh.faces), len(mesh.edges())) == (8, 6, 12)
    assert mesh.points[0].tolist() == [9.0, -2.0, -3.0]
    assert len(mesh.triangles()) == 12


def test_a_tube_is_drawn_with_the_segments_root_draws_it_with() -> None:
    mesh = build.tube(5.0, 10.0, 2.0, steps=20)
    assert (len(mesh.points), len(mesh.edges())) == (80, 160)
    assert half(mesh) == pytest.approx([10.0, 10.0, 2.0])


def test_a_tube_segment_has_flat_ends_and_one_more_point_round_each_ring() -> None:
    mesh = build.tube(5.0, 15.0, 5.0, 90.0, 270.0, steps=360)
    low, high = mesh.extent()
    assert (high[0] - low[0]) / 2 == pytest.approx(7.5)
    assert (high[1] - low[1]) / 2 == pytest.approx(15.0)
    assert len(mesh.points) == 4 * 361


def test_a_segment_running_through_zero_goes_the_way_round() -> None:
    mesh = build.cone(1.0, 0.0, 2.0, 0.0, 3.0, 270.0, 90.0, steps=18)
    assert mesh.points[:, 0].min() == pytest.approx(0.0, abs=ROUNDING)


def test_a_full_cone_with_no_hole_joins_its_walls_at_the_axis() -> None:
    mesh = build.cone(1.0, 0.0, 2.0, 0.0, 3.0, steps=6)
    assert all(a != b for a, b in mesh.edges())
    assert half(mesh)[2] == pytest.approx(1.0)


def test_trapezoids_put_their_corners_where_trap_does() -> None:
    mesh = build.trap(190.0, 0.0, 0.0, 60.0, 40.0, 90.0, 15.0, 120.0, 80.0, 180.0, 15.0)
    assert half(mesh) == pytest.approx([180.0, 120.0, 190.0])  # ROOT's GetDX, GetDY, GetDZ


def test_a_twisted_trapezoid_turns_its_faces_apart() -> None:
    plain = build.gtra(10.0, 0.0, 0.0, 0.0, 5.0, 3.0, 3.0, 0.0, 5.0, 3.0, 3.0, 0.0)
    twisted = build.gtra(10.0, 0.0, 0.0, 90.0, 5.0, 3.0, 3.0, 0.0, 5.0, 3.0, 3.0, 0.0)
    assert np.allclose(plain.points, build.box(3.0, 5.0, 10.0).points)
    assert not np.allclose(twisted.points[:4], plain.points[:4])


def test_the_other_flat_solids_are_eight_cornered_too() -> None:
    for mesh in (build.trd1(1, 2, 3, 4), build.trd2(1, 2, 3, 4, 5), build.para(1, 2, 3, 10, 20, 30),
                 build.arb8(1.0, [(0, 0)] * 8)):  # fmt: skip
        assert (len(mesh.points), len(mesh.faces)) == (8, 6)
    assert half(build.trd2(1, 2, 3, 4, 5)) == [2.0, 4.0, 5.0]


def test_a_polycone_and_a_polygon_are_made_of_their_sections() -> None:
    sections = [(-200, 50, 100), (-50, 50, 80), (50, 50, 80), (200, 50, 100)]
    assert half(build.pcon(0.0, 360.0, sections, steps=4))[2] == 200.0
    polygon = build.pgon(0.0, 360.0, 4, [(-1.0, 0.0, 1.0), (1.0, 0.0, 1.0)])
    assert np.hypot(*polygon.points[0, :2]) == pytest.approx(np.sqrt(2.0))


def test_a_sphere_torus_and_ellipse_reach_as_far_as_their_radii() -> None:
    assert half(build.sphere(1.0, 2.0, steps=360)) == pytest.approx([2.0, 2.0, 2.0], abs=1e-3)
    assert half(build.torus(10.0, 1.0, 2.0, steps=20)) == pytest.approx([12.0, 12.0, 2.0],
                                                                        abs=0.1)  # fmt: skip
    assert half(build.eltu(3.0, 1.0, 5.0, steps=20)) == pytest.approx([3.0, 1.0, 5.0])


def test_a_cut_tube_moves_its_ends_onto_the_cutting_planes() -> None:
    mesh = build.ctub(0.0, 1.0, 1.0, 0.0, 360.0, (0.0, -0.5, -1.0), (0.0, 0.5, 1.0), steps=4)
    top = mesh.points[mesh.points[:, 2] > 0]
    assert top[:, 2].max() == pytest.approx(1.5)


def test_a_paraboloid_and_a_hyperboloid_have_their_radii_at_their_ends() -> None:
    para = build.paraboloid(1.0, 3.0, 2.0, steps=20)
    ends = para.points[para.points[:, 2] == 2.0]
    assert np.hypot(ends[:, 0], ends[:, 1]).max() == pytest.approx(3.0)
    hype = build.hype(1.0, 0.0, 2.0, 45.0, 1.0, steps=20)
    assert np.hypot(*hype.points.T[:2]).max() == pytest.approx(np.sqrt(5.0))


def test_an_extruded_polygon_and_a_tessellated_solid_keep_their_outlines() -> None:
    xtru = build.xtru([(0, 0), (1, 0), (0, 1)], [(0, 0, 0, 1), (5, 1, 1, 2)])
    assert (len(xtru.points), len(xtru.faces)) == (6, 5)
    assert xtru.points[-1].tolist() == [1.0, 3.0, 5.0]
    tessellated = build.tessellated([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                                    [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])  # fmt: skip
    assert len(tessellated.edges()) == 6


def test_meshes_merge_and_move_together() -> None:
    moved = build.box(1, 1, 1).transformed(Matrix(translation=[5, 0, 0]))
    both = merged([build.box(1, 1, 1), moved])
    assert (len(both.points), len(both.faces)) == (16, 12)
    assert both.extent()[1].tolist() == [6.0, 1.0, 1.0]
    assert merged([]).points.shape == (0, 3)
