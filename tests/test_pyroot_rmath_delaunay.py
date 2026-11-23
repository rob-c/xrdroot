"""``ROOT.Math.Delaunay2D``: the numbers of ROOT's own ``testDelaunay2D`` gtest.

The fifteen values and the three beyond the hull are the test's, checked against ROOT
6.40's oracle: they are barycentric interpolation over the triangulation ROOT's CDT
library picks, whose diagonal across two cocircular points is the one the test names.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT

Math = ROOT.Math

#: The gtest's own points: a 250 by 100 rectangle split in two, and a triangle above it.
X = [750.0, 1000.0, 1000.0, 900.0, 750.0]
Y = [400.0, 400.0, 500.0, 600.0, 500.0]
Z = [100.0, 100.0, 500.0, 600.0, 250.0]
#: The box the gtest gives, ``d(5,x,y,z,170.,1000.,15.,600.)``.
BOX = (170.0, 1000.0, 15.0, 600.0)


@pytest.fixture
def surface():
    return Math.Delaunay2D(5, X, Y, Z, *BOX)


def _heights(surface, points):
    return [surface.Interpolate(x, y) for x, y in points]


def test_a_height_on_a_horizontal_edge_between_two_triangles_is_the_lines_own(surface):
    """The edge P2-P4, shared by two triangles: 250 at x=750 rising to 500 at x=1000."""
    asking = [(x, 500.0) for x in (760.0, 780.0, 800.0, 900.0)]
    assert _heights(surface, asking) == pytest.approx([260.0, 280.0, 300.0, 400.0], rel=1e-14)


def test_a_height_on_a_vertical_edge_between_two_triangles_is_the_lines_own(surface):
    """The edge P0-P4, shared by two triangles: 100 at y=400 rising to 250 at y=500."""
    asking = [(750.0, y) for y in (420.0, 450.0, 490.0)]
    assert _heights(surface, asking) == pytest.approx([130.0, 175.0, 235.0], rel=1e-14)


def test_a_height_on_the_diagonal_of_two_triangles_is_the_lines_own(surface):
    """P0-P2 is not an edge of the rectangle but the diagonal ROOT's CDT triangulates with:
    100 at (750, 400) rising to 500 at (1000, 500). The other diagonal, which scipy's
    triangulation takes, puts these three points inside the triangles it splits and
    answers 130, 220 and 370 instead."""
    asking = [(800.0, 420.0), (850.0, 440.0), (950.0, 480.0)]
    assert _heights(surface, asking) == pytest.approx([180.0, 260.0, 420.0], rel=1e-14)


def test_a_height_at_a_point_is_that_points_own(surface):
    assert _heights(surface, list(zip(X, Y, strict=True))) == Z


def test_a_point_just_beyond_the_hull_is_the_outer_value_which_is_zero(surface):
    asking = [(750.0 - 0.0001, 450.0), (800.0, 400.0 - 0.0001), (1001.0, 500.0)]
    assert surface.ZOuterValue() == 0.0
    assert _heights(surface, asking) == [0.0, 0.0, 0.0]


def test_the_triangles_are_counted_only_once_something_asks_for_one(surface):
    """ROOT triangulates lazily, so its count is zero until ``Interpolate`` runs."""
    assert surface.NumberOfTriangles() == 0
    surface.FindAllTriangles()
    assert surface.NumberOfTriangles() == 3


def test_the_outer_value_is_the_callers_to_set_and_outside_points_get_it(surface):
    surface.SetZOuterValue(170.0)
    assert (surface.ZOuterValue(), surface.Interpolate(1001.0, 500.0)) == (170.0, 170.0)
    surface.SetZOuterValue()
    assert surface.ZOuterValue() == 0.0


def test_the_normalized_range_is_the_unit_square_whatever_box_was_given(surface):
    assert (surface.XMin(), surface.XMax()) == (-0.5, 0.5)
    assert (surface.YMin(), surface.YMax()) == (-0.5, 0.5)
    from_points = Math.Delaunay2D(5, X, Y, Z)
    assert (from_points.XMin(), from_points.XMax()) == (-0.5, 0.5)


def test_no_points_at_all_is_the_outer_value_everywhere():
    empty = Math.Delaunay2D()
    assert (empty.NumberOfTriangles(), empty.Interpolate(1.0, 2.0)) == (0, 0.0)


def test_points_set_after_none_are_the_ones_interpolated_and_no_points_leave_them():
    surface = Math.Delaunay2D()
    surface.SetInputPoints(5, X, Y, Z, *BOX)
    assert surface.Interpolate(800.0, 420.0) == pytest.approx(180.0, rel=1e-14)
    surface.SetInputPoints(0, X, Y, Z, *BOX)
    assert surface.Interpolate(800.0, 420.0) == pytest.approx(180.0, rel=1e-14)


def test_points_in_a_line_make_no_triangles_so_the_outer_value_answers():
    strip = Math.Delaunay2D(3, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    assert (strip.NumberOfTriangles(), strip.Interpolate(1.0, 1.0)) == (0, 0.0)
    strip.FindAllTriangles()
    assert strip.NumberOfTriangles() == 0
