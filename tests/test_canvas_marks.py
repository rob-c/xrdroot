"""Markers as ``TImageDump`` draws them, and a graph's points with their error bars.

``TImageDump::DrawPolyMarker`` draws a marker ``8 * fMarkerSize`` pixels
across, less a quarter of a size for each two pixels of a thick outline:
20 (and 8) a filled disc of radius ``int(size / 2)`` half a pixel wider
than 24's (and 4's) ring, 21 a filled square, 25 its outline, and a style
it does not know as a square. ``TGraphPainter::PaintGraphAsymmErrors``
starts each arm of an error bar at the edge of the marker (``cxx`` and
``cyy`` of its size), and draws no arm the marker covers - the bars here
are the pixels ROOT's own PNG of the same graph sets. A graph's line needs
two points, a wholly transparent fill fills nothing, and a point outside the frame
gets no marker.
"""

from __future__ import annotations

import pytest

from test_canvas_draw import filled, fills, lines, make, marks, polylines
from xrdroot import Graph
from xrdroot.canvas import datapaint
from xrdroot.canvas.marks import marker_path, marker_pixels
from xrdroot.plot.model import Points


def _extent(style, size=1.0):
    path, solid, width = marker_path(style, size)
    return path.vertices.min(axis=0).tolist(), path.vertices.max(axis=0).tolist(), solid, width


def _attributes(g, name):
    """A graph's own ``name`` attributes, where it keeps them."""
    return (g.members.get("TGraph") or g.members)[name]


def test_a_disc_is_half_a_pixel_wider_than_the_ring_of_the_same_size():
    assert _extent(20) == ([-4.5, -4.5], [4.5, 4.5], True, 1)
    assert _extent(24) == ([-4.0, -4.0], [4.0, 4.0], False, 1)
    assert _extent(8) == _extent(20) and _extent(4) == _extent(24)


def test_a_thick_outline_shrinks_a_marker_by_a_quarter_of_a_size_for_each_two_pixels():
    assert marker_pixels(2020, 1.0) == 6.0
    assert _extent(2020) == ([-3.5, -3.5], [3.5, 3.5], True, 2)
    assert _extent(4024, 2.0)[1:] == ([6.0, 6.0], False, 4)


@pytest.mark.parametrize(
    ("style", "corners", "solid"),
    [(21, 4, True), (25, 5, False), (22, 3, True), (29, 10, True), (50, 4, True)],
)
def test_a_shaped_marker_is_its_polygon_filled_or_outlined_and_an_unknown_one_a_square(
    style, corners, solid
):
    path, filled_in, _ = marker_path(style, 1.0)
    # A filled polygon's path ends where it began: its last vertex only closes it.
    assert len(path.vertices) == corners + int(solid) and filled_in == solid
    assert path.vertices.min() == -4.0 and path.vertices.max() in (3.0, 4.0)
    if solid:
        assert path.vertices[0].tolist() == path.vertices[-1].tolist()
        assert len({tuple(v) for v in path.vertices.tolist()}) == corners  # none lost


def _big_discs():
    g = Graph.new("g", [1.0, 2.0, 3.0], [2.0, 4.0, 3.0], yerr=[0.5, 0.01, 0.5],
                  xerr=[0.1, 0.01, 0.1])  # fmt: skip
    _attributes(g, "TAttMarker").update(fMarkerStyle=20, fMarkerSize=2.0)
    return g


def test_an_error_bar_starts_at_the_edge_of_a_disc_and_one_the_disc_covers_is_not_drawn():
    ax = make([(_big_discs(), "ap")]).plot().axes[0]
    (centres,) = marks(ax)
    assert centres.get_offsets().tolist() == [[138.5, 350.5], [350.5, 85.5], [562.5, 217.5]]
    bars = polylines(ax, clipped=True)
    assert [[138, 342], [138, 284]] in bars and [[130, 350], [117, 350]] in bars
    assert not [bar for bar in bars if bar[0][0] in range(340, 361)]  # the middle point's


def test_points_without_a_marker_have_their_bars_drawn_and_no_markers(monkeypatch):
    painted = []

    def unmarked(scene, layer):
        painted.append(scene)
        datapaint.paint_points(scene, layer._replace(look=layer.look._replace(marker=None)))

    monkeypatch.setitem(datapaint.PAINTED, Points, unmarked)
    ax = make([(_big_discs(), "ap")]).plot().axes[0]
    assert painted and not marks(ax)
    # without a marker nothing covers the bars: each starts at its point
    assert [[350, 85], [350, 83]] in polylines(ax, clipped=True)


def test_a_point_outside_the_frame_is_given_no_marker():
    outside = Graph.new("g", [20.0, 21.0], [2.0, 4.0])
    ax = make([(filled(), "hist"), (outside, "p")]).plot().axes[0]
    assert not marks(ax)


def test_a_graph_of_one_point_draws_no_line():
    one = Graph.new("one", [1.0], [2.0])
    ax = make([(filled(), "hist"), (one, "l")]).plot().axes[0]
    assert len(lines(ax, clipped=True)) == 1  # the histogram's outline alone


def test_a_graph_filled_wholly_transparent_fills_nothing_and_filled_solid_fills_its_polygon():
    g = Graph.new("g", [1.0, 2.0, 3.0], [2.0, 4.0, 3.0])
    _attributes(g, "TAttFill").update(fFillColor=2, fFillStyle=4000)
    assert not fills(make([(g, "af")]).plot().axes[0])
    _attributes(g, "TAttFill").update(fFillStyle=1001)
    assert len(fills(make([(g, "af")]).plot().axes[0])) == 1
