"""``CONT1``, ``CONT2`` and ``CONT3``: a histogram's contour lines, found and drawn as ROOT does.

``THistPainter::PaintContour`` walks each cell of four neighbouring bin
centres round from its lowest corner, pairs the crossings of each of
twenty levels into segments, and paints each segment on its own: in the
level's colour for ``CONT1``, its line style for ``CONT2``, and the
histogram's own line for ``CONT3``, clipped to the frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_canvas_draw import FRAME, make, polylines
from xrdroot import Histogram
from xrdroot.canvas import contour
from xrdroot.canvas.contour import contour_segments, levels_of


def _cone() -> Histogram:
    """A cone of contents over a grid of 8 by 8: highest in the middle, lowest at the corners."""
    h = Histogram.book("h2", (8, -1.0, 1.0), (8, -1.0, 1.0))
    xs, ys = np.meshgrid(np.linspace(-0.875, 0.875, 8), np.linspace(-0.875, 0.875, 8))
    h.fill(xs.ravel(), ys.ravel(), weight=(2.0 - np.hypot(xs, ys)).ravel())
    return h


def test_the_levels_are_twenty_steps_up_from_the_lowest_content():
    assert levels_of(np.array([[1.0, 3.0]]), 4) == [1.0, 1.5, 2.0, 2.5]
    assert levels_of(np.array([[2.0, 2.0]]), 2) == [1.98, 2.0]  # one value is spread a hundredth


def test_a_level_crossing_a_cell_is_one_segment_where_its_edges_cross_it():
    values = np.array([[0.0, 0.0], [2.0, 2.0]])  # rising across x only
    ((start, end),) = contour_segments([0.0, 1.0], [0.0, 1.0], values, [0.0, 1.0])
    assert {(start.x, start.y), (end.x, end.y)} == {(0.5, 0.0), (0.5, 1.0)}
    assert start.level == end.level == 1
    assert contour_segments([0.0, 1.0], [0.0, 1.0], np.ones((2, 2)), [0.0, 5.0]) == []


def test_a_saddle_is_crossed_twice_and_every_crossing_is_paired_by_its_level():
    values = np.array([[0.0, 2.0], [2.0, 0.0]])
    segments = contour_segments([0.0, 1.0], [0.0, 1.0], values, [0.0, 1.0])
    assert len(segments) == 2
    assert all(start.level == end.level for start, end in segments)


def test_cont3_draws_every_level_in_the_histograms_line_clipped_to_the_frame():
    h = _cone()
    h._core["TAttLine"]["fLineColor"] = 4
    ax = make([(h, "cont3")]).plot().axes[0]
    drawn = [a for a in ax.get_children() if getattr(a, "pixel_clip", None) == FRAME]
    assert {a.color for a in drawn} == {"#0000ff"}
    assert len(polylines(ax, clipped=True)) > 20


@pytest.mark.parametrize(("option", "differ"), [("cont1", "color"), ("cont2", "dashes")])
def test_cont1_colours_each_level_and_cont2_gives_each_a_line_style(option, differ):
    ax = make([(_cone(), option)]).plot().axes[0]
    drawn = [a for a in ax.get_children() if getattr(a, "pixel_clip", None) == FRAME]
    assert len({str(getattr(a, differ)) for a in drawn}) > 1


def test_a_cell_whose_crossings_will_not_pair_is_let_go(monkeypatch):
    monkeypatch.setattr(contour, "REORDERS", -1)
    values = np.array([[0.0, 2.0], [2.0, 0.0]])
    assert contour_segments([0.0, 1.0], [0.0, 1.0], values, [0.0, 1.0]) == []
