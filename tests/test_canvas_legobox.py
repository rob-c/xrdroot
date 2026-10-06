"""A three-dimensional histogram drawn ``LEGO`` or ``BOX``: a box in each bin, hidden lines hidden.

``THistPainter::PaintH3BoxRaster`` sizes each bin's box by the cube root of
its share of the highest content, draws the boxes from the front to the
back through the raster screen, lines the back walls of the box round them
at the z axis's divisions where the boxes leave them in sight, and draws
the front edges and the three axes over everything.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_canvas_draw import lines, make, words
from xrdroot import Histogram


def _cube(*contents: float) -> Histogram:
    """A histogram of 2 by 2 by 2 bins over the unit cube, its first bins holding ``contents``."""
    h = Histogram.book("h3", (2, 0.0, 1.0), (2, 0.0, 1.0), (2, 0.0, 1.0))
    corners = [(0.25, 0.25, 0.25), (0.75, 0.75, 0.75), (0.25, 0.75, 0.25)][: len(contents)]
    for (x, y, z), content in zip(corners, contents, strict=False):
        h.fill(np.array([x]), np.array([y]), np.array([z]), weight=np.array([content]))
    h._core["TAttLine"]["fLineColor"] = 4
    return h


@pytest.mark.parametrize("option", ["lego", "box"])
def test_a_three_dimensional_histogram_is_a_box_per_bin_in_a_box_with_its_axes(option):
    ax = make([(_cube(8.0, 1.0), option)]).plot().axes[0]
    assert not ax.axison
    shapes = [[len(line) for line in a.lines] for a in lines(ax) if a.dashes == ()]
    assert [4, 4] in shapes  # the box's two front faces
    assert [a for a in lines(ax) if a.dashes == (1, 2)]  # its back walls, lined
    assert {"0", "0.5", "1"} <= set(words(ax))  # and its axes, labelled


def test_the_fuller_bin_has_the_bigger_box_and_an_empty_one_none():
    def across(*contents: float) -> int:
        ax = make([(_cube(*contents), "lego")]).plot().axes[0]
        edges = [line for a in lines(ax, (0.0, 0.0, 1.0)) for line in a.lines]
        xs = [p[0] for line in edges for p in line]
        return max(xs) - min(xs)

    # a second bin as full as the first has a box as big; an eighth as full, half as big (the
    # cube root of an eighth); an empty one, none - so the boxes reach less far across
    assert across(8.0, 8.0) > across(8.0, 1.0) > across(8.0)


def test_a_histogram_of_equal_contents_draws_no_boxes_and_one_without_lego_or_box_no_box():
    ax = make([(_cube(), "box")]).plot().axes[0]
    assert all(a.color == (0.0, 0.0, 0.0) for a in lines(ax))  # the box round them, and no boxes
    with pytest.warns(match="only LEGO, BOX and ISO"):
        make([(_cube(1.0), "")]).plot()
    ax = make([(_cube(1.0), "iso")]).plot().axes[0]  # a surface round the one full bin
    assert all(a.color == (0.0, 0.0, 0.0) for a in lines(ax))  # and no boxes
    assert [c for c in ax.collections if type(c).__name__ == "PolyCollection"]
    ax = make([(_cube(), "iso")]).plot().axes[0]  # contents all at their mean: no surface
    assert not ax.collections


def test_a_bin_below_the_histograms_own_minimum_has_no_box():
    """``PaintH3BoxRaster`` measures from ``GetMinimum``: ``fMinimum`` when it was set."""

    def blue(h):
        ax = make([(h, "lego")]).plot().axes[0]
        return sum(len(line) for a in lines(ax, (0.0, 0.0, 1.0)) for line in a.lines)

    both = _cube(8.0, 1.0)
    floor = _cube(8.0, 1.0)
    floor.members["TH3"]["TH1"]["fMinimum"] = 2.0
    assert 0 < blue(floor) < blue(both)
