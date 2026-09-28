"""``TPainter3dAlgorithms``' raster screen: a line is drawn only where nothing drawn hides it.

The screen is 1000 by 800 cells over the view's square from -1.1 to 1.1,
and ROOT truncates a point into its cell towards zero, 0.01 of a cell
short; a line is walked a cell at a time, and what is in sight is given as
fractions of it, from the end that is lower on the screen unless it was
turned round.
"""

from __future__ import annotations

import pytest

from xrdroot.canvas.raster3d import MOST, RasterScreen


def test_a_point_falls_in_the_cell_root_truncates_it_into():
    screen = RasterScreen()
    assert screen.cell(-1.1, -1.1) == (0, 0)  # -0.01 truncates to 0, towards zero
    assert screen.cell(0.0, 0.0) == (499, 399)
    assert screen.cell(1.1, 1.1) == (999, 799)


@pytest.mark.parametrize(
    ("p1", "p2"),
    [((-0.5, 0.0), (0.5, 0.1)), ((0.0, -0.5), (0.1, 0.5)), ((0.5, 0.5), (-0.5, -0.4))],
)
def test_a_line_on_a_clear_screen_is_all_in_sight(p1, p2):
    assert RasterScreen().visible(p1, p2) == [(0.0, 1.0)]


@pytest.mark.parametrize(
    ("p1", "p2"),
    [((-0.5, 1.2), (0.5, 1.3)), ((-0.5, -1.3), (0.5, -1.2)), ((1.2, 0.0), (1.3, 0.1)),
     ((-1.3, 0.0), (-1.2, 0.1))],
)  # fmt: skip
def test_a_line_off_the_screen_is_out_of_sight(p1, p2):
    assert RasterScreen().visible(p1, p2) == []


def test_a_face_drawn_hides_the_middle_of_a_line_behind_it():
    screen = RasterScreen()
    screen.fill([(-0.1, -0.5), (0.1, -0.5), (0.1, 0.5), (-0.1, 0.5)])
    (first, second) = screen.visible((-0.5, 0.0), (0.5, 0.0))
    assert first[0] == 0.0 and second[1] == 1.0
    assert first[1] == pytest.approx(0.4, abs=0.01) and second[0] == pytest.approx(0.6, abs=0.01)
    upright = screen.visible((0.0, -0.8), (0.0, 0.8))  # a steep line, through it
    assert len(upright) == 2 and upright[0][1] == pytest.approx(0.19, abs=0.01)
    turned = screen.visible((0.0, 0.8), (0.0, -0.8))  # the same line, from its top
    assert turned == [(1 - end, 1 - start) for start, end in upright[::-1]][::-1]


def test_a_line_wholly_behind_a_face_is_not_drawn_and_a_degenerate_face_covers_its_cell():
    screen = RasterScreen()
    screen.fill([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    assert screen.visible((-0.2, 0.0), (0.2, 0.1)) == []
    screen.fill([(0.9, 0.9)] * 4)
    assert screen.cells[screen.cell(0.9, 0.9)[::-1]]


def test_a_line_through_many_gaps_keeps_only_the_first_hundred():
    screen = RasterScreen()
    screen.cells[:, ::2] = True
    assert len(screen.visible((-1.0, 0.0), (1.0, 0.0))) == MOST
