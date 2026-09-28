"""``TPainter3dAlgorithms``' moving screen, and the faces a lego or surface is drawn with.

Seen from the front (``fTheta`` 0, ``fPhi`` 0) the unit box's x runs
across the screen and its z up it, scaled by ``2 / sqrt(3)``, so a line of
the box is a line of the screen. The screen's 2000 slices each keep the
highest and lowest point drawn in them (``ModifyScreen``); a line is drawn
only where it is above the one or below the other (``FindVisibleDraw``),
as fractions of it: whether it runs across slices, climbs or falls through
what is drawn inside one slice, or stands upright in a single slice. The
level lines of a face (``FindLevelLines``) are where it crosses each of the
z axis's primary divisions, and a face's edge is left out where nothing of
it shows.
"""

from __future__ import annotations

import math

import pytest

from xrdroot.canvas.lego import SLICES, MovingScreen
from xrdroot.canvas.legofaces import draw_face, level_lines
from xrdroot.canvas.view3d import View3D

#: One slice of the screen, in the unit box's x.
SLICE = 2.2 / SLICES * math.sqrt(3.0) / 2


def _front():
    return MovingScreen(View3D((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), -90.0, 90.0))


def _banded():
    """A screen on which the band from 0.4 to 0.6 high, 0.2 to 0.8 across, is drawn."""
    screen = _front()
    screen.cover((0.2, 0.0, 0.4), (0.8, 0.0, 0.4))
    screen.cover((0.2, 0.0, 0.6), (0.8, 0.0, 0.6))
    return screen


def test_a_line_across_a_clear_screen_is_seen_whole():
    assert _front().visible((0.0, 0.0, 0.2), (1.0, 0.0, 0.4)) == [(0.0, 1.0)]


@pytest.mark.parametrize(("low", "high"), [(0.0, 1.0), (1.0, 0.0)])
def test_a_steep_line_through_a_drawn_band_within_a_slice_is_hidden_where_it_crosses_it(low, high):
    parts = _banded().visible((0.5, 0.0, low), (0.5 + 1.5 * SLICE, 0.0, high))
    assert parts == [(0.0, pytest.approx(0.4)), (pytest.approx(0.6), 1.0)]


def test_an_upright_line_is_seen_below_and_above_what_its_slice_has_drawn():
    screen = _banded()
    assert screen.visible((0.5, 0.0, 0.1), (0.5, 0.0, 0.9)) == [
        (0.0, pytest.approx(0.375)), (pytest.approx(0.625), 1.0)]  # fmt: skip
    assert screen.visible((0.5, 0.0, 0.1), (0.5, 0.0, 0.5)) == [(0.0, pytest.approx(0.75))]
    assert screen.visible((0.5, 0.0, 0.5), (0.5, 0.0, 0.9)) == [(pytest.approx(0.25), 1.0)]
    assert screen.visible((0.5, 0.0, 0.45), (0.5, 0.0, 0.55)) == []


def test_an_upright_line_drawn_downwards_has_its_parts_as_fractions_from_its_top():
    parts = _banded().visible((0.5, 0.0, 0.9), (0.5, 0.0, 0.1))
    assert parts == [(1.0, pytest.approx(0.625)), (pytest.approx(0.375), 0.0)]


def test_a_line_in_the_screens_first_slice_is_seen_against_that_slice_alone():
    screen = _banded()
    left = (-1.1 - 14.5 * 2.2 / SLICES) / (2 / math.sqrt(3.0)) + 0.5
    assert screen.visible((left, 0.0, 0.1), (left, 0.0, 0.9)) == [(0.0, 1.0)]


def test_a_reset_screen_has_nothing_drawn_on_it():
    screen = _banded()
    screen.reset()
    assert screen.visible((0.5, 0.0, 0.1), (0.5, 0.0, 0.9)) == [(0.0, 1.0)]
    screen.reset(-2.0, 2.0)  # a wider screen, of slices twice as wide
    assert screen.dx == pytest.approx(2 * 2.2 / SLICES / 1.1)


def test_grid_levels_are_the_z_axiss_divisions_optimised_or_evenly_spaced_when_negative():
    screen = _front()
    screen.grid_levels(5)
    assert screen.levels == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    screen.grid_levels(-4)
    assert screen.levels == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_a_line_through_many_gaps_in_what_is_drawn_keeps_only_its_first_99_parts():
    screen = _front()
    for i in range(150):
        x = 0.1 + i * 4 * SLICE
        screen.cover((x, 0.0, 0.5), (x + 2 * SLICE, 0.0, 0.5))
    assert len(screen.visible((0.05, 0.0, 0.5), (0.95, 0.0, 0.5))) == 99


# -- faces --------------------------------------------------------------------------------


def _square(bottom, top, left=0.3, right=0.4):
    return [(left, 0.0, bottom), (right, 0.0, bottom), (right, 0.0, top), (left, 0.0, top)]


def test_a_face_has_no_level_lines_before_the_levels_are_set_or_when_it_is_beyond_them():
    screen = _front()
    face = _square(0.1, 0.3)
    assert level_lines(screen, face, [p[2] for p in face]) == []
    screen.grid_levels(-4)
    screen.levels = [0.5, 0.75]
    assert level_lines(screen, face, [p[2] for p in face]) == []
    assert level_lines(screen, _square(0.8, 0.9), [0.8, 0.8, 0.9, 0.9]) == []


def test_a_face_crossing_some_levels_has_a_level_line_at_each_it_reaches():
    screen = _front()
    screen.grid_levels(-4)
    face = _square(0.1, 0.6)
    found = level_lines(screen, face, [p[2] for p in face])
    assert [[round(p[2], 9) for p in line] for line in found] == [[0.25, 0.25], [0.5, 0.5]]
    assert [[round(p[0], 9) for p in line] for line in found] == [[0.4, 0.3], [0.4, 0.3]]


def test_a_face_drawn_with_its_levels_draws_them_dotted_then_its_edges():
    screen = _front()
    screen.grid_levels(-4)
    face = _square(0.1, 0.6)
    draw_face(screen, face, [p[2] for p in face], (2, 1, 1), (2, 1, 3))
    assert [s.style for s in screen.segments] == [3, 3, 1, 1, 1, 1]


def test_a_face_just_behind_one_drawn_leaves_out_even_the_slivers_of_its_edges():
    screen = _front()
    draw_face(screen, _square(0.4, 0.6, 0.2, 0.5), None, (1, 1, 1))
    drawn = len(screen.segments)
    draw_face(screen, _square(0.4 - 1e-12, 0.5), None, (1, 1, 1))
    assert len(screen.segments) == drawn
