"""Dashed and thick lines as ``TASImage`` sets their pixels, and how a pixel line is drawn.

A picture ROOT saves in batch goes through ``TImageDump``, which dashes a
line with a quarter of each of ``gStyle``'s dash lengths and hands it to
``TASImage``: ``DrawDashHLine`` and ``DrawDashVLine`` count the dashes a
pixel at a time along a straight line, ``DrawDashZLine`` walks Bresenham's
pixels with the dashes shortened by the cosine of the slope, and
``DrawDashZTLine`` stamps a square brush along a thick line, a dash at a
time. The pixels asserted here for thin and straight dashed lines are the
ones ROOT's own batch PNG of the same canvas sets. A pixel line fills
exactly its pixels in a raster, draws nothing when nothing of it is left,
and is stroked - dashed as ROOT dashes it - in a PDF or SVG.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.image import imread

from test_canvas_draw import NDC, lines, make, prim
from xrdroot.canvas.raster import (
    PixelLine,
    dashed_pixels,
    polyline_pixels,
    root_dashes,
    segment_pixels,
)

BITS = 0x03000000


def _line(x1, y1, x2, y2, **members):
    return prim("TLine", fX1=x1, fY1=y1, fX2=x2, fY2=y2, fBits=BITS | NDC, **members)


def test_a_dashed_style_is_a_quarter_of_each_of_gstyles_dash_lengths_and_a_solid_one_has_none():
    assert root_dashes(2) == (3, 3)
    assert root_dashes(3) == (1, 2)
    assert root_dashes(10) == (20, 10, 1, 10)
    assert root_dashes(1) == ()


def test_solid_lines_are_drawn_straight_by_the_column_or_by_bresenham_from_the_left_or_brushed():
    pixels = zip(*segment_pixels(3, 0, 3, 2, 2), strict=False)
    assert sorted(pixels) == [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
    # given right to left, Bresenham walks from the left and leaves the last pixel, (4, 2), off
    assert set(zip(*segment_pixels(4, 2, 0, 0), strict=False)) == {(0, 0), (1, 1), (2, 1), (3, 2)}
    assert set(zip(*segment_pixels(0, 0, 1, 4), strict=False)) == {(0, 0), (0, 1), (1, 2), (1, 3)}
    brushed = set(
        zip(*segment_pixels(0, 0, 2, 1, 2), strict=False)
    )  # a two-pixel brush on each of 3 pixels
    assert brushed == {(-1, -1), (-1, 0), (0, -1), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0),
                       (2, 1)}  # fmt: skip


def test_a_dashed_line_across_counts_its_dashes_and_gaps_a_pixel_at_a_time_from_its_left():
    xs, ys = dashed_pixels(20, 0, 0, 0, 1, (3, 2))
    assert xs == [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17, 20]
    assert set(ys) == {0}


def test_a_dashed_line_down_is_dashed_from_its_top_as_many_columns_as_it_is_thick():
    xs, ys = dashed_pixels(0, 5, 0, 0, 2, (2, 1))
    assert sorted(zip(ys, xs, strict=False)) == [
        (0, -1), (0, 0), (1, -1), (1, 0), (3, -1), (3, 0), (4, -1), (4, 0)
    ]  # fmt: skip


def test_a_thick_dashed_line_across_is_as_many_rows_as_it_is_thick_from_half_above_it():
    xs, ys = dashed_pixels(0, 10, 7, 10, 3, (2, 3))
    assert sorted(set(ys)) == [9, 10, 11]
    assert sorted(set(xs)) == [0, 1, 5, 6]


def test_a_thin_slanted_dashed_line_keeps_its_first_pixel_and_dashes_the_rest_shortened():
    whole = segment_pixels(0, 0, 10, 5)
    xs, ys = dashed_pixels(0, 0, 10, 5, 1, (3, 2))
    # 3 and 2 times the cosine of the slope round to 3 and 2: the second pixel is the first's
    assert list(zip(xs, ys, strict=False)) == [
        (0, 0),
        (0, 0),
        (1, 1),
        (2, 1),
        (5, 3),
        (6, 3),
        (7, 4),
    ]
    assert set(zip(xs, ys, strict=False)) <= set(zip(*whole, strict=False))


def test_a_thick_slanted_dashed_line_is_a_square_brush_stamped_a_dash_at_a_time():
    xs, ys = dashed_pixels(0, 0, 20, 10, 3, (4, 2))
    stamped = set(zip(xs, ys, strict=False))
    assert {(-1, -1), (1, 1), (19, 10)} <= stamped
    # dashes of 1.79 pixels across and gaps of 3.58: the brush leaves only column 9 bare
    assert sorted({x for x, _ in stamped}) == [*range(-1, 9), *range(10, 20)]
    assert max(x for x, _ in stamped) == 19


def test_a_thick_slanted_dashed_line_is_walked_from_its_left_end_whichever_way_it_is_given():
    down = dashed_pixels(0, 0, 20, 10, 3, (4, 2))
    assert dashed_pixels(20, 10, 0, 0, 3, (4, 2)) == down
    up = dashed_pixels(0, 10, 20, 0, 3, (4, 2))
    assert sorted(zip(up[0], (10 - y for y in up[1]), strict=False)) == sorted(
        zip(*down, strict=False)
    )


def test_dashes_of_an_odd_count_are_not_dashes_and_the_line_is_drawn_solid():
    assert polyline_pixels(np.array([(0, 0), (3, 0)]), 1, (1, 1, 1)).tolist() == [
        [0, 0], [1, 0], [2, 0], [3, 0]]  # fmt: skip


def test_a_line_that_never_leaves_its_first_pixel_sets_none():
    assert polyline_pixels(np.array([(4, 4), (4, 4)])).shape == (0, 2)


def test_a_dashed_tline_on_a_canvas_is_dashed_as_timagedump_dashes_it():
    fig = make([(_line(0.1, 0.2, 0.9, 0.2, fLineStyle=2), ""),
                (_line(0.1, 0.1, 0.9, 0.5, fLineStyle=3, fLineWidth=3), "")],
               width=300, height=300).plot()  # fmt: skip
    across, slanted = lines(fig.axes[0])
    assert across.dashes == (3, 3) and slanted.dashes == (1, 2) and slanted.thick == 3
    row = across.pixels()
    assert set(row[:, 1].tolist()) == {240}
    assert row[:8, 0].tolist() == [30, 31, 32, 36, 37, 38, 42, 43]


def test_a_canvas_saved_as_png_sets_exactly_the_pixels_its_dashed_lines_are_made_of(tmp_path):
    canvas = make([(_line(0.1, 0.1, 0.9, 0.5, fLineStyle=4), ""),
                   (_line(0.1, 0.9, 0.2, 0.3, fLineStyle=5, fLineWidth=2), "")],
                  width=300, height=300)  # fmt: skip
    path = tmp_path / "dashed.png"
    canvas.save(str(path))
    dark = np.argwhere(imread(str(path))[..., :3].sum(axis=2) < 1.5)[:, ::-1]
    wanted = np.unique(np.concatenate([a.pixels() for a in lines(canvas.plot().axes[0])]), axis=0)
    assert sorted(map(tuple, dark.tolist())) == sorted(map(tuple, wanted.tolist()))


def test_a_dashed_line_is_stroked_dashed_in_a_pdf_and_an_svg(tmp_path):
    canvas = make([(_line(0.1, 0.1, 0.9, 0.5, fLineStyle=7, fLineWidth=2), "")])
    canvas.save(str(tmp_path / "dashed.svg"))
    canvas.save(str(tmp_path / "dashed.pdf"))
    svg = (tmp_path / "dashed.svg").read_text()
    assert "stroke-dasharray: 3.6,3.6" in svg.replace("\n", "")
    assert (tmp_path / "dashed.pdf").read_bytes().startswith(b"%PDF")


def _scene(fig):
    ax = fig.axes[0]
    return SimpleNamespace(canvas=(700, 500), display=ax.transData, ax=ax)


def test_a_pixel_line_of_no_polyline_has_no_points_and_draws_nothing():
    fig = make([]).plot()
    empty = PixelLine(_scene(fig), [[(3, 3)]], "black", many=True)
    assert empty.lines == [] and empty.points.shape == (0, 2)
    fig.axes[0].add_artist(empty)
    FigureCanvasAgg(fig).draw()  # nothing to draw, and nothing fails


def test_a_hidden_pixel_line_or_one_clipped_away_sets_no_pixel():
    fig = make([]).plot()
    scene = _scene(fig)
    hidden = PixelLine(scene, [(10, 10), (60, 40)], "red")
    hidden.set_visible(False)
    outside = PixelLine(scene, [(10, 10), (60, 40)], "red", clip=(100, 100, 200, 200))
    assert outside.pixels().shape == (0, 2) and len(hidden.pixels())
    for artist in (hidden, outside):
        fig.axes[0].add_artist(artist)
    raster = FigureCanvasAgg(fig)
    raster.draw()
    rgb = np.asarray(raster.buffer_rgba())[..., :3]
    assert not ((rgb[..., 0] > 200) & (rgb[..., 1] < 50)).any()


def test_a_wide_line_that_goes_nowhere_is_not_drawn_as_asim_line_to_draws_none():
    """A dash of a thick dotted line rounds to no length where ROOT's brush is not stamped."""
    assert segment_pixels(5, 5, 5, 5, 3) == ([], [])
