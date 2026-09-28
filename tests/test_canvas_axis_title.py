"""An axis's title, its fonts of a precise size, and the frame's axes as a pad dresses them.

A ``TAxis`` can centre its title (``CenterTitle``) and turn it about
(``RotateTitle``), which ``TGaxis`` takes as bits; a font whose precision is
3 has its size in pixels rather than as a fraction of the pad. The title's
anchor, alignment and angle are what :func:`xrdroot.canvas.axis.paint_axis`
works out, before any font is measured, and ROOT's own frame put its titles
at the same anchors. The last tests draw whole canvases: a histogram's
``fNdivisions`` over 1000, whose primary divisions scale with its pad's
width, and graphs whose frame ROOT ranges by their points.
"""

from __future__ import annotations

import pytest

from test_canvas_draw import _drawn, filled, make, only, sub, words
from xrdroot import Graph
from xrdroot.canvas.axis import Axis, paint_axis

WIDTH, HEIGHT = 700, 500
ACROSS = (0.1, 0.1, 0.9, 0.1)
UP = (0.1, 0.1, 0.1, 0.9)
TURNED = frozenset({"rotatetitle"})


def pixel(u, v):
    """A point of the pad's NDC in the canvas's whole pixels."""
    return round(u * WIDTH), round((1 - v) * HEIGHT)


def title_of(ends, bits=frozenset(), **members):
    """The title a frame's axis from ``ends`` writes, as ``THistPainter`` paints it (``SDH``)."""
    axis = Axis(*ends, 0.0, 10.0, 510, "SDH", title="T", bits=bits, pad=(WIDTH, HEIGHT), **members)
    title = paint_axis(axis, pixel).labels[-1]
    assert title.text == "T"
    return title


def anchor(label):
    return round(label.u, 4), round(label.v, 4), label.align, round(label.angle, 1)


def test_a_title_is_right_aligned_at_the_far_end_of_its_axis_and_turned_along_it():
    assert anchor(title_of(ACROSS)) == (0.9, 0.044, 32, 0.0)
    assert anchor(title_of(UP))[2:] == (32, 90.0)


def test_a_rotated_title_is_turned_about_and_so_aligned_by_its_other_end():
    assert anchor(title_of(ACROSS, TURNED)) == (0.9, 0.044, 12, 180.0)
    assert anchor(title_of(UP, TURNED))[2:] == (12, 270.0)


def test_a_centred_rotated_title_is_centred_on_the_middle_of_its_axis():
    both = TURNED | {"centertitle"}
    assert anchor(title_of(ACROSS, both)) == (0.5, 0.044, 22, 180.0)
    assert anchor(title_of(UP, both))[1:] == (0.5, 22, 270.0)


def test_a_rotated_title_of_a_reversed_axis_is_aligned_by_its_near_end():
    title = title_of((0.9, 0.1, 0.1, 0.1), TURNED)
    assert (round(title.u, 9), title.align) == (0.1, 32)


def test_a_title_of_a_precise_size_is_that_many_pixels_of_the_pad_across_it():
    # 20 pixels of a 500 pixel high pad is 0.04 of it; of a 700 pixel wide one, 0.04 * 5 / 7.
    assert title_of(ACROSS, title_font=43, title_size=20).v == pytest.approx(0.1 - 1.6 * 0.04)
    assert title_of(UP, title_font=43, title_size=20).u == pytest.approx(0.1 - 1.6 * 20 / 700)
    assert title_of(ACROSS, title_font=43, title_size=20).size == 20


def test_labels_of_a_precise_size_stand_where_that_fraction_of_the_pad_would():
    def labels(**members):
        axis = Axis(0.1, 0.2, 0.9, 0.2, 0.0, 10.0, pad=(WIDTH, HEIGHT), **members)
        return paint_axis(axis, pixel).labels

    precise, relative = labels(label_font=43, label_size=20), labels(label_size=0.04)
    assert [label.v for label in precise] == pytest.approx([label.v for label in relative])
    assert {label.size for label in precise} == {20}


def test_an_unlabelled_linear_axis_still_writes_its_title():
    out = paint_axis(Axis(*ACROSS, 0.0, 10.0, 510, "U", title="T"), pixel)
    assert [label.text for label in out.labels] == ["T"]


# -- the frame's axes ---------------------------------------------------------------


def test_divisions_over_a_thousand_have_their_primaries_scaled_by_the_pads_width():
    h = filled()
    h._core["fXaxis"]["TAttAxis"]["fNdivisions"] = 1010  # ten, in half a canvas five
    pad = sub("p", [(h, "hist")], fXlowNDC=0.0, fWNDC=0.5)
    ax = only(make([(pad, "")]).plot(), "p")
    below = [t.get_text().strip() for t in ax.texts if t.get_position()[1] > 460]
    assert below == ["0", "2", "4", "6", "8", "10"]


def test_a_graph_all_at_one_x_is_framed_a_unit_wide_and_a_tenth_more():
    g = Graph.new("g", [2.0, 2.0], [1.0, 3.0])
    _fig, ax = _drawn(g, "AL")
    assert ax.get_xlim() == pytest.approx((1.9, 3.1))


def test_a_graph_of_no_positive_x_is_framed_to_end_at_zero():
    g = Graph.new("g", [-2.0, -1.0, 0.0], [1.0, 2.0, 3.0])
    _fig, ax = _drawn(g, "AL")
    assert ax.get_xlim() == pytest.approx((-2.2, 0.0))
    assert "0" in words(ax)
