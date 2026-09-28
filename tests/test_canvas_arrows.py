"""``TArrow::PaintArrow``'s closed heads, filled in the arrow's fill colour, and lines of no width.

A closed head (``|>`` or ``<|``) is a triangle: ROOT fills it in the
arrow's fill colour, unless that is 0, and outlines it in its line. A line
- a ``TLine``, an arrow's shaft or a head's outline - of width 0 is not
drawn at all, as ``TImageDump`` draws none; a head of such an arrow is
still filled.
"""

from __future__ import annotations

import math

import pytest
from matplotlib.colors import to_rgb

from test_canvas_draw import NDC, fills, lines, make, prim

BITS = 0x03000000 | NDC


def _arrow(option, **members):
    return prim("TArrow", fX1=0.1, fY1=0.5, fX2=0.9, fY2=0.5, fOption=option,
                fArrowSize=0.05, fBits=BITS, **members)  # fmt: skip


def test_a_closed_head_is_filled_in_the_arrows_fill_colour_and_outlined_in_its_line():
    ax = make([(_arrow("<|>", fFillColor=2), "")]).plot().axes[0]
    heads = fills(ax)
    assert [to_rgb(head.get_facecolor()) for head in heads] == [(1.0, 0.0, 0.0)] * 2
    # each tip, and its back corners 0.7 of the size back and 30 degrees either side
    tip, back = heads[0].get_xy()[1], heads[1].get_xy()[0]
    assert tip.tolist() == [630.0, 250.0] and heads[1].get_xy()[1].tolist() == [70.0, 250.0]
    assert back == pytest.approx([94.5, 250 - 24.5 * math.tan(math.pi / 6)])
    assert len(lines(ax)) == 3  # the shaft and the two outlines


def test_an_open_head_is_never_filled_whatever_the_arrows_fill_colour():
    ax = make([(_arrow("<>", fFillColor=2), "")]).plot().axes[0]
    assert not fills(ax)
    assert len(lines(ax)) == 3


def test_a_line_or_an_arrow_of_no_width_draws_no_line_but_its_closed_head_is_still_filled():
    tline = prim("TLine", fX1=0.1, fY1=0.2, fX2=0.9, fY2=0.2, fLineWidth=0, fBits=BITS)
    ax = make([(tline, ""), (_arrow("|>", fFillColor=4, fLineWidth=0), "")]).plot().axes[0]
    assert not lines(ax)
    (head,) = fills(ax)
    assert to_rgb(head.get_facecolor()) == (0.0, 0.0, 1.0)


def _drawn(option, drawn_with=""):
    """The polylines an arrow of ``option``, drawn with ``drawn_with``, is painted as."""
    ax = make([(_arrow(option), drawn_with)]).plot().axes[0]
    return [line.tolist() for a in lines(ax) for line in a.lines]


def test_a_bar_across_either_end_is_drawn_for_a_bar_in_the_option():
    start_bar, end_bar, shaft = _drawn("|-|")
    assert (start_bar[0][0], start_bar[1][0]) == (70, 70)  # upright, across the start
    assert (end_bar[0][0], end_bar[1][0]) == (630, 630)
    assert shaft == [[70, 250], [630, 250]]
    bar, _shaft, head = _drawn("|->")  # the bar read, what is left is an open head
    assert bar[0][0] == bar[1][0] == 70 and head[1] == [630, 250]


def test_a_head_pointing_back_mid_way_is_half_a_head_short_of_the_middle():
    _shaft, head = _drawn("-<-")
    assert head[1] == [338, 250]  # the middle, 350, less half of 0.7 of 0.05 of 700


def test_the_option_an_arrow_is_drawn_with_stands_before_its_own():
    assert len(_drawn("|>", drawn_with="")) == 2  # its own: a shaft and a closed head
    assert len(_drawn("|>", drawn_with="-")) == 1  # drawn "-": a line alone
