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
