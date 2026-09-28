"""Paves as ``TPave``, ``TPaveText``, ``TPaveLabel`` and ``TPaveStats`` paint them.

The corners of what a pave draws: an outline of no width, a hollow box, a
line too wide for its box, a line that is neither text nor a rule, a label
shrunk to fit or sized in pixels, and a stats box sized by its own text
size or holding nothing it paints.
"""

from __future__ import annotations

import pytest

from test_canvas_draw import fills, lines, make, polylines, prim, words
from xrdroot.buffer import Listed

#: A pave's corners in NDC, as a box ``fOption`` places in NDC.
CORNERS = {"fX1NDC": 0.1, "fY1NDC": 0.1, "fX2NDC": 0.4, "fY2NDC": 0.3, "fOption": "brNDC"}


def _pave(kind, **members):
    return prim(kind, **{**CORNERS, "fBorderSize": 1, **members})


def test_a_pave_outlined_in_a_line_of_no_width_has_no_outline_and_a_hollow_one_no_fill():
    ax = make([(_pave("TPave", fLineWidth=0), "")]).plot().axes[0]
    assert fills(ax) and not lines(ax)
    ax = make([(_pave("TPave", fFillStyle=0), "")]).plot().axes[0]
    assert not fills(ax) and polylines(ax) == [[[70, 450], [70, 350], [280, 350], [280, 450],
                                                 [70, 450]]]  # fmt: skip


def test_a_line_too_wide_for_its_pave_makes_all_its_lines_smaller():
    short = [prim("TLatex", fTitle="a", fTextSize=0.0)]
    long = [prim("TLatex", fTitle="a very long line indeed, too wide", fTextSize=0.0)]
    sizes = []
    for held in (short, long):
        ax = make([(_pave("TPaveText", fLines=held, fTextSize=0.0), "")]).plot().axes[0]
        sizes.append(max(t.get_fontsize() for t in ax.texts))
    assert sizes[1] < sizes[0]


def test_what_is_neither_text_nor_a_rule_in_a_pave_is_passed_over():
    held = Listed([prim("TBox"), prim("TText", fTitle="kept", fTextSize=0.0)])
    ax = make([(_pave("TPaveText", fLines=held, fTextSize=0.0), "")]).plot().axes[0]
    assert words(ax) == ["kept"]


def test_a_label_too_long_for_its_box_is_shrunk_until_it_fits():
    wide = _pave("TPaveLabel", fLabel="a label far too long for its little box", fTextSize=0.0)
    ax = make([(wide, "")]).plot().axes[0]
    ((text,),) = [ax.texts]
    assert text.get_fontsize() < 10


def test_a_label_shrunk_until_its_width_stops_changing_is_left_at_that_size():
    """In a box narrower than a pixel the label's width cannot shrink past a pixel or so, and
    the shrinking stops there."""
    narrow = {**CORNERS, "fX2NDC": 0.1005}
    ax = make([(prim("TPaveLabel", **narrow, fLabel="WW", fTextSize=0.0), "")]).plot().axes[0]
    assert all(t.get_fontsize() < 1 for t in ax.texts)


def test_a_label_in_a_pixel_font_is_its_size_in_pixels_and_a_blank_one_is_not_written():
    ax = make([(_pave("TPaveLabel", fLabel="px", fTextFont=43, fTextSize=20), "")]).plot().axes[0]
    ((text,),) = [ax.texts]
    assert text.get_fontsize() == pytest.approx(18 * 0.72)
    for blank in (" ", "\t"):  # nothing drawn, or nothing FreeType measures
        ax = make([(_pave("TPaveLabel", fLabel=blank), "")]).plot().axes[0]
        assert not ax.texts


def _stats(held, **members):
    return _pave("TPaveStats", fLines=Listed(held), fOptStat=11, **members)


def test_a_stats_box_sized_by_its_own_text_size_writes_at_that_size():
    held = [prim("TLatex", fTitle=text, fTextSize=0.0) for text in ("h", "Entries = 10")]
    ax = make([(_stats(held, fTextSize=0.03), "")]).plot().axes[0]
    assert {round(t.get_fontsize(), 2) for t in ax.texts} == {round(14 * 0.72, 2)}


def test_a_stats_box_of_nothing_it_paints_is_its_box_and_a_bar_line_is_not_measured():
    ax = make([(_stats([prim("TText", fTitle="not TLatex")]), "")]).plot().axes[0]
    assert not ax.texts and fills(ax)
    held = [prim("TLatex", fTitle="h"), prim("TLatex", fTitle="a | b"),
            prim("TLatex", fTitle="Entries = 10")]  # fmt: skip
    ax = make([(_stats(held, fTextSize=0.0), "")]).plot().axes[0]
    assert "a | b" in words(ax)
