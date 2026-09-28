"""A ``TLegend``'s corners: one column, a size of its own, and each kind of symbol.

``TLegend::PaintPrimitives`` gives a legend of one column its whole width
less the margin, writes at the legend's own text size where it has one,
drops a label centred on its row towards the row's bottom when the text is
taller than half the row, and draws each entry's symbol: an outlined box
for ``f``, a bar for ``e`` - with ends for a graph's - and a marker for ``p``.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_canvas_draw import _graph, filled, fills, lines, make, marks, prim, words
from xrdroot import Histogram
from xrdroot.buffer import Listed


def _legend(entries, **members):
    base = {"fX1NDC": 0.5, "fY1NDC": 0.5, "fX2NDC": 0.9, "fY2NDC": 0.9, "fOption": "brNDC",
            "fBorderSize": 1, "fTextSize": 0.0, "fNColumns": 1}  # fmt: skip
    return prim("TLegend", fPrimitives=Listed(entries), **{**base, **members})


def _entry(label, option, **members):
    return prim("TLegendEntry", fLabel=label, fOption=option, fTextSize=0.0, **members)


def test_a_legend_of_one_column_writes_each_label_after_the_margin_down_the_box():
    ax = make([(_legend([_entry("one", "l"), _entry("two", "l")]), "")]).plot().axes[0]
    one, two = (t.get_position() for t in ax.texts)
    assert one[0] == two[0] and one[1] < two[1]


def test_a_legend_with_a_text_size_of_its_own_writes_at_it_and_a_tall_label_sinks_its_row():
    small = make([(_legend([_entry("a", "")], fTextSize=0.03), "")]).plot().axes[0]
    assert {round(t.get_fontsize(), 2) for t in small.texts} == {round(14 * 0.72, 2)}
    rows = [_entry("tall", "", fTextAlign=22), _entry("x", "")]
    tall = make([(_legend(rows, fTextSize=0.2), "")]).plot().axes[0]
    assert "tall" in words(tall)


def test_a_fill_alone_is_outlined_and_an_error_bar_has_ends_only_for_a_graph():
    entries = [_entry("filled", "f", fFillColor=3), _entry("bar", "e", fObject=filled()),
               _entry("graph", "ep", fObject=_graph())]  # fmt: skip
    ax = make([(_legend(entries), "")]).plot().axes[0]
    assert fills(ax)  # the fill, in its own colour
    drawn = [line.tolist() for a in lines(ax) for line in a.lines]
    level = [line for line in drawn if len(line) == 2 and line[0][1] == line[1][1]]
    assert len(level) >= 4  # the box's top and bottom, and the graph's bar's ends
    assert marks(ax)  # the graph's marker


def test_a_two_dimensional_histogram_drawn_lego_same_is_drawn_flat_and_says_so():
    h = Histogram.book("h2", (2, 0.0, 2.0), (2, 0.0, 2.0))
    h.fill(np.array([0.5]), np.array([0.5]))
    with pytest.warns(match="three dimensions"):
        make([(h, "col"), (h, "lego same")]).plot()
