"""``THistPainter::PaintLego`` and ``PaintLegoAxis``: a 2D histogram's box, blocks and axes.

The z range is ``TableInit``'s: from the lowest bin (or ``fMinimum``) to
the highest (or ``fMaximum``) with five percent of the span on top, and
below unless that passes zero; a histogram of nothing but zeros is drawn
from -1 to 1, and one whose range is empty is widened by half its level
either way; ``PaintLego`` then adds another five percent on top. Seen from
below (a negative ``fTheta``) a block shows its bottom rather than its top.
The box's axes are ``TGaxis`` along its edges nearest the eye: a y axis
that is all but upright is made exactly so, an axis that is seen end on
(z, looked at from straight above) is left out, and a y axis with no title
offset has 1.5.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_canvas_draw import lines, make
from xrdroot import Histogram
from xrdroot.canvas import legofaces, legoplot


def _h2(values, **members):
    h = Histogram.book("h2", (2, 0.0, 2.0), (2, 0.0, 2.0))
    xs, ys = np.meshgrid([0.5, 1.5], [0.5, 1.5], indexing="ij")
    h.fill(xs.ravel(), ys.ravel(), weight=np.asarray(values, float).ravel())
    h.members["TH2"]["TH1"].update(members)
    return h


@pytest.mark.parametrize(
    ("values", "members", "wanted"),
    [
        ([[1, 2], [3, 4]], {}, (0.8425, 0.8425 + (4.15 - 0.8425) * 1.05)),
        ([[0.1, 2], [3, 4]], {}, (0.0, 4.195 * 1.05)),  # 5% below would pass zero
        ([[1, 2], [3, 4]], {"fMinimum": 0.5, "fMaximum": 6.0}, (0.5, 0.5 + 5.5 * 1.05)),
        ([[0, 0], [0, 0]], {}, (-1.0, -1.0 + 2 * 1.05)),
        ([[3, 3], [3, 3]], {}, (1.5, 1.5 + 3 * 1.05)),
        ([[-2, -2], [-2, -2]], {}, (-3.0, -3.0 + 2 * 1.05)),
    ],
)
def test_the_z_range_is_tableinits_with_five_percent_more_on_top(values, members, wanted):
    h = _h2(values, **members)
    assert legoplot._z_range(h.values(), h) == pytest.approx(wanted)


def _faces(monkeypatch, **pad):
    """Every block face a lego of four bins, one negative, is drawn with."""
    drawn = []

    def recorded(screen, points, values, look, levels=None):
        drawn.append([p[2] for p in points])
        return face(screen, points, values, look, levels)

    face = legofaces.draw_face
    monkeypatch.setattr(legofaces, "draw_face", recorded)
    make([(_h2([[1, 2], [3, -1]]), "lego")], **pad).plot()
    return drawn[:-2]  # the last two are the box's back walls


def test_a_lego_seen_from_below_draws_the_bottoms_of_its_blocks_and_not_their_tops(monkeypatch):
    bottom = legoplot._z_range(np.array([1.0, 2.0, 3.0, -1.0]), _h2([[1, 2], [3, -1]]))[0]
    below = _faces(monkeypatch, fTheta=-30.0)
    flat = [zs for zs in below if len(set(zs)) == 1]
    assert flat and all(zs[0] == bottom for zs in flat)
    above = _faces(monkeypatch)
    assert all(len(set(zs)) > 1 or zs[0] != bottom for zs in above)


def _axes(monkeypatch, values=((1, 2), (3, -1)), **pad):
    """The axes a lego is drawn with."""
    seen = []

    def recorded(axis, pixel):
        seen.append(axis)
        return paint(axis, pixel)

    paint = legoplot.paint_axis
    monkeypatch.setattr(legoplot, "paint_axis", recorded)
    fig = make([(_h2(values), "lego")], **pad).plot()
    return seen, fig


def test_a_lego_seen_from_the_front_has_its_y_axis_made_exactly_upright(monkeypatch):
    (xaxis, yaxis, zaxis), _fig = _axes(monkeypatch, fPhi=0.0)
    assert yaxis.x0 == yaxis.x1 and yaxis.y1 > yaxis.y0
    assert xaxis.y0 == xaxis.y1 and zaxis.x0 == zaxis.x1


def test_a_lego_seen_from_straight_above_has_no_z_axis(monkeypatch):
    seen, fig = _axes(monkeypatch, fTheta=90.0)
    assert len(seen) == 2
    assert not [a for a in lines(fig.axes[0]) if a.dashes]  # nor back walls lined at levels


def test_a_y_axis_without_a_title_offset_is_given_one_and_a_half(monkeypatch):
    h = _h2([[1, 2], [3, 4]])
    h.members["TH2"]["TH1"]["fYaxis"]["TAttAxis"]["fTitleOffset"] = 0.0
    seen = []
    paint = legoplot.paint_axis
    monkeypatch.setattr(legoplot, "paint_axis", lambda axis, pixel: seen.append(axis)
                        or paint(axis, pixel))  # fmt: skip
    make([(h, "lego")]).plot()
    assert [axis.title_offset for axis in seen] == [1.0, 1.5, 1.0]
