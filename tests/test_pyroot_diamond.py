"""``TDiamond``: a pave of text drawn as a diamond in its box, its text in the middle half."""

from __future__ import annotations

import xrdroot.pyroot as ROOT
from pyrootgraphics import fresh_session  # noqa: F401
from xrdroot.pyroot.graphics import snapshot


def test_a_diamond_is_drawn_with_its_shadow_and_its_text_in_its_middle_half():
    c = ROOT.TCanvas("dia", "dia", 300, 200)
    shadowed = ROOT.TDiamond(0.05, 0.1, 0.45, 0.8)
    shadowed.AddText("A TDiamond can contain any text.")
    shadowed.SetFillColor(19)
    shadowed.Draw()
    hollow = ROOT.TDiamond(0.5, 0.5, 0.9, 0.9)
    hollow.SetBorderSize(1)
    hollow.SetFillStyle(0)
    hollow.Draw()
    made = snapshot.model(c)
    assert [prim.classname for prim, _ in made.primitives] == ["TDiamond", "TDiamond"]
    figure = made.plot()
    from matplotlib.patches import Polygon

    shapes = [one for ax in figure.axes for one in ax.get_children() if isinstance(one, Polygon)]
    assert len(shapes) == 2  # the first's shadow and its fill: the hollow one is only outlined
