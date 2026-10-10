"""``ROOT.TScatter`` and ``ROOT.TScatter2D``: points with a colour and a size each, drawn."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh

pytest.importorskip("matplotlib")


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    from xrdroot.pyroot.graphics import hook, pads

    pads.CANVASES.clear()
    pads.set_current(None)
    session = fresh(tmp_path)
    next(session)
    hook.install()
    yield
    next(session, None)


def _scatter(count=6):
    r = ROOT.TRandom3(5)
    x, y = (np.array([100 * r.Rndm() for _ in range(count)]) for _ in range(2))
    c, s = np.arange(count, dtype=float) + 1, np.arange(count, dtype=float) * 10 + 1
    return ROOT.TScatter(count, x, y, c, s), (x, y, c, s)


def test_the_points_their_colours_and_sizes_are_roots_accessors(capsys):
    scatter, (x, y, c, s) = _scatter()
    assert (scatter.GetN(), scatter.ClassName(), scatter.GetName()) == (6, "TScatter", "Scatter")
    assert scatter.GetColor().tolist() == c.tolist() and scatter.GetSize().tolist() == s.tolist()
    assert scatter.GetGraph().GetN() == 6 and scatter.GetGraph().GetX()[0] == x[0]
    scatter.SetName("named")
    assert scatter._xrd.name == "named"
    scatter.Print()
    printed = capsys.readouterr().out
    assert printed.startswith(f"x[0]={x[0]:g}, y[0]={y[0]:g}, color[0]=1, size[0]=1")
    assert ROOT.TScatter(3).GetN() == 3 and ROOT.TScatter().GetN() == 0
    assert ROOT.TScatter(2, [1.0, 2.0], [3.0, 4.0]).GetColor() is None


def test_the_frame_its_axes_and_the_marker_sizes_follow_the_setters():
    scatter, (x, _y, _c, _s) = _scatter()
    scatter.SetTitle("Scatter plot title;X title;Y title;Z title")
    assert scatter.GetHistogram().GetXaxis().GetTitle() == "X title"
    assert scatter.GetZaxis().GetTitle() == "Z title" and scatter.GetYaxis().GetNbins() == 100
    scatter.GetZaxis().SetRangeUser(2.0, 4.0)
    assert scatter._xrd.colour_scale() == (2.0, 4.0)
    assert (scatter.GetMargin(), scatter.GetMaxMarkerSize(), scatter.GetMinMarkerSize()) == (
        0.1, 5.0, 1.0)
    scatter.SetMargin(0.2)
    scatter.SetMaxMarkerSize(3.0)
    scatter.SetMinMarkerSize(0.5)
    assert scatter.GetHistogram().GetXaxis().GetXmin() == pytest.approx(x.min() - 0.2 * np.ptp(x))
    assert scatter._xrd.marker_sizes().max() == 3.0 and scatter._xrd.marker_sizes().min() == 0.5
    scatter.SetHistogram(ROOT.TH2F("own", "own", 2, 0, 1, 2, 0, 1))
    assert scatter.GetHistogram().GetName() == "own"


def test_a_scatter_is_drawn_with_its_frame_and_colour_bar_and_another_over_it(tmp_path):
    canvas = ROOT.TCanvas("c", "c", 300, 300)
    canvas.SetRightMargin(0.14)
    scatter, _ = _scatter()
    scatter.SetMarkerStyle(20)
    scatter.SetTitle("t;x;y;z")
    scatter.GetXaxis().SetRangeUser(20.0, 90.0)
    scatter.Draw("A")
    other, _ = _scatter(4)
    other.SetMarkerStyle(21)
    other.Draw("logc logs")
    assert other._xrd.members["fLogC"] and other._xrd.members["fLogS"]
    canvas.SaveAs(str(tmp_path / "s.png"))
    assert (tmp_path / "s.png").stat().st_size > 3000
    ROOT.TScatter(2, [1.0, 2.0], [3.0, 4.0]).Draw("A")  # no colours: the marker's own
    canvas.SaveAs(str(tmp_path / "plain.png"))


def test_a_scatter_in_space_stands_in_a_box_with_its_markers(tmp_path):
    canvas = ROOT.TCanvas("c", "c", 300, 300)
    r = ROOT.TRandom3(7)
    x, y, z = (np.array([r.Rndm() for _ in range(8)]) for _ in range(3))
    cloud = ROOT.TScatter2D(8, x, y, z, np.arange(8.0) + 1, np.arange(8.0))
    cloud.SetTitle("t;x;y;z;c")
    cloud.SetMarkerStyle(20)
    cloud.Draw("logc")
    assert (cloud.ClassName(), cloud.GetGraph().GetN()) == ("TScatter2D", 8)
    other = ROOT.TScatter2D(8, x, y, z)
    other.Draw("SAME")
    canvas.SaveAs(str(tmp_path / "s3.png"))
    assert (tmp_path / "s3.png").stat().st_size > 3000
