"""``ROOT.TGraph2D`` and ``ROOT.TGraph2DErrors``: points set, interpolated, fitted and drawn.

A graph grows as ``SetPoint`` past its end grows ROOT's, takes a
``ctypes.c_double`` as PyROOT hands one over, interpolates on its Delaunay
surface, hangs its fit on itself for ``FindObject`` to find, and draws every
option ``TGraph2DPainter`` has.
"""

from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.canvas.graph2d import _flags

pytest.importorskip("iminuit")


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    from xrdroot.pyroot.graphics import hook, pads

    pads.CANVASES.clear()
    pads.set_current(None)
    session = fresh(tmp_path)
    next(session)
    hook.install()  # drawing puts things in pads, as the graphics make it
    yield
    next(session, None)


def _sombrero(cls, count=60):
    r = ROOT.TRandom(7)
    g = cls()
    for i in range(count):
        x, y = r.Uniform(-3, 5), r.Uniform(0, 1)
        g.SetPoint(i, x, y, math.sin(x) * math.cos(3 * y) + x * y)
    return g


def test_points_grow_the_graph_and_come_back_as_they_were_set():
    g = ROOT.TGraph2D()
    assert (g.GetName(), g.GetN(), g.ClassName()) == ("Graph2D", 0, "TGraph2D")
    g.SetPoint(2, ctypes.c_double(1.5), 2, 3)
    g.SetPoint(-1, 0, 0, 0)
    g.AddPoint(4, 5, 6)
    x, y, z = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    assert g.GetN() == 4 and g.GetPoint(2, x, y, z) == 2 and (x.value, z.value) == (1.5, 3.0)
    assert g.GetPoint(9, x, y, z) == -1
    assert (g.GetPointX(3), g.GetPointY(3), g.GetPointZ(3)) == (4.0, 5.0, 6.0)
    assert list(g.GetX()[:4]) == [0, 0, 1.5, 4] and len(g.GetY()) >= 4 and len(g.GetZ()) >= 4
    g.Set(2)
    assert g.GetN() == 2
    g.SetNameTitle("named", "a title")
    assert (g._xrd.name, g._xrd.title) == ("named", "a title")


def test_constructors_take_counts_arrays_and_names():
    g = ROOT.TGraph2D(3, np.array([0.0, 1, 2]), np.array([1.0, 2, 3]), np.array([4.0, 5, 6]))
    assert (g.GetXmin(), g.GetXmax(), g.GetYmin(), g.GetYmax(), g.GetZmin(), g.GetZmax()) == (
        0, 2, 1, 3, 4, 6)
    named = ROOT.TGraph2D("g", "t", 2, [0.0, 1], [0.0, 1], [0.0, 1])
    assert (named.GetName(), named.GetTitle(), named.GetN()) == ("g", "t", 2)
    assert ROOT.TGraph2D(np.array([1.0]), np.array([2.0]), np.array([3.0])).GetN() == 1
    e = ROOT.TGraph2DErrors(2, [0.0, 1], [0.0, 1], [0.0, 1], [0.1, 0.2])
    assert list(e.GetEX()) == [0.1, 0.2] and list(e.GetEZ()) == [0, 0]
    assert (e.GetXminE(), e.GetXmaxE(), e.GetYminE(), e.GetYmaxE()) == (-0.1, 1.2, 0, 1)
    assert (e.GetZminE(), e.GetZmaxE()) == (0, 1)
    assert ROOT.TGraph2D("only a name").GetTitle() == ""


def test_errors_are_set_point_by_point_and_read_back():
    e = ROOT.TGraph2DErrors(1)
    e.SetPointError(3, 0.1, 0.2, ctypes.c_double(0.3))
    e.SetPointError(-1, 1, 1, 1)
    assert (e.GetN(), e.GetErrorX(3), e.GetErrorY(3), e.GetErrorZ(3)) == (4, 0.1, 0.2, 0.3)
    assert e.GetErrorX(7) == -1.0 and list(e.GetEY()[:4]) == [0, 0, 0, 0.2]
    assert e.ClassName() == "TGraph2DErrors"


def test_the_surface_and_the_histogram_follow_the_settings(capsys):
    g = _sombrero(ROOT.TGraph2D)
    assert g.Interpolate(1.0, 0.5) == pytest.approx(g._xrd.interpolate(1.0, 0.5))
    g.SetNpx(2)
    g.SetNpy(50)
    g.SetMargin(0.1)
    g.SetMaxIter(10)
    h = g.GetHistogram()
    assert (h.GetNbinsX(), h.GetNbinsY(), g.GetNpx(), g.GetNpy()) == (4, 50, 4, 50)
    assert g.GetMargin() == 0.1
    assert g.GetXaxis().GetNbins() == 4 and g.GetYaxis().GetNbins() == 50
    assert g.GetZaxis() is not None and g.GetHistogram("empty").GetEntries() == 0
    g.SetMinimum(-5)
    g.SetMaximum(9)
    assert (g.GetMinimum(), g.GetMaximum(), g.GetHistogram().GetMaximum()) == (-5, 9, 9)
    assert ROOT.TGraph2D(2).Interpolate(0, 0) == 0.0
    assert "You need at least 3 points" in capsys.readouterr().err


def test_a_fit_is_hung_on_the_graph_and_found_by_name():
    g = _sombrero(ROOT.TGraph2D)
    f2 = ROOT.TF2("plane", "[0]*x + [1]*y", -3, 5, 0, 1)
    f2.SetParameters(1, 1)
    result = g.Fit(f2, "QS")
    assert int(result) == 0 and g.FindObject("plane") is g.GetFunction("plane")
    assert g.FindObject("plane").GetParameter(0) == pytest.approx(result.Parameter(0))
    assert g.GetListOfFunctions().GetSize() == 1
    copied = g.Clone("copy")
    assert copied.GetName() == "copy" and copied.GetN() == g.GetN() and g.Clone().GetN() == 60


def test_the_confidence_band_is_put_in_a_graph_of_errors():
    g = _sombrero(ROOT.TGraph2D)
    f2 = ROOT.TF2("plane", "[0]*x + [1]*y", -3, 5, 0, 1)
    g.Fit(f2, "Q")
    band = ROOT.TGraph2DErrors(3)
    for i in range(3):
        band.SetPoint(i, i, 0.5, 0)
    ROOT.TVirtualFitter.GetFitter().GetConfidenceIntervals(band)
    assert band.GetZ()[1] == pytest.approx(f2.Eval(1, 0.5)) and band.GetErrorZ(1) > 0


@pytest.mark.parametrize(
    "option", ["tri", "tri1", "tri2", "p0", "pcol", "line", "err", "surf1", "colz"]
)
def test_every_option_draws_without_anything_left_out(option, tmp_path):
    pytest.importorskip("matplotlib")
    import warnings

    g = _sombrero(ROOT.TGraph2DErrors)
    for i in range(g.GetN()):
        g.SetPointError(i, 0.1, 0.02, 0.2)
    g.SetMarkerStyle(20)
    g.SetLineColor(4)
    assert g._xrd.members["TAttMarker"]["fMarkerStyle"] == 20 and g.GetLineColor() == 4
    canvas = ROOT.TCanvas("c", "c", 300, 300)
    g.Draw(option)
    _sombrero(ROOT.TGraph2D, 10).Draw("same p")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        canvas.SaveAs(str(tmp_path / "g.png"))
    assert (tmp_path / "g.png").stat().st_size > 1000


def test_points_with_no_triangles_or_outside_the_box_draw_nothing_of_their_own(tmp_path):
    pytest.importorskip("matplotlib")
    line = ROOT.TGraph2D(3, [0.0, 1, 2], [0.0, 1, 2], [0.0, 1, 2])  # along a line: no area
    far = ROOT.TGraph2DErrors(2, [50.0, 60], [50.0, 60], [0.0, 1], [1.0, 1], [1.0, 1], [1.0, 1])
    canvas = ROOT.TCanvas("c", "c", 200, 200)
    line.Draw("tri")
    far.Draw("err same")
    canvas.SaveAs(str(tmp_path / "none.png"))
    assert (tmp_path / "none.png").exists()


def test_the_options_say_what_the_graph_draws_of_its_own():
    assert _flags("TRI1 SAME")["tri"] == "1" and _flags("tri1 same")["same"]
    assert not _flags("pol")["markers"] and _flags("P0")["p0"] and _flags("PCOL")["pcol"]
    assert _flags("E0 same")["err"] and not _flags("surf1")["err"]
