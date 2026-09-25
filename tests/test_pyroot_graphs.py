"""``TGraph``, its errors, ``THStack`` and ``TMultiGraph``."""

from __future__ import annotations

import array
import ctypes

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot import Graph


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_every_constructor_root_has(capsys):
    assert ROOT.TGraph().GetN() == 0
    assert ROOT.TGraph(3).GetN() == 3
    g = ROOT.TGraph(2, array.array("d", [1, 2]), array.array("d", [3, 4]))
    assert ROOT.TGraph([1.0, 2.0], [3.0, 4.0]).GetPointY(1) == 4
    assert ROOT.TGraph(g).GetPointX(0) == 1
    h = ROOT.TH1D("h", "hist", 2, 0, 2)
    h.Fill(0.5, 3.0)
    from_hist = ROOT.TGraphErrors(h)
    assert from_hist.GetN() == 2
    assert from_hist.GetPointY(0) == 3
    assert from_hist.GetErrorX(0) == 0.5
    passed, total = ROOT.TH1D("p", "", 2, 0, 2), ROOT.TH1D("t", "", 2, 0, 2)
    total.Fill(0.5)
    total.Fill(0.5)
    passed.Fill(0.5)
    ratio = ROOT.TGraphAsymmErrors(passed, total)
    assert ratio.GetN() == 1
    assert ratio.GetPointY(0) == 0.5
    assert ratio.GetErrorYlow(0) > 0
    divided = ROOT.TGraphAsymmErrors()
    divided.Divide(passed, total)
    assert divided.GetN() == 1
    assert ROOT.TGraphBentErrors is ROOT.TGraphAsymmErrors
    errors = ROOT.TGraphErrors(
        2, np.array([1.0, 2.0]), np.array([1.0, 2.0]), None, np.array([0.1, 0.2])
    )
    assert errors.GetErrorX(1) == 0
    assert errors.GetErrorY(1) == 0.2


def test_points_grow_move_and_go_as_roots_do():
    g = ROOT.TGraph()
    g.SetPoint(2, 5.0, 6.0)
    g.SetPoint(-1, 0, 0)
    assert g.GetN() == 3
    assert g.GetPointX(0) == 0
    assert g.GetPointX(2) == 5
    g.AddPoint(7.0, 8.0)
    g.SetPointX(0, 1.0)
    g.SetPointY(0, 2.0)
    g.SetPointX(9, 3.0)
    g.SetPointY(10, 4.0)
    assert g.GetN() == 11
    assert g.GetPointY(10) == 4
    assert g.GetPointX(10) == 0
    x, y = ctypes.c_double(0), ctypes.c_double(0)
    assert g.GetPoint(3, x, y) == 3
    assert (x.value, y.value) == (7.0, 8.0)
    assert g.GetPoint(99) == -1
    assert g.GetPointX(99) == -1
    assert g.GetPointY(-1) == -1
    assert g.RemovePoint(10) == 10
    assert g.RemovePoint(99) == -1
    assert g.GetN() == 10
    g.Set(2)
    assert g.GetN() == 2
    assert list(g.GetX()) == [1, 0]
    assert list(g.GetY()) == [2, 0]
    g.GetX()[1] = 9.0
    g.Sort()
    assert list(g.GetX()) == [1, 9]
    assert g._xrd.x.tolist() == [1, 9]
    assert g.GetEX() is None
    assert g.GetErrorX(0) == -1
    assert g.GetErrorY(0) == -1
    g.SetPointError(0, 1.0, 1.0)


def test_error_bars_each_side_and_both():
    e = ROOT.TGraphErrors(1)
    e.SetPointError(0, 0.1, 0.2)
    e.SetPointError(3, 0.3, 0.4)
    assert e.GetN() == 4
    assert list(e.GetEX()) == [0.1, 0, 0, 0.3]
    assert e.GetEY()[3] == 0.4
    assert e.GetErrorXlow(0) == e.GetErrorXhigh(0) == 0.1
    assert e.GetErrorYlow(3) == e.GetErrorYhigh(3) == 0.4
    assert e.GetErrorX(99) == -1
    assert e.GetEXlow() is None
    a = ROOT.TGraphAsymmErrors(1)
    a.SetPointError(0, 0.1, 0.3, 0.2, 0.4)
    assert list(a.GetEXlow()) == [0.1]
    assert list(a.GetEXhigh()) == [0.3]
    assert list(a.GetEYlow()) == [0.2]
    assert list(a.GetEYhigh()) == [0.4]
    assert a.GetErrorX(0) == pytest.approx(np.sqrt(0.05))
    assert a.GetErrorY(0) == pytest.approx(np.sqrt(0.1))
    assert (a.GetErrorXlow(0), a.GetErrorXhigh(0), a.GetErrorYlow(0), a.GetErrorYhigh(0)) == (
        0.1,
        0.3,
        0.2,
        0.4,
    )
    assert a._xrd.classname == "TGraphAsymmErrors"
    assert e._xrd.classname == "TGraphErrors"


def test_print_is_a_line_per_point_with_its_bars(capsys):
    ROOT.TGraph(1, [1.0], [2.0]).Print()
    ROOT.TGraphErrors(1, [1.0], [2.0], [0.1], [0.2]).Print()
    a = ROOT.TGraphAsymmErrors(1)
    a.SetPoint(0, 1, 2)
    a.SetPointError(0, 0.1, 0.2, 0.3, 0.4)
    a.Print()
    assert capsys.readouterr().out.splitlines() == [
        "x[0]=1, y[0]=2",
        "x[0]=1, y[0]=2, ex[0]=0.1, ey[0]=0.2",
        "x[0]=1, y[0]=2, exl[0]=0.1, exh[0]=0.2, eyl[0]=0.3, eyh[0]=0.4",
    ]


def test_what_is_worked_out_from_the_points():
    g = ROOT.TGraph(4, [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    assert g.Eval(2.5) == 5
    assert g.Integral() == 0.0
    assert g.GetMean() == 2.5
    assert g.GetMean(2) == 5
    assert g.GetRMS() == pytest.approx(np.std([1, 2, 3, 4]))
    assert g.GetStdDev(2) == g.GetRMS(2)
    assert g.GetCorrelationFactor() == pytest.approx(1.0)
    assert ROOT.TGraph().GetCovariance() == 0
    assert ROOT.TGraph(1).GetCorrelationFactor() == 0
    result = g.Fit("pol1", "QS")
    assert result.Parameter(1) == pytest.approx(2.0)
    assert g.GetFunction("pol1").GetParameter(1) == pytest.approx(2)
    assert g.GetFunction("nothing") is None
    assert g.GetListOfFunctions().GetSize() == 1


def test_the_frame_is_a_tenth_wider_than_the_points_and_titles_its_axes():
    g = ROOT.TGraph(2, [1.0, 3.0], [10.0, 20.0])
    g.SetTitle("t;the x;the y")
    frame = g.GetHistogram()
    assert frame.GetXaxis().GetXmin() == pytest.approx(0.8)
    assert frame.GetXaxis().GetXmax() == pytest.approx(3.2)
    assert frame.GetMinimum() == pytest.approx(9.0)
    assert frame.GetMaximum() == pytest.approx(21.0)
    assert g.GetXaxis().GetTitle() == "the x"
    assert g.GetYaxis().GetTitle() == "the y"
    assert g.GetTitle() == "t"
    assert frame.GetDirectory() is None
    assert frame.TestBit(ROOT.TH1.kNoStats)
    g.SetMinimum(0)
    g.SetMaximum(50)
    assert (g.GetMinimum(), g.GetMaximum(), frame.GetMaximum()) == (0, 50, 50)
    fresh = ROOT.TGraph(1, [0.0], [-1.0])
    fresh.SetMaximum(5)
    other = fresh.GetHistogram()
    assert other.GetMaximum() == 5
    assert other.GetXaxis().GetXmin() == 0.0
    assert other.GetXaxis().GetXmax() == pytest.approx(1.1)
    negative = ROOT.TGraph(2, [-3.0, -0.1], [1.0, 2.0])
    assert negative.GetHistogram().GetXaxis().GetXmax() == 0.0
    assert ROOT.TGraph().ComputeRange() == (0.0, 0.0, 0.0, 0.0)
    g.SetHistogram(frame)
    g.SetEditable(False)


def test_a_graph_keeps_its_look_and_name_in_what_it_writes():
    g = ROOT.TGraph(1, [1.0], [2.0])
    g.SetName("named")
    g.SetLineColor(ROOT.kRed)
    g.SetMarkerStyle(ROOT.kFullCircle)
    made = g._xrd
    assert made.name == "named"
    assert made._core["TAttLine"]["fLineColor"] == ROOT.kRed
    assert made._core["TAttMarker"]["fMarkerStyle"] == 20
    assert g.ClassName() == "TGraph"
    copy = g.Clone("copy")
    assert copy.GetName() == "copy"
    assert g.GetName() == "named"
    assert ROOT.TGraph(1).Clone().GetN() == 1


def test_a_graph_read_is_the_class_it_was_written_as():
    made = Graph.new("g", [1, 2], [3, 4], xerr=[0.1, 0.2], yerr=([0.1, 0.2], [0.3, 0.4]))
    made._core["TAttLine"]["fLineColor"] = 4
    wrapped = ROOT.TFile  # noqa: F841 - the namespace is loaded
    from xrdroot.pyroot.core.wrapping import wrap

    back = wrap(made)
    assert back.ClassName() == "TGraphAsymmErrors"
    assert back.GetErrorYhigh(1) == 0.4
    assert back.GetLineColor() == 4
    assert back._xrd is made
    assert wrap(made) is back


def test_a_stack_adds_up_its_histograms():
    stack = ROOT.THStack("hs", "a stack")
    assert stack.GetMaximum() == 0.0
    assert stack.GetMinimum() == 0.0
    assert stack.GetHistogram() is None
    a, b = ROOT.TH1F("a", "", 2, 0, 2), ROOT.TH1F("b", "", 2, 0, 2)
    a.Fill(0.5)
    b.Fill(0.5, 2)
    b.Fill(1.5)
    stack.Add(a)
    stack.Add(b, "hist")
    assert stack.GetNhists() == 2
    assert len(stack) == 2
    assert list(stack) == [a, b]
    assert stack.GetMaximum() == 3
    assert stack.GetMaximum("nostack") == 2
    assert stack.GetMinimum() == 0
    assert stack.GetStack().Last().GetBinContent(1) == 3
    assert stack.GetHists().At(1) is b
    assert stack.GetHistogram().GetEntries() == 0
    assert stack.GetXaxis().GetNbins() == 2
    assert stack.GetYaxis() is not None
    assert stack._xrd.name == "hs"
    stack.SetMaximum(10)
    stack.SetMinimum(-1)
    assert (stack.GetMaximum(), stack.GetMinimum()) == (10, -1)
    stack.RecursiveRemove(a)
    assert stack.GetNhists() == 1


def test_a_stack_prints_each_of_its_histograms(capsys):
    stack = ROOT.THStack("hs", "")
    stack.Add(ROOT.TH1F("a", "", 1, 0, 1))
    stack.Print()
    assert capsys.readouterr().out == "TH1.Print Name  = a, Entries= 0, Total sum= 0\n"


def test_a_multigraph_fits_every_graph_together():
    mg = ROOT.TMultiGraph("mg", "graphs")
    assert mg.GetListOfFunctions().GetSize() == 0
    assert mg.GetFunction("pol1") is None
    mg.Add(ROOT.TGraph(2, [1.0, 2.0], [1.0, 2.0]))
    mg.Add(ROOT.TGraph(2, [3.0, 4.0], [3.0, 4.0]), "p")
    assert mg.GetListOfGraphs().GetSize() == 2
    assert mg._xrd.classname == "TMultiGraph"
    result = mg.Fit("pol1", "QS")
    assert result.Parameter(1) == pytest.approx(1.0)
    assert mg.GetFunction("pol1") is not None
    assert mg.GetListOfFunctions().GetSize() == 1
    assert mg.GetFunction("nothing") is None
    assert mg.GetXaxis().GetXmin() == pytest.approx(0.7)
    assert mg.GetHistogram() is mg.GetHistogram()


def test_stacks_and_multigraphs_read_from_members():
    from xrdroot import MultiGraph
    from xrdroot.pyroot.core.wrapping import wrap

    made = MultiGraph("TMultiGraph", {"TNamed": {"fName": "m", "fTitle": "t"},
                                      "fGraphs": [Graph.new("g", [1], [2])]})  # fmt: skip
    back = wrap(made)
    assert back.GetName() == "m"
    assert back.GetListOfGraphs().At(0).GetPointY(0) == 2
