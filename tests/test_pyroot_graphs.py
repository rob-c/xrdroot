"""``TGraph``, its errors, ``THStack`` and ``TMultiGraph``."""

from __future__ import annotations

import array
import ctypes

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot import Graph


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_every_constructor_root_has(capsys):
    expect(
        (ROOT.TGraph().GetN(), 0),
        (ROOT.TGraph(3).GetN(), 3),
    )
    g = ROOT.TGraph(2, array.array("d", [1, 2]), array.array("d", [3, 4]))
    expect(
        (ROOT.TGraph([1.0, 2.0], [3.0, 4.0]).GetPointY(1), 4),
        (ROOT.TGraph(g).GetPointX(0), 1),
    )
    h = ROOT.TH1D("h", "hist", 2, 0, 2)
    h.Fill(0.5, 3.0)
    from_hist = ROOT.TGraphErrors(h)
    expect(
        (from_hist.GetN(), 2),
        (from_hist.GetPointY(0), 3),
        (from_hist.GetErrorX(0), 0.5),
    )
    passed, total = ROOT.TH1D("p", "", 2, 0, 2), ROOT.TH1D("t", "", 2, 0, 2)
    total.Fill(0.5)
    total.Fill(0.5)
    passed.Fill(0.5)
    ratio = ROOT.TGraphAsymmErrors(passed, total)
    expect(
        (ratio.GetN(), 1),
        (ratio.GetPointY(0), 0.5),
        (bool(ratio.GetErrorYlow(0) > 0), True),
    )
    divided = ROOT.TGraphAsymmErrors()
    divided.Divide(passed, total)
    expect(
        (divided.GetN(), 1),
        (bool(ROOT.TGraphBentErrors is not ROOT.TGraphAsymmErrors), True),
    )
    bent = ROOT.TGraphBentErrors(
        2, np.array([1.0, 2.0]), np.array([3.0, 4.0]),
        np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6]), np.array([0.7, 0.8]),
        np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0]), np.array([7.0, 8.0])
    )  # fmt: skip
    expect(
        (bent.GetN(), 2),
        (bent.GetEXlow().tolist(), [0.1, 0.2]),
        (bent.GetEYhigh().tolist(), [0.7, 0.8]),
        (bent.GetEXlowd().tolist(), [1.0, 2.0]),
        (bent.GetEXhighd().tolist(), [3.0, 4.0]),
        (bent.GetEYlowd().tolist(), [5.0, 6.0]),
        (bent.GetEYhighd().tolist(), [7.0, 8.0]),
        (bent.ClassName(), "TGraphBentErrors"),
    )
    errors = ROOT.TGraphErrors(
        2, np.array([1.0, 2.0]), np.array([1.0, 2.0]), None, np.array([0.1, 0.2])
    )
    expect(
        (errors.GetErrorX(1), 0),
        (errors.GetErrorY(1), 0.2),
    )


def test_points_grow_move_and_go_as_roots_do():
    g = ROOT.TGraph()
    g.SetPoint(2, 5.0, 6.0)
    g.SetPoint(-1, 0, 0)
    expect(
        (g.GetN(), 3),
        (g.GetPointX(0), 0),
        (g.GetPointX(2), 5),
    )
    g.AddPoint(7.0, 8.0)
    g.SetPointX(0, 1.0)
    g.SetPointY(0, 2.0)
    g.SetPointX(9, 3.0)
    g.SetPointY(10, 4.0)
    expect(
        (g.GetN(), 11),
        (g.GetPointY(10), 4),
        (g.GetPointX(10), 0),
    )
    x, y = ctypes.c_double(0), ctypes.c_double(0)
    expect(
        (g.GetPoint(3, x, y), 3),
        ((x.value, y.value), (7.0, 8.0)),
        (g.GetPoint(99), -1),
        (g.GetPointX(99), -1),
        (g.GetPointY(-1), -1),
        (g.RemovePoint(10), 10),
        (g.RemovePoint(99), -1),
        (g.GetN(), 10),
    )
    g.Set(2)
    expect(
        (g.GetN(), 2),
        (list(g.GetX()), [1, 0]),
        (list(g.GetY()), [2, 0]),
    )
    g.GetX()[1] = 9.0
    g.Sort()
    expect(
        (list(g.GetX()), [1, 9]),
        (g._xrd.x.tolist(), [1, 9]),
        (bool(g.GetEX() is None), True),
        (g.GetErrorX(0), -1),
        (g.GetErrorY(0), -1),
    )
    g.SetPointError(0, 1.0, 1.0)


def test_error_bars_each_side_and_both():
    e = ROOT.TGraphErrors(1)
    e.SetPointError(0, 0.1, 0.2)
    e.SetPointError(3, 0.3, 0.4)
    expect(
        (e.GetN(), 4),
        (list(e.GetEX()), [0.1, 0, 0, 0.3]),
        (e.GetEY()[3], 0.4),
        (bool(e.GetErrorXlow(0) == e.GetErrorXhigh(0) == 0.1), True),
        (bool(e.GetErrorYlow(3) == e.GetErrorYhigh(3) == 0.4), True),
        (e.GetErrorX(99), -1),
        (bool(e.GetEXlow() is None), True),
    )
    a = ROOT.TGraphAsymmErrors(1)
    a.SetPointError(0, 0.1, 0.3, 0.2, 0.4)
    expect(
        (list(a.GetEXlow()), [0.1]),
        (list(a.GetEXhigh()), [0.3]),
        (list(a.GetEYlow()), [0.2]),
        (list(a.GetEYhigh()), [0.4]),
        (a.GetErrorX(0), pytest.approx(np.sqrt(0.05))),
        (a.GetErrorY(0), pytest.approx(np.sqrt(0.1))),
        (
            (a.GetErrorXlow(0), a.GetErrorXhigh(0), a.GetErrorYlow(0), a.GetErrorYhigh(0)),
            (
                0.1,
                0.3,
                0.2,
                0.4,
            ),
        ),
        (a._xrd.classname, "TGraphAsymmErrors"),
        (e._xrd.classname, "TGraphErrors"),
    )


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
    expect(
        (g.Eval(2.5), 5),
        (g.Integral(), 0.0),
        (g.GetMean(), 2.5),
        (g.GetMean(2), 5),
        (g.GetRMS(), pytest.approx(np.std([1, 2, 3, 4]))),
        (g.GetStdDev(2), g.GetRMS(2)),
        (g.GetCorrelationFactor(), pytest.approx(1.0)),
        (ROOT.TGraph().GetCovariance(), 0),
        (ROOT.TGraph(1).GetCorrelationFactor(), 0),
    )
    result = g.Fit("pol1", "QS")
    expect(
        (result.Parameter(1), pytest.approx(2.0)),
        (g.GetFunction("pol1").GetParameter(1), pytest.approx(2)),
        (bool(g.GetFunction("nothing") is None), True),
        (g.GetListOfFunctions().GetSize(), 1),
    )


def test_the_frame_is_a_tenth_wider_than_the_points_and_titles_its_axes():
    g = ROOT.TGraph(2, [1.0, 3.0], [10.0, 20.0])
    g.SetTitle("t;the x;the y")
    frame = g.GetHistogram()
    expect(
        (frame.GetXaxis().GetXmin(), pytest.approx(0.8)),
        (frame.GetXaxis().GetXmax(), pytest.approx(3.2)),
        (frame.GetMinimum(), pytest.approx(9.0)),
        (frame.GetMaximum(), pytest.approx(21.0)),
        (g.GetXaxis().GetTitle(), "the x"),
        (g.GetYaxis().GetTitle(), "the y"),
        (g.GetTitle(), "t"),
        (bool(frame.GetDirectory() is None), True),
        (bool(frame.TestBit(ROOT.TH1.kNoStats)), True),
    )
    g.SetMinimum(0)
    g.SetMaximum(50)
    assert (g.GetMinimum(), g.GetMaximum(), frame.GetMaximum()) == (0, 50, 50)
    fresh = ROOT.TGraph(1, [0.0], [-1.0])
    fresh.SetMaximum(5)
    other = fresh.GetHistogram()
    expect(
        (other.GetMaximum(), 5),
        (other.GetXaxis().GetXmin(), 0.0),
        (other.GetXaxis().GetXmax(), pytest.approx(1.1)),
    )
    negative = ROOT.TGraph(2, [-3.0, -0.1], [1.0, 2.0])
    expect(
        (negative.GetHistogram().GetXaxis().GetXmax(), 0.0),
        (ROOT.TGraph().ComputeRange(), (0.0, 0.0, 0.0, 0.0)),
    )
    g.SetHistogram(frame)
    g.SetEditable(False)


def test_a_graph_keeps_its_look_and_name_in_what_it_writes():
    g = ROOT.TGraph(1, [1.0], [2.0])
    g.SetName("named")
    g.SetLineColor(ROOT.kRed)
    g.SetMarkerStyle(ROOT.kFullCircle)
    made = g._xrd
    expect(
        (made.name, "named"),
        (made._core["TAttLine"]["fLineColor"], ROOT.kRed),
        (made._core["TAttMarker"]["fMarkerStyle"], 20),
        (g.ClassName(), "TGraph"),
    )
    copy = g.Clone("copy")
    expect(
        (copy.GetName(), "copy"),
        (g.GetName(), "named"),
        (ROOT.TGraph(1).Clone().GetN(), 1),
    )


def test_a_graph_read_is_the_class_it_was_written_as():
    made = Graph.new("g", [1, 2], [3, 4], xerr=[0.1, 0.2], yerr=([0.1, 0.2], [0.3, 0.4]))
    made._core["TAttLine"]["fLineColor"] = 4
    wrapped = ROOT.TFile  # noqa: F841 - the namespace is loaded
    from xrdroot.pyroot.core.wrapping import wrap

    back = wrap(made)
    expect(
        (back.ClassName(), "TGraphAsymmErrors"),
        (back.GetErrorYhigh(1), 0.4),
        (back.GetLineColor(), 4),
        (bool(back._xrd is made), True),
        (bool(wrap(made) is back), True),
    )


def test_a_stack_adds_up_its_histograms():
    stack = ROOT.THStack("hs", "a stack")
    expect(
        (stack.GetMaximum(), 0.0),
        (stack.GetMinimum(), 0.0),
        (bool(stack.GetHistogram() is None), True),
    )
    a, b = ROOT.TH1F("a", "", 2, 0, 2), ROOT.TH1F("b", "", 2, 0, 2)
    a.Fill(0.5)
    b.Fill(0.5, 2)
    b.Fill(1.5)
    stack.Add(a)
    stack.Add(b, "hist")
    expect(
        (stack.GetNhists(), 2),
        (len(stack), 2),
        (list(stack), [a, b]),
        (stack.GetMaximum(), 3),
        (stack.GetMaximum("nostack"), 2),
        (stack.GetMinimum(), 1),  # the top of the stack's, as ROOT 6.40 has it
        (stack.GetStack().Last().GetBinContent(1), 3),
        (bool(stack.GetHists().At(1) is b), True),
        (stack.GetHistogram().GetEntries(), 0),
        (stack.GetXaxis().GetNbins(), 2),
        (bool(stack.GetYaxis() is not None), True),
        (stack._xrd.name, "hs"),
    )
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


def test_a_multigraph_fits_every_graph_together(monkeypatch):
    from xrdroot.pyroot.graphics import pads

    monkeypatch.setattr(pads, "_CURRENT", [None])  # ROOT's frame is the pad's business
    mg = ROOT.TMultiGraph("mg", "graphs")
    expect(
        (mg.GetListOfFunctions().GetSize(), 0),
        (bool(mg.GetFunction("pol1") is None), True),
    )
    mg.Add(ROOT.TGraph(2, [1.0, 2.0], [1.0, 2.0]))
    mg.Add(ROOT.TGraph(2, [3.0, 4.0], [3.0, 4.0]), "p")
    expect(
        (mg.GetListOfGraphs().GetSize(), 2),
        (mg._xrd.classname, "TMultiGraph"),
    )
    result = mg.Fit("pol1", "QS")
    expect(
        (result.Parameter(1), pytest.approx(1.0)),
        (bool(mg.GetFunction("pol1") is not None), True),
        (mg.GetListOfFunctions().GetSize(), 1),
        (bool(mg.GetFunction("nothing") is None), True),
        (mg.GetXaxis().GetXmin(), pytest.approx(0.85)),
        (bool(mg.GetHistogram() is mg.GetHistogram()), True),
    )


def test_a_multigraph_frame_on_a_log_pad_is_widened_by_its_ratios_logarithm():
    """``TMultiGraph::GetHistogram``: ROOT's own 0.65234 and 79.9602 for two falling graphs on
    a logarithmic pad, whose ends are pushed out by log10 of their ratio, not by a tenth."""
    ROOT.TCanvas().SetLogy()
    energy: list[float] = []
    steep: list[float] = []
    shallow: list[float] = []
    e = 1e-4
    while e < 2e7:
        energy.append(e)
        steep.append(10 * e**-0.1)
        shallow.append(15 * e**-0.15)
        e *= 1.1
    mg = ROOT.TMultiGraph("mg", "curves")
    mg.Add(ROOT.TGraph(len(energy), energy, steep))
    mg.Add(ROOT.TGraph(len(energy), energy, shallow))
    frame = mg.GetHistogram()
    expect(
        (frame.GetMinimum(), pytest.approx(0.65233814596432715)),
        (frame.GetMaximum(), pytest.approx(79.960188962911516)),
        (frame.GetNbinsX(), len(energy)),
        (frame.GetXaxis().GetXmin(), pytest.approx(-998097.69466180552)),
    )


def test_a_multigraph_frame_without_a_log_pad_is_a_twentieth_wider(monkeypatch):
    """A linear frame is ROOT's 5 % either way - not the tenth ``TGraph``'s own frame takes -
    and a multigraph of no graph still gets one, a unit wide."""
    from xrdroot.pyroot.graphics import pads

    monkeypatch.setattr(pads, "_CURRENT", [None])
    mg = ROOT.TMultiGraph("mg", "graphs")
    mg.Add(ROOT.TGraph())
    mg.Add(ROOT.TGraph(2, [1.0, 2.0], [1.0, 2.0]))
    mg.Add(ROOT.TGraph(2, [3.0, 4.0], [3.0, 4.0]))
    frame, empty = mg.GetHistogram(), ROOT.TMultiGraph().GetHistogram()
    expect(
        (frame.GetXaxis().GetXmin(), pytest.approx(0.85)),
        (frame.GetXaxis().GetXmax(), pytest.approx(4.15)),
        (frame.GetMinimum(), pytest.approx(0.85)),
        (frame.GetMaximum(), pytest.approx(4.15)),
        (frame.GetNbinsX(), 100),
        (bool(frame.GetDirectory() is None), True),
        (bool(frame.TestBit(ROOT.TH1.kNoStats)), True),
        (empty.GetXaxis().GetXmin(), pytest.approx(-0.05)),
        (empty.GetMaximum(), pytest.approx(1.05)),
    )


def test_a_log_frames_ends_are_floored_at_a_thousandth_when_points_reach_zero():
    """ROOT reads a bottom at or below zero as a thousandth of the top, a graph at a time, so
    a positive graph's least point does not rescue one that reaches below zero."""
    canvas = ROOT.TCanvas()
    canvas.SetLogx()
    canvas.SetLogy()
    mg = ROOT.TMultiGraph()
    mg.Add(ROOT.TGraph(2, [-5.0, 5.0], [-1.0, 5.0]))
    mg.Add(ROOT.TGraph(2, [2.0, 30.0], [2.0, 3.0]))
    frame = mg.GetHistogram()
    expect(
        (frame.GetXaxis().GetXmin(), pytest.approx(0.012)),
        (frame.GetXaxis().GetXmax(), pytest.approx(48.0)),
        (frame.GetMinimum(), pytest.approx(0.002)),
        (frame.GetMaximum(), pytest.approx(8.0)),
    )


def test_a_frame_of_one_point_is_widened_from_a_unit_span(monkeypatch):
    """ROOT makes the top of a span of no width one more than its bottom before padding it."""
    from xrdroot.pyroot.graphics import pads

    monkeypatch.setattr(pads, "_CURRENT", [None])
    mg = ROOT.TMultiGraph()
    mg.Add(ROOT.TGraph(1, [1.0], [1.0]))
    frame = mg.GetHistogram()
    expect(
        (frame.GetXaxis().GetXmin(), pytest.approx(0.95)),
        (frame.GetXaxis().GetXmax(), pytest.approx(2.05)),
        (frame.GetMinimum(), pytest.approx(0.95)),
        (frame.GetMaximum(), pytest.approx(2.05)),
    )


def test_stacks_and_multigraphs_read_from_members():
    from xrdroot import MultiGraph
    from xrdroot.pyroot.core.wrapping import wrap

    made = MultiGraph("TMultiGraph", {"TNamed": {"fName": "m", "fTitle": "t"},
                                      "fGraphs": [Graph.new("g", [1], [2])]})  # fmt: skip
    back = wrap(made)
    expect(
        (back.GetName(), "m"),
        (back.GetListOfGraphs().At(0).GetPointY(0), 2),
    )


def test_one_side_of_a_points_bar_is_set_and_the_graph_grown_to_it() -> None:
    """``SetPointEXlow`` and its kin: an asymmetric graph's sides; a symmetric one has none."""
    a = ROOT.TGraphAsymmErrors(1)
    a.SetPointEXlow(0, 0.1)
    a.SetPointEXhigh(0, 0.2)
    a.SetPointEYlow(2, 0.3)
    a.SetPointEYhigh(2, 0.4)
    assert (a.GetN(), a.GetErrorXlow(0), a.GetErrorXhigh(0)) == (3, 0.1, 0.2)
    assert (a.GetErrorYlow(2), a.GetErrorYhigh(2)) == (0.3, 0.4)
    e = ROOT.TGraphErrors(1)
    e.SetPointEYlow(0, 0.5)
    assert (e.GetN(), e.GetErrorYlow(0)) == (1, 0.0)


def test_a_function_is_applied_to_every_point_and_its_bars():
    plain = ROOT.TGraph(2, np.array([1.0, 2.0]), np.array([10.0, 20.0]))
    plain.Apply(ROOT.TF2("inv", "-1./y"))
    assert list(plain.GetY()) == [-0.1, -0.05]
    bars = ROOT.TGraphErrors(1, np.array([1.0]), np.array([2.0]), np.array([0.1]), np.array([1.0]))
    bars.Apply(ROOT.TF2("sq", "y*y"))
    assert (bars.GetPointY(0), bars.GetErrorY(0)) == (4.0, 4.0)  # (9 - 1) / 2
    asym = ROOT.TGraphAsymmErrors(1, np.array([1.0]), np.array([2.0]), np.array([0.0]),
                                  np.array([0.0]), np.array([1.0]), np.array([1.0]))  # fmt: skip
    asym.Apply(ROOT.TF2("neg", "-y"))
    assert (asym.GetErrorYlow(0), asym.GetErrorYhigh(0)) == (1.0, 1.0)


def test_a_stacks_extremes_take_the_sums_errors_or_each_histograms_as_root_does():
    """The values of ROOT's ``test_THStack.cxx``."""
    stack = ROOT.THStack("ext", "")
    assert stack.GetMinimum() == 0.0
    held = [ROOT.TH1F(f"e{i}", "", 5, 0, 5) for i in range(3)]
    for hist in held:
        for n in range(1, 6):
            hist.SetBinContent(n, n if n < 4 else 6 - n)
        stack.Add(hist)
    expect(
        (stack.GetMinimum(), 3), (stack.GetMaximum(), 9), (stack.GetMinimum("nostack"), 1),
        (stack.GetMinimum("e"), pytest.approx(1.267949, abs=1e-6)),
        (stack.GetMaximum("e"), pytest.approx(12)),
        (stack.GetMinimum("nostack e"), pytest.approx(0, abs=1e-12)),
        (stack.GetMaximum("nostack e"), pytest.approx(4.732051, abs=1e-6)),
    )
    held[2].SetBinError(4, 16)
    stack.Modified()
    assert stack.GetMinimum("e") == pytest.approx(-10.124515, abs=1e-6)
    assert stack.GetMaximum("nostack e") == pytest.approx(18)
