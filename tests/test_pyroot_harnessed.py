"""What the tutorial harness asked of the core next: function lists, templated ``Get``, 2D draws."""

from __future__ import annotations

import sys
import types

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.pyroot.core import troot


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_histograms_list_of_functions_is_its_own_and_keeps_what_is_added():
    h = ROOT.TH1D("h", "", 10, -3, 3)
    h.FillRandom("gaus", 200)
    h.Fit("gaus", "Q")
    listed = h.GetListOfFunctions()
    box, line = ROOT.TNamed("stats", ""), ROOT.TNamed("line", "")
    listed.Add(box)
    listed.AddFirst(line)
    extra = ROOT.TF1("extra", "x", 0, 1)
    h.GetListOfFunctions().AddLast(extra)
    h.GetListOfFunctions().AddFirst(ROOT.TF1("first", "x", 0, 1))
    again = h.GetListOfFunctions()
    expect(
        (again.GetSize(), 5),
        (again.FindObject("stats") is box, True),
        ([item.GetName() for item in again], ["first", "gaus", "extra", "line", "stats"]),
        (len(h._xrd.functions), 3),
    )
    assert again.Remove(box) is box
    assert again.Remove(extra) is extra
    assert again.Remove(ROOT.TNamed("nothing", "")) is None
    again.Clear()
    expect((h.GetListOfFunctions().GetSize(), 0), (h._xrd.functions, []))


def test_a_graphs_list_of_functions_is_its_own_too():
    g = ROOT.TGraph(3, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    g.GetListOfFunctions().Add(ROOT.TNamed("label", ""))
    g.Fit("pol1", "Q")
    names = [item.GetName() for item in g.GetListOfFunctions()]
    expect((names, ["pol1", "label"]), (g.GetFunction("pol1") is not None, True))


def test_get_takes_its_class_in_brackets_as_a_cpp_template_does():
    out = ROOT.TFile("t.root", "RECREATE")
    ROOT.TH1D("h", "", 1, 0, 1)
    out.Write()
    out.Close()
    back = ROOT.TFile("t.root")
    h = back.Get["TH1D"]("h")
    expect(
        (h is back.Get("h"), True),
        (back.GetKey("h").ReadObj[ROOT.TH1D]() is h, True),
        (ROOT.gROOT.Get["TH1"]("nothing"), None),
    )
    assert ROOT.TDirectory.Get.__name__ == "Get"
    back.Close()


def test_axes_that_may_grow_and_bars_are_noted():
    h = ROOT.TH1F("h", "", 3, 0, 3)
    expect((h.SetCanExtend(ROOT.TH1.kAllAxes), 0), (h.CanExtendAllAxes(), True))
    h2 = ROOT.TH2F("h2", "", 2, 0, 2, 2, 0, 2)
    h2.SetCanExtend(ROOT.TH1.kXaxis)
    h.SetBarWidth(0.4)
    h.SetBarOffset(0.1)
    expect(
        (h2.CanExtendAllAxes(), False),
        (h.GetBarWidth(), 0.4),
        (h.GetBarOffset(), 0.1),
        (h._xrd._core["fBarWidth"], 400),
    )


def test_macros_are_loaded_once_by_the_interpreter(monkeypatch):
    seen = []
    execute = types.ModuleType("xrdroot.cint.execute")
    execute.process_line = seen.append
    monkeypatch.setitem(sys.modules, "xrdroot.cint", types.ModuleType("xrdroot.cint"))
    monkeypatch.setitem(sys.modules, "xrdroot.cint.execute", execute)
    monkeypatch.setattr(troot.importlib.util, "find_spec", lambda name: True)
    try:
        assert not ROOT.gInterpreter.IsLoaded("m.C")
        ROOT.gInterpreter.LoadMacro("m.C")
        assert ROOT.gInterpreter.IsLoaded("m.C")
        ROOT.gInterpreter.ExecuteMacro("n.C")
    finally:
        troot.set_line_processor(None)
    assert seen == [".L m.C", ".x n.C"]


def test_a_two_dimensional_histogram_fills_at_random_as_root_does():
    ROOT.gRandom.SetSeed(5)
    ROOT.TF2("f2", "exp(-0.5*(x*x+y*y))", -3, 3, -3, 3)
    h = ROOT.TH2D("h", "", 10, -3, 3, 10, -3, 3)
    h.FillRandom("f2", 2000)
    expect(
        (h.GetEntries(), 2000),
        (round(h.GetMean(1), 6), -0.0066),
        (round(h.GetStdDev(2), 6), 1.028238),
        (h.GetBinContent(5, 5), 124),
        (h.GetBinContent(6, 6), 103),
    )
    g = ROOT.TH2D("g", "", 10, -3, 3, 10, -3, 3)
    g.FillRandom(h, 500)
    expect((g.GetEntries(), 500), (round(g.GetMean(1), 6), 0.011745))
    g.FillRandom(ROOT.TH2D("empty", "", 2, 0, 1, 2, 0, 1), 10)
    g.FillRandom("f2", 0, ROOT.TRandom3(1))
    assert g.GetEntries() == 500
