"""The corners of files, functions, graphs and histograms that a script meets now and then."""

from __future__ import annotations

import array
import pathlib

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot import Function, Graph
from xrdroot.pyroot.core import files, funcs, wrapping

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_key_made_here_has_no_time_of_its_own():
    assert files.TKey("k", "", "TH1F").GetDatime().GetYear() > 2000


def test_a_directory_made_in_an_updated_file_is_not_read_from_it():
    ROOT.TFile("u.root", "RECREATE").Close()
    added = ROOT.TFile("u.root", "UPDATE")
    fresh_dir = added.mkdir("brand_new")
    expect(
        (bool(fresh_dir._reader() is None), True),
        (bool(fresh_dir.GetListOfKeys().IsEmpty()), True),
    )
    added.Close()


def test_an_object_that_writes_itself_is_asked_to(tmp_path):
    written = []

    class Own(ROOT.TNamed):
        def _write_into(self, directory, name):
            written.append((directory.GetName(), name))
            return 7

    out = ROOT.TFile("own.root", "RECREATE")
    expect(
        (out.WriteTObject(Own("thing", "")), 7),
        (written, [("own.root", "thing")]),
    )
    out.Close()


def test_trees_are_written_when_their_file_closes_and_read_back_as_trees():
    out = ROOT.TFile("trees.root", "RECREATE")
    tree = ROOT.TTree("t", "a tree")
    x = array.array("d", [0.0])
    tree.Branch("x", x, "x/D")
    for value in (1.0, 2.0, 3.0):
        x[0] = value
        tree.Fill()
    assert (
        out.Get("t") is tree and tree.GetDirectory() is out
        if hasattr(tree, "GetDirectory")
        else True
    )
    out.Close()
    back = ROOT.TFile("trees.root")
    read = back.Get("t")
    expect(
        (read.ClassName(), "TTree"),
        (read.GetEntries(), 3),
        (back.GetKey("t").GetCycle(), 1),
    )
    back.Close()


def test_a_tree_written_by_hand_is_listed_once(capsys):
    out = ROOT.TFile("hand.root", "RECREATE", "given")
    tree = ROOT.TTree("t", "by hand")
    x = array.array("i", [0])
    tree.Branch("x", x, "x/I")
    tree.Fill()
    tree.Write()
    out.ls()
    expect(
        (bool("  KEY: TTree\tt;1\tby hand" in capsys.readouterr().out), True),
        (out._xrd.path, ""),
        (bool(out._xrd.name != ""), True),
    )
    out.Close()
    back = ROOT.TFile("hand.root", "READ", "a title of my own")
    assert back.GetTitle() == "a title of my own"
    back.Close()


def test_closing_from_inside_a_directory_goes_back_to_memory():
    out = ROOT.TFile("inside.root", "RECREATE")
    out.mkdir("sub").cd()
    out.Close()
    assert ROOT.gDirectory == ROOT.gROOT
    files._OPEN.append(out)
    out._writing = object.__new__(type("Closeable", (), {"close": lambda self: None}))
    out.Close()
    assert out not in files._OPEN
    other = ROOT.TFile("other.root", "RECREATE")
    files._OPEN.remove(other)
    ROOT.gROOT.cd()
    other.Close()


def test_tree_classes_without_the_trees_come_back_as_they_were_read(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None if "trees" in name else real(name)
    )
    expect(
        (bool(files._tree(object(), files.TKey("t", "", "TTree")) is None), True),
        (bool(files._tree(object(), files.TKey("h", "", "TH1F")) is None), True),
    )


def test_a_function_whose_signature_cannot_be_read_takes_x_and_p():
    class Opaque:
        __signature__ = property(lambda self: (_ for _ in ()).throw(ValueError()))

        def __call__(self, x, p):
            return x[0]

    assert funcs._arguments(Opaque()) == 2


def test_the_smaller_ways_of_asking_a_function():
    formula = ROOT.TFormula("form", "x")
    assert formula.ClassName() == "TFormula"
    f = ROOT.TF1("f", "[0]*x", 0, 1)
    f.SetParameter(0, 2)
    assert f(np.array([3.0]), np.array([5.0])) == 15
    none = ROOT.TF1("none", "x", 0, 1)
    none.Print("V")
    code = ROOT.TF1("code", lambda x: x[0], 0, 1)
    code.Print("V")
    saved = wrapping.wrap(
        Function.from_members("TF1", Function("s", "x", range=(0, 1)).written_members())
    )
    saved._xrd._model = None
    saved._xrd._tree = None
    saved.Print()


def test_saved_points_are_what_roots_compiled_functions_print(capsys):
    source = ROOT.TFile(str(DATA / "tformula.root"))
    source.Get("func2").Print()
    assert "Function based on a list of points" in capsys.readouterr().out
    source.Close()


def test_a_graph_read_without_attributes_keeps_its_own():
    made = Graph.new("g", [1.0], [2.0])
    for group in ("TAttLine", "TAttFill", "TAttMarker"):
        del made._core[group]
    back = wrapping.wrap(made)
    assert back.GetLineColor() == 1
    back.SetMinimum(-5)
    assert back.GetMinimum() == -5


def test_the_small_histogram_paths():
    h = ROOT.TH1D("h", "", 2, 0, 2)
    h.Fill(0.5)
    assert h.GetStats()[:2] == [1.0, 1.0]
    ROOT.TH1.AddDirectory(False)
    c = h.Clone("c")
    expect(
        (bool(ROOT.gROOT.FindObject("c") is None), True),
        (c.GetName(), "c"),
    )
    ROOT.TH1.AddDirectory(True)
    assert h.GetCumulative().GetSumw2N() == 0


def test_a_stack_without_histograms_has_no_frame_and_takes_one():
    stack = ROOT.THStack()
    assert stack.GetHistogram() is None
    frame = ROOT.TH1F("frame", "", 1, 0, 1)
    stack.SetHistogram(frame)
    expect(
        (bool(stack.GetHistogram() is frame), True),
        (bool(ROOT.TMultiGraph().GetHistogram() is not None), True),
    )
    mg = ROOT.TMultiGraph()
    mg.SetHistogram(frame)
    expect(
        (bool(mg.GetHistogram() is frame), True),
        (mg.GetXaxis().GetNbins(), 1),
    )


def test_an_efficiencys_title_titles_its_histograms_axes():
    e = ROOT.TEfficiency("e", "t", 2, 0, 2)
    e.SetTitle("eff;x axis;y axis")
    expect(
        (e.GetTitle(), "eff"),
        (e.GetTotalHistogram().GetXaxis().GetTitle(), "x axis"),
        (e.GetPassedHistogram().GetYaxis().GetTitle(), "y axis"),
    )
