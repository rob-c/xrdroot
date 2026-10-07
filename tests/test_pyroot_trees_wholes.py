"""Histograms and clones arrays a tree holds whole, and what is drawn of them.

``tree201_histograms.C`` keeps a ``TH1F``, a ``TH2F`` and a ``TProfile`` an
entry, and draws each entry's own ``Draw()`` and the numbers its methods
give; ``tree123_clonesarray.C`` keeps a ``TClonesArray`` of lines written
member by member. Both are written, read back and drawn here, and what
ROOT refuses to make of them is refused too.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.cint.runtime import Cell, construct_at
from xrdroot.errors import UnsupportedFeatureError


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def _histograms():
    out = ROOT.TFile("ht.root", "RECREATE")
    tree = ROOT.TTree("T", "test")
    hpx = Cell(ROOT.TH1F("hpx", "px", 10, -4, 4))
    hpxpy = Cell(ROOT.TH2F("hpxpy", "py vs px", 4, -4, 4, 4, -4, 4))
    hprof = Cell(ROOT.TProfile("hprof", "pz", 10, -4, 4, 0, 20))
    tree.Branch("hpx", "TH1F", hpx, 32000, 0)
    tree.Branch("hpxpy", "TH2F", hpxpy, 32000, 0)
    tree.Branch("hprof", "TProfile", hprof, 32000, 0)
    for i in range(6):
        hpx.value.Fill(i * 0.5)
        hpxpy.value.Fill(i * 0.5, -1)
        hprof.value.Fill(i * 0.5, i)
        tree.Fill()
    tree.Write()
    out.Close()
    return ROOT.TFile("ht.root").Get("T")


def test_histograms_held_whole_read_back_as_they_were_at_each_entry():
    back = _histograms()
    hpx, hpxpy, hprof = Cell(None), Cell(None), Cell(None)
    for name, held in (("hpx", hpx), ("hpxpy", hpxpy), ("hprof", hprof)):
        back.SetBranchAddress(name, held)
    back.GetEntry(3)
    assert (hpx.value.GetEntries(), hpxpy.value.GetEntries(), hprof.value.GetEntries()) == (4,) * 3
    assert hpx.value.GetMean() == pytest.approx(0.75)
    assert back.GetBranch("hpxpy").ClassName() == "TBranchElement"  # what pyroot calls one


def test_methods_of_each_entrys_object_are_drawn_as_numbers_or_draw_themselves():
    back = _histograms()
    assert back.Draw("hpx.GetMean():hprof.GetMean()", "", "goff") == 6
    assert np.allclose(back.GetV1(), [0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    assert len(back.GetV2()) == 6 and back.GetV3() is None and back.GetSelectedRows() == 6
    assert ROOT.gROOT.FindObject("htemp").GetTitle() == "hpx.GetMean():hprof.GetMean()"
    ROOT.TCanvas("c", "c")
    assert back.Draw("hpx.Draw()", "", "", 1, 4) == 1
    from xrdroot.pyroot.trees.methods import method_draw

    assert method_draw(back, "hpx.GetMean()", "hpx.GetEntries() > 2", "", None, 0) is None
    assert method_draw(back, "nothing.GetMean()", "", "", None, 0) is None
    assert method_draw(back, "hpx", "", "", None, 0) is None


def test_a_tree_drawn_before_anything_has_no_values_to_give():
    tree = ROOT.TTree("t", "t")
    assert tree.GetV1() is None and tree.GetSelectedRows() == 0


def test_a_clones_array_of_lines_is_written_member_by_member_and_read_back():
    out = ROOT.TFile("clones.root", "RECREATE")
    tree = ROOT.TTree("T", "lines")
    lines = Cell(ROOT.TClonesArray("TLine"))
    tree.Branch("tcl", lines, 256000, 0)
    lines.value.BypassStreamer()
    assert lines.value.CanBypassStreamer() and lines.value.GetClass().GetName() == "TLine"
    for i in range(3):
        lines.value.Clear()
        for k in range(i):
            construct_at(lines.value, k, ROOT.TLine(k, i, k + 1, i + 0.5))
        tree.Fill()
    tree.Write()
    out.Close()
    back = ROOT.TFile("clones.root").Get("T")
    read = Cell(ROOT.TClonesArray("TLine"))
    back.GetBranch("tcl").SetAutoDelete(False)
    back.SetBranchAddress("tcl", read)
    back.GetEntry(2)
    assert read.value.GetEntriesFast() == 2 and read.value.At(1).GetX2() == 2.0
    fresh_read = Cell(None)
    back.SetBranchAddress("tcl", fresh_read)
    back.GetEntry(1)
    assert fresh_read.value.At(0).GetY2() == 1.5


def test_a_clones_array_makes_what_its_slots_hold_when_asked():
    clones = ROOT.TClonesArray(ROOT.TClass.GetClass("TLine"))
    made = clones.ConstructedAt(2)
    assert clones.ConstructedAt(2) is made and clones.GetEntriesFast() == 3


def test_what_root_will_not_make_of_an_object_is_refused():
    tree = ROOT.TTree("t", "t")
    clones = Cell(ROOT.TClonesArray("TLine"))
    with pytest.raises(UnsupportedFeatureError, match="split into a branch per member"):
        tree.Branch("tcl", clones, 32000, 99)

    class Macro:
        _cxx_layout_ = ("Macro", (), (("fX", "double", "", ()),))

    with pytest.raises(UnsupportedFeatureError, match="at split level 0"):
        tree.Branch("m", Macro(), 32000, 0)
    assert tree.Write() == 0  # no branches: nothing to write
