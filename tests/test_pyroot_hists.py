"""``TH1``, ``TH2``, ``TH3``: made, named, kept, filled and printed as ROOT's are."""

from __future__ import annotations

import array

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_every_storage_kind_is_its_own_class():
    for dimension, args in (
        (1, (2, 0, 1)),
        (2, (2, 0, 1, 2, 0, 1)),
        (3, (2, 0, 1, 2, 0, 1, 2, 0, 1)),
    ):
        for kind in "CSIFD":
            cls = getattr(ROOT, f"TH{dimension}{kind}")
            h = cls(f"h{dimension}{kind}", "t", *args)
            assert h.ClassName() == f"TH{dimension}{kind}" and h.GetDimension() == dimension
            assert h.InheritsFrom("TH1") and h.KIND == kind
    assert (
        ROOT.TH1K is ROOT.TH1D
        and ROOT.TH1F.CLASS_TITLE == "1-Dim histograms (one float per channel)"
    )


def test_variable_bins_are_the_first_n_plus_one_edges():
    edges = array.array("d", [0, 1, 3, 7, 99])
    h = ROOT.TH1D("h", "", 3, edges)
    assert h.GetNbinsX() == 3 and h.GetXaxis().GetXmax() == 7 and h.GetBinWidth(2) == 2
    h2 = ROOT.TH2F("h2", "", 3, edges, 2, 0, 1)
    assert (h2.GetNbinsX(), h2.GetNbinsY(), h2.GetNcells()) == (3, 2, 20)
    h3 = ROOT.TH3D("h3", "", 1, 0, 1, 1, 0, 1, 3, np.array([0.0, 1.0, 2.0, 4.0]))
    assert h3.GetNbinsZ() == 3 and h3.GetZaxis().GetBinUpEdge(3) == 4


def test_a_default_histogram_is_kept_nowhere_and_a_copy_is():
    empty = ROOT.TH1F()
    assert empty.GetName() == "" and empty.GetNbinsX() == 1 and ROOT.gROOT.GetList().IsEmpty()
    h = ROOT.TH1D("h", "title;the x;the y", 4, 0, 4)
    copy = ROOT.TH1D(h)
    assert copy is not h and copy.GetName() == "h" and ROOT.gROOT.FindObject("h") is copy
    assert h.GetXaxis().GetTitle() == "the x" and h.GetYaxis().GetTitle() == "the y"
    assert h.GetTitle() == "title"


def test_the_directory_switches_are_roots(capsys):
    ROOT.TH1.AddDirectory(False)
    assert not ROOT.TH1.AddDirectoryStatus()
    ROOT.TH1D("floating", "", 1, 0, 1)
    assert ROOT.gROOT.FindObject("floating") is None
    ROOT.TH1.AddDirectory(True)
    ROOT.TH1.SetDefaultSumw2(True)
    assert ROOT.TH1.GetDefaultSumw2() and ROOT.TH1D("w", "", 2, 0, 1).GetSumw2N() == 4
    ROOT.TH1.SetDefaultSumw2(False)
    h = ROOT.TH1D("h", "", 1, 0, 1)
    below = ROOT.gROOT.mkdir("below")
    h.SetDirectory(below)
    assert h.GetDirectory() is below and ROOT.gROOT.FindObject("h") is None
    h.SetDirectory(ROOT.nullptr)
    assert h.GetDirectory() is None and below.FindObject("h") is None
    h.SetDirectory(ROOT.gDirectory)
    assert h.GetDirectory() is ROOT.gROOT
    h.SetDirectory(0)
    assert h.GetDirectory() is None


def test_names_titles_bits_and_options():
    h = ROOT.TH1F("h", "t", 2, 0, 1)
    h.SetNameTitle("n", "a;b")
    assert h.GetName() == "n" and h.GetTitle() == "a" and h.GetXaxis().GetTitle() == "b"
    h.SetStats(False)
    assert h.TestBit(ROOT.TH1.kNoStats) and h._xrd._core["TNamed"]["fBits"] & ROOT.TH1.kNoStats
    h.SetStats(True)
    assert not h.TestBit(ROOT.TH1.kNoStats)
    h.SetOption("hist")
    assert h.GetOption() == "hist"
    stats = np.zeros(13)
    h.Fill(0.25)
    assert h.GetStats(stats)[:4] == [1.0, 1.0, 0.25, 0.0625] and stats[2] == 0.25


def test_print_is_roots_summary_and_bin_listing(capsys):
    h = ROOT.TH1D("h1", "one", 1, 0, 1)
    h.Fill(0.5)
    h.Print()
    h.Print("all")
    h.Print("range")
    ROOT.TH2F("h2d", "t", 2, 0, 1, 2, 0, 1).Print("base")
    w = ROOT.TH1D("w", "", 1, 0, 1)
    w.Fill(0.5, 2)
    w.Print("all")
    lines = capsys.readouterr().out.splitlines()
    assert lines[:5] == [
        "TH1.Print Name  = h1, Entries= 1, Total sum= 1",
        "TH1.Print Name  = h1, Entries= 1, Total sum= 1",
        " fSumw[0]=0, x=-0.5",
        " fSumw[1]=1, x=0.5",
        " fSumw[2]=0, x=1.5",
    ]
    assert lines[5:8] == ["TH1.Print Name  = h1, Entries= 1, Total sum= 1", " fSumw[1]=1, x=0.5",
                          "TH1.Print Name  = h2d, Entries= 0, Total sum= 0"]  # fmt: skip
    assert lines[8:10] == [
        "          Title = t",
        "          NbinsX= 2, xmin= 0, xmax=1, NbinsY= 2, ymin= 0, ymax=1",
    ]
    assert (
        lines[11] == " fSumw[0]=0, x=-0.5, error=0" and lines[12] == " fSumw[1]=2, x=0.5, error=2"
    )


def test_ls_is_the_obj_line(capsys):
    ROOT.TH1D("h", "t", 1, 0, 1).ls("noaddr")
    ROOT.TH1D("g", "t", 1, 0, 1).ls()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "OBJ: TH1D\th\tt : 0" and lines[1].startswith("OBJ: TH1D\tg\tt : 0 at: 0x")


def test_filling_by_label_puts_categories_on_the_axis():
    h = ROOT.TH1F("h", "", 3, 0, 3)
    assert h.Fill("apple") == 1 and h.Fill("pear", 2) == 2 and h.Fill("apple") == 1
    assert h.GetXaxis().GetBinLabel(1) == "apple" and h.GetBinContent(2) == 2
    assert h.GetXaxis().FindBin("pear") == 2 and h.GetXaxis().FindFixBin("plum") == -1
    labels = h.GetXaxis().GetLabels()
    assert [str(item) for item in labels] == ["apple", "pear"] and labels.At(1).GetUniqueID() == 2
    assert ROOT.TH1F("g", "", 1, 0, 1).GetXaxis().GetLabels() is None


def test_fill_n_takes_arrays_of_entries():
    h = ROOT.TH1D("h", "", 4, 0, 4)
    h.FillN(3, array.array("d", [0.5, 1.5, 1.5, 9]), None)
    h.FillN(2, [2.5, 3.5], [2.0, 3.0])
    assert [h.GetBinContent(b) for b in range(1, 5)] == [1, 2, 2, 3] and h.GetEntries() == 5
    h2 = ROOT.TH2D("h2", "", 2, 0, 2, 2, 0, 2)
    h2.FillN(2, [0.5, 1.5], [0.5, 1.5], [1.0, 1.0])
    assert h2.GetBinContent(1, 1) == 1 and h2.GetBinContent(2, 2) == 1


def test_bins_are_read_and_set_by_global_or_per_axis_number():
    h2 = ROOT.TH2D("h2", "", 2, 0, 2, 3, 0, 3)
    assert h2.GetBin(1, 2) == 1 + 4 * 2 and h2.GetBin(9, -1) == 3
    h2.SetBinContent(1, 2, 5.0)
    h2.SetBinError(1, 2, 0.5)
    assert h2.GetBinContent(9) == 5 and h2.GetBinContent(1, 2) == 5 and h2.GetBinError(1, 2) == 0.5
    assert h2.GetBinErrorLow(9) == h2.GetBinErrorUp(9) == 0.5 and h2.GetEntries() == 1
    assert h2.GetBinContent(999) == 0 and h2.GetBinError(999) == 0 and h2.RetrieveBinContent(9) == 5
    h2.SetBinContent(999, 1.0)
    h2.SetBinError(999, 1.0)
    bx, by, bz = array.array("i", [0]), array.array("i", [0]), array.array("i", [0])
    assert h2.GetBinXYZ(9, bx, by, bz) == (1, 2, 0) and (bx[0], by[0], bz[0]) == (1, 2, 0)
    h3 = ROOT.TH3F("h3", "", 2, 0, 2, 2, 0, 2, 2, 0, 2)
    h3.SetBinContent(1, 1, 1, 4.0)
    assert h3.GetBinContent(1, 1, 1) == 4 and h3.GetBin(1, 1, 1) == 1 + 4 + 16
    h3.Fill(0.5, 0.5, 0.5, 2.0)
    assert h3.GetBinContent(1, 1, 1) == 6


def test_the_arrays_behind_the_bins():
    h = ROOT.TH1D("h", "", 2, 0, 2)
    h.AddBinContent(1)
    h.AddBinContent(2, 3.0)
    h.Sumw2()
    h.AddBinContent(2, 2.0)
    assert (
        list(h.GetArray()) == [0, 1, 5, 0]
        and h.GetSumw2N() == 4
        and list(h.GetSumw2()) == [0, 0, 0, 0]
    )
    h.Sumw2(False)
    assert h.GetSumw2N() == 0 and len(h.GetSumw2()) == 0
    h.SetBinsLength()
    h.GetArray()[1] = 9
    assert h.GetBinContent(1) == 9
    h.Reset("ICES")
    assert h.GetBinContent(1) == 9
    h.Reset()
    assert h.GetBinContent(1) == 0 and h.GetEntries() == 0


def test_reset_stats_works_the_entries_out_from_the_bins():
    h = ROOT.TH1D("h", "", 2, 0, 2)
    h.SetBinContent(1, 4.0)
    h.SetBinContent(2, 4.0)
    assert h.GetEntries() == 2
    h.ResetStats()
    assert h.GetEntries() == 8 and h.GetMean() == 1.0


def test_python_sees_a_histogram_as_uhi_does():
    h = ROOT.TH1D("h", "", 3, 0, 3)
    h.Fill(0.5, 2.0)
    assert list(h.values()) == [2, 0, 0] and list(h.variances()) == [4, 0, 0]
    assert list(h.counts()) == [1, 0, 0] and len(h.axes) == 1 and h.kind == "COUNT"
    assert len(h) == 5 and bool(h) and list(h) == [2, 0, 0]
    assert h[0] == pytest.approx(2) or h[0].value == pytest.approx(2)
    h[1] = 5.0
    assert h.GetBinContent(2) == 5
    rebinned = h[:: ROOT.__dict__.get("rebin", __import__("xrdroot").rebin)(3)]
    assert rebinned.GetNbinsX() == 1 and isinstance(rebinned, ROOT.TH1)
