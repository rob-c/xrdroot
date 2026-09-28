"""``TH1``, ``TH2``, ``TH3``: made, named, kept, filled and printed as ROOT's are."""

from __future__ import annotations

import array

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh


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
            expect(
                (h.ClassName(), f"TH{dimension}{kind}"),
                (h.GetDimension(), dimension),
                (bool(h.InheritsFrom("TH1")), True),
                (h.KIND, kind),
            )
    expect(
        (bool(ROOT.TH1K is ROOT.TH1D), True),
        (ROOT.TH1F.CLASS_TITLE, "1-Dim histograms (one float per channel)"),
    )


def test_variable_bins_are_the_first_n_plus_one_edges():
    edges = array.array("d", [0, 1, 3, 7, 99])
    h = ROOT.TH1D("h", "", 3, edges)
    expect(
        (h.GetNbinsX(), 3),
        (h.GetXaxis().GetXmax(), 7),
        (h.GetBinWidth(2), 2),
    )
    h2 = ROOT.TH2F("h2", "", 3, edges, 2, 0, 1)
    assert (h2.GetNbinsX(), h2.GetNbinsY(), h2.GetNcells()) == (3, 2, 20)
    h3 = ROOT.TH3D("h3", "", 1, 0, 1, 1, 0, 1, 3, np.array([0.0, 1.0, 2.0, 4.0]))
    expect(
        (h3.GetNbinsZ(), 3),
        (h3.GetZaxis().GetBinUpEdge(3), 4),
    )


def test_a_default_histogram_is_kept_nowhere_and_a_copy_is():
    empty = ROOT.TH1F()
    expect(
        (empty.GetName(), ""),
        (empty.GetNbinsX(), 1),
        (bool(ROOT.gROOT.GetList().IsEmpty()), True),
    )
    h = ROOT.TH1D("h", "title;the x;the y", 4, 0, 4)
    copy = ROOT.TH1D(h)
    expect(
        (bool(copy is not h), True),
        (copy.GetName(), "h"),
        (bool(ROOT.gROOT.FindObject("h") is copy), True),
        (h.GetXaxis().GetTitle(), "the x"),
        (h.GetYaxis().GetTitle(), "the y"),
        (h.GetTitle(), "title"),
    )


def test_the_directory_switches_are_roots(capsys):
    ROOT.TH1.AddDirectory(False)
    assert not ROOT.TH1.AddDirectoryStatus()
    ROOT.TH1D("floating", "", 1, 0, 1)
    assert ROOT.gROOT.FindObject("floating") is None
    ROOT.TH1.AddDirectory(True)
    ROOT.TH1.SetDefaultSumw2(True)
    expect(
        (bool(ROOT.TH1.GetDefaultSumw2()), True),
        (ROOT.TH1D("w", "", 2, 0, 1).GetSumw2N(), 4),
    )
    ROOT.TH1.SetDefaultSumw2(False)
    h = ROOT.TH1D("h", "", 1, 0, 1)
    below = ROOT.gROOT.mkdir("below")
    h.SetDirectory(below)
    expect(
        (bool(h.GetDirectory() is below), True),
        (bool(ROOT.gROOT.FindObject("h") is None), True),
    )
    h.SetDirectory(ROOT.nullptr)
    expect(
        (bool(h.GetDirectory() is None), True),
        (bool(below.FindObject("h") is None), True),
    )
    h.SetDirectory(ROOT.gDirectory)
    assert h.GetDirectory() is ROOT.gROOT
    h.SetDirectory(0)
    assert h.GetDirectory() is None


def test_names_titles_bits_and_options():
    h = ROOT.TH1F("h", "t", 2, 0, 1)
    h.SetNameTitle("n", "a;b")
    expect(
        (h.GetName(), "n"),
        (h.GetTitle(), "a"),
        (h.GetXaxis().GetTitle(), "b"),
    )
    h.SetStats(False)
    expect(
        (bool(h.TestBit(ROOT.TH1.kNoStats)), True),
        (bool(h._xrd._core["TNamed"]["fBits"] & ROOT.TH1.kNoStats), True),
    )
    h.SetStats(True)
    assert not h.TestBit(ROOT.TH1.kNoStats)
    h.SetOption("hist")
    assert h.GetOption() == "hist"
    stats = np.zeros(13)
    h.Fill(0.25)
    expect(
        (h.GetStats(stats)[:4], [1.0, 1.0, 0.25, 0.0625]),
        (stats[2], 0.25),
    )


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
    expect(
        (
            lines[:5],
            [
                "TH1.Print Name  = h1, Entries= 1, Total sum= 1",
                "TH1.Print Name  = h1, Entries= 1, Total sum= 1",
                " fSumw[0]=0, x=-0.5",
                " fSumw[1]=1, x=0.5",
                " fSumw[2]=0, x=1.5",
            ],
        ),
        (
            lines[5:8],
            [
                "TH1.Print Name  = h1, Entries= 1, Total sum= 1",
                " fSumw[1]=1, x=0.5",
                "TH1.Print Name  = h2d, Entries= 0, Total sum= 0",
            ],
        ),
        (
            lines[8:10],
            [
                "          Title = t",
                "          NbinsX= 2, xmin= 0, xmax=1, NbinsY= 2, ymin= 0, ymax=1",
            ],
        ),
        (lines[11], " fSumw[0]=0, x=-0.5, error=0"),
        (lines[12], " fSumw[1]=2, x=0.5, error=2"),
    )


def test_ls_is_the_obj_line(capsys):
    ROOT.TH1D("h", "t", 1, 0, 1).ls("noaddr")
    ROOT.TH1D("g", "t", 1, 0, 1).ls()
    lines = capsys.readouterr().out.splitlines()
    expect(
        (lines[0], "OBJ: TH1D\th\tt : 0"),
        (bool(lines[1].startswith("OBJ: TH1D\tg\tt : 0 at: 0x")), True),
    )


def test_filling_by_label_puts_categories_on_the_axis():
    h = ROOT.TH1F("h", "", 3, 0, 3)
    expect(
        (h.Fill("apple"), 1),
        (h.Fill("pear", 2), 2),
        (h.Fill("apple"), 1),
        (h.GetXaxis().GetBinLabel(1), "apple"),
        (h.GetBinContent(2), 2),
        (h.GetXaxis().FindBin("pear"), 2),
        (h.GetXaxis().FindFixBin("plum"), -1),
    )
    labels = h.GetXaxis().GetLabels()
    expect(
        ([str(item) for item in labels], ["apple", "pear"]),
        (labels.At(1).GetUniqueID(), 2),
        (bool(ROOT.TH1F("g", "", 1, 0, 1).GetXaxis().GetLabels() is None), True),
    )


def test_fill_n_takes_arrays_of_entries():
    h = ROOT.TH1D("h", "", 4, 0, 4)
    h.FillN(3, array.array("d", [0.5, 1.5, 1.5, 9]), None)
    h.FillN(2, [2.5, 3.5], [2.0, 3.0])
    expect(
        ([h.GetBinContent(b) for b in range(1, 5)], [1, 2, 2, 3]),
        (h.GetEntries(), 5),
    )
    h2 = ROOT.TH2D("h2", "", 2, 0, 2, 2, 0, 2)
    h2.FillN(2, [0.5, 1.5], [0.5, 1.5], [1.0, 1.0])
    expect(
        (h2.GetBinContent(1, 1), 1),
        (h2.GetBinContent(2, 2), 1),
    )


def test_bins_are_read_and_set_by_global_or_per_axis_number():
    h2 = ROOT.TH2D("h2", "", 2, 0, 2, 3, 0, 3)
    expect(
        (h2.GetBin(1, 2), 1 + 4 * 2),
        (h2.GetBin(9, -1), 3),
    )
    h2.SetBinContent(1, 2, 5.0)
    h2.SetBinError(1, 2, 0.5)
    expect(
        (h2.GetBinContent(9), 5),
        (h2.GetBinContent(1, 2), 5),
        (h2.GetBinError(1, 2), 0.5),
        (bool(h2.GetBinErrorLow(9) == h2.GetBinErrorUp(9) == 0.5), True),
        (h2.GetEntries(), 1),
        (h2.GetBinContent(999), 0),
        (h2.GetBinError(999), 0),
        (h2.RetrieveBinContent(9), 5),
    )
    h2.SetBinContent(999, 1.0)
    h2.SetBinError(999, 1.0)
    bx, by, bz = array.array("i", [0]), array.array("i", [0]), array.array("i", [0])
    expect(
        (h2.GetBinXYZ(9, bx, by, bz), (1, 2, 0)),
        ((bx[0], by[0], bz[0]), (1, 2, 0)),
    )
    h3 = ROOT.TH3F("h3", "", 2, 0, 2, 2, 0, 2, 2, 0, 2)
    h3.SetBinContent(1, 1, 1, 4.0)
    expect(
        (h3.GetBinContent(1, 1, 1), 4),
        (h3.GetBin(1, 1, 1), 1 + 4 + 16),
    )
    h3.Fill(0.5, 0.5, 0.5, 2.0)
    assert h3.GetBinContent(1, 1, 1) == 6


def test_the_arrays_behind_the_bins():
    h = ROOT.TH1D("h", "", 2, 0, 2)
    h.AddBinContent(1)
    h.AddBinContent(2, 3.0)
    h.Sumw2()
    h.AddBinContent(2, 2.0)
    expect(
        (list(h.GetArray()), [0, 1, 5, 0]),
        (h.GetSumw2N(), 4),
        (list(h.GetSumw2()), [0, 0, 0, 0]),
    )
    h.Sumw2(False)
    expect(
        (h.GetSumw2N(), 0),
        (len(h.GetSumw2()), 0),
    )
    h.SetBinsLength()
    h.GetArray()[1] = 9
    assert h.GetBinContent(1) == 9
    h.Reset("ICES")
    assert h.GetBinContent(1) == 9
    h.Reset()
    expect(
        (h.GetBinContent(1), 0),
        (h.GetEntries(), 0),
    )


def test_reset_stats_works_the_entries_out_from_the_bins():
    h = ROOT.TH1D("h", "", 2, 0, 2)
    h.SetBinContent(1, 4.0)
    h.SetBinContent(2, 4.0)
    assert h.GetEntries() == 2
    h.ResetStats()
    expect(
        (h.GetEntries(), 8),
        (h.GetMean(), 1.0),
    )


#: Ten fills, the first and last in the flow bins.
FLOW_FILLS = (-1.0, 0.5, 1.5, 1.5, 2.5, 3.5, 4.5, 4.5, 4.5, 7.0)


def test_reset_stats_counts_the_flow_bins_in_the_entries_as_root_640_does():
    # Each number is ROOT 6.40.04's own for the same histogram.
    h = ROOT.TH1D("h", "", 5, 0.0, 5.0)
    for x in FLOW_FILLS:
        h.Fill(x)
    h.ResetStats()
    assert h.GetEntries() == 10
    w = ROOT.TH1D("w", "", 5, 0.0, 5.0)
    w.Sumw2()
    for x in FLOW_FILLS:
        w.Fill(x, 2.0 if x > 4 else 1.0)
    w.ResetStats()
    assert w.GetEntries() == pytest.approx(8.909090909090908, rel=1e-15)  # 14 * 14 / 22
    w.Add(w, -0.5)
    assert w.GetEntries() == pytest.approx(1.7818181818181817, rel=1e-15)


def test_reset_stats_keeps_the_sign_of_a_negative_total():
    h = ROOT.TH1D("n", "", 5, 0.0, 5.0)
    h.Fill(-1.0)
    h.Fill(2.0)
    h.SetBinContent(0, -3.0)
    h.ResetStats()
    assert h.GetEntries() == -2
    s = ROOT.TH1D("s", "", 5, 0.0, 5.0)
    s.Sumw2()
    s.Fill(2.0)
    s.Fill(-1.0, -3.0)
    s.ResetStats()
    assert s.GetEntries() == pytest.approx(0.4, rel=1e-15)  # (-2)^2 / 10
    z = ROOT.TH1D("z", "", 5, 0.0, 5.0)
    z.Sumw2()
    z.Fill(2.0, 2.0)
    z.Fill(-1.0, -2.0)
    z.ResetStats()
    assert z.GetEntries() == 0


def test_reset_stats_of_a_grid_and_a_profile_takes_every_bin():
    g = ROOT.TH2D("g", "", 2, 0, 2, 2, 0, 2)
    for x, y in ((-1, -1), (0.5, 3), (0.5, 0.5)):
        g.Fill(x, y)
    g.ResetStats()
    assert g.GetEntries() == 3
    p = ROOT.TProfile("p", "", 2, 0, 2)
    for x, y in ((-1, 3), (0.5, 2), (0.5, 4)):
        p.Fill(x, y)
    p.ResetStats()
    # ROOT's 72: the means, 3 + 3, squared, over the one bin's error squared, 1/2.
    assert p.GetEntries() == pytest.approx(72.00000000000001, rel=1e-15)


def test_python_sees_a_histogram_as_uhi_does():
    h = ROOT.TH1D("h", "", 3, 0, 3)
    h.Fill(0.5, 2.0)
    expect(
        (list(h.values()), [2, 0, 0]),
        (list(h.variances()), [4, 0, 0]),
        (list(h.counts()), [1, 0, 0]),
        (len(h.axes), 1),
        (h.kind, "COUNT"),
        (len(h), 5),
        (bool(bool(h)), True),
        (list(h), [2, 0, 0]),
        (bool(h[0] == pytest.approx(2) or h[0].value == pytest.approx(2)), True),
    )
    h[1] = 5.0
    assert h.GetBinContent(2) == 5
    rebinned = h[:: ROOT.__dict__.get("rebin", __import__("xrdroot").rebin)(3)]
    expect(
        (rebinned.GetNbinsX(), 1),
        (bool(isinstance(rebinned, ROOT.TH1)), True),
    )
