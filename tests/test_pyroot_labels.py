"""Histograms of categories: filled by label, doubled when full, deflated and sorted as ROOT's."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


NAMES = ["a", "b", "c", "d", "e"]


def _labels(h):
    axis = h.GetXaxis()
    return [(axis.GetBinLabel(i), h.GetBinContent(i)) for i in range(1, h.GetNbinsX() + 1)]


def test_a_new_label_takes_the_next_bin_and_a_full_axis_doubles():
    h = ROOT.TH1D("h", "test", 3, 0, 3)
    h.SetCanExtend(ROOT.TH1.kAllAxes)
    for i in range(40):
        h.Fill(NAMES[(i * 7) % 5], 1 + i % 3)
    expect((h.GetNbinsX(), 6), (h.GetXaxis().GetXmax(), 6.0), (h.GetMean(), 0.0))
    h.LabelsDeflate()
    expect(
        (h.GetNbinsX(), 5),
        (h.GetEntries(), 40.0),
        (_labels(h)[:2], [("a", 16.0), ("c", 15.0)]),
        (round(h.GetBinError(3), 6), 6.403124),
    )
    h.LabelsDeflate()
    h.LabelsInflate()
    assert h.GetNbinsX() == 10


def test_a_first_label_makes_the_axis_one_of_categories_that_may_grow():
    h = ROOT.TH1F("h", "", 2, 0, 2)
    h.Fill("x")
    h.Fill("y")
    h.Fill("z", 2.0)
    axis = h.GetXaxis()
    expect(
        ((axis.CanExtend(), axis.IsAlphanumeric(), axis.CanBeAlphanumeric()), (True, True, True)),
        (h.GetNbinsX(), 4),
        (_labels(h)[2], ("z", 2.0)),
        (h.GetMean(), 0.0),
    )
    assert h.Fill("x") == 1


def test_an_axis_that_cannot_be_of_categories_ignores_a_new_label(capsys):
    h = ROOT.TH1D("h", "", 2, [0.0, 1.0, 3.0])
    plain = ROOT.TH1D("p", "", 2, 0, 2)
    plain.GetXaxis().SetNoAlphanumeric()
    expect(
        (h.Fill("x"), -1),
        (plain.Fill("y"), -1),
        (plain.GetXaxis().IsAlphanumeric(), False),
        ("not alphanumeric - ignore it" in capsys.readouterr().err, True),
    )
    plain.GetXaxis().SetNoAlphanumeric(False)
    assert plain.GetXaxis().CanBeAlphanumeric()


def test_labels_on_a_fixed_axis_fill_without_growing_it_and_keep_their_moments():
    h = ROOT.TH1D("h", "", 3, 0, 3)
    h.Fill("a")
    h.GetXaxis().SetCanExtend(False)
    h.Fill("b", 3.0)
    expect((h.GetNbinsX(), 3), (h.GetMean(), 1.125), (h.SetCanExtend(ROOT.TH1.kXaxis), 0))
    assert h.SetCanExtend(ROOT.TH1.kNoAxis) == ROOT.TH1.kXaxis


def test_both_axes_of_a_two_dimensional_histogram_grow_and_deflate():
    h2 = ROOT.TH2F("h2", "", 2, 0, 2, 2, 0, 2)
    h2.SetCanExtend(ROOT.TH1.kAllAxes)
    for i in range(30):
        h2.Fill(NAMES[i % 5], NAMES[(i * 3) % 4], 1.0)
    expect((h2.GetNbinsX(), 8), (h2.GetNbinsY(), 4))
    h2.LabelsDeflate("X")
    h2.LabelsDeflate("Y")
    h2.LabelsOption("v")
    cells = [h2.GetBinContent(i, j) for i in range(1, 6) for j in range(1, 5)]
    expect(
        ((h2.GetNbinsX(), h2.GetNbinsY(), h2.GetEntries()), (5, 4, 30.0)),
        (cells[:6], [2.0, 2.0, 1.0, 1.0, 1.0, 2.0]),
        (h2.GetXaxis()._row["_labels_option"], "v"),
    )


def test_labelled_bins_sort_by_label_or_by_content_errors_and_all():
    h = ROOT.TH1D("h", "", 3, 0, 3)
    h.Sumw2()
    for name, w in (("c", 1), ("a", 3), ("b", 2), ("d", 5), ("c", 1)):
        h.Fill(name, w)
    h.LabelsOption("a")
    alphabetical = _labels(h)[:4]
    h.LabelsOption(">")
    down = _labels(h)[:4]
    h.LabelsOption("<")
    expect(
        (alphabetical, [("a", 3.0), ("b", 2.0), ("c", 2.0), ("d", 5.0)]),
        (down, [("d", 5.0), ("a", 3.0), ("b", 2.0), ("c", 2.0)]),
        (_labels(h)[:4], [("b", 2.0), ("c", 2.0), ("a", 3.0), ("d", 5.0)]),
        (round(h.GetBinError(2), 4), 1.4142),
    )
    plain = ROOT.TH1D("p", "", 2, 0, 2)
    plain.Fill("q")
    plain.LabelsOption("a")
    assert _labels(plain) == [("q", 1.0), ("", 0.0)]
