"""The classes UHI's pythonizations name: ``TH1L`` and its kin, ``TProfile2Poly``,
``ROOT.uhi``, and the ``SetBins`` and ``Rebin3D`` slicing and rebinning rest on."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot._pythonization._th1 import _th1_derived_classes_to_pythonize
from xrdroot.pyroot._pythonization._th2 import _th2_derived_classes_to_pythonize
from xrdroot.pyroot._pythonization._th3 import _th3_derived_classes_to_pythonize


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_every_class_pyroot_pythonizes_is_in_the_namespace():
    for name in (*_th1_derived_classes_to_pythonize, *_th2_derived_classes_to_pythonize,
                 *_th3_derived_classes_to_pythonize):  # fmt: skip
        assert getattr(ROOT, name).__name__ == name


def test_a_histogram_of_64_bit_integers_keeps_its_bins_as_such():
    h = ROOT.TH3L("l3", "", 2, 0, 2, 2, 0, 2, 2, 0, 2)
    h.Fill(0.5, 0.5, 0.5, 2.5)
    expect((h.ClassName(), "TH3L"), (h.GetArray().dtype, np.dtype(np.int64)),
           (h.GetBinContent(1, 1, 1), 2.0), (ROOT.TH1L.CLASS_TITLE[-31:-1],
                                             "one 64 bit integer per channel"))  # fmt: skip


def test_the_uhi_module_holds_rooots_tags():
    expect(
        (ROOT.uhi.underflow(None, 0), 0),
        (ROOT.uhi.overflow(ROOT.TH1D("t", "", 4, 0, 1), 0), 5),
        (ROOT.uhi.rebin(3).ngroup, 3),
        (sorted(ROOT.uhi.TAGS), ["loc", "overflow", "rebin", "sum", "underflow"]),
    )


def test_set_bins_makes_new_axes_and_empties_every_bin():
    h = ROOT.TH2D("sb", "", 2, 0, 2, 2, 0, 2)
    h.Fill(0.5, 0.5)
    h.SetBins(3, np.array([0.0, 1.0, 2.0, 4.0]), 4, 0.0, 1.0)
    p = ROOT.TProfile("sp", "", 2, 0, 2)
    p.Fill(0.5, 3.0)
    p.SetBins(4, 0.0, 1.0)
    expect(
        (h.GetNbinsX(), 3), (h.GetXaxis().GetBinUpEdge(3), 4.0), (h.GetNbinsY(), 4),
        (h.values(flow=True).sum(), 0.0),
        (p.GetNbinsX(), 4), (p.GetBinEntries(1), 0.0), (p.GetBinContent(1), 0.0),
    )  # fmt: skip


def test_rebin3d_merges_blocks_and_sends_the_leftovers_to_the_overflow():
    """ROOT's own numbers: the fill at z = 4.5 of five z bins lands in the overflow."""
    h = ROOT.TH3D("r3", "", 4, 0, 4, 4, 0, 4, 5, 0, 5)
    h.Fill(1, 1, 1)
    h.Fill(3.5, 0.5, 4.5, 2)
    made = h.Rebin3D(2, 2, 2, "r3b")
    h.Rebin3D(2, 2, 1)
    expect(
        (made.GetNbinsZ(), 2), (made.Integral(), 1.0), (made.GetBinContent(2, 1, 3), 2.0),
        (made.GetEntries(), 2.0), (h.GetNbinsX(), 2), (h.GetNbinsZ(), 5),
    )  # fmt: skip


def test_a_poly_profile_averages_its_values_in_its_polygons():
    p = ROOT.TProfile2Poly("pp", "pp", 0, 4, 0, 4)
    first = p.AddBin(1, 1, 2, 2)
    expect(
        (p.Fill(1.5, 1.5, 3.0), first), (p.Fill(1.5, 1.5, 5.0), first), (p.Fill(9, 9, 7.0), -3),
        (p.GetBinContent(first), 4.0), (p.GetBinEntries(first), 2.0),
        (p.GetBinError(first), pytest.approx(1 / np.sqrt(2))), (p.GetBinError(5), 0.0),
        (p.GetEntries(), 2.0), (p.FindBin(1.5, 1.5), first), (p.ClassName(), "TProfile2Poly"),
        (p.kind, "MEAN"), (p.GetDimension(), 2),
    )  # fmt: skip


def test_a_poly_profile_is_seen_through_root_s_grid_of_a_hundred_bins():
    p = ROOT.TProfile2Poly("pg", "pg", 10, 1, 4, 10, 1, 4)
    p.Fill(2, 2, 1.0)
    first = p.AddBin(1, 1, 4, 4)
    p.Fill(2, 2, 6.0)
    cell = (first, 0)  # the grid cell whose ROOT number is the polygon's
    expect(
        (p.GetXaxis().GetNbins(), 100), (p.GetYaxis().GetXmax(), 4.0), (p.GetNbinsY(), 100),
        (p.GetNcells(), 102 * 102), (p.GetSumw2N(), 0), (p.GetBin(1, 1), 103),
        (p[0, 0], 0.0), (p[cell[0] - 1, ROOT.uhi.underflow], 6.0),
        (p.values().shape, (100, 100)), (p.counts(flow=True)[cell], 1.0),
        (p.variances(flow=True)[cell], 0.0),
    )  # fmt: skip
    with pytest.raises(UnsupportedFeatureError, match="cannot be sliced"):
        p[0:2, 0]
    with pytest.raises(UnsupportedFeatureError, match="cannot be set"):
        p[0, 0] = 1.0
    with pytest.raises(UnsupportedFeatureError, match="no axis for"):
        p._to_uhi_()


def test_effective_entries_count_only_the_bins_in_range():
    """ROOT's numbers: its ``GetStats`` works them out again from the bins in range."""
    h = ROOT.TH1D("e", "", 10, 0, 10)
    for i in range(10):
        h.Fill(i + 0.5, i + 1)
    h.Fill(4.5)
    h.GetXaxis().SetRange(3, 7)
    g = ROOT.TH1D("g", "", 10, 0, 10)
    g.GetXaxis().SetRange(9, 10)
    expect((h.GetEffectiveEntries(), pytest.approx(4.970588235294118, rel=1e-15)),
           (g.GetEffectiveEntries(), 0.0))  # fmt: skip
