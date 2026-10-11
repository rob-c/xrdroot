"""``TH2::FitSlicesX`` and ``FitSlicesY``: a fit to each slice, its parameters booked by bin.

Each x bin's y projection of a 2D Gaussian is fitted with a Gaussian; the
fitted constant, mean and width land in ``h_0``, ``h_1`` and ``h_2`` at the
slice's bin, the chi-square per degree of freedom in ``h_chi2`` - and a slice
of too few entries, or a fit of too few points, is skipped with ROOT's words.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.errors import UnsupportedFeatureError


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def _filled(name="h", nx=4, edges=None):
    rng = np.random.default_rng(3)
    if edges is None:
        h = ROOT.TH2D(name, "t", nx, 0, nx, 20, -5, 5)
    else:
        h = ROOT.TH2D(name, "t", len(edges) - 1, np.asarray(edges, dtype=float), 20, -5, 5)
    h.GetXaxis().SetTitle("x title")
    for column in range(nx):
        for y in rng.normal(0.5 * column - 1, 1.0, 400):
            h.Fill(column + 0.5, y)
    return h


def test_each_slice_fitted_books_its_parameters_and_chi_square_by_bin():
    h = _filled()
    h.FitSlicesY()
    means = ROOT.gDirectory.Get("h_1")
    assert means.GetTitle() == "Fitted value of par[1]=Mean"
    assert means.GetXaxis().GetTitle() == "x title" and means.GetNbinsX() == 4
    assert [means.GetBinContent(i) for i in range(1, 5)] == pytest.approx(
        [-1.0, -0.5, 0.0, 0.5], abs=0.15)  # fmt: skip
    assert ROOT.gDirectory.Get("h_2").GetBinContent(2) == pytest.approx(1.0, abs=0.15)
    chi2 = ROOT.gDirectory.Get("h_chi2")
    assert chi2.GetTitle() == "chisquare" and chi2.GetBinContent(1) > 0
    assert ROOT.gDirectory.Get("_temp") is None


def test_slices_grouped_cut_and_said_when_skipped(capsys):
    h = _filled()
    found = ROOT.TObjArray()
    h.FitSlicesY(None, 0, -1, 500, "NRG2", found)
    err = capsys.readouterr().err
    assert "Info in <TH2D::DoFitSlices>: Slice 0 skipped, the number of entries is zero or " in err
    assert "Info in <TH2D::DoFitSlices>: Slice fit 2 (2.000000,4.000000)" in err
    assert [one.GetName() for one in found] == ["h_0", "h_1", "h_2", "h_chi2"]
    assert "Fitted slice 2 skipped, the number of fitted points is too small, n=" in err
    assert found.At(0).GetNbinsX() == 2 and found.At(1).GetBinContent(2) == 0  # points < cut
    h.GetXaxis().SetRange(2, 3)
    h.FitSlicesY(None, 0, -1, 0, "QNRS")
    assert ROOT.gDirectory.Get("h_1").GetNbinsX() == 2
    h.GetXaxis().SetRange(0, 0)
    h.GetXaxis().SetBit(1 << 11)  # kAxisRange over bins 0 to 0, as a file may hold: all bins
    h.GetXaxis()._row["fFirst"] = h.GetXaxis()._row["fLast"] = 0
    h.FitSlicesY(None, 0, -1, 0, "QNR")
    assert ROOT.gDirectory.Get("h_1").GetNbinsX() == 4
    h.GetXaxis().SetBit(1 << 11, False)
    h.FitSlicesY(None, 3, 1, 0, "QNR")  # backwards: every bin, flows and all
    assert ROOT.gDirectory.Get("h_1").GetNbinsX() == 4
    with pytest.raises(UnsupportedFeatureError, match=r"graphical cut \[cut\]"):
        h.FitSlicesY(None, 0, -1, 0, "QNR[cut]")


def test_a_slice_along_x_fits_the_function_given_and_a_fit_of_too_few_points_is_skipped(capsys):
    h = _filled("v", 3, [0.0, 1.0, 2.5, 3.0])
    line = ROOT.TF1("line", "pol0", -5, 5)
    h.FitSlicesX(line, 0, -1, 0, "NR")
    h.FitSlicesY(ROOT.TF1("many", "pol9", -5, 5), 1, 1, 0, "NR")
    err = capsys.readouterr().err
    assert "Fitted slice 1 skipped, the number of fitted points is too small, n=" not in err
    assert ROOT.gDirectory.Get("v_0").GetXaxis().GetXbins()[1] == 1.0
    lone = ROOT.TH2D("lone", "lone", 2, 0, 2, 2, 0, 2)
    lone.Fill(0.5, 0.5)
    lone.FitSlicesY(None, 1, 1, 0, "NR")
    assert "Fitted slice 1 skipped, the number of fitted points is too small, n=1" in (
        capsys.readouterr().err)  # fmt: skip
    lone.FitSlicesY(None, 1, 1, 0, "QNR")
    assert capsys.readouterr().err == ""
    lone.FitSlicesY(ROOT.TF1("none", "1", 0, 2))  # a function of no parameters: nothing
    assert ROOT.gDirectory.Get("lone_chi2") is not None
