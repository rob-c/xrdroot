"""``ROOT::Math`` inside formulas, the Legendre polynomials, and a function's drawing frame."""

from __future__ import annotations

import math

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_the_legendre_polynomials_are_those_root_gives():
    M = ROOT.Math
    expect(
        (M.legendre(0, 0.3), 1.0),
        (M.legendre(1, 0.3), 0.3),
        (round(M.legendre(3, 0.3), 12), -0.3825),
        (round(M.legendre(5, -0.7), 12), 0.36519875),
        (round(M.assoc_legendre(3, 1, 0.3), 12), -0.786999841169),
        (M.assoc_legendre(4, 2, -0.5), 4.21875),
        (round(M.assoc_legendre(2, 2, 0.1), 12), 2.97),
        (math.isnan(M.assoc_legendre(1, 2, 0.3)), True),
        (round(M.sph_legendre(3, 1, 0.3), 12), -0.340321237293),
        (round(M.sph_legendre(4, 2, 1.5), 12), -0.321190930118),
        (round(M.sph_legendre(2, 0, 0.1), 12), 0.621352880682),
    )


def test_a_formula_may_call_any_root_math_function():
    f = ROOT.TF1("f", "ROOT::Math::normal_pdf(x, [0], [1])", -5, 5)
    f.SetParameters(1.5, 0.2)
    g = ROOT.TF1("g", "ROOT::Math::legendre([0], x)", -1, 1)
    g.SetParameter(0, 3)
    expect(
        (round(f.Eval(0.5), 12), 0.260695129317),
        (round(g.Eval(0.4), 12), -0.44),
    )


def test_a_functions_frame_is_kept_and_follows_its_limits():
    f = ROOT.TF1("f", "x*[0]", 0, 1)
    f.SetParameter(0, 1)
    f.GetXaxis().SetTitle("x")
    f.SetMaximum(3)
    f.SetMinimum(-1)
    f.SetParameter(0, 2)
    frame = f.GetHistogram()
    expect(
        (f.GetYaxis() is frame.GetYaxis(), True),
        (frame.GetXaxis().GetTitle(), "x"),
        (round(frame.GetBinContent(100), 6), 1.99),
        ((frame.GetMaximum(), frame.GetMinimum()), (3.0, -1.0)),
        ((f.GetMaximumStored(), f.GetMinimumStored()), (3.0, -1.0)),
        (f.GetZaxis() is frame.GetZaxis(), True),
    )
    f.SetNpx(10)
    assert f.GetHistogram() is not frame


def _fitted_line():
    xs = [0.1 * i for i in range(20)]
    ys = [1.0 + 2.0 * x + 0.3 * ((i * 7) % 5 - 2) for i, x in enumerate(xs)]
    g = ROOT.TGraph(len(xs), xs, ys)
    f = ROOT.TF1("fl", "pol1", 0, 2)
    return f, g.Fit(f, "QS")


def test_the_latest_fits_band_is_the_one_root_draws():
    assert ROOT.TVirtualFitter.GetFitter() is None
    f, r = _fitted_line()
    band = ROOT.TGraphErrors(3)
    for i in range(3):
        band.SetPoint(i, 0.5 * i, 0)
    fitter = ROOT.TVirtualFitter.GetFitter()
    fitter.GetConfidenceIntervals(band, 0.68)
    h = ROOT.TH1D("hci", "", 4, 0, 2)
    fitter.GetConfidenceIntervals(h)
    expect(
        ([round(band.GetPointY(i), 6) for i in range(3)], [0.914286, 1.959398, 3.004511]),
        ([round(band.GetErrorY(i), 6) for i in range(3)], [0.19021, 0.125192, 0.099066]),
        (round(h.GetBinContent(1), 6), 1.436842),
        (round(h.GetBinError(4), 6), 0.332665),
        (round(f.IntegralError(0, 1), 6), 0.12589),
        (
            round(
                f.IntegralError(0, 1, r.GetParams(), r.GetCovarianceMatrix().GetMatrixArray()), 6
            ),
            0.12589,
        ),
        (fitter.GetNumberTotalParameters(), 2),
        (round(fitter.GetParameter(1), 6), round(r.Parameter(1), 6)),
        (fitter.GetParError(0), r.ParError(0)),
        (fitter.GetCovarianceMatrixElement(0, 1), r.CovMatrix(0, 1)),
    )


def test_a_band_scaled_by_chi2_and_a_band_of_what_is_not_a_graph(capsys):
    from xrdroot.pyroot.core import fitters

    _, r = _fitted_line()
    ROOT.TVirtualFitter.GetFitter().GetConfidenceIntervals(ROOT.TGraph(1))
    _, half = fitters.band(r.Get()._xrd, [0.0], 0.68, norm=True)
    ROOT.TVirtualFitter.SetDefaultFitter("Minuit2")
    expect(
        (round(float(half[0]), 6), 0.086820),
        ("not supported" in capsys.readouterr().err, True),
        (ROOT.TVirtualFitter.GetDefaultFitter(), "Minuit2"),
    )
    ROOT.TVirtualFitter.SetDefaultFitter()
    assert ROOT.TVirtualFitter.GetDefaultFitter() == "Minuit"


def test_root_number_types_cast_as_cpp_does_and_titles_and_lists():
    h = ROOT.TH2D("h2", "", 2, 0, 1, 2, 0, 1)
    h.SetXTitle("x")
    h.SetYTitle("y")
    h.SetZTitle("z")
    ROOT.TF1("unlisted", "x", 0, 1, ROOT.TF1.EAddToList.kNo)
    ROOT.TF1("listed", "x", 0, 1, ROOT.TF1.EAddToList.kAdd)
    expect(
        (ROOT.Float_t(3) / 2, 1.5),
        (ROOT.Int_t(3.7), 3),
        (ROOT.Bool_t(2), True),
        (
            [h.GetXaxis().GetTitle(), h.GetYaxis().GetTitle(), h.GetZaxis().GetTitle()],
            ["x", "y", "z"],
        ),
        (ROOT.gROOT.GetListOfFunctions().FindObject("unlisted"), None),
        (ROOT.gROOT.GetListOfFunctions().FindObject("listed") is not None, True),
    )
