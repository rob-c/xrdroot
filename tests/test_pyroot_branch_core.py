"""The PyROOT layer's additions for RooStats' tutorials, held to what ROOT 6.40 answers.

``kNone``, ``ROOT.double`` and ``ROOT.sqrt``; matrices made empty and set an
element at a time; a histogram's under- and overflow bins and its contour
levels; the chi-square density at its origin and its quantile; the minimizer
defaults one table holds; colours named as strings; boxes whose corners are
put in order while a line's are kept; RooFit's HistFactory classes and range
casts; a workspace read from a file as the engine's own. Each expected value
was printed by ROOT itself for the same call.
"""

from __future__ import annotations

import array
import math
import pathlib
from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.fit import defaults
from xrdroot.pyroot.core.fits import TMatrixD, TMatrixDSym

DATA = pathlib.Path(__file__).parent / "data"


def test_knone_double_and_sqrt_are_what_a_script_takes_them_for() -> None:
    """``kNone`` is 0, ``ROOT.double(3)`` is 3.0 and ``ROOT.sqrt(4.0)`` is 2.0."""
    assert (ROOT.kNone, ROOT.double(3), ROOT.sqrt(4.0)) == (0, 3.0, 2.0)
    assert "roostats" in ROOT.SUBMODULES


def test_a_matrix_made_by_size_is_zeros_set_an_element_at_a_time() -> None:
    """``TMatrixDSym(2)`` and ``TMatrixD(2, 3)`` start at zero; ``m[i, j] = v`` sets one."""
    square, wide = TMatrixDSym(2), TMatrixD(2, 3)
    assert (square.GetNrows(), square.GetNcols(), wide.GetNcols()) == (2, 2, 3)
    square[0, 1] = 1.5
    square.__setcall__(1, 0, 2.5)
    assert (square[0, 1], square(1, 0), len(square)) == (1.5, 2.5, 2)
    square[1] = [4.0, 5.0]
    assert list(square[1]) == [4.0, 5.0]
    copied = TMatrixDSym(square)
    assert np.array_equal(np.asarray(copied), square.matrix())
    assert np.asarray(copied, dtype=np.float32).dtype == np.float32


def test_under_and_overflow_bins_are_found_along_any_axis_or_one() -> None:
    """``IsBinUnderflow`` and ``IsBinOverflow``, of every axis for ``iaxis`` 0, or of one."""
    h = ROOT.TH2D("hflows", "h", 2, 0, 2, 2, 0, 2)
    assert (h.IsBinUnderflow(0), h.IsBinUnderflow(5), h.IsBinOverflow(15), h.IsBinOverflow(5)) == (
        True, False, True, False)  # fmt: skip
    assert [h.IsBinUnderflow(1, 1), h.IsBinUnderflow(1, 2), h.IsBinOverflow(3, 1),
            h.IsBinOverflow(3, 2)] == [False, True, True, False]  # fmt: skip
    h1 = ROOT.TH1D("h1flows", "h1", 3, 0, 3)
    assert [h1.IsBinUnderflow(0), h1.IsBinOverflow(4), h1.IsBinOverflow(3),
            h1.IsBinUnderflow(0, 2)] == [True, True, False, True]  # fmt: skip
    assert (h1.IsBinUnderflow(2, 2), h1.IsBinOverflow(4, 2)) == (True, False)


def test_contour_levels_are_even_by_default_or_the_users_given() -> None:
    """``SetContour(4)`` spreads four levels from the lowest content; ``SetContour(3, levels)``
    keeps those and sets ``kUserContour``; ``SetContour(0)`` clears them."""
    h = ROOT.TH2D("hcont", "h", 2, 0, 2, 2, 0, 2)
    assert (h.GetContour(), h.GetContourLevel(0)) == (0, 0.0)
    h.Fill(0.5, 0.5, 1.0)
    h.Fill(1.5, 1.5, 5.0)
    h.SetContour(4)
    assert h.GetContour() == 4 and not h.TestBit(1 << 10)
    assert [h.GetContourLevel(i) for i in range(5)] == [0.0, 1.25, 2.5, 3.75, 0.0]
    h.SetContour(3, array.array("d", [0.5, 2.0, 3.0]))
    out = array.array("d", [0.0] * 3)
    assert (h.GetContour(out), list(out), h.TestBit(1 << 10)) == (3, [0.5, 2.0, 3.0], True)
    assert h.GetContourLevel(-1) == 0.0
    h.SetContour(0)
    assert (h.GetContour(), h.TestBit(1 << 10)) == (0, False)


def test_the_chi_square_density_at_its_origin_is_roots() -> None:
    """At ``x0``: infinite for one degree of freedom, a half for two, zero for three."""
    pdf = ROOT.Math.chisquared_pdf
    assert (pdf(0, 1), pdf(0, 2), pdf(0, 3), pdf(-1, 3)) == (math.inf, 0.5, 0.0, 0.0)
    assert pdf(2.5, 3) == pytest.approx(0.18072239266818127, rel=1e-15)
    assert ROOT.TMath.ChisquareQuantile(0.95, 3) == 7.814727903253585


def test_the_minimizer_defaults_are_the_one_table_the_engine_reads(monkeypatch: Any) -> None:
    """``SetDefaultMinimizer("Minuit")`` is what the engine's RooStats calculators see too."""
    monkeypatch.setattr(defaults, "DEFAULTS", dict(defaults.DEFAULTS))
    options = ROOT.Math.MinimizerOptions
    monkeypatch.setattr(options, "_defaults", defaults.DEFAULTS)
    options.SetDefaultMinimizer("Minuit")
    assert (options.DefaultMinimizerType(), options.DefaultMinimizerAlgo()) == ("Minuit", "Migrad")
    options.SetDefaultMinimizer("Minuit2", "Simplex")
    assert defaults.minimizer_algo() == "Simplex"


def test_a_colour_given_by_name_is_its_number() -> None:
    """``SetLineColor("kGreen")`` is 416 and ``"kRed+2"`` 634, as ``TColorNumber`` reads them;
    a ``#rrggbb`` is the colour ``TColor::GetColor`` finds."""
    line = ROOT.TLine(0, 1, 1, 0)
    line.SetLineColor("kGreen")
    assert line.GetLineColor() == 416
    line.SetLineColor("kRed+2")
    assert line.GetLineColor() == 634
    line.SetLineColor("#ff0000")
    assert line.GetLineColor() == ROOT.TColor.GetColor("#ff0000") == 2
    line.SetLineWidth(3)
    assert line.GetLineWidth() == 3


def test_a_box_and_a_pave_put_their_corners_in_order_but_a_line_keeps_its_own() -> None:
    """``TBox(1, 1, 0, 0)`` is ``(0, 0)-(1, 1)``; a ``TPaveText`` so too; a line falls."""
    box = ROOT.TBox(1, 1, 0, 0)
    assert (box.GetX1(), box.GetY1(), box.GetX2(), box.GetY2()) == (0.0, 0.0, 1.0, 1.0)
    pave = ROOT.TPaveText(0.9, 0.8, 0.1, 0.2, "NDC")
    assert (pave.GetX1(), pave.GetY1(), pave.GetX2(), pave.GetY2()) == (0.1, 0.2, 0.9, 0.8)
    line = ROOT.TLine(0, 1, 1, 0)
    assert (line.GetX1(), line.GetY1(), line.GetX2(), line.GetY2()) == (0.0, 1.0, 1.0, 0.0)


def test_histfactorys_classes_are_at_roots_top_level_without_roo() -> None:
    """``ROOT.ParamHistFunc`` and ``ROOT.PiecewiseInterpolation``, as ROOT has them."""
    from xrdroot.pyroot import roofit
    from xrdroot.roofit.pdfs.histfactory import ParamHistFunc, PiecewiseInterpolation

    assert roofit.ParamHistFunc is ParamHistFunc
    assert roofit.PiecewiseInterpolation is PiecewiseInterpolation
    assert "FlexibleInterpVar" not in roofit.__all__
    assert ROOT.ParamHistFunc is ParamHistFunc


def test_range_casts_hand_back_the_collection_they_are_given() -> None:
    """``static_range_cast<T>(c)`` and ``dynamic_range_cast<T>(c)`` are ``c``, as iterated."""
    from xrdroot.pyroot.roofit import dynamic_range_cast, static_range_cast

    held = ROOT.RooArgList()
    assert static_range_cast(held) is held
    assert dynamic_range_cast("RooRealVar", held) is held


def test_roofit_has_the_model_config_since_root_630() -> None:
    """``RooFit.ModelConfig`` is RooStats' ``ModelConfig``."""
    from xrdroot.roostats.modelconfig import ModelConfig

    assert ROOT.RooFit.ModelConfig is ModelConfig


def test_a_workspace_read_from_a_file_is_the_engines_own_object() -> None:
    """``TFile::Get`` of a ``RooWorkspace`` hands back the workspace, not a stand-in."""
    from xrdroot.roofit.workspace import RooWorkspace

    f = ROOT.TFile.Open(str(DATA / "roofit-workspace-zoo.root"))
    try:
        w = f.Get("w")
        assert isinstance(w, RooWorkspace)
        assert w.pdf("g").GetName() == "g"
    finally:
        f.Close()


def test_an_object_of_a_class_with_no_wrapper_is_still_a_stand_in() -> None:
    """Neither a ``TObject`` nor the engine's own - a ``TF1Convolution`` - is named by class."""
    from xrdroot.pyroot.core.files import TOther

    f = ROOT.TFile.Open(str(DATA / "tformula.root"))
    try:
        assert isinstance(f.Get("fconv"), TOther)
    finally:
        f.Close()


def test_a_matrix_made_from_rows_keeps_them() -> None:
    """``TMatrixDSym([[1, 2], [2, 3]])``: the values as given."""
    assert TMatrixDSym([[1, 2], [2, 3]])(1, 1) == 3.0


def test_the_engine_reads_what_is_not_a_tree_or_a_workspace_as_its_object() -> None:
    """A histogram in a file is read whole, not as a workspace."""
    from xrdroot import open_root

    with open_root(str(DATA / "gauss-h1.root")) as f:
        name = f.keys()[0]
        assert f[name] is not None
