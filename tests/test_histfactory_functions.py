"""HistFactory's functions: ``FlexibleInterpVar``, ``PiecewiseInterpolation``, ``ParamHistFunc``
and ``RooBinWidthFunction``, held to ROOT 6.40's values.

Each interpolation code of the two interpolating classes is evaluated at the same parameter
values ROOT evaluated it at, inside the boundary and beyond it on both sides; the gammas'
function is read bin by bin, integrated, and asked for its bins; the bin-width function is
switched off and on as a binned likelihood switches it.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.roofit.binning import RooBinning
from xrdroot.roofit.collections import RooArgList
from xrdroot.roofit.data.datahist import RooDataHist
from xrdroot.roofit.pdfs import histfactory as hf
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.pdfs.histfactory import (
    FlexibleInterpVar,
    ParamHistFunc,
    PiecewiseInterpolation,
    RooBinWidthFunction,
    interpolate,
)
from xrdroot.roofit.pdfs.histpdf import RooHistFunc
from xrdroot.roofit.variables import RooRealVar

POINTS = (-2.0, -1.0, -0.4, 0.0, 0.3, 1.0, 1.7)

#: ``FlexibleInterpVar`` of nominal 2, low 1.5 and high 3, at ``POINTS``, as ROOT 6.40 gives it.
FLEXIBLE = {
    0: [1.0, 1.5, 1.8, 2.0, 2.3, 3.0, 3.7],
    1: [1.125, 1.5, 1.7826024579660034, 2.0, 2.258693870913711, 3.0, 3.9846037198300026],
    2: [1.25, 1.5, 1.74, 2.0, 2.2475, 3.0, 3.875],
    4: [1.125, 1.5, 1.7658607003437548, 2.0, 2.2438775653153313, 3.0, 3.9846037198300026],
    5: [1.125, 1.5, 1.7658607003437548, 2.0, 2.2438775653153313, 3.0, 3.9846037198300026],
}

#: ``PiecewiseInterpolation`` of nominal 10, low 8 and high 13, at ``-4.5`` and ``POINTS[1:]``.
PIECEWISE = {
    0: [1.0, 8.0, 9.2, 10.0, 10.9, 13.0, 15.1],
    1: [3.6635737743356556, 8.0, 9.146101038546528, 10.0, 10.818897486445278, 13.0,
        15.620815356808382],
    2: [2.75, 8.0, 9.08, 10.0, 10.795, 13.0, 15.45],
    4: [1.0, 8.0, 9.134768, 10.0, 10.8294491875, 13.0, 15.1],
    5: [3.6635737743356556, 8.0, 9.118864347984681, 10.0, 10.793309775020541, 13.0,
        15.620815356808382],
    6: [1.0000000000000018, 8.0, 9.134768, 10.0, 10.8294491875, 13.0, 15.1],
}  # fmt: skip


@pytest.mark.parametrize("code", sorted(FLEXIBLE))
def test_a_flexible_interp_var_interpolates_each_code_as_root_does(code: int) -> None:
    """Codes 0, 1, 2, 4 (which is 5) and 5, inside and outside the boundary, both sides."""
    a = RooRealVar("a", "a", 0, -5, 5)
    f = FlexibleInterpVar("f", "", [a], 2.0, [1.5], [3.0])
    f.setInterpCode(a, code)
    found = []
    for x in POINTS:
        a.setVal(x)
        found.append(f.getVal())
    assert found == pytest.approx(FLEXIBLE[code], rel=1e-12)


def test_a_flexible_interp_var_of_two_parameters_and_its_accessors() -> None:
    """Both parameters' codes set at once; its numbers read back; ROOT's ``0.78109...``."""
    a, b = RooRealVar("a", "a", 0.5, -5, 5), RooRealVar("b", "b", -1.5, -5, 5)
    f = FlexibleInterpVar("f2", "", RooArgList([a, b]), 1.0, [0.9, 0.8], [1.2, 1.1])
    f.setAllInterpCodes(4)
    assert f.getVal() == pytest.approx(0.7810984695749845, rel=1e-12)
    assert (f.interpolationCodes(), f.low(), f.high(), f.nominal()) == ([4, 4], [0.9, 0.8],
                                                                        [1.2, 1.1], 1.0)
    f.setNominal(2.0)
    assert (f.nominal(), f.ClassName()) == (2.0, "RooStats::HistFactory::FlexibleInterpVar")


def test_a_flexible_interp_var_that_would_not_be_positive_is_the_smallest_double() -> None:
    """``TMath::Limits<double>::Min()``, for one value and for a batch of them."""
    a = RooRealVar("a", "a", -3.0, -5, 5)
    f = FlexibleInterpVar("f3", "", [a], 1.0, [0.2], [1.5])
    assert f.getVal() == 2.2250738585072014e-308
    assert list(f.compute({"a": np.array([-3.0, 0.0])})) == [2.2250738585072014e-308, 1.0]


def test_the_code_five_arrays_agree_with_the_code_five_numbers() -> None:
    """A batch of parameter values takes the arrays' path; each is the one-value path's."""
    xs = np.array(POINTS)
    batch = hf._code5(1.5, 3.0, 1.0, 2.0, xs, np.full(len(xs), 2.0))
    one = [hf._code5_one(1.5, 3.0, 1.0, 2.0, float(x), 2.0) for x in xs]
    assert list(batch) == pytest.approx(one, rel=1e-13)
    zero = hf._code5(np.zeros(1), np.zeros(1), 1.0, 2.0, np.array([0.5]), np.array([2.0]))
    assert list(zero) == [hf._code5_one(0.0, 0.0, 1.0, 2.0, 0.5, 2.0)] == [-1.15625]


def test_an_unknown_code_adds_nothing() -> None:
    assert list(interpolate(3, 1.5, 3.0, 1.0, 2.0, np.array([0.5, -2.0]), 2.0)) == [0.0, 0.0]


@pytest.mark.parametrize("code", sorted(PIECEWISE))
def test_a_piecewise_interpolation_interpolates_each_code_as_root_does(code: int) -> None:
    """Codes 0, 1, 2, 4, 5 and 6 of a nominal 10 between 8 and 13."""
    a = RooRealVar("a", "a", 0, -5, 5)
    nominal, low, high = (RooRealVar(n, "", v) for n, v in (("n", 10.0), ("l", 8.0),
                                                          ("h", 13.0)))  # fmt: skip
    p = PiecewiseInterpolation("p", "", nominal, [low], [high], [a])
    p.setInterpCode(a, code, True)
    found = []
    for x in (-4.5, *POINTS[1:]):
        a.setVal(x)
        found.append(p.getVal())
    assert found == pytest.approx(PIECEWISE[code], rel=1e-12)
    assert p.interpolationCodes() == [code]


def binned_x() -> RooRealVar:
    """``x`` in ``[0, 4]``, binned at 0, 1 and 4."""
    x = RooRealVar("x", "x", 0, 4)
    edges = RooBinning(0.0, 4.0)
    edges.addBoundary(1.0)
    x.setBinning(edges)
    return x


def hist_func(x: RooRealVar) -> RooHistFunc:
    from xrdroot.pyroot.core import TH1D

    h = TH1D("h", "", 2, np.array([0.0, 1.0, 4.0]))
    h.SetDirectory(0)
    h.SetBinContent(1, 3.0)
    h.SetBinContent(2, 6.0)
    return RooHistFunc("hf", "", [x], RooDataHist("dh", "", [x], h))


def test_a_positive_definite_interpolation_is_cut_at_zero_and_binned_as_its_nominal() -> None:
    """Pulled far below, the sum is zero; its bins are the nominal histogram function's."""
    a, x = RooRealVar("a", "a", -4.5, -5, 5), binned_x()
    nominal = hist_func(x)
    low, high = RooRealVar("l", "", 1.0), RooRealVar("h", "", 13.0)
    p = PiecewiseInterpolation("p", "", nominal, [low], [high], [a])
    p.setAllInterpCodes(0)
    x.setVal(2.5)
    assert p.getVal() == pytest.approx(6.0 - 4.5 * 5.0)
    p.setPositiveDefinite()
    assert (p.positiveDefinite(), p.getVal()) == (True, 0.0)
    assert list(p.compute({"a": np.array([-4.5, 0.0]), "x": np.array([2.5, 0.5])})) == [0.0, 3.0]
    assert (p.bin_boundaries("x"), p.bin_boundaries("a")) == ([0.0, 1.0, 4.0], None)
    assert p.isBinnedDistribution([x]) is True
    assert PiecewiseInterpolation("q", "", RooGaussian("g", "", x, a, low), [low], [high],
                                  [a]).isBinnedDistribution([x]) is False  # fmt: skip


def gammas_2d() -> tuple[ParamHistFunc, RooRealVar, RooRealVar, list[RooRealVar]]:
    x, y = binned_x(), RooRealVar("y", "y", 0, 2)
    y.setBins(2)
    gammas = [RooRealVar(f"g{i}", "", 1 + 0.5 * i, 0, 10) for i in range(4)]
    return ParamHistFunc("phf", "", [x, y], gammas), x, y, gammas


def test_a_param_hist_func_is_the_gamma_of_the_bin_first_variable_fastest() -> None:
    """At each of the four bins of ``x`` by ``y``, ROOT's gamma and its value."""
    f, x, y, _ = gammas_2d()
    found = []
    for xv, yv in ((0.5, 0.5), (2.5, 0.5), (0.5, 1.5), (2.5, 1.5)):
        x.setVal(xv)
        y.setVal(yv)
        found.append((f.getVal(), f.getParameter().GetName()))
    assert found == [(1.0, "g0"), (1.5, "g1"), (2.0, "g2"), (2.5, "g3")]
    batch = f.compute({"x": np.array([0.5, 2.5]), "y": np.array([1.5, 1.5]), "g2": 2.0,
                       "g3": 2.5, "g0": 1.0, "g1": 1.5})  # fmt: skip
    assert list(batch) == [2.0, 2.5]
    assert (f.numBins(), f.getParameter(1).GetName(), len(f.paramList())) == (4, "g1", 4)


def test_a_param_hist_funcs_integral_pairs_volumes_and_gammas_as_root_does() -> None:
    """ROOT's 16 - over both variables and over ``x`` alone - and 16 again numerically forced."""
    f, x, y, _ = gammas_2d()
    assert f.createIntegral([x, y]).getVal() == 16.0
    assert f.createIntegral([x]).getVal() == 16.0
    assert f.analytic_names(frozenset({"x", "z"}), None) == frozenset({"x"})
    f.setForceNumInt(True)
    assert f.analytic_names(frozenset({"x"}), None) == frozenset()


def test_a_param_hist_func_says_its_bins_and_sets_its_gammas_constant() -> None:
    f, *_ = gammas_2d()
    assert (f.bin_boundaries("y"), f.bin_boundaries("z"), f.isBinnedDistribution()) == (
        [0.0, 1.0, 2.0], None, True)  # fmt: skip
    f.setConstant()
    assert all(one.isConstant() for one in f.paramList())


def test_a_param_hist_func_with_a_gamma_short_is_refused() -> None:
    x = binned_x()
    with pytest.raises(ValueError, match="has 1 elements but the ParamHistFuncp has 2 bins"):
        ParamHistFunc("p", "", [x], [RooRealVar("g", "", 1.0)])
    assert ParamHistFunc("none", "").numBins() == 0


def test_a_bin_width_function_is_the_width_or_its_inverse_of_the_bin() -> None:
    """ROOT's 3 and 1/3 at ``x = 2.5``; one outside the histogram, off, or in a binned fit."""
    x = binned_x()
    func = hist_func(x)
    width, inverse = RooBinWidthFunction("bw", "", func), RooBinWidthFunction("bd", "", func, True)
    x.setVal(2.5)
    assert (width.getVal(), inverse.getVal()) == (3.0, 1.0 / 3.0)
    assert list(width.compute({"x": np.array([0.5, 9.0])})) == [1.0, 1.0]
    with hf.binned_likelihood():
        assert width.getVal() == 1.0
    RooBinWidthFunction.disableClass()
    try:
        assert (RooBinWidthFunction.isClassEnabled(), width.getVal()) == (False, 1.0)
    finally:
        RooBinWidthFunction.enableClass()
    assert (inverse.divideByBinWidth(), inverse.histFunc() is func) == (True, True)
    assert (width.bin_boundaries("x"), width.isBinnedDistribution([x])) == ([0.0, 1.0, 4.0], True)


def test_bins_are_found_through_functions_that_have_them_and_depend_on_the_variable() -> None:
    """The first function with edges in ``x``; binned only if every dependent one is."""
    x, a = binned_x(), RooRealVar("a", "a", 0, -1, 1)
    gauss = RooGaussian("g", "", x, a, RooRealVar("s", "", 1.0))
    width = RooBinWidthFunction("bw", "", hist_func(x))
    assert hf._boundaries_of([a, gauss, width], "x") == [0.0, 1.0, 4.0]
    assert hf._boundaries_of([PiecewiseInterpolation("p", "", gauss)], "x") is None
    assert hf._binned([a, width], frozenset({"x"})) is True
    assert hf._binned([width, unbinned_product(x)], [x]) is False


def unbinned_product(x: RooRealVar) -> object:
    """A function of ``x`` that cannot say whether it is binned."""
    from xrdroot.roofit.functions import RooProduct

    return RooProduct("prod", "", [x])
