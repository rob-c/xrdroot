"""Extended sums use the same cached normalization that ROOT uses for their density."""

import pytest

from xrdroot import pyroot as ROOT


@pytest.fixture
def summed():
    x = ROOT.RooRealVar("cached_sum_x", "", 0., -1., 3.)
    x.setBins(2)
    histogram = ROOT.TH1D("cached_sum_hist", "", 2, -1., 3.)
    histogram.SetBinContent(1, 2.)
    histogram.SetBinContent(2, 7.)
    data = ROOT.RooDataHist("cached_sum_data", "", [x], histogram)
    function = ROOT.RooHistFunc("cached_sum_function", "", [x], data)
    coefficient = ROOT.RooRealVar("cached_sum_coefficient", "", 3.7, 0., 100.)
    pdf = ROOT.RooRealSumPdf("cached_sum_pdf", "", [function], [coefficient], True)
    return x, coefficient, pdf


def test_expected_events_and_the_density_share_their_normalization(summed, monkeypatch):
    x, _, pdf = summed
    original, calls = pdf.integrate, []

    def integrate(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(pdf, "integrate", integrate)
    assert pdf.expectedEvents([x]) == 66.60000000000001
    assert pdf.getNorm([x]) == 66.60000000000001
    x.setVal(2.)
    assert pdf.expectedEvents([x]) == 66.60000000000001
    assert len(calls) == 1


def test_changing_a_coefficient_recomputes_the_expected_events(summed):
    x, coefficient, pdf = summed
    assert pdf.expectedEvents([x]) == 66.60000000000001
    coefficient.setVal(2.3)
    assert pdf.expectedEvents([x]) == 41.4


def test_expected_events_follow_changes_to_the_normalization_range(summed):
    x, coefficient, pdf = summed
    coefficient.setVal(2.)
    assert pdf.expectedEvents([x]) == 36.
    x.setRange("cached_sum_window", -1., 1.)
    pdf.setNormRange("cached_sum_window")
    assert pdf.expectedEvents([x]) == 8.
    x.setRange("cached_sum_window", -1., 2.)
    assert pdf.expectedEvents([x]) == 22.
    pdf.setNormRange("")
    assert pdf.expectedEvents([x]) == 36.


def test_an_empty_normalization_set_has_unit_expected_events(summed):
    x, coefficient, pdf = summed
    assert pdf.expectedEvents(ROOT.RooArgSet()) == 1.
    x.setVal(2.)
    coefficient.setVal(2.3)
    assert pdf.expectedEvents(ROOT.RooArgSet()) == 1.


@pytest.fixture
def interpolated(summed):
    x, _, original = summed
    alpha = ROOT.RooRealVar("changed_sum_alpha", "", .5, -2., 2.)
    coefficient = ROOT.RooStats.HistFactory.FlexibleInterpVar(
        "changed_sum_coefficient", "", [alpha], 1., [.25], [4.])
    pdf = ROOT.RooRealSumPdf("changed_sum_pdf", "", original.funcs, [coefficient], True)
    return x, alpha, coefficient, pdf, original


def test_interpolation_settings_invalidate_expected_events(interpolated):
    x, alpha, coefficient, pdf, _ = interpolated
    assert pdf.expectedEvents([x]) == 45.
    coefficient.setInterpCode(alpha, 1)
    assert pdf.expectedEvents([x]) == 36.
    coefficient.setAllInterpCodes(0)
    assert pdf.expectedEvents([x]) == 45.
    coefficient.setNominal(3.)
    assert pdf.expectedEvents([x]) == 63.


def test_resetting_interpolation_codes_marks_the_normalization_dirty(interpolated, monkeypatch):
    x, alpha, coefficient, pdf, _ = interpolated
    original, calls = pdf.integrate, []

    def integrate(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(pdf, "integrate", integrate)
    assert pdf.expectedEvents([x]) == 45.
    coefficient.setInterpCode(alpha, 0)
    assert pdf.expectedEvents([x]) == 45.
    coefficient.setAllInterpCodes(0)
    assert pdf.expectedEvents([x]) == 45.
    assert len(calls) == 3


def test_configuration_changes_leave_unrelated_normalizations_cached(interpolated, monkeypatch):
    x, _, coefficient, changed, pdf = interpolated
    original, calls = pdf.integrate, []

    def integrate(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(pdf, "integrate", integrate)
    assert pdf.expectedEvents([x]) == 66.60000000000001
    assert changed.expectedEvents([x]) == 45.
    coefficient.setAllInterpCodes(1)
    assert changed.expectedEvents([x]) == 36.
    assert pdf.expectedEvents([x]) == 66.60000000000001
    assert len(calls) == 1


def test_cloned_pdfs_follow_configuration_of_their_shared_inputs(interpolated):
    x, _, coefficient, pdf, _ = interpolated
    assert pdf.expectedEvents([x]) == 45.
    copied = pdf.Clone("shared_sum_copy")
    assert copied.expectedEvents([x]) == 45.
    coefficient.setAllInterpCodes(1)
    assert pdf.expectedEvents([x]) == copied.expectedEvents([x]) == 36.


def test_deeply_cloned_models_keep_their_own_configuration(interpolated):
    x, _, _, pdf, _ = interpolated
    assert pdf.expectedEvents([x]) == 45.
    copied = pdf.cloneTree("independent_sum_copy")
    observable = copied.getVariables().find(x.GetName())
    assert copied.expectedEvents([observable]) == 45.
    copied.coefs[0].setAllInterpCodes(1)
    assert copied.expectedEvents([observable]) == 36.
    assert pdf.expectedEvents([x]) == 45.


def test_numerical_shape_normalization_keeps_its_integral_configuration(summed):
    x, _, original = summed
    function = original.funcs[0]
    alpha = ROOT.RooRealVar("piecewise_sum_alpha", "", .5, -2., 2.)
    low = ROOT.RooProduct("piecewise_sum_low", "", [function, ROOT.RooConstVar("lo", "", .25)])
    high = ROOT.RooProduct("piecewise_sum_high", "", [function, ROOT.RooConstVar("hi", "", 4.)])
    piece = ROOT.PiecewiseInterpolation(
        "piecewise_sum_function", "", function, [low], [high], [alpha])
    pdf = ROOT.RooRealSumPdf(
        "piecewise_sum_pdf", "", [piece], [ROOT.RooConstVar("n", "", 1.)], True)
    assert pdf.expectedEvents([x]) == 45.
    assert piece.getVal() == 5.
    piece.setInterpCode(alpha, 1)
    assert piece.getVal() == 4.
    # ROOT's cached numerical shape integral retains its initial interpolation setting.
    assert pdf.expectedEvents([x]) == 45.
    piece.setAllInterpCodes(0)
    assert pdf.expectedEvents([x]) == 45.


def test_a_shared_interpolation_factor_refreshes_both_sample_terms(interpolated):
    x, _, coefficient, original, _ = interpolated
    function = original.funcs[0]
    first = ROOT.RooProduct("shared_sum_first", "", [function, coefficient])
    second = ROOT.RooProduct("shared_sum_second", "", [coefficient, function])
    unit = ROOT.RooConstVar("shared_sum_unit", "", 1.)
    pdf = ROOT.RooRealSumPdf("shared_sum_pdf", "", [first, second], [unit, unit], True)
    assert pdf.expectedEvents([x]) == 90.
    coefficient.setAllInterpCodes(1)
    assert pdf.expectedEvents([x]) == 72.


def _empty_codes_keep_the_cache(pdf, x, interpolation, expected, monkeypatch):
    original, calls = pdf.integrate, []

    def integrate(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(pdf, "integrate", integrate)
    assert pdf.expectedEvents([x]) == expected
    interpolation.setAllInterpCodes(1)
    assert pdf.expectedEvents([x]) == expected
    assert len(calls) == 1


def test_empty_normalization_interpolations_have_no_codes_to_dirty(summed, monkeypatch):
    x, _, original = summed
    interpolation = ROOT.RooStats.HistFactory.FlexibleInterpVar(
        "empty_sum_interpolation", "", [], 2., [], [])
    pdf = ROOT.RooRealSumPdf("empty_sum_pdf", "", original.funcs, [interpolation], True)
    _empty_codes_keep_the_cache(pdf, x, interpolation, 36., monkeypatch)


def test_empty_shape_interpolations_have_no_codes_to_dirty(summed, monkeypatch):
    x, _, original = summed
    interpolation = ROOT.PiecewiseInterpolation(
        "empty_shape_interpolation", "", original.funcs[0], [], [], [])
    pdf = ROOT.RooRealSumPdf(
        "empty_shape_pdf", "", [interpolation], [ROOT.RooConstVar("scale", "", 1.)], True)
    _empty_codes_keep_the_cache(pdf, x, interpolation, 18., monkeypatch)
