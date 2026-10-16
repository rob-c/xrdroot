"""Cached histogram values follow mutations to coordinates, data and binning."""

import numpy as np
import pytest

from xrdroot.roofit.binning import RooBinning, RooUniformBinning
from xrdroot.roofit.data.datahist import RooDataHist
from xrdroot.roofit.pdfs.histfactory import ParamHistFunc
from xrdroot.roofit.pdfs.histpdf import RooHistFunc, RooHistPdf
from xrdroot.roofit.variables import RooRealVar


def made(cls=RooHistFunc):
    x = RooRealVar("x", "", 0.0, 2.0)
    x.setBinning(RooBinning(2, 0.0, 2.0))
    data = RooDataHist("data", "", [x])
    data.set(0, 10.0)
    data.set(1, 20.0)
    return cls("h", "", [x], data), data


def same(h, ctx):
    actual = np.asarray(h.compute(ctx))
    expected = h._uncached_weights(h._columns(ctx))
    assert actual.shape == expected.shape
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    return actual


@pytest.mark.parametrize("cls", [RooHistFunc, RooHistPdf])
def test_weights_and_ranges_remain_live_after_a_repeated_lookup(cls):
    h, data = made(cls)
    ctx = {"x": np.asarray([0.25, 1.25])}
    assert np.array_equal(same(h, ctx), [10.0, 20.0])
    assert np.array_equal(same(h, ctx), [10.0, 20.0])
    data.set(0, 11.0)
    assert np.array_equal(same(h, ctx), [11.0, 20.0])
    var = data.get().find("x")
    var.setBinning(RooUniformBinning(0.0, 4.0, 2))
    expected = [11.0, 11.0] if cls is RooHistFunc else [5.5, 5.5]
    assert np.array_equal(same(h, ctx), expected)


@pytest.mark.parametrize("cls", [RooHistFunc, RooHistPdf])
def test_moving_an_interior_edge_without_changing_the_range_invalidates_a_lookup(cls):
    h, data = made(cls)
    ctx = {"x": np.asarray([0.75])}
    assert np.array_equal(same(h, ctx), [10.0])
    binning = data.get().find("x").getBinning()
    assert not binning.removeBoundary(1.0)
    binning.addBoundary(0.5)
    expected = 20.0 if cls is RooHistFunc else 20.0 / 1.5
    assert np.array_equal(same(h, ctx), [expected])


@pytest.mark.parametrize("cls", [RooHistFunc, RooHistPdf])
def test_mutating_returned_values_does_not_change_a_later_evaluation(cls):
    h, _ = made(cls)
    ctx = {"x": np.asarray([0.25, 1.25])}
    first = h.compute(ctx)
    first[0] = 999.0
    hit = h.compute(ctx)
    assert np.array_equal(hit, [10.0, 20.0])
    hit[1] = 999.0
    assert np.array_equal(h.compute(ctx), [10.0, 20.0])


def test_mutating_coordinates_and_reshaping_them_cannot_reuse_previous_values():
    h, _ = made()
    coordinates = np.asarray([0.25, 1.25])
    ctx = {"x": coordinates}
    assert np.array_equal(same(h, ctx), [10.0, 20.0])
    coordinates[0] = 1.25
    assert np.array_equal(same(h, ctx), [20.0, 20.0])
    result = same(h, {"x": coordinates.reshape(2, 1)})
    assert np.array_equal(result, [[20.0], [20.0]])
    assert same(h, {"x": 0.25}).shape == ()
    assert same(h, {"x": 0.25}) == 10.0


def test_values_outside_the_histogram_still_have_zero_weight():
    h, _ = made()
    ctx = {"x": np.asarray([-1.0, 2.0, 3.0, np.nan])}
    assert np.array_equal(same(h, ctx), [0.0, 20.0, 0.0, 0.0])
    assert np.array_equal(same(h, ctx), [0.0, 20.0, 0.0, 0.0])


@pytest.mark.parametrize("cls", [RooHistFunc, RooHistPdf])
def test_interpolation_settings_and_later_weights_are_not_hidden_by_a_cache(cls):
    h, data = made(cls)
    ctx = {"x": np.asarray([0.75, 1.25])}
    same(h, ctx)
    h.setInterpolationOrder(1)
    same(h, ctx)
    h.setCdfBoundaries(True)
    same(h, ctx)
    data.set(0, 12.0)
    same(h, ctx)
    h.setInterpolationOrder(0)
    same(h, ctx)


def test_custom_data_histograms_keep_their_own_weights():
    class Data(RooDataHist):
        scale = 1.0

        def weights(self):
            return super().weights() * self.scale

    x = RooRealVar("x", "", 0.0, 2.0)
    x.setBins(2)
    data = Data("data", "", [x])
    data.set(0, 10.0)
    data.set(1, 20.0)
    h = RooHistFunc("h", "", [x], data)
    ctx = {"x": np.asarray([0.25, 1.25])}
    assert np.array_equal(same(h, ctx), [10.0, 20.0])
    data.scale = 2.0
    assert np.array_equal(same(h, ctx), [20.0, 40.0])


def test_custom_binning_can_change_its_interior_edges():
    class Binning(RooUniformBinning):
        middle = 1.0

        def array(self):
            return np.asarray([0.0, self.middle, 2.0])

    x = RooRealVar("x", "", 0.0, 2.0)
    x.setBinning(Binning(0.0, 2.0, 2))
    data = RooDataHist("data", "", [x])
    data.set(0, 10.0)
    data.set(1, 20.0)
    h = RooHistFunc("h", "", [x], data)
    ctx = {"x": np.asarray([0.75])}
    assert np.array_equal(same(h, ctx), [10.0])
    data.get().find("x").getBinning().middle = 0.5
    assert np.array_equal(same(h, ctx), [20.0])


def test_custom_histogram_functions_keep_their_own_bin_lookup():
    class Hist(RooHistFunc):
        shift = 0

        def _located(self, columns, shape):
            index, inside = super()._located(columns, shape)
            return (index + self.shift) % 2, inside

    h, _ = made(Hist)
    ctx = {"x": np.asarray([0.25, 1.25])}
    assert np.array_equal(same(h, ctx), [10.0, 20.0])
    h.shift = 1
    assert np.array_equal(same(h, ctx), [20.0, 10.0])


def test_large_coordinate_columns_keep_their_complete_output():
    h, _ = made()
    coordinates = np.resize([0.25, 1.25], 4097)
    assert np.array_equal(same(h, {"x": coordinates}), np.resize([10.0, 20.0], 4097))
    coordinates[0] = 1.25
    assert same(h, {"x": coordinates})[0] == 20.0


def test_large_histograms_keep_their_live_weights():
    x = RooRealVar("x", "", 0.0, 4097.0)
    x.setBins(4097)
    data = RooDataHist("data", "", [x])
    data.set(0, 10.0)
    h = RooHistFunc("h", "", [x], data)
    ctx = {"x": np.asarray([0.25])}
    assert np.array_equal(same(h, ctx), [10.0])
    data.set(0, 20.0)
    assert np.array_equal(same(h, ctx), [20.0])


def test_custom_variables_keep_their_dynamic_range():
    class Variable(RooRealVar):
        threshold = 0.0

        def getMin(self, name=None):
            return max(super().getMin(name), self.threshold)

    x = Variable("x", "", 0.0, 2.0)
    x.setBins(2)
    data = RooDataHist("data", "", [x])
    data.set(0, 10.0)
    data.set(1, 20.0)
    h = RooHistFunc("h", "", [x], data)
    ctx = {"x": np.asarray([0.25, 1.25])}
    assert np.array_equal(same(h, ctx), [10.0, 20.0])
    data.get().find("x").threshold = 0.5
    assert np.array_equal(same(h, ctx), [0.0, 20.0])


def test_histogram_parameters_can_vary_for_each_event():
    x = RooRealVar("x", "", 0.0, 2.0)
    x.setBins(2)
    first, second = RooRealVar("first", "", 1.0), RooRealVar("second", "", 2.0)
    hist = ParamHistFunc("parameters", "", [x], [first, second])
    values = hist.compute({"x": np.array([0.25, 1.25]), "first": np.array([1.0, 2.0]),
                           "second": np.array([3.0, 4.0])})
    assert np.array_equal(values, [1.0, 4.0])
    first_values = np.array([5.0, 6.0])
    assert np.array_equal(hist.compute({"x": np.array([0.25, 0.25]),
                                       "first": first_values}), first_values)


def test_scalar_histogram_lookups_accept_zero_dimensional_parameter_arrays():
    x = RooRealVar("x", "", 0.0, 2.0)
    x.setBins(2)
    first, second = RooRealVar("first", "", 1.0), RooRealVar("second", "", 2.0)
    hist = ParamHistFunc("parameters", "", [x], [first, second])
    result = hist.compute({"x": 0.25, "first": np.asarray(1.5)})
    assert type(result) is float and result == 1.5


def test_an_empty_parameter_histogram_has_no_value_to_look_up():
    hist = ParamHistFunc("empty", "", [], [])
    with pytest.raises(IndexError):
        hist.compute({})
