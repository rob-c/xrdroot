"""A one-event likelihood keeps the general evaluator's bits and mutable model state."""
from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot.roofit import evalerrors, kernels, mathfuncs
from xrdroot.roofit.arg import graph_changed
from xrdroot.roofit.data.dataset import RooDataSet
from xrdroot.roofit.fitting.nll import RooNLLVar
from xrdroot.roofit.fitting.scalar import _density, _gaussian_integral, _NotScalar, one_entry
from xrdroot.roofit.functions import RooAddition, RooProduct
from xrdroot.roofit.pdfs.basic import RooExponential, RooGaussian
from xrdroot.roofit.pdfs.prodpdf import RooProdPdf
from xrdroot.roofit.pdfs.shapes import RooPoisson
from xrdroot.roofit.variables import RooConstVar, RooRealVar


def _data(*variables, weight=1.0):
    data = RooDataSet("data", "", variables)
    data.add_columns({one.GetName(): [one.getVal()] for one in variables}, [weight])
    return data


def _same(nll):
    actual = one_entry(nll)
    assert actual is not None
    nll._badness = 0.0
    expected = nll._general_sum()
    assert nll._badness == 0.0
    assert np.asarray(actual).view(np.uint64) == np.asarray(expected).view(np.uint64)
    return actual


@pytest.mark.parametrize("low,high", [(-10., 10.), (-8., -3.), (2., 10.),
                                     (-math.inf, math.inf), (0., 0.), (-1., 0.), (0., 1.)])
def test_scalar_gaussian_normalization_keeps_each_upper_tail_bit(low, high):
    actual = _gaussian_integral(low, high, 0., 1.3)
    expected = mathfuncs.gaussian_integral(low, high, 0., 1.3)
    assert np.asarray(actual).view(np.uint64) == np.asarray(expected).view(np.uint64)


@pytest.mark.parametrize("observed", [0., 0.25, 1., 150., 299.9])
@pytest.mark.parametrize("weight", [1., 0.25, -2.])
@pytest.mark.parametrize("squared", [False, True])
def test_a_weighted_count_keeps_the_general_likelihoods_bits(observed, weight, squared):
    x, mean = RooRealVar("x", "", observed, 0., 300.), RooRealVar("mu", "", 150., 0., 500.)
    pdf = RooPoisson("p", "", x, mean)
    nll = RooNLLVar(pdf, _data(x, weight=weight))
    nll.applyWeightSquared(squared)
    _same(nll)
    pdf.setNoRounding()
    _same(nll)
    mean.setVal(225.)
    x.setMax(400.)
    _same(nll)


@pytest.mark.parametrize("normalize", ["x", "mean", "neither"])
def test_a_gaussian_normalizes_over_the_requested_variable(normalize):
    x = RooRealVar("x", "", 0.25, -5., 5.)
    mean = RooRealVar("mean", "", 0.1, -2., 2.)
    pdf = RooGaussian("g", "", x, mean, 1.3)
    obs = {"x": x, "mean": mean, "neither": RooRealVar("unused", "", 2.)}[normalize]
    nll = RooNLLVar(pdf, _data(obs))
    _same(nll)
    x.setVal(-1.)
    mean.setVal(0.3)
    mean.setMax(3.)
    _same(nll)


def _counting():
    x = RooRealVar("x", "", 150., 0., 300.)
    signal = RooRealVar("s", "", 50., 0., 120.)
    efficiency = RooRealVar("eff", "", 1., 0., 3.)
    scaled = RooProduct("scaled", "", [signal, efficiency])
    background = RooConstVar("b", "", 100.)
    mean = RooAddition("mean", "", [scaled, background])
    count = RooPoisson("count", "", x, mean)
    nominal = RooRealVar("nominal", "", 1., 0., 3.)
    constraint = RooGaussian("constraint", "", nominal, efficiency, 0.05)
    model = RooProdPdf("model", "", [count, constraint])
    nll = RooNLLVar(model, _data(x), constraints=[constraint], constrained=frozenset(["eff"]),
                    global_values={"nominal": 1.01, "b": -1000.})
    return nll, signal, efficiency, background


def test_counting_constraints_and_later_parameter_changes_keep_the_same_sum():
    nll, signal, efficiency, background = _counting()
    _same(nll)
    signal.setVal(80.)
    efficiency.setVal(0.95)
    _same(nll)
    # A named constant ignores a context entry with the same name.
    assert background.getVal() == 100.
    nll._global_values["nominal"] = 1.02
    _same(nll)
    replacement = _data(nll.pdf.pdfs[0].x, weight=0.25)
    nll.setData(replacement)
    _same(nll)


@pytest.mark.parametrize("size", [0, 2, 9, 17, 100])
def test_long_parameter_sums_keep_numpy_reduction_order(size):
    x = RooRealVar("x", "", 3., 0., 300.)
    values = [1e16, 1., -1e16, 1.] * size + [10.]
    terms = [RooConstVar(f"term{i}", "", value) for i, value in enumerate(values)]
    mean = RooAddition("mean", "", terms)
    nll = RooNLLVar(RooPoisson("p", "", x, mean), _data(x))
    _same(nll)
    # Low-level proxy edits invalidate the graph, as server redirection does.
    mean.terms.add(RooConstVar("extra", "", 2.))
    graph_changed()
    _same(nll)


def test_empty_parameter_sums_and_products_keep_their_identity():
    x = RooRealVar("x", "", 0.25, -5., 5.)
    zero, one = RooAddition("zero", "", []), RooProduct("one", "", [])
    _same(RooNLLVar(RooGaussian("g", "", x, zero, one), _data(x)))


def test_offset_likelihoods_keep_the_original_first_value_and_reset_for_new_data():
    nll, signal, _, _ = _counting()
    nll.offset = True
    original = _same(nll)
    assert nll.getVal() == 0.
    signal.setVal(75.)
    assert nll.getVal() == _same(nll) - original
    nll.setData(nll.data)
    assert nll.getVal() == 0.


def test_raw_products_keep_factors_that_a_normalized_product_omits():
    nll, _, _, _ = _counting()
    pdf = nll.pdf
    for names in [frozenset(), frozenset(["x"]), frozenset(["unused"])]:
        ctx = {"x": 140.}
        with kernels.likelihood():
            actual = _density(pdf, ctx, names)
            expected = np.asarray(pdf.value({"x": np.asarray([140.])}, names)).item()
        assert np.asarray(actual).view(np.uint64) == np.asarray(expected).view(np.uint64)


def test_user_pdf_subclasses_retain_their_own_evaluator():
    class Gaussian(RooGaussian):
        def compute(self, ctx):
            return super().compute(ctx) * 0.5

    x = RooRealVar("x", "", 0., -5., 5.)
    nll = RooNLLVar(Gaussian("g", "", x, 0., 1.), _data(x))
    assert one_entry(nll) is None
    assert math.isfinite(nll.getVal())


def test_unknown_formulas_retain_their_general_evaluator():
    x = RooRealVar("x", "", 1., 0., 5.)
    nll = RooNLLVar(RooExponential("p", "", x, -1.), _data(x))
    assert one_entry(nll) is None
    assert nll.getVal() == nll._general_sum()


def test_array_global_observables_keep_the_general_evaluation_error():
    nll, _, _, _ = _counting()
    nll._global_values["nominal"] = np.asarray([1., 1.])
    assert one_entry(nll) is None
    with pytest.raises(TypeError, match="arrays"):
        nll.getVal()


def test_unsupported_real_nodes_use_the_complete_model_evaluator():
    from xrdroot.roofit.functions import RooFormulaVar

    x = RooRealVar("x", "", 3., 0., 300.)
    parameter = RooRealVar("parameter", "", 5., 0., 10.)
    formula = RooFormulaVar("mean", "@0+1", [parameter])
    nll = RooNLLVar(RooPoisson("p", "", x, formula), _data(x))
    assert one_entry(nll) is None
    assert math.isfinite(nll.getVal())


def test_overlapping_product_factors_keep_their_joint_integral():
    x = RooRealVar("x", "", 0., -5., 5.)
    pdf = RooProdPdf("p", "", [RooGaussian("g1", "", x, 0., 1.),
                               RooGaussian("g2", "", x, 1., 2.)])
    nll = RooNLLVar(pdf, _data(x))
    assert one_entry(nll) is None
    assert math.isfinite(nll.getVal())


@pytest.mark.parametrize("change", ["named", "numeric", "parameterized"])
def test_alternative_normalization_rules_keep_the_general_integrator(change):
    x = RooRealVar("x", "", 0., -5., 5.)
    pdf = RooGaussian("g", "", x, 0., 1.)
    if change == "named":
        x.setRange("fit", -2., 2.)
        pdf.setNormRange("fit")
    elif change == "numeric":
        pdf.setForceNumInt()
    else:
        low, high = RooRealVar("low", "", -2.), RooRealVar("high", "", 2.)
        x.setRange(low, high)
    nll = RooNLLVar(pdf, _data(x))
    assert one_entry(nll) is None
    assert math.isfinite(nll.getVal())


@pytest.mark.parametrize("pdf_kind", ["gaussian", "poisson"])
def test_integrating_multiple_parameters_keeps_the_general_normalization(pdf_kind):
    x, mean = RooRealVar("x", "", 1., 0., 5.), RooRealVar("mean", "", 2., 0., 5.)
    pdf = (RooGaussian("p", "", x, mean, 1.) if pdf_kind == "gaussian"
           else RooPoisson("p", "", x, mean))
    with pytest.raises(_NotScalar):
        _density(pdf, {"x": 1., "mean": 2.}, frozenset(["x", "mean"]))


@pytest.mark.parametrize("pdf_kind", ["gaussian", "poisson"])
def test_invalid_points_keep_the_general_likelihood_and_error_messages(pdf_kind):
    x = RooRealVar("x", "", 1., 0., 300.)
    parameter = RooRealVar("parameter", "", 0., -5., 5.)
    if pdf_kind == "gaussian":
        parameter.setVal(0.0001)  # a positive width whose observed event underflows to zero
    pdf = (RooGaussian("p", "", x, 0., parameter) if pdf_kind == "gaussian"
           else RooPoisson("p", "", x, parameter))
    nll = RooNLLVar(pdf, _data(x))
    assert one_entry(nll) is None
    evalerrors.clear()
    evalerrors.collecting(True)
    try:
        with np.errstate(all="ignore"):
            nll._badness = 0.
            reference = nll._general_sum()
            expected_badness = nll._badness
            messages, number = evalerrors.text(10), evalerrors.count()
            evalerrors.clear()
            actual = nll.getVal()
        assert math.isnan(reference) == math.isnan(actual)
        assert actual == reference == math.inf
        assert nll._badness == expected_badness
        assert evalerrors.count() == number == 1
        assert evalerrors.text(10) == messages
    finally:
        evalerrors.collecting(False)
        evalerrors.clear()


@pytest.mark.parametrize("entries", [0, 2])
def test_empty_and_multi_event_datasets_retain_the_general_sum(entries):
    x, mean = RooRealVar("x", "", 3., 0., 300.), RooRealVar("mu", "", 5., 0., 10.)
    pdf = RooPoisson("p", "", x, mean)
    data = RooDataSet("data", "", [x])
    data.add_columns({"x": np.full(entries, 3.)})
    nll = RooNLLVar(pdf, data)
    assert one_entry(nll) is None
    assert nll.getVal() == nll._general_sum()
