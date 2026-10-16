"""Batching accepts only independent numeric bindings and known normalization contracts."""

import math
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from test_roofit_parameter_batches import model, scalar
from test_roostats_bayes_batches import like_for
from xrdroot.roofit.categories import RooCategory
from xrdroot.roofit.data.datahist import RooDataHist
from xrdroot.roofit.data.dataset import RooDataSet
from xrdroot.roofit.fitting import batchmodel
from xrdroot.roofit.fitting.nll import RooNLLVar
from xrdroot.roofit.fitting.parameterbatch import evaluate
from xrdroot.roofit.functions import RooProduct
from xrdroot.roofit.pdfs.basic import RooGaussian, RooUniform
from xrdroot.roofit.pdfs.prodpdf import RooProdPdf
from xrdroot.roofit.pdfs.simultaneous import RooSimultaneous
from xrdroot.roofit.variables import RooConstVar, RooRealVar
from xrdroot.roostats import bayesbatch, bayesscan
from xrdroot.roostats.bayesfuncs import Functor, Posterior, PosteriorFunction


@pytest.mark.parametrize("points", [[1.0, 1.0, 0.0], [[1.0, 1.0]], [[1.0, 1.0, math.nan]]])
def test_incomplete_or_nonfinite_parameter_points_use_scalar_evaluation(points):
    nll, parameters = model()
    assert evaluate(nll, parameters, points) is None


def test_an_empty_parameter_batch_has_no_values_or_assignments():
    nll, parameters = model()
    before = [p.getVal() for p in parameters]
    assert evaluate(nll, parameters, np.empty((0, 3))).shape == (0,)
    assert [p.getVal() for p in parameters] == before


@pytest.mark.parametrize("value", [np.float32(0.1), Fraction(1, 3), "0", None, math.nan, 10**1000])
def test_non_double_global_values_keep_the_original_arithmetic(value):
    nll, parameters = model()
    nll._global_values["nominal"] = value
    assert evaluate(nll, parameters, np.asarray([[1.0, 1.0, 0.0]])) is None


@pytest.mark.parametrize("option", [{"offset": True}, {"rng": "named"}, {"copies": {1: "state"}}])
def test_range_offset_and_copy_state_are_owned_by_the_scalar_evaluator(option):
    old, parameters = model()
    nll = RooNLLVar(old.pdf, old.data, constraints=old.constraints, **option)
    assert evaluate(nll, parameters, np.asarray([[1.0, 1.0, 0.0]])) is None


@pytest.mark.parametrize("weight", [-1.0, math.nan])
def test_negative_or_nonfinite_bin_counts_do_not_enter_the_batch_path(weight):
    nll, parameters = model()
    nll.data.set(0, weight)
    nll.setData(nll.data)
    assert evaluate(nll, parameters, np.asarray([[1.0, 1.0, 0.0]])) is None


def test_empty_bins_keep_the_original_zero_expectation_rule():
    nll, parameters = model()
    observed = RooDataHist("observed_bins", "", nll.data.get())
    observed.set(0, 0.0)
    observed.set(1, 20.0)
    nll.setData(observed)
    parameters[0].setMin(1e-15)
    points = np.asarray([[1e-15, 1.0, 0.0], [1.0, 1.0, 0.0]])
    actual = evaluate(nll, parameters, points)
    assert actual is not None
    assert np.array_equal(actual.view(np.uint64), scalar(nll, parameters, points).view(np.uint64))


def test_zero_expected_counts_return_to_the_original_binned_diagnostics():
    nll, parameters = model()
    parameters[1].setMin(0.0)
    assert evaluate(nll, parameters, np.asarray([[1.0, 0.0, 0.0]])) is None


def test_zero_gaussian_width_keeps_the_scalar_division_error():
    nll, parameters = model()
    nll.constraints[0].sigma._val = 0.0
    assert evaluate(nll, parameters, np.asarray([[1.0, 1.0, 0.0]])) is None
    with pytest.raises(ZeroDivisionError):
        scalar(nll, parameters, [[1.0, 1.0, 0.0]])


@pytest.mark.parametrize("names", [["alpha", "nominal"], ["gamma"]])
def test_unrecognized_constraint_normalization_sets_use_the_original_integrals(names):
    old, parameters = model()
    nll = RooNLLVar(old.pdf, old.data, constraints=old.constraints, constrained=frozenset(names))
    assert evaluate(nll, parameters, np.asarray([[1.0, 1.0, 0.0]])) is None


def test_composite_constraints_retain_their_general_normalization():
    old, parameters = model()
    constraint = RooProdPdf("constraint", "", old.constraints)
    nll = RooNLLVar(old.pdf, old.data, constraints=[constraint])
    assert evaluate(nll, parameters, np.asarray([[1.0, 1.0, 0.0]])) is None


def test_multichannel_likelihoods_keep_channel_weights_and_ignore_absent_channels():
    old, parameters = model(False)
    x = old.pdf.variable("x")
    category = RooCategory("category", "")
    for label in ["a", "b", "absent"]:
        category.defineType(label)
    pdf = RooSimultaneous("sim", "", category)
    for label in ["a", "b", "absent"]:
        pdf.addPdf(old.pdf, label)
    weight = RooRealVar("weight", "", 1.0, 0.0, 100.0)
    data = RooDataSet("combined", "", [x, category, weight], weight_var="weight")
    data.add_columns(
        {
            "x": np.array([0.5, 1.5, 0.5]),
            "category": np.array([0, 0, 1]),
            "weight": np.array([10.0, 20.0, 15.0]),
        }
    )
    nll = RooNLLVar(pdf, data)
    points = np.asarray([[1.0, 1.0], [2.0, 0.5], [0.7, 1.8]])
    actual = evaluate(nll, parameters, points)
    assert actual is not None
    assert np.array_equal(actual.view(np.uint64), scalar(nll, parameters, points).view(np.uint64))


def test_parameter_dimensions_must_match_the_bound_variables():
    nll, parameters = model()
    like = like_for(nll, parameters)
    assert bayesbatch.integrand(like, 4) is None


def test_foreign_callables_retain_their_own_binding_protocol():
    assert bayesbatch.integrand(lambda x: 1.0) is None


def test_prior_bindings_must_not_assign_different_variables_after_the_likelihood():
    nll, parameters = model()
    like = like_for(nll, parameters)
    other = RooRealVar("other", "", 1.0, 0.0, 2.0)
    like.prior = Functor(RooUniform("other_prior", "", [other]), [other])
    assert bayesbatch.integrand(like) is None
    like.prior = SimpleNamespace()
    assert bayesbatch.integrand(like) is None


@pytest.mark.parametrize("setting", ["gaussian", "numeric", "range"])
def test_custom_or_numerically_normalized_priors_use_scalar_evaluation(setting):
    nll, parameters = model()
    like = like_for(nll, parameters)
    if setting == "gaussian":
        like.prior = Functor(nll.constraints[0], parameters)
    elif setting == "numeric":
        like.prior.func.setForceNumInt(True)
    else:
        like.prior.func.setNormRange("range")
    assert bayesbatch.integrand(like) is None


def test_tiny_memory_budgets_retain_the_original_scalar_path(monkeypatch):
    nll, parameters = model()
    like = like_for(nll, parameters)
    monkeypatch.setattr(bayesbatch, "_MAX_CELLS", 1)
    assert bayesbatch.integrand(like) is None


def test_an_empty_binding_batch_does_not_advance_evaluation_counts():
    nll, parameters = model()
    like = like_for(nll, parameters)
    assert bayesbatch.integrand(like)(np.empty((0, 3))).shape == (0,)
    assert like.nll.calls == like.prior.calls == 0


def posterior_for(prior=True):
    nll, parameters = model()
    scale, gamma, alpha = parameters
    uniform = RooUniform("prior", "", [scale]) if prior else None
    post = PosteriorFunction(nll, scale, [gamma, alpha], uniform, 1.0, 0.0, calls=300)
    return nll, parameters, post, Posterior("posterior", post, scale)


def remove_scan_bound(nll, parameters, post, condition):
    if condition == "offset":
        post.like.offset = 100.0
    elif condition == "sigma":
        nll.constraints[0].sigma._val = 2.0
    elif condition == "gaussian_range":
        nll.constraints[0].mean.setRange(-0.5, 0.5)
    elif condition == "raw_poisson":
        nll._constrained_over = frozenset(["nominal"])
        nll.__dict__.pop("_constrained_cache", None)
    elif condition == "large_count":
        observed = RooDataHist("large", "", nll.data.get())
        observed.set(0, 1000001.0)
        observed.set(1, 20.0)
        nll.setData(observed)
    elif condition == "globals":
        nll._global_values["nominal"] = 0.0
    else:
        parameters[0].setRange(0.1, 0.5)


@pytest.mark.parametrize(
    "condition",
    ["offset", "sigma", "gaussian_range", "raw_poisson", "large_count", "globals", "narrow_prior"],
)
def test_scans_without_a_safe_likelihood_bound_keep_sequential_evaluation(condition):
    nll, parameters, post, exact = posterior_for()
    remove_scan_bound(nll, parameters, post, condition)
    before = [p.getVal() for p in parameters]
    assert bayesscan.sample(exact, parameters[0], 0.1, 3.0, 4) is None
    assert post.like.nll.calls == post.like.prior.calls == 0
    assert [p.getVal() for p in parameters] == before


@pytest.mark.parametrize(
    "condition", ["bins", "one_dimension", "different_poi", "foreign_function"]
)
def test_unsupported_scan_bindings_leave_the_original_callable_in_charge(condition):
    _, parameters, post, exact = posterior_for()
    poi = parameters[0]
    bins = 4
    if condition == "bins":
        bins = 0
    elif condition == "one_dimension":
        post.lows = post.lows[:1]
        post.highs = post.highs[:1]
    elif condition == "different_poi":
        poi = RooRealVar(poi)
    else:
        exact = Posterior("custom", lambda x: x * x, poi)
    assert bayesscan.sample(exact, poi, 0.1, 3.0, bins) is None
    assert post.like.nll.calls == 0


def test_gaussian_constraints_can_be_normalized_over_their_first_variable():
    old, parameters = model()
    nll = RooNLLVar(
        old.pdf, old.data, constraints=old.constraints, constrained=frozenset(["alpha", "observed"])
    )
    like = like_for(nll, parameters, False)
    assert batchmodel.bounded_posterior(like, parameters)
    points = np.asarray([[1.0, 1.0, 0.0], [0.5, 1.8, 2.0]])
    actual = evaluate(nll, parameters, points)
    assert actual is not None
    assert np.array_equal(actual.view(np.uint64), scalar(nll, parameters, points).view(np.uint64))


def test_raw_gaussian_constraints_have_no_independent_scan_bound():
    old, parameters = model()
    nll = RooNLLVar(
        old.pdf, old.data, constraints=old.constraints, constrained=frozenset(["observed"])
    )
    assert not batchmodel.bounded_posterior(like_for(nll, parameters, False), parameters)


def test_negative_or_unknown_products_have_no_nonnegative_interval_bound():
    gamma = RooRealVar("gamma", "", 1.0, 0.1, 2.0)
    positive = RooConstVar("positive", "", 100.0)
    assert batchmodel._nonnegative_interval(
        RooProduct("mean", "", [gamma, positive]), {"gamma"}
    ) == (10.0, 200.0)
    negative = RooConstVar("negative", "", -1.0)
    assert batchmodel._nonnegative_interval(
        RooProduct("bad", "", [gamma, negative]), {"gamma"}
    ) == (-math.inf, math.inf)
    assert batchmodel._nonnegative_interval(object(), set()) == (-math.inf, math.inf)


def test_nonfundamental_gaussian_axes_keep_the_general_normalization():
    _, parameters = model()
    gamma = parameters[1]
    derived = RooProduct("derived", "", [gamma, RooConstVar("factor", "", 1.0)])
    assert not batchmodel._covered_domain(derived, gamma, {"gamma"})
    assert not batchmodel._covered_domain(gamma, derived, {"gamma"})


def test_nonfinite_constraint_means_do_not_have_a_uniform_gaussian_bound():
    nll, parameters = model()
    parameters[2].setRange(-math.inf, math.inf)
    assert not batchmodel.bounded_posterior(like_for(nll, parameters, False), parameters)


def test_poisson_bounds_reject_varying_counts_and_large_expectations():
    nll, parameters = model()
    poisson = nll.constraints[1]
    assert not batchmodel._poisson_bound(nll, poisson, {"observed", "gamma"})
    parameters[1].setMax(20000.0)
    assert not batchmodel.bounded_posterior(like_for(nll, parameters, False), parameters)


def test_pointwise_gaussian_normalization_retains_numerical_integration_options():
    nll, parameters = model()
    nll.constraints[0].setForceNumInt(True)
    assert evaluate(nll, parameters, np.asarray([[1.0, 1.0, 0.0]])) is None


def test_a_mixture_of_binned_and_unbinned_channels_keeps_scalar_evaluation():
    old, parameters = model(False)
    x = old.pdf.variable("x")
    category = RooCategory("mixed_category", "")
    category.defineType("binned")
    category.defineType("unbinned")
    pdf = RooSimultaneous("mixed", "", category)
    pdf.addPdf(old.pdf, "binned")
    pdf.addPdf(RooGaussian("continuous", "", x, 1.0, 1.0), "unbinned")
    weight = RooRealVar("weight", "", 1.0, 0.0, 100.0)
    data = RooDataSet("mixed_data", "", [x, category, weight], weight_var="weight")
    data.add_columns(
        {
            "x": np.array([0.5, 1.5]),
            "mixed_category": np.array([0, 1]),
            "weight": np.array([10.0, 20.0]),
        }
    )
    nll = RooNLLVar(pdf, data)
    assert nll.binned
    assert evaluate(nll, parameters, np.asarray([[1.0, 1.0]])) is None


def test_nonuniform_priors_do_not_get_an_independent_scan_bound():
    nll, parameters = model()
    like = like_for(nll, parameters)
    like.prior = Functor(nll.constraints[0], parameters)
    assert not batchmodel.bounded_posterior(like, parameters)


def test_a_varying_zero_width_returns_to_scalar_gaussian_diagnostics():
    old, parameters = model()
    sigma = RooRealVar("sigma", "", 1.0, 0.0, 2.0)
    previous = old.constraints[0]
    gaussian = RooGaussian("varying_constraint", "", previous.x, previous.mean, sigma)
    nll = RooNLLVar(
        old.pdf,
        old.data,
        constraints=[gaussian, old.constraints[1]],
        constrained=frozenset(["nominal", "observed"]),
    )
    assert evaluate(nll, [*parameters, sigma], np.asarray([[1.0, 1.0, 0.0, 0.0]])) is None
    with pytest.raises(ZeroDivisionError):
        scalar(nll, [*parameters, sigma], [[1.0, 1.0, 0.0, 0.0]])


def test_invalid_binned_points_fall_back_without_duplicating_calls():
    nll, parameters = model()
    parameters[1].setMin(0.0)
    like = like_for(nll, parameters, False)
    points = np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    expected = np.asarray([like(point) for point in points])
    like.nll.calls = 0
    like.max = 0.0
    actual = bayesbatch.integrand(like)(points)
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert like.nll.calls == 2


def test_a_retained_nll_offset_is_included_in_the_warning_bound():
    nll, parameters, post, exact = posterior_for()
    nll._offset_value = 100.0
    assert bayesscan.sample(exact, parameters[0], 0.1, 3.0, 4) is None
    assert post.like.nll.calls == 0
