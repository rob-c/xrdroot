"""Prepared native inputs keep every constraint's original kernel and fallback behavior."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from test_roofit_hf_fit import model, same
from xrdroot.roofit.collections import RooArgList, RooArgSet
from xrdroot.roofit.fitting import hfconstraints, hffit, scalar
from xrdroot.roofit.fitting.minimizer import RooMinimizer
from xrdroot.roofit.functions import RooAddition, RooProduct
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.pdfs.gamma import RooGamma
from xrdroot.roofit.pdfs.prodpdf import RooProdPdf
from xrdroot.roofit.pdfs.shapes import RooPoisson
from xrdroot.roofit.variables import RooConstVar, RooRealVar


def constrained():
    channel, _, _ = model()
    background = RooRealVar('nb', '', 1., .05, 5.)
    nominal = RooRealVar('nominal', '', 1.)
    sigma = RooRealVar('sigma', '', .2)
    gaussian = RooGaussian('gaussian', '', nominal, background, sigma)
    count = RooRealVar('count', '', 5.4)
    scale = RooRealVar('scale', '', 7.)
    mean = RooProduct('mean', '', RooArgList(background, scale))
    poisson = RooPoisson('poisson', '', count, mean)
    prod = RooProdPdf('prod', '', RooArgList(channel.sum('main', background), gaussian, poisson))
    data = channel.hist('observed', (5., 7., 2., 1.))
    nll = prod.createNLL(data, Constrain=RooArgSet(background), GlobalObservables=RooArgSet())
    minimizer = RooMinimizer(nll)
    minimizer._optimizing = True
    minimizer.setPrintLevel(-1)
    return nll, minimizer, background, nominal, sigma, count, scale, poisson


def reference(nll):
    return scalar.constraint_sum(nll, nll._constraint_value)


def test_prepared_constraints_read_live_parameters_and_keep_the_kernel_bits():
    nll, minimizer, background, *_ = constrained()
    assert hffit.constraints(nll) is None
    with hffit.within(minimizer, minimizer._migrad):
        assert hffit._ACTIVE.get().constraints is not None
        for value in (.1, .7, 1., 1.3, 2.):
            background.setVal(value)
            same(hffit.constraints(nll), reference(nll))
            same(nll._constraint_sum(), reference(nll))
    assert hffit.constraints(nll) is None


@pytest.mark.parametrize('rounding', [False, True])
def test_fractional_poisson_counts_keep_the_configured_rounding(rounding):
    nll, minimizer, *_, poisson = constrained()
    poisson.setNoRounding(rounding)
    with hffit.within(minimizer, minimizer._migrad):
        assert hffit._ACTIVE.get().constraints is not None
        same(hffit.constraints(nll), reference(nll))


def test_global_observations_are_refreshed_between_fits():
    nll, minimizer, _, nominal, *_ = constrained()
    for value in (.8, 1., 1.2):
        nll._global_values[nominal.GetName()] = value
        with hffit.within(minimizer, minimizer._migrad):
            assert hffit._ACTIVE.get().constraints is not None
            same(hffit.constraints(nll), reference(nll))


@pytest.mark.parametrize('sigma_value', [0., -1., math.nan, math.inf, 1e-200])
def test_invalid_gaussian_points_return_to_the_original_evaluator(sigma_value):
    nll, minimizer, _, _, sigma, *_ = constrained()
    with hffit.within(minimizer, minimizer._migrad):
        assert hffit._ACTIVE.get().constraints is not None
        sigma.setVal(sigma_value)
        assert hffit.constraints(nll) is None
        if sigma_value * sigma_value == 0.:
            # The existing scalar kernel divides by zero before reporting this point.
            with pytest.raises(ZeroDivisionError):
                reference(nll)
            with pytest.raises(ZeroDivisionError):
                nll._constraint_sum()
        else:
            with np.errstate(all='ignore'):
                same(nll._constraint_sum(), reference(nll))


@pytest.mark.parametrize('count_value', [-1., math.nan, math.inf])
def test_invalid_poisson_counts_return_to_the_original_evaluator(count_value):
    nll, minimizer, _, _, _, count, *_ = constrained()
    with hffit.within(minimizer, minimizer._migrad):
        assert hffit._ACTIVE.get().constraints is not None
        count.setVal(count_value)
        assert hffit.constraints(nll) is None
        with np.errstate(all='ignore'):
            same(nll._constraint_sum(), reference(nll))


@pytest.mark.parametrize('mean_value', [0., -1., math.nan, math.inf])
def test_invalid_poisson_expectations_keep_the_original_log_rules(mean_value):
    nll, minimizer, _, _, _, _, scale, _ = constrained()
    with hffit.within(minimizer, minimizer._migrad):
        assert hffit._ACTIVE.get().constraints is not None
        scale.setVal(mean_value)
        assert hffit.constraints(nll) is None
        with np.errstate(all='ignore'):
            same(nll._constraint_sum(), reference(nll))


def test_normalized_constraints_keep_the_existing_integral():
    nll, minimizer, background, *_ = constrained()
    nll._constrained_over = frozenset([background.GetName()])
    nll.__dict__.pop('_constrained_cache', None)
    with hffit.within(minimizer, minimizer._migrad):
        assert hffit._ACTIVE.get().constraints is None
        assert hffit.constraints(nll) is None
        same(nll._constraint_sum(), reference(nll))


def test_gamma_constraints_keep_the_existing_evaluator():
    nll, minimizer, background, *_ = constrained()
    gamma = RooGamma('gamma', '', background, RooConstVar('shape', '', 2.3),
                     RooConstVar('width', '', .7), RooConstVar('origin', '', 0.))
    nll.constraints.append(gamma)
    with hffit.within(minimizer, minimizer._migrad):
        assert hffit._ACTIVE.get().constraints is None
        same(nll._constraint_sum(), reference(nll))


def test_a_cyclic_input_does_not_leave_the_builder_active():
    node = RooProduct('cycle', '', [])
    node.terms._list.append(node)
    inputs = hfconstraints.Inputs({})
    with pytest.raises(hfconstraints.NotPrepared):
        inputs.bind(node)
    assert not inputs.active
    assert inputs.bind(RooConstVar('valid', '', .7))() == .7


def test_unknown_real_expressions_keep_the_existing_evaluator():
    inputs = hfconstraints.Inputs({})
    node = RooAddition('sum', '', RooArgList(RooConstVar('one', '', 1.)))
    with pytest.raises(hfconstraints.NotPrepared):
        inputs.bind(node)


def test_an_invalid_rounding_flag_is_not_evaluated_while_preparing():
    nll, _, *_, poisson = constrained()
    class Callback:
        def __bool__(self):
            raise AssertionError('A preparation guard ran a callback.')
    poisson._no_rounding = Callback()
    assert hfconstraints.prepare(nll) is None


def test_negative_numpy_widths_keep_the_original_warning_count():
    nll, minimizer, _, _, sigma, *_ = constrained()
    sigma._val = np.float64(-1e200)
    with hffit.within(minimizer, minimizer._migrad):
        assert hffit._ACTIVE.get().constraints is None
        with warnings.catch_warnings(record=True) as expected:
            warnings.simplefilter('always')
            reference(nll)
        with warnings.catch_warnings(record=True) as actual:
            warnings.simplefilter('always')
            nll._constraint_sum()
    assert [(one.category, str(one.message)) for one in actual] == [
        (one.category, str(one.message)) for one in expected]


def test_numpy_constants_keep_the_original_python_double_arithmetic():
    node = RooConstVar('huge', '', 1.)
    node._val = np.float64(1e308)
    expression = RooProduct('product', '', RooArgList(RooConstVar('two', '', 2.), node))
    evaluate = hfconstraints.Inputs({}).bind(expression)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        expected = scalar._real(expression, {})
        actual = evaluate()
    assert not recorded
    assert type(actual) is float
    same(actual, expected)


def test_numpy_variable_products_keep_the_original_warning_count():
    nll, minimizer, background, _, _, _, scale, _ = constrained()
    background.setVal(2.)
    scale._val = np.float64(1e308)
    with hffit.within(minimizer, minimizer._migrad):
        assert hffit._ACTIVE.get().constraints is None
        with warnings.catch_warnings(record=True) as expected:
            warnings.simplefilter('always')
            reference(nll)
        with warnings.catch_warnings(record=True) as actual:
            warnings.simplefilter('always')
            nll._constraint_sum()
    assert [(one.category, str(one.message)) for one in actual] == [
        (one.category, str(one.message)) for one in expected]


@pytest.mark.parametrize('position', [2, 3, 4, 5, 6])
def test_numpy_variables_use_the_existing_constraint_evaluator(position):
    parts = constrained()
    nll, minimizer = parts[:2]
    variable = parts[position]
    variable._val = np.float64(variable.getVal())
    with hffit.within(minimizer, minimizer._migrad):
        assert hffit._ACTIVE.get().constraints is None
        same(nll._constraint_sum(), reference(nll))


def test_global_observations_keep_their_original_scalar_conversion():
    nll, minimizer, _, nominal, *_ = constrained()
    nominal._val = np.float64(.7)
    nll._global_values[nominal.GetName()] = np.float64(.8)
    with hffit.within(minimizer, minimizer._migrad):
        assert hffit._ACTIVE.get().constraints is not None
        same(hffit.constraints(nll), reference(nll))
