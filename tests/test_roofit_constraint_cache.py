"""Repeated inputs can reuse native kernels while every live getter is still read."""
from __future__ import annotations

import gc
import math
import warnings
import weakref

import numpy as np
import pytest

from test_roofit_bound_constraints import constrained, reference
from test_roofit_hf_fit import same
from xrdroot.roofit import kernels
from xrdroot.roofit.fitting import hfconstraints, hffit


def prepared(kind):
    fixture = constrained()
    result = hfconstraints.prepare(fixture[0])
    assert result is not None
    return next(term for term in result.terms if isinstance(term, kind)), fixture


def counted_exp(monkeypatch):
    calls = []
    original = kernels.fast_exp
    def exp(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(kernels, 'fast_exp', exp)
    return calls


@pytest.mark.parametrize('kind', [hfconstraints.Gaussian, hfconstraints.Poisson])
def test_only_the_last_constraint_input_is_reused_and_updates_are_read_live(kind, monkeypatch):
    term, fixture = prepared(kind)
    background = fixture[2]
    calls = counted_exp(monkeypatch)
    values = []
    for value in (1., 1., 1.2, 1.2, 1., 1.):
        background.setVal(value)
        values.append(term.value())
    assert values[0] == values[1] == values[4] == values[5]
    assert values[2] == values[3] and values[2] != values[0]
    assert len(calls) == 3


@pytest.mark.parametrize('kind,names', [(hfconstraints.Gaussian, ['x', 'mean', 'sigma']),
                                      (hfconstraints.Poisson, ['x', 'mean'])])
def test_cache_hits_preserve_the_order_and_number_of_live_input_reads(kind, names):
    term, _ = prepared(kind)
    calls = []
    def reader(name, original):
        def read():
            calls.append(name)
            return original()
        return read
    for name in names:
        setattr(term, name, reader(name, getattr(term, name)))
    found = [term.value() for _ in range(3)]
    assert found[0] == found[1] == found[2]
    assert calls == names * 3


class ForeignFloat(float):
    def __hash__(self):
        raise AssertionError('A non-native input was used as a cache key.')


@pytest.mark.parametrize('kind,field', [
    (hfconstraints.Gaussian, 'x'), (hfconstraints.Gaussian, 'mean'),
    (hfconstraints.Gaussian, 'sigma'), (hfconstraints.Poisson, 'x'),
    (hfconstraints.Poisson, 'mean')])
@pytest.mark.parametrize('value', [1, np.float64(1.), np.array(1.), ForeignFloat(1.)])
def test_non_builtin_floats_keep_the_uncached_kernel_and_never_enter_hashing(
    kind, field, value, monkeypatch
):
    term, _ = prepared(kind)
    setattr(term, field, lambda: value)
    calls = counted_exp(monkeypatch)
    result = [term.value() for _ in range(3)]
    assert result[0] == result[1] == result[2]
    assert len(calls) == 3


@pytest.mark.parametrize('kind', [hfconstraints.Gaussian, hfconstraints.Poisson])
def test_cache_storage_is_owned_by_one_constraint_not_by_its_model(kind):
    def references():
        term, fixture = prepared(kind)
        term.value()
        return weakref.ref(term), weakref.ref(fixture[2])
    term, parameter = references()
    gc.collect()
    assert term() is None and parameter() is None


@pytest.mark.parametrize('count', [0., 5.4, 1.25])
def test_poisson_rounding_changes_are_included_in_the_cached_inputs(count):
    term, _ = prepared(hfconstraints.Poisson)
    term.x = lambda: count
    term.mean = lambda: 7.
    from xrdroot.roofit.pdfs.shapes import _poisson_single
    for rounding in (False, False, True, True, False):
        term.no_rounding = rounding
        expected = _poisson_single(count if rounding else float(math.floor(count)), 7.)
        same(term.value(), expected)


@pytest.mark.parametrize('field,value,expected', [
    ('x', math.nan, math.nan), ('mean', math.inf, 0.),
    ('sigma', 0., math.nan), ('sigma', -1., math.nan), ('sigma', 1e-200, math.nan)])
def test_invalid_gaussian_inputs_after_a_cache_hit_keep_the_original_refusal(
    field, value, expected
):
    term, _ = prepared(hfconstraints.Gaussian)
    first = term.value()
    assert term.value() == first
    original = getattr(term, field)
    setattr(term, field, lambda: value)
    same(term.value(), expected)
    same(term.value(), expected)
    setattr(term, field, original)
    same(term.value(), first)


@pytest.mark.parametrize('field,value', [('x', math.nan), ('x', -1.), ('x', math.inf),
                                       ('mean', 0.), ('mean', -1.), ('mean', math.inf)])
def test_invalid_poisson_inputs_after_a_cache_hit_keep_the_original_refusal(field, value):
    term, _ = prepared(hfconstraints.Poisson)
    first = term.value()
    assert term.value() == first
    original = getattr(term, field)
    setattr(term, field, lambda: value)
    assert math.isnan(term.value()) and math.isnan(term.value())
    setattr(term, field, original)
    same(term.value(), first)


def test_non_native_poisson_rounding_metadata_does_not_enter_the_cache(monkeypatch):
    term, _ = prepared(hfconstraints.Poisson)
    term.no_rounding = np.bool_(True)
    calls = counted_exp(monkeypatch)
    found = [term.value() for _ in range(3)]
    assert found[0] == found[1] == found[2] and len(calls) == 3


def test_cached_constraint_inputs_follow_every_parameter_and_global_observation_change():
    fixture = constrained()
    nll, minimizer, background, nominal, sigma, count, scale, _ = fixture
    with hffit.within(minimizer, minimizer._migrad):
        assert hffit._ACTIVE.get().constraints is not None
        for variable in (background, nominal, sigma, count, scale):
            initial = variable.getVal()
            for value in (initial, initial, initial * 1.125, initial * 1.125, initial):
                variable.setVal(value)
                same(hffit.constraints(nll), reference(nll))


def test_non_native_inputs_preserve_a_warning_on_every_kernel_evaluation():
    term, _ = prepared(hfconstraints.Gaussian)
    term.x = lambda: np.float64(1e200)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert term.value() == 0.
        assert term.value() == 0.
    assert len(caught) == 2
    assert all(item.category is RuntimeWarning and 'overflow' in str(item.message)
               for item in caught)
