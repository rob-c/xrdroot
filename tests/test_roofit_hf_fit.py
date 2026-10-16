"""A fit may reuse native geometry; its data, callbacks and numeric behavior stay observable."""
from __future__ import annotations

import gc
import math
import weakref

import numpy as np
import pytest

from test_roofit_branch_binned import Channel, _constrained
from xrdroot.roofit.fitting import hffit
from xrdroot.roofit.fitting.minimizer import RooMinimizer
from xrdroot.roofit.variables import RooConstVar


def model():
    channel = Channel(signal=(1., 4., 2., 1.))
    nll = channel.model.createNLL(channel.hist('observed', (5., 7., 2., 1.)))
    minimizer = RooMinimizer(nll)
    minimizer._optimizing = True
    minimizer.setPrintLevel(-1)
    return channel, nll, minimizer


def same(actual, expected):
    assert np.asarray(actual).view(np.uint64) == np.asarray(expected).view(np.uint64)


def test_live_parameters_keep_the_original_bin_arithmetic():
    channel, nll, minimizer = model()
    assert hffit.total(nll) is None
    with hffit.within(minimizer, minimizer._migrad):
        plan = hffit._ACTIVE.get()
        assert plan is not None
        for value in (.01, .5, 1., 1.3, 4., 9.99):
            channel.mu.setVal(value)
            expected = nll.channel(nll.pdf, None)
            same(hffit.total(nll), expected)
            same(nll.getVal(), expected)
    assert hffit.total(nll) is None


def test_constraints_keep_their_original_values_and_order():
    channel, _, _ = model()
    prod, background, nominal = _constrained(channel)
    nll = prod.createNLL(channel.hist('observed', (5., 7., 2., 1.)),
                        GlobalObservables=[nominal])
    minimizer = RooMinimizer(nll)
    minimizer._optimizing = True
    minimizer.setPrintLevel(-1)
    with hffit.within(minimizer, minimizer._migrad):
        assert hffit._ACTIVE.get() is not None
        for value in (.8, 1., 1.3):
            background.setVal(value)
            same(nll.getVal(), nll.channel(nll.pdf, None) + nll._constraint_sum())


def test_a_new_fit_uses_changed_data_and_histogram_weights():
    channel, nll, minimizer = model()
    before = hffit._prepare(minimizer, minimizer._migrad)
    assert before is not None
    channel.sig.data._weights *= 1.5
    nll.setData(channel.hist('replacement', (3., 4., 5., 2.)))
    after = hffit._prepare(minimizer, minimizer._migrad)
    assert after is not None
    assert before.predict() != after.predict()
    same(after.total(), nll.channel(nll.pdf, None))


@pytest.mark.parametrize('squared', [False, True])
def test_fractional_and_empty_counts_keep_the_original_poisson_sum(squared):
    channel, nll, minimizer = model()
    nll.setData(channel.hist('fractional', (0., .125, 2.3, 1e-12)))
    nll.applyWeightSquared(squared)
    plan = hffit._prepare(minimizer, minimizer._migrad)
    assert plan is not None
    same(plan.total(), nll.channel(nll.pdf, None))


@pytest.mark.parametrize('value', [0., -1., math.nan, math.inf])
def test_invalid_expectations_use_the_original_diagnostics(value):
    _, _, minimizer = model()
    plan = hffit._prepare(minimizer, minimizer._migrad)
    assert plan is not None
    plan.predict = lambda: (value, 1., 1., 1.)
    assert plan.total() is None


def test_a_tiny_positive_empty_bin_keeps_the_zero_term():
    _, _, minimizer = model()
    plan = hffit._prepare(minimizer, minimizer._migrad)
    assert plan is not None
    plan.predict = lambda: (1e-12, 1., 1., 1.)
    plan.counts[0] = 1e-12
    plan.factorials[0] = 0.
    assert math.isfinite(plan.total())


def test_nested_fits_and_exceptions_restore_the_outer_plan():
    _, nll, minimizer = model()
    _, other, inner = model()
    with hffit.within(minimizer, minimizer._migrad):
        outer = hffit._ACTIVE.get()
        with pytest.raises(RuntimeError, match='interrupted'):
            with hffit.within(inner, inner._migrad):
                assert hffit.total(nll) is None
                assert hffit.total(other) is not None
                raise RuntimeError('interrupted')
        assert hffit._ACTIVE.get() is outer
    assert hffit._ACTIVE.get() is None


def test_a_finished_fit_does_not_retain_the_model():
    _, nll, minimizer = model()
    reference = weakref.ref(nll)
    with hffit.within(minimizer, minimizer._migrad):
        assert hffit._ACTIVE.get() is not None
    del nll, minimizer
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize('field,value', [('binned', False), ('rng', 'range'),
    ('_copies', {1:object()}), ('_copies', []), ('constraints', ()),
    ('_global_values', {'bad':object()}), ('w', np.ones(4,dtype=np.float32)),
    ('w', np.zeros(0)), ('w', np.ones(9)), ('columns', {'x': np.zeros(3)}),
    ('columns', {'x': [1.,2.,3.,4.]})])
def test_unsupported_likelihood_state_keeps_the_array_evaluator(field, value):
    _, nll, minimizer = model()
    setattr(nll, field, value)
    assert hffit._prepare(minimizer, minimizer._migrad) is None


@pytest.mark.parametrize('field,value', [('_optimizing',False), ('verbose',True),
                                         ('print_level',0)])
def test_only_quiet_constant_optimized_fits_reuse_structure(field, value):
    _, _, minimizer = model()
    setattr(minimizer,field,value)
    assert hffit._prepare(minimizer, minimizer._migrad) is None


def test_a_custom_minuit_driver_keeps_model_mutations_observable():
    _, _, minimizer = model()
    assert hffit._prepare(minimizer, lambda m: None) is None


def test_explicit_numpy_underflow_reporting_keeps_array_warnings():
    _, _, minimizer = model()
    with np.errstate(under='warn'):
        assert hffit._prepare(minimizer, minimizer._migrad) is None


@pytest.mark.parametrize('value', [-1., math.nan, math.inf, 1e101])
def test_counts_outside_the_finite_nonnegative_domain_are_not_cached(value):
    _, nll, minimizer = model()
    nll.w[0] = value
    assert hffit._prepare(minimizer, minimizer._migrad) is None


def test_custom_model_methods_are_called_by_the_original_evaluator():
    channel, nll, minimizer = model()
    calls=[]
    compute=channel.sig.compute
    channel.sig.compute=lambda ctx: (calls.append(1) or compute(ctx))
    assert hffit._prepare(minimizer, minimizer._migrad) is None
    nll.getVal()
    assert calls == [1]


def test_direct_collection_edits_are_checked_without_a_graph_version_change():
    channel, _, minimizer = model()
    channel.model.funcs._list.append(RooConstVar('extra','',2.))
    assert hffit._prepare(minimizer, minimizer._migrad) is None


def test_float32_observable_arithmetic_keeps_numpy_promotion_rules():
    _, nll, minimizer = model()
    nll.columns['x']=nll.columns['x'].astype(np.float32)
    assert hffit._prepare(minimizer, minimizer._migrad) is None


def test_a_removed_binned_attribute_is_checked_at_the_next_fit():
    channel, _, minimizer = model()
    channel.model.setAttribute('BinnedLikelihood',False)
    assert hffit._prepare(minimizer,minimizer._migrad) is None


def test_a_histogram_mapping_requires_string_field_names():
    channel, _, minimizer = model()
    channel.sig._given={1:channel.x}
    assert hffit._prepare(minimizer,minimizer._migrad) is None


def test_custom_minimizer_subclasses_keep_the_original_evaluator():
    _, nll, _ = model()
    class CustomMinimizer(RooMinimizer):
        pass
    minimizer=CustomMinimizer(nll)
    minimizer._optimizing=True
    assert hffit._prepare(minimizer,minimizer._migrad) is None


@pytest.mark.parametrize('value', [1e75, 1e200, -1e200, math.inf, math.nan])
@pytest.mark.parametrize('overflow', ['warn', 'raise'])
def test_rejected_squared_weights_do_not_emit_a_planning_warning(value, overflow):
    import warnings

    channel, nll, minimizer = model()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nll.setData(channel.hist('large', (value, 7., 2., 1.)))
    nll.applyWeightSquared(True)
    with np.errstate(over=overflow), warnings.catch_warnings(record=True) as found:
        warnings.simplefilter('always')
        assert hffit._prepare(minimizer, minimizer._migrad) is None
    assert not found
