"""Per-fit interpolation retains ROOT's operation order with fixed polynomial coefficients."""
import numpy as np
import pytest

from test_roofit_hf_fit import same
from xrdroot.roofit.collections import RooArgList
from xrdroot.roofit.fitting import hfinterpolation
from xrdroot.roofit.pdfs.histfactory import FlexibleInterpVar
from xrdroot.roofit.variables import RooConstVar, RooRealVar


@pytest.mark.parametrize('code', [4, 5])
@pytest.mark.parametrize('nominal,low,high,boundary', [
    (1., .9, 1.1, 1.), (.3, .1, .7, .4), (3., 1., 6., 2.),
    (.01, 100., .01, 4.), (100., .01, 100., .01)])
def test_live_parameters_keep_every_bit_across_the_interpolation_boundary(
    code, nominal, low, high, boundary
):
    parameter = RooRealVar('alpha', '', 0., -8., 8.)
    node = FlexibleInterpVar('shape', '', RooArgList(parameter), nominal, [low], [high], [code])
    node._boundary = boundary
    evaluate = hfinterpolation.prepare(node)
    assert evaluate is not None
    values = np.r_[np.linspace(-8., 8., 101), -boundary, boundary, 0.]
    for value in values:
        parameter.setVal(float(value))
        same(evaluate(), node.compute({}))


def test_multiple_variations_keep_the_original_add_then_multiply_rounding():
    parameters = [RooRealVar('alpha'+str(i), '', 0., -8., 8.) for i in range(4)]
    node = FlexibleInterpVar('shape', '', RooArgList(*parameters), .7,
                             [.6, .3, .4, .9], [.8, 1.2, .9, .1], [4, 5, 4, 5])
    evaluate = hfinterpolation.prepare(node)
    assert evaluate is not None
    rng = np.random.default_rng(711)
    for values in rng.uniform(-8., 8., (1000, 4)):
        for parameter, value in zip(parameters, values, strict=True):
            parameter.setVal(float(value))
        same(evaluate(), node.compute({}))


@pytest.mark.parametrize('code', [0, 1, 2, 3, 6])
def test_other_interpolation_codes_keep_the_existing_evaluator(code):
    node = FlexibleInterpVar('shape', '', RooArgList(RooConstVar('alpha', '', .2)),
                             1., [.9], [1.1], [code])
    assert hfinterpolation.prepare(node) is None


def test_numpy_metadata_keeps_the_existing_arithmetic():
    node = FlexibleInterpVar('shape', '', RooArgList(RooConstVar('alpha', '', .2)),
                             1., [.9], [1.1], [4])
    node._nominal = np.float64(1.)
    assert hfinterpolation.prepare(node) is None


def test_numpy_parameters_keep_the_existing_arithmetic():
    parameter = RooRealVar('alpha', '', .2)
    parameter._val = np.float64(.2)
    node = FlexibleInterpVar('shape', '', RooArgList(parameter), 1., [.9], [1.1], [4])
    assert hfinterpolation.prepare(node) is None


def curved_fit():
    from test_roofit_hf_fit import model
    from xrdroot.roofit.fitting.minimizer import RooMinimizer

    channel, _, _ = model()
    alpha = RooRealVar('alpha', '', .3, -2., 2.)
    shape = FlexibleInterpVar('shape', '', RooArgList(alpha), 1., [.8], [1.2], [4])
    pdf = channel.sum('curved', shape)
    nll = pdf.createNLL(channel.hist('curved_obs', (5., 7., 2., 1.)))
    minimizer = RooMinimizer(nll)
    minimizer._optimizing = True
    minimizer.setPrintLevel(-1)
    return shape, alpha, nll, minimizer


def test_new_fits_refresh_interpolation_metadata_and_code_changes():
    from xrdroot.roofit.fitting import hffit

    shape, _, nll, minimizer = curved_fit()
    changes = [lambda: None, lambda: shape.setNominal(.7),
               lambda: shape._low.__setitem__(0, .6),
               lambda: shape._high.__setitem__(0, 1.4),
               lambda: setattr(shape, '_boundary', .5),
               lambda: shape.setAllInterpCodes(1)]
    for change in changes:
        change()
        expected = nll.getVal()
        with hffit.within(minimizer, minimizer._migrad):
            assert hffit._ACTIVE.get() is not None
            same(hffit.total(nll), expected)


def test_interpolation_plans_do_not_retain_models_after_the_fit():
    import gc
    import weakref

    from xrdroot.roofit.fitting import hffit

    def completed():
        shape, alpha, _nll, minimizer = curved_fit()
        with hffit.within(minimizer, minimizer._migrad):
            plan = hffit._ACTIVE.get()
            assert plan is not None
            assert any(isinstance(getattr(ref, '__self__', None), hfinterpolation.Interpolation)
                       for ref in plan.predict.__globals__['_refs'])
        return weakref.ref(shape), weakref.ref(alpha)

    references = completed()
    gc.collect()
    assert all(ref() is None for ref in references)


def test_parameter_compute_semantics_are_preserved_when_getval_has_a_native_alias(monkeypatch):
    parameter = RooRealVar('alpha', '', .2, -2., 2.)
    node = FlexibleInterpVar('shape', '', RooArgList(parameter), 1., [.8], [1.2], [4])
    monkeypatch.setattr(RooRealVar, 'getVal', RooConstVar.isConstant)
    evaluate = hfinterpolation.prepare(node)
    assert evaluate is not None
    same(evaluate(), node.compute({}))


@pytest.mark.parametrize('target', ['parameter', 'interpolator'])
def test_native_compute_aliases_retain_the_original_evaluator(target, monkeypatch):
    from xrdroot.roofit.real import RooAbsReal

    parameter = RooRealVar('alpha', '', .2)
    node = FlexibleInterpVar('shape', '', RooArgList(parameter), 1., [.8], [1.2], [4])
    monkeypatch.setattr(RooRealVar if target == 'parameter' else FlexibleInterpVar,
                        'compute', RooAbsReal.compute)
    assert hfinterpolation.prepare(node) is None


def test_compute_descriptors_are_not_evaluated_during_preparation(monkeypatch):
    parameter = RooRealVar('alpha', '', .2)
    node = FlexibleInterpVar('shape', '', RooArgList(parameter), 1., [.8], [1.2], [4])
    def callback(self):
        raise AssertionError('Preparation ran a compute descriptor.')
    monkeypatch.setattr(FlexibleInterpVar, 'compute', property(callback))
    assert hfinterpolation.prepare(node) is None


@pytest.mark.parametrize('field', ['__module__', '__qualname__'])
def test_nonstandard_function_metadata_is_not_coerced_during_preparation(field, monkeypatch):
    parameter = RooRealVar('alpha', '', .2)
    node = FlexibleInterpVar('shape', '', RooArgList(parameter), 1., [.8], [1.2], [4])
    class CallbackString(str):
        def __eq__(self, other):
            raise AssertionError('Preparation compared custom function metadata.')
    value = object() if field == '__module__' else CallbackString('FlexibleInterpVar.compute')
    monkeypatch.setattr(FlexibleInterpVar.compute, field, value)
    assert hfinterpolation.prepare(node) is None


def test_derived_parameter_expressions_keep_the_existing_evaluator():
    from xrdroot.roofit.functions import RooProduct

    parameter = RooProduct('derived', '', RooArgList(RooConstVar('one', '', .2)))
    node = FlexibleInterpVar('shape', '', RooArgList(parameter), 1., [.8], [1.2], [4])
    assert hfinterpolation.prepare(node) is None
