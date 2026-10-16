"""Shared bytecode must never share a fit's parameters, histogram values or lifetime."""
import gc
import weakref

import numpy as np

from test_roofit_hf_fit import model
from xrdroot.roofit.fitting import hffit
from xrdroot.roofit.fitting.hfprogram import Program
from xrdroot.roofit.variables import RooRealVar


def test_identical_code_keeps_each_models_parameter_references():
    first=RooRealVar('first','',.3,-5.,5.)
    second=RooRealVar('second','',.7,-5.,5.)
    a,source_a=Program({},4).build(first)
    b,source_b=Program({},4).build(second)
    assert source_a==source_b and a.__code__ is b.__code__
    assert a.__globals__ is not b.__globals__
    assert a()==(.3,)*4 and b()==(.7,)*4
    first.setVal(-.4)
    assert a()==(-.4,)*4 and b()==(.7,)*4


def test_histogram_changes_make_distinct_code_and_keep_earlier_predictions():
    channel,nll,minimizer=model()
    before=hffit._prepare(minimizer,minimizer._migrad)
    assert before is not None
    expected=before.predict()
    channel.sig.data._weights*=1.5
    after=hffit._prepare(minimizer,minimizer._migrad)
    assert after is not None
    assert before.source!=after.source
    assert before.predict()==expected and before.predict()!=after.predict()
    np.testing.assert_array_equal(np.asarray(after.predict()),nll.pdf.compute(nll.columns))


def test_shared_bytecode_does_not_retain_models_after_the_last_fit():
    _,nll,minimizer=model()
    reference=weakref.ref(nll)
    with hffit.within(minimizer,minimizer._migrad):
        assert hffit._ACTIVE.get() is not None
    del nll,minimizer
    gc.collect()
    assert reference() is None
