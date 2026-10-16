"""Small native expressions preserve binary64 arithmetic and reject broader numeric domains."""
from __future__ import annotations

import math

import numpy as np
import pytest

from test_roofit_hf_fit import model, same
from xrdroot.roofit.collections import RooArgList
from xrdroot.roofit.fitting import hffit
from xrdroot.roofit.fitting.hfprogram import NotCompiled, Program
from xrdroot.roofit.functions import RooFormulaVar, RooProduct
from xrdroot.roofit.pdfs.histfactory import FlexibleInterpVar, ParamHistFunc, RooBinWidthFunction
from xrdroot.roofit.variables import RooConstVar, RooRealVar


def evaluate(node, columns=None, count=4):
    columns = {} if columns is None else columns
    predict, source = Program(columns,count).build(node)
    expected = np.broadcast_to(node.compute(columns),(count,))
    np.testing.assert_array_equal(np.asarray(predict()).view(np.uint64),
                                  expected.copy().view(np.uint64))
    return predict, source


def test_shared_variables_are_read_live_and_constants_ignore_name_collisions():
    value=RooRealVar('value','',.3,-5.,5.)
    constant=RooConstVar('value','',.7)
    node=RooProduct('product','',RooArgList(value,constant,value))
    predict,_=evaluate(node)
    for number in (.1,.9,-.3):
        value.setVal(number)
        same(predict()[0],node.getVal())
    evaluate(constant,{'value':np.arange(4,dtype=float)})
    evaluate(value,{'value':np.arange(4,dtype=float)})


def test_histfactory_statistical_parameters_and_widths_match_the_array_evaluator():
    channel, nll, _=model()
    gammas=RooArgList(*[RooRealVar(f'g{i}','',1.+i/10.,.1,3.) for i in range(4)])
    function=ParamHistFunc('stat','',RooArgList(channel.x),gammas)
    predict,_=evaluate(function,nll.columns)
    gammas[2].setVal(.37)
    same(predict()[2],.37)
    width=RooBinWidthFunction('width','',channel.sig,True)
    from xrdroot.roofit.pdfs.histfactory import binned_likelihood
    with binned_likelihood():
        evaluate(width,nll.columns)


@pytest.mark.parametrize('code',[0,1,4,5])
def test_native_interpolation_reuses_its_original_kernel(code):
    parameter=RooRealVar('alpha','',.2,-5.,5.)
    function=FlexibleInterpVar('interp','',[parameter],1.,[.95],[1.05])
    function.setAllInterpCodes(code)
    predict,_=evaluate(function)
    for value in (-4.,-.3,0.,.7,3.):
        parameter.setVal(value)
        same(predict()[0],float(function.compute({})))


@pytest.mark.parametrize('field,value',[('_boundary',0.),('_boundary',5.),('_nominal',0.),
    ('_low',[0.]),('_high',[101.]),('_codes',[2])])
def test_interpolation_outside_the_bounded_domain_uses_the_general_evaluator(field,value):
    parameter=RooRealVar('alpha','',.2,-5.,5.)
    function=FlexibleInterpVar('interp','',[parameter],1.,[.95],[1.05])
    setattr(function,field,value)
    with pytest.raises(NotCompiled):
        Program({},4).build(function)


def test_interpolation_parameters_must_be_scalar_and_bounded():
    parameter=RooRealVar('alpha','',.2,-9.,9.)
    function=FlexibleInterpVar('interp','',[parameter],1.,[.95],[1.05])
    with pytest.raises(NotCompiled):
        Program({},4).build(function)
    with pytest.raises(NotCompiled):
        Program({'alpha':np.zeros(4)},4).build(function)
    function=FlexibleInterpVar('many','',[],1.,[],[])
    function.params._list=[parameter]*5
    with pytest.raises(NotCompiled):
        Program({},4).build(function)


@pytest.mark.parametrize('value',[math.inf,math.nan,1e201])
def test_nonfinite_or_overflowing_intermediate_domains_are_not_compiled(value):
    node=RooConstVar('bad','',value)
    with pytest.raises(NotCompiled):
        Program({},4).build(node)


def test_cycles_and_unknown_expression_nodes_use_the_general_evaluator():
    node=RooProduct('cycle','',[])
    node.terms._list.append(node)
    with pytest.raises(NotCompiled):
        Program({},4).build(node)
    formula=RooFormulaVar('formula','','1',[])
    with pytest.raises(NotCompiled):
        Program({},4).build(formula)


@pytest.mark.parametrize('change',[lambda ch:setattr(ch.sig,'order',1),
    lambda ch:setattr(ch.sig,'density',True),
    lambda ch:setattr(ch.sig,'_hist_vars',[]),
    lambda ch:ch.sig.observables._list.clear(),
    lambda ch:ch.sig.data.get().find('x').setRange(-math.inf,4.)])
def test_histogram_interpolation_density_and_unbounded_axes_are_not_cached(change):
    channel, _, minimizer=model()
    change(channel)
    assert hffit._prepare(minimizer,minimizer._migrad) is None


def test_histogram_observables_must_be_plain_columns():
    channel,_,_=model()
    program=Program({},4)
    with pytest.raises(NotCompiled):
        program.build(channel.sig)
    with pytest.raises(NotCompiled):
        Program({'x':np.zeros(4,dtype=np.float32)},4).build(channel.sig)


def test_an_observed_variable_with_float32_storage_keeps_array_promotion():
    variable=RooRealVar('x','',.3,-5.,5.)
    with pytest.raises(NotCompiled):
        Program({'x':np.zeros(4,dtype=np.float32)},4).build(variable)


@pytest.mark.parametrize('overflow', ['warn', 'raise'])
def test_overflowing_histogram_widths_keep_the_original_warning_path(overflow):
    import warnings

    channel, _, minimizer = model()
    channel.sig.data.get().find('x').setRange(-1e308, 1e308)
    with np.errstate(over=overflow), warnings.catch_warnings(record=True) as found:
        warnings.simplefilter('always')
        assert hffit._prepare(minimizer, minimizer._migrad) is None
    assert not found
