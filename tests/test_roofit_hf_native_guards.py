"""Unknown storage and callbacks must fall back before a supposedly constant value is read."""
from __future__ import annotations

import math

import numpy as np
import pytest

from test_roofit_hf_fit import model
from xrdroot.roofit.fitting import hffit
from xrdroot.roofit.variables import RooRealVar


class Foreign:
    def __float__(self):
        raise AssertionError('A planning guard ran a numeric callback.')

    def __get__(self, instance, owner):
        raise AssertionError('A planning guard ran a descriptor callback.')


@pytest.mark.parametrize('value',[Foreign(),np.float32(1.),np.asarray(1.),10**400])
def test_numeric_callbacks_and_non_native_scalars_are_rejected(value):
    _,nll,minimizer=model()
    nll._global_values['foreign']=value
    assert hffit._prepare(minimizer,minimizer._migrad) is None


def test_foreign_class_descriptors_are_rejected_without_access(monkeypatch):
    _,_,minimizer=model()
    monkeypatch.setattr(RooRealVar,'foreign',Foreign(),raising=False)
    assert hffit._prepare(minimizer,minimizer._migrad) is None


@pytest.mark.parametrize('member',[lambda self,ctx:1.,classmethod(lambda cls:1.),
                                  staticmethod(lambda:1.),property(lambda self:1.)])
def test_changed_class_callbacks_are_checked_again_for_every_fit(member,monkeypatch):
    _,_,minimizer=model()
    assert hffit._prepare(minimizer,minimizer._migrad) is not None
    with monkeypatch.context() as patch:
        patch.setattr(RooRealVar,'foreign',member,raising=False)
        assert hffit._prepare(minimizer,minimizer._migrad) is None
    assert hffit._prepare(minimizer,minimizer._migrad) is not None


def test_a_class_from_a_foreign_module_is_not_a_native_callback(monkeypatch):
    _,_,minimizer=model()
    monkeypatch.setattr(RooRealVar,'__module__','foreign')
    assert hffit._prepare(minimizer,minimizer._migrad) is None


@pytest.mark.parametrize('target',['minimizer','nll','variable','data','collection','binning'])
def test_instance_method_overrides_keep_the_general_evaluator(target):
    channel,nll,minimizer=model()
    choices={'minimizer':(minimizer,'_evaluate'),'nll':(nll,'evaluate_nll'),
             'variable':(channel.mu,'getVal'),'data':(channel.sig.data,'weights'),
             'collection':(channel.model.funcs,'__iter__'),
             'binning':(channel.mu.getBinning(),'lowBound')}
    node,name=choices[target]
    setattr(node,name,lambda *args:1.)
    assert hffit._prepare(minimizer,minimizer._migrad) is None


@pytest.mark.parametrize('field,value',[('_val',Foreign()),('_error',Foreign()),
    ('_asym',[]),('_asym',(1.,)),('_asym',(1.,Foreign())),('_shared',{'range':Foreign()}),
    ('_binning',Foreign())])
def test_variable_storage_is_checked_before_numeric_methods_run(field,value):
    channel,_,minimizer=model()
    setattr(channel.mu,field,value)
    assert hffit._prepare(minimizer,minimizer._migrad) is None


@pytest.mark.parametrize('field,value',[('_low',math.nan),('_high',math.nan),('_bins',0),
                                       ('_uniform',1),('_edges',[Foreign()])])
def test_parameter_binnings_require_fixed_native_numeric_storage(field,value):
    channel,_,minimizer=model()
    setattr(channel.mu.getBinning(),field,value)
    assert hffit._prepare(minimizer,minimizer._migrad) is None


@pytest.mark.parametrize('field,value',[('_proxies',()),('_proxies',[Foreign()]),
    ('_name',Foreign()),('_attributes',[]),('_attributes',{1}),('_strings',{'key':Foreign()}),
    ('_strings',{1:'value'})])
def test_graph_metadata_cannot_hide_python_callbacks(field,value):
    channel,_,minimizer=model()
    setattr(channel.model,field,value)
    assert hffit._prepare(minimizer,minimizer._migrad) is None


def test_a_proxy_cannot_supply_a_foreign_collection():
    channel,_,minimizer=model()
    channel.model._proxies[0].target=Foreign()
    assert hffit._prepare(minimizer,minimizer._migrad) is None


def test_a_non_boolean_proxy_mode_is_not_evaluated():
    channel,_,minimizer=model()
    channel.model._proxies[0].many=Foreign()
    assert hffit._prepare(minimizer,minimizer._migrad) is None


@pytest.mark.parametrize('field,value',[('_weights',[1.,2.]),('_weights',np.zeros((2,2))),
                                       ('_columns',{'x':np.asarray([Foreign()],dtype=object)})])
def test_histogram_data_requires_native_array_storage(field,value):
    channel,_,minimizer=model()
    setattr(channel.sig.data,field,value)
    assert hffit._prepare(minimizer,minimizer._migrad) is None


def test_collection_storage_cannot_use_a_foreign_iterator():
    channel,_,minimizer=model()
    channel.model.funcs._list=Foreign()
    assert hffit._prepare(minimizer,minimizer._migrad) is None


def test_planning_large_native_graphs_leaves_the_array_evaluator_available():
    channel,_,minimizer=model()
    channel.model.coefs._list=[RooRealVar(f'p{i}','',1.) for i in range(150)]
    assert hffit._prepare(minimizer,minimizer._migrad) is None
