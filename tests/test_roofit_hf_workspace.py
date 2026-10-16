"""The actual HistFactory graph exercises shared normalizations and per-bin statistical gammas."""
from __future__ import annotations

import numpy as np
import pytest
from tools.maintainability import analyze_source

from test_histfactory_models import Built, hf001
from test_roofit_hf_fit import same
from xrdroot.roofit.fitting import hffit
from xrdroot.roofit.fitting.minimizer import RooMinimizer


@pytest.fixture
def workspace(tmp_path):
    return Built(tmp_path,hf001).ws


def test_a_real_histfactory_model_keeps_all_likelihood_bits(workspace):
    pdf=workspace.pdf('simPdf')
    data=workspace.data('obsData')
    nll=pdf.createNLL(data)
    minimizer=RooMinimizer(nll)
    minimizer._optimizing=True
    minimizer.setPrintLevel(-1)
    plan=hffit._prepare(minimizer,minimizer._migrad)
    assert plan is not None
    poi=workspace.var('SigXsecOverSM')
    alpha=workspace.var('alpha_syst2')
    gamma=workspace.var('gamma_stat_channel1_bin_1')
    for value in (.2,.7,1.,1.8,2.9):
        poi.setVal(value)
        alpha.setVal(value-1.)
        gamma.setVal(.8+value/10.)
        expected=nll._general_sum()
        with hffit.within(minimizer,minimizer._migrad):
            assert hffit._ACTIVE.get() is not None
            same(nll._general_sum(),expected)
    limits={'ccn':10,'cognitive':15,'npath':200,'halstead_volume':500,'max_nesting':4}
    for metric in analyze_source(plan.source):
        assert all(getattr(metric,key)<=limit for key,limit in limits.items()),metric


def test_a_fit_preserves_minuit_steps_and_results(workspace,monkeypatch):
    pytest.importorskip('iminuit')
    nll=workspace.pdf('simPdf').createNLL(workspace.data('obsData'))
    parameters=list(nll.getParameters())
    saved=[(p.getVal(),p.getError()) for p in parameters]
    prepared=hffit._prepare
    results=[]
    for enabled in (False,True):
        for parameter,(value,error) in zip(parameters,saved,strict=False):
            parameter.setVal(value)
            parameter.setError(error)
        minimizer=RooMinimizer(nll)
        minimizer._optimizing=True
        minimizer.setPrintLevel(-1)
        with monkeypatch.context() as patch:
            patch.setattr(hffit,'_prepare',prepared if enabled else lambda *args:None)
            assert minimizer.migrad()==0
        results.append(([p.getVal() for p in parameters],nll.getVal(),minimizer.evaluations))
    np.testing.assert_array_equal(np.asarray(results[0][0]).view(np.uint64),
                                  np.asarray(results[1][0]).view(np.uint64))
    assert results[0][1:] == results[1][1:]


@pytest.mark.parametrize('changed',['category','states','channels','parameters'])
def test_live_simultaneous_model_edits_are_checked_at_the_next_fit(workspace,changed):
    pdf=workspace.pdf('simPdf')
    nll=pdf.createNLL(workspace.data('obsData'))
    minimizer=RooMinimizer(nll)
    minimizer._optimizing=True
    minimizer.setPrintLevel(-1)
    if changed=='category':
        nll.columns[pdf.index.GetName()][0]=999
    elif changed=='states':
        pdf.index._states={'channel1':object()}
    elif changed=='channels':
        pdf.channels['second']=pdf.channels['channel1']
    else:
        workspace.var('alpha_syst2').compute=lambda ctx:0.
    assert hffit._prepare(minimizer,minimizer._migrad) is None
