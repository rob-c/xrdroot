"""An interval owns its chain; a declared adapter owns its public ROOT histogram."""

from __future__ import annotations

from statisticalproviders import install_provider
from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.variables import RooRealVar
from xrdroot.roostats.markov import MarkovChain
from xrdroot.roostats.mcmc import MCMCInterval


def test_the_interval_passes_itself_to_the_declared_posterior_adapter(monkeypatch):
    received = []
    expected = object()

    def posterior(interval):
        received.append(interval)
        return expected

    install_provider(monkeypatch, "roostats.posterior_hist", posterior)
    interval = MCMCInterval("interval")
    assert interval.GetPosteriorHist() is expected
    assert received == [interval]


def test_the_public_posterior_retains_float_storage_weights_and_detachment():
    from xrdroot.pyroot.core.hists import TH1F

    x = RooRealVar("x", "x", 0, 2)
    x.setBins(2)
    chain = MarkovChain()
    for value, weight in ((0.5, 2.0), (1.5, 3.0)):
        x.setVal(value)
        chain.Add(RooArgSet(x), 0.0, weight)
    interval = MCMCInterval("interval", RooArgSet(x), chain)
    histogram = interval.GetPosteriorHist()
    assert isinstance(histogram, TH1F)
    assert histogram.GetName() == "MCMCposterior_hist"
    assert [histogram.GetBinContent(i) for i in (1, 2)] == [2.0, 3.0]
    assert histogram.GetDirectory() is None
