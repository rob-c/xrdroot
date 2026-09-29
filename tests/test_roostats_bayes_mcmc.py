"""``MCMCCalculator`` and ``MCMCInterval``: Bayesian intervals from a Markov chain, ROOT's own.

The model is IntervalExamples' Gaussian - its width free too - fitted to
ten events drawn from seed 3001, and each chain is 300 steps of RooStats'
Metropolis-Hastings: the uniform proposal, the sequential one, and the
``ProposalHelper``'s Gaussian with or without a covariance matrix and a
uniform part. Every chain, interval, level and cutoff is the one ROOT 6.40
found for the same calls. The intervals of chains made by hand pin down
how the shortest interval gathers bins - strictly or not - and how the
tail-fraction interval walks in from each end.
"""

from __future__ import annotations

from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roostats.markov import SequentialProposal
from xrdroot.roostats.mcmc import MCMCCalculator
from xrdroot.roostats.modelconfig import ModelConfig
from xrdroot.roostats.proposals import ProposalHelper

FIRST = (-0.5746123255230486, 0.8710162937641144, 2.0, 15.61690352544702)


def gaussian(pois: str = "mu") -> tuple[Any, Any, Any]:
    """The workspace, its model configuration and ten events drawn from seed 3001."""
    ROOT.RooRandom.randomGenerator().SetSeed(3001)
    w = ROOT.RooWorkspace("w")
    w.factory("Gaussian::normal(x[-10,10],mu[-1,1],sigma[1,0.5,2])")
    w.defineSet("poi", pois)
    w.defineSet("obs", "x")
    config = ModelConfig("cfg", w)
    config.SetPdf(w["normal"])
    config.SetParametersOfInterest(w.set("poi"))
    config.SetObservables(w.set("obs"))
    return w, config, w["normal"].generate(w.set("obs"), 10)


def calculator(pois: str = "mu") -> tuple[Any, MCMCCalculator]:
    """300 steps, the first 20 burn-in, 20 bins, at 90%."""
    w, config, data = gaussian(pois)
    mc = MCMCCalculator(data, config)
    mc.SetConfidenceLevel(0.9)
    mc.SetNumBins(20)
    mc.SetNumBurnInSteps(20)
    mc.SetNumIters(300)
    return w, mc


def entry(chain: Any, i: int) -> tuple[float, float, float, float]:
    found = chain.Get(i)
    return found.getRealValue("mu"), found.getRealValue("sigma"), chain.Weight(), chain.NLL()


def summary(interval: Any, w: Any) -> tuple[float, float, float]:
    mu = w["mu"]
    return interval.LowerLimit(mu), interval.UpperLimit(mu), interval.GetActualConfidenceLevel()


def test_the_shortest_interval_of_a_uniform_proposals_chain_is_roots(capfd) -> None:
    """The default: uniform proposals, the shortest interval of the chain's histogram."""
    w, mc = calculator()
    interval = mc.GetInterval()
    assert summary(interval, w) == (-0.75, 0.4500000000000002, 0.9157088122605364)
    assert interval.GetHistCutoff() == 9.0
    chain = interval.GetChain()
    assert chain.Size() == 147
    assert entry(chain, 0) == FIRST
    assert entry(chain, 146) == (-0.5148513833992183, 1.2027744750957936, 1.0,
                                 14.938476231312256)  # fmt: skip
    assert (interval.GetName(), interval.GetNumBurnInSteps()) == ("MCMCInterval_", 20)
    assert interval.ConfidenceLevel() == pytest.approx(0.9)
    out = capfd.readouterr().out
    said = ["Metropolis-Hastings progress: ....", "Proposal acceptance rate: 49%"]
    assert [line for line in said if line not in out] == []


def test_a_central_interval_walks_in_from_both_tails() -> None:
    """A left-side fraction of one half: the tail-fraction interval of the same chain."""
    w, mc = calculator()
    mc.SetLeftSideTailFraction(0.5)
    assert summary(mc.GetInterval(), w) == (-0.791468839161098, 0.47402061289176345,
                                            0.896551724137931)  # fmt: skip


def test_a_flat_prior_leaves_the_chain_as_it_is() -> None:
    """The likelihood times a flat prior: the same chain and interval."""
    w, mc = calculator()
    w.factory("Uniform::prior(mu)")
    mc.SetPriorPdf(w["prior"])
    interval = mc.GetInterval()
    assert summary(interval, w) == (-0.75, 0.4500000000000002, 0.9157088122605364)
    assert entry(interval.GetChain(), 0) == FIRST


def test_sequential_proposals_move_one_parameter_at_a_time() -> None:
    """``SequentialProposal(0.1)``: ROOT's chain of 178 points."""
    w, mc = calculator()
    mc.SetProposalFunction(SequentialProposal(0.1))
    interval = mc.GetInterval()
    assert summary(interval, w) == (-0.6499999999999999, 0.9500000000000002, 0.91015625)
    assert interval.GetHistCutoff() == 8.0
    chain = interval.GetChain()
    assert (chain.Size(), entry(chain, 1)) == (
        178, (-0.04387512989342213, 1.6231277854271298, 2.0, 15.818736309808795))


def helped(w: Any, cache: int = 50) -> ProposalHelper:
    helper = ProposalHelper()
    helper.SetVariables(ROOT.RooArgList(w["mu"], w["sigma"]))
    helper.SetUpdateProposalParameters(True)
    helper.SetCacheSize(cache)
    return helper


def _helped_chain(helper: ProposalHelper, w: Any, config: Any, data: Any) -> tuple[Any, ...]:
    mc = MCMCCalculator(data, config)
    mc.SetConfidenceLevel(0.9)
    mc.SetNumBins(20)
    mc.SetNumBurnInSteps(20)
    mc.SetNumIters(300)
    mc.SetProposalFunction(helper.GetProposalFunction())
    interval = mc.GetInterval()
    mu = w["mu"]
    return (interval.LowerLimit(mu), interval.UpperLimit(mu), interval.GetActualConfidenceLevel(),
            interval.GetChain().Size())


@pytest.mark.parametrize(("settings", "expected"), [
    ({}, (-0.95, 0.6500000000000001, 1.0, 146)),
    ({"SetUniformFraction": 0.2, "SetWidthRangeDivisor": 3.0},
     (-0.95, 0.8500000000000001, 1.0, 123)),
    ({"SetCovMatrix": [[0.04, 0.005], [0.005, 0.02]]}, (-0.85, 0.55, 1.0, 222)),
])  # fmt: skip
def test_the_proposal_helpers_gaussian_walks_as_roots(settings: Any, expected: Any) -> None:
    """ROOT 6.40's chains of the helper's Gaussian - with its widths a third of the ranges and
    a fifth uniform, or a covariance matrix given - their intervals and lengths."""
    w, config, data = gaussian("mu,sigma")
    helper = helped(w)
    for name, value in settings.items():
        getattr(helper, name)(value)
    assert _helped_chain(helper, w, config, data) == expected


def test_the_helper_ignores_what_it_cannot_use_and_refuses_clues(capsys: Any) -> None:
    from xrdroot.errors import UnsupportedFeatureError

    w, _, _ = gaussian("mu,sigma")
    helper = helped(w)
    helper.SetWidthRangeDivisor(0.0)
    helper.SetCacheSize(0)
    assert "Requested non-positive cache size: 0. Cache size unchanged." in (
        capsys.readouterr().out)  # fmt: skip
    helper.SetPdf(w["normal"])
    assert helper.GetProposalFunction().GetPdf().GetName() == "proposalFunction"
    fixed = ProposalHelper()  # its Gaussian's means left where they start
    fixed.SetVariables(ROOT.RooArgList(w["mu"], w["sigma"]))
    assert fixed.GetProposalFunction().GetPdf().GetName() == "proposalFunction"
    with pytest.raises(UnsupportedFeatureError, match="RooNDKeysPdf"):
        helper.SetClues(None)
