"""``MetropolisHastings``, ``MarkovChain`` and the proposal functions, number for number ROOT's.

Each chain is RooStats' from the same seed of RooFit's generator: the same
proposals drawn, the same steps taken, the same points kept with the same
weights and negative log-likelihoods, for each type and sign of function and
for uniform, sequential and density proposals - the references printed by
ROOT 6.40 for the same calls. A function that is nowhere positive shows what
RooStats does without a starting point: a single point, its likelihood's
logarithm infinite, as C's arithmetic has it.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roostats.markov import (
    MarkovChain,
    MetropolisHastings,
    ProposalFunction,
    SequentialProposal,
    UniformProposal,
)
from xrdroot.roostats.proposals import PdfProposal

MH = MetropolisHastings
GAUSS = "exp(-5*(@0*@0+@1*@1))"


def plane() -> tuple[Any, Any]:
    return ROOT.RooRealVar("a", "a", 0.1, -1, 1), ROOT.RooRealVar("b", "b", 0.2, -1, 1)


def density(a: Any, b: Any, cache: int = 20) -> PdfProposal:
    """A Gaussian of width 0.3 about the chain's point, a cache of ``cache`` at a time."""
    ma, mb = ROOT.RooRealVar("ma", "ma", 0, -1, 1), ROOT.RooRealVar("mb", "mb", 0, -1, 1)
    width = ROOT.RooRealVar("sa", "sa", 0.3)
    gauss = ROOT.RooProdPdf("gp", "gp", [ROOT.RooGaussian("g1", "g1", a, ma, width),
                                         ROOT.RooGaussian("g2", "g2", b, mb, width)])  # fmt: skip
    proposal = PdfProposal(gauss)
    proposal.AddMapping(ma, a)
    proposal.AddMapping(mb, b)
    proposal.SetCacheSize(cache)
    return proposal


def chain_of(formula: str, kind: int, sign: int, proposal: Any = None, iters: int = 200,
             chained: bool = False) -> MarkovChain:  # fmt: skip
    """RooStats' chain of ``formula`` in ``a`` and ``b`` from seed 4357."""
    ROOT.RooRandom.randomGenerator().SetSeed(4357)
    a, b = plane()
    made = density(a, b) if proposal == "pdf" else proposal or UniformProposal()
    mh = MH(ROOT.RooFormulaVar("f", formula, [a, b]), ROOT.RooArgSet(a, b), made, iters)
    mh.SetType(kind)
    mh.SetSign(sign)
    if chained:
        mh.SetChainParameters(ROOT.RooArgSet(a))
    return mh.ConstructChain()


def point(chain: MarkovChain, i: int) -> tuple[list[float], float, float]:
    """The chain's ``i``-th point, its weight and its negative log-likelihood."""
    found = chain.Get(i)
    return [found.getRealValue(n) for n in ("a", "b") if found.find(n)], chain.Weight(), chain.NLL()


def test_a_regular_likelihood_walks_as_roots_does(capfd) -> None:
    """A positive likelihood with uniform proposals: ROOT's 48 points out of 200 steps."""
    chain = chain_of(GAUSS, MH.kRegular, MH.kPositive)
    assert chain.Size() == 48
    assert point(chain, 0) == ([-0.5366869145072997, -0.03005277132615447], 3.0,
                               pytest.approx(1.4446800663387385, rel=1e-15))  # fmt: skip
    assert point(chain, 1) == ([-0.6628551562316716, -0.04894053936004639], 1.0,
                               pytest.approx(2.20886067267883, rel=1e-15))  # fmt: skip
    assert point(chain, 47) == ([0.062461471650749445, -0.046633278485387564], 5.0,
                                pytest.approx(0.030380490515365382, rel=1e-14))  # fmt: skip
    out = capfd.readouterr().out
    assert "Proposal acceptance rate: 24%" in out
    assert "Number of steps in chain: 48" in out


def test_a_negative_likelihood_takes_the_same_walk() -> None:
    """The likelihood's negative, said to be negative, is the same chain."""
    positive = chain_of(GAUSS, MH.kRegular, MH.kPositive)
    negative = chain_of(f"-{GAUSS}", MH.kRegular, MH.kNegative)
    assert [point(negative, i) for i in range(48)] == [point(positive, i) for i in range(48)]


def test_a_log_likelihood_with_sequential_steps_keeps_only_the_chain_parameters() -> None:
    """A log-likelihood, one parameter moved at a time, and only ``a`` kept in the chain."""
    chain = chain_of("-5*(@0*@0+@1*@1)", MH.kLog, MH.kPositive, SequentialProposal(0.5),
                     chained=True)  # fmt: skip
    assert chain.Size() == 101
    assert [a.GetName() for a in chain.GetParameters()] == ["a"]
    assert point(chain, 0) == ([-0.5366869145072997], 1.0, 1.7702089796123877)
    assert point(chain, 1) == ([-0.5366869145072997], 2.0, 1.6939043868049974)
    assert point(chain, 100) == ([-0.02976127502292769], 1.0, 0.9930769101728493)


def test_density_proposals_weigh_the_steps_by_their_densities_both_ways() -> None:
    """A Gaussian proposal about the point is not symmetric: each step's ratio is corrected
    by the densities there and back - for a negative log-likelihood and a likelihood alike."""
    nll = chain_of("5*(@0*@0+@1*@1)", MH.kLog, MH.kNegative, "pdf")
    like = chain_of(GAUSS, MH.kRegular, MH.kPositive, "pdf")
    assert nll.Size() == like.Size() == 114
    first = [-0.28940780900342145, -0.047067968969863955]
    assert point(nll, 0) == (first, 2.0, pytest.approx(0.4298613680755447, rel=1e-15))
    assert point(like, 0) == (first, 2.0, pytest.approx(0.4298613680755447, rel=1e-15))
    last = [0.08256888815024724, 0.13051013423212535]
    assert point(nll, 113) == (last, 2.0, pytest.approx(0.11925258213827707, rel=1e-14))
    assert point(like, 113)[:2] == (last, 2.0)


def test_a_function_nowhere_positive_leaves_one_point_of_infinite_nll(capfd) -> None:
    """With no starting point in a thousand tries RooStats says so and walks anyway; every
    step's ratio is ``0/0``, never taken, so the chain is its one point, of weight 200."""
    chain = chain_of("0*@0", MH.kRegular, MH.kPositive)
    assert chain.Size() == 1
    assert point(chain, 0) == ([-0.34455810207873583, 0.5702735935337842], 200.0, float("inf"))
    out = capfd.readouterr().out
    assert "Problem finding a good starting point in MetropolisHastings::ConstructChain()" in out
    assert "Proposal acceptance rate: 0.5%" in out


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_log_likelihood_that_fails_is_never_stepped_to() -> None:
    """Where the log-likelihood is not a number the start is drawn again, and a candidate
    there is refused - so every point of the chain is where it is defined."""
    chain = chain_of("-log(@0)-log(@1)", MH.kLog, MH.kNegative, iters=100)
    assert all(point(chain, i)[0][0] > 0 and point(chain, i)[0][1] > 0
               for i in range(chain.Size()))  # fmt: skip
    assert sum(chain.weights()) == 100


def test_a_function_of_another_type_still_walks() -> None:
    """A type RooStats does not know is neither log nor regular: its ratio is the quotient,
    and every step, uphill or not, is left to the generator."""
    chain = chain_of(GAUSS, 3, MH.kPositive, iters=50)
    assert sum(chain.weights()) == 50


def test_a_chain_of_no_steps_is_empty() -> None:
    """No iterations, no points."""
    assert chain_of(GAUSS, MH.kRegular, MH.kPositive, iters=0).Size() == 0


def test_metropolis_hastings_refuses_to_start_without_its_members(capfd) -> None:
    """No parameters, proposal or function - or no type and sign - and there is no chain."""
    a, b = plane()
    mh = MH()
    assert mh.ConstructChain() is None
    assert ("Critical members uninitialized: parameters, proposal  function, or (log) "
            "likelihood function") in capfd.readouterr().out  # fmt: skip
    mh.SetFunction(ROOT.RooFormulaVar("f", GAUSS, [a, b]))
    mh.SetParameters(ROOT.RooArgSet(a, b))
    mh.SetProposalFunction(UniformProposal())
    mh.SetNumIters(10)
    assert mh.ConstructChain() is None
    assert ("Please set type and sign of your function using MetropolisHastings::SetType() and "
            "MetropolisHastings::SetSign()") in capfd.readouterr().out  # fmt: skip
    mh.SetType(MH.kRegular)
    mh.SetSign(MH.kPositive)
    assert sum(mh.ConstructChain().weights()) == 10


def test_a_markov_chain_keeps_its_points_weights_and_likelihoods() -> None:
    """``MarkovChain``: named or not, its parameters given or taken from the first point; each
    point loaded into the chain's own copies, the last loaded the one asked of by default."""
    a, b = plane()
    assert (MarkovChain().GetName(), MarkovChain().GetTitle()) == ("_markov_chain", "Markov Chain")
    named = MarkovChain("c", "chain", ROOT.RooArgSet(a))
    assert (named.GetName(), named.GetTitle(), named.Size()) == ("c", "chain", 0)
    chain = MarkovChain()
    for x, weight in ((0.25, 2.0), (0.5, 1.0), (-0.5, 3.0)):
        a.setVal(x)
        chain.Add(ROOT.RooArgSet(a, b), -x, weight)
    assert chain.Size() == 3
    loaded = chain.Get(1)
    assert loaded.getRealValue("a") == 0.5 and loaded.find("a") is not a
    assert (chain.Get().getRealValue("a"), chain.Weight(), chain.NLL()) == (0.5, 1.0, -0.5)
    assert (chain.Weight(2), chain.NLL(0)) == (3.0, -0.25)
    assert (chain.values("a", 1), chain.values("b"), chain.weights(1)) == (
        [0.5, -0.5], [0.2, 0.2, 0.2], [1.0, 3.0])


def test_the_uniform_and_sequential_proposals() -> None:
    """A uniform proposal is anywhere in the ranges, of density one over their volume; a
    sequential one moves one parameter by a Gaussian, wrapped round its range."""
    a, b = plane()
    base = ProposalFunction()
    with pytest.raises(NotImplementedError):
        base.Propose(ROOT.RooArgSet(a), ROOT.RooArgSet(a))
    assert (base.IsSymmetric(None, None), base.GetProposalDensity(None, None)) == (True, 1.0)
    uniform = UniformProposal()
    assert uniform.GetProposalDensity(None, ROOT.RooArgSet(a, b)) == 0.25
    ROOT.RooRandom.randomGenerator().SetSeed(1)
    moved, here = ROOT.RooArgSet(a, b).snapshot(), ROOT.RooArgSet(a, b)
    for _ in range(20):
        SequentialProposal(0.05).Propose(moved, here)
        values = [moved.getRealValue("a"), moved.getRealValue("b")]
        assert all(-1 <= v <= 1 for v in values)
        assert values[0] == a.getVal() or values[1] == b.getVal()


def test_a_density_proposal_without_mappings_draws_its_cache_in_turn(capfd) -> None:
    """Without mappings the density stays put: its candidates are its cache, in turn, a new
    cache drawn when one is used up; a cache size that is not positive is refused."""
    a, _b = plane()
    gauss = ROOT.RooGaussian("g", "g", a, ROOT.RooFit.RooConst(0.0), ROOT.RooFit.RooConst(0.3))
    proposal = PdfProposal()
    proposal.SetPdf(gauss)
    proposal.SetOwnsPdf(True)
    assert proposal.GetPdf() is gauss
    proposal.SetCacheSize(0)
    assert "Warning: Requested non-positive cache size: 0. Cache size unchanged." in (
        capfd.readouterr().out)
    proposal.SetCacheSize(2)
    ROOT.RooRandom.randomGenerator().SetSeed(7)
    moved, here = ROOT.RooArgSet(a).snapshot(), ROOT.RooArgSet(a)
    drawn = []
    for _ in range(5):
        proposal.Propose(moved, here)
        drawn.append(moved.getRealValue("a"))
    assert len(set(drawn)) == 5 and all(-1 <= v <= 1 for v in drawn)
    assert not proposal.IsSymmetric(moved, here)
    a.setVal(0.0)
    peak = 1 / (0.3 * math.sqrt(2 * math.pi)) / math.erf(1 / (0.3 * math.sqrt(2)))
    assert proposal.GetProposalDensity(ROOT.RooArgSet(a), here) == pytest.approx(peak, rel=1e-12)


def test_points_of_other_variables_are_not_the_same_point() -> None:
    """``Equals``: two sets are the same point only if they hold the same variables."""
    from xrdroot.roostats.proposals import _equal

    a, b = plane()
    assert _equal(ROOT.RooArgSet(a), ROOT.RooArgSet(a).snapshot())
    assert not _equal(ROOT.RooArgSet(a), ROOT.RooArgSet(b))
    assert not _equal(ROOT.RooArgSet(a), ROOT.RooArgSet(a, b))


def test_the_proposal_helper_takes_a_density_and_refuses_what_it_cannot(capfd) -> None:
    """A density given is the helper's Gaussian; a divisor not positive and a cache size not
    positive are ignored, and clues - a keys density - are refused."""
    from xrdroot.errors import UnsupportedFeatureError
    from xrdroot.roostats.proposals import ProposalHelper

    a, _b = plane()
    helper = ProposalHelper()
    helper.SetVariables(ROOT.RooArgList(a))
    gauss = ROOT.RooGaussian("g", "g", a, ROOT.RooFit.RooConst(0.0), ROOT.RooFit.RooConst(0.3))
    helper.SetPdf(gauss)
    helper.SetWidthRangeDivisor(0.0)
    helper.SetCacheSize(-1)
    assert "Requested non-positive cache size: -1. Cache size unchanged." in capfd.readouterr().out
    made = helper.GetProposalFunction().GetPdf()
    assert (made.GetName(), [p.GetName() for p in made.pdfList()]) == ("proposalFunction", ["g"])
    with pytest.raises(UnsupportedFeatureError, match="RooNDKeysPdf"):
        helper.SetClues(None)
