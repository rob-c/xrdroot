"""A chain's points as a dataset, and a dataset's numbers as a tree: RooStats' two bridges.

``rs101_limitexample`` scans a Markov chain and draws it with ``TTree::Draw``, so RooStats
hands the chain over as a ``RooDataSet`` (``MarkovChain::GetAsDataSet``, whose columns are the
parameters, ``nll_MarkovChain_local_`` and the steps as the weight) and then as a ``TTree``
(``RooStats::GetAsTTree``, ``"%s/D"`` a branch, in ``std::map``'s alphabetical order, with an
``_err`` or ``_errlo``/``_errhi`` branch beside a variable that carries an error).
"""

from __future__ import annotations

import pytest

from xrdroot.pyroot.roostats.asdata import INVALID, GetAsTTree
from xrdroot.roofit.data.dataset import RooDataSet
from xrdroot.roofit.variables import RooRealVar
from xrdroot.roostats.markov import DATASET, NLL, MarkovChain

#: Three points: the parameters' values, the likelihood, and the steps it stayed.
POINTS = (((0.5, 2.0), 10.0, 1.0), ((1.5, 3.0), 12.0, 2.0), ((2.5, 4.0), 14.0, 3.0))


def a_chain() -> MarkovChain:
    """Three points of two parameters, each with its likelihood and its weight."""
    chain = MarkovChain("chain", "chain", [RooRealVar("mu", "", 0.0), RooRealVar("s", "", 1.0)])
    for row, value, weight in POINTS:
        entry = [RooRealVar("mu", "", row[0]), RooRealVar("s", "", row[1])]
        chain.Add(entry, value, weight)
    return chain


def test_a_chain_is_handed_over_as_a_dataset_of_its_points() -> None:
    """The columns are the parameters and the likelihood, the steps are the weight."""
    data = a_chain().GetAsDataSet()
    assert [one.GetName() for one in data.get()] == ["mu", "s", NLL]
    assert data.numEntries() == 3
    assert data.get(1).getRealValue("mu") == 1.5
    assert data.get(1).getRealValue("s") == 3.0
    assert data.get(1).getRealValue(NLL) == 12.0
    assert list(data.weights()) == [1.0, 2.0, 3.0]
    assert data.GetName() == DATASET


def test_the_variables_asked_for_are_the_ones_the_dataset_has() -> None:
    """``GetAsDataSet(&vars)`` reduces to those columns - the likelihood still comes along."""
    data = a_chain().GetAsDataSet([RooRealVar("s", "", 0.0)])
    assert [one.GetName() for one in data.get()] == ["s", NLL]
    assert [data.get(i).getRealValue("s") for i in range(3)] == [2.0, 3.0, 4.0]


def test_an_interval_hands_its_chain_over_the_same_way() -> None:
    from xrdroot.roostats.mcmc import MCMCInterval

    interval = MCMCInterval("mc", [RooRealVar("mu", "", 0.0)], a_chain())
    data = interval.GetChainAsDataSet()
    assert data.numEntries() == 3
    assert [one.GetName() for one in data.get()] == ["mu", "s", NLL]


def test_a_dataset_is_drawn_from_a_tree_with_a_branch_a_column() -> None:
    """One ``"%s/D"`` branch a column, and ``Draw`` reads them back - which is what ``rs101``
    does with its chain: ``chain.Draw("s:ratioSigEff:ratioBkgEff", "nll_...", "box")``."""
    tree = GetAsTTree("chain", "chain", a_chain().GetAsDataSet())
    assert tree.GetName() == "chain"
    assert tree.GetEntries() == 3
    branches = list(tree.GetListOfBranches())
    assert [one.GetName() for one in branches] == ["mu", NLL, "s"]
    for one in branches:
        assert one.GetLeaf(one.GetName()).GetTypeName() == "Double_t"
    assert tree.Draw("mu") == 3
    assert tree.GetHistogram().Integral() == pytest.approx(3.0)


def test_a_variable_with_an_error_gets_a_branch_for_it() -> None:
    """``hasAsymError`` first, as ROOT asks: ``_errlo``/``_errhi``, else one ``_err``."""
    plain = RooRealVar("plain", "", 1.0)
    spread, both = RooRealVar("spread", "", 2.0), RooRealVar("both", "", 3.0)
    spread.setError(0.5)
    both.setAsymError(-0.25, 0.75)
    data = RooDataSet("d", "d", [plain, spread, both])
    data.add([plain, spread, both])
    tree = GetAsTTree("t", "t", data)
    assert [one.GetName() for one in tree.GetListOfBranches()] == [
        "both", "both_errhi", "both_errlo", "plain", "spread", "spread_err",
    ]
    assert INVALID == -999.0
