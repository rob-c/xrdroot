"""RuleFit: its forest's rules, its linear terms, its gradient-directed path, its weight file."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from tmvafitsupport import reader
from tmvasupport import classify, loader, session
from xrdroot.tmva import TMVAError
from xrdroot.tmva.rulefitting import fit_path, scan_tau
from xrdroot.tmva.rules import Rule, _node_rules, grow_rules, rule_matrix

__all__ = ["session"]

RULEFIT = ROOT.TMVA.Types.kRuleFit
#: A forest and a path small enough to be quick.
SMALL = "!H:!V:nTrees=3:GDNSteps=300:GDTau=0.5"
#: What a path is fitted with.
PATH = {"GDStep": 0.01, "GDNSteps": 300, "GDErrScale": 1.1, "GDTau": -1.0, "GDTauPrec": 0.01}


def _extremes(title: str) -> tuple[float, float]:
    made = reader(title, silent=True)
    return (
        made.EvaluateMVA([2.0, 2.0, 2.0, 2.0], title),
        made.EvaluateMVA([-2.0, -2.0, -2.0, -2.0], title),
    )


@pytest.mark.parametrize(
    "options",
    [
        f"{SMALL}:Model=ModRuleLinear",
        f"{SMALL}:Model=ModRule:ForestType=Random",
        f"{SMALL}:Model=ModLinear",
        "!H:!V:nTrees=2:GDNSteps=200:GDTau=-1:GDTauPrec=0.05",
    ],
)
def test_every_model_is_fitted_printed_and_read_back_by_a_reader(session, capsys, options):
    classify([(RULEFIT, "RuleFit", options)])
    printed = capsys.readouterr().out
    assert "RULE ENSEMBLE SUMMARY" in printed and "M o d e l" in printed
    high, low = _extremes("RuleFit")
    assert high > low


def test_a_forest_of_many_rules_prints_only_the_ten_most_important(session, capsys):
    classify([(RULEFIT, "RuleFit", "!H:!V:nTrees=6:GDNSteps=300:GDTau=0:MinImp=0")])
    assert "Skipping the next" in capsys.readouterr().out


def test_rules_all_below_the_importance_cut_are_all_removed(session, capsys):
    classify([(RULEFIT, "RuleFit", f"{SMALL}:MinImp=2")])
    printed = capsys.readouterr().out
    assert "none passed!" in printed and "Rule terms were disabled" not in printed


@pytest.mark.parametrize(
    ("options", "said"),
    [
        ("RuleFitModule=RFFriedman", "Friedman's own RuleFit"),
        ("Model=ModTree", "not a RuleFit model"),
    ],
)
def test_a_module_or_model_xrdroot_does_not_have_is_refused(session, options, said):
    factory = ROOT.TMVA.Factory("job", "!V:Silent:AnalysisType=Classification")
    with pytest.raises(TMVAError, match=said):
        factory.BookMethod(loader(), RULEFIT, "RuleFit", options)


def test_a_rule_of_no_rules_is_a_matrix_of_no_columns():
    assert rule_matrix([], np.zeros((3, 2))).shape == (3, 0)
    inside = rule_matrix(
        [Rule({0: (None, 1.0), 1: (0.0, None)})], np.array([[0.5, 0.5], [2.0, 0.5]])
    )
    assert inside[:, 0].tolist() == [1.0, 0.0]


def test_a_path_with_nothing_to_move_stops_at_once():
    zeros = (np.zeros((4, 2)), np.ones(4), np.ones(4))
    found = fit_path(zeros, zeros, 0.5, PATH)
    assert found.step == 0 and found.coefficients.tolist() == [0.0, 0.0]


def test_a_path_whose_validation_risk_grows_is_stopped_where_it_was_least():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(40, 3))
    target = np.where(features[:, 0] > 0, 1.0, -1.0)
    sample = (features, target, np.ones(40))
    valid = (features, -target, np.ones(40))
    found = fit_path(sample, valid, 0.0, {**PATH, "GDStep": 0.5, "GDNSteps": 1000})
    assert found.step == 0
    tolerant = fit_path(sample, valid, 0.0, {**PATH, "GDStep": 0.5, "GDErrScale": 1e9})
    assert tolerant.step == 0
    assert scan_tau(sample, sample, {**PATH, "GDTau": 0.3}).tau == 0.3


def test_a_cut_off_between_the_tenths_is_kept_when_its_path_is_better():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 3))
    target = np.where(features[:, 0] + 0.5 * rng.normal(size=40) > 0, 1.0, -1.0)
    sample = (features[:20], target[:20], np.ones(20))
    valid = (features[20:], target[20:], np.ones(20))
    found = scan_tau(sample, valid, {**PATH, "GDNSteps": 1000, "GDTauPrec": 0.01})
    assert found.tau not in [tenth / 10 for tenth in range(11)]


def test_a_threshold_between_single_precision_values_cuts_strictly_between_them():
    from sklearn.tree import DecisionTreeClassifier

    below = np.float32(16) + np.float32(2**-19)
    above = np.float32(16) + np.float32(2**-18)
    values = np.array([[below], [above]], dtype=np.float64)
    tree = DecisionTreeClassifier().fit(values, [0, 1])
    left, right = _node_rules(tree)
    assert left.cuts[0][0] is None and float(below) < left.cuts[0][1] <= float(above)
    assert float(below) <= right.cuts[0][0] < float(above)


def test_a_forest_of_perfect_trees_is_not_reweighted():
    options = {"ForestType": "AdaBoost", "nTrees": 2, "fEventsMin": 0.1, "fEventsMax": 0.2}
    values = np.array([[0.0], [1.0], [2.0], [3.0]])
    rules, generated = grow_rules(values, np.array([0, 0, 1, 1]), np.ones(4), options)
    assert generated == 4 and len(rules) == 2
