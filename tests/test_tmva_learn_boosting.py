"""TMVA's boosting between scikit-learn's trees, at the edges a Factory rarely reaches."""

from __future__ import annotations

import numpy as np

from xrdroot.tmva import boosting
from xrdroot.tmva.boosting import Booster, Settings


def test_the_classes_are_left_unweighted_when_one_of_them_has_no_weight():
    signal = np.array([True, True, False])
    assert list(boosting.normalised(signal, np.array([1.0, 1.0, 0.0]))) == [1.0, 1.0, 1.0]
    assert list(boosting.normalised(signal, np.array([1.0, 3.0, 2.0]))) == [0.375, 0.375, 0.75]


def test_a_tree_wrong_on_most_events_stops_the_boosting_unless_it_is_a_single_leaf():
    values = np.zeros((5, 2))
    signal = np.array([True, True, True, False, False])
    booster = Booster(Settings(), values, np.ones(5))
    # Every event called background: the three signal events are wrong, three fifths of them.
    assert boosting._ada_boost(booster, signal, np.zeros(5), 3) == -1.0
    assert boosting._ada_boost(booster, signal, np.zeros(5), 1) == 0.0
    assert booster.forest.errors == [0.5]


def test_adaboost_over_events_that_cannot_be_told_apart_stops_after_its_first_tree():
    values = np.zeros((20, 2))
    signal = np.arange(20) % 2 == 0
    forest = boosting.adaboost(Settings(ntrees=5), values, signal, np.ones(20))
    assert len(forest.trees) == 1 and forest.weights == [0.0]
    assert list(forest.importance) == [0.0, 0.0]


def test_adaboost_r2_stops_at_a_tree_whose_loss_is_half_the_events_or_more():
    rng = np.random.default_rng(3)
    values = rng.normal(size=(40, 2))
    target = np.where(rng.random(40) < 0.5, -1.0, 1.0)
    forest = boosting.adaboost_r2(Settings(ntrees=5, max_depth=1), values, target, np.ones(40))
    assert len(forest.trees) == 1 and forest.weights == [0.0]


def test_adaboost_r2_over_a_single_leaf_as_far_from_every_event_is_given_no_weight():
    values = np.zeros((6, 1))
    target = np.array([-1.0, 1.0] * 3)
    forest = boosting.adaboost_r2(Settings(ntrees=3), values, target, np.ones(6))
    assert forest.weights == [0.0] and forest.errors == [0.5]


def test_the_r2_losses_are_each_of_tmvas_three():
    values = np.linspace(0, 1, 30)[:, None]
    target = values[:, 0] ** 2
    for loss in ("linear", "quadratic", "exponential"):
        forest = boosting.adaboost_r2(
            Settings(ntrees=2, max_depth=2, r2_loss=loss), values, target, np.ones(30)
        )
        assert len(forest.trees) == 2 and 0 < forest.errors[0] < 0.5


def test_a_gradient_regression_whose_last_bagged_sample_is_empty_has_no_targets_to_set():
    # TMVA's draw for a third tree of sixteen events, a fifth each, draws none of them.
    assert boosting.bagged(16, 2, 0.2).sum() == 0
    values = np.linspace(0, 1, 16)[:, None]
    settings = Settings(ntrees=2, max_depth=2, bagged=True, fraction=0.2, min_node=0.0)
    forest = boosting.regression(settings, values, 3 * values[:, 0], np.ones(16))
    assert len(forest.trees) == 2


def test_the_huber_transition_is_the_first_residual_that_is_not_zero_when_the_quantile_is():
    loss = boosting.RegressionLoss("Huber", 0.5)
    found = loss.targets(np.array([0.0, 0.0, 0.0, 2.0, -3.0]), np.ones(5))
    assert loss.transition == 2.0 and list(found) == [0.0, 0.0, 0.0, 2.0, -2.0]


def test_a_weighted_quantile_of_one_residual_or_of_none_of_the_weight_is_the_least_residual():
    assert boosting._weighted_quantile(np.array([4.0]), np.ones(1), 0.5) == 4.0
    assert boosting._weighted_quantile(np.array([3.0, 1.0]), np.ones(2), 0.0) == 1.0
    assert boosting._weighted_quantile(np.array([3.0, 1.0, 2.0]), np.ones(3), 1.0) == 3.0


def test_the_mean_of_the_occurrences_boosted_in_turn_is_the_count_when_nothing_is_boosted():
    assert list(boosting._geometric(np.array([1.0, 2.0]), np.array([3.0, 2.0]))) == [3.0, 6.0]
