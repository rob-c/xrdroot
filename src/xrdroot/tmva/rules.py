"""The rules of ``RuleFit``: every node of a forest of trees, as the cuts that lead to it.

A rule is a box - per variable an optional lower and upper cut, ``min < x``
and ``x < max`` as ``RuleCut::EvalEvent`` has them - and is 1 inside, 0
outside. The forest is grown as TMVA's ``RuleFit`` grows it: ``nTrees``
trees, each's leaves holding at least a fraction of the events' weight drawn
uniformly between ``fEventsMin`` and half ``fEventsMax``, the events
reweighted by AdaBoost between trees; the trees themselves are
scikit-learn's. Rules that repeat another, or are no rule at all, are
dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..random.mersenne import TRandom3

__all__ = ["Rule", "grow_rules", "rule_matrix"]


@dataclass
class Rule:
    """One rule: its cuts, ``{variable: (low, high)}`` with ``None`` for no cut, and its fit."""

    cuts: dict[int, tuple[float | None, float | None]] = field(default_factory=dict)
    coefficient: float = 0.0
    importance: float = 0.0
    support: float = 0.0
    sigma: float = 0.0

    def inside(self, values: Any) -> Any:
        """``RuleCut::EvalEvent`` of every row of ``values``."""
        found = np.ones(len(values), dtype=bool)
        for index, (low, high) in self.cuts.items():
            column = values[:, index]
            if low is not None:
                found &= column > low
            if high is not None:
                found &= column < high
        return found

    def key(self) -> tuple[Any, ...]:
        return tuple(sorted((k, v) for k, v in self.cuts.items()))


def _node_rules(tree: Any) -> list[Rule]:
    """Every node below the root of a fitted scikit-learn tree, as the rule of its path."""
    inner = tree.tree_
    found: list[Rule] = []
    stack: list[tuple[int, dict[int, tuple[float | None, float | None]]]] = [(0, {})]
    while stack:
        node, cuts = stack.pop()
        if node != 0:
            found.append(Rule(dict(cuts)))
        left, right = inner.children_left[node], inner.children_right[node]
        if left < 0:
            continue
        variable, threshold = int(inner.feature[node]), float(inner.threshold[node])
        # scikit-learn sends x <= t left; for single-precision values that is x < ``edge`` and
        # x > ``floor``, the rule's strict cuts.
        floor = np.float32(threshold)
        if floor > threshold:
            floor = np.nextafter(floor, np.float32(-np.inf))
        edge = float(np.nextafter(floor, np.float32(np.inf)))
        low, high = cuts.get(variable, (None, None))
        below = dict(cuts)
        below[variable] = (low, edge if high is None else min(high, edge))
        above = dict(cuts)
        above[variable] = (float(floor) if low is None else max(low, float(floor)), high)
        stack.extend(((int(right), above), (int(left), below)))
    return found


def grow_rules(
    values: Any, signal: Any, weights: Any, options: dict[str, Any]
) -> tuple[list[Rule], int]:
    """``MakeForest`` and ``MakeRules``: the forest's distinct rules, and how many there were."""
    from sklearn.tree import DecisionTreeClassifier

    random = TRandom3(4357)
    boost = str(options["ForestType"]).lower() == "adaboost"
    current = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    rules: list[Rule] = []
    for number_ in range(int(options["nTrees"])):
        low, high = float(options["fEventsMin"]), 0.5 * float(options["fEventsMax"])
        fraction = float(random.uniform(low, high))
        tree = DecisionTreeClassifier(min_weight_fraction_leaf=fraction, random_state=number_)
        tree.fit(values, signal, sample_weight=current)
        rules.extend(_node_rules(tree))
        if boost:
            wrong = tree.predict(values) != signal
            error = float(np.sum(current[wrong]))
            if 0 < error < 0.5:
                current = np.where(wrong, current * (1 - error) / error, current)
                current /= current.sum()
    generated = len(rules)
    distinct: dict[tuple[Any, ...], Rule] = {}
    for rule in rules:
        if rule.cuts:
            distinct.setdefault(rule.key(), rule)
    return list(distinct.values()), generated


def rule_matrix(rules: list[Rule], values: Any) -> Any:
    """Each rule's value for every event: a column of 0s and 1s per rule."""
    if not rules:
        return np.zeros((len(values), 0))
    return np.stack([rule.inside(values) for rule in rules], axis=1).astype(np.float64)
