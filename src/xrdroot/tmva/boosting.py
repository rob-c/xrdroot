"""TMVA's boosting, over trees grown by scikit-learn: AdaBoost, bagging and gradient boosting.

What TMVA does between trees is done as TMVA does it - the classes'
weights normalised to each other first, a Poisson-drawn bagged sample from
``TRandom3(100 * trees + 1234)`` for every tree (exactly TMVA's draws), the
AdaBoost weight ``beta * ln((1 - err) / err)`` and its reweighting of each
drawn occurrence, the gradient boosting's residuals and leaf responses, the
Huber loss's transition point - while each tree itself is grown by
scikit-learn's ``DecisionTreeClassifier`` or ``DecisionTreeRegressor``,
which cut at the best point rather than at TMVA's ``nCuts`` grid points.
The forest is then TMVA's own (:mod:`.trees`), and so is everything that
reads it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..random.mersenne import TRandom3
from .trees import Tree, from_sklearn

__all__ = ["Forest", "Settings", "bagged"]


@dataclass
class Settings:
    """The options the boosting and the trees are grown with."""

    ntrees: int = 800
    max_depth: int = 3
    min_node: float = 0.05
    boost: str = "AdaBoost"
    beta: float = 0.5
    shrinkage: float = 1.0
    bagged: bool = False
    fraction: float = 0.6
    yes_no: bool = True
    purity_limit: float = 0.5
    criterion: str = "gini"
    randomised: bool = False
    nvars: int = 2
    loss: str = "huber"
    huber_quantile: float = 0.7
    r2_loss: str = "quadratic"
    inverse_negative: bool = True


@dataclass
class Forest:
    """A trained forest: its trees, their boost weights, and what the boosting recorded."""

    trees: list[Tree] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)
    importance: Any = None


def bagged(size: int, trees: int, fraction: float) -> Any:
    """``GetBaggedSubSample``: how often each event is drawn, as TMVA draws it for tree ``trees``."""
    return np.asarray(TRandom3(100 * trees + 1234).poisson_d(fraction, size), dtype=np.float64)


def _sklearn() -> Any:
    try:
        from sklearn import tree
    except ImportError as why:  # pragma: no cover - scikit-learn is installed with the tmva extra
        from .log import Logger

        raise Logger("BDT").fatal(
            "Training a BDT needs scikit-learn, which is not installed; "
            "pip install xrdroot[tmva] installs it"
        ) from why
    return tree


def _grow(
    settings: Settings, values: Any, labels: Any, weights: Any, seed: int, classify: bool
) -> Any:
    """One scikit-learn tree, grown as the options ask, on the events with non-zero weight."""
    module = _sklearn()
    keep = weights > 0
    common = {
        "max_depth": settings.max_depth,
        "min_weight_fraction_leaf": min(settings.min_node, 0.5),
        "random_state": seed,
        "max_features": settings.nvars if settings.randomised else None,
    }
    if classify:
        grown = module.DecisionTreeClassifier(criterion=settings.criterion, **common)
    else:
        grown = module.DecisionTreeRegressor(criterion="squared_error", **common)
    grown.fit(np.asarray(values[keep], dtype=np.float32), labels[keep], sample_weight=weights[keep])
    return grown


def _importance(fitted: Any) -> Any:
    """``DecisionTree::GetVariableImportance``: each variable's share of the squared weighted gains."""
    inner = fitted.tree_
    found = np.zeros(fitted.n_features_in_)
    for node in range(inner.node_count):
        left, right = inner.children_left[node], inner.children_right[node]
        if left < 0:
            continue
        total = inner.weighted_n_node_samples[node]
        gain = (
            inner.impurity[node]
            - (
                inner.weighted_n_node_samples[left] * inner.impurity[left]
                + inner.weighted_n_node_samples[right] * inner.impurity[right]
            )
            / total
        )
        found[inner.feature[node]] += (gain * total) ** 2
    total = found.sum()
    return found / total if total > np.finfo(np.float64).eps else found


def normalised(signal: Any, weights: Any) -> Any:
    """``InitEventSample``: boost weights making signal and background each half the events."""
    total = float(len(weights))
    sum_s, sum_b = float(np.sum(weights[signal])), float(np.sum(weights[~signal]))
    if not sum_s or not sum_b:
        return np.ones(len(weights))
    return np.where(signal, total / (2 * sum_s), total / (2 * sum_b))


def _geometric(factor: Any, counts: Any) -> Any:
    """``f + f**2 + ... + f**c``: what ``c`` occurrences add to the sum, each boosted in turn."""
    with np.errstate(divide="ignore", invalid="ignore"):
        series = factor * (factor**counts - 1) / (factor - 1)
    return np.where(np.isclose(factor, 1.0), counts, series)


class Booster:
    """One forest being grown: the events, their boost weights, the bagged sample, the trees."""

    def __init__(self, settings: Settings, values: Any, weights: Any) -> None:
        self.settings, self.values, self.weights = settings, values, np.asarray(weights, float)
        self.boost = np.ones(len(values))
        self.forest = Forest()
        self.importance = np.zeros(values.shape[1])
        self.counts = self._bag()

    def _bag(self) -> Any:
        if not self.settings.bagged:
            return np.ones(len(self.values))
        return bagged(len(self.values), len(self.forest.trees), self.settings.fraction)

    def sample_weights(self) -> Any:
        return self.weights * self.boost * self.counts

    def add(self, fitted: Any, tree: Tree, weight: float) -> None:
        """A tree grown and boosted: kept with its weight, and the next bagged sample drawn."""
        self.forest.trees.append(tree)
        self.forest.weights.append(max(weight, 0.0))
        self.importance += max(weight, 0.0) * _importance(fitted)
        self.counts = self._bag()

    def finish(self) -> Forest:
        importance = np.sqrt(self.importance)
        total = importance.sum()
        self.forest.importance = importance / total if total > 0 else importance
        return self.forest


def _ada_error(booster: Booster, signal: Any, output: Any) -> tuple[float, Any, Any]:
    """The weighted error of a tree, the sign of each event's truth, and its tree output."""
    settings = booster.settings
    occurrence = booster.weights * booster.boost * booster.counts
    truth = np.where(signal, 1.0, -1.0)
    if settings.yes_no:
        wrong = (output > settings.purity_limit) != signal
        return float(np.sum(occurrence[wrong])) / float(np.sum(occurrence)), truth, wrong
    scaled = (output - 0.5) * 2.0
    return float(np.sum(occurrence * truth * scaled)) / float(np.sum(occurrence)), truth, scaled


def _ada_boost(booster: Booster, signal: Any, output: Any, nodes: int) -> float:
    """``MethodBDT::AdaBoost``: the tree's boost weight, and every drawn event reweighted."""
    settings = booster.settings
    error, truth, what = _ada_error(booster, signal, output)
    if error >= 0.5 and settings.yes_no:
        if nodes != 1:
            return -1.0
        error = 0.5
    error = abs(error)
    if settings.yes_no:
        weight = float(np.log((1.0 - error) / error)) * settings.beta
        factor = np.where(what, np.exp(weight), 1.0)
    else:
        weight = float(np.log((1.0 + error) / (1.0 - error))) * settings.beta
        factor = np.exp(-weight * truth * what)
    if settings.inverse_negative:
        factor = np.where(booster.weights > 0, factor, 1.0 / factor)
    counts = booster.counts
    added = booster.weights * booster.boost * _geometric(factor, counts)
    booster.boost = booster.boost * factor**counts
    norm = float(np.sum(counts)) / float(np.sum(added))
    booster.boost = booster.boost * norm**counts
    booster.forest.errors.append(error)
    return weight


def adaboost(settings: Settings, values: Any, signal: Any, weights: Any) -> Forest:
    """AdaBoost, RealAdaBoost or bagging of classification trees, as ``MethodBDT::Train`` runs them."""
    booster = Booster(settings, values, weights)
    booster.boost = normalised(signal, booster.weights)
    booster.counts = booster._bag()
    for itree in range(settings.ntrees):
        fitted = _grow(settings, values, signal.astype(int), booster.sample_weights(), itree, True)
        column = list(fitted.classes_).index(1) if 1 in fitted.classes_ else None
        tree = from_sklearn(fitted, column, settings.purity_limit)
        if settings.boost == "Bagging":
            weight = 1.0
            booster.forest.errors.append(0.0)
        else:
            output = tree.respond(values, settings.yes_no, "purity")
            weight = _ada_boost(booster, signal, output, fitted.tree_.node_count)
        booster.add(fitted, tree, weight)
        if weight <= 0:
            break
    return booster.finish()


def _leaf_responses(fitted: Any, leaves: Any, numerator: Any, denominator: Any) -> Any:
    """Each node's response: the ratio of two sums over the drawn events in it, a leaf's own."""
    size = fitted.tree_.node_count
    top = np.bincount(leaves, numerator, size)
    bottom = np.maximum(np.bincount(leaves, denominator, size), 1e-30)
    return top / bottom


def gradboost(settings: Settings, values: Any, signal: Any, weights: Any) -> Forest:
    """``BoostType=Grad`` for two classes: the binomial log-likelihood, TMVA's residuals and steps."""
    booster = Booster(settings, values, weights)
    booster.boost = normalised(signal, booster.weights)
    booster.counts = booster._bag()
    label = signal.astype(np.float64)
    residual = np.zeros(len(values))
    target = label - 0.5
    for itree in range(settings.ntrees):
        sample = booster.sample_weights()
        fitted = _grow(settings, values, target, sample, itree, False)
        leaves = fitted.apply(np.asarray(values, dtype=np.float32))
        spread = np.abs(target) * (1.0 - np.abs(target))
        response = (
            settings.shrinkage
            * 0.5
            * _leaf_responses(fitted, leaves, target * sample, spread * sample)
        )
        tree = from_sklearn(fitted, None, settings.purity_limit, response)
        residual += tree.respond(values, False)
        target = label - 1.0 / (1.0 + np.exp(-2.0 * residual))
        booster.forest.errors.append(0.0)
        booster.add(fitted, tree, 1.0)
    return booster.finish()


def softmax(scores: Any) -> Any:
    """Each row's class probabilities, ``exp(F_k) / sum exp(F_i)``."""
    exponentials = np.exp(scores)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def multiclass(
    settings: Settings, values: Any, classes: Any, weights: Any, nclasses: int
) -> Forest:
    """``BoostType=Grad`` for several classes: a tree per class per step, and the softmax residuals."""
    booster = Booster(settings, values, weights)
    truth = (classes[:, None] == np.arange(nclasses)).astype(np.float64)
    target = truth - 1.0 / nclasses
    residual = np.zeros((len(values), nclasses))
    factor = settings.shrinkage * (nclasses - 1) / nclasses
    for itree in range(settings.ntrees):
        for cls in range(nclasses):
            sample = booster.sample_weights()
            t = target[:, cls]
            fitted = _grow(settings, values, t, sample, itree * nclasses + cls, False)
            leaves = fitted.apply(np.asarray(values, dtype=np.float32))
            spread = np.abs(t) * (1.0 - np.abs(t))
            response = factor * _leaf_responses(fitted, leaves, t * sample, spread * sample)
            tree = from_sklearn(fitted, None, settings.purity_limit, response)
            residual[:, cls] += tree.respond(values, False)
            if cls == nclasses - 1:
                target = truth - softmax(residual)
            booster.forest.errors.append(0.0)
            booster.add(fitted, tree, 1.0)
    return booster.finish()


def _weighted_quantile(residuals: Any, weights: Any, quantile: float) -> float:
    """``HuberLossFunction::CalculateQuantile``: the residual where the weight passes the quantile."""
    order = np.argsort(residuals, kind="stable")
    ordered, running = residuals[order], np.cumsum(weights[order])
    if quantile == 0 or len(ordered) == 1:
        return float(ordered[0])
    total = running[-1]
    before = np.concatenate(([0.0], running[:-1]))
    index = int(np.searchsorted(before > total * quantile, True))
    return float(ordered[min(index, len(ordered) - 1)])


class RegressionLoss:
    """``HuberLossFunctionBDT``, ``LeastSquaresLossFunctionBDT`` or ``AbsoluteDeviation...``."""

    def __init__(self, name: str, quantile: float) -> None:
        self.name, self.quantile = name.lower(), quantile
        self.transition = 0.0

    def initial(self, residuals: Any, weights: Any) -> float:
        """``Init``: the constant the boosting starts from - a weighted median, or mean."""
        if self.name == "leastsquares":
            return float(np.dot(residuals, weights) / np.sum(weights))
        return _weighted_quantile(residuals, weights, 0.5)

    def targets(self, residuals: Any, weights: Any) -> Any:
        """``SetTargets``: what the next tree is fitted to."""
        if self.name == "leastsquares":
            return residuals
        if self.name == "absolutedeviation":
            return np.where(residuals < 0, -1.0, 1.0)
        size = np.abs(residuals)
        self.transition = _weighted_quantile(size, weights, self.quantile)
        if self.transition == 0 and np.any(size != 0):
            self.transition = float(size[np.flatnonzero(size != 0)[0]])
        return np.clip(residuals, -self.transition, self.transition)

    def fit(self, residuals: Any, weights: Any) -> float:
        """``Fit``: a leaf's response, from the residuals of the drawn events in it."""
        if self.name == "leastsquares":
            return float(np.dot(residuals, weights) / np.sum(weights))
        median = _weighted_quantile(residuals, weights, 0.5)
        if self.name == "absolutedeviation":
            return median
        difference = residuals - median
        shift = np.sum(np.sign(difference) * np.minimum(self.transition, np.abs(difference)))
        return median + float(shift) / len(residuals)


def regression(settings: Settings, values: Any, target: Any, weights: Any) -> Forest:
    """``BoostType=Grad`` for a regression: TMVA's loss functions, fitted leaf by leaf."""
    booster = Booster(settings, values, weights)
    loss = RegressionLoss(settings.loss, settings.huber_quantile)
    start = loss.initial(target, booster.weights)
    predicted = np.full(len(values), start)
    drawn = booster.counts > 0
    fit_to = loss.targets((target - predicted)[drawn], booster.weights[drawn])
    current = np.zeros(len(values))
    current[drawn] = fit_to
    for itree in range(settings.ntrees):
        sample = booster.sample_weights()
        fitted = _grow(settings, values, current, sample, itree, False)
        leaves = fitted.apply(np.asarray(values, dtype=np.float32))
        response = _leaf_fits(loss, fitted, leaves, target - predicted, booster)
        tree = from_sklearn(fitted, None, settings.purity_limit, settings.shrinkage * response)
        predicted = predicted + tree.respond(values, False)
        booster.forest.errors.append(0.0)
        booster.add(fitted, tree, start if itree == 0 else 1.0)
        drawn = booster.counts > 0
        if drawn.any():
            current[drawn] = loss.targets((target - predicted)[drawn], booster.weights[drawn])
    return booster.finish()


def _leaf_fits(
    loss: RegressionLoss, fitted: Any, leaves: Any, residuals: Any, booster: Booster
) -> Any:
    """Each leaf's fit over its drawn events - an event drawn twice counting twice."""
    found = np.zeros(fitted.tree_.node_count)
    counts = booster.counts.astype(np.int64)
    for leaf in np.unique(leaves[counts > 0]):
        at = (leaves == leaf) & (counts > 0)
        repeat = counts[at]
        found[leaf] = loss.fit(
            np.repeat(residuals[at], repeat), np.repeat(booster.weights[at], repeat)
        )
    return found


def adaboost_r2(settings: Settings, values: Any, target: Any, weights: Any) -> Forest:
    """``BoostType=AdaBoostR2``: regression trees, reweighted by how far each is off."""
    booster = Booster(settings, values, weights)
    for itree in range(settings.ntrees):
        fitted = _grow(settings, values, target, booster.sample_weights(), itree, False)
        tree = from_sklearn(fitted, None, settings.purity_limit, fitted.tree_.value[:, 0, 0])
        deviation = np.abs(tree.respond(values, False) - target)
        weight = _r2_boost(booster, deviation, fitted.tree_.node_count)
        booster.add(fitted, tree, weight)
        if weight <= 0:
            break
    return booster.finish()


def _r2_boost(booster: Booster, deviation: Any, nodes: int) -> float:
    """``MethodBDT::AdaBoostR2``: the loss-weighted error, and each event's new weight."""
    occurrence = booster.weights * booster.boost * booster.counts
    total = float(np.sum(occurrence))
    largest = float(np.max(deviation[booster.counts > 0])) if np.any(booster.counts > 0) else 0.0
    largest = largest or 1.0
    scaled = deviation / largest
    loss = {"linear": scaled, "quadratic": scaled**2, "exponential": 1 - np.exp(-scaled)}
    error = float(np.sum(occurrence * loss.get(booster.settings.r2_loss, scaled**2))) / total
    if error >= 0.5:
        if nodes != 1:
            return -1.0
        error = 0.5
    ratio = error / (1.0 - error)
    factor = ratio ** (1.0 - scaled)
    factor = np.where(booster.weights > 0, factor, 1.0 / factor)
    booster.boost = booster.boost * factor**booster.counts
    new_total = float(np.sum(booster.weights * booster.boost * booster.counts))
    booster.boost = booster.boost * total / new_total
    booster.forest.errors.append(error)
    return float(np.log(1.0 / ratio))
