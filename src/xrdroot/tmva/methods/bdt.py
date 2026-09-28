"""``BDT``: boosted decision trees - TMVA's boosting over scikit-learn's trees.

``BoostType=AdaBoost`` (the default for classification), ``RealAdaBoost``,
``Bagging`` and ``Grad`` for classification; ``Grad`` for several classes;
``AdaBoostR2`` (the default) and ``Grad`` for regression - each with TMVA's
options: ``NTrees``, ``MaxDepth``, ``MinNodeSize`` (a leaf's least share of
the weight), ``AdaBoostBeta``, ``Shrinkage``, ``UseBaggedBoost`` and
``BaggedSampleFraction``, ``SeparationType`` (``GiniIndex`` or
``CrossEntropy``, scikit-learn's two), ``UseRandomisedTrees``/``UseNvars``,
``RegressionLossFunctionBDTG`` and ``HuberQuantile``. ``nCuts`` has no
counterpart: scikit-learn cuts at the best point, not on a grid. The forest
is written as TMVA's ``<BinaryTree>`` XML, and TMVA's own BDT weight files
are read and evaluated too.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import boosting
from ..dataset import Events
from ..log import Logger
from ..method import CLASSIFICATION, MULTICLASS, REGRESSION, Method
from ..trees import read_tree
from ..xmlfile import Node, children

__all__ = ["MethodBDT"]

#: ``SeparationType`` as scikit-learn's ``criterion``.
CRITERIA = {"giniindex": "gini", "giniindexwithlaplace": "gini", "crossentropy": "entropy"}


class MethodBDT(Method):
    """``TMVA::MethodBDT``."""

    type_name = "BDT"
    analyses = frozenset({CLASSIFICATION, REGRESSION, MULTICLASS})
    defaults = {
        "NTrees": 800,
        "MaxDepth": 3,
        "MinNodeSize": "5%",
        "nCuts": 20,
        "BoostType": "AdaBoost",
        "AdaBoostR2Loss": "Quadratic",
        "UseBaggedBoost": False,
        "Shrinkage": 1.0,
        "AdaBoostBeta": 0.5,
        "UseRandomisedTrees": False,
        "UseNvars": 2,
        "UsePoissonNvars": True,
        "BaggedSampleFraction": 0.6,
        "UseYesNoLeaf": True,
        "NegWeightTreatment": "InverseBoostNegWeights",
        "NodePurityLimit": 0.5,
        "SeparationType": "GiniIndex",
        "RegressionLossFunctionBDTG": "Huber",
        "HuberQuantile": 0.7,
        "DoBoostMonitor": False,
        "UseFisherCuts": False,
        "SigToBkgFraction": 1.0,
        "PruneMethod": "NoPruning",
        "PruneStrength": 0.0,
        "SkipNormalization": False,
    }

    def process_options(self) -> None:
        regression = self.analysis == REGRESSION
        given = self.options.given
        if regression and not given("MaxDepth"):
            self.values["MaxDepth"] = 50
        if regression and not given("BoostType"):
            self.values["BoostType"] = "AdaBoostR2"
        if regression and not given("MinNodeSize"):
            self.values["MinNodeSize"] = "0.2%"
        if not given("UseNvars"):
            self.values["UseNvars"] = int(np.sqrt(self.dsi.GetNVariables()) + 0.6)
        boost = str(self.opt("BoostType"))
        self.boost = "AdaBoost" if boost == "RealAdaBoost" else boost
        self.yes_no = bool(self.opt("UseYesNoLeaf")) and boost not in ("RealAdaBoost", "AdaCost")
        if regression:
            self.yes_no = False
        self.forest = boosting.Forest()

    def settings(self) -> boosting.Settings:
        text = str(self.opt("MinNodeSize")).rstrip("%")
        return boosting.Settings(
            ntrees=int(self.opt("NTrees")),
            max_depth=int(self.opt("MaxDepth")),
            min_node=float(text) / 100.0,
            boost=self.boost,
            beta=float(self.opt("AdaBoostBeta")),
            shrinkage=float(self.opt("Shrinkage")),
            bagged=bool(self.opt("UseBaggedBoost")) or self.boost == "Bagging",
            fraction=float(self.opt("BaggedSampleFraction")),
            yes_no=self.yes_no,
            purity_limit=float(self.opt("NodePurityLimit")),
            criterion=CRITERIA.get(str(self.opt("SeparationType")).lower(), "gini"),
            randomised=bool(self.opt("UseRandomisedTrees")),
            nvars=int(self.opt("UseNvars")),
            loss=str(self.opt("RegressionLossFunctionBDTG")),
            huber_quantile=float(self.opt("HuberQuantile")),
            r2_loss=str(self.opt("AdaBoostR2Loss")).lower(),
            inverse_negative=str(self.opt("NegWeightTreatment")).lower() == "inverseboostnegweights",
        )

    def train(self, events: Events) -> None:
        events = events.take(events.weights != 0)
        if self.opt("IgnoreNegWeightsInTraining"):
            events = events.take(events.weights > 0)
        settings = self.settings()
        signal = events.classes == self.dsi.GetSignalClassIndex()
        if self.analysis == CLASSIFICATION:
            self._announce(signal, events.weights)
        if self.analysis == REGRESSION:
            self.log.info(f"Regression Loss Function: {settings.loss}")
        self.log.info(f"Training {settings.ntrees} Decision Trees ... patience please")
        self.forest = self._grow(settings, events, signal)

    def _announce(self, signal: Any, weights: Any) -> None:
        boost = boosting.normalised(signal, weights)
        reweighted_s = float(np.sum((weights * boost)[signal]))
        reweighted_b = float(np.sum((weights * boost)[~signal]))
        self.log.header(f"#events: (reweighted) sig: {reweighted_s:g} bkg: {reweighted_b:g}")
        self.log.info(f"#events: (unweighted) sig: {int(signal.sum())} bkg: {int((~signal).sum())}")

    def _grow(self, settings: boosting.Settings, events: Events, signal: Any) -> boosting.Forest:
        values, weights = events.values, events.weights
        if self.analysis == MULTICLASS:
            if settings.boost != "Grad":
                raise self.log.fatal(
                    "Multiclass is currently only supported by gradient boost. "
                    "Please change boost option accordingly (BoostType=Grad)."
                )
            return boosting.multiclass(settings, values, events.classes, weights, self.dsi.GetNClasses())
        if self.analysis == REGRESSION:
            target = events.targets[:, 0]
            if settings.boost == "Grad":
                return boosting.regression(settings, values, target, weights)
            return boosting.adaboost_r2(settings, values, target, weights)
        if settings.boost == "Grad":
            return boosting.gradboost(settings, values, signal, weights)
        if settings.boost not in ("AdaBoost", "Bagging"):
            raise self.log.fatal(f"<Boost> unknown boost option {settings.boost} called")
        return boosting.adaboost(settings, values, signal, weights)

    # -- the output -----------------------------------------------------------------------

    def evaluate(self, values: Any) -> Any:
        values = np.asarray(values, dtype=np.float32)
        trees, weights = self.forest.trees, np.asarray(self.forest.weights, dtype=np.float64)
        if self.analysis == MULTICLASS:
            return self._multiclass(values)
        if self.analysis == REGRESSION:
            return self._regression(values)[:, None]
        if self.boost == "Grad":
            total = sum(tree.respond(values, False) for tree in trees)
            return 2.0 / (1.0 + np.exp(-2.0 * total)) - 1.0
        what = "ntype" if self.yes_no else "purity"
        total = sum(w * tree.respond(values, self.yes_no, what) for w, tree in zip(weights, trees))
        norm = float(np.sum(weights))
        return total / norm if norm > np.finfo(np.float64).eps else np.zeros(len(values))

    def _multiclass(self, values: Any) -> Any:
        nclasses = self.dsi.GetNClasses()
        scores = np.zeros((len(values), nclasses))
        for index, tree in enumerate(self.forest.trees):
            scores[:, index % nclasses] += tree.respond(values, False)
        return boosting.softmax(scores)

    def _regression(self, values: Any) -> Any:
        trees, weights = self.forest.trees, np.asarray(self.forest.weights, dtype=np.float64)
        responses = np.array([tree.respond(values, False) for tree in trees])
        if self.boost == "Grad":
            output = responses.sum(axis=0) + weights[0]
        elif self.boost == "AdaBoostR2":
            output = _weighted_median(responses, weights)
        else:
            output = (weights[:, None] * responses).sum(axis=0) / np.sum(weights)
        return self.handler.inverse_targets(output[:, None])[:, 0]

    def ranking(self) -> tuple[str, list[tuple[str, float]]] | None:
        if self.forest.importance is None:
            return None
        labels = [variable.label for variable in self.dsi.variables]
        return "Variable Importance", list(zip(labels, self.forest.importance))

    def monitoring(self, output: Any, directory: str) -> None:
        """``WriteMonitoringHistosToFile``: the ``MonitorNtuple`` of each tree's boosting."""
        self.log.info(f"{output.GetName()}:/{directory}")
        count = len(self.forest.trees)
        columns = {
            "iTree": np.arange(count, dtype=np.int32),
            "boostWeight": np.asarray(self.forest.weights, dtype=np.float64),
            "errorFraction": np.asarray(self.forest.errors[:count], dtype=np.float64),
        }
        output.write_tree(directory, "MonitorNtuple", columns, "BDT variables")

    # -- the weight file ------------------------------------------------------------------

    def add_weights(self, node: Node) -> None:
        weights = node.add("Weights", NTrees=len(self.forest.trees), AnalysisType=self.analysis)
        for index, (tree, weight) in enumerate(zip(self.forest.trees, self.forest.weights)):
            tree.add_xml(weights, weight, index, {})

    def read_weights(self, node: Any) -> None:
        trees = [read_tree(item) for item in children(node, "BinaryTree")]
        weights = [float(np.float32(item.get("boostWeight"))) for item in children(node, "BinaryTree")]
        self.forest = boosting.Forest(trees, weights)
        if not trees:
            raise Logger("BDT").fatal("The weight file holds no trees")


def _weighted_median(responses: Any, weights: Any) -> Any:
    """``AdaBoostR2``'s answer: the mean of the responses around each event's weighted median."""
    ntrees = len(weights)
    order = np.argsort(responses, axis=0, kind="stable")
    ordered = np.take_along_axis(responses, order, axis=0)
    cumulative = np.cumsum(weights[order], axis=0)
    total = float(np.sum(weights))
    first = np.argmax(cumulative > total / 2.0, axis=0) + 1
    low = np.maximum(first - ntrees // 6, 0)
    high = np.minimum(first + ntrees // 6, ntrees)
    answer = np.empty(responses.shape[1])
    for event in range(responses.shape[1]):
        answer[event] = ordered[low[event] : high[event], event].mean()
    return answer
