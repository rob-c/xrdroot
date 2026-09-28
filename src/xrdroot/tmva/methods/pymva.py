"""PyMVA's scikit-learn methods: ``PyRandomForest``, ``PyAdaBoost`` and ``PyGTB``.

Each is TMVA's wrapper of a scikit-learn ensemble - ``RandomForestClassifier``,
``AdaBoostClassifier``, ``GradientBoostingClassifier`` - its options passed
under TMVA's names, trained on the events and their weights, its output
the probability of the signal class. As TMVA does, the fitted classifier is
pickled beside the weight file (``PyRFModel_<name>.PyData`` and the like).
``PyKeras`` and ``PyTorch`` train a user's own model file, which needs
Keras or PyTorch; xrdroot refuses them by name.
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np

from ..dataset import Events
from ..method import Method
from ..xmlfile import Node

__all__ = ["MethodPyAdaBoost", "MethodPyGTB", "MethodPyRandomForest"]


class _PyMethod(Method):
    """The part the three share: options to keyword arguments, the pickle, the output."""

    #: The scikit-learn estimator's module and class, and the pickle's prefix.
    estimator: tuple[str, str, str] = ("", "", "")
    #: TMVA's option name, and the estimator's keyword and how its value converts.
    keywords: dict[str, tuple[str, Any]] = {}

    def _model_file(self) -> str:
        folder = os.path.dirname(self.source or self.weight_file)
        name = f"{self.estimator[2]}_{self.name}.PyData"
        return os.path.join(folder, name) if folder else name

    def _made(self) -> Any:
        import importlib

        module, kind, _ = self.estimator
        arguments = {
            keyword: convert(self.opt(option))
            for option, (keyword, convert) in self.keywords.items()
        }
        return getattr(importlib.import_module(module), kind)(**arguments)

    def train(self, events: Events) -> None:
        signal = events.classes == self.dsi.GetSignalClassIndex()
        self.model = self._made()
        self.model.fit(
            np.asarray(events.values, dtype=np.float64),
            np.where(signal, 0, 1),
            sample_weight=np.asarray(events.weights, dtype=np.float64),
        )

    def evaluate(self, values: Any) -> Any:
        found = self.model.predict_proba(np.asarray(values, dtype=np.float64))
        return found[:, list(self.model.classes_).index(0)]

    def add_weights(self, node: Node) -> None:
        node.add("Weights")
        with open(self._model_file(), "wb") as out:
            pickle.dump(self.model, out)

    def read_weights(self, node: Any) -> None:
        with open(self._model_file(), "rb") as source:
            self.model = pickle.load(source)  # noqa: S301 - the file this package wrote


def _depth(value: Any) -> Any:
    return None if str(value) in ("None", "") else int(value)


class MethodPyRandomForest(_PyMethod):
    """``TMVA::MethodPyRandomForest``."""

    type_name = "PyRandomForest"
    estimator = ("sklearn.ensemble", "RandomForestClassifier", "PyRFModel")
    defaults = {
        "NEstimators": 10,
        "Criterion": "gini",
        "MaxDepth": "None",
        "MinSamplesSplit": 2,
        "MinSamplesLeaf": 1,
        "MinWeightFractionLeaf": 0.0,
        "MaxFeatures": "sqrt",
        "MaxLeafNodes": "None",
        "Bootstrap": True,
        "RandomState": "None",
    }
    keywords = {
        "NEstimators": ("n_estimators", int),
        "Criterion": ("criterion", str),
        "MaxDepth": ("max_depth", _depth),
        "MinSamplesSplit": ("min_samples_split", int),
        "MinSamplesLeaf": ("min_samples_leaf", int),
        "MinWeightFractionLeaf": ("min_weight_fraction_leaf", float),
        "MaxFeatures": ("max_features", lambda v: str(v).strip("'")),
        "MaxLeafNodes": ("max_leaf_nodes", _depth),
        "Bootstrap": ("bootstrap", bool),
        "RandomState": ("random_state", _depth),
    }


class MethodPyAdaBoost(_PyMethod):
    """``TMVA::MethodPyAdaBoost``."""

    type_name = "PyAdaBoost"
    estimator = ("sklearn.ensemble", "AdaBoostClassifier", "PyAdaBoostModel")
    defaults = {"NEstimators": 50, "LearningRate": 1.0, "RandomState": "None"}
    keywords = {
        "NEstimators": ("n_estimators", int),
        "LearningRate": ("learning_rate", float),
        "RandomState": ("random_state", _depth),
    }


class MethodPyGTB(_PyMethod):
    """``TMVA::MethodPyGTB``."""

    type_name = "PyGTB"
    estimator = ("sklearn.ensemble", "GradientBoostingClassifier", "PyGTBModel")
    defaults = {
        "Loss": "log_loss",
        "LearningRate": 0.1,
        "NEstimators": 100,
        "Subsample": 1.0,
        "MinSamplesSplit": 2,
        "MinSamplesLeaf": 1,
        "MaxDepth": 3,
        "RandomState": "None",
    }
    keywords = {
        "Loss": ("loss", lambda v: "log_loss" if str(v) == "deviance" else str(v)),
        "LearningRate": ("learning_rate", float),
        "NEstimators": ("n_estimators", int),
        "Subsample": ("subsample", float),
        "MinSamplesSplit": ("min_samples_split", int),
        "MinSamplesLeaf": ("min_samples_leaf", int),
        "MaxDepth": ("max_depth", _depth),
        "RandomState": ("random_state", _depth),
    }
