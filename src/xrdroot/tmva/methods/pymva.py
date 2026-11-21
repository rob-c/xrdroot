"""PyMVA's Python methods: the three scikit-learn ensembles, and the user's own PyTorch model.

``PyRandomForest``, ``PyAdaBoost`` and ``PyGTB`` are TMVA's wrappers of a
scikit-learn ensemble - ``RandomForestClassifier``, ``AdaBoostClassifier``,
``GradientBoostingClassifier`` - its options passed under TMVA's names,
trained on the events and their weights, its output the probability of the
signal class. As TMVA does, the fitted classifier is pickled beside the
weight file (``PyRFModel_<name>.PyData`` and the like).

``PyTorch`` trains nothing itself: the user's Python file builds a TorchScript
model and hands TMVA the loop that trains it, so the training is the user's
and what is left here is the plumbing. ``PyKeras`` trains a Keras model the
same way and needs TensorFlow; xrdroot refuses it by name.
"""

from __future__ import annotations

import os
import pickle
import re
from typing import Any, ClassVar, NamedTuple

import numpy as np

from .. import torchscript
from ..dataset import Events
from ..log import Logger
from ..method import CLASSIFICATION, MULTICLASS, REGRESSION, Method
from ..xmlfile import Node

__all__ = ["CustomObjects", "MethodPyAdaBoost", "MethodPyGTB", "MethodPyRandomForest",
           "MethodPyTorch"]


class _PyMethod(Method):
    """The part the three share: options to keyword arguments, the pickle, the output."""

    #: A scikit-learn classifier is a classifier: one class against the rest is either analysis.
    analyses = frozenset({CLASSIFICATION, MULTICLASS})

    #: The scikit-learn estimator's module and class, and the pickle's prefix.
    estimator: ClassVar[tuple[str, str, str]] = ("", "", "")
    #: TMVA's option name, and the estimator's keyword and how its value converts.
    keywords: ClassVar[dict[str, tuple[str, Any]]] = {}

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

    def _targets(self, events: Events) -> Any:
        """What the estimator is fitted to: the class number, or which side of signal it is on.

        A multiclass fit needs every class at once, which is the class index the data set
        gave each event; a classification fit is TMVA's two classes, with the signal's 0.
        """
        if self.analysis == MULTICLASS:
            return np.asarray(events.classes)
        signal = events.classes == self.dsi.GetSignalClassIndex()
        return np.where(signal, 0, 1)

    def train(self, events: Events) -> None:
        self.model = self._made()
        self.model.fit(
            np.asarray(events.values, dtype=np.float64),
            self._targets(events),
            sample_weight=np.asarray(events.weights, dtype=np.float64),
        )

    def evaluate(self, values: Any) -> Any:
        found = self.model.predict_proba(np.asarray(values, dtype=np.float64))
        listed = list(self.model.classes_)
        if self.analysis == MULTICLASS:
            # One column per class in the data set's order, whatever order the fit sorted in.
            return found[:, [listed.index(cls) for cls in range(self.dsi.GetNClasses())]]
        return found[:, listed.index(0)]

    def add_weights(self, node: Node) -> None:
        node.add("Weights")
        with open(self._model_file(), "wb") as out:
            pickle.dump(self.model, out)

    def read_weights(self, node: Any) -> None:
        with open(self._model_file(), "rb") as source:
            # Unpickling runs what the file says; this is the file ``add_weights`` wrote.
            self.model = pickle.load(source)


def _depth(value: Any) -> Any:
    return None if str(value) in ("None", "") else int(value)


class CustomObjects(NamedTuple):
    """What the user's file hands TMVA: the loop, the optimiser, the loss and the prediction."""

    fit: Any
    optimizer: Any
    criterion: Any
    predict: Any


#: What a user's file must leave in ``load_model_custom_objects``, and what each thing does.
CUSTOM: dict[str, str] = {
    "train_func": "the training loop",
    "criterion": "the loss function",
    "predict_func": "the prediction function",
}

#: How ``ValidationSize`` may be written: ``20%``, ``0.2``, or ``100`` events.
VALIDATION = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))(%?)$")


def custom_objects(torch: Any, namespace: Any, log: Logger) -> CustomObjects:
    """The four things TMVA calls, refused by name when the user's file leaves one out.

    ROOT reads them out of the same dict with the same three keys, and falls back to
    a plain SGD optimiser when the file names none.
    """
    found = namespace.get("load_model_custom_objects")
    if not isinstance(found, dict):
        raise log.refuse(
            "The PyTorch method's UserCode must leave a dict called load_model_custom_objects "
            "holding train_func, criterion and predict_func: add it to the file that builds "
            "the model, as ROOT's own PyMVA examples do"
        )
    missing = [key for key in CUSTOM if key not in found]
    if missing:
        said = ", ".join(f"{key} ({CUSTOM[key]})" for key in missing)
        raise log.refuse(
            f"The PyTorch method's UserCode leaves no {said} in load_model_custom_objects: give "
            "them there, as ROOT's own PyMVA examples do"
        )
    return CustomObjects(
        found["train_func"],
        found.get("optimizer", torch.optim.SGD),
        found["criterion"],
        found["predict_func"],
    )


class MethodPyTorch(Method):
    """``TMVA::MethodPyTorch``: a TorchScript model, trained by the file that made it.

    The events reach the user's loop as PyTorch dataloaders of NumPy's values, one row
    an event, with the targets beside them: a class one-hot for a classification or a
    multiclass, the targets themselves for a regression. The last ``ValidationSize``
    of the training events are what the loop validates on, and the trained model is
    written to ``FilenameTrainedModel`` - the weight file holds no weights, only the
    options that name that file, exactly as ROOT's method writes none.
    """

    type_name = "PyTorch"
    analyses = frozenset({CLASSIFICATION, REGRESSION, MULTICLASS})
    defaults: ClassVar[dict[str, Any]] = {
        "FilenameModel": "",
        "FilenameTrainedModel": "",
        "BatchSize": 100,
        "NumEpochs": 10,
        "ContinueTraining": False,
        "SaveBestOnly": True,
        "LearningRateSchedule": "",
        "ValidationSize": "20%",
        "UserCode": "",
    }

    def process_options(self) -> None:
        """Configure without loading anything: the model is read when it is first used."""
        if not torchscript.available():
            raise self.log.refuse(
                "PyMVA's PyTorch method trains and evaluates the user's TorchScript model with "
                "torch, which is not installed: pip install 'xrdroot[pymva]', or book another "
                "method"
            )
        self.torch = torchscript.module()
        self.filename_model = str(self.opt("FilenameModel"))
        self.trained_model = str(self.opt("FilenameTrainedModel")) or self._default_model()
        self._custom: CustomObjects | None = None
        self._model: Any = None

    def _default_model(self) -> str:
        """ROOT's default ``FilenameTrainedModel``: ``TrainedModel_<name>.pt`` by the weights."""
        folder = os.path.dirname(self.source or self.weight_file)
        name = f"TrainedModel_{self.name}.pt"
        return os.path.join(folder, name) if folder else name

    def _functions(self) -> CustomObjects:
        """The user's loop, optimiser, loss and prediction, from its file, once."""
        if self._custom is None:
            self._custom = custom_objects(self.torch, self._user_code(), self.log)
        return self._custom

    def _user_code(self) -> dict[str, Any]:
        """The names the user's file defines, run as ROOT runs it."""
        path = str(self.opt("UserCode"))
        if not path:
            raise self.log.refuse(
                "The PyTorch method needs the Python file that builds and trains its model: "
                "name it with the option UserCode=<file>, as ROOT's own PyMVA examples do"
            )
        if not os.path.exists(path):
            raise self.log.refuse(
                f'The PyTorch method\'s UserCode "{path}" does not exist: name the file that '
                "builds the model with UserCode=<file>"
            )
        return torchscript.user_code(path)

    def _loaded(self, path: str, option: str) -> Any:
        """The TorchScript model at ``path``, refused clearly when nothing was written there."""
        if not os.path.exists(path):
            raise self.log.refuse(
                f'The PyTorch model "{path}", which {option} names, does not exist: train the '
                "method first, or point that option at a model torch.jit.save wrote"
            )
        return torchscript.load(self.torch, path)

    # -- training -------------------------------------------------------------------------

    def _n_validation(self, count: int) -> int:
        """``GetNumValidationSamples``: how many of the training events are validated on."""
        text = str(self.opt("ValidationSize")).strip()
        found = VALIDATION.match(text)
        if found is None:
            raise self.log.fatal(
                f'Cannot parse number "{text}". Expected string like "20%" or "20.0%".'
            )
        share = float(found.group(1)) / (100.0 if found.group(2) else 1.0)
        # A percentage is always a share of the events; a plain number is a share below
        # one and a count of events at or above it, as ROOT reads both.
        events = int(count * share) if found.group(2) or share < 1.0 else int(share)
        if events < 0:
            raise self.log.fatal(f'Validation size "{text}" is negative.')
        if events == 0:
            raise self.log.fatal(f'Validation size "{text}" is zero.')
        if events >= count:
            raise self.log.fatal(
                f'Validation size "{text}" is larger than or equal in size to training set '
                f'(size="{count}").'
            )
        return events

    def _targets(self, events: Events) -> Any:
        """What the model is trained towards: the class one-hot, or the regression's targets."""
        if self.analysis == REGRESSION:
            return np.asarray(events.targets, dtype=np.float64)
        classes = np.arange(self.dsi.GetNClasses())
        return (np.asarray(events.classes)[:, None] == classes).astype(np.float64)

    def _initial(self) -> tuple[Any, str]:
        """The model to train on, and the option that names it: the trained one if continuing."""
        if self.opt("ContinueTraining"):
            return self.trained_model, "FilenameTrainedModel"
        return self.filename_model, "FilenameModel"

    def train(self, events: Events) -> None:
        """The user's loop, on the events as dataloaders, its best model written as it goes."""
        # The user's own file runs first: it is what builds and saves the model to be trained.
        custom = self._functions()
        source, option = self._initial()
        self._model = self._loaded(source, option)
        n_valid = self._n_validation(len(events))
        train, valid = events.take(slice(0, len(events) - n_valid)), events.take(
            slice(len(events) - n_valid, len(events))
        )
        self.log.info(
            f"Split TMVA training data in {len(train)} training events and {len(valid)} "
            "validation events"
        )
        batch = int(self.opt("BatchSize"))
        custom.fit(
            self._model,
            torchscript.loader(self.torch, train.values, self._targets(train), batch),
            torchscript.loader(self.torch, valid.values, self._targets(valid), batch),
            num_epochs=int(self.opt("NumEpochs")),
            batch_size=batch,
            optimizer=custom.optimizer,
            criterion=custom.criterion,
            save_best=self._best() if self.opt("SaveBestOnly") else None,
            scheduler=torchscript.schedule(str(self.opt("LearningRateSchedule"))),
        )
        if not self.opt("SaveBestOnly"):
            torchscript.save(self.torch, self._model, self.trained_model)

    def _best(self) -> Any:
        """``SaveBestOnly``'s writer: the model put where the trained one is expected."""
        return torchscript.best_checkpoint(self.torch, self.trained_model)

    # -- evaluation -----------------------------------------------------------------------

    def evaluate(self, values: Any) -> Any:
        """The user's prediction function: the signal's chance, every class, or the targets."""
        if self._model is None:
            self._model = self._loaded(self.trained_model, "FilenameTrainedModel")
        found = torchscript.predictions(self._functions().predict, self._model, values)
        if self.analysis == REGRESSION:
            return self.handler.inverse_targets(found)
        if self.analysis == MULTICLASS:
            return found
        return found[:, self.dsi.GetSignalClassIndex()]

    def add_weights(self, node: Node) -> None:
        """Nothing: the weights are the trained model file, which the options name."""

    def read_weights(self, node: Any) -> None:
        """Nothing: the model is loaded from ``FilenameTrainedModel`` on first evaluation."""


class MethodPyRandomForest(_PyMethod):
    """``TMVA::MethodPyRandomForest``."""

    type_name = "PyRandomForest"
    estimator: ClassVar[tuple[str, str, str]] = (
        "sklearn.ensemble",
        "RandomForestClassifier",
        "PyRFModel",
    )
    defaults: ClassVar[dict[str, Any]] = {
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
    keywords: ClassVar[dict[str, tuple[str, Any]]] = {
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
    estimator: ClassVar[tuple[str, str, str]] = (
        "sklearn.ensemble",
        "AdaBoostClassifier",
        "PyAdaBoostModel",
    )
    defaults: ClassVar[dict[str, Any]] = {
        "NEstimators": 50,
        "LearningRate": 1.0,
        "RandomState": "None",
    }
    keywords: ClassVar[dict[str, tuple[str, Any]]] = {
        "NEstimators": ("n_estimators", int),
        "LearningRate": ("learning_rate", float),
        "RandomState": ("random_state", _depth),
    }


class MethodPyGTB(_PyMethod):
    """``TMVA::MethodPyGTB``."""

    type_name = "PyGTB"
    estimator: ClassVar[tuple[str, str, str]] = (
        "sklearn.ensemble",
        "GradientBoostingClassifier",
        "PyGTBModel",
    )
    defaults: ClassVar[dict[str, Any]] = {
        "Loss": "log_loss",
        "LearningRate": 0.1,
        "NEstimators": 100,
        "Subsample": 1.0,
        "MinSamplesSplit": 2,
        "MinSamplesLeaf": 1,
        "MaxDepth": 3,
        "RandomState": "None",
    }
    keywords: ClassVar[dict[str, tuple[str, Any]]] = {
        "Loss": ("loss", lambda v: "log_loss" if str(v) == "deviance" else str(v)),
        "LearningRate": ("learning_rate", float),
        "NEstimators": ("n_estimators", int),
        "Subsample": ("subsample", float),
        "MinSamplesSplit": ("min_samples_split", int),
        "MinSamplesLeaf": ("min_samples_leaf", int),
        "MaxDepth": ("max_depth", _depth),
        "RandomState": ("random_state", _depth),
    }
