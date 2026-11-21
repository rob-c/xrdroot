"""PyMVA's Python methods: the three scikit-learn ensembles, and the user's own models.

``PyRandomForest``, ``PyAdaBoost`` and ``PyGTB`` are TMVA's wrappers of a
scikit-learn ensemble - ``RandomForestClassifier``, ``AdaBoostClassifier``,
``GradientBoostingClassifier`` - its options passed under TMVA's names,
trained on the events and their weights, its output the probability of the
signal class. As TMVA does, the fitted classifier is pickled beside the
weight file (``PyRFModel_<name>.PyData`` and the like).

``PyTorch`` trains nothing itself: the user's Python file builds a TorchScript
model and hands TMVA the loop that trains it, so the training is the user's
and what is left here is the plumbing. ``PyKeras`` trains a Keras model the
same way, except that Keras does the training: the user's file builds and
compiles the model, and ``model.fit`` takes the events and the callbacks that
ROOT hands it.
"""

from __future__ import annotations

import os
import pickle
import re
from typing import Any, ClassVar, NamedTuple

import numpy as np

from .. import kerasmodel, torchscript
from ..dataset import Events
from ..log import Logger
from ..method import CLASSIFICATION, MULTICLASS, REGRESSION, Method
from ..xmlfile import Node

__all__ = [
    "CustomObjects",
    "MethodPyAdaBoost",
    "MethodPyGTB",
    "MethodPyKeras",
    "MethodPyRandomForest",
    "MethodPyTorch",
]


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


def _beside(name: str, weight_file: str) -> str:
    """``name`` in the folder of ``weight_file``: where a method writes what it trains."""
    folder = os.path.dirname(weight_file)
    return os.path.join(folder, name) if folder else name


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


class _PyModel(Method):
    """What PyMVA's two framework methods share: which model is trained, and how much is held back.

    Both take a model file the user's own script wrote, train it on the
    training events with the tail of them held back for validation, and write
    the result to a file of their own - so the weight file has no weights in
    it, only the options that name where the model went.
    """

    #: What the default trained model's file is called, after ``TrainedModel_<name>``.
    model_suffix: ClassVar[str] = ""
    #: The model file the options name to train, and the one they name to write it to.
    filename_model: str
    trained_model: str

    def _default_model(self) -> str:
        """ROOT's default ``FilenameTrainedModel``: ``TrainedModel_<name>`` beside the weights."""
        return _beside(
            f"TrainedModel_{self.name}{self.model_suffix}", self.source or self.weight_file
        )

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

    def _initial(self) -> tuple[Any, str]:
        """The model to train on, and the option that names it: the trained one if continuing."""
        if self.opt("ContinueTraining"):
            return self.trained_model, "FilenameTrainedModel"
        return self.filename_model, "FilenameModel"

    def _split(self, events: Events) -> tuple[Events, Events]:
        """The training events and the last ``ValidationSize`` of them, as ROOT splits them."""
        n_valid = self._n_validation(len(events))
        cut = len(events) - n_valid
        self.log.info(
            f"Split TMVA training data in {cut} training events and {n_valid} validation events"
        )
        return events.take(slice(0, cut)), events.take(slice(cut, len(events)))


class MethodPyTorch(_PyModel):
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
    model_suffix = ".pt"
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

    def _targets(self, events: Events) -> Any:
        """What the model is trained towards: the class one-hot, or the regression's targets."""
        if self.analysis == REGRESSION:
            return np.asarray(events.targets, dtype=np.float64)
        classes = np.arange(self.dsi.GetNClasses())
        return (np.asarray(events.classes)[:, None] == classes).astype(np.float64)

    def train(self, events: Events) -> None:
        """The user's loop, on the events as dataloaders, its best model written as it goes."""
        # The user's own file runs first: it is what builds and saves the model to be trained.
        custom = self._functions()
        source, option = self._initial()
        self._model = self._loaded(source, option)
        train, valid = self._split(events)
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


class MethodPyKeras(_PyModel):
    """``TMVA::MethodPyKeras``: a compiled Keras model, trained by Keras on the events.

    The events reach ``model.fit`` as NumPy arrays of single precision - one row an
    event, the class one-hot or the regression's targets beside it, and the event's
    weight as the sample's - with the last ``ValidationSize`` of them held back to
    validate on. What ROOT asks of Keras through callbacks is a callback here too:
    ``SaveBestOnly`` writes the model whenever the validation loss improves,
    ``TriesEarlyStopping`` stops the training, and ``LearningRateSchedule`` and
    ``TensorBoard`` set the rate and log the run. The trained model is written to
    ``FilenameTrainedModel``, and the weight file holds no weights, only the options
    that name that file, exactly as ROOT's method writes none.

    A multiclass model's last layer is a softmax, so its row of probabilities is
    what TMVA wants one output per class for; a classification's is the chance of
    the signal class, which is the model's first output as it is ROOT's.
    """

    type_name = "PyKeras"
    analyses = frozenset({CLASSIFICATION, REGRESSION, MULTICLASS})
    model_suffix = ".keras"
    defaults: ClassVar[dict[str, Any]] = {
        "FilenameModel": "",
        "FilenameTrainedModel": "",
        "BatchSize": 100,
        "NumEpochs": 10,
        "NumThreads": 0,
        "GpuOptions": "",
        "tf.keras": True,
        "Verbose": 1,
        "ContinueTraining": False,
        "SaveBestOnly": True,
        "TriesEarlyStopping": -1,
        "LearningRateSchedule": "",
        "TensorBoard": "",
        "ValidationSize": "20%",
        "UserCode": "",
    }

    def process_options(self) -> None:
        """Configure without loading anything: the model is read when it is first used."""
        if not kerasmodel.available():
            raise self.log.refuse(
                "PyMVA's PyKeras method trains and evaluates the user's Keras model with "
                "TensorFlow, which is not installed: pip install 'xrdroot[pymva]', or book "
                "another method"
            )
        self.tensorflow = kerasmodel.module()
        self.filename_model = str(self.opt("FilenameModel"))
        self.trained_model = str(self.opt("FilenameTrainedModel")) or self._default_model()
        self._model: Any = None
        # What Keras recorded of the training: each quantity, an (epoch, value) per epoch.
        self.history: dict[str, list[tuple[int, float]]] = {}
        kerasmodel.threads(self.tensorflow, int(self.opt("NumThreads")))
        kerasmodel.memory_growth(self.tensorflow, str(self.opt("GpuOptions")))

    def _custom_objects(self) -> Any:
        """What ``load_model`` is given to rebuild classes of the user's own file, if any."""
        path = str(self.opt("UserCode"))
        if not path:
            return None
        if not os.path.exists(path):
            raise self.log.refuse(
                f'The PyKeras method\'s UserCode "{path}" does not exist: name the file that '
                "leaves load_model_custom_objects holding the classes to load the model with"
            )
        return kerasmodel.user_code(path).get("load_model_custom_objects")

    def _loaded(self, path: str, option: str) -> Any:
        """The Keras model at ``path``, refused clearly when nothing was written there."""
        if not os.path.exists(path):
            raise self.log.refuse(
                f'The Keras model "{path}", which {option} names, does not exist: train the '
                "method first, or point that option at a model model.save wrote"
            )
        return kerasmodel.load(self.tensorflow, path, self._custom_objects())

    # -- training -------------------------------------------------------------------------

    def _sample(self, events: Events) -> tuple[Any, Any, Any]:
        """The events as Keras takes them: the values, what they train towards, their weights.

        A classification's or a multiclass model is trained towards the class
        one-hot, ``1`` in the column of the event's class, as ROOT fills it.
        """
        if self.analysis == REGRESSION:
            targets = np.asarray(events.targets, dtype=np.float32)
        else:
            classes = np.arange(self.dsi.GetNClasses())
            targets = (np.asarray(events.classes)[:, None] == classes).astype(np.float32)
        return (
            np.ascontiguousarray(events.values, dtype=np.float32),
            targets,
            np.asarray(events.weights, dtype=np.float32),
        )

    def _callbacks(self) -> list[Any]:
        """The callbacks ROOT's options ask for, in the order ROOT appends them."""
        verbose = int(self.opt("Verbose"))
        made = []
        if self.opt("SaveBestOnly"):
            made.append(kerasmodel.checkpoint(self.tensorflow, self.trained_model, verbose))
        tries = int(self.opt("TriesEarlyStopping"))
        if tries >= 0:
            made.append(kerasmodel.early_stopping(self.tensorflow, tries, verbose))
        rate_for = kerasmodel.schedule(str(self.opt("LearningRateSchedule")))
        if rate_for is not None:
            made.append(kerasmodel.learning_rate(self.tensorflow, rate_for, verbose))
        log_dir = str(self.opt("TensorBoard"))
        if log_dir:
            made.append(kerasmodel.tensor_board(self.tensorflow, log_dir))
        return made

    def train(self, events: Events) -> None:
        """``model.fit`` on the events, its best model or its last written where it goes."""
        source, option = self._initial()
        self._model = self._loaded(source, option)
        if not kerasmodel.compiled(self._model):
            raise self.log.refuse(
                f'The Keras model "{source}" was saved without being compiled: call '
                "model.compile(loss=..., optimizer=...) in the script that builds it, before "
                "model.save, so that Keras can train it"
            )
        train, valid = self._split(events)
        batch = int(self.opt("BatchSize"))
        self.history = kerasmodel.fit(
            self._model,
            self._sample(train),
            self._sample(valid),
            batch,
            int(self.opt("NumEpochs")),
            int(self.opt("Verbose")),
            self._callbacks(),
        )
        if not self.opt("SaveBestOnly"):
            kerasmodel.save(self._model, self.trained_model)
        # ROOT reloads FilenameTrainedModel for every evaluation, so what the Factory scores is
        # what the file holds - with the default SaveBestOnly that is the best epoch's model, not
        # the last one training came to. The model in memory is let go for the same reason, once
        # something has written the file: the checkpoint callback does, and the save above does.
        if os.path.exists(self.trained_model):
            self._model = None

    # -- evaluation -----------------------------------------------------------------------

    def evaluate(self, values: Any) -> Any:
        """The model's answers: the signal's chance, every class's, or the targets."""
        if self._model is None:
            self._model = self._loaded(self.trained_model, "FilenameTrainedModel")
        found = kerasmodel.predictions(self._model, values, int(self.opt("Verbose")))
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
