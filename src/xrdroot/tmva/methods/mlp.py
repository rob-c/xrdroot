"""``MLP``: TMVA's multilayer perceptron, trained by BFGS or by back propagation in NumPy.

``HiddenLayers=N+5`` (``N`` the number of variables, several layers
comma-separated), ``NeuronType=tanh`` for the hidden layers, a sigmoid
output with the cross entropy for a classification and a linear one with
the squared error otherwise, ``NCycles`` iterations of ``TrainingMethod``:
``BFGS`` is SciPy's limited-memory BFGS, ``BP`` and ``GA`` minibatch
gradient descent at ``LearningRate``. ``UseRegulator`` - TMVA's Bayesian
regulator, which tunes the weight decay as it goes - is a fixed weight
decay here. The network is written in TMVA's MLP ``<Layout>``, and TMVA's
MLP weight files are read.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..dataset import Events
from ..method import CLASSIFICATION, MULTICLASS, REGRESSION, Method
from ..nettrain import Descent, initial_network, loss_and_gradient, train_bfgs, train_descent
from ..networks import ACTIVATIONS, Network, mlp_from_xml
from ..xmlfile import Node

__all__ = ["MethodMLP", "layer_sizes", "targets_of"]

#: The weight decay standing in for the Bayesian regulator.
REGULATOR_DECAY = 1e-3


def layer_sizes(spec: str, nvar: int, nout: int) -> list[int]:
    """``ParseLayoutString``: ``"N+5,N"`` as every layer's neurons, input and output included."""
    sizes = [nvar]
    for piece in (part.strip() for part in spec.split(",") if part.strip()):
        count = 0
        if piece[0] in "nN":
            count, piece = nvar, piece[1:]
        sizes.append(count + (int(piece) if piece else 0))
    return [*sizes, nout]


def targets_of(method: Method, events: Events) -> tuple[Any, str]:
    """What a network is trained towards, and the loss: the class, the classes, or the targets."""
    if method.analysis == REGRESSION:
        return events.targets, "mse"
    if method.analysis == MULTICLASS:
        classes = np.arange(method.dsi.GetNClasses())
        return (events.classes[:, None] == classes).astype(np.float64), "softmax"
    signal = events.classes == method.dsi.GetSignalClassIndex()
    return signal.astype(np.float64)[:, None], "ce"


def outputs(method: Method, network: Network, values: Any) -> Any:
    """The network's answer: one value per event, a class distribution, or the targets."""
    found = network(values)
    if method.analysis == CLASSIFICATION:
        return found[:, 0]
    if method.analysis == REGRESSION:
        return method.handler.inverse_targets(found)
    return found


class MethodMLP(Method):
    """``TMVA::MethodMLP``."""

    type_name = "MLP"
    analyses = frozenset({CLASSIFICATION, REGRESSION, MULTICLASS})
    defaults: ClassVar[dict[str, Any]] = {
        "NCycles": 500,
        "HiddenLayers": "N,N-1",
        "NeuronType": "sigmoid",
        "RandomSeed": 1,
        "EstimatorType": "MSE",
        "NeuronInputType": "sum",
        "TrainingMethod": "BP",
        "LearningRate": 0.02,
        "DecayRate": 0.01,
        "TestRate": 10,
        "BatchSize": -1,
        "UseRegulator": False,
        "WeightRange": 1.0,
    }

    def process_options(self) -> None:
        nout = {REGRESSION: self.dsi.GetNTargets(), MULTICLASS: self.dsi.GetNClasses()}
        self.sizes = layer_sizes(
            str(self.opt("HiddenLayers")), self.dsi.GetNVariables(), nout.get(self.analysis, 1)
        )
        self.hidden = ACTIVATIONS.get(str(self.opt("NeuronType")).lower(), "sigmoid")
        output = {CLASSIFICATION: "sigmoid", MULTICLASS: "softmax"}.get(self.analysis, "linear")
        activations = [self.hidden] * (len(self.sizes) - 2) + ["linear"]
        self.network = initial_network(self.sizes, activations, output, int(self.opt("RandomSeed")))

    def booked(self) -> None:
        self.log.header("Building Network. ")
        self.log.info("Initializing weights")

    def train(self, events: Events) -> None:
        target, kind = targets_of(self, events)
        self.log.info("Training Network")
        self.log.info("")
        decay = REGULATOR_DECAY if self.opt("UseRegulator") else 0.0
        cycles = int(self.opt("NCycles"))
        values = events.values
        if str(self.opt("TrainingMethod")).upper() == "BFGS":
            train_bfgs(self.network, values, target, events.weights, kind, cycles, decay)
        else:
            batch = int(self.opt("BatchSize"))
            settings = Descent(
                learning_rate=float(self.opt("LearningRate")),
                momentum=0.0,
                batch_size=batch if batch > 0 else 1 if len(values) < 2 else min(len(values), 32),
                convergence_steps=cycles,
                max_epochs=cycles,
                decay=decay,
                optimizer="SGD",
            )
            data = (values, target, events.weights)
            self.network = train_descent(self.network, data, data, kind, settings)
        if self.opt("UseRegulator"):
            self._regulator_line(events, target, kind)
        self._statistics = events

    def _regulator_line(self, events: Events, target: Any, kind: str) -> None:
        """TMVA's closing line of a regulated training: the training and test losses."""
        train_error = loss_and_gradient(self.network, events.values, target, events.weights, kind)[
            0
        ]
        test = self.loader.dataset().test if self.loader is not None else events
        transformed = self.handler.apply(test)
        test_target, _ = targets_of(self, transformed)
        test_error = loss_and_gradient(
            self.network, transformed.values, test_target, transformed.weights, kind
        )[0]
        self.log.info(
            f"Finalizing handling of Regulator terms, trainE={train_error:g} testE={test_error:g}"
        )

    def evaluate(self, values: Any) -> Any:
        return outputs(self, self.network, values)

    def ranking(self) -> tuple[str, list[tuple[str, float]]]:
        """``MethodANNBase::CreateRanking``: each input's squared weights times its mean squared."""
        events = self._statistics
        signal = events.classes == self.dsi.GetSignalClassIndex()
        first = self.network.layers[0][0]
        found = []
        for index, info in enumerate(self.dsi.variables):
            column = events.values[:, index]
            mean_s, mean_b = column[signal].mean(), column[~signal].mean()
            rms = (column[signal].std() + column[~signal].std()) / 2.0
            average = max((abs(mean_s) + abs(mean_b)) / 2.0, rms)
            found.append((info.label, float(np.sum(first[:, index] ** 2)) * average * average))
        return "Importance", found

    def monitoring(self, output: Any, directory: str) -> None:
        self.log.info(f"Write special histos to file: {output.GetName()}:/{directory}")

    def add_weights(self, node: Node) -> None:
        self.network.mlp_xml(node)

    def read_weights(self, node: Any) -> None:
        self.log.header("Building Network. ")
        self.log.info("Initializing weights")
        self.network = mlp_from_xml(node, self.hidden, self.network.output)
