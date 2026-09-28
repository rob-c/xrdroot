"""``DL`` - and its old name ``DNN`` - TMVA's deep network: dense layers, trained by minibatches.

``Layout=TANH|128,TANH|128,LINEAR`` gives each layer's activation and
width (a width may be written in ``N``, the number of inputs); the
``TrainingStrategy`` phases - ``LearningRate``, ``Momentum``,
``ConvergenceSteps``, ``BatchSize``, ``WeightDecay``, ``DropConfig``,
``Optimizer=ADAM`` or ``SGD``, ``MaxEpochs`` - are run in turn on the first
``1 - ValidationSize`` of the training events, stopping each once the rest
have not improved for ``ConvergenceSteps`` epochs. The training is
PyTorch's when PyTorch is installed and NumPy's otherwise; either way the
network is TMVA's DL ``<DenseLayer>`` XML. Convolutional and recurrent
layers are refused by name.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from ..dataset import Events
from ..method import CLASSIFICATION, MULTICLASS, REGRESSION, Method
from ..nettrain import Descent, initial_network, train_descent
from ..networks import ACTIVATIONS, dl_from_xml
from ..xmlfile import Node
from .mlp import outputs, targets_of

__all__ = ["MethodDL", "MethodDNN", "parse_layout", "strategy_phases"]

#: The arithmetic a layer width may be written in.
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.Div: operator.floordiv, ast.FloorDiv: operator.floordiv}
#: The layer kinds TMVA's DL has that this does not.
REFUSED = ("CONV", "MAXPOOL", "RESHAPE", "RNN", "LSTM", "GRU", "BNORM", "PADDING2D")


def _width(text: str, nvar: int) -> int:
    """A width as written - ``128``, ``N+10``, ``(N+100)*2`` - worked out."""

    def value(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in ("N", "n"):
            return nvar
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return int(OPERATORS[type(node.op)](value(node.left), value(node.right)))
        raise ValueError(f"the layer width {text!r} is not arithmetic in N")

    return value(ast.parse(text.strip(), mode="eval").body)


def parse_layout(layout: str, nvar: int, nout: int) -> tuple[list[int], list[str]]:
    """``ParseDenseLayer`` of each layer: the network's sizes and its layers' activations."""
    sizes, activations = [nvar], []
    for layer in (piece for piece in layout.split(",") if piece.strip()):
        parts = [part.strip() for part in layer.split("|") if part.strip()]
        if parts and parts[0].upper() in REFUSED:
            raise ValueError(
                f"a {parts[0].upper()} layer is a kind of TMVA DL layer xrdroot does not have; "
                "it has DENSE layers"
            )
        parts = [part for part in parts if part.upper() != "DENSE"]
        activation, width = "tanh", 0
        for part in parts:
            if part.lower() in ACTIVATIONS:
                activation = ACTIVATIONS[part.lower()]
            else:
                width = _width(part, nvar)
        activations.append(activation)
        sizes.append(width)
    sizes[-1] = nout
    return sizes, activations


def strategy_phases(text: str) -> list[Descent]:
    """``TrainingStrategy``: each ``|``-separated phase's settings, TMVA's defaults for the rest."""
    phases = []
    for block in (piece for piece in text.split("|") if piece.strip()):
        values = dict(item.split("=", 1) for item in block.split(",") if "=" in item)
        drop = tuple(float(x) for x in values.get("DropConfig", "0.0").split("+") if x)
        phases.append(Descent(
            learning_rate=float(values.get("LearningRate", 1e-5)),
            momentum=float(values.get("Momentum", 0.3)),
            batch_size=int(float(values.get("BatchSize", 30))),
            convergence_steps=int(float(values.get("ConvergenceSteps", 100))),
            max_epochs=int(float(values.get("MaxEpochs", 2000))),
            decay=float(values.get("WeightDecay", 0.0)) if values.get("Regularization", "NONE").upper() == "L2" else 0.0,
            dropout=drop,
            optimizer=values.get("Optimizer", "ADAM").upper(),
        ))
    return phases


class MethodDL(Method):
    """``TMVA::MethodDL``: dense layers."""

    type_name = "DL"
    analyses = frozenset({CLASSIFICATION, REGRESSION, MULTICLASS})
    defaults = {
        "InputLayout": "0|0|0",
        "BatchLayout": "0|0|0",
        "Layout": "DENSE|(N+100)*2|SOFTSIGN,DENSE|0|LINEAR",
        "ErrorStrategy": "CROSSENTROPY",
        "WeightInitialization": "XAVIER",
        "RandomSeed": 0,
        "ValidationSize": "20%",
        "Architecture": "CPU",
        "TrainingStrategy": (
            "LearningRate=1e-3,Momentum=0.0,ConvergenceSteps=100,MaxEpochs=2000,Optimizer=ADAM,"
            "BatchSize=30,TestRepetitions=1,WeightDecay=0.0,Regularization=None,DropConfig=0.0"
        ),
    }

    def process_options(self) -> None:
        nout = {REGRESSION: self.dsi.GetNTargets(), MULTICLASS: self.dsi.GetNClasses()}
        try:
            self.sizes, self.activations = parse_layout(
                str(self.opt("Layout")), self.dsi.GetNVariables(), nout.get(self.analysis, 1)
            )
        except ValueError as why:
            raise self.log.fatal(str(why)) from why
        output = {CLASSIFICATION: "sigmoid", MULTICLASS: "softmax"}.get(self.analysis, "linear")
        seed = int(self.opt("RandomSeed"))
        self.network = initial_network(self.sizes, self.activations, output, seed,
                                       str(self.opt("WeightInitialization")))

    def _validation_count(self, total: int) -> int:
        text = str(self.opt("ValidationSize")).strip()
        if text.endswith("%"):
            return int(total * float(text[:-1]) / 100.0)
        fraction = float(text)
        return int(total * fraction) if fraction < 1 else int(fraction)

    def train(self, events: Events) -> None:
        target, kind = targets_of(self, events)
        nvalid = self._validation_count(len(events))
        ntrain = len(events) - nvalid
        self.log.info("Start of deep neural network training on CPU using MT,  nthreads = 1")
        self.log.info("")
        self.handler.print_stats(events)
        self._describe(ntrain, nvalid)
        train = (events.values[:ntrain], target[:ntrain], events.weights[:ntrain])
        valid = (events.values[ntrain:], target[ntrain:], events.weights[ntrain:])
        self.history: list[float] = []
        for number, phase in enumerate(strategy_phases(str(self.opt("TrainingStrategy"))), 1):
            phase.seed = int(self.opt("RandomSeed")) + number
            self._phase_header(number, phase)
            self.network = _trainer()(self.network, train, valid, kind, phase, self._report)
        self.log.info("")

    def _describe(self, ntrain: int, nvalid: int) -> None:
        import sys

        self.log.info("*****   Deep Learning Network *****")
        loss = "C" if self.analysis == CLASSIFICATION else "M" if self.analysis == REGRESSION else "S"
        sys.stdout.write(
            f"DEEP NEURAL NETWORK:   Depth = {len(self.sizes) - 1}  Input = ( 1, 1, {self.sizes[0]} )  "
            f"Loss function = {loss}\n"
        )
        for index, (width, name) in enumerate(zip(self.sizes[1:], self.activations)):
            sys.stdout.write(
                f"\tLayer {index}\t DENSE Layer: \t ( Input = {self.sizes[index]:5d} , Width = "
                f"{width:5d} ) \t Activation Function = {name.capitalize()}\n"
            )
        self.log.info(f"Using {ntrain} events for training and {nvalid} for testing")

    def _phase_header(self, number: int, phase: Descent) -> None:
        self.log.info(
            f"Training phase {number}:  Optimizer {phase.optimizer} Learning rate = "
            f"{phase.learning_rate:g}"
        )
        self.log.info("-" * 62)
        self.log.info("     Epoch |   Train Err.   Val. Err. Conv. Steps")
        self.log.info("-" * 62)
        self.log.info("   Start epoch iteration ...")

    def _report(self, epoch: int, train: float, valid: float, improved: bool, since: int) -> None:
        self.history.append(train)
        if improved:
            self.log.info(f"{epoch:>10d} Minimum Test error found - save the configuration ")
        self.log.info(f"{epoch:>10d} | {train:>12.6g} {valid:>11.6g} {since:>11d}")

    def evaluate(self, values: Any) -> Any:
        return outputs(self, self.network, values)

    def add_weights(self, node: Node) -> None:
        loss = {CLASSIFICATION: "C", REGRESSION: "R", MULTICLASS: "M"}[self.analysis]
        output = {CLASSIFICATION: "S", MULTICLASS: "T"}.get(self.analysis, "I")
        self.network.dl_xml(node, loss, output)

    def read_weights(self, node: Any) -> None:
        self.network = dl_from_xml(node)
        if self.analysis == MULTICLASS:
            self.network.output = "softmax"


class MethodDNN(MethodDL):
    """``TMVA::MethodDNN``: ``kDNN``, the deep network's old name, trained as ``DL`` is."""

    type_name = "DNN"


def _trainer() -> Any:
    """PyTorch's training when PyTorch is there, NumPy's when it is not."""
    from .. import torchnet

    return torchnet.train_descent if torchnet.available() else train_descent
