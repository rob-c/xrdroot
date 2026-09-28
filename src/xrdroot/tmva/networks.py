"""Feed-forward networks: their layers, their evaluation, and TMVA's two XML layouts of them.

``MLP`` and ``DL`` are both a chain of dense layers - a weight matrix, a
bias, an activation - ending in the output's own function: a sigmoid for a
two-class cross entropy, a softmax over several classes, nothing for a
regression. However a network is trained (:mod:`.nettrain`, or PyTorch
through :mod:`.torchnet`), it ends up as a :class:`Network` here, is
evaluated here in NumPy, and is written as TMVA's MLP ``<Layout>`` or its
DL ``<DenseLayer>`` XML - both of which are read back, TMVA's own files
included.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .xmlfile import Node, children, floats, number

__all__ = ["ACTIVATIONS", "Network", "activate", "dl_from_xml", "mlp_from_xml"]

#: TMVA's DL ``EActivationFunction``, by number, as its weight files write them.
DL_ACTIVATIONS = ("linear", "relu", "sigmoid", "tanh", "symmrelu", "softsign", "gauss", "fasttanh")
#: Every activation's name in TMVA's option strings, lower-cased, and what it is here.
ACTIVATIONS = {
    "linear": "linear",
    "identity": "linear",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "fasttanh": "tanh",
    "ftanh": "tanh",
    "relu": "relu",
    "symmrelu": "symmrelu",
    "softsign": "softsign",
    "gauss": "gauss",
    "radial": "gauss",
}


def activate(name: str, x: Any) -> Any:
    """``x`` through an activation."""
    if name == "tanh":
        return np.tanh(x)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "symmrelu":
        return np.abs(x)
    if name == "softsign":
        return x / (1.0 + np.abs(x))
    if name == "gauss":
        return np.exp(-x * x)
    return x


def softmax(x: Any) -> Any:
    shifted = np.exp(x - x.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


@dataclass
class Network:
    """Dense layers - ``(weights, bias)`` with ``weights`` of shape (out, in) - and activations."""

    layers: list[tuple[Any, Any]] = field(default_factory=list)
    activations: list[str] = field(default_factory=list)
    #: What the last layer's values are put through: ``sigmoid``, ``softmax`` or ``linear``.
    output: str = "linear"

    def raw(self, values: Any) -> Any:
        """The last layer's values, before the output function."""
        x = np.asarray(values, dtype=np.float64)
        for (weights, bias), name in zip(self.layers, self.activations):
            x = activate(name, x @ weights.T + bias)
        return x

    def __call__(self, values: Any) -> Any:
        x = self.raw(values)
        if self.output == "sigmoid":
            return activate("sigmoid", x)
        if self.output == "softmax":
            return softmax(x)
        return x

    def mlp_xml(self, parent: Node) -> None:
        """``MethodANNBase::AddWeightsXMLTo``: each layer's neurons, a bias neuron last, and their synapses."""
        weights = parent.add("Weights")
        layout = weights.add("Layout", NLayers=len(self.layers) + 1)
        for index, (matrix, bias) in enumerate(self.layers):
            layer = layout.add("Layer", Index=index, NNeurons=matrix.shape[1] + 1)
            for neuron in np.vstack([matrix.T, bias[None, :]]):
                layer.add("Neuron", NSynapses=len(neuron)).block(neuron, 16)
        last = self.layers[-1][0].shape[0]
        output = layout.add("Layer", Index=len(self.layers), NNeurons=last)
        for _ in range(last):
            output.add("Neuron", NSynapses=0)

    def dl_xml(self, parent: Node, loss: str, output: str) -> None:
        """``MethodDL::AddWeightsXMLTo``: the dense layers as TMVA's DL writes them."""
        first = self.layers[0][0].shape[1]
        weights = parent.add(
            "Weights",
            NetDepth=len(self.layers),
            InputDepth=1,
            InputHeight=1,
            InputWidth=first,
            BatchSize=1,
            BatchDepth=1,
            BatchHeight=1,
            BatchWidth=first,
            LossFunction=loss,
            Initialization="F",
            Regularization=0,
            OutputFunction=output,
            WeightDecay=number(0.0),
        )
        for (matrix, bias), name in zip(self.layers, self.activations):
            layer = weights.add(
                "DenseLayer", Width=matrix.shape[0], ActivationFunction=DL_ACTIVATIONS.index(name)
            )
            layer.add("Weights", Rows=matrix.shape[0], Columns=matrix.shape[1]).block(
                np.ravel(matrix), 16
            )
            layer.add("Biases", Rows=matrix.shape[0], Columns=1).block(bias, 16)


def mlp_from_xml(node: Any, hidden: str, output: str) -> Network:
    """A network from ``MethodANNBase``'s ``<Layout>``: every layer's synapses, bias neurons last."""
    layers = children(node.find("Layout"), "Layer")
    made = Network(output=output)
    for layer in layers[:-1]:
        neurons = [floats(item) for item in children(layer, "Neuron")]
        matrix = np.array(neurons[:-1])
        made.layers.append((matrix.T.copy(), np.array(neurons[-1])))
        made.activations.append(hidden)
    made.activations[-1] = "linear"
    return made


def dl_from_xml(node: Any) -> Network:
    """A network from ``MethodDL``'s ``<DenseLayer>`` elements."""
    functions = {"S": "sigmoid", "T": "softmax"}
    made = Network(output=functions.get(str(node.get("OutputFunction", "")), "linear"))
    for layer in children(node, "DenseLayer"):
        matrix, bias = layer.find("Weights"), layer.find("Biases")
        rows, columns = int(matrix.get("Rows")), int(matrix.get("Columns"))
        made.layers.append(
            (np.array(floats(matrix)).reshape(rows, columns), np.array(floats(bias)))
        )
        made.activations.append(DL_ACTIVATIONS[int(layer.get("ActivationFunction", 0))])
    return made
