"""The deep network's minibatch training in PyTorch, when PyTorch is installed.

This is :func:`~.nettrain.train_descent` done by PyTorch's autograd and
optimisers: the :class:`~.networks.Network` is copied into ``torch.nn.Linear``
layers, trained epoch by epoch on shuffled minibatches - ADAM or momentum
SGD, dropout on each layer's input, the weighted loss divided by the
batch's weight - and the best network on the validation events is copied
back out. Only the handful of PyTorch calls below are used, so that the
whole of it is exercised in the tests against a small stand-in for
PyTorch where PyTorch itself is not installed.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import numpy as np

from .nettrain import Descent
from .networks import Network

__all__ = ["available", "train_descent"]


def available() -> bool:
    """Is PyTorch installed?"""
    return importlib.util.find_spec("torch") is not None


def _layers(torch: Any, net: Network) -> list[Any]:
    """The network's layers as ``torch.nn.Linear``, its weights copied in."""
    made = []
    for weights, bias in net.layers:
        layer = torch.nn.Linear(weights.shape[1], weights.shape[0], dtype=torch.float64)
        with torch.no_grad():
            layer.weight.copy_(torch.as_tensor(weights, dtype=torch.float64))
            layer.bias.copy_(torch.as_tensor(bias, dtype=torch.float64))
        made.append(layer)
    return made


def _activation(torch: Any, name: str, x: Any) -> Any:
    functions = {
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "relu": torch.relu,
        "symmrelu": torch.abs,
        "softsign": torch.nn.functional.softsign,
    }
    if name == "gauss":
        return torch.exp(-x * x)
    return functions.get(name, lambda value: value)(x)


def _forward(torch: Any, layers: list[Any], net: Network, x: Any, dropout: tuple[float, ...], training: bool) -> Any:
    for index, (layer, name) in enumerate(zip(layers, net.activations)):
        rate = dropout[index] if index < len(dropout) else 0.0
        if training and rate > 0:
            x = torch.nn.functional.dropout(x, rate, True)
        x = _activation(torch, name, layer(x))
    return x


def _loss(torch: Any, kind: str, raw: Any, target: Any, weights: Any) -> Any:
    functional = torch.nn.functional
    total = weights.sum()
    if kind == "ce":
        each = functional.binary_cross_entropy_with_logits(raw, target, reduction="none")
    elif kind == "softmax":
        each = -(target * functional.log_softmax(raw, 1)).sum(1, keepdim=True)
    else:
        each = 0.5 * (raw - target) ** 2
    return (each.sum(1) * weights).sum() / total


def _optimiser(torch: Any, layers: list[Any], settings: Descent) -> Any:
    parameters = [p for layer in layers for p in layer.parameters()]
    if settings.optimizer.upper() == "ADAM":
        return torch.optim.Adam(parameters, lr=settings.learning_rate, eps=1e-7,
                                weight_decay=settings.decay)
    return torch.optim.SGD(parameters, lr=settings.learning_rate, momentum=settings.momentum,
                           weight_decay=settings.decay)


def _copied(layers: list[Any], net: Network) -> Network:
    pairs = [(layer.weight.detach().numpy().copy(), layer.bias.detach().numpy().copy()) for layer in layers]
    return Network(pairs, list(net.activations), net.output)


def _evaluate(torch: Any, layers: list[Any], net: Network, data: tuple[Any, Any, Any], kind: str) -> float:
    with torch.no_grad():
        raw = _forward(torch, layers, net, data[0], (), False)
        return float(_loss(torch, kind, raw, data[1], data[2]))


def train_descent(net: Network, train: tuple[Any, Any, Any], valid: tuple[Any, Any, Any], kind: str,
                  settings: Descent, report: Any = None) -> Network:
    """Minibatch epochs in PyTorch until the validation loss stops improving; the best network."""
    torch = importlib.import_module("torch")
    torch.manual_seed(settings.seed)
    tensors = [tuple(torch.as_tensor(np.asarray(part, dtype=np.float64)) for part in data)
               for data in (train, valid)]
    layers = _layers(torch, net)
    optimiser = _optimiser(torch, layers, settings)
    rng = np.random.default_rng(settings.seed)
    best, best_loss, since = net, np.inf, 0
    size = len(train[0])
    for epoch in range(1, settings.max_epochs + 1):
        order = rng.permutation(size)
        for start in range(0, size, settings.batch_size):
            batch = torch.as_tensor(order[start : start + settings.batch_size])
            values, target, weights = (part[batch] for part in tensors[0])
            optimiser.zero_grad()
            raw = _forward(torch, layers, net, values, settings.dropout, True)
            _loss(torch, kind, raw, target, weights).backward()
            optimiser.step()
        train_loss = _evaluate(torch, layers, net, tensors[0], kind)
        valid_loss = _evaluate(torch, layers, net, tensors[1], kind)
        improved = valid_loss < best_loss
        if improved:
            best, best_loss, since = _copied(layers, net), valid_loss, 0
        else:
            since += 1
        if report is not None:
            report(epoch, train_loss, valid_loss, improved, since)
        if since >= settings.convergence_steps:
            break
    return best
