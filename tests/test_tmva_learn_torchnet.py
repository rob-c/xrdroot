"""The deep network's PyTorch training, against a stand-in for PyTorch - and PyTorch, if there.

PyTorch is not a dependency, so the adapter is exercised against a module of
NumPy that has the handful of PyTorch's calls it makes: tensors that are
arrays, ``nn.Linear``, the functional losses, and optimisers whose step
moves every parameter by a fixed amount, so that the training changes the
network and its losses and the best network seen is the one handed back.
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import sys
import types
from typing import Any, ClassVar

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from test_tmva_learn_bdt import train_one
from tmvasupport import session
from xrdroot.tmva import nettrain, torchnet

__all__ = ["session"]


class Tensor(np.ndarray):
    """A PyTorch tensor as a NumPy array: ``sum(dim, keepdim)``, ``copy_``, ``detach``..."""

    def sum(  # type: ignore[override]
        self, dim: Any = None, keepdim: bool = False, **named: Any
    ) -> Any:
        axis = named.pop("axis", dim)
        return np.ndarray.sum(self, axis=axis, keepdims=named.pop("keepdims", keepdim), **named)

    def copy_(self, other: Any) -> Tensor:
        self[...] = other
        return self

    def detach(self) -> Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return np.asarray(self)

    def backward(self) -> None:
        """The gradients, which the stand-in's optimisers do without."""


def as_tensor(data: Any, dtype: Any = None) -> Tensor:
    return np.array(data, dtype=dtype).view(Tensor)


class Linear:
    """``nn.Linear``: ``x @ W.T + b``, its parameters starting at zero."""

    def __init__(self, inputs: int, outputs: int, dtype: Any = None) -> None:
        self.weight = as_tensor(np.zeros((outputs, inputs)), dtype)
        self.bias = as_tensor(np.zeros(outputs), dtype)

    def __call__(self, x: Any) -> Any:
        return x @ self.weight.T + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]


class Optimiser:
    """``optim.Adam`` or ``optim.SGD``: each step moves every parameter by the learning rate."""

    made: ClassVar[list[Optimiser]] = []

    def __init__(self, parameters: list[Tensor], lr: float, **settings: Any) -> None:
        self.parameters, self.lr, self.settings = parameters, lr, settings
        self.steps = 0
        Optimiser.made.append(self)

    def zero_grad(self) -> None:
        """No gradients to clear."""

    def step(self) -> None:
        self.steps += 1
        for parameter in self.parameters:
            parameter -= self.lr


def binary_cross_entropy_with_logits(raw: Any, target: Any, reduction: str = "mean") -> Any:
    assert reduction == "none"
    return np.maximum(raw, 0) - raw * target + np.log1p(np.exp(-np.abs(raw)))


def log_softmax(raw: Any, dim: int) -> Any:
    top = raw.max(axis=dim, keepdims=True)
    return raw - top - np.log(np.exp(raw - top).sum(axis=dim, keepdims=True))


def fake_torch() -> types.ModuleType:
    """The stand-in: ``torch``, ``torch.nn``, ``torch.nn.functional`` and ``torch.optim``."""
    torch = types.ModuleType("torch")
    torch.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    functional = types.SimpleNamespace(
        softsign=lambda x: x / (1 + np.abs(x)),
        dropout=lambda x, rate, training: x * (1.0 / (1.0 - rate)),
        binary_cross_entropy_with_logits=binary_cross_entropy_with_logits,
        log_softmax=log_softmax,
    )
    torch.nn = types.SimpleNamespace(Linear=Linear, functional=functional)
    torch.optim = types.SimpleNamespace(Adam=Optimiser, SGD=Optimiser)
    torch.float64 = np.float64
    torch.as_tensor = as_tensor
    torch.no_grad = contextlib.nullcontext
    torch.manual_seed = lambda seed: None
    torch.tanh, torch.relu, torch.abs, torch.exp = (
        np.tanh,
        lambda x: np.maximum(x, 0),
        np.abs,
        np.exp,
    )
    torch.sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    return torch


@pytest.fixture
def torch(monkeypatch):
    """The stand-in, where ``import torch`` finds it."""
    made = fake_torch()
    monkeypatch.setitem(sys.modules, "torch", made)
    Optimiser.made.clear()
    return made


def data(kind: str, outputs: int = 1, count: int = 24) -> tuple[Any, Any, Any]:
    """Events of two variables, their targets for the loss ``kind``, and their weights."""
    rng = np.random.default_rng(1)
    values = rng.normal(size=(count, 2))
    if kind == "mse":
        return values, values[:, :1] * 2.0, np.ones(count)
    return values, np.eye(outputs)[rng.integers(0, outputs, count)], rng.uniform(0.5, 1.5, count)


@pytest.mark.parametrize(
    "name", ["tanh", "sigmoid", "relu", "symmrelu", "softsign", "gauss", "linear"]
)
def test_every_activation_is_pytorchs_function_of_the_same_name(torch, name):
    net = nettrain.initial_network([2, 3, 1], [name, "linear"], "linear", 2)
    values = data("mse")[0]
    layers = torchnet._layers(torch, net)
    found = torchnet._forward(torch, layers, net, as_tensor(values), (), False)
    assert np.allclose(found, net.raw(values))


@pytest.mark.parametrize("kind, outputs", [("ce", 1), ("softmax", 3), ("mse", 1)])
def test_the_losses_are_the_numpy_descents_own(torch, kind, outputs):
    net = nettrain.initial_network([2, 3, outputs], ["tanh", "linear"], "linear", 2)
    values, target, weights = data(kind, outputs)
    layers = torchnet._layers(torch, net)
    found = torchnet._evaluate(
        torch, layers, net, (as_tensor(values), as_tensor(target), as_tensor(weights)), kind
    )
    assert found == pytest.approx(nettrain.loss_and_gradient(net, values, target, weights, kind)[0])


@pytest.mark.parametrize("optimizer", ["ADAM", "SGD"])
def test_the_training_hands_back_the_best_network_on_the_validation_events(torch, optimizer):
    net = nettrain.initial_network([2, 2, 1], ["tanh", "linear"], "linear", 2)
    train, valid = data("mse"), data("mse", count=12)
    settings = nettrain.Descent(
        learning_rate=0.05,
        optimizer=optimizer,
        batch_size=8,
        convergence_steps=2,
        max_epochs=40,
        dropout=(0.5,),
        decay=0.1,
    )
    lines = []
    best = torchnet.train_descent(net, train, valid, "mse", settings, lambda *l: lines.append(l))
    assert Optimiser.made[0].settings["weight_decay"] == 0.1
    assert Optimiser.made[0].steps == 3 * len(lines)
    assert lines[-1][4] == 2 and len(lines) < 40
    best_line = min(lines, key=lambda line: line[2])
    assert best_line[3] and best is not net
    assert nettrain.loss_and_gradient(best, *valid, "mse")[0] == pytest.approx(best_line[2])


def test_a_training_of_no_epochs_hands_back_the_network_it_was_given(torch):
    net = nettrain.initial_network([2, 2, 1], ["tanh", "linear"], "linear", 2)
    settings = nettrain.Descent(max_epochs=0)
    assert torchnet.train_descent(net, data("mse"), data("mse"), "mse", settings) is net


def test_a_training_reports_to_no_one_when_it_is_given_no_one_to_report_to(torch):
    net = nettrain.initial_network([2, 2, 1], ["tanh", "linear"], "linear", 2)
    settings = nettrain.Descent(max_epochs=2, batch_size=24)
    assert torchnet.train_descent(net, data("mse"), data("mse"), "mse", settings) is not net
    assert Optimiser.made[0].steps == 2


def test_pytorch_is_available_when_it_can_be_imported(torch, monkeypatch):
    assert torchnet.available()
    monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.setattr(sys, "path", [])
    assert not torchnet.available()


def test_a_deep_network_is_trained_by_pytorch_when_pytorch_is_there(session, torch):
    strategy = "TrainingStrategy=LearningRate=1e-3,BatchSize=40,MaxEpochs=3,ConvergenceSteps=1"
    method = train_one(ROOT.TMVA.Types.kDL, "DL", f"Layout=TANH|4,LINEAR:{strategy}")
    assert Optimiser.made and Optimiser.made[0].steps > 0
    assert len(method.history["valError"]) >= 1


def test_the_real_pytorch_trains_a_network_to_a_lower_validation_loss():
    pytest.importorskip("torch")
    net = nettrain.initial_network([2, 4, 1], ["tanh", "linear"], "linear", 2)
    train, valid = data("mse", count=64), data("mse", count=32)
    before = nettrain.loss_and_gradient(net, *valid, "mse")[0]
    settings = nettrain.Descent(learning_rate=1e-2, batch_size=16, max_epochs=20)
    best = torchnet.train_descent(net, train, valid, "mse", settings)
    assert nettrain.loss_and_gradient(best, *valid, "mse")[0] < before
