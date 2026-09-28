"""Training a :class:`~.networks.Network` in NumPy: by BFGS (with SciPy) or by minibatch descent.

TMVA's ``MLP`` trains with BFGS (``TrainingMethod=BFGS``) or by back
propagation; its ``DL`` with ADAM or momentum SGD on minibatches, with
dropout, weight decay and an early stop once the validation loss has not
improved for ``ConvergenceSteps`` epochs. These are those, over the event
weights: the loss is the weighted cross entropy (or squared error, for a
regression), divided by the sum of weights, and its gradient is worked out
by back propagation through the layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .networks import Network, activate, softmax

__all__ = ["Descent", "initial_network", "loss_and_gradient", "train_bfgs", "train_descent"]


def initial_network(sizes: list[int], activations: list[str], output: str, seed: int = 0,
                    scheme: str = "xavieruniform") -> Network:
    """A network of the given layer sizes, its weights drawn as ``WeightInitialization`` says."""
    rng = np.random.default_rng(seed)
    made = Network(output=output)
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        if scheme.lower() == "xavier":
            weights = rng.normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)), (fan_out, fan_in))
        else:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            weights = rng.uniform(-limit, limit, (fan_out, fan_in))
        made.layers.append((weights, np.zeros(fan_out)))
    made.activations = list(activations)
    return made


def _derivative(name: str, value: Any, before: Any) -> Any:
    """An activation's derivative, from its output (and, for some, its input)."""
    if name == "tanh":
        return 1.0 - value * value
    if name == "sigmoid":
        return value * (1.0 - value)
    if name == "relu":
        return (before > 0).astype(np.float64)
    if name == "symmrelu":
        return np.sign(before)
    if name == "softsign":
        return 1.0 / (1.0 + np.abs(before)) ** 2
    if name == "gauss":
        return -2.0 * before * value
    return np.ones_like(value)


def _forward(net: Network, values: Any, masks: list[Any] | None) -> tuple[list[Any], list[Any]]:
    """Every layer's input and output, dropping out the inputs ``masks`` says to."""
    inputs, outputs = [], []
    x = values
    for index, ((weights, bias), name) in enumerate(zip(net.layers, net.activations)):
        if masks is not None and masks[index] is not None:
            x = x * masks[index]
        inputs.append(x)
        before = x @ weights.T + bias
        x = activate(name, before)
        outputs.append((before, x))
    return inputs, outputs


def _loss(kind: str, raw: Any, target: Any, weights: Any) -> tuple[float, Any]:
    """The weighted loss per unit weight, and its gradient with respect to the last layer's values."""
    total = float(np.sum(weights))
    if kind == "ce":
        p = np.clip(activate("sigmoid", raw), 1e-15, 1 - 1e-15)
        value = -np.sum(weights[:, None] * (target * np.log(p) + (1 - target) * np.log(1 - p)))
        return value / total, weights[:, None] * (p - target) / total
    if kind == "softmax":
        p = np.clip(softmax(raw), 1e-15, 1.0)
        value = -np.sum(weights[:, None] * target * np.log(p))
        return value / total, weights[:, None] * (p - target) / total
    difference = raw - target
    value = 0.5 * np.sum(weights[:, None] * difference * difference)
    return value / total, weights[:, None] * difference / total


def loss_and_gradient(net: Network, values: Any, target: Any, weights: Any, kind: str,
                      decay: float = 0.0, masks: list[Any] | None = None) -> tuple[float, list[Any]]:
    """The loss and every layer's ``(dW, db)``, by back propagation."""
    inputs, outputs = _forward(net, values, masks)
    value, delta = _loss(kind, outputs[-1][1], target, weights)
    grads: list[Any] = [None] * len(net.layers)
    for index in range(len(net.layers) - 1, -1, -1):
        before, after = outputs[index]
        delta = delta * _derivative(net.activations[index], after, before)
        weights_l = net.layers[index][0]
        grads[index] = (delta.T @ inputs[index] + decay * weights_l, delta.sum(axis=0))
        delta = delta @ weights_l
        if masks is not None and masks[index] is not None:
            delta = delta * masks[index]
    value += 0.5 * decay * sum(float(np.sum(w * w)) for w, _ in net.layers)
    return value, grads


def _flat(net: Network) -> Any:
    return np.concatenate([np.concatenate([w.ravel(), b]) for w, b in net.layers])


def _unflat(net: Network, flat: Any) -> None:
    position = 0
    for index, (weights, bias) in enumerate(net.layers):
        size = weights.size
        new_w = flat[position : position + size].reshape(weights.shape)
        new_b = flat[position + size : position + size + len(bias)]
        net.layers[index] = (new_w.copy(), new_b.copy())
        position += size + len(bias)


def train_bfgs(net: Network, values: Any, target: Any, weights: Any, kind: str, cycles: int,
               decay: float = 0.0) -> float:
    """BFGS - SciPy's limited-memory one - on the whole training sample; the final loss."""
    from scipy.optimize import minimize

    def objective(flat: Any) -> tuple[float, Any]:
        _unflat(net, flat)
        value, grads = loss_and_gradient(net, values, target, weights, kind, decay)
        return value, np.concatenate([np.concatenate([g.ravel(), b]) for g, b in grads])

    found = minimize(objective, _flat(net), jac=True, method="L-BFGS-B",
                     options={"maxiter": max(int(cycles), 1)})
    _unflat(net, found.x)
    return float(found.fun)


@dataclass
class Descent:
    """A minibatch descent's settings: TMVA DL's ``TrainingStrategy`` for one phase."""

    learning_rate: float = 1e-5
    momentum: float = 0.3
    batch_size: int = 30
    convergence_steps: int = 100
    max_epochs: int = 2000
    decay: float = 0.0
    dropout: tuple[float, ...] = ()
    optimizer: str = "ADAM"
    seed: int = 0


def _masks(rng: Any, net: Network, size: int, dropout: tuple[float, ...]) -> list[Any]:
    """Each layer's input kept (scaled) or dropped, as ``DropConfig`` gives the drop probability."""
    made: list[Any] = []
    for index, (weights, _) in enumerate(net.layers):
        rate = dropout[index] if index < len(dropout) else 0.0
        if rate <= 0:
            made.append(None)
            continue
        keep = rng.random((size, weights.shape[1])) >= rate
        made.append(keep / (1.0 - rate))
    return made


def _step(net: Network, grads: list[Any], state: dict[str, Any], settings: Descent) -> None:
    """One update of every layer, by ADAM or by momentum SGD."""
    state["t"] += 1
    for index, ((w, b), (gw, gb)) in enumerate(zip(net.layers, grads)):
        moments = state.setdefault(index, [np.zeros_like(w), np.zeros_like(b), np.zeros_like(w), np.zeros_like(b)])
        if settings.optimizer.upper() == "ADAM":
            new = _adam(moments, (w, b), (gw, gb), state["t"], settings.learning_rate)
        else:
            moments[0] = settings.momentum * moments[0] - settings.learning_rate * gw
            moments[1] = settings.momentum * moments[1] - settings.learning_rate * gb
            new = (w + moments[0], b + moments[1])
        net.layers[index] = new


def _adam(moments: list[Any], params: tuple[Any, Any], grads: tuple[Any, Any], t: int, rate: float) -> tuple[Any, Any]:
    beta1, beta2, eps = 0.9, 0.999, 1e-7
    updated = []
    for which in range(2):
        moments[which] = beta1 * moments[which] + (1 - beta1) * grads[which]
        moments[which + 2] = beta2 * moments[which + 2] + (1 - beta2) * grads[which] ** 2
        first = moments[which] / (1 - beta1**t)
        second = moments[which + 2] / (1 - beta2**t)
        updated.append(params[which] - rate * first / (np.sqrt(second) + eps))
    return updated[0], updated[1]


def train_descent(net: Network, train: tuple[Any, Any, Any], valid: tuple[Any, Any, Any], kind: str,
                  settings: Descent, report: Any = None) -> Network:
    """Minibatch epochs until the validation loss stops improving; the best network seen."""
    rng = np.random.default_rng(settings.seed)
    values, target, weights = train
    best, best_loss, since, state = _copy(net), np.inf, 0, {"t": 0}
    for epoch in range(1, settings.max_epochs + 1):
        order = rng.permutation(len(values))
        for start in range(0, len(values), settings.batch_size):
            batch = order[start : start + settings.batch_size]
            masks = _masks(rng, net, len(batch), settings.dropout)
            _, grads = loss_and_gradient(net, values[batch], target[batch], weights[batch], kind,
                                         settings.decay, masks)
            _step(net, grads, state, settings)
        train_loss = loss_and_gradient(net, *train, kind)[0]
        valid_loss = loss_and_gradient(net, *valid, kind)[0]
        improved = valid_loss < best_loss
        if improved:
            best, best_loss, since = _copy(net), valid_loss, 0
        else:
            since += 1
        if report is not None:
            report(epoch, train_loss, valid_loss, improved, since)
        if since >= settings.convergence_steps:
            break
    return best


def _copy(net: Network) -> Network:
    return Network([(w.copy(), b.copy()) for w, b in net.layers], list(net.activations), net.output)
