"""Networks in NumPy: every activation's gradient, the minibatch descent, and the layouts parsed."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from xrdroot.tmva import nettrain
from xrdroot.tmva.methods.dl import parse_layout, strategy_phases
from xrdroot.tmva.methods.mlp import layer_sizes
from xrdroot.tmva.networks import Network, activate, dl_from_xml, mlp_from_xml
from xrdroot.tmva.xmlfile import Node

#: Every activation a layer can have.
NAMES = ("linear", "sigmoid", "tanh", "relu", "symmrelu", "softsign", "gauss")


def numeric_gradient(net: Network, values, target, weights, kind: str) -> np.ndarray:
    """The first layer's weight gradient by central differences."""
    matrix = net.layers[0][0]
    found = np.zeros_like(matrix)
    for index in np.ndindex(matrix.shape):
        saved = matrix[index]
        matrix[index] = saved + 1e-6
        up = nettrain.loss_and_gradient(net, values, target, weights, kind)[0]
        matrix[index] = saved - 1e-6
        down = nettrain.loss_and_gradient(net, values, target, weights, kind)[0]
        matrix[index] = saved
        found[index] = (up - down) / 2e-6
    return found


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("kind", ["ce", "softmax", "mse"])
def test_back_propagation_through_every_activation_is_the_losss_own_gradient(name, kind):
    rng = np.random.default_rng(5)
    outputs = 3 if kind == "softmax" else 1
    net = nettrain.initial_network([2, 3, outputs], [name, "linear"], "linear", 1, "xavier")
    values = rng.normal(0.3, 1.0, (6, 2))
    target = (
        np.eye(outputs)[rng.integers(0, outputs, 6)] if kind != "mse" else rng.normal(size=(6, 1))
    )
    weights = rng.uniform(0.5, 1.5, 6)
    _, grads = nettrain.loss_and_gradient(net, values, target, weights, kind)
    assert np.allclose(grads[0][0], numeric_gradient(net, values, target, weights, kind), atol=1e-5)


def test_the_activations_are_tmvas_functions_of_their_input():
    x = np.array([-2.0, 0.5])
    assert list(activate("symmrelu", x)) == [2.0, 0.5]
    assert list(activate("softsign", x)) == [-2.0 / 3.0, 0.5 / 1.5]
    assert np.allclose(activate("gauss", x), np.exp(-x * x))
    assert list(activate("linear", x)) == [-2.0, 0.5]


def test_a_network_puts_its_last_layer_through_its_output_function():
    net = Network([(np.array([[1.0, 0.0], [0.0, 2.0]]), np.zeros(2))], ["linear"], "softmax")
    assert np.allclose(net(np.array([[0.0, 0.0]])), [[0.5, 0.5]])
    net.output = "sigmoid"
    assert np.allclose(net(np.array([[0.0, 0.0]])), [[0.5, 0.5]])
    net.output = "linear"
    assert np.allclose(net(np.array([[1.0, 1.0]])), [[1.0, 2.0]])


def test_the_descent_stops_once_the_validation_loss_has_not_improved_for_its_steps():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(40, 2))
    target = (values[:, :1] > 0).astype(float)
    data = (values, target, np.ones(40))
    net = nettrain.initial_network([2, 3, 1], ["tanh", "linear"], "sigmoid")
    reported = []
    settings = nettrain.Descent(
        learning_rate=5.0,
        momentum=0.9,
        optimizer="SGD",
        batch_size=8,
        convergence_steps=2,
        max_epochs=50,
        dropout=(0.3,),
    )
    best = nettrain.train_descent(
        net, data, data, "ce", settings, lambda *line: reported.append(line)
    )
    assert len(reported) < 50 and reported[-1][4] == 2
    assert not all(line[3] for line in reported)
    assert best.layers[0][0].shape == (3, 2)


def test_bfgs_brings_the_loss_down_from_where_it_started():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(30, 2))
    target = values @ np.array([[1.0], [-2.0]])
    net = nettrain.initial_network([2, 2, 1], ["tanh", "linear"], "linear")
    before = nettrain.loss_and_gradient(net, values, target, np.ones(30), "mse")[0]
    assert nettrain.train_bfgs(net, values, target, np.ones(30), "mse", 20, 1e-3) < before


def test_the_mlps_layers_are_counted_in_the_number_of_variables():
    assert layer_sizes("N+5, ,N,3,n", 4, 1) == [4, 9, 4, 3, 4, 1]


def test_a_deep_layout_gives_each_layer_its_width_and_activation():
    sizes, names = parse_layout("DENSE|(N+100)*2|SOFTSIGN,,RELU|N/2,DENSE|64", 4, 2)
    assert sizes == [4, 208, 2, 2] and names == ["softsign", "relu", "tanh"]


def test_a_layer_width_that_is_not_arithmetic_in_n_is_refused():
    with pytest.raises(ValueError, match="not arithmetic in N"):
        parse_layout("DENSE|M+1|TANH,LINEAR", 4, 1)
    with pytest.raises(ValueError, match="not arithmetic in N"):
        parse_layout("DENSE|N**2|TANH,LINEAR", 4, 1)


def test_the_training_strategy_gives_each_phase_its_settings_and_tmvas_defaults():
    first, second = strategy_phases(
        "LearningRate=0.1,Regularization=L1,WeightDecay=0.5,DropConfig=0.2+0.1,Optimizer=sgd|"
        "MaxEpochs=7,Regularization=L2,WeightDecay=0.5"
    )
    assert first.learning_rate == 0.1 and first.decay == 0.0 and first.regularization == "1"
    assert first.dropout == (0.2, 0.1) and first.optimizer == "SGD"
    assert second.max_epochs == 7 and second.decay == 0.5 and second.regularization == "2"
    assert second.batch_size == 30 and second.learning_rate == 1e-5
    assert strategy_phases(" | ") == []


def test_an_mlp_layout_is_read_back_as_the_network_it_was_written_from():
    net = nettrain.initial_network([2, 3, 1], ["tanh", "linear"], "sigmoid", 3)
    parent = Node("MethodSetup")
    net.mlp_xml(parent)
    back = mlp_from_xml(ET.fromstring("\n".join(parent.lines()))[0], "tanh", "sigmoid")
    values = np.array([[0.2, -0.4]])
    assert back.activations == ["tanh", "linear"]
    assert np.allclose(back(values), net(values))


def test_a_deep_layout_without_an_output_function_or_activations_is_read_as_linear():
    text = (
        '<Weights><DenseLayer Width="1"><Weights Rows="1" Columns="2">1 2</Weights>'
        '<Biases Rows="1" Columns="1">0.5</Biases></DenseLayer></Weights>'
    )
    net = dl_from_xml(ET.fromstring(text))
    assert net.output == "linear" and net.activations == ["linear"]
    assert net(np.array([[1.0, 1.0]]))[0, 0] == 3.5
