"""The MLP and the deep network end to end: trained through the Factory, read back, and asked."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from test_tmva_learn_bdt import read_back, train_one
from tmvasupport import classify, session
from xrdroot.tmva import TMVAError

__all__ = ["session"]

#: A deep network's two small layers and a phase of a few epochs.
LAYOUT = "Layout=TANH|6,LINEAR"
PHASE = "TrainingStrategy=LearningRate=1e-2,BatchSize=20,MaxEpochs=3,ConvergenceSteps=1"


def agrees(method: object, title: str) -> None:
    """The method read back from its weight file answers as the trained one does."""
    reader, cells = read_back(title)
    for cell in cells.values():
        cell[0] = 0.5
    direct = method.evaluate(np.full((1, 4), 0.5))[0]
    assert reader.EvaluateMVA(title) == pytest.approx(direct, abs=1e-5)


@pytest.mark.parametrize(
    "text",
    [
        "NCycles=5:HiddenLayers=N+1,3:TrainingMethod=BFGS:NeuronType=tanh",
        "NCycles=2:HiddenLayers=4:TrainingMethod=BP:BatchSize=16:UseRegulator",
        "NCycles=2:HiddenLayers=N:NeuronType=relu",
    ],
)
def test_an_mlp_is_trained_and_answers_as_its_weight_file_does(session, text):
    method = train_one(ROOT.TMVA.Types.kMLP, "MLP", text)
    agrees(method, "MLP")
    label, ranked = method.ranking()
    assert label == "Importance" and [name for name, _ in ranked] == [
        "var1",
        "var2",
        "var3",
        "var4",
    ]


def test_an_mlp_regression_and_multiclass_give_their_targets_and_classes(session):
    method = train_one(
        ROOT.TMVA.Types.kMLP, "MLPR", "NCycles=3:HiddenLayers=3:TrainingMethod=BFGS", "Regression"
    )
    assert method.evaluate(np.ones((2, 2))).shape == (2, 1)
    method = train_one(ROOT.TMVA.Types.kMLP, "MLPM", "NCycles=3:HiddenLayers=3", "Multiclass")
    found = method.evaluate(np.zeros((2, 4)))
    assert found.shape == (2, 3) and np.allclose(found.sum(axis=1), 1.0)


def test_an_mlp_regulated_after_its_factory_is_gone_measures_its_test_loss_on_its_training(
    session, capsys
):
    factory = classify(
        [(ROOT.TMVA.Types.kMLP, "MLP", "NCycles=1:HiddenLayers=2:UseRegulator")],
        "!V:!Silent:AnalysisType=Classification",
        "",
    )
    method = factory.GetMethod("dataset", "MLP")
    events = method.loader.dataset().train
    method.loader = None
    capsys.readouterr()
    method.train(method.handler.apply(events))
    assert "Finalizing handling of Regulator terms" in capsys.readouterr().out


def test_an_mlp_trained_into_a_file_says_where_its_special_histograms_go(session, capsys):
    classify(
        [(ROOT.TMVA.Types.kMLP, "MLP", "NCycles=1:HiddenLayers=2")],
        "!V:!Silent:AnalysisType=Classification",
    )
    assert "Write special histos to file: out.root:/" in capsys.readouterr().out


@pytest.mark.parametrize(
    "text",
    [
        f"{LAYOUT}:{PHASE}",
        "Layout=DENSE|N+2|RELU,DENSE|(N*2)//4|SIGMOID,LINEAR:WeightInitialization=XAVIERUNIFORM:"
        "ValidationSize=0.25:TrainingStrategy=LearningRate=1e-2,Momentum=0.5,Optimizer=SGD,"
        "MaxEpochs=2,DropConfig=0.0+0.5+0.0,Regularization=L2,WeightDecay=1e-3|"
        "LearningRate=1e-3,MaxEpochs=1",
        f"{LAYOUT}:ValidationSize=50:{PHASE}:Architecture=GPU",
    ],
)
def test_a_deep_network_is_trained_and_answers_as_its_weight_file_does(session, text):
    method = train_one(ROOT.TMVA.Types.kDL, "DL", text)
    agrees(method, "DL")
    assert len(method.history["valError"]) >= 1


def test_a_deep_network_regression_and_multiclass_are_trained_and_read_back(session):
    method = train_one(ROOT.TMVA.Types.kDL, "DLR", f"{LAYOUT}:{PHASE}", "Regression")
    reader, _ = read_back("DLR", ("var1", "var2"))
    direct = method.evaluate(np.zeros((1, 2)))[0, 0]
    assert reader.EvaluateRegression("DLR")[0] == pytest.approx(direct, rel=1e-5)
    method = train_one(ROOT.TMVA.Types.kDL, "DLM", f"{LAYOUT}:{PHASE}", "Multiclass")
    reader, _ = read_back("DLM")
    assert list(reader.EvaluateMulticlass("DLM")) == pytest.approx(
        list(method.evaluate(np.zeros((1, 4)))[0]), abs=1e-5
    )


def test_a_deep_network_by_its_old_name_is_trained_as_dl_is(session):
    method = train_one(ROOT.TMVA.Types.kDNN, "DNN", f"{LAYOUT}:{PHASE}")
    assert method.GetMethodTypeName() == "DNN"


def test_a_deep_network_with_no_training_phase_keeps_the_network_it_started_with(session):
    method = train_one(ROOT.TMVA.Types.kDL, "DL", f"{LAYOUT}:TrainingStrategy=")
    assert method.history == {"trainingError": [], "valError": []}


def test_a_deep_network_asked_to_be_verbose_prints_the_options_it_parsed(session, capsys):
    classify(
        [(ROOT.TMVA.Types.kDL, "DL", f"V:{LAYOUT}:{PHASE}")],
        "!V:!Silent:AnalysisType=Classification",
        "",
    )
    assert "Boost_num" in capsys.readouterr().out


def test_a_convolutional_layer_is_refused_by_name(session):
    with pytest.raises(TMVAError, match="a CONV layer is a kind of TMVA DL layer"):
        train_one(ROOT.TMVA.Types.kDL, "DL", "Layout=CONV|2|3|3,LINEAR")


def test_a_deep_network_descending_too_fast_stops_once_it_no_longer_improves(session):
    strategy = (
        "TrainingStrategy=LearningRate=5,Optimizer=SGD,Momentum=0.9,BatchSize=10,"
        "MaxEpochs=30,ConvergenceSteps=2"
    )
    method = train_one(ROOT.TMVA.Types.kDL, "DL", f"{LAYOUT}:{strategy}")
    errors = [error for _, error in method.history["valError"]]
    assert len(errors) < 30 and min(errors) < errors[-1]
