"""Boosted decision trees end to end: every boosting, the three analyses, and the weight files."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from tmvasupport import VARIABLES, classify, gaussian, loader, regression_tree, session, weights
from xrdroot.tmva import TMVAError

__all__ = ["multiclass_loader", "read_back", "regression_loader", "session", "train_one", "without"]

#: What a Factory prints nothing but its results with.
QUIET = "!V:Silent:AnalysisType="


def multiclass_loader(name: str = "dataset") -> Any:
    """A loader of the four variables over three classes of Gaussians apart in their means."""
    made = ROOT.TMVA.DataLoader(name)
    for variable in VARIABLES:
        made.AddVariable(variable, "F")
    for index, shift in enumerate((-1.5, 0.0, 1.5)):
        made.AddTree(gaussian(f"Tree{index}", shift, 10 + index, 120), f"class{index}")
    made.PrepareTrainingAndTestTree("", "SplitMode=Random:NormMode=NumEvents:!V")
    return made


def regression_loader(name: str = "dataset") -> Any:
    """A loader of two variables and the smooth target :func:`regression_tree` makes of them."""
    made = ROOT.TMVA.DataLoader(name)
    made.AddVariable("var1", "F")
    made.AddVariable("var2", "F")
    made.AddTarget("fvalue")
    made.AddRegressionTree(regression_tree())
    made.PrepareTrainingAndTestTree("", "SplitMode=Random:NormMode=NumEvents:!V")
    return made


def train_one(
    kind: Any, title: str, text: str, analysis: str = "Classification", data: Any = None
) -> Any:
    """The method as it was trained - not as the Factory read it back - tested and evaluated."""
    made = {
        "Classification": loader,
        "Regression": regression_loader,
        "Multiclass": multiclass_loader,
    }[analysis]
    factory = ROOT.TMVA.Factory("job", QUIET + analysis)
    method = factory.BookMethod(data if data is not None else made(), kind, title, text)
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()
    return method


def without(path: str, tag: str) -> None:
    """The weight file at ``path`` rewritten without any of its ``tag`` elements."""
    with open(path) as source:
        text = source.read()
    with open(path, "w") as out:
        out.write(re.sub(rf"<{tag}\b.*?</{tag}>", "", text, flags=re.DOTALL))


def read_back(title: str, variables: tuple[str, ...] = VARIABLES) -> Any:
    """A Reader that has booked the method's weight file, and the cells of its variables."""
    reader = ROOT.TMVA.Reader("!Color:Silent")
    cells = {name: np.zeros(1, dtype=np.float32) for name in variables}
    for name, cell in cells.items():
        reader.AddVariable(name, cell)
    reader.BookMVA(title, weights(title))
    return reader, cells


@pytest.mark.parametrize(
    "text",
    [
        "NTrees=5:MaxDepth=2:BoostType=AdaBoost",
        "NTrees=5:BoostType=RealAdaBoost:SeparationType=CrossEntropy",
        "NTrees=5:BoostType=Bagging:UseRandomisedTrees:UseNvars=2",
        "NTrees=5:BoostType=Grad:Shrinkage=0.2:UseBaggedBoost:BaggedSampleFraction=0.5",
        "NTrees=5:UseYesNoLeaf=False:NegWeightTreatment=Pray",
    ],
)
def test_a_classification_forest_answers_as_its_weight_file_does(session, text):
    method = train_one(ROOT.TMVA.Types.kBDT, "BDT", text)
    assert len(method.forest.trees) >= 1
    reader, cells = read_back("BDT")
    for cell in cells.values():
        cell[0] = 0.25
    direct = method.evaluate(np.full((1, 4), 0.25))[0]
    assert reader.EvaluateMVA("BDT") == pytest.approx(direct, abs=1e-5)
    label, ranked = method.ranking()
    assert label == "Variable Importance" and len(ranked) == 4
    # The forest read back from its weight file has no importance of its variables to rank.
    assert reader.FindMVA("BDT").ranking() is None


@pytest.mark.parametrize(
    "text",
    [
        "NTrees=5:MaxDepth=3",
        "NTrees=5:BoostType=Grad:RegressionLossFunctionBDTG=Huber",
        "NTrees=5:MaxDepth=3:BoostType=Grad:RegressionLossFunctionBDTG=LeastSquares",
        "NTrees=5:BoostType=Grad:RegressionLossFunctionBDTG=AbsoluteDeviation:UseBaggedBoost",
    ],
)
def test_a_regression_forest_is_grown_by_each_boosting_and_read_back(session, text):
    method = train_one(ROOT.TMVA.Types.kBDT, "BDTR", text, "Regression")
    reader, cells = read_back("BDTR", ("var1", "var2"))
    cells["var1"][0], cells["var2"][0] = 1.0, 2.0
    direct = method.evaluate(np.array([[1.0, 2.0]]))[0, 0]
    assert reader.EvaluateRegression("BDTR")[0] == pytest.approx(direct, rel=1e-5)


def test_a_multiclass_forest_gives_every_class_a_probability(session):
    method = train_one(ROOT.TMVA.Types.kBDT, "BDTG", "NTrees=3:BoostType=Grad", "Multiclass")
    found = method.evaluate(np.zeros((2, 4)))
    assert found.shape == (2, 3) and np.allclose(found.sum(axis=1), 1.0)
    reader, _ = read_back("BDTG")
    assert sum(reader.EvaluateMulticlass("BDTG")) == pytest.approx(1.0)


def test_a_multiclass_forest_is_refused_any_boosting_but_the_gradient(session):
    with pytest.raises(TMVAError, match="only supported by gradient boost"):
        train_one(ROOT.TMVA.Types.kBDT, "BDT", "NTrees=3", "Multiclass")


def test_a_boosting_tmva_does_not_have_is_refused(session):
    with pytest.raises(TMVAError, match="unknown boost option"):
        train_one(ROOT.TMVA.Types.kBDT, "BDT", "NTrees=3:BoostType=AdaCost")


def test_a_forest_trained_without_the_negative_weights_still_answers(session):
    method = train_one(ROOT.TMVA.Types.kBDT, "BDT", "NTrees=3:IgnoreNegWeightsInTraining")
    assert 0 < len(method.forest.trees) <= 3


def test_a_bagged_regression_forest_answers_with_its_trees_weighted_mean(session):
    method = train_one(
        ROOT.TMVA.Types.kBDT, "BDTR", "NTrees=3:MaxDepth=2:BoostType=Bagging", "Regression"
    )
    found = method.evaluate(np.array([[1.0, 2.0], [4.0, 4.0]]))
    assert found.shape == (2, 1) and found[0, 0] < found[1, 0]


def test_a_gradient_boosting_told_how_to_treat_negative_weights_is_not_told_it_again(
    session, capsys
):
    loud = "!V:!Silent:AnalysisType=Classification"
    classify(
        [(ROOT.TMVA.Types.kBDT, "BDT", "NTrees=2:BoostType=Grad:NegWeightTreatment=Pray")], loud, ""
    )
    assert "change to new default" not in capsys.readouterr().out
    classify([(ROOT.TMVA.Types.kBDT, "BDT", "NTrees=2:BoostType=Grad")], loud, "")
    assert "change to new default NegWeightTreatment=Pray" in capsys.readouterr().out


def test_a_forest_trained_into_a_file_writes_its_boosting_in_a_monitor_ntuple(session):
    classify([(ROOT.TMVA.Types.kBDT, "BDT", "NTrees=4")], "!V:!Silent:AnalysisType=Classification")
    with xrdroot.open_root("out.root") as found:
        monitor = found["dataset/Method_BDT/BDT/MonitorNtuple"]
        assert list(monitor.arrays(["iTree"])["iTree"]) == [0, 1, 2, 3]


def test_a_weight_file_without_trees_is_refused(session):
    train_one(ROOT.TMVA.Types.kBDT, "BDT", "NTrees=2")
    without(weights("BDT"), "BinaryTree")
    with pytest.raises(TMVAError, match="holds no trees"):
        read_back("BDT")
