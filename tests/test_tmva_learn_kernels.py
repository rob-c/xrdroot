"""The support vector machine, the nearest neighbours and PyMVA's ensembles, end to end."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from test_tmva_learn_bdt import read_back, train_one, without
from tmvasupport import session, weights
from xrdroot.tmva import TMVAError
from xrdroot.tmva.method import CLASSIFICATION, MULTICLASS
from xrdroot.tmva.methods.knn import scale_widths

__all__ = ["session"]


def agrees(method: object, title: str, variables: tuple[str, ...] = ()) -> None:
    """The method read back from its weight file answers as the trained one does."""
    reader, cells = read_back(title, variables) if variables else read_back(title)
    for cell in cells.values():
        cell[0] = 0.5
    direct = np.ravel(method.evaluate(np.full((1, len(cells)), 0.5)))[0]
    found = reader.EvaluateRegression(title)[0] if variables else reader.EvaluateMVA(title)
    assert found == pytest.approx(direct, rel=1e-4, abs=1e-5)


@pytest.mark.parametrize(
    "text",
    [
        "Gamma=0.25:C=1.0",
        "Kernel=Polynomial:Order=2:Theta=1.0:C=0.5",
        "Kernel=MultiGauss:GammaList=0.5,1,2,4",
        "Kernel=MultiGauss",
    ],
)
def test_a_support_vector_machine_answers_as_its_weight_file_does(session, text):
    agrees(train_one(ROOT.TMVA.Types.kSVM, "SVM", text), "SVM")


def test_a_support_vector_regression_answers_as_its_weight_file_does(session):
    method = train_one(ROOT.TMVA.Types.kSVM, "SVMR", "Gamma=0.5", "Regression")
    assert method.opt("C") == 0.002
    agrees(method, "SVMR", ("var1", "var2"))


def test_a_support_vector_machine_weighs_the_smaller_class_up(session):
    data = ROOT.TMVA.DataLoader("dataset")
    for variable in ("var1", "var2", "var3", "var4"):
        data.AddVariable(variable, "F")
    from tmvasupport import gaussian

    data.AddSignalTree(gaussian("TreeS", 1.0, 1, 300))
    data.AddBackgroundTree(gaussian("TreeB", -1.0, 2, 100))
    data.PrepareTrainingAndTestTree("", "SplitMode=Random:NormMode=NumEvents:!V")
    agrees(train_one(ROOT.TMVA.Types.kSVM, "SVM", "Gamma=0.25", data=data), "SVM")


def test_a_kernel_tmva_does_not_have_is_refused(session):
    with pytest.raises(TMVAError, match="Sigmoid is not a recognised kernel function"):
        train_one(ROOT.TMVA.Types.kSVM, "SVM", "Kernel=Sigmoid")


def test_a_support_vector_weight_file_without_support_vectors_is_refused(session):
    train_one(ROOT.TMVA.Types.kSVM, "SVM", "Gamma=0.25")
    without(weights("SVM"), "SupportVector")
    with pytest.raises(TMVAError, match="holds no support vectors"):
        read_back("SVM")


@pytest.mark.parametrize(
    "text",
    [
        "nkNN=15",
        "nkNN=10:UseKernel:Kernel=Gaus:SigmaFact=2:!UseWeight",
        "nkNN=10:UseKernel:Kernel=Poln:ScaleFrac=0",
        "nkNN=10:IgnoreNegWeightsInTraining",
    ],
)
def test_the_nearest_neighbours_answer_as_their_weight_file_does(session, text):
    agrees(train_one(ROOT.TMVA.Types.kKNN, "KNN", text), "KNN")


def test_the_nearest_neighbours_of_a_regression_answer_with_their_mean_target(session):
    agrees(
        train_one(ROOT.TMVA.Types.kKNN, "KNNR", "nkNN=5", "Regression"), "KNNR", ("var1", "var2")
    )


def test_a_scale_too_fine_for_the_events_there_are_is_their_whole_range():
    values = np.array([[0.0], [4.0], [1.0]])
    assert scale_widths(values, 80) == [4.0]
    assert scale_widths(np.arange(10.0)[:, None], 80) == [8.0]


@pytest.mark.parametrize(
    "kind, title, text",
    [
        (ROOT.TMVA.Types.kPyRandomForest, "PyRF", "NEstimators=5:MaxDepth=3:RandomState=1"),
        (ROOT.TMVA.Types.kPyAdaBoost, "PyAB", "NEstimators=5:RandomState=2"),
        (ROOT.TMVA.Types.kPyGTB, "PyGTB", "NEstimators=5:Loss=deviance:MaxDepth=2"),
    ],
)
def test_a_scikit_learn_ensemble_is_pickled_beside_its_weight_file_and_read_back(
    session, kind, title, text
):
    method = train_one(kind, title, text)
    prefix = method.estimator[2]
    assert os.path.exists(os.path.join("dataset", "weights", f"{prefix}_{title}.PyData"))
    agrees(method, title)


def test_pymva_starts_an_interpreter_that_is_already_this_process():
    """PyMVA's embedded interpreter is the one running the test, and its executable is this."""
    assert ROOT.TMVA.Python_Executable() == sys.executable
    assert ROOT.TMVA.PyMethodBase.PyInitialize() is None
    assert ROOT.TMVA.PyMethodBase.PyFinalize() is None


def test_a_python_ensemble_answers_with_one_probability_for_each_class(session):
    """A multiclass fit targets the class number, and answers with every class's chance."""
    from test_tmva_frame_multiclass import multiclass

    factory = multiclass([(ROOT.TMVA.Types.kPyGTB, "PyGTB", "!H:!V:NEstimators=5:MaxDepth=2")])
    method = factory.GetMethod("dataset", "PyGTB")
    assert method.analysis == MULTICLASS
    reader, cells = read_back("PyGTB")
    # The third class's means, which are the variables' first four multiples of its shift.
    for cell, mean in zip(cells.values(), (0.375, 0.75, 1.125, 1.5), strict=True):
        cell[0] = mean
    outputs = list(reader.EvaluateMulticlass("PyGTB"))
    assert len(outputs) == 3
    assert sum(outputs) == pytest.approx(1.0, abs=1e-4)
    assert outputs.index(max(outputs)) == 2
    assert reader.EvaluateMulticlass(2, "PyGTB") == pytest.approx(outputs[2], abs=1e-6)


def test_a_python_ensemble_of_two_classes_answers_with_the_signal_probability(session):
    """One class against the rest is a classification: the answer is the signal's chance."""
    method = train_one(ROOT.TMVA.Types.kPyGTB, "PyGTB", "NEstimators=5:MaxDepth=2")
    assert method.analysis == CLASSIFICATION
    assert method.evaluate(np.array([[1.0, 1.0, 1.0, 1.0]]))[0] > 0.5
    assert method.evaluate(np.array([[-1.0, -1.0, -1.0, -1.0]]))[0] < 0.5
