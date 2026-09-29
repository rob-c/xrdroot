"""The Factory over three classes: the per-class tables, the confusion matrices and the reader."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from tmvasupport import VARIABLES, gaussian, session
from xrdroot.tmva import TMVAError

__all__ = ["session"]

#: A small, quick gradient-boosted forest.
BDTG = "!H:!V:NTrees=5:BoostType=Grad:Shrinkage=0.5:MaxDepth=2"


def three_classes(name: str = "dataset") -> object:
    """A loader of the four variables over three classes, apart in their means."""
    made = ROOT.TMVA.DataLoader(name)
    for variable in VARIABLES:
        made.AddVariable(variable, "F")
    for number, shift in enumerate((-1.5, 0.0, 1.5)):
        made.AddTree(gaussian(f"Tree{number}", shift, 10 + number, 120), f"class{number}")
    made.PrepareTrainingAndTestTree("", "SplitMode=Random:NormMode=NumEvents:!V")
    return made


def multiclass(methods: list[tuple[str, str, str]], output: str = "multi.root") -> object:
    """A multiclass Factory that has booked, trained, tested and evaluated ``methods``."""
    target = ROOT.TFile.Open(output, "RECREATE")
    factory = ROOT.TMVA.Factory("job", target, "!V:!Silent:AnalysisType=multiclass")
    data = three_classes()
    for kind, title, text in methods:
        factory.BookMethod(data, kind, title, text)
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()
    target.Close()
    return factory


def test_a_multiclass_forest_prints_its_tables_and_writes_its_histograms(session, capsys):
    factory = multiclass([("BDT", "BDTG", BDTG), ("BDT", "BDTN", BDTG + ":VarTransform=N")])
    printed = capsys.readouterr().out
    assert "1-vs-rest performance metrics per class" in printed
    assert "=== Showing confusion matrix for method : BDTG" in printed
    assert "Multiclass evaluation of BDTG on testing sample" in printed
    with xrdroot.open_root("multi.root") as found:
        method = found["dataset/Method_BDT/BDTG"].keys()
        assert any("MVA_BDTG_Train_class0_prob_for_class1" in name for name in method)
        assert any("1v1rejBvsS_class0_vs_class2" in name for name in method)
        assert any("CorrelationMatrixclass1" in name for name in found["dataset"].keys())
    auc = factory.GetROCIntegral("dataset", "BDTG", 2)
    assert 0.5 < auc <= 1.0


def test_a_multiclass_forest_read_back_answers_every_class(session):
    multiclass([("BDT", "BDTG", BDTG)])
    reader = ROOT.TMVA.Reader("!Color:Silent")
    cells = [np.zeros(1, dtype=np.float32) for _ in VARIABLES]
    for variable, cell in zip(VARIABLES, cells):
        reader.AddVariable(variable, cell)
    reader.BookMVA("BDTG", "dataset/weights/job_BDTG.weights.xml")
    for cell in cells:
        cell[0] = 1.5
    every = reader.EvaluateMulticlass("BDTG")
    assert len(every) == 3 and abs(sum(every) - 1.0) < 1e-5
    assert reader.EvaluateMulticlass(2, "BDTG") == every[2]
    assert reader.EvaluateMulticlass([1.5] * 4, "BDTG")[0] == every[0]
    with pytest.raises(TMVAError, match="was not trained for regression"):
        reader.EvaluateRegression("BDTG")


def test_a_reader_refuses_to_answer_every_class_of_a_classifier(session):
    from tmvasupport import classify, weights

    classify([(ROOT.TMVA.Types.kLD, "LD", "")], "!V:Silent:AnalysisType=Classification", "")
    reader = ROOT.TMVA.Reader(list(VARIABLES), "Silent")
    reader.BookMVA("LD", weights("LD"))
    with pytest.raises(TMVAError, match="was not trained for multiclass"):
        reader.EvaluateMulticlass([0.0] * 4, "LD")
