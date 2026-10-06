"""Cross-validation: methods trained fold by fold, their results, and the method left behind."""

from __future__ import annotations

import os

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from tmvasupport import VARIABLES, loader, session
from xrdroot.tmva import TMVAError, cvresults
from xrdroot.tmva.dataset import DataSetInfo, Events
from xrdroot.tmva.methods.crossvalidation import MethodCrossValidation

__all__ = ["session"]

#: A split of the events by their spectator, into as many folds as asked.
SPLIT = "int(fabs([spec])*100)%int([NumFolds])"
#: The options of a quiet, deterministic cross-validation.
DETERMINISTIC = f"!V:Silent:AnalysisType=Classification:SplitType=Deterministic:SplitExpr={SPLIT}"


#: Every per-fold quantity a result lists.
GETTERS = ("ROC", "Sig", "Sep", "Eff01", "Eff10", "Eff30", "EffArea", "TrainEff01", "TrainEff10")


def _random(output: str = "cv.root") -> object:
    """A two-fold random cross-validation of LD and Fisher, written to ``output``."""
    target = ROOT.TFile.Open(output, "RECREATE")
    options = "!V:!Silent:AnalysisType=Classification:NumFolds=2:Transformations=I"
    cv = ROOT.TMVA.CrossValidation("job", loader(), target, options)
    cv.BookMethod(ROOT.TMVA.Types.kLD, "LD", "!H:!V")
    cv.BookMethod("Fisher", "Fisher")
    cv.Evaluate()
    target.Close()
    return cv


def test_a_random_cross_validation_processes_the_folds_then_the_whole(session, capsys):
    _random()
    printed = capsys.readouterr().out
    assert "Processing folds for method LD" in printed
    assert "Folds processed for all methods, evaluating." in printed
    with xrdroot.open_root("cv.root") as found:
        assert "LD" in found["dataset/TestTree"].keys()
    assert os.path.exists("dataset/weights/job_LD_fold1.weights.xml")


def test_a_random_cross_validation_gives_each_fold_its_numbers_and_curve(session, capsys):
    results = _random().GetResults()
    assert [r.GetNumFolds() for r in results] == [2, 2]
    first = results[0]
    assert 0.5 < first.GetROCAverage() < 1
    assert first.GetROCStandardDeviation() >= 0
    counts = {len(getattr(first, f"Get{getter}Values")()) for getter in GETTERS}
    assert counts == {2} and len(first.GetTrainEff30Values()) == 2
    assert sorted(first.GetROCCurves()) == [0, 1]
    capsys.readouterr()
    first.Print()
    first.DrawAvgROCCurve()
    assert "Average ROC-Int :" in capsys.readouterr().out


def test_the_cross_validation_answers_what_it_was_booked_with(session):
    data = loader()
    cv = ROOT.TMVA.CrossValidation("job", data, "!V:Silent:AnalysisType=Classification")
    cv.BookMethod("LD", "LD", "!H")
    info = cv.GetMethods()[0]
    assert info.GetValue("MethodName") == "LD"
    assert info.GetValue["TString"]("MethodOptions") == "!H"
    assert cv.GetNumFolds() == 2 and cv.GetDataLoader() is data
    assert cv.GetFactory() is cv.factory
    cv.SetNumFolds(3)
    cv.SetSplitExpr(SPLIT)
    assert cv.GetNumFolds() == 3 and cv.split_expr == SPLIT
    with pytest.raises(TMVAError, match="No cross-validation results available"):
        cv.GetResults()


def test_a_split_expression_without_deterministic_splitting_is_refused(session):
    with pytest.raises(TMVAError, match="SplitExpr can only be used with Deterministic"):
        ROOT.TMVA.CrossValidation("job", loader(), f"!V:Silent:SplitExpr={SPLIT}")


def _deterministic(ensembling: str = "None", options: str = DETERMINISTIC) -> object:
    data = loader(spectator=True)
    cv = ROOT.TMVA.CrossValidation("job", data, f"{options}:OutputEnsembling={ensembling}")
    cv.BookMethod("LD", "LD", "")
    cv.Evaluate()
    return cv


def _reader(tag: str = "cv") -> object:
    reader = ROOT.TMVA.Reader("!Color:Silent")
    for variable in VARIABLES:
        reader.AddVariable(variable, np.zeros(1, dtype=np.float32))
    reader.AddSpectator("spec := var1*2", np.zeros(1, dtype=np.float32))
    reader.BookMVA(tag, "dataset/weights/job_LD.weights.xml")
    return reader


def test_a_deterministic_cross_validation_is_read_back_and_applied_fold_by_fold(session):
    _deterministic()
    reader = _reader()
    folds = reader.FindMVA("cv").folds
    assert len(folds) == 2
    event = np.array([[0.1, 0.2, 0.3, 0.4]])
    method = reader.FindMVA("cv")
    events = Events(event, np.zeros((1, 1)), np.array([[0.004]]), np.zeros(1, int), np.ones(1))
    expected = folds[0].mva(events)[0]
    assert method.mva(events)[0] == expected


def test_an_averaging_cross_validation_answers_the_mean_of_its_folds(session):
    _deterministic("Avg")
    method = _reader().FindMVA("cv")
    events = Events(
        np.array([[0.1, 0.2, 0.3, 0.4]]),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.zeros(1, int),
        np.ones(1),
    )
    answers = [fold.mva(events)[0] for fold in method.folds]
    assert method.mva(events)[0] == pytest.approx(np.mean(answers))


def test_a_stratified_split_puts_each_class_across_the_folds(session):
    cv = ROOT.TMVA.CrossValidation(
        "job", loader(), "!V:Silent:AnalysisType=Classification:SplitType=RandomStratified"
    )
    folds = cv._folds()
    classes = cv.loader.dataset().train.classes
    for members in folds:
        assert set(classes[members]) == {0, 1}


def test_a_regression_cross_validation_keeps_no_roc_of_its_folds(session):
    from test_tmva_frame_regression import regression_loader

    cv = ROOT.TMVA.CrossValidation("job", regression_loader(), "!V:Silent:AnalysisType=Regression")
    cv.BookMethod("LD", "LD", "")
    cv.Evaluate()
    first = cv.GetResults()[0]
    assert first.GetROCValues()[0] == 0.0 and first.results[0].roc is None
    assert os.path.exists("dataset/weights/job_LD.weights.xml")


def test_a_method_that_cannot_do_the_analysis_leaves_no_folds_to_read(session, capsys):
    from test_tmva_frame_regression import regression_loader

    cv = ROOT.TMVA.CrossValidation("job", regression_loader(), "!V:AnalysisType=Regression")
    cv.BookMethod("Likelihood", "Likelihood", "")
    with pytest.raises(TMVAError, match=r"Unable to open input weight file: .*_fold1"):
        cv.Evaluate()
    assert "not capable of handling regression with 1 targets" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("expression", "refusal"),
    [
        ("int([spec]", "is not a valid TFormula"),
        ("[nothing]", 'Spectator "nothing" not found'),
        ("int([numFolds])", "should be a non-negativeinteger"),
    ],
)
def test_a_split_expression_that_cannot_split_is_refused(session, expression, refusal):
    options = f"!V:Silent:SplitType=Deterministic:SplitExpr={expression}"
    cv = ROOT.TMVA.CrossValidation("job", loader(spectator=True), options)
    with pytest.raises(TMVAError, match=refusal):
        cv._folds()


def test_the_split_of_no_events_is_as_many_empty_folds(session):
    empty = Events(
        np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 0)), np.zeros(0, int), np.ones(0)
    )
    folds = cvresults.split_folds(empty, [], 3, "", True)
    assert [len(members) for members in folds] == [0, 0, 0]


def test_the_standard_deviation_of_one_fold_is_zero():
    result = cvresults.CrossValidationResult(1)
    assert result.GetROCAverage() == 0.0
    result.Fill(cvresults.CrossValidationFoldResult(0, roc_integral=0.75))
    assert result.GetROCStandardDeviation() == 0.0 and result.GetROCAverage() == 0.75


def test_a_randomly_split_cross_validation_cannot_be_read_back(session):
    cv = ROOT.TMVA.CrossValidation("job", loader(), "!V:Silent:AnalysisType=Classification")
    cv.BookMethod("LD", "LD", "")
    cv.Evaluate()
    with pytest.raises(TMVAError, match="XML reading only for deterministic splitting"):
        _reader()


def _unbooked(options: str) -> MethodCrossValidation:
    info = DataSetInfo("dataset")
    return MethodCrossValidation("job", "cv", info, options)


def test_a_cross_validation_method_without_its_folds_cannot_say_which_answers(session):
    method = _unbooked("NumFolds=2:EncapsulatedMethodName=LD")
    event = Events(
        np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 0)), np.zeros(1, int), np.ones(1)
    )
    assert method.fold_file(0) == "job_LD_fold1.weights.xml"
    with pytest.raises(TMVAError, match="XML reading only for deterministic splitting"):
        method.mva(event)
    with pytest.raises(TMVAError, match="which carry their spectators"):
        method.evaluate(np.zeros((1, 1)))


def test_a_cross_validation_method_of_an_unknown_ensembling_is_refused(session):
    method = _unbooked("OutputEnsembling=Median")
    event = Events(
        np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 0)), np.zeros(1, int), np.ones(1)
    )
    with pytest.raises(TMVAError, match="Ensembling type Median unknown"):
        method.mva(event)


def test_a_spectator_named_twice_in_a_split_expression_is_read_once(session):
    data = loader(spectator=True)
    events = data.dataset().train
    twice = cvresults.split_folds(
        events, data.info.spectators, 2, "int(fabs([spec]+[spec])*50)%2", False
    )
    once = cvresults.split_folds(
        events, data.info.spectators, 2, SPLIT.replace("[NumFolds]", "2"), False
    )
    assert all(np.array_equal(a, b) for a, b in zip(twice, once, strict=False))
