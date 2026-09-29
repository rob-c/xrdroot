"""The Factory over a classification: its options, its queries and what it writes of each method."""

from __future__ import annotations

import os

import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from tmvasupport import classify, loader, session, weights
from xrdroot.tmva import TMVAError

__all__ = ["session"]

#: A factory that says nothing and writes no file.
QUIET = "!V:Silent:AnalysisType=Classification"


def test_transformations_correlations_and_a_method_transform_are_all_written(session, capsys):
    classify(
        [(ROOT.TMVA.Types.kLD, "LD", "!H:!V:VarTransform=D")],
        "!V:!Silent:Correlations:Transformations=I;D;P;G,D:AnalysisType=Classification",
    )
    printed = capsys.readouterr().out
    assert "Ranking input variables (method unspecific)" in printed
    with xrdroot.open_root("out.root") as found:
        names = found["dataset"].keys()
        assert any("InputVariables_Id" in name for name in names)
        assert any("InputVariables_Deco" in name for name in names)
        assert any("CorrelationMatrixS" in name for name in names)


def test_transformations_without_the_identity_rank_no_input_variables(session, capsys):
    classify([(ROOT.TMVA.Types.kLD, "LD", "")], "!V:!Silent:Transformations=D")
    assert "Ranking input variables (method unspecific)" not in capsys.readouterr().out
    with xrdroot.open_root("out.root") as found:
        assert any("InputVariables_Deco" in name for name in found["dataset"].keys())


def test_a_likelihood_with_output_densities_writes_its_probability_and_rarity(session, capsys):
    classify([(ROOT.TMVA.Types.kLikelihood, "Likelihood", "!H:!V:CreateMVAPdfs")])
    printed = capsys.readouterr().out
    assert "Also filling probability and rarity histograms" in printed
    assert "<CreateMVAPdfs> Separation from histogram (PDF)" in printed
    with xrdroot.open_root("out.root") as found:
        test = found["dataset/TestTree"]
        assert "prob_Likelihood" in test.keys()
        assert "prob_Likelihood" in found["dataset/TrainTree"].keys()
        method = found["dataset/Method_Likelihood/Likelihood"].keys()
        assert any("Rarity_S" in name for name in method)


def test_help_asked_for_is_printed_before_the_method_trains(session, capsys):
    classify([(ROOT.TMVA.Types.kLD, "LD", "H:!V")], output="")
    printed = capsys.readouterr().out
    assert "H e l p   f o r   M V A   m e t h o d   [ LD ]" in printed
    assert "Suppress this message" in printed


def test_a_factory_without_the_roc_prints_no_tables(session, capsys):
    classify([(ROOT.TMVA.Types.kLD, "LD", "")], "!V:!ROC:AnalysisType=Classification")
    assert "Evaluation results ranked" not in capsys.readouterr().out


def test_a_factory_without_persistence_writes_no_weight_file(session):
    factory = classify([(ROOT.TMVA.Types.kFisher, "Fisher", "")], QUIET + ":!ModelPersistence")
    assert not os.path.exists(weights("Fisher"))
    assert factory.GetMethod("dataset", "Fisher").weight_file == "job_Fisher.weights.xml"


def test_a_factory_with_nothing_booked_says_it_has_nothing_to_do(session, capsys):
    factory = ROOT.TMVA.Factory("job", "!V:AnalysisType=Classification")
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()
    printed = capsys.readouterr().out
    for what in ("train", "test", "evaluate"):
        assert f"...nothing found to {what}" in printed


def test_the_analysis_is_found_from_the_classes_when_it_is_left_automatic(session):
    factory = ROOT.TMVA.Factory("job", "!V:Silent")
    factory.BookMethod(loader(), ROOT.TMVA.Types.kLD, "LD")
    assert factory.analysis == 0


def test_a_loader_of_one_class_has_no_analysis_to_be_found(session):
    data = ROOT.TMVA.DataLoader("dataset")
    data.AddVariable("var1", "F")
    with pytest.raises(TMVAError, match="No analysis type for 0 classes"):
        ROOT.TMVA.Factory("job", "!V:Silent").BookMethod(data, "LD", "LD")


def test_a_method_booked_by_a_type_number_out_of_range_is_refused_by_that_number(session):
    factory = ROOT.TMVA.Factory("job", QUIET)
    with pytest.raises(xrdroot.UnsupportedFeatureError, match="999"):
        factory.BookMethod(loader(), 999, "odd")


def test_the_roc_integral_is_asked_of_the_test_and_the_training_sample(session):
    factory = classify([(ROOT.TMVA.Types.kLD, "LD", "")], QUIET)
    test = factory.GetROCIntegral("dataset", "LD")
    train = factory.GetROCIntegral(loader(), "LD", 0, 0)
    assert 0.5 < test < 1 and 0.5 < train < 1 and test != train


def test_a_roc_curve_is_a_graph_titled_unless_asked_not_to_be(session):
    factory = classify([(ROOT.TMVA.Types.kLD, "LD", "")], QUIET)
    titled = factory.GetROCCurve("dataset", "LD")
    assert titled.GetTitle() == "Signal efficiency vs. Background rejection"
    assert titled.GetName() == "LD"
    bare = factory.GetROCCurve("dataset", "LD", False, 0, 0)
    assert bare.GetTitle() != "Signal efficiency vs. Background rejection"
    assert bare.GetN() == titled.GetN()


def test_the_roc_curves_of_a_loader_are_drawn_on_one_canvas_with_a_legend(session):
    kinds = ("LD", "Fisher", "Likelihood", "LD", "Fisher", "LD", "Fisher", "LD", "Fisher", "LD")
    booked = [(kind, f"{kind}{i}", "") for i, kind in enumerate(kinds)]
    factory = classify(booked, QUIET, output="")
    canvas = factory.GetROCCurve(loader())
    assert canvas.GetName() == "dataset"
    assert canvas._tmva_legend.GetNRows() == len(kinds) + 1  # the header's row too
    assert factory.GetROCCurve("dataset", True).GetName() == "dataset"


def test_a_roc_query_of_a_method_never_booked_is_refused(session):
    factory = classify([(ROOT.TMVA.Types.kLD, "LD", "")], QUIET, output="")
    with pytest.raises(TMVAError, match="Method = BDT not found with Dataset = dataset"):
        factory.GetROCIntegral("dataset", "BDT")


def test_setting_verbose_and_deleting_the_methods_change_the_factory(session):
    factory = ROOT.TMVA.Factory("job", QUIET)
    factory.BookMethod(loader(), "LD", "LD")
    factory.SetVerbose()
    assert factory.options.flag("V", False)
    factory.SetVerbose(False)
    assert not factory.options.flag("V", True)
    factory.DeleteAllMethods()
    assert not factory.HasMethod("dataset", "LD")


def test_a_method_that_cannot_do_the_analysis_is_warned_of_and_not_booked(session, capsys):
    factory = ROOT.TMVA.Factory("job", "!V:AnalysisType=Multiclass")
    assert factory.BookMethod(loader(), "LD", "LD") is None
    printed = capsys.readouterr().out
    assert "not capable of handling multiclass classification with 2 classes" in printed


def test_a_method_with_no_ranking_and_a_recorded_history_is_said_so(session, capsys):
    target = ROOT.TFile.Open("out.root", "RECREATE")
    factory = ROOT.TMVA.Factory("job", target, "!V:AnalysisType=Classification")
    method = factory.BookMethod(loader(), "LD", "LD")
    method.ranking = lambda: None
    method.history = {"loss": [(1, 0.5), (2, 0.25), (3, 0.125)], "one": [(4, 1.0)], "none": []}
    factory.TrainAllMethods()
    target.Close()
    printed = capsys.readouterr().out
    assert "No variable ranking supplied by classifier: LD" in printed
    assert "TH1.Print Name  = TrainingHistory_LD_loss, Entries= 0, Total sum= 0.875" in printed
    assert "TrainingHistory_LD_one" in printed and "TrainingHistory_LD_none" not in printed


def test_a_regression_factory_refuses_to_give_a_roc_integral(session, capsys):
    factory = ROOT.TMVA.Factory("job", "!V:AnalysisType=Regression")
    assert factory.GetROCIntegral("dataset", "LD") == 0.0
    assert "Can only generate ROC integral" in capsys.readouterr().out


def test_training_a_regression_without_a_target_is_refused(session):
    factory = ROOT.TMVA.Factory("job", "!V:Silent:AnalysisType=Regression")
    factory.BookMethod(loader(), "LD", "LD")
    with pytest.raises(TMVAError, match="regression training without specifying a target"):
        factory.TrainAllMethods()


def test_training_a_classifier_of_fewer_than_two_classes_is_refused(session):
    factory = ROOT.TMVA.Factory("job", QUIET)
    data = loader()
    factory.BookMethod(data, "LD", "LD")
    data.info.classes = data.info.classes[:1]
    with pytest.raises(TMVAError, match="less than two classes"):
        factory.TrainAllMethods()
