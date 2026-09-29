"""The Category method: a method of its own for each region of the events, through the Factory."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from tmvasupport import VARIABLES, gaussian, loader, session, weights
from xrdroot.tmva import TMVAError
from xrdroot.tmva.dataset import DataSetInfo
from xrdroot.tmva.methods.category import MethodCategory, category_info

__all__ = ["session"]


def _categories(factory: object, data: object, name: str = "Category") -> object:
    category = factory.BookMethod(data, ROOT.TMVA.Types.kCategory, name, "")
    category.AddMethod("abs(var1)<=1", "var1:var2:var3:var4", ROOT.TMVA.Types.kFisher, "F", "")
    category.AddMethod("abs(var1)>1", "var2:var3:spec", "LD", "L", "VarTransform=N")
    return category


def _run(factory: object) -> None:
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()


def test_a_category_trains_a_method_per_region_and_ranks_their_variables(session, capsys):
    target = ROOT.TFile.Open("cat.root", "RECREATE")
    factory = ROOT.TMVA.Factory("job", target, "!V:!Silent:AnalysisType=Classification")
    category = _categories(factory, loader(spectator=True))
    category.AddMethod("abs(var1)>1", "", "KNN", "none", "nkNN=5")
    _run(factory)
    target.Close()
    printed = capsys.readouterr().out
    assert "Adding sub-classifier: Fisher::F" in printed
    assert "Train all sub-classifiers for Classification ..." in printed
    assert "Begin ranking of input variables..." in printed
    assert "No variable ranking supplied by classifier: none" in printed
    assert "Recreating sub-classifiers from XML-file" in printed
    assert "[F_dsi] : Evaluation of F on testing sample" in printed
    with xrdroot.open_root("cat.root") as found:
        assert "Category" in found["dataset/TestTree"].keys()
    assert [s.label for s in category.dsi.spectators][-3:] == [
        "Category_cat1",
        "Category_cat2",
        "Category_cat3",
    ]


def test_a_category_read_by_a_reader_answers_by_each_region_s_cut(session):
    factory = ROOT.TMVA.Factory("job", "!V:Silent:AnalysisType=Classification")
    _categories(factory, loader(spectator=True))
    _run(factory)
    reader = ROOT.TMVA.Reader("!Color:Silent")
    cells = [np.zeros(1, dtype=np.float32) for _ in VARIABLES]
    for variable, cell in zip(VARIABLES, cells):
        reader.AddVariable(variable, cell)
    reader.AddSpectator("spec := var1*2", np.zeros(1, dtype=np.float32))
    reader.BookMVA("cat", weights("Category"))
    inner = reader.EvaluateMVA([0.5, 0.1, 0.2, 0.3], "cat")
    outer = reader.EvaluateMVA([2.0, 0.1, 0.2, 0.3], "cat")
    assert inner != outer
    subs = reader.FindMVA("cat").subs
    assert [sub["method"].type_name for sub in subs] == ["Fisher", "LD"]


def test_an_event_in_no_region_is_answered_zero(session):
    factory = ROOT.TMVA.Factory("job", "!V:Silent:AnalysisType=Classification")
    category = factory.BookMethod(loader(), "Category", "Category", "")
    category.AddMethod("var1>0", "var1:var2", "Fisher", "F", "")
    _run(factory)
    reader = ROOT.TMVA.Reader(list(VARIABLES), "!Color:Silent")
    reader.BookMVA("cat", weights("Category"))
    assert reader.EvaluateMVA([-1.0, 0.1, 0.2, 0.3], "cat") == 0.0
    assert reader.EvaluateMVA([1.0, 0.1, 0.2, 0.3], "cat") != 0.0


def test_a_category_over_a_regression_trains_without_ranking(session, capsys):
    from test_tmva_frame_regression import regression_loader

    factory = ROOT.TMVA.Factory("job", "!V:!Silent:AnalysisType=Regression")
    category = factory.BookMethod(regression_loader(), "Category", "Category", "")
    category.AddMethod("var1<2.5", "var1:var2", "LD", "low", "")
    category.AddMethod("var1>=2.5", "var1:var2", "LD", "high", "")
    _run(factory)
    printed = capsys.readouterr().out
    assert "Train all sub-classifiers for Regression ..." in printed
    assert "Begin ranking" not in printed


def test_a_region_over_a_variable_the_loader_does_not_have_is_refused(session):
    factory = ROOT.TMVA.Factory("job", "!V:Silent:AnalysisType=Classification")
    category = factory.BookMethod(loader(), "Category", "Category", "")
    with pytest.raises(TMVAError, match="The variable var9 was not found"):
        category.AddMethod("var1>0", "var1:var9", "Fisher", "F", "")


def test_a_category_answers_events_not_values(session):
    method = MethodCategory("job", "Category", DataSetInfo("dataset"))
    with pytest.raises(TMVAError, match="Category answers events"):
        method.evaluate(np.zeros((1, 1)))


def test_a_category_made_by_hand_books_its_regions_without_a_loader(session):
    info = DataSetInfo("dataset")
    info.variables = loader().info.variables
    method = MethodCategory("job", "Category", info)
    made = method.AddMethod("var1>0", "", "Fisher", "F", "")
    assert made.loader is None and len(made.dsi.variables) == 4
    region = category_info(info, "var1>0", "var1", "R")
    assert [v.label for v in region.variables] == ["var1"]


def test_a_region_s_loader_builds_its_data_set_once_and_is_named_by_it(session):
    factory = ROOT.TMVA.Factory("job", "!V:Silent:AnalysisType=Classification")
    category = factory.BookMethod(loader(), "Category", "Category", "")
    made = category.AddMethod("var1>0", "var1:var2", "Fisher", "F", "")
    assert made.loader.GetName() == "F_dsi"
    assert made.loader.dataset() is made.loader.dataset()


def test_a_category_trained_without_an_output_writes_nothing_of_its_regions(session, capsys):
    factory = ROOT.TMVA.Factory("job", "!V:AnalysisType=Classification")
    category = factory.BookMethod(loader(), "Category", "Category", "")
    category.AddMethod("var1>0", "var1:var2", "Fisher", "L", "")
    category.output = None
    category.train(category.loader.dataset().train)
    assert "Train method: L for Classification" in capsys.readouterr().out


def test_a_likelihood_in_a_region_too_small_for_its_bins_books_one_bin(session):
    target = ROOT.TFile.Open("catlik.root", "RECREATE")
    factory = ROOT.TMVA.Factory("job", target, "Silent:AnalysisType=Classification")
    category = factory.BookMethod(loader(spectator=True), "Category", "Category", "")
    category.AddMethod("var1>0", "var1:var2", "Likelihood", "Lik", "")
    _run(factory)
    target.Close()
    assert 0.0 <= factory.GetROCIntegral("dataset", "Category") <= 1.0


def test_a_reader_decides_the_region_by_the_spectator_it_was_pointed_at(session):
    made = ROOT.TMVA.DataLoader("dataset")
    for variable in VARIABLES[:3]:
        made.AddVariable(variable, "F")
    made.AddSpectator("var4", "F")  # a branch, as eta is in TMVA's tutorial
    made.AddSignalTree(gaussian("TreeS", 1.0, 1))
    made.AddBackgroundTree(gaussian("TreeB", -1.0, 2))
    made.PrepareTrainingAndTestTree("", "", "SplitMode=Random:NormMode=NumEvents:!V")
    factory = ROOT.TMVA.Factory("job", "!V:Silent:AnalysisType=Classification")
    category = factory.BookMethod(made, "Category", "Category", "")
    category.AddMethod("var4<=0", "var1:var2", "Fisher", "low", "")
    category.AddMethod("var4>0", "var1:var2:var3", "LD", "high", "")
    _run(factory)
    reader = ROOT.TMVA.Reader("!Color:Silent")
    for variable in VARIABLES[:3]:
        reader.AddVariable(variable, np.full(1, 0.3, dtype=np.float32))
    spectator = np.zeros(1, dtype=np.float32)
    reader.AddSpectator("var4", spectator)
    reader.BookMVA("cat", weights("Category"))
    spectator[0] = -1.0
    low = reader.EvaluateMVA("cat")
    spectator[0] = 1.0
    assert reader.EvaluateMVA("cat") != low
