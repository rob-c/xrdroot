"""The DataLoader: its variables, trees, events one by one, weights, cuts and how it is split."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from tmvasupport import gaussian, loader, regression_tree, session
from xrdroot.tmva import TMVAError
from xrdroot.tmva.dataset import TESTING, TRAINING

__all__ = ["session"]


def _tree(name: str, columns: dict[str, Any]) -> Any:
    """A pyroot tree of float columns, filled entry by entry."""
    tree = ROOT.TTree(name, name)
    cells = {key: np.zeros(1, dtype=np.float32) for key in columns}
    for key, cell in cells.items():
        tree.Branch(key, cell, f"{key}/F")
    for row in range(len(next(iter(columns.values())))):
        for key, cell in cells.items():
            cell[0] = columns[key][row]
        tree.Fill()
    return tree


def _array_tree(name: str, count: int, seed: int, jagged: bool) -> Any:
    """A tree of an array branch ``arr`` of three values an entry - or, jagged, three or four."""
    rng = np.random.default_rng(seed)
    tree = ROOT.TTree(name, name)
    size, values = np.zeros(1, dtype=np.int32), np.zeros(4, dtype=np.float32)
    if jagged:
        tree.Branch("n", size, "n/I")
    tree.Branch("arr", values, "arr[n]/F" if jagged else "arr[3]/F")
    for row in range(count):
        size[0] = 3 + row % 2
        values[:] = rng.normal(0, 1, 4)
        tree.Fill()
    return tree


def test_a_variable_is_declared_by_its_type_or_by_its_title_unit_and_type(session):
    made = ROOT.TMVA.DataLoader("dataset")
    made.AddVariable("a", ord("F"))
    made.AddVariable("b", "I", -1.0, 1.0)
    made.AddVariable("c", "Title of c", "cm", "F", 0, 5)
    made.AddVariable("d", "Title of d")
    made.AddVariable("e")
    infos = made.GetDataSetInfo().GetVariableInfos()
    assert [info.GetVarType() for info in infos] == ["F", "I", "F", "F", "F"]
    assert (infos[1].GetMin(), infos[1].GetMax()) == (-1.0, 1.0)
    assert (infos[2].GetTitle(), infos[2].GetUnit()) == ("Title of c", "cm")
    assert made.GetName() == "dataset" and made.DefaultDataSetInfo() is made.GetDataSetInfo()


@pytest.mark.parametrize("jagged", [False, True])
def test_an_array_of_variables_is_one_variable_per_element(session, jagged):
    made = ROOT.TMVA.DataLoader("dataset")
    made.AddVariablesArray("arr", 3)
    made.AddSignalTree(_array_tree("S", 40, 1, jagged))
    made.AddBackgroundTree(_array_tree("B", 40, 2, jagged))
    made.PrepareTrainingAndTestTree("", "SplitMode=Block:!V")
    events = made.dataset()
    info = made.GetDataSetInfo()
    assert [v.GetInternalName() for v in info.variables] == ["arr[0]", "arr[1]", "arr[2]"]
    assert info.GetVariableInfo(2).GetTitle() == "arr[2]" and events.train.values.shape[1] == 3


def test_a_tree_that_does_not_exist_is_refused(session):
    with pytest.raises(TMVAError, match="Tree does not exist"):
        ROOT.TMVA.DataLoader("dataset").AddTree(None, "Signal")


@pytest.mark.parametrize(
    "kind, wanted",
    [("Training", TRAINING), ("Test", TESTING), ("Training and Testing", 2), (TESTING, TESTING)],
)
def test_a_tree_is_for_training_or_testing_as_its_type_says(session, kind, wanted):
    made = ROOT.TMVA.DataLoader("dataset")
    made.AddTree(gaussian("TreeS", 1.0, 1, 10), "Signal", 1.0, "", kind)
    assert made.inputs[0].tree_type == wanted


def test_a_tree_type_neither_training_nor_testing_is_refused(session):
    made = ROOT.TMVA.DataLoader("dataset")
    with pytest.raises(TMVAError, match="cannot interpret tree type"):
        made.AddTree(gaussian("TreeS", 1.0, 1, 10), "Signal", 1.0, "", "Validation")


def test_a_cut_on_a_tree_keeps_only_the_entries_that_pass_it(session):
    made = ROOT.TMVA.DataLoader("dataset")
    made.AddTree(gaussian("TreeS", 1.0, 1, 50), "Signal", 2.0, "var1>0")
    assert made.inputs[0].tree.GetEntries() < 50 and made.inputs[0].weight == 2.0


def test_a_regression_reads_its_target_and_spectator_from_its_tree(session):
    made = ROOT.TMVA.DataLoader("dataset")
    made.AddVariable("var1", "F")
    made.AddVariable("var2", "F")
    made.AddTarget("fvalue")
    made.AddRegressionTarget("half := fvalue/2", "Half", "u")
    made.AddSpectator("var1*3")
    made.AddRegressionTree(regression_tree(count=60))
    made.SetWeightExpression("1", "Regression")
    made.PrepareTrainingAndTestTree("", "SplitMode=Random:NormMode=None:!V")
    assert not made.has_dataset()
    events = made.dataset()
    assert made.has_dataset() and events.train.targets.shape == (30, 2)
    assert np.allclose(events.train.targets[:, 1], events.train.targets[:, 0] / 2, rtol=1e-6)
    info = made.GetDataSetInfo()
    assert (info.GetNTargets(), info.GetNSpectators()) == (2, 1)


def test_input_trees_are_given_together_with_their_weights(session):
    made = ROOT.TMVA.DataLoader("dataset")
    made.SetInputTrees(gaussian("S", 1.0, 1, 10), gaussian("B", -1.0, 2, 10), 2.0, 3.0)
    made.SetSignalTree(gaussian("S2", 1.0, 3, 10))
    made.SetBackgroundTree(gaussian("B2", 1.0, 4, 10))
    assert [item.weight for item in made.inputs] == [2.0, 3.0, 1.0, 1.0]
    assert [item.class_name for item in made.inputs] == ["Signal", "Background"] * 2


def _by_events(made: Any, count: int = 30) -> None:
    rng = np.random.default_rng(3)
    for _ in range(count):
        made.AddSignalTrainingEvent(list(rng.normal(1, 1, 2)), 1.0)
        made.AddBackgroundTrainingEvent(list(rng.normal(-1, 1, 2)), 2.0)
        made.AddSignalTestEvent(list(rng.normal(1, 1, 2)))
        made.AddBackgroundTestEvent(list(rng.normal(-1, 1, 2)))


def test_events_added_one_by_one_are_trees_of_their_class_and_purpose(session):
    made = ROOT.TMVA.DataLoader("dataset")
    made.AddVariable("x", "F")
    made.AddVariable("y", "F")
    _by_events(made)
    trees = made._event_trees
    assert trees[("Signal", TRAINING)].GetName() == "TrainAssignTree_Signal"
    assert trees[("Background", TESTING)].GetEntries() == len(trees[("Background", TESTING)]) == 30
    made.PrepareTrainingAndTestTree("", "NormMode=None:!V")
    events = made.dataset()
    assert (events.GetNTrainingEvents(), events.GetNTestEvents()) == (60, 60)
    assert events.GetNEvents() == 60 and sorted(set(events.train.weights)) == [1.0, 2.0]


def test_events_of_any_class_are_added_for_training_or_testing_by_name(session):
    made = ROOT.TMVA.DataLoader("dataset")
    made.AddVariable("x", "F")
    for row in range(20):
        made.AddTrainingEvent("One", [row])
        made.AddTestEvent("Two", [row], 0.5)
        made.AddTrainingEvent("Two", [-row])
        made.AddTestEvent("One", [-row])
    events = made.dataset()
    assert len(events.train) == 40 and set(events.test.weights) == {0.5, 1.0}
    assert made.GetDataSetInfo().GetSignalClassIndex() == 0


def test_a_weight_expression_for_no_class_is_the_signals_and_the_backgrounds(session):
    made = loader()
    made.SetWeightExpression("weight*2")
    info = made.GetDataSetInfo()
    assert [info.GetClassInfo(n).GetWeight() for n in range(2)] == ["weight*2", "weight*2"]
    made.SetSignalWeightExpression("1")
    assert info.GetClassInfo("Signal").GetWeight() == "1"


def test_a_cut_is_set_or_added_to_what_a_class_has(session):
    made = loader()
    made.SetCut("var1>0", "Signal")
    made.AddCut("var2>0", "Signal")
    made.AddCut("")
    made.SetCut("var3<5")
    info = made.GetDataSetInfo()
    assert info.GetClassInfo("Signal").GetCut() == "var3<5"
    made.AddCut("var4>-5", "Background")
    assert info.GetClassInfo(1).GetCut() == "(var3<5)&&(var4>-5)"
    assert info.GetClassInfo(7) is None and info.GetClassInfo("None") is None


def _plain(name: str = "dataset") -> Any:
    made = ROOT.TMVA.DataLoader(name)
    for variable in ("var1", "var2"):
        made.AddVariable(variable, "F")
    made.AddSignalTree(gaussian("TreeS", 1.0, 1, 60))
    made.AddBackgroundTree(gaussian("TreeB", -1.0, 2, 60))
    return made


def test_numbers_of_training_and_testing_events_stand_for_their_options(session):
    made = _plain()
    made.PrepareTrainingAndTestTree("var1>-9", 20, 30)
    options = made.info.split_options
    assert "nTrain_Signal=20" in options and "nTest_Background=30" in options
    events = made.dataset()
    assert (len(events.train), len(events.test)) == (40, 60)


def test_four_numbers_are_each_classs_training_and_testing_events(session):
    made = _plain()
    made.PrepareTrainingAndTestTree("", 10, 20, 30, 40, "!V:NormMode=None")
    events = made.dataset()
    assert (len(events.train), len(events.test)) == (30, 70)
    assert made.info.split_options.endswith("nTest_Background=40:!V:NormMode=None")


def test_one_number_trains_on_that_many_and_tests_on_the_rest(session):
    made = _plain()
    made.PrepareTrainingAndTestTree("", 25)
    assert "nTest_Signal=-1" in made.info.split_options


def test_a_signal_and_a_background_cut_are_each_their_classs(session, capsys):
    made = _plain()
    made.PrepareTrainingAndTestTree("var1>0", "var1<0", "!V")
    events = made.dataset()
    printed = capsys.readouterr().out
    assert 'Signal     requirement: "var1>0"' in printed
    assert np.all(events.train.values[events.train.classes == 0, 0] > 0)
    assert np.all(events.train.values[events.train.classes == 1, 0] < 0)


def test_one_cut_and_the_options_name_every_class_first(session, capsys):
    _plain().PrepareTrainingAndTestTree("", "!V")
    assert "Class index : 1  name : Background" in capsys.readouterr().out


def test_options_alone_split_as_they_say(session):
    made = _plain()
    made.PrepareTrainingAndTestTree("SplitMode=Block")
    assert made.info.split_options == "SplitMode=Block"
    made.PrepareTrainingAndTestTree()
    assert made.info.split_options == ""


def test_an_event_with_a_value_that_is_not_finite_is_left_out_and_said_so(session, capsys):
    made = ROOT.TMVA.DataLoader("dataset")
    made.AddVariable("x", "F")
    values = np.arange(20.0)
    values[3] = np.nan
    made.AddSignalTree(_tree("S", {"x": values}))
    made.AddBackgroundTree(_tree("B", {"x": -values - 1}))
    made.PrepareTrainingAndTestTree("", "!V")
    events = made.dataset()
    assert len(events.train) + len(events.test) == 36
    assert "NaN or +-inf in Event 3" in capsys.readouterr().out


def test_a_constant_variable_is_warned_about(session, capsys):
    made = ROOT.TMVA.DataLoader("dataset")
    made.AddVariable("x", "F")
    made.AddVariable("c", "F")
    made.AddSignalTree(_tree("S", {"x": np.arange(10.0), "c": np.ones(10)}))
    made.AddBackgroundTree(_tree("B", {"x": -np.arange(10.0), "c": np.ones(10)}))
    made.dataset()
    assert "Variable c is constant" in capsys.readouterr().out


def test_a_constant_variable_is_plotted_and_scattered_without_failing(session):
    made = ROOT.TMVA.DataLoader("dataset")
    made.AddVariable("x", "F")
    made.AddVariable("c", "F")
    made.AddSignalTree(_tree("S", {"x": np.arange(10.0), "c": np.ones(10)}))
    made.AddBackgroundTree(_tree("B", {"x": -np.arange(10.0) - 1, "c": np.ones(10)}))
    made.PrepareTrainingAndTestTree("", "SplitMode=Alternate:!V")
    output = ROOT.TFile.Open("constant.root", "RECREATE")
    factory = ROOT.TMVA.Factory("job", output, "!V:Silent:AnalysisType=Classification")
    factory.BookMethod(made, "Cuts", "Cuts", "FitMethod=MC:SampleSize=50")
    factory.TrainAllMethods()
    output.Close()
    assert Path("dataset/weights/job_Cuts.weights.xml").exists()
