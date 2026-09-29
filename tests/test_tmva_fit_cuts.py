"""The rectangular cut optimisation: its fitters, its variable properties, and its weight file."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from tmvafitsupport import SPLIT
from tmvafitsupport import reader as fit_reader
from tmvasupport import classify, gaussian, loader, session
from xrdroot.tmva import TMVAError
from xrdroot.tmva.cutsfit import CutTable, Sample

__all__ = ["session"]

CUTS = ROOT.TMVA.Types.kCuts
#: A Monte Carlo fit small enough to be quick.
MC = "!H:!V:FitMethod=MC:SampleSize=400"
#: A genetic fit small enough to be quick.
GA = "!H:!V:FitMethod=GA:PopSize=20:Steps=3:Cycles=1"


def _flipped():
    """A loader whose signal sits below its background, so that FSmart keeps the lower edge open."""
    made = ROOT.TMVA.DataLoader("dataset")
    for variable in ("var1", "var2", "var3", "var4"):
        made.AddVariable(variable, "F")
    made.AddSignalTree(gaussian("TreeS", -1.0, 1))
    made.AddBackgroundTree(gaussian("TreeB", 1.0, 2))
    made.PrepareTrainingAndTestTree("", "", SPLIT)
    return made


def test_monte_carlo_cuts_are_trained_evaluated_and_read_back_by_a_reader(session, capsys):
    classify([(CUTS, "CutsMC", MC + ":VarProp[1]=FMin:VarProp[2]=FMax")])
    printed = capsys.readouterr().out
    assert 'Use optimization method: "Monte Carlo"' in printed
    assert "Cut values for requested signal efficiency" in printed
    reader = fit_reader("CutsMC")
    assert "sample of MC events" in capsys.readouterr().out
    assert reader.EvaluateMVA([0.0, 0.0, 0.0, 0.0], "CutsMC") == 0.0
    assert reader.EvaluateMVA([5.0, 5.0, 5.0, 5.0], "CutsMC", 0.9) in (0.0, 1.0)
    method = reader.FindCutsMVA("CutsMC")
    low, high = ROOT.std.vector("double")(), ROOT.std.vector("double")()
    assert method.GetCuts(0.5, low, high) == 0.5 and len(low) == 4
    assert method.GetCuts(0.5) == 0.5 and method.GetInputVar(0) == "var1"


def test_genetic_cuts_with_smart_properties_both_ways_and_limited_ranges(session, capsys):
    options = GA + ":VarProp=FSmart:CutRangeMin[0]=-2:CutRangeMax[0]=2"
    classify([(CUTS, "CutsGA", options)])
    assert 'Use "FSmart" cuts' in capsys.readouterr().out
    classify([(CUTS, "CutsGA", GA + ":VarProp=FSmart")], data=_flipped())
    assert 'Use optimization method: "Genetic Algorithm"' in capsys.readouterr().out


def test_cuts_on_integer_variables_draw_from_discrete_intervals(session):
    made = ROOT.TMVA.DataLoader("dataset")
    for variable in ("var1", "var2", "var3", "var4"):
        made.AddVariable(f"int({variable}*2)", "I")
    made.AddSignalTree(gaussian("TreeS", 1.0, 1))
    made.AddBackgroundTree(gaussian("TreeB", -1.0, 2))
    made.PrepareTrainingAndTestTree("", "", SPLIT)
    factory = classify([(CUTS, "CutsI", MC)], data=made)
    assert factory.GetMethod("dataset", "CutsI") is not None


@pytest.mark.parametrize(
    ("transform", "said"),
    [
        ("D", "[var1]"),
        ("N", "var1_[transformed]"),
        ("N,D", "var1 [transformed]"),
    ],
)
def test_cuts_on_transformed_variables_say_what_the_cuts_are_on(session, capsys, transform, said):
    classify([(CUTS, "CutsT", MC + f":VarTransform={transform}")])
    printed = capsys.readouterr().out
    assert said in printed
    assert "Transformation applied to input variables : None" not in printed


def test_a_fit_method_xrdroot_does_not_have_is_refused_by_name(session):
    factory = ROOT.TMVA.Factory("job", "!V:Silent:AnalysisType=Classification")
    with pytest.raises(TMVAError, match="FitMethod=SA"):
        factory.BookMethod(loader(), CUTS, "CutsSA", "FitMethod=SA")


def test_a_table_keeps_the_first_box_of_least_background_in_each_bin_and_ignores_the_rest():
    table = CutTable(nbins=4, nvar=1)
    box = np.array([[0.0], [1.0], [2.0]])
    table.offer_batch(np.array([0.3, 0.3, 1.0]), np.array([0.5, 0.2, 0.1]), box, box + 1)
    assert table.effb[1] == 0.2 and table.lower[1, 0] == 1.0
    table.offer_batch(np.array([0.3]), np.array([0.4]), box[:1], box[:1] + 1)
    assert table.effb[1] == 0.2
    table.offer_batch(np.array([1.0]), np.array([0.0]), box[:1], box[:1] + 1)
    assert list(table.effb) == [-0.1, 0.2, -0.1, -0.1]


def test_the_efficiencies_of_a_class_without_weight_are_zero():
    sample = Sample(np.array([[0.5], [1.5]]), np.array([True, True]), np.array([1.0, 1.0]))
    effs, effb = sample.efficiencies(np.array([[0.0]]), np.array([[1.0]]))
    assert effs.tolist() == [0.5] and effb.tolist() == [0.0]
