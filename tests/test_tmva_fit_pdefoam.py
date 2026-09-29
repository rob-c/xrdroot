"""PDE-Foam: its foams for every analysis, how they are grown, and the file they are kept in."""

from __future__ import annotations

import os

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from tmvafitsupport import MULTICLASS, REGRESSION, multiclass_loader, reader, regression_loader
from tmvasupport import classify, loader, session, weights
from xrdroot.tmva import TMVAError
from xrdroot.tmva.foam import Density, Foam, ranges
from xrdroot.tmva.foamcells import Cells
from xrdroot.tmva.log import Logger
from xrdroot.tmva.methods.pdefoamio import read_foams, write_foams

__all__ = ["session"]

FOAM = ROOT.TMVA.Types.kPDEFoam
#: A foam small enough to be quick.
SMALL = "!H:!V:nActiveCells=12:nSampl=60:nBin=5:Nmin=5"


def _signal_and_background(made, title: str) -> tuple[float, float]:
    high = made.EvaluateMVA([2.0, 2.0, 2.0, 2.0], title)
    low = made.EvaluateMVA([-2.0, -2.0, -2.0, -2.0], title)
    return high, low


@pytest.mark.parametrize(
    "options",
    [
        SMALL,
        f"{SMALL}:SigBgSeparate:MaxDepth=3",
        f"{SMALL}:UseYesNoCell:Nmin=0",
    ],
)
def test_a_classification_foam_is_grown_written_and_read_back_by_a_reader(session, options):
    classify([(FOAM, "PDEFoam", options)])
    assert os.path.exists(weights("PDEFoam").replace(".xml", "_foams.root"))
    high, low = _signal_and_background(reader("PDEFoam", silent=True), "PDEFoam")
    assert low <= high


def test_equal_numbers_of_events_are_not_warned_of(session, capsys):
    made = loader(split="SplitMode=Random:NormMode=EqualNumEvents:!V")
    classify([(FOAM, "PDEFoam", SMALL)], data=made)
    assert "only NormMode=EqualNumEvents ensures" not in capsys.readouterr().out


def test_a_regression_foam_gives_empty_cells_their_neighbours_value(session):
    options = "!H:!V:nActiveCells=40:nSampl=40:nBin=4:Nmin=0:VolFrac=0.02"
    classify([(FOAM, "PDEFoam", options)], options=REGRESSION, data=regression_loader())
    made = reader("PDEFoam", ("var1", "var2"), silent=True)
    for x, y in ((1.0, 1.0), (4.0, 4.0), (0.1, 4.9), (4.9, 0.1)):
        assert np.isfinite(made.EvaluateRegression(0, [x, y], "PDEFoam"))


def test_every_class_has_its_foam_and_their_outputs_are_a_softmax(session):
    classify([(FOAM, "PDEFoam", SMALL)], options=MULTICLASS, data=multiclass_loader(count=80))
    outputs = reader("PDEFoam", silent=True).EvaluateMulticlass([1.0, 1.0, 1.0, 1.0], "PDEFoam")
    assert len(outputs) == 3 and sum(outputs) == pytest.approx(1.0, abs=1e-5)


def test_a_foam_of_one_cell_ranks_every_variable_at_zero(session):
    factory = classify([(FOAM, "PDEFoam", "!H:!V:nActiveCells=1:nSampl=20")])
    title, ranked = factory.GetMethod("dataset", "PDEFoam").ranking()
    assert title == "Variable Importance" and [value for _, value in ranked] == [0.0] * 4


@pytest.mark.parametrize(
    ("options", "said"),
    [
        ("DTLogic=GiniIndex", "DTLogic=GiniIndex"),
        ("Kernel=Gauss", "Kernel=Gauss"),
        ("MultiTargetRegression", "MultiTargetRegression=True"),
    ],
)
def test_what_xrdroot_s_foam_does_not_have_is_refused(session, options, said):
    factory = ROOT.TMVA.Factory("job", "!V:Silent:AnalysisType=Classification")
    with pytest.raises(TMVAError, match=said):
        factory.BookMethod(loader(), FOAM, "PDEFoam", options)


def test_a_regression_of_two_targets_is_refused(session):
    made = regression_loader()
    made.AddTarget("var1+var2")
    factory = ROOT.TMVA.Factory("job", "!V:Silent:AnalysisType=Regression")
    with pytest.raises(TMVAError, match="multi-target regression"):
        factory.BookMethod(made, FOAM, "PDEFoam", SMALL)


def test_a_foam_file_xrdroot_did_not_write_is_refused(session):
    classify([(FOAM, "PDEFoam", SMALL)])
    with xrdroot.create(weights("PDEFoam").replace(".xml", "_foams.root")) as out:
        out.tree("Other", {"x": "f8"}).extend({"x": np.zeros(2)})
    with pytest.raises(TMVAError, match="is not one xrdroot wrote"):
        reader("PDEFoam", silent=True)


def test_a_tail_cut_of_every_event_leaves_the_whole_range():
    values = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]])
    lows, highs = ranges(values, 1.0)
    assert lows.tolist() == [0.0, 1.0] and highs.tolist() == [2.0, 5.0]


def test_a_cell_whose_best_cut_is_at_its_edge_is_not_cut(session, capsys):
    values = np.array([[0.2, 0.2], [0.8, 0.8]])
    density = Density(values, np.ones(2), np.array([0.1, 0.1]), "event")
    options = {"nActiveCells": 3, "nSampl": 10, "nBin": 5, "Nmin": 0, "MaxDepth": 0}
    foam = Foam("EdgeFoam", np.zeros(2), np.ones(2), density, options)
    foam.fill(None)
    foam.driv[0], foam.xdiv[0] = 1.0, 1.0
    assert foam.peek() == -1
    assert "no more candidate cells" in capsys.readouterr().out


def test_foams_written_beside_the_working_directory_are_read_back(session):
    cells = Cells(2, 1)
    cells.fill(None)
    write_foams("foams.root", [cells], ["OneFoam"])
    back = read_foams("foams.root", ["OneFoam"], 2, Logger("PDEFoam"))
    assert back[0].columns()["status"].tolist() == [1]
