"""What the Factory writes of the data: its output file's trees and plots, and weights' XML."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from tmvasupport import classify, gaussian, loader, session, weights
from xrdroot.tmva import TMVAError, hists
from xrdroot.tmva.output import Output
from xrdroot.tmva.plots import _range
from xrdroot.tmva.variables import VariableInfo
from xrdroot.tmva.varrank import correlation_ratio
from xrdroot.tmva.xmlfile import Node, child, children

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


def test_the_test_tree_holds_the_spectators_beside_the_variables(session):
    classify([(ROOT.TMVA.Types.kLD, "LD", "!H:!V")], data=loader(spectator=True))
    with xrdroot.open_root("out.root") as found:
        test = found["dataset/TestTree"]
        assert "spec" in test.keys() and "var4" in test.keys()


def _three_classes() -> Any:
    made = ROOT.TMVA.DataLoader("dataset")
    for variable in ("var1", "var2"):
        made.AddVariable(variable, "F")
    for number, name in enumerate(("Signal", "bg0", "bg1")):
        made.AddTree(gaussian(f"Tree{number}", number - 1.0, number + 1, 60), name)
    made.PrepareTrainingAndTestTree("", "SplitMode=Random:NormMode=NumEvents:!V")
    return made


def test_a_multiclass_output_is_a_leaf_for_every_class(session):
    classify(
        [(ROOT.TMVA.Types.kBDT, "BDTG", "!H:!V:NTrees=5:BoostType=Grad:MaxDepth=2")],
        options="!V:!Silent:AnalysisType=Multiclass",
        data=_three_classes(),
    )
    with xrdroot.open_root("out.root") as found:
        test = found["dataset/TestTree"]
        assert "BDTG.bg1" in test.keys() and test.num_entries == 90


def test_a_file_with_no_file_writes_nothing():
    silent = Output(None)
    silent.write("dataset", "text")
    silent.write_tree("dataset", "Tree", {"x": np.zeros(2)})
    assert silent.silent and silent.GetName() == ""


def test_a_directory_already_in_the_file_is_found_rather_than_made(session):
    target = ROOT.TFile.Open("f.root", "RECREATE")
    Output(target).directory("dataset/inner", "Inner")
    again = Output(target)
    assert again.directory("dataset/inner") is not None and again.exists("dataset/inner")
    assert again.GetName() == "f.root"
    target.Close()


def test_an_integer_variable_is_plotted_a_bin_per_value(session):
    made = ROOT.TMVA.DataLoader("dataset")
    made.AddVariable("n", "I")
    made.AddVariable("x", "F")
    rng = np.random.default_rng(1)
    for name, shift in (("Signal", 1), ("Background", -1)):
        count = rng.integers(0, 5, 60).astype(float)
        made.AddTree(_tree(name, {"n": count, "x": rng.normal(shift, 1, 60)}), name)
    made.PrepareTrainingAndTestTree("", "!V")
    classify([(ROOT.TMVA.Types.kLD, "LD", "!H:!V")], data=made)
    with xrdroot.open_root("out.root") as found:
        plot = found["dataset/InputVariables_Id/n__Signal_Id"]
        assert len(plot.axes[0]) == 5


@pytest.mark.parametrize("value, top", [(2.0, 2.2), (-2.0, -1.0), (0.0, 1.0)])
def test_a_constant_variable_is_plotted_over_a_range_of_its_own(value, top):
    nbins, low, high = _range(VariableInfo("c"), (value, 0.0, value, value))
    assert (nbins, low) == (40, value) and high == pytest.approx(top + (top - value) / 40)


def test_more_than_twenty_variables_have_no_scatter_plots(session):
    made = ROOT.TMVA.DataLoader("dataset")
    names = [f"v{index}" for index in range(21)]
    for name in names:
        made.AddVariable(name, "F")
    rng = np.random.default_rng(1)
    for name, shift in (("Signal", 1), ("Background", -1)):
        made.AddTree(_tree(name, {v: rng.normal(shift, 1, 30) for v in names}), name)
    made.PrepareTrainingAndTestTree("", "!V:!Correlations")
    classify([(ROOT.TMVA.Types.kLD, "LD", "!H:!V")], data=made)
    with xrdroot.open_root("out.root") as found:
        assert "CorrelationPlots" not in found["dataset/InputVariables_Id"].keys()


def test_a_histogram_is_booked_filled_and_read_by_its_bins():
    empty = hists.filled("e", "e", (4, 0.0, 4.0), [])
    assert hists.bins(empty).sum() == 0.0
    assert list(hists.centers(empty)) == [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    assert list(hists.find_bin(empty, np.array([-1.0, 0.2, 3.9, 9.0]))) == [0, 1, 4, 5]


def test_a_correlation_ratio_of_nothing_in_range_is_minus_one():
    assert correlation_ratio(np.zeros((302, 302)), (0.0, 1.0)) == -1.0


def test_an_element_with_text_and_children_writes_the_text_first():
    node = Node("Top", A=1).set("B", "two")
    node.text = "words & more"
    node.add("Inner")
    assert node.lines() == ['<Top A="1" B="two">', "  words &amp; more", "  <Inner/>", "</Top>"]
    parsed = ET.fromstring("\n".join(node.lines()))
    assert child(parsed, "Inner") is not None and len(children(parsed)) == 1


def _identity_weights() -> str:
    classify([(ROOT.TMVA.Types.kLD, "LD", "!H:!V:VarTransform=I")], output="")
    return weights("LD")


def test_an_identity_transformation_is_written_and_read_back(session):
    path = _identity_weights()
    transform = child(ET.parse(path).getroot(), "Transformations").find("Transform")
    assert transform.get("Name") == "Id"
    reader = ROOT.TMVA.Reader("!Color:Silent")
    for variable in ("var1", "var2", "var3", "var4"):
        reader.AddVariable(variable, np.zeros(1, dtype=np.float32))
    method = reader.BookMVA("LD", path)
    assert method.handler.transforms[0].params == [None]


def test_a_transformation_the_reader_does_not_know_is_refused(session):
    path = _identity_weights()
    with open(path) as source:
        text = source.read().replace('Name="Id"', 'Name="Bogus"')
    with open(path, "w") as out:
        out.write(text)
    reader = ROOT.TMVA.Reader("!Color:Silent")
    for variable in ("var1", "var2", "var3", "var4"):
        reader.AddVariable(variable, np.zeros(1, dtype=np.float32))
    with pytest.raises(TMVAError, match="Variable transform 'Bogus' unknown"):
        reader.BookMVA("LD", path)
