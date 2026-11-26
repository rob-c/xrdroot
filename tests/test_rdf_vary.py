"""``Vary``: systematic variations of a column, written as a branch each by ``Snapshot``."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT


def _frame():
    """A frame with one column and its variation: nominal 1.01, down 1.005, up 1.02."""
    frame = ROOT.RDataFrame(3).Define("pt", "ROOT::RVec<float>(2, 41.)")
    frame = frame.Define("scale", "1+0.01*(pt-40)")
    return frame.Vary(
        "scale",
        "ROOT::VecOps::RVec<ROOT::VecOps::RVec<float>>({1+0.005*(pt-40), 1+0.02*(pt-40)})",
        ["down", "up"],
        "unc",
    )


def _options(include: bool):
    options = ROOT.RDF.RSnapshotOptions("RECREATE", 1, 0, 0, 99, False)
    options.fIncludeVariations = include
    return options


def test_a_snapshot_asked_for_variations_writes_a_branch_for_each_tag(tmp_path):
    written = tmp_path / "varied.root"
    _frame().Snapshot("Events", str(written), ["scale"], _options(include=True))
    with ROOT.TFile.Open(str(written)) as found:
        tree = found.Get("Events")
        assert [branch.GetName() for branch in tree.GetListOfBranches()] == [
            "scale",
            "scale:unc:down",
            "scale:unc:up",
        ]
        assert tree.GetEntries() == 3
    again = ROOT.RDataFrame("Events", str(written))
    # The branch holds ``float``s - C++'s ``RVec<float>`` - so the values are compared as such.
    for column, nominal in (("scale", 1.01), ("scale:unc:down", 1.005), ("scale:unc:up", 1.02)):
        for row in again.Take(column):
            assert list(row) == pytest.approx([nominal, nominal], rel=1e-6)


def test_a_snapshot_not_asked_for_variations_writes_the_nominal_column_only(tmp_path):
    written = tmp_path / "nominal.root"
    _frame().Snapshot("Events", str(written), ["scale"], _options(include=False))
    with ROOT.TFile.Open(str(written)) as found:
        names = [branch.GetName() for branch in found.Get("Events").GetListOfBranches()]
    assert names == ["scale"]


def test_a_column_with_no_variation_is_written_named_and_nothing_more(tmp_path):
    """A Snapshot of several columns varies only the one ``Vary`` was given, and writes the rest
    as they are."""
    frame = _frame().Define("other", "pt[0]")
    written = tmp_path / "mixed.root"
    frame.Snapshot("Events", str(written), ["other", "scale"], _options(include=True))
    with ROOT.TFile.Open(str(written)) as found:
        names = [branch.GetName() for branch in found.Get("Events").GetListOfBranches()]
    assert names == ["other", "scale", "scale:unc:down", "scale:unc:up"]


def test_a_variation_with_no_name_is_a_branch_called_column_and_tag(tmp_path):
    """ROOT's ``<column>:<tag>``, the spelling its own ``VariationsFor`` results are named by."""
    frame = ROOT.RDataFrame(2).Define("x", "1").Vary("x", "ROOT::RVecI{-2, 2}", ["down", "up"])
    written = tmp_path / "unnamed.root"
    frame.Snapshot("Events", str(written), ["x"], _options(include=True))
    with ROOT.TFile.Open(str(written)) as found:
        names = [branch.GetName() for branch in found.Get("Events").GetListOfBranches()]
    assert names == ["x", "x:down", "x:up"]
    again = ROOT.RDataFrame("Events", str(written))
    assert list(again.Take("x")) == [1, 1] and list(again.Take("x:up")) == [2, 2]


def test_n_variations_tags_them_by_number_as_root_does():
    frame = ROOT.RDataFrame(1).Define("x", "1").Vary("x", "ROOT::RVecI{-1, 0, 2}", nVariations=3)
    taken = frame.Take("x").GetValue()  # the nominal is what the column had
    assert list(taken) == [1]
    assert frame._inner._varied["x"].tags == ("0", "1", "2")


def test_vary_refuses_a_column_that_is_not_there_and_a_call_with_no_tags():
    frame = ROOT.RDataFrame(1).Define("x", "1")
    with pytest.raises(ValueError, match="it has none called 'y': Define it before varying it"):
        frame.Vary("y", "ROOT::RVecI{-1, 2}", ["down", "up"])
    with pytest.raises(ValueError, match="needs the tags of its variations"):
        frame.Vary("x", "ROOT::RVecI{-1, 2}")
