"""``TFileMerger``: ROOT's ``hadd``, from the API the merging tutorial uses."""

from __future__ import annotations

import pytest

import xrdroot
import xrdroot.pyroot as ROOT


@pytest.fixture
def two_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for index in range(2):
        histogram = ROOT.TH1F(f"h{index}", f"h{index}", 10, 0.0, 1.0)
        for _ in range(3 + index):
            histogram.Fill(0.5)
        written = ROOT.TFile(f"tomerge{index}.root", "RECREATE")
        histogram.Write()
        written.Close()
    return tmp_path


def test_the_merging_tutorials_flag_combination_merges_every_key(two_files):
    with ROOT.TFileMerger(False) as merger:
        assert merger.OutputFile("merged.root") is True
        assert merger.GetOutputFile().GetTitle() == "merged.root"
        for index in range(2):
            assert merger.AddFile(f"tomerge{index}.root") is True
        mode = ROOT.TFileMerger.kAll | ROOT.TFileMerger.kIncremental
        assert merger.PartialMerge(mode) is True
        merger.Reset()

    merged = xrdroot.open_root("merged.root")
    assert sorted(merged.keys()) == ["h0", "h1"]
    assert merged["h0"].values().sum() == 3.0
    assert merged["h1"].values().sum() == 4.0


def test_an_open_file_can_be_added_by_its_own_object(two_files):
    merger = ROOT.TFileMerger()
    merger.OutputFile("merged.root")
    opened = ROOT.TFile("tomerge0.root")
    assert merger.AddFile(opened) is True
    opened.Close()
    assert merger.Merge() is True
    assert xrdroot.open_root("merged.root").keys() == ["h0"]


def test_a_merger_without_an_output_or_without_inputs_merges_nothing():
    merger = ROOT.TFileMerger()
    assert merger.GetOutputFile() is None
    assert merger.PartialMerge() is False
    assert merger.Merge() is False
    merger.OutputFile("nowhere.root")
    assert merger.Merge() is False
    assert "0 input(s)" in repr(merger)
    # the output object answers with its name as ROOT's TFile does, and says so in repr
    named = ROOT.TFileMergerOutput("out.root")
    assert named.GetName() == "out.root"
    assert "out.root" in repr(named)


def test_the_flags_are_kept_and_a_merge_without_trees_leaves_them_out(two_files):
    merger = ROOT.TFileMerger()
    merger.OutputFile("merged.root")
    merger.SetMergeOptions(ROOT.TFileMerger.kNoTrees)
    assert merger.GetMergeOptions() == ROOT.TFileMerger.kNoTrees
    merger.SetFastMethod(True)
    merger.SetCompressionAlgorithm(0)
    merger.SetMsgLevel(0)
    merger.AddFileImpl("tomerge0.root")
    assert merger.PartialMerge(ROOT.TFileMerger.kNoTrees) is True
    # a merger that is left without merging merges on the way out of the with
    other = ROOT.TFileMerger(False)
    other.OutputFile("other.root")
    other.AddFile("tomerge1.root")
    with other:
        pass
    assert xrdroot.open_root("other.root").keys() == ["h1"]


def test_an_object_without_a_name_is_added_as_it_is(two_files):
    class Opened:
        pass

    merger = ROOT.TFileMerger()
    merger.OutputFile("merged.root")
    assert merger.AddFile(Opened()) is True
    assert "1 input(s)" in repr(merger)
