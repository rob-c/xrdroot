"""A directory's ``d["name"]``, kept once found; a ``TDirectoryFile`` in no file; the version."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT


def test_a_directory_item_is_found_by_name_and_kept_once_found():
    top = ROOT.TDirectory("items_top", "top")
    histogram = ROOT.TH1F("items_h", "h", 2, 0, 1)
    histogram.SetDirectory(top)
    assert "_cached_items" not in top.__dict__
    assert top["items_h"] is histogram and top._cached_items["items_h"] is histogram
    assert top["items_h"] is histogram
    with pytest.raises(KeyError, match="has no object 'missing'"):
        top["missing"]


def test_a_directory_file_in_no_file_makes_its_directories_in_memory():
    top = ROOT.TDirectoryFile("loose_top", "top")
    below = top.mkdir("below")
    deeper = below.mkdir("deeper")
    histogram = ROOT.TH1F("loose_h", "h", 2, 0, 1)
    histogram.SetDirectory(deeper)
    assert top["below"]["deeper"]["loose_h"] is histogram
    assert top.Get("below/deeper/loose_h") is histogram and top._writable() is None


def test_the_namespace_has_the_version_groot_has():
    assert ROOT.__version__ == ROOT.gROOT.GetVersion() and "/" not in ROOT.__version__


def test_a_frames_description_shows_as_its_text():
    description = ROOT.RDataFrame(1).Describe()
    assert repr(description) == description.AsString() == str(description)


def test_a_histogram_prints_its_name_title_and_bins():
    histogram = ROOT.TH1F("printed_h", "a title", 10, 0, 1)
    assert str(histogram) == "Name: printed_h Title: a title NbinsX: 10"
    assert repr(histogram).startswith("<cppyy.gbl.TH1F object at 0x")


def test_rmdir_removes_a_directory_and_everything_that_was_inside_it():
    top = ROOT.TDirectory("rmdir_top", "top")
    below = top.mkdir("below")
    deeper = below.mkdir("deeper")
    ROOT.TH1F("rmdir_h", "h", 2, 0, 1).SetDirectory(deeper)
    assert top.rmdir("below") is None
    assert top.GetList().GetSize() == 0 and top.Get("below") is None


def test_rmdir_removes_a_key_that_is_not_a_directory_as_readily_as_one():
    top = ROOT.TDirectory("rmdir_h_top", "top")
    kept = top.mkdir("kept")
    ROOT.TH1F("rmdir_plain", "h", 2, 0, 1).SetDirectory(top)
    top.rmdir("rmdir_plain")
    assert [obj.GetName() for obj in top.GetList()] == ["kept"] and kept.GetMother() is top


def test_rmdir_of_a_name_this_directory_does_not_hold_does_nothing_at_all():
    top = ROOT.TDirectory("rmdir_missing_top", "top")
    below = top.mkdir("below")
    top.rmdir("nowhere")
    top.rmdir("")
    assert top.Get("below") is below


def test_rmdir_looks_in_this_directory_only_so_a_path_or_a_dotdot_finds_nothing():
    top = ROOT.TDirectory("rmdir_path_top", "top")
    below = top.mkdir("below")
    below.mkdir("deeper")
    top.rmdir("below/deeper")
    top.rmdir("..")
    assert top.Get("below") is below and below.Get("deeper") is not None


def test_a_directory_removed_while_it_is_current_leaves_its_mother_current():
    top = ROOT.TDirectory("rmdir_cwd_top", "top")
    below = top.mkdir("below")
    with ROOT.TDirectory.TContext(top):
        below.cd()
        top.rmdir("below")
        assert ROOT.gDirectory == top


def test_a_histogram_booked_with_no_directory_current_is_kept_nowhere():
    """``TDirectory::TContext ctx{nullptr}`` leaves no directory current, and ROOT then books a
    histogram without registering it - hist017's and the parallel fill tutorials' first line."""
    before = ROOT.gDirectory
    with ROOT.TDirectory.TContext(None):
        nowhere = ROOT.TH1D("booked_nowhere", "nowhere", 4, 0.0, 1.0)
        nowhere.Fill(0.5)
        assert repr(ROOT.gDirectory) == "None"  # gDirectory answers as whatever is current
        assert nowhere.GetDirectory() is None
    assert ROOT.gDirectory is before
    assert before.Get("booked_nowhere") is None and nowhere.GetEntries() == 1
