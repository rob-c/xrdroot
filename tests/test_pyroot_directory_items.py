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
