"""What TMVA's tutorials needed of the rest of xrdroot: leaf lists, directories, trees, models."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from xrdroot.cint.runtime.streams import ostream
from xrdroot.rdf.models import is_model
from xrdroot.tmva.tools import CxxVector

from tmvasupport import session

__all__ = ["session"]


def test_a_leaf_list_column_is_one_branch_of_a_leaf_per_name(tmp_path):
    path = tmp_path / "leaves.root"
    with xrdroot.create(str(path)) as out:
        tree = out.tree("T", {"a": "f", "out": ("f", ("Signal", "bg0"))})
        tree.extend(
            {
                "a": np.arange(2, dtype=np.float32),
                "out": np.arange(4, dtype=np.float32).reshape(2, 2),
            }
        )
    with xrdroot.open_root(str(path)) as read:
        found = read["T"].arrays(["out.Signal", "out.bg0"])
    assert list(found["out.Signal"]) == [0, 2] and list(found["out.bg0"]) == [1, 3]


def test_a_leaf_list_of_bad_names_or_of_no_names_is_refused(tmp_path):
    with xrdroot.create(str(tmp_path / "bad.root")) as out:
        with pytest.raises(ValueError, match="leaf"):
            out.tree("T", {"out": ("f", ("a b",))})
        with pytest.raises(ValueError, match="values per entry"):
            out.tree("U", {"out": ("f", ())})


def test_the_root_session_changes_into_a_directory_of_an_open_file_by_its_path(session):
    made = ROOT.TFile("x.root", "RECREATE")
    made.mkdir("inner")
    assert ROOT.gROOT.cd("x.root:/inner") and ROOT.gDirectory.GetPath() == "x.root:/inner"
    assert ROOT.gROOT.cd("x.root:/") and ROOT.gDirectory.GetPath() == "x.root:/"
    assert not ROOT.gROOT.cd("other.root:/")
    made.Close()


def test_a_tree_with_no_name_is_not_written_when_its_file_closes(session):
    made = ROOT.TFile("t.root", "RECREATE")
    nameless = ROOT.TTree()
    value = np.zeros(1)
    nameless.Branch("x", value, "x/D")
    nameless.Fill()
    named = ROOT.TTree("kept", "kept")
    named.Branch("x", value, "x/D")
    named.Fill()
    made.Close()
    with xrdroot.open_root("t.root") as read:
        assert [key.name for key in read.all_keys()] == ["kept"]


def test_a_branch_declared_by_class_name_takes_the_object_after_it(session):
    tree = ROOT.TTree("t", "t")
    values = ROOT.std.vector["float"]([1.0, 2.0])
    tree.Branch("vars", "std::vector<float>", values)
    tree.Fill()
    assert tree.GetEntries() == 1


def test_a_braced_model_is_a_model_and_a_list_of_names_is_not():
    assert is_model(["h", "title", 10, 0.0, 1.0])
    assert not is_model(["x", "y"]) and not is_model(["h"])


def test_a_stream_writes_what_an_iterator_points_at():
    items = CxxVector([1.5, 2.5])
    assert ostream().format(items.begin()) == "1.5"
