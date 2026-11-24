"""``TTree::Branch(TCollection*)``: one branch for each object a collection holds.

ROOT's form names the branches after the collection and the object it reached, walking a
collection inside a collection with both names - which is how PyROOT's own pythonization
test (``bindings/pyroot/pythonizations/test/ttree_branch.py``) ends: a ``TList`` holding a
``TObjArray`` named ``myobjarray`` of ``one`` and ``two`` read back as ``entry.myobjarray_one``
and ``entry.myobjarray_two``.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.core.collections import TClonesArray, TList, TObjArray, TObjString
from xrdroot.pyroot.core.strings import TString
from xrdroot.pyroot.trees import TTree
from xrdroot.pyroot.trees.objects import BY_ROW, row_object


def _two_names() -> TList:
    """The list the pythonization test builds: a tokenized ``TString`` under a name."""
    split = TString("one:two").Tokenize(TString(":"))
    split.SetName("myobjarray")
    items = TList()
    items.Add(split)
    return items


def _strings() -> TList:
    """A list of the two strings themselves, with no collection between them and the tree."""
    items = TList()
    for text in ("one", "two"):
        items.Add(TObjString(text))
    return items


def _names(tree: TTree) -> list[str]:
    return [branch.GetName() for branch in tree.GetListOfBranches()]


def test_a_collection_of_objects_is_one_branch_each_named_for_the_collection_and_the_object():
    tree = TTree("t", "t")
    assert tree.Branch(_two_names()) == 2
    assert _names(tree) == ["myobjarray_one", "myobjarray_two"]
    assert [branch.GetClassName() for branch in tree.GetListOfBranches()] == ["TObjString"] * 2


def test_the_objects_a_collection_of_strings_wrote_read_back_as_the_objects_they_are(tmp_path):
    tree = TTree("t", "t")
    tree.Branch(_two_names())
    tree.Fill()
    path = tmp_path / "strings.root"
    handle = ROOT.TFile(str(path), "RECREATE")
    tree.SetDirectory(handle)
    tree.Write()
    handle.Close()
    read = ROOT.TFile(str(path))
    back = read.Get("t")
    back.GetEntry(0)
    assert back.myobjarray_one.GetName() == "one"
    assert back.myobjarray_two.GetString() == "two"


def test_the_name_a_collection_is_branched_under_comes_before_the_objects():
    tree = TTree("t", "t")  # ROOT's form is Branch(list, bufsize, splitlevel, name)
    assert tree.Branch(_two_names(), 32000, 99, "top") == 2
    assert _names(tree) == ["top_myobjarray_one", "top_myobjarray_two"]


def test_the_name_an_object_is_branched_under_is_the_one_it_was_given_and_an_underscore():
    tree = TTree("t", "t")
    tree.Branch(_strings(), 32000, 99, "top")
    assert _names(tree) == ["top_one", "top_two"]


def test_a_name_ending_in_an_underscore_is_not_given_another_one():
    tree = TTree("t", "t")
    tree.Branch(_strings(), 32000, 99, "top_")
    assert _names(tree) == ["top_one", "top_two"]


def test_a_split_level_above_ninety_nine_ends_the_branch_name_with_a_dot():
    tree = TTree("t", "t")
    tree.Branch(_two_names(), None, 101)
    assert _names(tree) == ["myobjarray_one.", "myobjarray_two."]


def test_a_collection_given_no_buffer_size_or_split_level_takes_roots_own():
    tree = TTree("t", "t")
    assert tree.Branch(_two_names(), None, None) == 2
    assert _names(tree) == ["myobjarray_one", "myobjarray_two"]


def test_a_collection_at_split_level_one_is_branched_whole_and_refused_by_name():
    tree = TTree("t", "t")
    with pytest.raises(UnsupportedFeatureError, match="holds a TObjArray"):
        tree.Branch(_two_names(), 32000, 1)


def test_branch_of_a_tclones_array_is_refused_as_root_refuses_it():
    tree = TTree("t", "t")
    with pytest.raises(UnsupportedFeatureError, match="cannot be called for a TClonesArray"):
        tree.Branch(TClonesArray("TObjString", 2))


def test_a_collection_branch_adds_to_a_tree_being_filled_as_every_branch_does():
    tree = TTree("t", "t")
    x = TObjArray()
    x.SetName("kept")
    x.Add(TObjString("one"))
    tree.Branch(x)
    tree.Fill()
    with pytest.raises(ValueError, match="already has 1 entries"):
        tree.Branch(x)


def test_a_row_of_a_tobjstring_makes_the_object_that_answers_its_name():
    made = row_object({"TObject": {"fUniqueID": 0, "fBits": 0}, "fString": "one"}, "TObjString")
    assert (made.GetName(), made.GetString(), str(made)) == ("one", "one", "one")
    assert set(BY_ROW) == {"TObjString"}


def test_a_row_of_a_class_no_declaration_makes_answers_nothing():
    assert row_object({"fString": "one"}, "TNothingOfOurs") is None
