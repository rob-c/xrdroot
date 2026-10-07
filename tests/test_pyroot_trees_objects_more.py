"""The edges of branches of objects: printing a collection, reading between fills,
pointers ROOT fills with a vector or an object of its own making, and draws' values.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.cint.runtime import Cell
from xrdroot.pyroot.trees.addresses import Vector, address_of
from xrdroot.pyroot.trees.layout import BranchInfo
from xrdroot.pyroot.trees.printing import element_lines


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def _four_vectors(entries):
    v, many = ROOT.Math.XYZTVector(), ROOT.std.vector["ROOT::Math::XYZTVector"]()
    tree = ROOT.TTree("T", "objects")
    tree.Branch("lv", "ROOT::Math::XYZTVector", v)
    tree.Branch("lvs", "std::vector<ROOT::Math::XYZTVector>", Cell(many))
    for i in range(entries):
        v.SetCoordinates(i, 0, 0, i)
        many.clear()
        many.push_back(ROOT.Math.XYZTVector(1, 2, 3, i))
        tree.Fill()
    return tree


def test_a_collection_and_an_object_member_print_as_branches_themselves(capsys):
    out = ROOT.TFile("lv.root", "RECREATE")
    tree = _four_vectors(3)
    tree.Write()
    tree.Print()
    printed = capsys.readouterr().out
    assert "*Br    0 :fCoordinates :" in printed and "One basket in memory" in printed
    assert "*Br    5 :lvs       : Int_t lvs_" in printed
    assert "*Br    6 :lvs.fCoordinates.fX : Double_t fX[lvs_]" in printed
    tree.Print("lvs*")
    assert "fCoordinates :" not in capsys.readouterr().out
    out.Close()


def test_a_split_heading_names_its_title_when_it_has_one_of_its_own():
    leaf = BranchInfo("x", "x", "", [], 1, 0, 0, 0, 32000)
    top = BranchInfo("event", "the event", "Event", [], 1, 0, 0, 0, 32000, 0, (leaf,), -2, 0)
    lines, after = element_lines(top, 0)
    assert lines[0] == f"*Branch  :{'event':<9} : {'the event':<54} *" and after == 1


def test_what_is_filled_after_the_tree_was_read_is_kept_with_what_went_before():
    tree = _four_vectors(2)
    assert tree.Draw("fCoordinates.fX", "", "goff") == 2  # read back: two entries sealed
    repr(tree._store.slots["lv"])
    for _ in range(2):
        tree.Fill()
    assert tree.Draw("lvs.fCoordinates.fT", "", "goff") == 4
    assert list(tree.GetV1()) == [0.0, 1.0, 1.0, 1.0]


def test_a_draws_values_are_each_expression_for_every_entry_it_kept():
    x, y = np.zeros(1), np.zeros(1)
    tree = ROOT.TTree("t", "t")
    tree.Branch("x", x, "x/D")
    tree.Branch("y", y, "y/D")
    for i in range(5):
        x[0], y[0] = i, -i
        tree.Fill()
    assert tree.Draw("x:y:2*x", "x > 1", "goff") == 3
    assert (list(tree.GetV1()), list(tree.GetV3())) == ([2.0, 3.0, 4.0], [4.0, 6.0, 8.0])
    assert tree.GetV2() is tree.GetV2() and tree.GetV4() is None  # worked out once


def test_a_null_pointer_to_a_vector_is_given_a_vector_to_read_into():
    out = ROOT.TFile("v.root", "RECREATE")
    vpx = ROOT.std.vector["float"]()
    tree = ROOT.TTree("t", "t")
    tree.Branch("vpx", vpx)
    for i in range(3):
        vpx.assign([0.5] * i)
        tree.Fill()
    tree.Write()
    out.Close()
    back = ROOT.TFile("v.root").Get("t")
    pointer = Cell(0)
    back.SetBranchAddress("vpx", pointer)
    back.GetEntry(2)
    assert pointer.value.size() == 2 and pointer.value.capacity() >= 2


def test_a_pointer_to_a_vector_is_the_vector_it_points_at():
    vector = ROOT.std.vector["float"]()
    assert isinstance(address_of(Cell(vector)), Vector)
    with pytest.raises(TypeError, match="a tree reads and fills only what"):
        address_of(object())


def test_a_vector_of_objects_makes_room_as_it_grows():
    many = ROOT.std.vector["ROOT::Math::XYZTVector"]()
    many.reserve(10)
    many.push_back(ROOT.Math.XYZTVector())
    assert (many.capacity(), many.size()) == (1, 1)
