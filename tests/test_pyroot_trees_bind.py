"""What a branch of objects is read into: members of one object, or the object made again.

A pointer may come null, for an object of the branch's class to be made -
which needs the class declared - or already pointing at one; the object
itself may be handed over instead of a pointer to it.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.cint.runtime import Cell
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.trees.bind import _class, _member
from xrdroot.pyroot.trees.methods import _arguments
from xrdroot.tmva.dataset import Labels


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


class Point:
    _cxx_layout_ = ("Point", (), (("fX", "double", "", ()), ("lambda", "int", "", ())))

    def __init__(self):
        self.fX, self.lambda_ = 0.0, 0


def test_each_member_is_read_into_its_attribute_or_its_coordinate():
    point = Point()
    member = _member(point, "lambda")
    member.put(4)
    assert (point.lambda_, member.get()) == (4, 4)
    vector = ROOT.Math.XYZTVector(1, 2, 3, 4)
    coordinate = _member(vector, "fCoordinates.fZ")
    coordinate.put(7)
    assert (coordinate.get(), vector.Pz(), vector.E()) == (7.0, 7.0, 4.0)


def test_an_object_handed_over_whole_is_filled_in_place():
    out = ROOT.TFile("p.root", "RECREATE")
    tree = ROOT.TTree("T", "points")
    point = Point()
    tree.Branch("p", Cell(point), 32000, 1)
    for i in range(3):
        point.fX, point.lambda_ = i * 1.5, i
        tree.Fill()
    tree.Write()
    out.Close()
    back = ROOT.TFile("p.root").Get("T")
    into = Point()
    back.SetBranchAddress("p", into)
    back.GetEntry(2)
    assert (into.fX, into.lambda_) == (3.0, 2)
    pointed = Cell(Point())
    back.SetBranchAddress("p", pointed)
    back.GetEntry(1)
    assert pointed.value.fX == 1.5
    with pytest.raises(UnsupportedFeatureError, match="no declaration of Point"):
        back.SetBranchAddress("p", Cell(None))


def test_a_whole_object_handed_over_itself_is_filled_where_it_is():
    out = ROOT.TFile("t.root", "RECREATE")
    tree = ROOT.TTree("T", "vectors")
    v = ROOT.TLorentzVector()
    tree.Branch("tlv", "TLorentzVector", v, 32000, 0)
    for i in range(2):
        v.SetPxPyPzE(i, 0, 0, 1)
        tree.Fill()
    assert tree.Draw("tlv.Px()", "", "goff") == 2  # read back between fills
    v.SetPxPyPzE(5, 0, 0, 1)
    tree.Fill()
    tree.Write()
    out.Close()
    back = ROOT.TFile("t.root").Get("T")
    into = ROOT.TLorentzVector()
    back.SetBranchAddress("tlv", into)
    back.GetEntry(2)
    assert into.Px() == 5.0


def test_a_class_no_one_declared_cannot_be_made():
    assert _class("ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >") is (
        ROOT.Math.XYZTVector
    )
    with pytest.raises(UnsupportedFeatureError, match="run the macro that declares it"):
        _class("Nobody")


def test_a_method_is_called_with_the_numbers_and_strings_it_was_written_with():
    assert _arguments(" 3, 'x' , 2.5 ") == (3, "x", 2.5) and _arguments("") == ()


def test_a_list_of_labels_says_its_size_as_a_vector_does():
    assert Labels(["a", "b"]).size() == 2
