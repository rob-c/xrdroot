"""``T->Branch("v3", &v)`` of an object, filled, written, printed and read back.

Each kind of object a tutorial hands a tree - a macro's class, GenVector's
four-vector and a vector of them, a ``TLorentzVector``, histograms and a
``TClonesArray`` of lines - is filled entry by entry, written, and read
back through ``SetBranchAddress`` into the object a pointer is made to
point at, as ROOT reads it.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.cint.runtime import Cell
from xrdroot.cint.runtime.root import DECLARED
from xrdroot.wobjects import IGNORED


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


class Vector3:
    """A macro's class, as the translation of ``tree122_vector3.C`` makes it."""

    _cxx_layout_ = ("Vector3", (), (("fX", "Double_t", "", ()), ("fY", "Double_t", "", ()),
                                    ("fZ", "Double_t", "", ())))  # fmt: skip

    def __init__(self):
        self.fX = self.fY = self.fZ = 0.0


def _filled(name, branch, fill, entries=4):
    """A tree in a file of its own, given one branch and filled ``entries`` times."""
    out = ROOT.TFile(f"{name}.root", "RECREATE")
    tree = ROOT.TTree("T", "objects")
    branch(tree)
    for i in range(entries):
        fill(i)
        tree.Fill()
    tree.Write()
    return out, tree


def test_a_macros_class_is_split_member_by_member_and_read_back_as_one(capsys, monkeypatch):
    v = Cell(Vector3())

    def fill(i):
        v.value.fX, v.value.fY, v.value.fZ = i, 2.0 * i, -1.0

    out, tree = _filled("v3", lambda t: t.Branch("v3", v, 32000, 1), fill)
    tree.Print()
    printed = capsys.readouterr().out
    assert "*Branch  :v3" in printed and "BranchElement (see below)" in printed
    assert "*Br    1 :fY        : Double_t" in printed
    assert [b.GetName() for b in tree.GetBranch("v3").GetListOfBranches()] == ["fX", "fY", "fZ"]
    out.Close()
    monkeypatch.setitem(DECLARED, "Vector3", Vector3)
    back = ROOT.TFile("v3.root").Get("T")
    read = Cell(0)
    back.SetBranchAddress("v3", read)
    back.GetEntry(3)
    assert (read.value.fX, read.value.fY, read.value.fZ) == (3.0, 6.0, -1.0)
    back.GetBranch("fY").GetEntry(1)
    assert (read.value.fX, read.value.fY) == (3.0, 2.0)  # one branch read: one member
    assert back.GetBranch("fY").GetSplitLevel() == 0 and back.GetBranch("v3").GetSplitLevel()


def test_a_four_vector_and_a_vector_of_them_are_split_and_read_back():
    v, many = ROOT.Math.XYZTVector(), ROOT.std.vector["ROOT::Math::XYZTVector"]()

    def branches(tree):
        tree.Branch("lv", "ROOT::Math::XYZTVector", v)
        tree.Branch("lvs", "std::vector<ROOT::Math::XYZTVector>", Cell(many))

    def fill(i):
        v.SetCoordinates(i, i, i, 10 + i)
        many.clear()
        for k in range(i):
            many.push_back(ROOT.Math.XYZTVector(k, 0, 0, 1))

    out, _tree = _filled("lv", branches, fill)
    out.Close()
    back = ROOT.TFile("lv.root").Get("T")
    one, held = Cell(None), Cell(None)
    back.SetBranchAddress("lv", one)
    back.SetBranchAddress("lvs", held)
    back.GetEntry(2)
    assert (one.value.E(), held.value.size(), held.value[1].Px()) == (12.0, 2, 1.0)
    back.GetEntry(3)
    assert held.value.size() == 3


def test_a_lorentz_vector_is_never_split_and_ignores_its_tobject_when_told(capsys):
    v = Cell(ROOT.TLorentzVector())
    ROOT.TLorentzVector.Class().IgnoreTObjectStreamer()
    assert ROOT.TLorentzVector.Class().CanIgnoreTObjectStreamer()
    try:
        out, _tree = _filled(
            "tlv",
            lambda t: t.Branch("tlv", "TLorentzVector", v, 16000, 2),
            lambda i: v.value.SetPxPyPzE(i, -i, 0, 5 + i),
        )
        out.Close()
    finally:
        ROOT.TLorentzVector.Class().IgnoreTObjectStreamer(False)
    assert "TLorentzVector cannot be split" in capsys.readouterr().err
    assert "TLorentzVector" not in IGNORED
    back = ROOT.TFile("tlv.root").Get("T")
    read = Cell(None)
    back.SetBranchAddress("tlv", read)
    back.GetEntry(2)
    assert (read.value.Px(), read.value.Py(), read.value.E()) == (2.0, -2.0, 7.0)
