"""``T->Branch("v3", &v)`` of an object, filled, written, printed and read back.

Each kind of object a tutorial hands a tree - a macro's class, GenVector's
four-vector and a vector of them, a ``TLorentzVector``, histograms and a
``TClonesArray`` of lines - is filled entry by entry, written, and read
back through ``SetBranchAddress`` into the object a pointer is made to
point at, as ROOT reads it - or through ``TTreeReaderValue<T>``, whose
``Get()`` and ``operator->`` are the object the branch holds.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pycint.runtime import Cell
from xrdroot.pycint.runtime.root import DECLARED
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


class Holder:
    """A macro's class, as ``roottest``'s ``ArrayHolderT<int>`` is translated: a vector and a
    fixed array of numbers, and a method of its own that a macro calls through the reader."""

    _cxx_layout_ = ("Holder", (), (("m_v", "std::vector<double>", "", ()),
                                   ("m_a", "int", "", (3,))))  # fmt: skip

    def __init__(self):
        self.m_v = ROOT.std.vector["double"]()
        self.m_a = np.zeros(3, "i4")

    def ToString(self) -> str:
        return "-".join([*(str(int(a)) for a in self.m_a), *(str(float(v)) for v in self.m_v)])


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


def test_bronch_is_roots_older_spelling_of_branch_for_an_object() -> None:
    """``tree->Bronch("v", "Vector3", &v, bufsize, split)``: the class named before the object,
    which is the branch ``Branch`` makes from the same arguments - roottest's
    ``io/bigevent`` declares its super branch that way."""
    from xrdroot.pyroot.trees import TTree

    def filled(branch: str) -> TTree:
        held = Cell(Vector3())
        tree = TTree("t", "")
        made = getattr(tree, branch)("v", "Vector3", held, 16000, 1)
        assert made is not None and made.GetName() == "v"
        for step in range(3):
            held.value.fX, held.value.fY, held.value.fZ = step, step * 2.0, step * 3.0
            tree.Fill()
        return tree

    through_bronch, through_branch = filled("Bronch"), filled("Branch")
    assert [branch.GetName() for branch in through_bronch.GetListOfBranches()] == [
        branch.GetName() for branch in through_branch.GetListOfBranches()]
    assert through_bronch.GetEntries() == 3


def test_a_reader_value_of_a_class_branch_is_the_object_and_runs_its_methods(monkeypatch):
    """``TTreeReaderValue<ArrayHolder> r(reader, "h")``, then ``r->ToString()``: ROOT hands the
    macro the object the branch holds, so ``operator->`` reaches the class's own method -
    roottest's ``io/stdarray/colWiseRead_21`` compares that text with a fixed reference."""
    held = Cell(Holder())

    def fill(i):
        held.value.m_a[:] = [i, i + 1, i + 2]
        held.value.m_v.assign([i + 0.5])

    out, _tree = _filled("holder", lambda t: t.Branch("h", held, 32000, 1), fill, entries=3)
    out.Close()
    monkeypatch.setitem(DECLARED, "Holder", Holder)
    back = ROOT.TFile("holder.root").Get("T")
    reader = ROOT.TTreeReader(back)
    value = ROOT.TTreeReaderValue["Holder"](reader, "h")
    seen = []
    while reader.Next():
        obj = value.__deref__()
        # The translation writes r->ToString() as r.ToString(): the reader's unknown name is
        # the object's, and a name of the reader's own - Get - is still the reader's.
        seen.append((type(obj).__name__, obj.m_a.tolist(), list(obj.m_v), value.ToString()))
    assert seen == [("Holder", [0, 1, 2], [0.5], "0-1-2-0.5"),
                    ("Holder", [1, 2, 3], [1.5], "1-2-3-1.5"),
                    ("Holder", [2, 3, 4], [2.5], "2-3-4-2.5")]  # fmt: skip
    assert isinstance(value.Get(), Holder) and value.Get().m_a.tolist() == [2, 3, 4]


def test_a_reader_value_of_a_class_never_declared_here_gives_the_members_read(monkeypatch):
    """A file is read without the class's declaration as readily: the reader then hands back
    the members ROOT's own description gave, as it always did, and a private name is still
    the reader's own rather than the object's."""
    held = Cell(Holder())
    out, _tree = _filled("undeclared", lambda t: t.Branch("h", held, 32000, 1),
                         lambda i: None, entries=1)  # fmt: skip
    out.Close()
    monkeypatch.delitem(DECLARED, "Holder", raising=False)
    back = ROOT.TFile("undeclared.root").Get("T")
    reader = ROOT.TTreeReader(back)
    value = ROOT.TTreeReaderValue["Holder"](reader, "h")
    assert reader.Next()
    members = value.__deref__()
    assert isinstance(members, dict) and members["m_a"].tolist() == [0, 0, 0]
    with pytest.raises(AttributeError, match="has no attribute '_private'"):
        value._private  # noqa: B018 - a name of the reader's own is never the object's


def test_a_vector_of_vectors_is_written_whole_with_the_counts_of_both_levels():
    """ROOT lays a ``std::vector`` of containers out whole - the count of the outer one and
    each inner one's own - since a split branch of this writer counts one level; the file
    gives both counts back, entry for entry, to the vector read into."""
    outer = ROOT.std.vector["vector<double>"]()
    held = Cell(outer)

    def fill(i):
        outer.clear()
        for j in range(i + 1):
            inner = ROOT.std.vector["double"]()
            inner.push_back(float(j))
            inner.push_back(float(j) + 0.5)
            outer.push_back(inner)

    out, tree = _filled("vecvec", lambda t: t.Branch("vv", held), fill, entries=3)
    branch = tree.GetBranch("vv")
    assert branch.GetClassName() == "vector<vector<double> >"
    assert [leaf.GetName() for leaf in branch.GetListOfLeaves()] == ["vv"]
    out.Close()
    back = ROOT.TFile("vecvec.root").Get("T")
    read = ROOT.std.vector["vector<double>"]()
    assert back.SetBranchAddress("vv", read) == 0
    seen = []
    for i in range(back.GetEntries()):
        back.GetEntry(i)
        seen.append([list(inner) for inner in read])
    assert seen == [[[0.0, 0.5]], [[0.0, 0.5], [1.0, 1.5]],
                    [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]]]  # fmt: skip


def test_a_collection_of_objects_whose_own_width_is_unknown_is_still_refused():
    """The write of a collection whole needs the class its elements are, spelled as ROOT
    spells it; a vector of a class this writer does not write whole has no such name, and
    a branch laying out one count cannot say what its elements hold."""
    held = Cell(ROOT.std.vector["HalfKnown"]())
    tree = ROOT.TTree("t", "t")
    with pytest.raises(UnsupportedFeatureError, match="carry sizes of their own"):
        tree.Branch("hv", held)
