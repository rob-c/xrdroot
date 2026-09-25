"""``SetBranchAddress`` and ``GetEntry``: a tree read an entry at a time, as ``tree1.C`` reads.

A file is written with :mod:`xrdroot.pyroot.trees` and read back through
:func:`~xrdroot.pyroot.trees.wrap`, which is what ``TFile::Get`` hands back,
and every value read into an address is compared with what was filled.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

import xrdroot
from xrdroot.pyroot.stl import std
from xrdroot.pyroot.trees import TTree, wrap
from xrdroot.pyroot.trees import batch as batching

ENTRIES = 30


class Cell:
    def __init__(self, value):
        self.value = value


def _filled() -> TTree:
    """The tree every test reads: numbers, a counted array, a vector, text, a leaf list."""
    px, n, arr = np.zeros(1, "f"), np.zeros(1, "i"), np.zeros(3, "d")
    v, s, rec = std.vector["int"](), std.string(), np.zeros(1, dtype=[("a", "i4"), ("b", "f8")])
    t = TTree("t", "for reading")
    t.Branch("px", px, "px/F")
    t.Branch("n", n, "n/I")
    t.Branch("arr", arr, "arr[n]/D")
    t.Branch("v", v)
    t.Branch("s", s)
    t.Branch("rec", rec, "a/I:b/D")
    for i in range(ENTRIES):
        px[0], n[0], arr[:] = i * 0.5, i % 4 if i % 4 < 4 else 0, [i, i + 1, i + 2]
        n[0] = min(n[0], 3)
        v.assign(range(i % 3))
        s.assign(f"e{i}")
        rec["a"], rec["b"] = i, -i
        t.Fill()
    return t


@pytest.fixture
def path(tmp_path):
    out = tmp_path / "read.root"
    with xrdroot.create(str(out)) as f:
        t = _filled()
        t.SetDirectory(f)
        t.Write()
    return str(out)


@pytest.fixture(params=["file", "memory"])
def tree(request, path):
    """The same tree read from its file, and while it is still being filled."""
    if request.param == "memory":
        yield _filled()
        return
    with xrdroot.open_root(path) as f:
        yield wrap(f["t"])


def test_get_entry_puts_each_entry_into_every_address_bound_to_a_branch(tree, monkeypatch):
    monkeypatch.setattr(batching, "STEP", 7)
    px, n, arr, v, s = ctypes.c_float(), Cell(0), np.zeros(3), std.vector["int"](), std.string()
    for name, address in [("px", px), ("n", n), ("arr", arr), ("v", v), ("s", s)]:
        assert tree.SetBranchAddress(name, address) == 0
    for i in range(tree.GetEntries()):
        assert tree.GetEntry(i) > 0
        count = min(i % 4, 3)
        assert px.value == i * 0.5 and n.value == count and tree.GetReadEntry() == i
        assert arr[:count].tolist() == [i, i + 1, i + 2][:count]
        assert list(v) == list(range(i % 3)) and str(s) == f"e{i}"
    assert tree.GetEntry(ENTRIES) == 0 and tree.GetEntry(-1) == 0


def test_a_leaf_list_branch_reads_into_the_members_of_what_it_is_given(tree):
    rec = np.zeros(1, dtype=[("a", "i4"), ("b", "f8")])
    one = Cell(0.0)
    if tree.GetBranch("rec") is not None:  # a file has its leaves as branches of their own
        assert tree.SetBranchAddress("rec", rec) == 0
        tree.GetEntry(4)
        assert rec["a"][0] == 4 and rec["b"][0] == -4
    assert tree.SetBranchAddress("b", one) == 0
    tree.GetEntry(5)
    assert one.value == -5


def test_an_address_holding_an_array_is_filled_where_it_is(tree):
    held = Cell(np.zeros(3))
    before = held.value
    text = bytearray(8)
    tree.SetBranchAddress("arr", held)
    tree.SetBranchAddress("s", text)
    tree.GetEntry(3)
    assert held.value is before and before.tolist() == [3, 4, 5]
    assert bytes(text).startswith(b"e3\\0")


def test_an_address_too_small_for_an_entry_is_refused_with_both_sizes(tree):
    tree.SetBranchAddress("arr", np.zeros(1))
    with pytest.raises(ValueError, match="holds 3 values and the array it is read into has room"):
        tree.GetEntry(3)


def test_an_unknown_branch_is_reported_as_root_reports_it(tree, capsys):
    assert tree.SetBranchAddress("nothing", np.zeros(1)) == -5
    assert "Error in <TTree::SetBranchAddress>: unknown branch -> nothing" in capsys.readouterr().err


def test_a_branch_switched_off_is_not_read_and_is_read_again_when_on(tree):
    px, n = np.zeros(1, "f"), np.zeros(1, "i")
    tree.SetBranchAddress("px", px)
    tree.SetBranchAddress("n", n)
    tree.SetBranchStatus("*", 0)
    tree.SetBranchStatus("px", 1)
    assert tree.GetBranchStatus("px") and not tree.GetBranchStatus("n")
    tree.GetEntry(2)
    assert px[0] == 1.0 and n[0] == 0
    tree.ResetBranchAddress(tree.GetBranch("px"))
    tree.ResetBranchAddress("n")
    tree.GetEntry(3)
    assert px[0] == 1.0
    tree.SetBranchAddress("px", px)
    tree.ResetBranchAddresses()
    tree.GetEntry(4)
    assert px[0] == 1.0


def test_pyroot_reads_a_branch_as_an_attribute_and_walks_the_tree_entry_by_entry(tree):
    seen = [(event.px, event.n, list(event.v), event.s) for event in tree]
    assert len(seen) == ENTRIES and seen[5] == (2.5, 1, [0, 1], "e5")
    assert list(tree.v) == [0, 1] and tree.arr.tolist() == [29]
    assert isinstance(tree.v, std.vector["int"]) or tree.GetBranch("nv") is not None
    assert tree.GetLineColor() == 1
    tree.SetLineColor(4)
    assert tree.GetLineColor() == 4
    with pytest.raises(AttributeError, match="xrdroot.pyroot's does not"):
        tree.NotAMethod  # noqa: B018
    with pytest.raises(AttributeError):
        tree._private  # noqa: B018


def test_branches_and_leaves_answer_roots_questions_about_themselves(tree):
    names = [branch.GetName() for branch in tree.GetListOfBranches()]
    assert names == ["px", "n", "arr", "v", "s", "rec"] or "nv" in names
    leaf = tree.GetLeaf("arr")
    assert leaf.GetTypeName() == "Double_t" and leaf.GetLeafCount().GetName() == "n"
    assert leaf.GetValue() == 0.0 and leaf.GetLenStatic() == 1 and not leaf.IsUnsigned()
    tree.GetEntry(6)
    assert leaf.GetLen() == leaf.GetNdata() == 2 and leaf.GetValue(1) == 7.0
    assert leaf.GetValue(5) == 0.0 and leaf.GetValueLong64() == 6
    assert leaf.GetValuePointer().tolist() == [6.0, 7.0] and leaf.GetBranch().GetName() == "arr"
    assert "Double_t" in repr(leaf) and tree.GetLeaf("px").GetLeafCount() is None
    assert tree.GetLeaf("s").GetValue() == 2.0 and tree.GetLeaf("s").GetLen() == 1
    assert tree.GetLeaf("nothing") is None and tree.GetLeaf("px", "px").GetName() == "px"
    assert tree.GetLeaf("nothing", "px") is None
    assert len(tree.GetListOfLeaves()) >= 7
    branch = tree.GetBranch("px")
    assert branch.GetEntries() == ENTRIES and branch.GetTree() is tree and branch.GetMother() is branch
    assert branch.GetListOfBranches().GetEntries() == 0 and branch.GetClassName() == ""
    assert branch.GetBasketSize() > 0 and branch.GetTotBytes() >= 0 and branch.GetZipBytes() >= 0
    assert branch.GetWriteBasket() >= 0 and branch.GetReadEntry() == 6 and "TBranch" in repr(branch)
    px = np.zeros(1, "f")
    branch.SetAddress(px)
    assert branch.GetAddress() is not None and branch.GetEntry(8) > 0 and px[0] == 4.0
    assert tree.GetBranch("nothing") is None


def test_a_vector_branch_says_the_class_it_holds(path):
    t = _filled()
    assert t.GetBranch("v").GetClassName() == "vector<int>"
    assert t.GetBranch("v").ClassName() == "TBranchElement"
    assert t.GetLeaf("v").ClassName() == "TLeafElement"


def test_load_tree_and_get_tree_are_the_tree_itself_for_a_tree(tree):
    assert tree.LoadTree(3) == 3 and tree.LoadTree(ENTRIES) == -2
    assert tree.GetTree() is tree and tree.GetTreeNumber() == 0
    assert tree.GetEntryNumber(4) == 4 and tree.GetEntryNumber(ENTRIES) == -1
    assert tree.GetEntries("n > 2") == 7 and tree.GetEntriesFast() == ENTRIES
    assert "entries" in repr(tree) and tree._xrd is not None
