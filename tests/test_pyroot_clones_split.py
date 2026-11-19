"""A ``TClonesArray`` split into a branch per member, as ``TTree::Branch`` splits one.

The top branch counts the objects each entry holds (``fType`` 3) and each
member of every object is a branch of its own (``fType`` 31) - the objects'
``TObject`` part split into ``fUniqueID`` and ``fBits``, a ``TString`` member a
row of strings. ``TTree::Print`` of such a tree is ROOT 6.40's, line for line,
and ROOT reads the file back object for object (checked with the oracle when
this was written); here it is read back by xrdroot both ways.
"""

from __future__ import annotations

import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.trees import objects
from xrdroot.wbranch import CloneMember, clones_members

#: ``TTree::Print`` of ROOT 6.40 for the tree :func:`_written` makes, its sizes and all.
PRINTED = """\
*Br    0 :N         : Int_t N_                                               *
*Entries :        3 : Total  Size=       3322 bytes  File Size  =         97 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.00     *
*............................................................................*
*Br    1 :N.fUniqueID : UInt_t fUniqueID[N_]                                 *
*Entries :        3 : Total  Size=        708 bytes  File Size  =        106 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.12     *
*............................................................................*
*Br    2 :N.fBits   : UInt_t fBits[N_]                                       *
*Entries :        3 : Total  Size=        688 bytes  File Size  =        102 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br    3 :N.fName   : TString fName[N_]                                      *
*Entries :        3 : Total  Size=        700 bytes  File Size  =        124 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.02     *
*............................................................................*
*Br    4 :N.fTitle  : TString fTitle[N_]                                     *
*Entries :        3 : Total  Size=        705 bytes  File Size  =        125 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.02     *
*............................................................................*
"""


def _written(path: str) -> None:
    """Three entries of one, two and three ``TNamed``, the branch split as ROOT splits it."""
    out = ROOT.TFile(path, "RECREATE")
    tree = ROOT.TTree("T", "T")
    held = ROOT.TClonesArray("TNamed")
    tree.Branch("N", held)
    for entry in range(3):
        held.Clear()
        for at in range(entry + 1):
            held.AddAt(ROOT.TNamed(f"N_{entry}_{at}", f"N_{entry}_{at}"), at)
        tree.Fill()
    tree.Write()
    tree.Print()
    out.Close()


def test_a_split_clones_array_is_printed_as_root_prints_it(tmp_path, capsys):
    _written(str(tmp_path / "clones.root"))
    assert PRINTED in capsys.readouterr().out


def test_each_member_reads_back_as_a_row_per_entry(tmp_path):
    path = tmp_path / "clones.root"
    _written(str(path))
    with xrdroot.open_root(path) as found:
        tree = found["T"]
        assert tree.keys() == ["N", "N.fUniqueID", "N.fBits", "N.fName", "N.fTitle"]
        assert tree.arrays(["N.fName"])["N.fName"][2] == ["N_2_0", "N_2_1", "N_2_2"]
        record = tree.branches["N.fBits"].record
        assert record.classname == "TObject"
        assert record.element[1:] == ("", 2417737773, 1, 1, 31, 15, 0)


def test_a_script_reads_the_objects_back_into_its_clones_array_or_a_new_one(tmp_path):
    path = str(tmp_path / "clones.root")
    _written(path)
    back = ROOT.TFile(path)
    tree = back.Get("T")
    mine = ROOT.TClonesArray("TNamed")
    tree.SetBranchAddress("N", mine)
    tree.GetEntry(1)
    assert [o.GetName() for o in mine] == ["N_1_0", "N_1_1"] and mine[1].GetTitle() == "N_1_1"
    back.Close()
    back = ROOT.TFile(path)
    tree = back.Get("T")
    cell = ROOT.MakeNullPointer(ROOT.TClonesArray) if hasattr(ROOT, "MakeNullPointer") else None
    holder = type("Cell", (), {"value": cell})()
    tree.SetBranchAddress("N", holder)
    tree.GetEntry(2)
    assert holder.value.GetClass().GetName() == "TNamed" and len(holder.value) == 3
    back.Close()


def test_long_names_go_out_with_four_bytes_of_length(tmp_path):
    path = str(tmp_path / "long.root")
    out = ROOT.TFile(path, "RECREATE")
    tree = ROOT.TTree("T", "T")
    held = ROOT.TClonesArray("TObjString")
    tree.Branch("S", held, 32000, 1)
    held.AddAt(ROOT.TObjString("x" * 300), 0)
    tree.Fill()
    tree.Write()
    out.Close()
    with xrdroot.open_root(path) as found:
        assert found["T"].arrays(["S.fString"])["S.fString"][0] == ["x" * 300]


def test_a_class_without_a_description_or_with_members_not_split_here_is_refused():
    with pytest.raises(UnsupportedFeatureError, match="carries a description of"):
        clones_members("MyMacroClass")
    with pytest.raises(UnsupportedFeatureError, match="TH1::fXaxis is a Object member"):
        clones_members("TH1F")
    names = [(m.owner, m.name) for m in clones_members("TLine")]
    assert ("TAttLine", "fLineColor") in names and ("TLine", "fX1") in names


def _cloned(tmp_path, name: str, entries: int = -1, cut: str | None = None) -> ROOT.TTree:
    """The tree of a written file, cloned or copied by a cut into a file of its own."""
    path = str(tmp_path / "clones.root")
    _written(path)
    source = ROOT.TFile(path)
    upstream = source.Get("T")
    clone = upstream.CopyTree(cut) if cut else upstream.CloneTree(entries)
    clone.SetName(name)
    source.Close()
    with xrdroot.create(str(tmp_path / "clone.root")) as out:
        clone.SetDirectory(out)
        clone.Write()
    return clone


def test_a_clone_hands_the_writer_one_branch_for_the_split_collection(tmp_path):
    """ROOT writes a split ``TClonesArray`` as a count and a branch per member, so a clone
    of one is that same branch again, not a column for each member of its own.

    A member's column is ``N.fName``, and no column of a tree may be named with a dot.
    """
    clone = _cloned(tmp_path, "C")
    assert clone.GetEntries() == 3
    assert list(clone._store.slots) == ["N"] and "ClonedSlot" in repr(clone._store.slots["N"])
    assert sorted(clone._store.columns()) == [
        "N",
        "N\x00N.fBits",
        "N\x00N.fName",
        "N\x00N.fTitle",
        "N\x00N.fUniqueID",
    ]
    with xrdroot.open_root(str(tmp_path / "clone.root")) as found:
        written = found["C"]
        assert written.keys() == ["N", "N.fUniqueID", "N.fBits", "N.fName", "N.fTitle"]
        assert written.arrays(["N.fName"])["N.fName"][2] == ["N_2_0", "N_2_1", "N_2_2"]
        assert written.branches["N"].record.element == ("", "TNamed", 3066088035, 4, 0, 3, -1, 3)
        bits = written.branches["N.fBits"].record.element
        assert bits == ("TNamed", "", 2417737773, 1, 1, 31, 15, 0)


def test_a_clone_filled_entry_by_entry_copies_the_objects_each_entry_holds(tmp_path):
    """``CloneTree(0)`` then ``GetEntry`` and ``Fill``: the clone reads its objects from the
    entry the source last read, as ROOT's clone shares its source's branch buffers."""
    path = str(tmp_path / "clones.root")
    _written(path)
    source = ROOT.TFile(path)
    upstream = source.Get("T")
    clone = upstream.CloneTree(0)
    assert upstream.GetEntry(2)
    assert clone.Fill()
    source.Close()
    with xrdroot.create(str(tmp_path / "filled.root")) as out:
        clone.SetDirectory(out)
        clone.Write()
    with xrdroot.open_root(str(tmp_path / "filled.root")) as found:
        assert found["T"].arrays(["N.fName"])["N.fName"] == [["N_2_0", "N_2_1", "N_2_2"]]


def test_a_copy_tree_of_a_split_collection_keeps_the_entries_the_cut_keeps(tmp_path):
    clone = _cloned(tmp_path, "C", cut="Entry$ % 2 == 0")
    assert clone.GetEntries() == 2
    with xrdroot.open_root(str(tmp_path / "clone.root")) as found:
        assert found["C"].arrays(["N.fName"])["N.fName"] == [["N_0_0"], ["N_2_0", "N_2_1", "N_2_2"]]


def test_a_clone_says_so_when_a_member_of_a_split_collection_is_not_copied(tmp_path):
    """A collection is one branch, so a member the clone does not read leaves it half written."""
    path = str(tmp_path / "clones.root")
    _written(path)
    source = ROOT.TFile(path)
    upstream = source.Get("T")
    upstream.SetBranchStatus("N.fName", 0)
    with pytest.raises(UnsupportedFeatureError, match="turn that branch back on"):
        upstream.CloneTree()


class _Pid:
    """The shape a macro's ``tref_test_pid`` gives the split: a name and one whole ``TRef``."""

    def __init__(self, name: str, ref: object) -> None:
        self.fName = name
        self.fRef = ROOT.TRef(ref)


def test_a_member_that_is_one_whole_object_a_row_is_that_object_and_not_a_number(monkeypatch):
    """ROOT splits ``tref_test_pid::fRef`` as a whole ``TRef`` for every object of the
    collection (``fType`` 31, ``fStreamerType`` 61), and a ``TRef`` holds no number to pack:
    its rows are the references themselves, one list per entry, which a column streams.

    The class is one the file describes rather than one of ROOT's own, so its members are
    spelled out here - ``clones_members`` reads them once the writer carries its description.
    """
    members = [
        CloneMember("TNamed", 1, "fName", 65, ""),
        CloneMember("tref_test_pid", 0, "fRef", 61, "a reference"),
    ]
    monkeypatch.setattr(objects, "clones_members", lambda element: members)
    kind = objects.SplitArray("tref_test_pid")
    named = [ROOT.TNamed(f"n{at}", "") for at in range(3)]
    entries = [
        kind.entry([_Pid("a", named[0]), _Pid("b", named[1])]),
        kind.entry([_Pid("c", named[2])]),
    ]
    keys = ["r1", "r1.fName", "r1.fRef"]
    columns = kind.columns(keys, entries)
    assert list(columns["r1"]) == [2, 1]
    assert columns["r1.fName"] == [["a", "b"], ["c"]]
    refs = columns["r1.fRef"]
    assert [(row[0].GetObject().GetName(), row[1].GetObject().GetName()) for row in refs[:1]] == [
        ("n0", "n1")
    ]
    assert [ref.GetObject().GetName() for ref in refs[1]] == ["n2"]
