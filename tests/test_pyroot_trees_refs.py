"""``TTree::BranchRef``: the table saying which branch holds each referenced object.

ROOT keeps it in the tree record's ``fBranchRef`` slot, outside
``fBranches``, and fills its one basket at every ``Fill`` with the branch
number of every object a ``TRef`` names. What a tree being filled knows here
is the table itself: the branch of each referenced object read, as its
identifier and the process that gave it say.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.buffer import Buffer
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pycint.runtime import Cell
from xrdroot.pyroot.trees.branches import TBranchRef, TRefTable, _referenced
from xrdroot.writer import WBuffer


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_branch_ref_is_kept_in_its_own_slot_and_not_among_the_branches():
    tree = ROOT.TTree("t", "a tree")
    ref = tree.BranchRef()
    assert isinstance(ref, TBranchRef)
    assert (ref.GetName(), ref.GetTitle()) == (
        "TRefTable", "List of branch numbers with referenced objects"
    )
    assert tree.GetBranchRef() is ref and tree.BranchRef() is ref
    assert ref.GetRefTable().GetOwner() is ref
    assert tree.GetBranch("TRefTable") is None
    assert tree.GetListOfBranches() == []


def test_a_process_the_table_has_not_seen_yet_gets_a_row_of_its_own():
    table = TRefTable()
    ROOT.TProcessID.AddProcessID()  # the session's own, which Add(1) below uses
    other = ROOT.TProcessID.AddProcessID()
    table.SetParent("a branch")
    table.Add(1, other)
    table.Add(1)  # the session's own process, a row the loop has to walk past ``other`` for
    assert len(table._described()["rows"]) == 2


def test_a_cleared_tables_row_is_written_as_empty():
    table = TRefTable()
    table.SetParent("a branch")
    table.Add(3)
    table.Clear()
    assert table._described()["rows"] == [[]]  # trailing zeroes are not written


def _two_entries_of_two_referenced_names(tree, clones):
    """Fill ``tree`` twice, two numbered ``TNamed``s a branch entry."""
    for entry in range(2):
        clones.Clear()
        for slot in range(2):
            named = clones.ConstructedAt(slot)
            named.SetName(f"N_{entry}_{slot}")
            ROOT.TRef(named)  # numbers it, so the branch holding it is recorded
        tree.Fill()


def test_each_fill_records_the_branch_holding_every_referenced_object():
    tree = ROOT.TTree("t", "a tree")
    clones = ROOT.TClonesArray("TNamed")
    tree.Branch("n", Cell(clones))
    table = tree.BranchRef().GetRefTable()
    _two_entries_of_two_referenced_names(tree, clones)
    branch = tree.GetBranch("n")
    assert branch is not None
    assert table.GetN() == 5  # ROOT's fN: the last identifier plus one, not how many there are
    assert table.GetSize() >= 5
    assert table.GetParent(3) is table.GetParents().At(0)
    assert table.GetParent(4) is table.GetParents().At(0)
    assert table.GetParent(3).GetName() == branch.GetName()
    assert table.GetParent(0) is None
    assert table.GetParent(99) is None
    assert table.GetParents().GetEntriesFast() == 1


def test_a_reference_is_recorded_under_the_process_that_numbered_it():
    tree = ROOT.TTree("t", "a tree")
    clones = ROOT.TClonesArray("TNamed")
    tree.Branch("n", Cell(clones))
    ref = tree.BranchRef()
    named = clones.ConstructedAt(0)
    named.SetName("N")
    ROOT.TRef(named)
    tree.Fill()
    table = ref.GetRefTable()
    assert table.GetProcessGUIDs() == [ROOT.TProcessID.GetSessionProcessID().GetTitle()]
    ref.SetRequestedEntry(0)
    assert ref.Notify() is True
    table._owner = ref
    assert table.Notify() is True


def test_the_table_round_trips_through_the_bytes_root_writes():
    made = TRefTable()
    made.SetParent("a branch", 0)
    made.Add(1)
    made.Add(3)
    made.SetParent("another branch", 1)
    made.Add(2)
    buffer = WBuffer()
    made.FillBuffer(buffer)
    read = TRefTable()
    read.ReadBuffer(Buffer(bytes(buffer.data)))
    assert read._described()["rows"] == made._described()["rows"]
    assert read.GetParent(3) is None  # the parents themselves are back-references on file
    old = TRefTable()
    old.ReadBuffer(Buffer(b"\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x05"))
    assert old.GetN() == 2 and old.GetParent(1) is None


def test_a_cleared_table_forgets_the_entry_but_keeps_its_branches():
    table = TRefTable()
    table.SetParent("a branch")
    table.Add(2)
    table.SetUID(2, None)
    assert (table.GetUID(), table.GetUIDContext()) == (2, None)
    table.Clear()
    assert table.GetN() == 0 and table.GetParent(2) is None
    assert table.GetParents().GetEntriesFast() == 1
    table.SetParent("a branch")
    table.Add(2)
    table.Reset()
    assert table.GetN() == 0
    assert table.GetParents().GetEntriesFast() == 0


def test_the_current_table_is_the_one_set_for_the_tree_being_read():
    made = TRefTable()
    assert TRefTable.GetRefTable() is made
    TRefTable.SetRefTable(None)
    assert TRefTable.GetRefTable() is None
    TRefTable.SetRefTable(TRefTable())


def test_a_table_with_no_parent_refuses_to_record_an_object():
    table = TRefTable()
    assert table.Add(7) == -1
    assert table.GetParent(7) is None


def test_a_branch_ref_counts_its_entries_and_can_be_printed(capsys):
    tree = ROOT.TTree("t", "a tree")
    tree.Branch("x", np.zeros(1), "x/D")
    ref = tree.BranchRef()
    tree.Fill()
    assert ref.GetEntries() == 1
    ref.Print()
    assert "TRefTable" in capsys.readouterr().out
    ref.Reset()
    assert ref.GetRefTable().GetParents().GetEntriesFast() == 0


def test_a_reference_notify_reads_the_branch_of_the_entry_asked_for():
    out = ROOT.TFile("a.root", "RECREATE")
    tree = ROOT.TTree("t", "a tree")
    x = np.zeros(1)
    tree.Branch("x", x, "x/D")
    for value in (1.0, 2.0):
        x[0] = value
        tree.Fill()
    out.Close()
    back = ROOT.TFile("a.root").Get("t")
    ref = TBranchRef(back)  # what the tree record's fBranchRef slot gives a reader
    ref.SetRequestedEntry(1)
    branch = back.GetBranch("x")
    assert branch is not None
    ref.GetRefTable().SetParent(branch, -1)
    ref.GetRefTable().Add(1)
    ref.GetRefTable().SetUID(1)
    ref.Notify()
    assert branch.GetReadEntry() == 1
    ref.SetParent(branch, 0)
    assert ref.GetRefTable().GetParents().At(0) is branch


def test_only_objects_a_reference_may_point_at_are_recorded():
    named = ROOT.TNamed("N", "N")
    assert list(_referenced(named)) == []
    ROOT.TRef(named)
    assert list(_referenced(named)) == [named]
    assert list(_referenced([named, (1, 2)])) == [named]


def test_a_fill_with_no_object_branch_has_nothing_to_record():
    tree = ROOT.TTree("t", "a tree")
    tree.Branch("x", np.zeros(1), "x/D")
    ref = tree.BranchRef()
    missing = SimpleNamespace(kind="object", branch="not a branch", address=object())
    ref._account(SimpleNamespace(slots={"x": SimpleNamespace(kind="scalar"), "m": missing},
                                 entries=2))  # fmt: skip
    assert ref._has_rows() is False


def test_a_cloned_split_collection_is_not_accounted_for_by_an_address_it_has_not():
    """A split ``TClonesArray`` a clone copies holds no address of its own - its objects are
    read out of its columns - so a ``BranchRef`` filling such a tree has nothing to number
    through it and steps past it, as ROOT's table, which the writing branch feeds, does."""
    tree = ROOT.TTree("t", "a tree")
    tree.Branch("x", np.zeros(1), "x/D")
    ref = tree.BranchRef()
    cloned = SimpleNamespace(kind="object", branch="r1", follows=[])
    ref._account(SimpleNamespace(slots={"r1": cloned}, entries=1))
    assert ref._has_rows() is False


def test_a_tree_whose_table_cannot_be_written_refuses_rather_than_losing_it():
    out = ROOT.TFile("a.root", "RECREATE")
    tree = ROOT.TTree("t", "a tree")
    clones = ROOT.TClonesArray("TNamed")
    tree.Branch("n", Cell(clones))
    tree.BranchRef()
    named = clones.ConstructedAt(0)
    named.SetName("N")
    ROOT.TRef(named)
    tree.Fill()
    with pytest.raises(UnsupportedFeatureError, match=r"puts no fBranchRef in the tree record"):
        tree.Write()
    tree.SetDirectory(None)
    out.Close()


def test_a_writer_that_carries_the_table_is_given_it_beside_the_tree():
    import io

    import xrdroot

    tree = ROOT.TTree("t", "a tree")
    clones = ROOT.TClonesArray("TNamed")
    tree.Branch("n", Cell(clones))
    tree.BranchRef()
    named = clones.ConstructedAt(0)
    named.SetName("N")
    ROOT.TRef(named)
    tree.Fill()
    with xrdroot.create(io.BytesIO()) as out:
        out.writes_ref_table = True  # what a writer able to put fBranchRef in its record says
        tree.SetDirectory(out)
        assert tree.Write() > 0
        described = tree._wrote._ref_table
    uid = int(named.GetUniqueID()) & 0xFFFFFF
    assert described["name"] == "TRefTable" and described["rows"][0][uid] == 1


def test_a_branch_ref_with_nothing_recorded_writes_the_tree_as_ever():
    out = ROOT.TFile("a.root", "RECREATE")
    tree = ROOT.TTree("t", "a tree")
    x = np.zeros(1)
    tree.Branch("x", x, "x/D")
    tree.BranchRef()
    x[0] = 1.5
    tree.Fill()
    assert tree.Write() > 0
    out.Close()
    assert ROOT.TFile("a.root").Get("t").GetEntry(0) > 0
