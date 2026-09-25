"""Friends, indices, entry lists and copies: ``tree109_friend.C`` to ``tree112_copy.C``.

The parent tree has ``Run``, ``Event`` and ``x``; the friend is a copy of
some of its entries, indexed by ``Run`` and ``Event``, and reading the
parent must find in the friend the entry with the same two numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot
from xrdroot.pyroot.trees import TEntryList, TTree, hooks, wrap

ENTRIES = 40


def _parent() -> TTree:
    run, event, x = np.zeros(1, "i"), np.zeros(1, "i"), np.zeros(1, "f")
    t = TTree("T", "test friend trees")
    t.Branch("Run", run, "Run/I")
    t.Branch("Event", event, "Event/I")
    t.Branch("x", x, "x/F")
    for i in range(ENTRIES):
        run[0], event[0], x[0] = 1 + i // 20, i, i * 0.25
        t.Fill()
    return t


def _write(tmp_path, tree, name):
    path = tmp_path / f"{name}.root"
    with xrdroot.create(str(path)) as f:
        tree.SetDirectory(f)
        tree.Write()
    return str(path)


def test_copy_tree_keeps_what_the_cut_keeps_and_an_index_finds_each_again():
    parent = _parent()
    friend = parent.CopyTree("x < 5 && Event % 2 == 0")
    friend.SetName("TF")
    assert friend.GetEntries() == 10 and friend.BuildIndex("Run", "Event") == 10
    assert friend.GetEntryNumberWithIndex(1, 8) == 4 and friend.GetEntryNumberWithIndex(1, 7) == -1
    fx = np.zeros(1, "f")
    friend.SetBranchAddress("x", fx)
    assert friend.GetEntryWithIndex(1, 8) > 0 and fx[0] == 2.0
    assert friend.GetEntryWithIndex(2, 99) == -1
    assert parent.GetEntryNumberWithIndex(1, 1) == -1


def test_a_friend_with_an_index_is_read_at_the_entry_whose_numbers_match():
    parent = _parent()
    friend = parent.CopyTree("Event % 3 == 0")
    friend.BuildIndex("Run", "Event")
    x, fx, fevent = np.zeros(1, "f"), np.zeros(1, "f"), np.zeros(1, "i")
    parent.SetBranchAddress("x", x)
    friend.SetBranchAddress("x", fx)
    friend.SetBranchAddress("Event", fevent)
    assert parent.AddFriend(friend, "TF") is friend
    matched = 0
    for i in range(ENTRIES):
        parent.GetEntry(i)
        matched += bool(fevent[0] == i and fx[0] == x[0])
    assert matched == 14
    assert parent.GetListOfFriends()[0] is friend and parent.GetFriend("TF") is friend
    assert parent.GetFriend("nobody") is None
    parent.RemoveFriend(friend)
    assert parent.GetListOfFriends().GetEntries() == 0


def test_an_index_of_a_number_rather_than_a_branch_keys_on_the_number():
    parent = _parent()
    friend = parent.CloneTree()
    friend.BuildIndex("Event")
    fevent = np.zeros(1, "i")
    friend.SetBranchAddress("Event", fevent)
    parent.AddFriend(friend)
    parent.GetEntry(7)
    assert fevent[0] == 7
    other = parent.CloneTree()
    other.BuildIndex("Run", "Event")
    other._index.major = "2"
    parent.AddFriend(other, "o")
    oevent = np.zeros(1, "i")
    other.SetBranchAddress("Event", oevent)
    parent.GetEntry(25)
    assert oevent[0] == 25
    other._index.major = "Event * 1"
    parent.GetEntry(3)
    assert oevent[0] == 25  # an index value of no branch is the entry, which matches nothing


def test_a_friend_of_as_many_entries_is_drawn_beside_the_tree(tmp_path):
    parent = _parent()
    weights = TTree("W", "")
    w = np.zeros(1)
    weights.Branch("w", w, "w/D")
    for i in range(ENTRIES):
        w[0] = i
        weights.Fill()
    parent.AddFriend(weights)
    assert parent.Draw("W.w", "x > 9", "goff") == 3
    with xrdroot.open_root(_write(tmp_path, _parent(), "parent")) as f:
        written = wrap(f["T"])
        opened = written.AddFriend("W", _write(tmp_path, weights, "weights"))
        assert written.Draw("w", "w < 10", "goff") == 10
        opened._xrd._source.close()  # the friend's file, opened for it, as ROOT opens it
    with xrdroot.open_root(_write(tmp_path, _parent(), "again")) as f:
        with xrdroot.open_root(str(tmp_path / "weights.root")) as g:
            by_alias = wrap(f["T"])
            by_alias.AddFriend("ww=W", g)
            assert by_alias.GetFriend("ww").GetEntries() == ENTRIES


def test_draw_to_an_entry_list_keeps_the_entries_and_set_entry_list_walks_only_them():
    parent = _parent()
    assert parent.Draw(">>elist", "Event % 5 == 0", "entrylist") == 8
    elist = hooks.registry()["elist"]
    assert isinstance(elist, TEntryList) and elist.GetN() == 8 and elist.GetTreeName() == "T"
    assert parent.Draw(">>+elist", "Event == 1") == 1 and elist.GetN() == 9
    assert parent.Draw(">>fresh", "Event < 30", "", 10, 5) == 10
    parent.SetEntryList(elist)
    assert parent.GetEntryList() is elist and parent.GetEntryNumber(2) == 5
    assert [int(event.Event) for event in parent] == [0, 1, 5, 10, 15, 20, 25, 30, 35]
    assert parent.Draw("x", "", "goff") == 9 and parent.Draw(">>sub", "Event > 20") == 3
    assert list(hooks.registry()["sub"]) == [25, 30, 35]
    assert parent.Scan("Event", "Event > 20") == 3
    parent.SetEntryList(None)
    assert parent.Draw("x", "", "goff") == ENTRIES


def test_an_entry_list_keeps_its_entries_once_each_and_in_order(capsys):
    elist = TEntryList("e", "some entries")
    assert elist.Enter(5) and elist.Enter(2) and not elist.Enter(5)
    assert list(elist) == [2, 5] and len(elist) == 2 and elist.Contains(2) == 1
    assert elist.GetEntry(1) == 5 and elist.GetEntry(9) == -1 and "2 entries" in repr(elist)
    assert elist.Next() == 2 and elist.Next() == 5 and elist.Next() == -1
    assert elist.Remove(2) and not elist.Remove(2)
    other = TEntryList("o", "", _parent())
    other.Enter(7)
    elist.Add(other)
    elist.SetTree("T")
    elist.Print("all")
    assert capsys.readouterr().out == "T  2\n5\n7\n"
    elist.SetTree(_parent())
    assert other.GetTreeName() == "T"
    elist.Reset()
    assert elist.GetN() == 0


def test_clone_tree_of_nothing_then_fill_copies_the_entries_the_loop_picks():
    parent = _parent()
    parent.SetBranchStatus("*", 0)
    parent.SetBranchStatus("Event", 1)
    parent.SetBranchStatus("x", 1)
    clone = parent.CloneTree(0)
    assert [b.GetName() for b in clone.GetListOfBranches()] == ["Event", "x"]
    with pytest.raises(ValueError, match="has read none yet"):
        clone.Fill()
    for i in range(ENTRIES):
        parent.GetEntry(i)
        if parent.x > 8:
            clone.Fill()
    assert clone.GetEntries() == 7 and clone.Draw("Event", "Event > 35", "goff") == 4
    bound = np.zeros(1, "f")
    parent.SetBranchAddress("x", bound)
    parent.GetEntry(0)
    bound[0] = 99.0
    clone.Fill()
    assert clone.GetEntries("x == 99") == 1
    assert parent.CloneTree(5).GetEntries() == 5 and parent.CloneTree().GetEntries() == ENTRIES


def test_a_clone_takes_counters_vectors_and_text_with_what_they_count(tmp_path):
    from xrdroot.pyroot.stl import std

    n, a, v, s = np.zeros(1, "i"), np.zeros(3), std.vector["float"](), std.string()
    t = TTree("t", "")
    t.Branch("n", n, "n/I")
    t.Branch("a", a, "a[n]/D")
    t.Branch("v", v)
    t.Branch("s", s)
    for i in range(4):
        n[0], a[:] = i % 3, i
        v.assign([i] * i)
        s.assign(str(i))
        t.Fill()
    t.SetBranchStatus("n", 0)
    copy = t.CopyTree("n > 0")
    assert copy.GetEntries() == 2
    with xrdroot.open_root(_write(tmp_path, copy, "copy")) as f:
        read = wrap(f["t"])
        got = read._xrd.arrays()
        assert got["s"] == ["1", "2"] and [list(r) for r in got["v"]] == [[1.0], [2.0, 2.0]]
        assert [list(r) for r in got["a"]] == [[1.0], [2.0, 2.0]]
        assert read.CloneTree().GetEntries() == 2


def test_a_clone_filled_entry_by_entry_reads_arrays_from_what_the_source_read(capsys):
    n, a = np.zeros(1, "i"), np.zeros(3)
    t = TTree("t", "")
    t.Branch("n", n, "n/I")
    t.Branch("a", a, "a[n]/D")
    for i in range(3):
        n[0], a[:] = i, [i, i, i]
        t.Fill()
    clone = t.CloneTree(0)
    for i in range(3):
        t.GetEntry(i)
        clone.Fill()
    assert [list(row) for row in clone._xrd.arrays()["a"]] == [[], [1.0], [2.0, 2.0]]
    empty = TEntryList("none")
    empty.Print("all")
    empty.Print()
    assert capsys.readouterr().out == "  0\n  0\n"
