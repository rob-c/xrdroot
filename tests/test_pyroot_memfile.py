"""``TMemFile``: a ROOT file in memory; and a tree moved between directories."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT


def test_a_memory_file_is_written_read_back_from_its_bytes_and_never_touches_disk(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    made = ROOT.TMemFile("scratch.root", "RECREATE")
    histogram = ROOT.TH1D("h", "h", 4, 0, 1)
    histogram.Fill(0.5)
    histogram.Write()
    made.Close()
    data = bytearray(made.GetSize())
    assert made.CopyTo(data) == len(data) > 0
    again = ROOT.TMemFile("again.root", data, len(data))
    assert not again.IsZombie() and again.Get("h").GetEntries() == 1
    assert ROOT.TMemFile("whole.root", bytes(data)).Get("h").GetEntries() == 1
    assert not list(tmp_path.iterdir())


def test_a_memory_file_with_nothing_to_read_is_a_zombie(capfd):
    assert ROOT.TMemFile("empty.root", "READ").IsZombie()
    assert "no buffer to read empty.root" in capfd.readouterr().err


def test_a_tree_set_to_no_directory_leaves_the_file_it_was_in_and_can_move_to_another():
    first = ROOT.TMemFile("first.root", "RECREATE")
    tree = ROOT.TTree("t", "t")
    assert tree.GetDirectory() is first and first.GetList().Contains(tree)
    tree.SetDirectory(ROOT.nullptr)
    assert tree.GetDirectory() is None and not first.GetList().Contains(tree)
    first.Close()
    second = ROOT.TMemFile("second.root", "RECREATE")
    tree.SetDirectory(second)
    assert tree.GetDirectory() is second and second.GetList().Contains(tree)
    second.Close()


def test_a_tree_given_a_directory_with_no_list_of_its_own_is_written_there_still():
    class Bare:
        """A directory to write into that keeps no list of what it holds."""

    tree = ROOT.TTree("t", "t")
    tree.SetDirectory(None)
    bare = Bare()
    tree.SetDirectory(bare)
    assert tree.GetDirectory() is bare
    tree.SetDirectory(0)
    assert tree.GetDirectory() is None


def test_a_leaf_list_over_a_macros_struct_takes_its_members_in_order_whatever_their_names(
    tmp_path,
):
    ROOT.gInterpreter.Declare("struct LeafPair { int first; int second; };")
    ROOT.gInterpreter.Declare("struct LeafOne { int only; };")
    pair, named = ROOT.LeafPair(), ROOT.LeafPair()
    path = str(tmp_path / "pairs.root")
    made = ROOT.TFile(path, "RECREATE")
    tree = ROOT.TTree("t", "t")
    tree.Branch("pair", pair, "a/I:b/I")
    tree.Branch("named", named, "first/I:second/I")
    with pytest.raises(TypeError, match="LeafOne"):
        tree.Branch("long", ROOT.LeafOne(), "x/I:y/I")
    pair.first, pair.second, named.second = 3, 4, 5
    tree.Fill()
    made.Write()
    made.Close()
    with ROOT.TFile(path) as read:
        back = read.Get("t")
        back.GetEntry(0)
        assert [back.a, back.b, back.second] == [3, 4, 5]


def test_the_address_of_a_member_is_a_place_a_tree_reads_it_from(tmp_path):
    ROOT.gInterpreter.Declare("struct AddressedPair { int first; int second; };")
    pair = ROOT.AddressedPair()
    assert ROOT.addressof(pair) == id(pair)
    place = ROOT.addressof(pair, "second")
    place.value = 9
    assert pair.second == 9
    path = str(tmp_path / "addressed.root")
    made = ROOT.TFile(path, "RECREATE")
    tree = ROOT.TTree("t", "t")
    tree.Branch("second", place, "second/I")
    tree.Fill()
    made.Write()
    made.Close()
    with ROOT.TFile(path) as read:
        back = read.Get("t")
        back.GetEntry(0)
        assert back.second == 9
