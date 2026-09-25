"""``TChain`` and ``TTreeReader``: files read as one tree, and a tree walked by a reader.

The files are written here, three of ten entries each, so that entry
numbers across the chain and within each file can both be checked.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot
from xrdroot.pyroot.stl import std
from xrdroot.pyroot.trees import (
    TChain,
    TTree,
    TTreeReader,
    TTreeReaderArray,
    TTreeReaderValue,
    wrap,
)

PER_FILE = 10


@pytest.fixture
def files(tmp_path):
    made = []
    for number in range(3):
        x, v = np.zeros(1), std.vector["float"]()
        t = TTree("events", "")
        t.Branch("x", x, "x/D")
        t.Branch("v", v)
        for i in range(PER_FILE):
            x[0] = number * PER_FILE + i
            v.assign([x[0]] * (i % 3))
            t.Fill()
        path = tmp_path / f"run{number}.root"
        with xrdroot.create(str(path)) as f:
            t.SetDirectory(f)
            t.Write()
        made.append(str(path))
    return made


def test_a_chain_adds_files_by_name_by_wildcard_and_by_another_chain(files, tmp_path):
    chain = TChain("events")
    assert chain.GetEntries() == 0 and chain.GetNtrees() == 0
    with pytest.raises(ValueError, match="has no files; Add some first"):
        chain.Draw("x")
    assert chain.Add(files[0]) == 1
    assert chain.Add(str(tmp_path / "run[12].root")) == 2
    assert chain.GetNtrees() == 3 and chain.GetEntries() == 30
    assert [element.GetTitle() for element in chain.GetListOfFiles()] == files
    assert chain.GetListOfFiles()[0].GetName() == "events"
    other = TChain("events")
    assert other.Add(chain) == 3 and other.AddFile(files[0], -1, "events") == 1
    assert other.GetEntries() == 40 and other.GetEntries("x < 5") == 10
    assert other.ClassName() == "TChain"
    named = TChain("ignored")
    named.Add(f"{files[1]}/events")
    assert named.GetEntries() == PER_FILE
    for each in (chain, other, named):
        each._xrd.close()


def test_a_chain_reads_entries_across_its_files_and_knows_which_file_each_is_in(files):
    chain = TChain("events")
    for path in files:
        chain.Add(path)
    assert chain.GetTreeNumber() == -1
    x = np.zeros(1)
    chain.SetBranchAddress("x", x)
    assert chain.LoadTree(25) == 5 and chain.GetTreeNumber() == 2
    assert chain.LoadTree(99) == -2 and chain.GetTreeOffset() == [0, 10, 20, 30]
    total = 0.0
    for i in range(chain.GetEntries()):
        chain.GetEntry(i)
        total += x[0]
    assert total == sum(range(30)) and chain.GetTreeNumber() == 2
    tree = chain.GetTree()
    assert tree.GetEntries() == PER_FILE and tree.ClassName() == "TTree"
    assert chain.Draw("x", "x > 14", "goff") == 15
    assert [branch.GetName() for branch in chain.GetListOfBranches()] == ["x", "nv", "v"]
    wrapped = wrap(chain._xrd)
    assert wrapped.GetEntries() == 30 and wrapped.GetNtrees() == 3
    chain._xrd.close()


def test_a_chain_of_trees_of_different_names_is_refused(files):
    chain = TChain("events")
    chain.Add(files[0])
    chain.Add(f"{files[1]}/other")
    with pytest.raises(ValueError, match="holds trees called events, other"):
        chain.GetEntries()


def test_a_reader_walks_every_entry_and_its_values_read_as_cplusplus_reads_them(files):
    with xrdroot.open_root(files[0]) as f:
        reader = TTreeReader("events", f)
        x = TTreeReaderValue["double"](reader, "x")
        v = TTreeReaderArray["float"](reader, "v")
        with pytest.raises(ValueError, match="call Next first"):
            x.Get()
        seen = []
        while reader.Next():
            seen.append((x.Get()[0], x.__deref__(), float(x), int(x), list(v), len(v)))
        assert len(seen) == PER_FILE and seen[2] == (2.0, 2.0, 2.0, 2, [2.0, 2.0], 2)
        assert reader.GetCurrentEntry() == PER_FILE and x.IsValid() and x.GetBranchName() == "x"
        reader.Restart()
        reader.SetEntriesRange(4, 7)
        assert list(reader) == [4, 5, 6]
        assert reader.SetEntry(8) == 8 and reader.SetEntry(3) == 0 and x.__deref__() == 3.0
        reader.SetEntry(5)
        assert v.GetSize() == v.size() == 2 and v[1] == 5.0 and v.At(0) == 5.0
        assert not v.IsEmpty() and list(iter(v)) == [5.0, 5.0]
        assert reader.GetEntries() == PER_FILE and not reader.IsChain()
        with pytest.raises(KeyError, match="no branch called 'nope'"):
            TTreeReaderValue(reader, "nope")


def test_a_reader_is_made_from_a_tree_a_chain_a_path_or_nothing(files):
    chain = TChain("events")
    chain.Add(files[0])
    chain.Add(files[1])
    reader = TTreeReader(chain)
    assert reader.IsChain() and reader.GetEntries() == 20 and reader.GetTree() is chain
    chain._xrd.close()
    by_path = TTreeReader("events", files[2])
    x = TTreeReaderValue(by_path, "x")
    by_path.Next()
    assert x.__deref__() == 20.0
    by_path.GetTree()._xrd._source.close()
    with pytest.raises(ValueError, match="needs the file it is in"):
        TTreeReader("events")
    empty = TTreeReader()
    assert not empty.Next() and empty.GetEntries() == 0 and not empty.IsChain()
    assert not TTreeReaderValue(empty, "anything").IsValid()
    with xrdroot.open_root(files[0]) as f:
        empty.SetTree(f["events"])
        assert empty.Next() and empty.GetTree().GetEntries() == PER_FILE
        from_xrd = TTreeReader(f["events"])
        assert from_xrd.GetEntries() == PER_FILE


def test_a_reader_given_an_entry_list_walks_only_its_entries(files):
    from xrdroot.pyroot.trees import TEntryList

    with xrdroot.open_root(files[0]) as f:
        tree = wrap(f["events"])
        chosen = TEntryList("chosen")
        for entry in (1, 3, 8):
            chosen.Enter(entry)
        reader = TTreeReader(tree, None, chosen)
        x = TTreeReaderValue["double"](reader, "x")
        values = []
        while reader.Next():
            values.append(x.__deref__())
        assert values == [1.0, 3.0, 8.0] and reader.GetEntries() == 3
        vector = TTreeReaderValue(TTreeReader(_vectors()), "v")
        vector._reader.Next()
        vector._reader.Next()
        assert list(vector.Get()) == [1.0] and isinstance(vector.Get(), std.vector["float"])


def _vectors():
    v = std.vector["float"]()
    t = TTree("t", "")
    t.Branch("v", v)
    for i in range(3):
        v.assign([1.0] * i)
        t.Fill()
    return t
