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
    assert (chain.GetEntries(), chain.GetNtrees()) == (0, 0)
    with pytest.raises(ValueError, match="has no files; Add some first"):
        chain.Draw("x")
    added = (chain.Add(files[0]), chain.Add(str(tmp_path / "run[12].root")))
    assert (added, chain.GetNtrees(), chain.GetEntries()) == ((1, 2), 3, 30)
    assert [element.GetTitle() for element in chain.GetListOfFiles()] == files
    assert chain.GetListOfFiles()[0].GetName() == "events"
    other = TChain("events")
    added = (other.Add(chain), other.AddFile(files[0], -1, "events"))
    assert (added, other.GetEntries(), other.GetEntries("x < 5")) == ((3, 1), 40, 10)
    named = TChain("ignored")
    named.Add(f"{files[1]}/events")
    assert (named.GetEntries(), other.ClassName()) == (PER_FILE, "TChain")
    for each in (chain, other, named):
        each._xrd.close()


def _chained(files):
    chain = TChain("events")
    for path in files:
        chain.Add(path)
    return chain


def test_a_chain_reads_entries_across_its_files_in_order(files):
    chain = _chained(files)
    x = np.zeros(1)
    chain.SetBranchAddress("x", x)
    total = 0.0
    for i in range(chain.GetEntries()):
        chain.GetEntry(i)
        total += x[0]
    assert (total, chain.GetTreeNumber()) == (sum(range(30)), 2)
    assert chain.Draw("x", "x > 14", "goff") == 15
    assert [branch.GetName() for branch in chain.GetListOfBranches()] == ["x", "v"]
    wrapped = wrap(chain._xrd)
    assert (wrapped.GetEntries(), wrapped.GetNtrees()) == (30, 3)
    chain._xrd.close()


def test_a_chain_knows_which_file_each_entry_is_in(files):
    chain = _chained(files)
    assert chain.GetTreeNumber() == -1
    assert (chain.LoadTree(25), chain.GetTreeNumber(), chain.LoadTree(99)) == (5, 2, -2)
    assert chain.GetTreeOffset() == [0, 10, 20, 30]
    tree = chain.GetTree()
    assert (tree.GetEntries(), tree.ClassName()) == (PER_FILE, "TTree")
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
        assert (len(seen), seen[2]) == (PER_FILE, (2.0, 2.0, 2.0, 2, [2.0, 2.0], 2))
        assert (reader.GetCurrentEntry(), x.IsValid(), x.GetBranchName()) == (PER_FILE, True, "x")
        with pytest.raises(KeyError, match="no branch called 'nope'"):
            TTreeReaderValue(reader, "nope")


def test_a_reader_walks_a_range_and_is_set_to_any_entry(files):
    with xrdroot.open_root(files[0]) as f:
        reader = TTreeReader("events", f)
        x = TTreeReaderValue["double"](reader, "x")
        v = TTreeReaderArray["float"](reader, "v")
        reader.Restart()
        reader.SetEntriesRange(4, 7)
        assert list(reader) == [4, 5, 6]
        assert (reader.SetEntry(8), reader.SetEntry(3), x.__deref__()) == (7, 0, 3.0)
        reader.SetEntry(5)
        assert (v.GetSize(), v.size(), v[1], v.At(0), v.IsEmpty()) == (2, 2, 5.0, 5.0, False)
        assert list(iter(v)) == [5.0, 5.0]
        assert (reader.GetEntries(), reader.IsChain()) == (PER_FILE, False)


def test_a_reader_is_made_from_a_tree_a_chain_a_path_or_nothing(files):
    chain = TChain("events")
    chain.Add(files[0])
    chain.Add(files[1])
    reader = TTreeReader(chain)
    assert (reader.IsChain(), reader.GetEntries(), reader.GetTree()) == (True, 20, chain)
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
