"""Entries with no value of a column: a chain of trees that differ, a friend matched by index.

ROOT 6.32's ``DefaultValueFor``, ``FilterAvailable`` and ``FilterMissing``
over the two ways an entry comes to have no value - df036's chain of
``file?#tree`` parts with different branches, and df037's friends indexed by
``BuildIndex`` that match only some entries - held to what ROOT 6.40 shows.
"""

from __future__ import annotations

import array
import pickle

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.rdf.frame import RDataFrame
from xrdroot.rdf.patchwork import Befriended, Friend, Patched
from xrdroot.rdf.sources import Table

LOWEST = -2147483648


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def _tree(path, name, columns, rows, vector=False):
    """A file holding one tree of int branches ``columns``, filled with ``rows``."""
    with ROOT.TFile(str(path), "RECREATE"):
        t = ROOT.TTree(name, name)
        cells = {c: array.array("i", [0]) for c in columns}
        for c in columns:
            t.Branch(c, cells[c], f"{c}/I")
        held = ROOT.std.vector["float"]()
        if vector:
            t.Branch("v", held)
        for row in rows:
            for c, v in zip(columns, row, strict=True):
                cells[c][0] = v
            held.clear()
            held.push_back(1.0)
            t.Fill()
        t.Write()
    return str(path)


def _patched(tmp_path, vector=False):
    chain = ROOT.TChain()
    chain.Add(_tree(tmp_path / "a.root", "t1", ["x", "y"], [(1, 2), (2, 4)]) + "?#t1")
    chain.Add(_tree(tmp_path / "b.root", "t2", ["y"], [(3,), (6,)], vector) + "?#t2")
    return chain


def test_a_chain_of_trees_that_differ_has_no_value_where_a_tree_lacks_the_branch(tmp_path):
    df = ROOT.RDataFrame(_patched(tmp_path))
    filled = df.DefaultValueFor("x", LOWEST)
    assert list(filled.Take["int"]("x").GetValue()) == [1, 2, LOWEST, LOWEST]
    assert list(df.FilterAvailable("x").Take["int"]("y").GetValue()) == [2, 4]
    assert list(df.FilterMissing("x").Take["int"]("y").GetValue()) == [3, 6]
    box = df.FilterMissing("x").Display(columnList=("y",), nRows=5).GetValue().AsString()
    assert "| 2   | 3 | " in box  # the row is the entry's number, as ROOT's box shows it
    with pytest.raises(LookupError, match="entry 2 has no value of the column 'x'"):
        df.Take["int"]("x").GetValue()
    assert df.Count().GetValue() == 4
    after = df.Filter("y > 3")  # the missing value of a later entry, past a filter
    assert list(after.FilterMissing("x").Take["int"]("y").GetValue()) == [6]
    with pytest.raises(LookupError, match="entry 3 has no value of the column 'x'"):
        after.Take["int"]("x").GetValue()
    df.close()
    plain = RDataFrame(4).Define("z", "1")
    with pytest.raises(ValueError, match="is one made here"):
        plain.FilterAvailable("z")


def test_a_patchwork_reads_each_tree_in_turn_and_keeps_only_its_parts_when_pickled(tmp_path):
    chain = _patched(tmp_path, vector=True)
    source = Patched(chain._files)
    assert source.names() == ["x", "y", "v"] and "v" in source.readable()
    assert source.available("y", 0, 4) is None and len(source.available("x", 4, 4)) == 0
    assert source.read("x", 4, 4).dtype == source.read("x", 0, 1).dtype
    assert source.sample(1).AsString().endswith("b.root/t2") and "?#t2" in source.describe()
    assert source.cxx_type("x") is None
    with pytest.raises(UnsupportedFeatureError, match="holds more than one number"):
        source.read("v", 0, 4)
    again = pickle.loads(pickle.dumps(source))
    assert again.parts == source.parts and len(again) == 4
    source.close()
    again.close()


def test_a_friend_indexed_by_build_index_matches_the_entries_it_can(tmp_path):
    main = ROOT.TChain("events")
    main.Add(_tree(tmp_path / "m.root", "events", ["idx", "x"], [(1, 1), (2, 2), (3, 3)]))
    aux = ROOT.TChain("aux")
    aux.Add(_tree(tmp_path / "x.root", "aux", ["idx", "y"], [(1, 4), (3, 7)]))
    aux.BuildIndex("idx")
    main.AddFriend(aux)
    df = ROOT.RDataFrame(main)
    assert list(df.DefaultValueFor("aux.y", -1).Take["int"]("aux.y").GetValue()) == [4, -1, 7]
    assert list(df.FilterAvailable("aux.y").Take["int"]("x").GetValue()) == [1, 3]
    assert list(df.FilterMissing("aux.idx").Take["int"]("x").GetValue()) == [2]
    with pytest.raises(ValueError, match="is one made here"):
        df.Define("z", "x * 2").DefaultValueFor("z", 0)
    with pytest.raises(KeyError, match="names a column this frame has none called"):
        df.FilterAvailable("nothing")
    main._xrd.close()
    aux._xrd.close()


def _short(tmp_path):
    """Files of a tree of three entries and of a friend of two, and a source of them both."""
    from xrdroot import open_root

    files = [open_root(_tree(tmp_path / "m.root", "t", ["x"], [(1,), (2,), (3,)])),
             open_root(_tree(tmp_path / "s.root", "s", ["y"], [(5,), (6,)], True))]  # fmt: skip
    tree, short = Table(files[0]["t"]), Table(files[1]["s"])
    return files, short, Befriended(tree, [Friend("s", short)])


def test_a_friend_shorter_than_its_tree_has_no_value_past_its_end(tmp_path):
    files, _, source = _short(tmp_path)
    assert source.names() == ["x", "s.y", "s.v"]
    assert "s.y" in source.readable()
    assert source.read("s.y", 0, 2).tolist() == [5, 6]
    assert source.available("s.y", 0, 2) is None
    assert source.read("s.y", 0, 3).tolist() == [5, 6, 0]
    assert source.available("s.y", 0, 3).tolist() == [True, True, False]
    with pytest.raises(UnsupportedFeatureError, match=r"'s\.v' is missing in some entries"):
        source.read("s.v", 0, 3)
    assert source.read("s.y", 1, 3).tolist() == [6, 0]  # read once, and kept
    source.close()
    for file in files:
        file.close()


def test_a_befriended_source_says_what_it_is_and_closes_what_it_opened(tmp_path):
    files, short, source = _short(tmp_path)
    assert len(source) == 3 and source.boundaries() == [0, 3]
    assert source.available("x", 0, 3) is None and source.read("x", 1, 2).tolist() == [2]
    assert source.cxx_type("x") is None and source.cxx_type("s.y") is None
    assert source.describe().endswith("with friends s")
    assert source.sample(0).EntryRange() == (0, 3)
    pair = Befriended(Table(short.table), [Friend("s", short)])  # a friend as long as the tree
    with pytest.raises(UnsupportedFeatureError, match="holds more than one in each"):
        RDataFrame(pair).DefaultValueFor("s.v", 0.0).Take("s.v").GetValue()
    owned = Befriended(Patched([(_tree(tmp_path / "p.root", "p", ["x"], [(1,)]), "p")]),
                       [Friend("f", Patched([(str(tmp_path / "p.root"), "p")]))])  # fmt: skip
    assert owned.owned and owned.read("f.x", 0, 1).tolist() == [1]
    owned.close()
    for file in files:
        file.close()


def test_every_entry_of_a_tree_alone_has_its_columns(tmp_path):
    from xrdroot import open_root

    with open_root(_tree(tmp_path / "t.root", "t", ["x"], [(1,), (2,)])) as file:
        df = RDataFrame(file["t"])
        assert df.FilterAvailable("x").Filter("x > 1").FilterAvailable("x").Count().GetValue() == 1
        assert df.FilterMissing("x").Count().GetValue() == 0
