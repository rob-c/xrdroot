"""Entry lists, and reading the entries one names rather than a range.

No file in the corpus holds a ``TEntryList`` or a ``TEventList``, so the ones
here are made by ``crafted.py``, laid out member for member as the classes'
own headers declare them: the reading is checked against that layout, which
is the file's own description of itself, rather than against ROOT.

The trees they are applied to are ROOT's: ``pod-advanced.root``'s ``orange``
holds its hundred entries in two baskets split at entry 77, and its
``Evtake_iwant`` is the entry's own number, so a value read says which
entry it came from.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from crafted import craft
from xrdroot import EntryList, FormatError, UnsupportedFeatureError, open_root
from xrdroot.entries import BLOCK, selected

DATA = pathlib.Path(__file__).parent / "data"


def opened(name: str):
    return open_root(str(DATA / f"{name}.root"))


def bitmap(*entries: int) -> dict:
    """A block that keeps its entries as a bit each, sixteen to a word."""
    words = np.zeros(BLOCK // 16, np.uint16)
    for entry in entries:
        words[entry >> 4] |= 1 << (entry & 15)
    return {"fNPassed": len(entries), "fN": len(words), "fIndices": words, "fType": 0}


def listed(*entries: int, passing: bool = True, passed: int | None = None) -> dict:
    """A block that keeps a sorted list: of the entries that passed, or that did not."""
    count = len(entries) if passed is None else passed
    return {
        "fNPassed": count,
        "fN": len(entries),
        "fIndices": list(entries),
        "fType": 1,
        "fPassing": passing,
    }


def entry_list(name: str, tree: str, *blocks: dict, file_name: str = "", lists=None) -> dict:
    return {
        "TNamed": {"fName": name, "fTitle": f"entries of {tree}"},
        "fLists": lists,
        "fNBlocks": len(blocks),
        "fBlocks": [("TEntryListBlock", block) for block in blocks] if blocks else None,
        "fN": 0,
        "fTreeName": tree,
        "fFileName": file_name,
    }


@pytest.fixture
def lists(tmp_path):
    """A file of entry lists: every kind of block, one per tree, and the old kind."""
    everything = entry_list(
        "blocks",
        "orange",
        bitmap(1, 5, 17, 3999),
        listed(3, 10),
        listed(*range(4, BLOCK), passing=False, passed=4),
    )
    per_tree = entry_list(
        "per_tree",
        "",
        lists=[
            ("TEntryList", entry_list("a", "orange", listed(2, 80), file_name="/x/pod.root")),
            ("TEntryList", entry_list("b", "orange", listed(7), file_name="/y/other.root")),
            ("TEntryList", entry_list("c", "tree", listed(1, 2))),
        ],
    )
    old = {"TNamed": {"fName": "old"}, "fN": 3, "fSize": 100, "fDelta": 100, "fList": [4, 9, 16]}
    path = craft(
        tmp_path / "lists.root",
        [
            ("TEntryList", "blocks", everything),
            ("TEntryList", "per_tree", per_tree),
            ("TEventList", "old", old),
            ("TEntryList", "empty", entry_list("empty", "orange")),
        ],
    )
    with open_root(str(path)) as handle:
        yield handle


def test_every_kind_of_block_gives_back_the_entries_it_kept(lists):
    kept = lists["blocks"]
    assert isinstance(kept, EntryList)
    assert kept.entries.tolist() == [1, 5, 17, 3999, 4003, 4010, 8000, 8001, 8002, 8003]
    assert kept.entries.dtype == np.int64
    assert (kept.name, kept.title, kept.tree_name, len(kept)) == (
        "blocks",
        "entries of orange",
        "orange",
        10,
    )
    assert repr(kept) == "<TEntryList 'blocks' of 'orange', 10 entries>"


def test_a_list_with_nothing_in_it_is_empty(lists):
    empty = lists["empty"]
    assert empty.entries.tolist() == []
    assert empty.for_tree("anything") is empty  # one list is the list for any tree


def test_an_event_list_is_its_sorted_array(lists):
    old = lists["old"]
    assert old.entries.tolist() == [4, 9, 16]
    assert repr(old) == "<TEventList 'old', 3 entries>"


def test_a_list_made_over_a_chain_keeps_a_list_per_tree(lists):
    per_tree = lists["per_tree"]
    assert [one.name for one in per_tree.lists] == ["a", "b", "c"]
    assert len(per_tree) == 5
    assert per_tree.for_tree("tree").entries.tolist() == [1, 2]
    assert per_tree.for_tree("orange", "/data/pod.root").name == "a"
    assert per_tree.lists[1].file_name == "/y/other.root"
    with pytest.raises(UnsupportedFeatureError, match=r"keeps a list per tree \(orange in"):
        _ = per_tree.entries
    with pytest.raises(KeyError, match="has 2 lists for 'orange' among its 3"):
        per_tree.for_tree("orange")
    with pytest.raises(KeyError, match=r"has no lists for 'nope' in x\.root"):
        per_tree.for_tree("nope", "x.root")


def test_a_list_given_as_members_rather_than_read_is_one_too():
    one = {"TNamed": {"fName": "a"}, "fTreeName": "t", "fBlocks": [listed(3)]}
    made = EntryList("TEntryList", {"fLists": [one]})
    assert made.for_tree("t").entries.tolist() == [3]


def test_a_list_whose_blocks_are_not_blocks_is_refused():
    with pytest.raises(FormatError, match="blocks that are not TEntryListBlocks"):
        EntryList("TEntryList", {"fBlocks": ["TEntryListBlock"]})


def test_a_tree_reads_the_entries_a_list_kept_and_only_their_baskets(lists, monkeypatch):
    from xrdroot.tree import Branch

    read: list[int] = []
    real = Branch.basket

    def counted(self, index):
        read.append(index)
        return real(self, index)

    monkeypatch.setattr(Branch, "basket", counted)
    with opened("pod-advanced") as handle:
        tree = handle["orange"]
        picked = tree.arrays(["orange.Evtake_iwant"], entries=np.array([3, 10, 12]))
        assert picked["orange.Evtake_iwant"].tolist() == [3, 10, 12]
        assert read == [0]  # the first basket, and not the second
        per_file = tree.arrays(["orange.Mc_x"], entries=lists["per_tree"].lists[0])
        assert per_file["orange.Mc_x"].tolist() == [7.0, 85.0]


def test_a_list_made_over_a_chain_gives_a_tree_the_list_for_it(lists, tmp_path):
    import shutil

    shutil.copy(DATA / "pod-advanced.root", tmp_path / "pod.root")
    with open_root(str(tmp_path / "pod.root")) as handle:
        picked = handle["orange"].arrays(["orange.Evtake_iwant"], entries=lists["per_tree"])
    assert picked["orange.Evtake_iwant"].tolist() == [2, 80]


def test_entries_asked_for_out_of_order_come_back_in_that_order():
    with opened("pod-advanced") as handle:
        column = handle["orange"]["orange.Evtake_iwant"]
        assert column.array(entries=[90, 3, 77, 3]).tolist() == [90, 3, 77, 3]
        assert column.array(entries=[]).tolist() == []


def test_a_mask_picks_the_entries_it_says_yes_to():
    with opened("pod-advanced") as handle:
        column = handle["orange"]["orange.Evtake_iwant"]
        mask = np.zeros(100, bool)
        mask[[5, 95]] = True
        assert column.array(entries=mask).tolist() == [5, 95]
        with pytest.raises(ValueError, match="a mask of 3 entries was given for a tree of 100"):
            column.array(entries=[True, False, True])


def test_entries_that_are_not_entry_numbers_are_refused():
    with opened("pod-advanced") as handle:
        column = handle["orange"]["orange.Evtake_iwant"]
        with pytest.raises(IndexError, match="entry 100 is not in a tree of 100 entries"):
            column.array(entries=[1, 100])
        with pytest.raises(IndexError, match="entry -1 is not"):
            column.array(entries=[-1])
        with pytest.raises(ValueError, match="one-dimensional run of whole entry numbers"):
            column.array(entries=[1.5])
        with pytest.raises(ValueError, match="one-dimensional"):
            column.array(entries=[[1]])


def test_a_list_per_tree_handed_to_one_branch_is_refused_for_want_of_a_tree_name(lists):
    with opened("pod-advanced") as handle:
        with pytest.raises(UnsupportedFeatureError, match="keeps a list per tree"):
            handle["orange"]["orange.Mc_x"].array(entries=lists["per_tree"])
        assert handle["orange"]["orange.Mc_x"].array(entries=lists["old"]).tolist() == [
            9.0,
            14.0,
            21.0,
        ]


def test_rows_of_different_lengths_are_picked_row_by_row():
    with opened("chain.flat.1") as handle:
        rows = handle["tree"]["SliI32"].array(entries=[4, 1])
    assert rows.tolist() == [[-4] * 4, [-1]]


def test_a_split_object_is_put_back_together_for_the_entries_picked():
    with opened("small-evnt-tree-fullsplit") as handle:
        rows = handle["tree"]["evt"].array(entries=[42, 7])
    assert [row["Beg"] for row in rows] == ["beg-042", "beg-007"]
    assert [row["I32"] for row in rows] == [42, 7]


def test_iterating_over_entries_walks_them_a_step_at_a_time(lists):
    with opened("pod-advanced") as handle:
        batches = handle["orange"].iterate(
            ["orange.Evtake_iwant"], step=2, entries=np.array([1, 2, 80, 99, 50])
        )
        assert [batch["orange.Evtake_iwant"].tolist() for batch in batches] == [
            [1, 2],
            [80, 99],
            [50],
        ]


def test_selected_entries_are_checked_against_the_tree_they_are_for():
    assert selected(np.array([0, 4]), 5).tolist() == [0, 4]
    assert selected(np.array([], dtype=np.int64), 5).tolist() == []


def test_a_list_made_over_a_chain_picks_each_file_s_entries_out_of_the_chain(tmp_path):
    from xrdroot import chain

    per_file = entry_list(
        "per_file",
        "",
        lists=[
            ("TEntryList", entry_list("a", "tree", listed(1, 3), file_name="/x/chain.flat.1.root")),
            ("TEntryList", entry_list("b", "tree", listed(0), file_name="chain.flat.2.root")),
        ],
    )
    path = craft(tmp_path / "per_file.root", [("TEntryList", "per_file", per_file)])
    with open_root(str(path)) as handle, chain("tree", [DATA / "chain.flat.*.root"]) as made:
        kept = handle["per_file"]
        assert made.arrays(["I32"], entries=kept)["I32"].tolist() == [-1, -3, -5]
        assert made["I32"].array(entries=kept).tolist() == [-1, -3, -5]
