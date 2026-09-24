"""Chains: one tree across many files, read end to end.

``chain.flat.1.root`` and ``chain.flat.2.root`` are ROOT's own, from go-hep:
the same thirty-five columns in each, five entries apiece, with every value
the negative of its entry number counted across both files - so a range that
crosses from one into the other says by its values alone whether it did.
"""

from __future__ import annotations

import pathlib
import pickle
import shutil

import numpy as np
import pytest

import xrdroot
from support import plain
from xrdroot import Chain, Jagged, UnsupportedFeatureError, chain, create, open_root

DATA = pathlib.Path(__file__).parent / "data"
FLAT = [str(DATA / "chain.flat.1.root"), str(DATA / "chain.flat.2.root")]


@pytest.fixture
def flat():
    with chain("tree", FLAT) as made:
        yield made


def test_a_chain_is_as_long_as_its_files_put_together(flat):
    assert (len(flat), flat.counts, flat.num_entries) == (10, [5, 5], 10)
    assert flat.files == FLAT
    assert flat.starts() == [0, 5, 10]
    assert repr(flat) == "<Chain 'tree' over 2 files>"
    assert flat.name == "tree"


def test_a_chain_names_its_columns_as_its_first_file_does(flat):
    assert flat.keys()[:3] == ["B", "Str", "I8"]
    assert list(flat)[:3] == ["B", "Str", "I8"]
    assert "I32" in flat and "nothing" not in flat
    assert flat.typenames()["F64"] == "float64"
    assert flat.show().splitlines()[4].split() == ["I32", "int32"]
    assert "ArrI32" in flat.readable()


def test_a_range_across_the_boundary_reads_from_both_files(flat):
    assert flat["I32"].array(3, 7).tolist() == [-3, -4, -5, -6]
    assert flat["Str"].array(4, 6) == ["str-4", "str-5"]
    assert flat["ArrI32"].array(4, 6).tolist() == [[-4] * 10, [-5] * 10]
    assert flat["I32"].array(-2).tolist() == [-8, -9]  # counted from the end, as in Python


def test_rows_of_different_lengths_carry_their_offsets_on_into_the_next_file(flat):
    rows = flat["SliI32"].array(3, 7)
    assert isinstance(rows, Jagged)
    assert rows.tolist() == [[-3] * 3, [-4] * 4, [-5] * 5, [-6] * 6]
    assert rows.offsets.tolist() == [0, 3, 7, 12, 18]


def test_an_empty_range_is_an_empty_column_of_the_right_kind(flat):
    assert flat["I32"].array(7, 7).tolist() == []
    assert flat["I32"].array(7, 7).dtype == np.int32
    assert flat["SliI32"].array(20, 30).tolist() == []


def test_a_batch_of_a_chain_runs_off_the_end_of_one_file_into_the_next(flat):
    batches = [batch["I32"].tolist() for batch in flat.iterate(["I32"], step=3)]
    assert batches == [[0, -1, -2], [-3, -4, -5], [-6, -7, -8], [-9]]
    ranged = list(flat.iterate(["I32"], step=4, entry_start=2, entry_stop=-1))
    assert [batch["I32"].tolist() for batch in ranged] == [[-2, -3, -4, -5], [-6, -7, -8]]


def test_a_chain_asked_for_nothing_in_particular_reads_every_readable_column(flat):
    everything = flat.arrays(entry_start=4, entry_stop=6)
    assert everything["I64"].tolist() == [-4, -5]
    assert len(everything) == len(flat.readable())


def test_a_chain_hands_its_columns_to_pandas_like_a_tree():
    with chain("tree", FLAT) as made:
        frame = made.arrays(["I32", "F64"], 4, 6, library="pd")
    assert frame["I32"].tolist() == [-4, -5]


def test_a_step_of_nothing_is_refused(flat):
    with pytest.raises(ValueError, match="at least one entry"):
        next(flat.iterate(step=0))


def test_a_glob_stands_for_every_file_it_matches_in_order():
    with chain("tree", [str(DATA / "chain.flat.*.root")]) as made:
        assert made.files == FLAT
        assert len(made) == 10


def test_a_glob_that_matches_nothing_is_refused_rather_than_an_empty_chain():
    with pytest.raises(FileNotFoundError, match="matches no file"):
        chain("tree", [str(DATA / "no-such-*.root")])


def test_a_chain_of_no_files_at_all_is_refused():
    with pytest.raises(ValueError, match="at least one file"):
        chain("tree", [])


def test_a_path_object_is_a_file_like_its_string(flat):
    with chain("tree", [pathlib.Path(FLAT[0])]) as made:
        assert len(made) == 5


def test_files_are_opened_only_when_something_needs_them():
    made = chain("tree", FLAT)
    assert all(link.file is None for link in made._links)
    made["I32"].array(0, 2)  # the first file, and the second for the check of its type
    made.close()
    assert all(link.file is None for link in made._links)


def test_a_file_already_open_is_used_as_it_is_and_left_open():
    with open_root(FLAT[0]) as first:
        made = chain("tree", [first, FLAT[1]])
        assert made["I32"].array(4, 6).tolist() == [-4, -5]
        assert made.files == FLAT
        made.close()
        assert first["tree"]["I32"].array(0, 1).tolist() == [0]  # still open


def test_a_chain_goes_to_another_process_as_where_its_files_are(flat):
    len(flat)
    arrived = pickle.loads(pickle.dumps(flat))
    try:
        assert arrived.counts == [5, 5]
        assert all(link.file is None for link in arrived._links)
        assert arrived["I32"].array(4, 6).tolist() == [-4, -5]
    finally:
        arrived.close()
    with open_root(FLAT[0]) as first:
        arrived = pickle.loads(pickle.dumps(chain("tree", [first])))
        assert arrived.files == [FLAT[0]]
        arrived.close()


def test_a_chain_of_a_file_object_cannot_go_to_another_process():
    with open(FLAT[0], "rb") as handle, open_root(handle) as opened:
        with pytest.raises(UnsupportedFeatureError, match="opened from a file object"):
            pickle.dumps(chain("tree", [opened]))


def test_the_same_whole_objects_in_two_files_come_back_as_one_list():
    with chain("tree", [DATA / "chain.1.root", DATA / "chain.2.root"]) as made:
        rows = made["evt"].array(9, 11)
    assert [row["Beg"] for row in rows] == ["beg-009", "beg-010"]
    assert plain(rows[1]["StlVecF64"]) == []


def test_a_branch_the_first_file_does_not_have_is_named_in_the_refusal(flat):
    with pytest.raises(KeyError, match="'nothing' is not a branch of 'tree'"):
        flat["nothing"]


def test_a_branch_a_later_file_does_not_have_says_which_file(tmp_path):
    with create(str(tmp_path / "a.root")) as out:
        out["t"] = {"x": np.arange(3, dtype=np.int32), "y": np.arange(3.0)}
    with create(str(tmp_path / "b.root")) as out:
        out["t"] = {"x": np.arange(2, dtype=np.int32)}
    with chain("t", [tmp_path / "a.root", tmp_path / "b.root"]) as made:
        with pytest.raises(KeyError, match=r"'y' is a branch of 't' in .*a\.root but not in"):
            made["y"].array()


def test_files_that_disagree_about_a_column_are_refused_by_name(tmp_path):
    with create(str(tmp_path / "a.root")) as out:
        out["t"] = {"x": np.arange(3, dtype=np.int32)}
    with create(str(tmp_path / "b.root")) as out:
        out["t"] = {"x": np.arange(3.0)}
    with chain("t", [tmp_path / "a.root", tmp_path / "b.root"]) as made:
        with pytest.raises(UnsupportedFeatureError, match=r"'x' holds int32 in .* and float64"):
            made["x"].array()
        with pytest.raises(UnsupportedFeatureError, match="reads a column as one type"):
            made["x"].pick(np.array([0]))


def test_a_chained_branch_says_what_it_is_and_how_long(flat):
    column = flat["I32"]
    assert (len(column), column.typename, column.name) == (10, "int32", "I32")
    assert repr(column) == "<ChainedBranch 'I32' over 2 files>"


def test_entries_picked_across_a_chain_come_back_in_the_order_asked(flat):
    assert flat["I32"].array(entries=[9, 0, 5, 4]).tolist() == [-9, 0, -5, -4]
    picked = flat.arrays(["I32", "SliI32"], entries=np.array([6, 2]))
    assert picked["I32"].tolist() == [-6, -2]
    assert picked["SliI32"].tolist() == [[-6] * 6, [-2] * 2]
    assert flat["I32"].array(entries=[]).tolist() == []


def test_a_chain_is_exported_from_the_package():
    assert xrdroot.chain is chain and xrdroot.Chain is Chain


def test_a_chain_copied_file_by_file_reads_the_same(tmp_path):
    for name in FLAT:
        shutil.copy(name, tmp_path)
    with chain("tree", [str(tmp_path / "*.root")]) as made:
        assert made["I32"].array().tolist() == list(range(0, -10, -1))
