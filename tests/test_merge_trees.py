"""Merging trees: baskets copied across as they are, or entries read and written.

The fast way is checked where it matters - that every column of the merged
tree is every input's column one after another, that the baskets went across
byte for byte and are as many as the inputs had, and that it copes with what
ROOT's own files hold: wide basket keys, baskets kept inside the branch,
counters of their own type, strings. So is each way a basket has to be
packed again - another compression, a key that grows - and each way a tree
is refused, by name, rather than merged wrongly.
"""

from __future__ import annotations

import io
import pathlib
import shutil
import struct

import numpy as np
import pytest

import xrdroot
from xrdroot import Jagged, UnsupportedFeatureError, create, merge, open_root
from xrdroot import writer as writing
from xrdroot.buffer import Buffer
from xrdroot.file import Key
from xrdroot.merging.baskets import _shifted
from xrdroot.merging.trees import TreeMerge, exact_spec, normal_codes, plan

DATA = pathlib.Path(__file__).parent / "data"

#: Carrying over what is not merged is said in a warning, which the tests that
#: are about it ask for by name; everywhere else it is only noise.
pytestmark = pytest.mark.filterwarnings("ignore::xrdroot.MergeWarning")
#: Two halves of one tree, ROOT 6 wrote them, every leaf type and counted runs.
HALVES = [DATA / "chain.flat.1.root", DATA / "chain.flat.2.root"]


def joined(parts):
    first = parts[0]
    if isinstance(first, np.ndarray):
        return np.concatenate(parts)
    if isinstance(first, Jagged):
        return Jagged.join(parts)
    return [value for part in parts for value in part]


def same(one, other) -> bool:
    if isinstance(one, np.ndarray):
        return bool(np.array_equal(one, other))
    return bool(one == other)


def columns(path, name="tree"):
    with open_root(str(path)) as f:
        tree = f[name]
        return {branch: tree[branch].array() for branch in tree.keys()}, tree.num_entries


def require_concatenated(out, inputs, name="tree"):
    got, entries = columns(out, name)
    parts = [columns(path, name) for path in inputs]
    assert entries == sum(count for _, count in parts)
    assert list(got) == list(parts[0][0])
    for branch, values in got.items():
        assert same(values, joined([part[branch] for part, _ in parts])), branch


def small(path, **options) -> pathlib.Path:
    """A file of one tree ``events`` written here, a number, a string and a counted run."""
    rows = options.pop("rows", [[1.5], [], [2.5, 3.5]])
    with create(str(path), **options) as out:
        out["events"] = {
            "x": np.arange(len(rows), dtype=np.float64),
            "label": [f"entry {i}" for i in range(len(rows))],
            "jets": [np.asarray(row, np.float32) for row in rows] or Jagged(np.zeros(0), [0]),
        }
    return path


# -- the fast way -------------------------------------------------------------


def test_trees_root_wrote_merge_basket_for_basket_with_every_column_intact(tmp_path):
    out = tmp_path / "out.root"
    merged = merge(out, HALVES)
    assert merged.objects == {"tree": "tree"}
    assert merged.baskets.verbatim == 70 and merged.baskets.repacked == 0
    assert merged.entries == 0  # nothing went the slow way
    require_concatenated(out, HALVES)
    with open_root(str(out)) as f:
        assert all(branch.num_baskets == 2 for branch in f["tree"].branches.values())


def test_a_basket_copied_as_it_is_is_the_very_bytes_it_was(tmp_path):
    out = tmp_path / "out.root"
    merge(out, HALVES[:1])
    with open_root(str(HALVES[0])) as f, open_root(str(out)) as g:
        for name in ("F64", "Str", "SliF32"):
            before, after = f["tree"][name].record, g["tree"][name].record
            old = f._source.read(before.basket_seek[0], before.basket_bytes[0])
            new = g._source.read(after.basket_seek[0], after.basket_bytes[0])
            skip = Key(Buffer(old)).keylen
            assert old[skip:] == new[skip:], name
            assert Key(Buffer(new)).keylen == skip  # the key kept its wide form


def test_the_merged_tree_says_what_its_counters_and_strings_are_at_most(tmp_path):
    out = tmp_path / "out.root"
    merge(out, HALVES)
    with open_root(str(out)) as f:
        tree = f["tree"]
        counts = tree["N"].array()
        assert tree["N"].leaf.maximum == counts.max()
        assert tree["Str"].leaf.length == max(len(text) for text in tree["Str"].array()) + 1
        assert tree["SliF32"].leaf.count is tree["N"].leaf


def test_a_counter_of_its_own_type_is_merged_as_that_type(tmp_path):
    inputs = []
    for index, top in enumerate((200, 3)):
        path = tmp_path / f"in{index}.root"
        with create(str(path)) as out:
            tree = out.tree("t", {"v": ("h", "n")}, counters={"n": "B"})
            tree.extend({"v": [np.arange(top) % 7, np.arange(2)]})
        inputs.append(path)
    out = tmp_path / "out.root"
    assert merge(out, inputs).baskets.verbatim == 4
    with open_root(str(out)) as f:
        tree = f["t"]
        assert tree["n"].leaf.classname == "TLeafB" and tree["n"].leaf.unsigned
        assert tree["n"].leaf.maximum == -56  # 200, as the byte it is read as
        assert tree["n"].array().tolist() == [200, 2, 3, 2]
    require_concatenated(out, inputs, "t")


def test_a_counter_that_is_not_an_integer_type_is_refused(tmp_path):
    with create(str(tmp_path / "x.root")) as out:
        with pytest.raises(ValueError, match=r"counter 'n' is declared as 'f'"):
            out.tree("t", {"v": ("h", "n")}, counters={"n": "f"})


def test_baskets_kept_inside_their_branch_become_baskets_of_their_own(tmp_path):
    out = tmp_path / "out.root"
    xrdroot.copy(DATA / "g4-like.root", out, "mytree", columns=["i32", "f64"])
    got, entries = columns(out, "mytree")
    want, _ = columns(DATA / "g4-like.root", "mytree")
    assert entries == 5
    assert all(same(got[name], want[name]) for name in ("i32", "f64"))
    with open_root(str(out)) as f:
        assert f["mytree"]["i32"].record.basket_seek  # on file now, not in the branch


def test_a_string_column_kept_inside_its_branch_keeps_its_table_of_entries(tmp_path):
    source = tmp_path / "in.root"
    small(source)
    out = tmp_path / "out.root"
    with open_root(str(source)) as f:
        tree = f["events"]
        for name in ("label", "jets", "x"):
            record = tree[name].record
            keyed = [xrdroot.tree.Basket.keyed(f._source, s, n, record.entry_offset_len > 0)
                     for s, n in zip(record.basket_seek, record.basket_bytes)]
            record.baskets = keyed  # as if the tree had never flushed them
        with create(str(out)) as written:
            merging = TreeMerge(written, "events", tree, "'events'")
            merging.add(tree, "'events'", verbatim=True)
    require_concatenated(out, [source], "events")


def test_trees_compressed_otherwise_are_packed_again_without_decoding_an_entry(tmp_path):
    out = tmp_path / "out.root"
    merged = merge(out, HALVES, compression="lzma")
    assert merged.baskets.repacked == 70 and merged.baskets.verbatim == 0
    assert merged.entries == 0
    require_concatenated(out, HALVES)
    with open_root(str(out)) as f:
        assert f.compression == 206


def test_keeping_the_compression_copies_baskets_whatever_they_were(tmp_path):
    out = tmp_path / "out.root"
    merged = merge(out, HALVES, compression=505, keep_compression=True)
    assert merged.baskets.verbatim == 70
    require_concatenated(out, HALVES)


def test_a_file_that_grows_past_2_gb_moves_the_tables_of_small_keyed_baskets(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(writing, "BIG", 0)  # every record past it: every key wide
    inputs = [DATA / "join1.root", DATA / "join2.root".replace("join2", "join1")]
    out = tmp_path / "out.root"
    merged = merge(out, inputs)
    # the number and the int copied as they were, their ends moved; the strings repacked
    assert merged.baskets.verbatim == 4 and merged.baskets.repacked == 2
    monkeypatch.undo()
    require_concatenated(out, inputs, "j1")


def test_a_tree_renamed_on_the_way_has_its_baskets_keys_made_to_fit(tmp_path):
    out = tmp_path / "out.root"
    xrdroot.copy(HALVES[0], out, {"tree": "a/much/longer/place/for/the/tree"})
    got, _ = columns(out, "a/much/longer/place/for/the/tree")
    want, _ = columns(HALVES[0])
    assert all(same(got[name], want[name]) for name in want)


def test_a_table_of_entries_moves_along_but_its_last_empty_slot_does_not():
    table = struct.pack(">i3i", 3, 70, 75, 0) + b"tail"
    assert _shifted(table, 8) == struct.pack(">i3i", 3, 78, 83, 0) + b"tail"
    assert _shifted(b"", 8) == b""


def test_a_basket_written_with_ioi_features_is_refused_by_name(tmp_path):
    source = tmp_path / "features.root"
    shutil.copy(HALVES[0], source)
    with open_root(str(source)) as f:
        record = f["tree"]["I32"].record
        seek = record.basket_seek[0]
        key = Key(Buffer(f._source.read(seek, record.basket_bytes[0])))
    data = bytearray(source.read_bytes())
    at = seek + key.keylen - 19 + 2 + 4  # the basket's fNevBufSize, in its key
    struct.pack_into(">i", data, at, -4)
    source.write_bytes(bytes(data))
    with pytest.raises(UnsupportedFeatureError, match=r"I/O features.*fast=False"):
        merge(tmp_path / "out.root", [source])
    assert not (tmp_path / "out.root").stat().st_size  # taken back whole


def test_the_basket_size_a_branch_says_is_the_size_its_baskets_were_built_with(tmp_path):
    out = tmp_path / "out.root"
    merge(out, [small(tmp_path / "in.root")])
    with open_root(str(out)) as f:
        record = f["events"]["x"].record
        raw = f._source.read(record.basket_seek[0], record.basket_bytes[0])
        buf = Buffer(raw)
        Key(buf)
        buf.i16()
        assert buf.i32() == writing.BASKET_BYTES


# -- the slow way, and deciding between them ----------------------------------


def test_leaves_this_writer_does_not_make_go_the_slow_way_and_read_back_the_same(tmp_path):
    source = DATA / "leaves.root"
    out = tmp_path / "out.root"
    merged = merge(out, [source, source])
    assert merged.baskets == (0, 0, 0)
    assert merged.entries == 20
    got, entries = columns(out)
    want, _ = columns(source)
    assert entries == 20
    for name, values in want.items():
        assert same(got[name], joined([values, values])), name
    with open_root(str(out)) as f:
        assert f["tree"]["G64"].leaf.classname == "TLeafL"  # the nearest this writer makes


def test_asking_for_the_slow_way_reads_and_writes_every_entry(tmp_path):
    out = tmp_path / "out.root"
    merged = merge(out, HALVES, fast=False)
    assert merged.baskets == (0, 0, 0) and merged.entries == 10
    require_concatenated(out, HALVES)


def test_a_tree_with_nothing_in_it_merges_into_a_tree_with_nothing_in_it(tmp_path):
    empty = small(tmp_path / "empty.root", rows=[])
    out = tmp_path / "out.root"
    merge(out, [empty, empty], fast=False)
    with open_root(str(out)) as f:
        assert f["events"].num_entries == 0
        assert set(f["events"].keys()) == {"x", "label", "njets", "jets"}
    computed = tmp_path / "computed.root"
    xrdroot.copy(empty, computed, "events", columns={"twice": "2 * x"})
    with open_root(str(computed)) as f:
        assert f["events"].num_entries == 0 and f["events"].keys() == ["twice"]


def test_one_input_can_go_fast_and_the_next_slow_into_the_same_tree(tmp_path):
    first, second = small(tmp_path / "a.root"), small(tmp_path / "b.root")
    out = tmp_path / "out.root"
    with open_root(str(first)) as f, open_root(str(second)) as g, create(str(out)) as w:
        merging = TreeMerge(w, "events", f["events"], "'events'")
        merging.add(f["events"], "a", verbatim=True)
        slow = g["events"]
        slow["x"].record.branches = [slow["x"].record]  # not a branch made the same way
        merging.add(slow, "b", verbatim=True)
        merging.add(f["events"], "a again", verbatim=True)
        assert merging.moved.verbatim == 8 and merging.slow_entries == 3
    require_concatenated(out, [first, second, first], "events")


def test_a_tree_whose_branches_start_part_way_through_goes_the_slow_way(tmp_path):
    source = small(tmp_path / "a.root")
    with open_root(str(source)) as f, create(io.BytesIO()) as w:
        tree = f["events"]
        merging = TreeMerge(w, "events", tree, "'events'")
        tree["x"].record.first_entry = 1
        assert not merging._fits(tree)
        tree["x"].record.first_entry = 0
        tree["x"].record.entries = 2
        assert not merging._fits(tree)


def test_a_basket_from_elsewhere_is_not_put_before_entries_waiting_for_one(tmp_path):
    with create(io.BytesIO()) as out:
        tree = out.tree("t", {"x": float})
        tree.fill(x=1.0)
        with pytest.raises(ValueError, match=r"land before them"):
            tree._adopt(tree._branch("x"), (100, 10), 1, 10)
        with pytest.raises(KeyError, match=r"'y' is not a branch of 't'"):
            tree._branch("y")


def test_a_cut_keeps_the_entries_that_pass_and_their_counters_with_them(tmp_path):
    out = tmp_path / "out.root"
    xrdroot.copy(HALVES[0], out, "tree", cut="N >= 2", columns=["N", "SliF64", "Str"])
    with open_root(str(out)) as f, open_root(str(HALVES[0])) as g:
        tree, source = f["tree"], g["tree"]
        chosen = source["N"].array() >= 2
        assert tree.keys() == ["N", "SliF64", "Str"]
        assert tree["N"].array().tolist() == source["N"].array()[chosen].tolist()
        assert tree["SliF64"].array() == source["SliF64"].array().take(np.flatnonzero(chosen))
        assert tree["SliF64"].leaf.count is tree["N"].leaf


def test_a_cut_nothing_passes_still_writes_the_tree(tmp_path):
    out = tmp_path / "out.root"
    xrdroot.copy(HALVES[0], out, "tree", cut="N > 100", columns=["N", "F32"])
    with open_root(str(out)) as f:
        assert f["tree"].num_entries == 0 and f["tree"].keys() == ["N", "F32"]
    again = tmp_path / "again.root"
    xrdroot.copy(HALVES[0], again, "tree", cut="N > 100", columns={"half": "F64 / 2"})
    with open_root(str(again)) as f:
        assert f["tree"].num_entries == 0 and f["tree"]["half"].typename == "float64"


def test_columns_computed_from_expressions_are_written_under_their_names(tmp_path):
    out = tmp_path / "out.root"
    xrdroot.copy(HALVES[0], out, "tree", columns={"twice": "2 * F64", "count": "N"})
    with open_root(str(out)) as f, open_root(str(HALVES[0])) as g:
        assert f["tree"]["twice"].array().tolist() == (2 * g["tree"]["F64"].array()).tolist()
        assert f["tree"]["count"].array().tolist() == g["tree"]["N"].array().tolist()


def test_a_run_chosen_without_its_counter_is_given_one_of_its_own(tmp_path):
    with open_root(str(HALVES[0])) as f:
        chosen = plan(f["tree"], ["SliI16", "F32"])
    assert chosen.inexact == ["SliI16"] and chosen.counters == {}


def test_a_counter_this_writer_does_not_make_the_same_way_leaves_its_runs_their_own():
    with open_root(str(HALVES[0])) as f:
        tree = f["tree"]
        tree["N"].leaf.classname = "TLeafG"  # as though ROOT had counted with a long
        chosen = plan(tree, ["N", "SliI16"])
    assert chosen.counters == {} and chosen.inexact == ["N", "SliI16"]


def test_a_counter_chosen_without_the_runs_it_counts_is_a_column_of_numbers(tmp_path):
    with open_root(str(HALVES[0])) as f:
        chosen = plan(f["tree"], ["N", "F32"])
    assert chosen.columns == {"N": "i", "F32": "f"} and not chosen.inexact


def test_what_this_writer_makes_the_same_way_is_told_from_what_it_does_not():
    with open_root(str(DATA / "leaves.root")) as f:
        tree = f["tree"]
        counted = {id(branch.leaf): name for name, branch in tree.branches.items()}
        assert exact_spec(tree["ArrU16"], counted) == ("H", 10)
        assert exact_spec(tree["SliU8"], counted) == ("B", "N")
        assert exact_spec(tree["Str"], counted) is str
        assert exact_spec(tree["G64"], counted) is None
        assert exact_spec(tree["SliI8"], {}) is None  # its counter not among them
    with open_root(str(DATA / "padding.root")) as f:
        assert exact_spec(f["tree"]["pad.x1"], {}) is None  # one leaf of several
    with open_root(str(DATA / "small-evnt-tree-fullsplit.root")) as f:
        assert exact_spec(f["tree"]["evt"], {}) is None


def test_a_string_counted_by_another_branch_is_not_one_this_writer_makes():
    with open_root(str(DATA / "leaves.root")) as f:
        branch = f["tree"]["Str"]
        branch.leaf.count = f["tree"]["N"].leaf
        assert exact_spec(branch, {}) is None


def test_a_file_s_compression_is_read_the_same_whatever_its_vintage():
    assert normal_codes(1) == 101
    assert normal_codes(0) == normal_codes(300) == 0
    assert normal_codes(505) == 505


# -- refusals -------------------------------------------------------------------


def test_trees_with_different_branches_are_refused_naming_the_branches(tmp_path):
    other = tmp_path / "other.root"
    with create(str(other)) as out:
        out["tree"] = {"F64": np.zeros(2), "extra": np.zeros(2, np.int32)}
    with pytest.raises(ValueError, match=r"extra \(int32 here, missing in the first\)"):
        merge(tmp_path / "out.root", [HALVES[0], other])


def test_a_split_object_or_a_leaf_list_is_refused_rather_than_dropped(tmp_path):
    with pytest.raises(UnsupportedFeatureError, match=r"evt: a split object"):
        merge(tmp_path / "a.root", [DATA / "small-evnt-tree-fullsplit.root"])
    with pytest.raises(UnsupportedFeatureError, match=r"pad.x1: one leaf of a branch of several"):
        merge(tmp_path / "b.root", [DATA / "padding.root"])


def test_a_column_this_writer_cannot_write_refuses_the_tree_naming_it(tmp_path):
    with pytest.raises(UnsupportedFeatureError, match=r"'tree' in .*tdatime.root cannot be"):
        merge(tmp_path / "out.root", [DATA / "tdatime.root"], only_keys=["tree"])


def test_a_tree_merged_into_a_file_being_added_to_leaves_its_baskets_where_they_are(tmp_path):
    out = tmp_path / "out.root"
    shutil.copy(HALVES[0], out)
    before = out.stat().st_size
    merged = merge(out, HALVES[1:], append=True)
    assert merged.baskets.in_place == 35 and merged.baskets.verbatim == 35
    require_concatenated(out, HALVES)
    assert out.stat().st_size < before + HALVES[1].stat().st_size + 20_000
