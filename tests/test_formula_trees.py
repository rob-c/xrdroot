"""Expressions and cuts read from trees ROOT wrote, as ``arrays`` reads them.

The expected values are go-hep's ``rdraw`` tests, which draw from
``small-flat-tree.root``: for its ``i``-th of a hundred entries it holds
``Int32 = i``, ``N = i % 10``, a slice of ``N`` copies of ``i`` and an array of
ten copies of ``i``. That is enough to say exactly what every expression here
should give, and each is checked against NumPy doing the same by hand.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from xrdroot import FormulaError, Jagged, chain, compile_formula, create, open_root
from xrdroot.formula.select import TreeNames, dimensions

DATA = pathlib.Path(__file__).parent / "data"

#: What small-flat-tree.root holds, worked out: entry i has i % 10 copies of i in its slice.
ENTRY = np.arange(100)
SLICE = [[float(i)] * (i % 10) for i in range(100)]


@pytest.fixture
def flat():
    with open_root(str(DATA / "small-flat-tree.root")) as f:
        yield f["tree"]


@pytest.fixture
def split():
    with open_root(str(DATA / "small-evnt-tree-fullsplit.root")) as f:
        yield f["tree"]


def test_a_fixed_array_is_looped_over_ten_values_an_entry(flat):
    values = flat.arrays(["ArrayFloat64 * 1"])["ArrayFloat64 * 1"]
    assert isinstance(values, Jagged)
    assert len(values.flat) == 1000
    assert values.flat.mean() == pytest.approx(np.repeat(ENTRY, 10).mean())


def test_a_slice_is_looped_over_its_own_length_an_entry(flat):
    values = flat.arrays(["SliceFloat64 + 0"])["SliceFloat64 + 0"]
    assert values.tolist() == SLICE


def test_an_index_picks_one_element_and_one_value_per_entry(flat):
    got = flat.arrays(["ArrayFloat64[0]", "Float64"])
    assert got["ArrayFloat64[0]"].tolist() == got["Float64"].tolist()


def test_the_reducers_make_one_value_per_entry_of_the_slice(flat):
    got = flat.arrays(["Length$(SliceFloat64)", "Sum$(SliceFloat64)", "Max$(SliceFloat64)"])
    assert got["Length$(SliceFloat64)"].tolist() == [len(row) for row in SLICE]
    assert got["Sum$(SliceFloat64)"].tolist() == [sum(row) for row in SLICE]
    assert got["Max$(SliceFloat64)"].tolist() == [max(row, default=0) for row in SLICE]


def test_a_cut_over_elements_keeps_elements(flat):
    got = flat.arrays(["ArrayFloat64", "Int32"], cut="ArrayFloat64 > 50")
    assert got["ArrayFloat64"].lengths().tolist() == [10] * 49
    assert got["Int32"].tolist() == list(range(51, 100))


def test_a_cut_over_the_entry_keeps_entries_and_all_their_elements(flat):
    got = flat.arrays(["SliceFloat64"], cut="Length$(SliceFloat64) >= 5")
    assert got["SliceFloat64"].tolist() == [row for row in SLICE if len(row) >= 5]
    assert (
        flat.arrays(["SliceFloat64"], cut="Length$(SliceFloat64) == 0")["SliceFloat64"].tolist()
        == [[]] * 10
    )


def test_a_number_per_entry_goes_with_every_element(flat):
    values = flat.arrays(["ArrayFloat64 - Float64"])["ArrayFloat64 - Float64"]
    assert len(values.flat) == 1000 and not values.flat.any()


def test_two_collections_of_different_lengths_pair_up_to_the_shorter(flat):
    values = flat.arrays(["SliceFloat64 + ArrayFloat64"])["SliceFloat64 + ArrayFloat64"]
    assert values.tolist() == [[2 * x for x in row] for row in SLICE]


def test_entry_and_iteration_say_where_the_loop_is(flat):
    got = flat.arrays(["Entry$ - Int32", "Iteration$ + ArrayFloat64*0"], 10, 20)
    assert got["Entry$ - Int32"].tolist() == [0] * 10
    assert got["Iteration$ + ArrayFloat64*0"].tolist() == [list(range(10))] * 10


def test_a_cut_on_a_range_and_on_entries_numbered(flat):
    got = flat.arrays(["Int32 * 2"], 20, 40, cut="Int32 % 2 == 0")
    assert got["Int32 * 2"].tolist() == list(range(40, 80, 4))
    picked = flat.arrays(["Entry$", "N"], entries=[3, 97, 50], cut="N > 0")
    assert picked["Entry$"].tolist() == [3, 97] and picked["N"].tolist() == [3, 7]


def test_strings_compare_and_select(flat):
    got = flat.arrays(["Str", 'strstr(Str, "-00")'], cut='Str == "evt-003" || Str == "evt-042"')
    assert got["Str"] == ["evt-003", "evt-042"]
    assert got['strstr(Str, "-00")'].tolist() == [True, False]


def test_only_the_branches_the_expressions_need_are_read(flat, monkeypatch):
    asked = []
    original = type(flat["Int32"]).array

    def counting(self, *args, **kwargs):
        asked.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(flat["Int32"]), "array", counting)
    flat.arrays(["Int32 + Float64", "Int32 * 2"], cut="Float64 > 3")
    assert sorted(asked) == ["Float64", "Int32"]


def test_aliases_stand_for_expressions_in_names_and_cuts(flat):
    aliases = {"twice": "2 * Int32", "big": "twice > 150"}
    got = flat.arrays(["twice"], cut="big", aliases=aliases)
    assert got["twice"].tolist() == list(range(152, 200, 2))


def test_iterate_takes_expressions_and_cuts_batch_by_batch(flat):
    batches = list(flat.iterate(["Int32 + 1"], step=30, cut="N == 0"))
    assert [batch["Int32 + 1"].tolist() for batch in batches] == [
        [1, 11, 21],
        [31, 41, 51],
        [61, 71, 81],
        [91],
    ]
    picked = list(flat.iterate(["N"], step=2, entries=[1, 2, 3], cut="N != 2"))
    assert [batch["N"].tolist() for batch in picked] == [[1], [3]]


def test_a_split_object_s_members_are_named_with_or_without_the_object(split):
    got = split.arrays(["evt.P3.Px + P3.Pz", "ArrayI16[2]", "StlVecF64.size()"], 0, 3)
    assert got["evt.P3.Px + P3.Pz"].tolist() == [-2, 0, 2]
    assert got["ArrayI16[2]"].tolist() == [0, 1, 2]
    assert got["StlVecF64.size()"].tolist() == [0, 1, 2]
    got = split.arrays(['StlVecStr == "vec-002"', "Sum$(SliceU64)"], 0, 3)
    assert got['StlVecStr == "vec-002"'].tolist() == [[], [False], [True, True]]
    assert got["Sum$(SliceU64)"].tolist() == [0, 1, 4]


def test_a_misspelled_branch_is_refused_with_the_ones_near_it(split):
    with pytest.raises(FormulaError, match=r"not a member of 'P3' .* the nearest are 'P3\.Px'"):
        split.arrays(["P3.Pxx * 2"])


def test_the_dimensions_of_a_branch_come_from_its_type_and_title(split):
    names = TreeNames(split)
    assert names["I32"] == 0
    assert names["ArrayI16[10]"] == 1
    assert names["SliceI16"] == 1
    assert names["StlVecStr"] == 1
    assert names["evt"] == 0
    assert "P3.Px" in names and len(names) == len(split.keys())
    assert compile_formula("SliceI16 + I32", names).per_element


def test_vectors_of_vectors_from_a_file_loop_twice():
    with open_root(str(DATA / "std-containers-split00.root")) as f:
        tree = f.tree()
        assert dimensions(tree["vec_vec_i32"]) == 2
        got = tree.arrays(["vec_vec_i32 * 1", "@vec_vec_i32.size()", 'vec_str == "two"'])
        assert got["vec_vec_i32 * 1"].tolist() == [[-1], [-1, -1, -2]]
        assert got["@vec_vec_i32.size()"].tolist() == [1, 2]
        assert got['vec_str == "two"'].tolist() == [[False], [False, True]]
        got = tree.arrays(["vec_i32", "vec_vec_i32[1]"], cut="vec_i32 < 0")
        assert got["vec_i32"].tolist() == [[-1], [-1, -2]]


def test_a_title_s_dimensions_shape_what_the_file_gives_flat(tmp_path):
    path = tmp_path / "shaped.root"
    with create(str(path)) as f:
        f["t"] = {
            "m": np.arange(18.0).reshape(2, 9),
            "x": Jagged(np.arange(12.0), np.array([0, 6, 12])),
        }
    with open_root(str(path)) as f:
        tree = f["t"]
        tree["m"].leaf.title = "m[3][3]"
        tree["x"].leaf.title = "x[n][3]"
        assert dimensions(tree["m"]) == 2 and dimensions(tree["x"]) == 2
        got = tree.arrays(["m[1][2]", "m[][0]", "x[1][0]", "Sum$(x[][2])", "m"])
        assert got["m[1][2]"].tolist() == [5, 14]
        assert got["m[][0]"].tolist() == [[0, 3, 6], [9, 12, 15]]
        assert got["x[1][0]"].tolist() == [3, 9]
        assert got["Sum$(x[][2])"].tolist() == [7, 19]
        assert got["m"].shape == (2, 9)  # a branch asked for by name is as the file has it
        got = tree.arrays(["m", "x"], cut="m > 13")
        assert got["m"].tolist() == [[14, 15, 16, 17]] and got["x"].tolist() == [[11.0]]


def test_a_chain_counts_entry_across_its_files_and_local_entry_within_each():
    files = [str(DATA / "chain.flat.1.root"), str(DATA / "chain.flat.2.root")]
    with chain("tree", files) as events:
        got = events.arrays(["Entry$", "LocalEntry$", "I32 + Entry$", "Entries$"], 3, 8)
        assert got["Entry$"].tolist() == [3, 4, 5, 6, 7]
        assert got["LocalEntry$"].tolist() == [3, 4, 0, 1, 2]
        assert got["I32 + Entry$"].tolist() == [0] * 5
        assert got["Entries$"].tolist() == [10] * 5
        batches = list(events.iterate(["N"], step=4, cut="Sum$(SliF64) > 20"))
        assert [batch["N"].tolist() for batch in batches] == [[], [5, 6, 7], [8, 9]]


def test_a_friend_s_columns_are_there_to_an_expression():
    with open_root(str(DATA / "join1.root")) as one, open_root(str(DATA / "join2.root")) as two:
        tree = one["j1"]
        tree.add_friend(two["j2"], "second")
        got = tree.arrays(["second.b20 - b10", "b20 * 1"], 0, 3)
        assert got["second.b20 - b10"].tolist() == [100.0] * 3
        assert got["b20 * 1"].tolist() == [201.0, 202.0, 203.0]


def test_a_tree_s_expression_names_what_it_refuses(flat):
    with pytest.raises(FormulaError, match="could not be parsed"):
        flat.arrays(["Int32 +"])
    with pytest.raises(FormulaError, match="could not be parsed"):
        flat.arrays(["Int32"], cut="Int32 +")
    got = flat.arrays(cut="Int32 < 2")
    assert got["Int32"].tolist() == [0, 1] and len(got) == len(flat.readable())


def test_a_cut_can_hand_its_columns_to_pandas(flat):
    pd = pytest.importorskip("pandas")
    frame = flat.arrays(["Int32", "Float64 / 2"], cut="Int32 >= 98", library="pd")
    assert isinstance(frame, pd.DataFrame)
    assert frame["Float64 / 2"].tolist() == [49.0, 49.5]
