"""Writing tree columns whose entries differ in size: rows of numbers, and text.

A row of varying length is laid out as ROOT lays out ``x[nx]/F`` - a counter
branch, a data leaf pointing at the counter's leaf, and baskets with a table
of where each entry begins - and a string as a ``TLeafC``. What is checked
here, beyond values going in and coming back out through this library's own
reader (which was written against files ROOT wrote): the counter comes before
what it counts and is the very leaf the data points at, the tables and the
leaf's bookkeeping hold what the ROOT 6.08 donor ``small-flat-tree.root``
holds in the same places, filling entry by entry and a column at a time make
the same baskets, and every way of giving a row wrongly is refused by name.
"""

from __future__ import annotations

import io
import pathlib
import struct

import awkward as ak
import numpy as np
import pytest

from test_root_wtree import at_branches, branch_fields, tree_record
from xrdroot import Jagged, create, open_root
from xrdroot.objects import CLASSES
from xrdroot.writer import _keylen
from xrdroot.wtree import (
    MIN_OFFSET_LEN,
    OFFSET_LEN,
    _adapted,
    _is_rows,
    _is_text,
    _room,
    _typecode,
    spec_of,
)

DATA = pathlib.Path(__file__).parent / "data"

#: Five entries of rows of different lengths, one of them empty.
ROWS = [[1.5, 2.5], [], [3.5], [4.5, 5.5, 6.5], [7.5]]


def written(columns, fill, *, basket_size=32_000, **options) -> bytes:
    """A file holding one tree ``events``, declared and then given ``fill``."""
    buf = io.BytesIO()
    with create(buf, **options) as out:
        fill(out.tree("events", columns, basket_size=basket_size))
    return buf.getvalue()


def extended(columns, arrays, **kwargs) -> bytes:
    return written(columns, lambda tree: tree.extend(arrays), **kwargs)


def read_back(data: bytes):
    return open_root(io.BytesIO(data))


def jagged(rows, dtype=np.float64) -> Jagged:
    lengths = [len(row) for row in rows]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    return Jagged(np.asarray([v for row in rows for v in row], dtype=dtype), offsets)


def _objects(rows) -> np.ndarray:
    """Rows in a one-dimensional object array, the way a frame's column of lists is."""
    out = np.empty(len(rows), dtype=object)
    for index, row in enumerate(rows):
        out[index] = row
    return out


def basket_bytes(data: bytes, branch: str, index: int = 0) -> tuple[bytes, int]:
    """One basket of an uncompressed file, whole: its bytes and its key's length."""
    with read_back(data) as back:
        record = back["events"][branch].record
        seek, nbytes = record.basket_seek[index], record.basket_bytes[index]
    raw = data[seek : seek + nbytes]
    return raw, struct.unpack_from(">h", raw, 14)[0]


# -- rows of numbers, in and out -------------------------------------------


def test_rows_of_different_lengths_come_back_as_they_went_in():
    data = written({"x": ("f", None)}, lambda tree: [tree.fill(x=row) for row in ROWS])
    with read_back(data) as back:
        tree = back["events"]
        assert tree.keys() == ["nx", "x"]
        assert tree["x"].is_jagged
        assert tree["x"].array().tolist() == ROWS
        assert tree["x"].array(1, 4).tolist() == ROWS[1:4]
        assert tree["nx"].array().tolist() == [2, 0, 1, 3, 1]
        assert tree.typenames() == {"nx": "int32", "x": "float32"}


@pytest.mark.parametrize(
    "given",
    [
        jagged(ROWS),
        [np.asarray(row) for row in ROWS],
        ROWS,
        ak.Array(ROWS),
        np.asarray([np.asarray(row) for row in ROWS], dtype=object),
    ],
    ids=["jagged", "arrays", "lists", "awkward", "objects"],
)
def test_a_column_of_rows_can_be_given_every_way_rows_are_kept(given):
    with read_back(extended({"x": ("d", None)}, {"x": given})) as back:
        assert back["events"]["x"].array().tolist() == ROWS


def test_a_rectangle_given_to_a_varying_column_is_rows_that_happen_to_agree():
    square = np.arange(6, dtype=np.int32).reshape(3, 2)
    with read_back(extended({"x": ("i", None)}, {"x": square})) as back:
        assert back["events"]["x"].array().tolist() == square.tolist()
        assert back["events"]["nx"].array().tolist() == [2, 2, 2]


def test_rows_of_every_kind_of_number_round_trip():
    columns = {code: (code, None) for code in "?bBhHiIqQfd"}
    rows = [[1, 0, 1], [], [0]]
    arrays = {code: [np.asarray(row, dtype=code) for row in rows] for code in columns}
    with read_back(extended(columns, arrays)) as back:
        tree = back["events"]
        for code in columns:
            assert tree[code].array().tolist() == rows, code


def test_columns_can_share_one_counter_by_naming_it():
    px, py = jagged(ROWS), jagged([[-v for v in row] for row in ROWS])
    data = extended(
        {"px": ("f", "nmu"), "py": ("f", "nmu"), "e": float},
        {"px": px, "py": py, "e": np.arange(5.0)},
    )
    with read_back(data) as back:
        tree = back["events"]
        assert tree.keys() == ["nmu", "px", "py", "e"]
        assert tree["py"].array() == py
        assert tree["nmu"].array().tolist() == [2, 0, 1, 3, 1]


def test_a_column_of_rows_fills_many_baskets_the_same_way_either_way_it_is_given():
    rng = np.random.default_rng(7)
    rows = [rng.normal(size=int(n)) for n in rng.integers(0, 6, 300)]
    columns = {"x": ("d", None), "s": str}
    words = [f"entry {i}" * (i % 4) for i in range(300)]

    def by_rows(tree):
        for row, word in zip(rows, words):
            tree.fill(x=row, s=word)

    def by_columns(tree):
        tree.extend({"x": rows[:17], "s": words[:17]})
        tree.fill(x=rows[17], s=words[17])
        tree.extend({"x": rows[18:], "s": words[18:]})

    with (
        read_back(written(columns, by_rows, basket_size=100)) as one,
        read_back(written(columns, by_columns, basket_size=100)) as two,
    ):
        for name in ("nx", "x", "s"):
            first, second = one["events"][name], two["events"][name]
            assert list(first.record.basket_entry) == list(second.record.basket_entry), name
            assert first.num_baskets > 10
        assert one["events"]["x"].array().tolist() == [row.tolist() for row in rows]
        assert two["events"]["s"].array() == words
        assert two["events"]["x"].array(290, 300).tolist() == [r.tolist() for r in rows[290:]]


def test_a_basket_goes_out_with_the_row_that_fills_it_and_empty_rows_wait():
    """A row of nothing takes no bytes, so one after a full basket starts the next."""
    rows = [[1.0, 2.0], [], [3.0, 4.0], []]
    data = extended({"x": ("d", None)}, {"x": rows}, basket_size=16)
    with read_back(data) as back:
        branch = back["events"]["x"]
        assert list(branch.record.basket_entry) == [0, 1, 3, 4]
        assert branch.array().tolist() == rows


# -- how a column of rows is laid out --------------------------------------


def test_the_data_leaf_points_at_the_very_leaf_of_its_counter():
    data = extended({"x": ("f", None)}, {"x": ROWS})
    buf, _version, end = tree_record(data)
    counter, rows = at_branches(buf).objarray(CLASSES)
    leaves = buf.objarray(CLASSES)
    assert rows.leaves[0].count is counter.leaves[0]
    assert [id(leaf) for leaf in leaves] == [id(counter.leaves[0]), id(rows.leaves[0])]
    assert counter.leaves[0].count is None
    buf.resume(end)


def test_a_column_of_rows_is_titled_the_way_root_spells_a_leaf_list():
    data = extended({"x": ("f", None), "y": ("d", "n")}, {"x": ROWS, "y": ROWS})
    with read_back(data) as back:
        tree = back["events"]
        assert tree["x"].record.title == "x[nx]/F"
        assert tree["x"].title == "x[nx]"
        assert tree["y"].record.title == "y[n]/D"
        assert tree["nx"].record.title == "nx/I"
        assert (tree["x"].leaf.length, tree["x"].leaf.etype) == (1, 4)


def test_a_counter_keeps_its_largest_count_as_root_does_and_nothing_else_does():
    data = extended({"x": ("f", None)}, {"x": ROWS})
    buf, _version, end = tree_record(data)
    counter, rows = at_branches(buf).objarray({"TBranch": _branch_and_leaf})
    assert counter["leaf"] == {"is_range": 1, "minimum": 0, "maximum": 3}
    assert rows["leaf"] == {"is_range": 0, "minimum": 0, "maximum": 0}
    assert counter["entry_offset_len"] == 0  # a counter's entries are all one size
    assert rows["entry_offset_len"] == 4 * len(ROWS)
    buf.resume(end)


def _branch_and_leaf(buf):
    """A branch's ``fEntryOffsetLen``, then its one leaf's range fields."""
    _version, end = buf.header()
    buf.skip_record()  # TNamed
    buf.skip_record()  # TAttFill
    buf.i32s(2)  # fCompress, fBasketSize
    entry_offset_len = buf.i32()
    buf.i32()  # fWriteBasket
    buf.i64()  # fEntryNumber
    buf.i32s(3)  # fOffset, fMaxBaskets, fSplitLevel
    buf.i64s(4)  # fEntries, fFirstEntry, fTotBytes, fZipBytes
    buf.skip_record()  # fBranches
    buf.header()  # fLeaves
    buf.tobject()
    buf.string()
    buf.i32s(2)  # how many, and from where
    buf.u32(), buf.u32()  # the leaf's byte count and new-class tag
    buf.cstring()
    buf.header()
    _version, inner = buf.header()
    buf.skip_record()  # TNamed
    buf.i32s(3)  # fLen, fLenType, fOffset
    is_range = buf.u8()
    buf.u8()  # fIsUnsigned
    buf.u32()  # fLeafCount
    buf.resume(inner)
    leaf = {"is_range": is_range, "minimum": buf.i32(), "maximum": buf.i32()}
    buf.resume(end)
    return {"entry_offset_len": entry_offset_len, "leaf": leaf}


def test_a_basket_of_rows_ends_in_the_table_root_writes_after_fLast():
    """``small-flat-tree.root`` writes one more slot than entries, the last zero."""
    data = extended({"x": ("d", None)}, {"x": ROWS}, compression=None)
    raw, keylen = basket_bytes(data, "x")
    assert keylen == _keylen("TBasket", "x", "events", extra=19)
    _version, _size, nevsize, nevbuf, last, flag = struct.unpack_from(">hiiiiB", raw, keylen - 19)
    assert (nevsize, nevbuf, flag) == (OFFSET_LEN, 5, 0)
    assert last == keylen + 8 * 7
    count = struct.unpack_from(">i", raw, last)[0]
    table = struct.unpack_from(f">{count}i", raw, last + 4)
    assert count == nevbuf + 1
    assert table == (*(keylen + 8 * at for at in (0, 2, 2, 3, 6)), 0)
    assert len(raw) == last + 4 + 4 * count


def test_a_branch_of_rows_counts_its_tables_into_its_bytes():
    data = extended({"x": ("d", None)}, {"x": ROWS}, compression=None)
    buf, _version, end = tree_record(data)
    _counter, rows = at_branches(buf).objarray({"TBranch": branch_fields})
    keylen = _keylen("TBasket", "x", "events", extra=19)
    assert rows["tot_bytes"] == keylen + 8 * 7 + 4 + 4 * 6
    assert rows["entry_offset_len"] == 4 * 5  # ROOT's guess, shrunk to fit
    buf.resume(end)


def test_the_guess_at_a_basket_table_moves_the_way_root_moves_it():
    assert _adapted(OFFSET_LEN, 100) == 400  # what small-flat-tree.root says
    assert _adapted(OFFSET_LEN, 2) == MIN_OFFSET_LEN
    assert _adapted(OFFSET_LEN, 3000) == 6000
    assert _adapted(OFFSET_LEN, 400) == OFFSET_LEN
    assert _adapted(MIN_OFFSET_LEN, 1) == MIN_OFFSET_LEN
    assert _room(OFFSET_LEN, 5) == OFFSET_LEN
    assert _room(OFFSET_LEN, 1000) == 2000
    assert _room(OFFSET_LEN, 5000) == 8000


def test_a_tree_of_rows_describes_its_classes_as_the_donor_does():
    with open_root(str(DATA / "small-flat-tree.root")) as donor:
        theirs = dict(donor._source.streamers())
    data = extended({"x": ("f", None), "s": str}, {"x": ROWS, "s": list("abcde")})
    with read_back(data) as back:
        ours = back._source.streamers()
    for classname in ("TLeafI", "TLeafF", "TLeafC"):
        assert classname in ours
        for name, member in ours[classname].items():
            other = theirs[classname][name]
            assert (member.title, member.stype, member.typename) == (
                other.title,
                other.stype,
                other.typename,
            )


# -- text -------------------------------------------------------------------


WORDS = ["", "one", "ünïcode", "x" * 254, "y" * 255, "z" * 1000]


@pytest.mark.parametrize(
    "given",
    [WORDS, tuple(WORDS), np.asarray(WORDS), np.asarray(WORDS, dtype=object), ak.Array(WORDS)],
    ids=["list", "tuple", "numpy", "objects", "awkward"],
)
def test_a_column_of_text_comes_back_as_it_went_in(given):
    with read_back(extended({"s": str}, {"s": given})) as back:
        branch = back["events"]["s"]
        assert branch.array() == WORDS
        assert branch.typename == "str"
        assert branch.record.title == "s/C"


def test_text_filled_an_entry_at_a_time_reads_back_by_range():
    data = written({"s": str, "n": int}, lambda tree: [tree.fill(s=w, n=1) for w in WORDS])
    with read_back(data) as back:
        assert back["events"]["s"].array(2, 4) == WORDS[2:4]


def test_a_text_leaf_says_how_long_its_longest_string_was():
    """ROOT's ``TLeafC`` leaves room for a NUL: ``evt-000`` makes an ``fLen`` of 8."""
    with read_back(extended({"s": str}, {"s": ["evt-000", "e"]})) as back:
        leaf = back["events"]["s"].leaf
        assert (leaf.length, leaf.etype, leaf.classname) == (8, 1, "TLeafC")
    buf, _version, end = tree_record(extended({"s": str}, {"s": ["evt-000", "e"]}))
    (fields,) = at_branches(buf).objarray({"TBranch": _branch_and_leaf})
    assert fields["leaf"] == {"is_range": 0, "minimum": 0, "maximum": 8}
    buf.resume(end)


def test_a_text_leaf_with_nothing_in_it_still_has_room_for_one():
    with read_back(extended({"s": str}, {"s": []})) as back:
        assert back["events"]["s"].leaf.length == 1
        assert back["events"]["s"].array() == []


def test_a_basket_of_text_holds_each_string_behind_its_length():
    data = extended({"s": str}, {"s": ["ab", "c" * 300]}, compression=None)
    raw, keylen = basket_bytes(data, "s")
    body = raw[keylen:]
    assert body[:3] == b"\x02ab"
    assert body[3:8] == b"\xff" + struct.pack(">i", 300)
    count, first, second, end = struct.unpack_from(">4i", body, 3 + 5 + 300)
    assert (count, first, second, end) == (3, keylen, keylen + 3, 0)


# -- declared and inferred ---------------------------------------------------


def test_a_varying_column_is_declared_with_none_or_a_counter_name():
    assert _typecode("x", ("f", None)) == ("f", "nx")
    assert _typecode("x", (float, "count")) == ("d", "count")


def test_a_tree_says_what_its_varying_columns_hold_and_are_made_of():
    with create(io.BytesIO()) as out:
        tree = out.tree("events", {"x": ("f", None), "y": ("h", "nx"), "s": str})
        assert tree.columns == {"nx": "int32", "x": "float32[nx]", "y": "int16[nx]", "s": "str"}
        assert tree.classes == ("TTree", "TBranch", "TLeafI", "TLeafF", "TLeafS", "TLeafC")
        assert repr(tree) == "<WritableTree 'events' with 4 columns and 0 entries so far>"


@pytest.mark.parametrize(
    ("values", "spec"),
    [
        (jagged(ROWS, np.float32), (np.dtype("float32"), None)),
        (ak.Array([[1, 2], [3]]), (np.dtype("int64"), None)),
        ([np.zeros(2, np.int16), np.zeros(2, np.int16)], (np.dtype("int16"), None)),
        ([[1.0], [2.0, 3.0]], (np.dtype("float64"), None)),
        (np.asarray([[1], [2, 3]], dtype=object), (np.dtype("int64"), None)),
        (["a", "b"], str),
        (np.asarray(["a", "bc"]), str),
        (np.asarray(["a", "b"], dtype=object), str),
        (ak.Array(["a", "bc"]), str),
        ([[], []], (np.dtype("float64"), None)),
        (_objects([[1, 2], [3, 4]]), (np.dtype("int64"), None)),
        ([[1, 2], [3, 4]], (np.dtype("int64"), 2)),
        (ak.Array([1.0, 2.0]), np.dtype("float64")),
        ([], np.dtype("float64")),
    ],
)
def test_what_a_column_holds_is_read_off_the_values(values, spec):
    assert spec_of("x", values) == spec


def test_a_list_that_is_neither_rows_nor_text_is_not_mistaken_for_either():
    assert not _is_rows([1, [2, 3]])
    assert not _is_rows(np.zeros(3))
    assert not _is_text(["a", 1])
    assert not _is_text(ak.Array([[1.0]]))


def test_a_table_of_rows_and_text_written_under_a_name_becomes_a_tree():
    buf = io.BytesIO()
    with create(buf) as out:
        out["events"] = {"x": jagged(ROWS), "s": list("abcde"), "e": np.arange(5)}
    with read_back(buf.getvalue()) as back:
        tree = back["events"]
        assert tree.keys() == ["nx", "x", "s", "e"]
        assert tree["x"].array().tolist() == ROWS
        assert tree["s"].array() == list("abcde")


def test_a_frame_with_a_column_of_lists_and_one_of_strings_is_a_tree_too():
    import pandas as pd

    frame = pd.DataFrame(
        {"x": [[1.0, 2.0], [], [3.0]], "y": [[1, 2], [3, 4], [5, 6]], "s": ["a", "b", "c"]}
    )
    buf = io.BytesIO()
    with create(buf) as out:
        out["events"] = frame
    with read_back(buf.getvalue()) as back:
        assert back["events"]["x"].array().tolist() == [[1.0, 2.0], [], [3.0]]
        # A frame cannot hold a rectangle, so lists of one length are still rows.
        assert back["events"]["y"].array().tolist() == [[1, 2], [3, 4], [5, 6]]
        assert back["events"]["s"].array() == ["a", "b", "c"]


# -- what a varying column refuses ------------------------------------------


def test_a_counter_cannot_also_be_declared_as_a_column():
    with create(io.BytesIO()) as out:
        with pytest.raises(ValueError, match="nx is declared as a column and named as a counter"):
            out.tree("events", {"x": ("f", None), "nx": int})


def test_a_counter_needs_a_name_a_column_could_have():
    with create(io.BytesIO()) as out:
        with pytest.raises(ValueError, match="not a name a counter can have"):
            out.tree("events", {"x": ("f", "n[x]")})


def test_a_counter_is_never_filled_by_hand():
    with create(io.BytesIO()) as out:
        tree = out.tree("events", {"x": ("f", None)})
        with pytest.raises(ValueError, match="nx, which is a counter filled from the rows"):
            tree.fill(x=[1.0], nx=1)
        with pytest.raises(ValueError, match="nothing for x and nx, which is a counter"):
            tree.fill(nx=1)
        assert len(tree) == 0


def test_columns_sharing_a_counter_must_agree_about_every_entry():
    with create(io.BytesIO()) as out:
        tree = out.tree("events", {"px": ("f", "n"), "py": ("f", "n")})
        tree.fill(px=[1.0], py=[2.0])
        with pytest.raises(ValueError, match="share the counter 'n', and entry 1 has 1 values"):
            tree.fill(px=[1.0], py=[2.0, 3.0])
        with pytest.raises(ValueError, match="entry 3 has 2 values in one and 0 in the other"):
            tree.extend({"px": [[1.0], [1.0], [1.0, 2.0]], "py": [[1.0], [1.0], []]})
        assert len(tree) == 1


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ([[1.0, 2.0]], r"shaped \(1, 2\); a row is one-dimensional"),
        ([[1.0], [2.0, 3.0]], "this one is ragged"),
        (2.0, r"shaped \(\); a row"),
        ([1.5], "holds int32 values, and these are float64"),
        ([2**40], "from -2147483648 to 2147483647"),
    ],
)
def test_a_row_that_is_not_a_flat_run_of_the_right_numbers_is_refused(value, match):
    with create(io.BytesIO()) as out:
        tree = out.tree("events", {"x": ("i", None)})
        with pytest.raises(ValueError, match=match):
            tree.fill(x=value)
        assert len(tree) == 0


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (5, "given one int rather than a sequence of runs"),
        ("abc", "given one str rather than a sequence of runs"),
        (np.zeros(3), r"shaped \(\); a row"),
        (np.zeros((2, 2, 2)), r"shaped \(2, 2\); a row"),
        (ak.Array([1.0, 2.0]), "an Awkward Array of 2 \\* float64 is not that"),
        (ak.Array([[1.0, None]]), "is not that"),
        (ak.Array([[[1.0, 2.0]], [[3.0, 4.0]]]), r"rows hold values shaped \(2,\)"),
        (Jagged(np.zeros((4, 2)), [0, 2, 4]), r"rows hold values shaped \(2,\)"),
    ],
)
def test_a_column_of_rows_that_is_not_rows_of_numbers_is_refused(values, match):
    with create(io.BytesIO()) as out:
        tree = out.tree("events", {"x": ("d", None)})
        with pytest.raises(ValueError, match=match):
            tree.extend({"x": values})
        assert len(tree) == 0


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (b"bytes", "this entry is of type bytes, not str"),
        (3, "this entry is of type int, not str"),
        ("a\x00b", "has a NUL in it"),
    ],
)
def test_an_entry_of_text_that_is_not_a_plain_string_is_refused(value, match):
    with create(io.BytesIO()) as out:
        tree = out.tree("events", {"s": str})
        with pytest.raises(ValueError, match=match):
            tree.fill(s=value)


@pytest.mark.parametrize("values", ["abc", 3])
def test_a_column_of_text_given_one_thing_rather_than_many_is_refused(values):
    with create(io.BytesIO()) as out:
        tree = out.tree("events", {"s": str})
        with pytest.raises(ValueError, match="rather than a sequence of them"):
            tree.extend({"s": values})


def test_str_inside_a_pair_says_it_stands_alone():
    with pytest.raises(ValueError, match="str, on its own rather than in a pair"):
        _typecode("x", (str, 3))
