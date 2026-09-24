"""Writing RNTuples, and reading back exactly what went in.

A written RNTuple is read back here by the reader that reads ROOT's own, so
what is asserted is that the two agree - values, types, clusters and pages -
and that the writer lays things out the way ROOT does: the column encodings
ROOT picks, a checksum after every page, the anchor described in the file's
streamer information exactly as ROOT describes it. That uproot reads these
files too was checked outside the suite, which does not depend on it.
"""

from __future__ import annotations

import io
import pathlib
import struct

import numpy as np
import pytest

import xrdroot
from support import plain
from xrdroot import Jagged, RNTuple, WritableRNTuple, create, open_root
from xrdroot.rntuple import checksum as xxh3
from xrdroot.rntuple.columns import ENCODINGS
from xrdroot.rntuple.writer import spec_of
from xrdroot.streamers import read_streamers

DATA = pathlib.Path(__file__).parent / "data" / "rntuple"


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    """Every checksum checked on the way back in, with or without ``xxhash``."""
    monkeypatch.setattr(xxh3, "_fast", xxh3.xxh3_64)


def roundtrip(fields, batches, *, compression="zlib", name="events", **options) -> RNTuple:
    """An RNTuple written to memory from ``batches``, and read back."""
    buf = io.BytesIO()
    with create(buf, compression=compression) as f:
        ntuple = f.rntuple(name, fields, **options)
        for batch in batches:
            ntuple.extend(batch)
    buf.seek(0)
    found = open_root(buf)[name]
    assert isinstance(found, RNTuple)
    return found


NUMBERS = {
    "b": np.array([True, False, True, True, False, False, True, False, True]),
    "i8": np.arange(-4, 5, dtype=np.int8),
    "u8": np.arange(9, dtype=np.uint8),
    "i16": np.arange(-4, 5, dtype=np.int16) * 1000,
    "u16": np.arange(9, dtype=np.uint16) * 7000,
    "i32": np.arange(-4, 5, dtype=np.int32) * 10**9 // 4,
    "u32": np.arange(9, dtype=np.uint32) * 5 * 10**8,
    "i64": np.arange(-4, 5, dtype=np.int64) * 10**18,
    "u64": np.arange(9, dtype=np.uint64) * 2 * 10**18,
    "f32": np.linspace(-1, 1, 9, dtype=np.float32),
    "f64": np.linspace(-1e300, 1e300, 9),
}


@pytest.mark.parametrize("compression", ["zlib", "lz4", "lzma", None])
def test_every_number_comes_back_as_it_went_in_under_every_compression(compression):
    fields = {name: values.dtype for name, values in NUMBERS.items()}
    found = roundtrip(fields, [NUMBERS], compression=compression)
    batch = found.arrays()
    for name, values in NUMBERS.items():
        assert batch[name].dtype == values.dtype
        assert batch[name].tolist() == values.tolist()


def test_the_encodings_are_the_split_ones_when_compressed_and_the_plain_ones_when_not():
    fields = {"i": np.int32, "v": [np.float32], "s": str, "b": bool}
    batch = {"i": [1, 2], "v": [[1.0], []], "s": ["a", "b"], "b": [True, False]}
    squeezed = roundtrip(fields, [batch])._store.schema.columns
    plain_ = roundtrip(fields, [batch], compression=None)._store.schema.columns
    assert [ENCODINGS[c.type].name for c in squeezed] == [
        "SplitInt32", "SplitIndex64", "SplitReal32", "SplitIndex64", "Char", "Bit",
    ]  # fmt: skip
    assert [ENCODINGS[c.type].name for c in plain_] == [
        "Int32", "Index64", "Real32", "Index64", "Char", "Bit",
    ]  # fmt: skip
    assert [(c.bits, c.field) for c in squeezed] == [
        (32, 0), (64, 1), (32, 2), (64, 3), (8, 3), (1, 4),
    ]  # fmt: skip


def test_fields_are_declared_in_any_of_the_spellings_a_physicist_reaches_for():
    fields = {
        "a": int,
        "b": float,
        "c": bool,
        "d": "std::uint16_t",
        "e": "double",
        "f": np.int8,
        "g": "float32",
        "h": "std::vector<std::int32_t>",
        "i": "ROOT::VecOps::RVec<float>",
        "j": [int],
        "k": "std::string",
        "l": str,
        "m": [np.dtype(">u2")],
    }
    batch = {name: [1] if name not in "hijmkl" else [[1]] for name in fields}
    batch["k"] = batch["l"] = ["x"]
    batch["c"] = [True]
    found = roundtrip(fields, [batch])
    assert found.cxx_types() == {
        "a": "std::int64_t",
        "b": "double",
        "c": "bool",
        "d": "std::uint16_t",
        "e": "double",
        "f": "std::int8_t",
        "g": "float",
        "h": "std::vector<std::int32_t>",
        "i": "std::vector<float>",
        "j": "std::vector<std::int64_t>",
        "k": "std::string",
        "l": "std::string",
        "m": "std::vector<std::uint16_t>",
    }
    assert plain(found.arrays()) == {**{name: [1] for name in "abcdefg"}, "c": [True], **{
        name: [[1]] for name in "hijm"}, "k": ["x"], "l": ["x"]}  # fmt: skip


def test_strings_and_vectors_come_back_as_they_went_in_from_every_shape_they_come_in():
    rows = [[1.5, 2.5], [], [3.5]]
    jagged = Jagged(np.array([9.0, 1.5, 2.5, 3.5]), [1, 3, 3, 4])  # not starting at zero
    found = roundtrip(
        {"v": [np.float64], "w": [np.float64], "x": [np.float64], "s": str},
        [
            {"v": rows, "w": jagged, "x": [np.array(r) for r in rows], "s": ["", "é", "b"]},
            {"v": [[4.0]], "w": [[4.0]], "x": [[4.0]], "s": [b"raw"]},
        ],
    )
    batch = plain(found.arrays())
    want = [*rows, [4.0]]
    assert batch["v"] == batch["w"] == batch["x"] == want
    assert batch["s"] == ["", "é", "b", "raw"]


def test_awkward_arrays_are_written_as_they_are_held():
    ak = pytest.importorskip("awkward")
    table = {
        "n": ak.Array([1, 2, 3]),
        "pt": ak.Array([[1.0, 2.0], [], [3.0]]),
        "s": ak.Array(["a", "bb", "ccc"]),
    }
    buf = io.BytesIO()
    with create(buf) as f:
        f.write("events", table, rntuple=True)
    buf.seek(0)
    found = open_root(buf)["events"]
    assert found.cxx_types() == {
        "n": "std::int64_t",
        "pt": "std::vector<double>",
        "s": "std::string",
    }
    assert plain(found.arrays()) == {
        "n": [1, 2, 3],
        "pt": [[1.0, 2.0], [], [3.0]],
        "s": ["a", "bb", "ccc"],
    }


def test_a_table_written_as_an_rntuple_takes_its_types_from_its_columns():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"x": np.arange(4, dtype=np.int16), "s": list("abcd")})
    buf = io.BytesIO()
    with create(buf) as f:
        f.write("frame", frame, rntuple=True)
        f.write("plain", {"j": Jagged([1, 2], [0, 1, 2]), "r": [[1, 2], [3]]}, rntuple=True)
    buf.seek(0)
    with open_root(buf) as f:
        assert f.classnames() == {"frame": "ROOT::RNTuple", "plain": "ROOT::RNTuple"}
        assert f["frame"].typenames() == {"x": "int16", "s": "str"}
        assert plain(f["frame"].arrays()) == {"x": [0, 1, 2, 3], "s": ["a", "b", "c", "d"]}
        assert f["plain"].cxx_types()["r"] == "std::vector<std::int64_t>"


def test_what_is_not_a_table_is_refused_when_asked_for_as_an_rntuple():
    with create(io.BytesIO()) as f:
        with pytest.raises(ValueError, match="a str is not one"):
            f.write("x", "text", rntuple=True)


def test_entries_can_be_added_a_batch_or_a_row_at_a_time():
    buf = io.BytesIO()
    with create(buf) as f:
        ntuple = f.rntuple("events", {"n": np.int32, "v": [np.int16], "s": str})
        assert isinstance(ntuple, WritableRNTuple)
        ntuple.fill(n=1, v=[1, 2], s="one")
        ntuple.extend([{"n": 2, "v": [], "s": "two"}, {"n": 3, "v": [3], "s": "three"}])
        ntuple.extend({"n": [4], "v": [[4, 4]], "s": ["four"]})
        assert len(ntuple) == ntuple.num_entries == 4
        assert ntuple.fields == {
            "n": "std::int32_t",
            "v": "std::vector<std::int16_t>",
            "s": "std::string",
        }
        assert repr(ntuple) == "<WritableRNTuple 'events' with 3 fields and 4 entries so far>"
    buf.seek(0)
    batch = plain(open_root(buf)["events"].arrays())
    assert batch == {
        "n": [1, 2, 3, 4],
        "v": [[1, 2], [], [3], [4, 4]],
        "s": ["one", "two", "three", "four"],
    }


def test_clusters_close_at_the_size_asked_for_and_pages_at_theirs():
    n = 10_000
    values = np.arange(n, dtype=np.int32)
    flags = values % 3 == 0
    found = roundtrip(
        {"i": np.int32, "b": bool}, [{"i": values, "b": flags}], cluster_size=5000, page_size=1000
    )
    assert found.num_clusters == -(-n * 5 // 5000)
    cluster = found._store.clusters(0)[0]
    assert cluster.columns[0].counts == [250, 250, 250, 250]
    assert cluster.columns[1].counts == [1000]
    assert found["i"].array().tolist() == values.tolist()
    assert found["b"].array(4990, 5010).tolist() == flags[4990:5010].tolist()


def test_a_batch_bigger_than_a_cluster_is_cut_where_the_cluster_fills():
    rows = [list(range(k % 5)) for k in range(1000)]
    found = roundtrip({"v": [np.int64]}, [{"v": rows}, {"v": rows}], cluster_size=2000)
    clusters = [c for g in range(len(found._store.groups)) for c in found._store.clusters(g)]
    assert sum(c.entries for c in clusters) == 2000
    assert all(c.entries < 200 for c in clusters)
    assert plain(found["v"].array()) == rows + rows


def test_a_row_larger_than_a_whole_cluster_still_goes_in_one_of_its_own():
    found = roundtrip({"v": [np.int64]}, [{"v": [list(range(100)), [1]]}], cluster_size=16)
    assert found.num_clusters == 2
    assert plain(found["v"].array()) == [list(range(100)), [1]]


def test_an_rntuple_with_no_entries_is_still_one():
    found = roundtrip({"x": float, "v": [float], "s": str}, [])
    assert len(found) == 0 and found.num_clusters == 0
    assert found["x"].array().tolist() == [] and found["s"].array() == []
    assert len(found["v"].array()) == 0


def test_data_that_does_not_compress_is_stored_as_it_is():
    noise = np.random.default_rng(3).integers(0, 2**63, 64, dtype=np.int64)
    found = roundtrip({"x": np.int64}, [{"x": noise}])
    pages = found._store.clusters(0)[0].columns[0]
    assert pages.sizes == [8 * 64]
    assert found["x"].array().tolist() == noise.tolist()


def test_the_file_describes_the_anchor_exactly_as_root_does():
    buf = io.BytesIO()
    with create(buf) as f:
        f.rntuple("events", {"x": float})
    buf.seek(0)
    with open_root(buf) as f:
        mine = read_streamers(f._source)["ROOT::RNTuple"]
    with open_root(DATA / "test_int_float_rntuple_v1-0-0-0.root") as f:
        roots = read_streamers(f._source)["ROOT::RNTuple"]
    assert [(m.name, m.stype, m.typename) for m in mine.values()] == [
        (m.name, m.stype, m.typename) for m in roots.values()
    ]


def test_the_anchor_header_and_footer_say_what_root_would():
    found = roundtrip({"x": float}, [{"x": [1.0]}], name="nt")
    assert found.version == (1, 0, 0, 0)
    assert found.writer == "xrdroot" and found.name == "nt"
    assert found._store.anchor.max_key == 1 << 30


def test_several_rntuples_and_a_tree_share_a_file():
    buf = io.BytesIO()
    with create(buf) as f:
        f.rntuple("a", {"x": np.int16}).extend({"x": [1, 2]})
        f.rntuple("b", {"y": str}, description="the second").fill(y="z")
        f["tree"] = {"t": np.arange(3.0)}
        f.rntuple("a", {"x": np.int16}).fill(x=9)
    buf.seek(0)
    with open_root(buf) as f:
        assert f["a"]["x"].array().tolist() == [9]
        assert f["a;1"]["x"].array().tolist() == [1, 2]
        assert f["b"].description == "the second"
        assert f["tree"]["t"].array().tolist() == [0.0, 1.0, 2.0]


def test_a_file_written_without_xxhash_carries_the_same_checksums(monkeypatch):
    monkeypatch.setattr(xxh3, "_fast", None)
    buf = io.BytesIO()
    with create(buf) as f:
        f.rntuple("events", {"x": np.float32}).extend({"x": np.arange(1000, dtype=np.float32)})
    monkeypatch.setattr(xxh3, "_fast", xxh3.xxh3_64)
    buf.seek(0)
    assert open_root(buf)["events"]["x"].array().sum() == sum(range(1000))


# What is refused, and how it is said.


@pytest.mark.parametrize(
    "spec",
    [
        complex,
        "std::vector<std::string>",
        [str],
        [[float]],
        np.float16,
        "TLorentzVector",
        3,
        object,
    ],
)
def test_a_field_this_writer_cannot_lay_out_is_refused_by_name(spec):
    with create(io.BytesIO()) as f:
        with pytest.raises(ValueError, match="Records, nested collections and variants"):
            f.rntuple("events", {"x": spec})


@pytest.mark.parametrize(
    ("name", "fields", "options", "message"),
    [
        ("", {"x": float}, {}, "a string with something in it"),
        ("a.b", {"x": float}, {}, "full stops"),
        ("ok", {"x y": float}, {}, "spaces"),
        ("ok", {"x\n": float}, {}, "control characters"),
        ("ok", {}, {}, "needs at least one"),
        ("ok", [("x", float)], {}, "mapping of field name to type"),
        ("ok", {"x": float}, {"cluster_size": 0}, "cluster size is 0"),
        ("ok", {"x": float}, {"page_size": True}, "page size is True"),
    ],
)
def test_a_badly_declared_rntuple_is_refused_with_the_reason(name, fields, options, message):
    with create(io.BytesIO()) as f:
        with pytest.raises(ValueError, match=message):
            WritableRNTuple(f, name, fields, 1, **options)


@pytest.mark.parametrize(
    ("fields", "batch", "message"),
    [
        ({"x": np.int8}, {"x": [300]}, "from -128 to 127, and these run from 300 to 300"),
        ({"x": np.int32}, {"x": [1.5]}, "these are float64, which would not go into it"),
        ({"x": np.int32}, {"x": [[1, 2]]}, r"shaped \(1, 2\)"),
        ({"x": [np.int32]}, {"x": [[[1]]]}, "not a row of numbers"),
        ({"x": [np.int32]}, {"x": [[0.5]]}, "would not go into it"),
        ({"x": str}, {"x": "abc"}, "one str for the whole batch"),
        ({"x": str}, {"x": [3]}, "an entry given for it is a int"),
        ({"x": float, "y": float}, {"x": [1.0], "y": [1.0, 2.0]}, "x: 1, y: 2"),
        ({"x": float}, {"y": [1.0]}, "has no x and y, which it does not have"),
    ],
)
def test_a_batch_that_does_not_fit_is_refused_whole(fields, batch, message):
    buf = io.BytesIO()
    with create(buf) as f:
        ntuple = f.rntuple("events", fields)
        with pytest.raises(ValueError, match=message):
            ntuple.extend(batch)
        assert len(ntuple) == 0


def test_nothing_is_added_once_the_file_is_closed():
    with create(io.BytesIO()) as f:
        ntuple = f.rntuple("events", {"x": float})
    with pytest.raises(ValueError, match="the file this RNTuple is in is closed"):
        ntuple.fill(x=1.0)


def test_the_type_of_a_column_is_read_off_its_values():
    ak = pytest.importorskip("awkward")
    assert spec_of("x", np.arange(3.0)) == np.float64
    assert spec_of("x", np.array(["a", "b"])) is str
    assert spec_of("x", ["a", b"b"]) is str
    assert spec_of("x", [[1, 2], [3]]) == [np.dtype(int)]
    assert spec_of("x", Jagged(np.zeros(2, np.float32), [0, 2])) == [np.float32]
    assert spec_of("x", ak.Array([[1.5], []])) == [np.float64]
    assert spec_of("x", ak.Array(["a"])) is str
    assert spec_of("x", ak.Array([True])) == np.bool_
    assert spec_of("x", np.zeros((2, 3), np.int8)) == [np.int8]  # rows, all of a length


def test_values_that_are_no_type_an_rntuple_is_written_from_are_refused():
    for bad in ([{"a": 1}], [["a"], ["b"]], []):
        with pytest.raises(ValueError, match="neither numbers, nor strings, nor rows"):
            spec_of("x", bad)


def test_the_page_after_every_page_is_its_checksum():
    buf = io.BytesIO()
    with create(buf, compression=None) as f:
        f.rntuple("events", {"x": np.int32}).extend({"x": [7, 8]})
    data = buf.getvalue()
    buf.seek(0)
    found = open_root(buf)["events"]
    pages = found._store.clusters(0)[0].columns[0]
    at, size = pages.offsets[0], pages.sizes[0]
    assert pages.checked == [True]
    assert data[at : at + size] == struct.pack("<ii", 7, 8)
    assert int.from_bytes(data[at + size : at + size + 8], "little") == xxh3.xxh3_64(
        data[at : at + size]
    )


def test_an_rntuple_written_to_a_path_reads_back_from_it(tmp_path):
    path = tmp_path / "out.root"
    with xrdroot.create(str(path)) as f:
        f.write("events", {"x": np.arange(5)}, rntuple=True)
    with xrdroot.open_root(str(path)) as f:
        assert f["events"]["x"].array().tolist() == [0, 1, 2, 3, 4]


def test_an_rntuple_goes_into_the_directory_its_name_says_even_in_an_update(tmp_path):
    import xrdroot

    path = str(tmp_path / "nested.root")
    with xrdroot.create(path) as out:
        out.write("runs/a/nt", {"n": np.arange(3, dtype=np.int32)}, rntuple=True)
    with xrdroot.update(path) as out:
        out.rntuple("runs/b/more", {"x": np.float64}).extend({"x": np.ones(2)})
    with xrdroot.open_root(path) as back:
        assert back["runs/a/nt"]["n"].array().tolist() == [0, 1, 2]
        assert back["runs/b/more"]["x"].array().tolist() == [1.0, 1.0]
