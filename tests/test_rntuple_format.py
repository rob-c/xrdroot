"""The RNTuple format piece by piece: frames, envelopes, encodings and field kinds.

The files ROOT wrote cover what ROOT writes. What it could write and has not
here - a ``std::map``, a ``std::optional``, a field streamed whole, a column
type from the future, a damaged frame, a block split across keys - is built
in these tests out of the same pieces the reader takes apart, so that every
branch of the reader has met the thing it is there for and says what it
should about it.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np
import pytest

from support import plain
from xrdroot import FormatError, Jagged, UnsupportedFeatureError
from xrdroot.rntuple import checksum as xxh3
from xrdroot.rntuple.columns import ENCODINGS, decode, encode, encoding, stored_size
from xrdroot.rntuple.envelope import (
    Anchor,
    Builder,
    Cursor,
    Link,
    _chunks,
    envelope,
    open_envelope,
    read_blob,
)
from xrdroot.rntuple.fields import join, rows
from xrdroot.rntuple.reader import RNTuple
from xrdroot.rntuple.schema import (
    COLLECTION,
    LEAF,
    RECORD,
    REPETITIVE,
    STREAMER,
    Cluster,
    Column,
    Field,
    Group,
    Schema,
    read_description,
    read_page_list,
    write_description,
)

BY_NAME = {kind.name: kind.code for kind in ENCODINGS.values()}


# Frames, envelopes and locators.


def cursor(data: bytes) -> Cursor:
    return Cursor(data, 0, len(data), "test")


def test_a_read_past_the_end_of_an_envelope_is_a_damaged_file():
    with pytest.raises(FormatError, match="ends at byte 2, where 4 bytes were wanted"):
        cursor(b"\x01\x02").u32()


def test_a_frame_of_the_wrong_kind_or_size_is_a_damaged_file():
    with pytest.raises(FormatError, match="record frame at byte 0 where a list frame"):
        cursor(struct.pack("<q", 8)).frame(listed=True)
    with pytest.raises(FormatError, match="declaring 4 bytes"):
        cursor(struct.pack("<q", 4)).frame(listed=False)
    with pytest.raises(FormatError, match="declaring 99 bytes"):
        cursor(struct.pack("<qI", -99, 0)).frame(listed=True)
    walk = cursor(struct.pack("<qq", 8, 0))
    walk.frame(listed=False)
    walk.u64()
    with pytest.raises(FormatError, match="before what it holds was over"):
        walk.seek(8)


def test_a_large_locator_is_read_and_one_of_another_kind_refused_by_name():
    large = struct.pack("<iQQ", -(0x01 << 24 | 20), 1 << 40, 4096) + b"\xaa"
    walk = cursor(large)
    assert walk.locator() == (1 << 40, 4096)
    assert walk.pos == 20
    other = struct.pack("<iQ", -(0x02 << 24 | 12), 0)
    with pytest.raises(UnsupportedFeatureError, match="locator of type 2"):
        cursor(other).locator()


def test_feature_flags_run_on_while_the_top_bit_is_set_and_any_set_bit_is_refused():
    assert cursor(struct.pack("<QQ", 1 << 63, 0)).flags() == [1 << 63, 0]
    with pytest.raises(UnsupportedFeatureError, match="feature flag 0 - deferred columns"):
        cursor(struct.pack("<Q", 1)).flags()
    with pytest.raises(UnsupportedFeatureError, match="feature flag 64 - a feature newer"):
        cursor(struct.pack("<QQ", 1 << 63, 2)).flags()


def test_an_envelope_says_what_it_is_and_how_long_and_ends_with_its_checksum(monkeypatch):
    monkeypatch.setattr(xxh3, "_fast", xxh3.xxh3_64)
    raw = envelope(2, b"payload!")
    assert open_envelope(raw, 2).take(8) == b"payload!"
    with pytest.raises(FormatError, match="too few to be one"):
        open_envelope(raw[:10], 2)
    with pytest.raises(FormatError, match="labelled as type 2 of 24 bytes, where a header"):
        open_envelope(raw, 1)
    damaged = raw[:9] + b"P" + raw[10:]
    with pytest.raises(FormatError, match="footer envelope does not match the checksum"):
        open_envelope(damaged, 2)


class Bytes:
    """A source over bytes in memory, counting its reads."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.reads: list[tuple[int, int]] = []

    def read(self, offset: int, size: int) -> bytes:
        self.reads.append((offset, size))
        return self.data[offset : offset + size]


def split_blob(payload: bytes, max_key: int) -> tuple[bytes, int]:
    """A payload laid out the way ROOT splits one across keys: the rest, then the head."""
    count = _chunks(len(payload), max_key)
    head = max_key - (count - 1) * 8
    out, places, at = bytearray(), [], head
    while at < len(payload):
        places.append(len(out))
        out += payload[at : at + max_key]
        at += max_key
    start = len(out)
    out += payload[:head] + struct.pack(f"<{len(places)}Q", *places)
    return bytes(out), start


@pytest.mark.parametrize("size", [100, 128, 256, 300, 257])
def test_a_block_bigger_than_a_key_is_put_back_together_from_the_keys_it_spans(size):
    payload = bytes(range(256)) * 2
    payload = payload[:size]
    data, start = split_blob(payload, 64)
    source = Bytes(data)
    assert read_blob(source, start, size, 64) == payload  # type: ignore[arg-type]
    assert read_blob(source, start, 32, 0) == data[start : start + 32]  # type: ignore[arg-type]


def test_the_number_of_keys_a_block_spans_leaves_room_for_the_offsets():
    assert _chunks(128, 64) == 3  # two whole keys would leave no room for the offset
    assert _chunks(100, 64) == 2
    assert _chunks(120, 64) == 2  # the last key has exactly the room the offset needs
    assert _chunks(121, 64) == 3


def test_an_anchor_goes_out_and_comes_back_the_same():
    anchor = Anchor((1, 0, 0, 0), Link(100, 50, 70), Link(300, 20, 20), 1 << 30)
    back = Anchor.parse(anchor.payload(), "a")
    assert (back.version, back.max_key) == ((1, 0, 0, 0), 1 << 30)
    assert (back.header.offset, back.header.size, back.header.length) == (100, 50, 70)
    assert repr(back.footer) == "<Link 20 bytes at 300, 20 unpacked>"
    with pytest.raises(FormatError, match="too short to be an RNTuple anchor"):
        Anchor.parse(anchor.payload()[:40], "a")


# Column encodings.


class Col:
    """Just enough of a column description for the decoder."""

    def __init__(self, name: str, bits: int = 0, low: float = 0.0, high: float = 0.0) -> None:
        self.id = 7
        self.type = BY_NAME[name]
        self.bits = bits
        self.low, self.high = low, high


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("Bit", np.array([True, False, True, True, False, False, False, True, True])),
        ("Int8", np.array([-3, 0, 5], np.int8)),
        ("SplitInt16", np.array([-300, 2, 7], np.int16)),
        ("SplitUInt16", np.array([60000, 2, 7], np.uint16)),
        ("SplitInt32", np.array([-(2**31), 2**31 - 1, 0], np.int32)),
        ("SplitInt64", np.array([-(2**63), 5, -1], np.int64)),
        ("SplitReal32", np.array([1.5, -2.25, 0.0], np.float32)),
        ("SplitReal64", np.array([1e300, -2.25, 0.0])),
        ("SplitIndex32", np.array([0, 3, 3, 9], np.uint32)),
        ("SplitIndex64", np.array([2, 3, 3, 90], np.uint64)),
        ("Real64", np.array([1.25, -7.5])),
        ("Index64", np.array([5, 9], np.uint64)),
    ],
)
def test_every_encoding_written_decodes_back_to_what_went_in(name, values):
    kind = encoding(BY_NAME[name])
    raw = encode(kind, values)
    assert len(raw) == stored_size(
        kind, 1 if name == "Bit" else 8 * kind.dtype.itemsize, len(values)
    )
    assert decode(Col(name, 1), raw, len(values)).tolist() == values.tolist()


def test_split_bytes_are_every_first_byte_then_every_second():
    raw = encode(encoding(BY_NAME["SplitUInt16"]), np.array([0x0102, 0x0304], np.uint16))
    assert raw == bytes([0x02, 0x04, 0x01, 0x03])


def test_half_precision_floats_read():
    raw = np.array([1.5, -0.25], "<f2").tobytes()
    assert decode(Col("Real16"), raw, 2).tolist() == [1.5, -0.25]
    split = encode(encoding(BY_NAME["SplitReal16"]), np.array([1.5, -0.25], np.float16))
    assert decode(Col("SplitReal16"), split, 2).tolist() == [1.5, -0.25]


def test_quantized_and_truncated_floats_with_impossible_widths_are_refused():
    with pytest.raises(FormatError, match="keeps 9 bits, not 10 to 31"):
        decode(Col("Real32Trunc", 9), bytes(8), 2)
    with pytest.raises(FormatError, match="keeps 33 bits, not 1 to 32"):
        decode(Col("Real32Quant", 33, -1.0, 1.0), bytes(16), 2)


def test_a_page_too_short_for_its_elements_is_a_damaged_file():
    with pytest.raises(FormatError, match="holds 3 bytes, too few for the 2 Int16"):
        decode(Col("Int16"), b"abc", 2)


def test_a_column_type_from_after_the_specification_is_refused_by_number():
    with pytest.raises(UnsupportedFeatureError, match="column type 0x42 is newer"):
        encoding(0x42)
    assert repr(encoding(0)) == "<Encoding Bit>"


# Fields, built from a schema of the test's own making.


def schema(fields: list[tuple[int, int, str, str]], columns: list[tuple[str, int]]) -> Schema:
    """``(parent, role, name, type)`` per field, ``(column type, field)`` per column."""
    made = Schema("made", "made here", "tests")
    for id, (parent, role, name, cxx) in enumerate(fields):
        made.fields.append(Field(id, parent, role, name, cxx))
    for id, (kind, field) in enumerate(columns):
        made.columns.append(Column(id, BY_NAME[kind], 8, field))
    made.link()
    return made


class Store:
    """A store whose columns are arrays already decoded, in one cluster."""

    def __init__(self, made: Schema, values: dict[int, Any], entries: int) -> None:
        self.schema = made
        self.values = values
        self.cluster = Cluster(0, entries)
        self.groups = [Group(0, entries, 1, Link(0, 0, 0))]

    def covering(self, start: int, stop: int) -> Any:
        return iter([self.cluster] if stop > start else [])

    def elements(self, cluster: Any, reps: Any, slot: int, lo: int, hi: int, base: int) -> Any:
        column = reps[0][slot]
        kind = ENCODINGS[self.schema.columns[column].type].dtype
        return np.asarray(self.values.get(column, []), kind)[lo:hi]


def made(fields: Any, columns: Any, values: dict[int, Any], entries: int) -> RNTuple:
    built = schema(fields, columns)
    return RNTuple("made", Store(built, values, entries))  # type: ignore[arg-type]


def test_a_map_is_a_dict_per_entry():
    found = made(
        [
            (0, COLLECTION, "m", "std::map<std::string,float>"),
            (0, RECORD, "_0", "std::pair<std::string,float>"),
            (1, LEAF, "_0", "std::string"),
            (1, LEAF, "_1", "float"),
        ],
        [("Index64", 0), ("Index64", 2), ("Char", 2), ("Real32", 3)],
        {0: [2, 2, 3], 1: [1, 3, 4], 2: list(b"abcd"), 3: [1.5, 2.5, 3.5]},
        3,
    )
    assert found.typenames() == {"m": "dict[str, float32]"}
    assert found["m"].array() == [{"a": 1.5, "bc": 2.5}, {}, {"d": 3.5}]


def test_an_optional_is_the_value_or_none():
    found = made(
        [(0, COLLECTION, "o", "std::optional<std::int32_t>"), (0, LEAF, "_0", "std::int32_t")],
        [("Index64", 0), ("Int32", 1)],
        {0: [1, 1, 2], 1: [5, 7]},
        3,
    )
    field = found["o"]
    assert (field.typename, field.is_jagged) == ("int32 | None", False)
    assert field.array() == [5, None, 7]


def test_a_wrapper_reads_as_what_it_wraps_and_an_unknown_type_as_its_column():
    found = made(
        [
            (0, LEAF, "w", "std::atomic<std::vector<float>>"),
            (0, COLLECTION, "_0", "std::vector<float>"),
            (1, LEAF, "_0", "float"),
            (3, LEAF, "e", "MyEnum"),
        ],
        [("Index64", 1), ("Real32", 2), ("Int16", 3)],
        {0: [1, 3], 1: [1.0, 2.0, 3.0], 2: [4, -2]},
        2,
    )
    assert found["w"].is_jagged and found["w"].array().tolist() == [[1.0], [2.0, 3.0]]
    assert found.typenames()["e"] == "int16"
    assert found["e"].array().tolist() == [4, -2]


def test_fields_this_reader_does_not_decode_are_refused_by_name_and_listed():
    found = made(
        [
            (0, STREAMER, "streamed", "TLorentzVector"),
            (1, LEAF, "future", "float"),
            (2, COLLECTION, "orphan", "std::vector<float>"),
            (3, LEAF, "nothing", "Empty"),
            (4, LEAF, "offsets", "Weird"),
            (5, LEAF, "switch", "Odd"),
        ],
        [("Index64", 0), ("Byte", 0), ("Real32", 1), ("Index64", 2), ("Index64", 4), ("Switch", 5)],
        {},
        1,
    )
    found._store.schema.columns[2].type = 0x42  # a column type from the future
    found = RNTuple("made", found._store)
    assert found.readable() == []
    reasons = found.unreadable
    assert "the ROOT streamer wrote whole" in reasons["streamed"]
    assert "newer than the RNTuple specification" in reasons["future"]
    assert "with 0 fields under it, where exactly one belongs" in reasons["orphan"]
    assert "no columns and nothing under it" in reasons["nothing"]
    assert "stored as Index64, which is not a number" in reasons["offsets"]
    assert "stored as Switch" in reasons["switch"]
    assert found.typenames()["streamed"] == "? (TLorentzVector)"
    with pytest.raises(UnsupportedFeatureError, match="'orphan' holds a std::vector<float>"):
        found["orphan"].array()


def test_an_array_of_arrays_is_a_block_of_three_dimensions():
    found = made(
        [
            (0, LEAF, "a", "std::array<std::array<float,2>,3>"),
            (0, LEAF, "_0", "std::array<float,2>"),
            (1, LEAF, "_0", "float"),
        ],
        [("Real32", 2)],
        {0: list(range(12))},
        2,
    )
    for field, size in zip(found._store.schema.fields[:2], (3, 2)):
        field.flags, field.array_size = REPETITIVE, size
    found = RNTuple("made", found._store)
    assert found["a"].array().shape == (2, 3, 2)
    assert found.typenames()["a"] == "float32[2][3]"


def test_values_are_made_rows_and_pieces_joined_whatever_their_shape():
    assert rows(np.arange(4).reshape(2, 2))[1].tolist() == [2, 3]
    assert rows(np.arange(2)) == [0, 1]
    joined = join([Jagged([1, 2], [0, 2]), Jagged([3], [0, 0, 1])], None)
    assert plain(joined) == [[1, 2], [], [3]]
    assert join([[1], [2, 3]], None) == [1, 2, 3]
    assert join([], "empty") == "empty"


def test_a_node_says_what_it_is():
    found = made([(0, LEAF, "x", "float")], [("Real32", 0)], {0: [1.0]}, 1)
    assert repr(found["x"].node) == "<Leaf 'x' of float32>"
    assert repr(found._store.schema.fields[0]) == "<Field 0 'x' of 'float'>"
    assert repr(found._store.schema.columns[0]) == "<Column 0 of type 0xc for field 0>"


# Descriptions, written and read back.


def test_a_schema_description_goes_out_and_comes_back_the_same():
    written = schema(
        [(0, LEAF, "a", "std::array<float,4>"), (0, LEAF, "_0", "float")], [("Real32", 1)]
    )
    written.fields[0].flags, written.fields[0].array_size = REPETITIVE, 4
    out = Builder()
    write_description(out, written)
    back = Schema("back", "", "")
    read_description(Cursor(bytes(out.data), 0, len(out.data), "test"), back)
    back.link()
    assert [(f.name, f.type, f.array_size) for f in back.fields] == [
        ("a", "std::array<float,4>", 4),
        ("_0", "float", 0),
    ]
    assert back.fields[1].columns == [[0]]


def test_a_sharded_cluster_is_refused_as_a_later_version_of_the_format():
    out = Builder()
    out.pack("Q", 0)
    mark = out.list(1)
    record = out.record()
    out.pack("QQ", 0, 10 | (1 << 56))
    out.close(record)
    out.close(mark)
    with pytest.raises(UnsupportedFeatureError, match="sharded cluster at entry 0"):
        read_page_list(Cursor(bytes(out.data), 0, len(out.data), "page list"))
