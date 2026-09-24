"""Reading RNTuples ROOT wrote, and getting back what the macros put in them.

The files under ``tests/data/rntuple`` were written by ROOT from the macros in
scikit-hep-testdata, and the values asserted here are the ones those macros
filled: a reader checked against its own writer proves nothing. Between them
they have every column encoding the format defines, collections of every
depth, records, bases, variants, several clusters and cluster groups, a schema
that grew while it was written, a field written through two encodings, three
versions of the format, and two files of real CMS open data.
"""

from __future__ import annotations

import io
import math
import pathlib

import numpy as np
import pytest

import xrdroot
from support import plain
from xrdroot import FormatError, Jagged, RNTuple, UnsupportedFeatureError
from xrdroot.rntuple import checksum as xxh3

DATA = pathlib.Path(__file__).parent / "data" / "rntuple"
FILES = sorted(DATA.glob("*.root"))


def ntuple(name: str, key: str = "ntuple") -> RNTuple:
    """An RNTuple from one of the files, read from memory so that nothing is left open."""
    found = xrdroot.open_root(io.BytesIO((DATA / f"{name}.root").read_bytes()))[key]
    assert isinstance(found, RNTuple)
    return found


@pytest.fixture
def verified(monkeypatch):
    """Checksums checked on the way in, as they are with ``xxhash`` installed."""
    monkeypatch.setattr(xxh3, "_fast", xxh3.xxh3_64)


@pytest.fixture
def unverified(monkeypatch):
    """Checksums passed over on the way in, as they are without ``xxhash``."""
    monkeypatch.setattr(xxh3, "_fast", None)


@pytest.mark.parametrize("path", FILES, ids=[path.stem for path in FILES])
def test_every_rntuple_root_wrote_opens_and_every_field_reads(path, verified):
    with xrdroot.open_root(path) as f:
        names = [name for name, kind in f.classnames().items() if kind == "ROOT::RNTuple"]
        assert names
        for name in names:
            found = f[name]
            assert not found.unreadable
            batch = found.arrays(entry_stop=20)
            assert set(batch) == set(found.keys())
            assert all(len(values) == min(20, len(found)) for values in batch.values())


def test_numbers_come_back_as_arrays_of_the_type_the_field_says():
    found = ntuple("test_int_float_rntuple_v1-0-0-0")
    assert len(found) == 10
    assert found.typenames() == {"one_integers": "int32", "two_floats": "float32"}
    assert not found["one_integers"].is_jagged
    assert found["one_integers"].array().tolist() == list(range(9, -1, -1))
    floats = found["two_floats"].array()
    assert floats.dtype == np.float32
    assert np.allclose(floats, np.arange(9, -1, -1) * 1.1)


def test_a_vector_of_numbers_is_jagged():
    found = ntuple("test_1jag_int_float_rntuple_v1-0-0-0")
    ints = found["one_v_integers"].array()
    floats = found["two_v_floats"].array()
    assert isinstance(ints, Jagged) and found["one_v_integers"].is_jagged
    for entry in range(100):
        base, count = 100 - 10 * (entry // 10), entry % 10
        assert ints[entry].tolist() == [base - k for k in range(count)]
        assert np.allclose(floats[entry], [(base - k) / 10 for k in range(count)])


def test_strings_are_a_list_of_str():
    found = ntuple("ntpl001_staff_rntuple_v1-0-0-0", "Staff")
    assert len(found) == 3354
    first = found.arrays(["Category", "Age", "Cost", "Division", "Nation"], 0, 3)
    assert first["Category"].tolist() == [202, 530, 316]
    assert first["Age"].tolist() == [58, 63, 56]
    assert first["Cost"].tolist() == [11975, 10228, 10730]
    assert first["Division"] == ["PS", "EP", "PS"]
    assert first["Nation"] == ["DE", "CH", "FR"]
    assert all(len(nation) == 2 for nation in found["Nation"].array())


def test_a_footer_of_the_newer_format_version_reads_the_same():
    older = ntuple("ntpl001_staff_rntuple_v1-0-0-0", "Staff")
    newer = ntuple("ntpl001_staff_rntuple_v1-0-1-0", "Staff")
    assert newer.version == (1, 0, 1, 0) and older.version == (1, 0, 0, 0)
    assert newer["Nation"].array() == older["Nation"].array()


def test_an_uncompressed_rntuple_in_the_plain_encodings_reads():
    found = ntuple("rntviewer-testfile-uncomp-single-rntuple-v1-0-0-0", "Contributors")
    names = found.arrays(["firstName", "lastName"], 0, 3)
    assert [f"{a} {b}" for a, b in zip(names["firstName"], names["lastName"])] == [
        "Jakob Blomer",
        "Philippe Canal",
        "Axel Naumann",
    ]


def test_one_file_can_hold_several_rntuples():
    assert ntuple("rntviewer-testfile-multiple-rntuples-v1-0-0-0", "A")["f"].array(
        0, 3
    ).tolist() == [0.0, 1.0, 2.0]
    assert ntuple("rntviewer-testfile-multiple-rntuples-v1-0-0-0", "B")["g"].array(
        0, 3
    ).tolist() == [0, 100, 200]


def test_a_record_is_a_dict_per_entry_and_its_members_are_reachable_with_dots():
    found = ntuple("test_nested_structs_rntuple_v1-0-0-0")
    rows = plain(found["my_struct"].array())
    assert rows[3] == {
        "i": 3,
        "sub_struct": {"i": 4, "sub_sub_struct": {"i": 5, "v": [3, 4]}},
    }
    inner = found["my_struct.sub_struct.sub_sub_struct.v"]
    assert inner.array(2, 4).tolist() == [[2, 3], [3, 4]]
    assert found["my_struct.sub_struct.i"].array().tolist() == list(range(1, 11))
    assert "my_struct.sub_struct" in found
    assert "my_struct.nothing" not in found
    assert "my_struct.i.deeper" not in found
    assert "nothing.i" not in found
    assert 3 not in found


def test_a_base_class_is_a_member_named_after_the_class():
    found = ntuple("test_class_inheritance_rntuple_v1-0-0-1", "rntpl")
    child = plain(found["child"].array(2, 3))[0]
    assert child == {
        "BaseA": {"base_a1": 2, "base_a2": pytest.approx(0.2), "base_a3": [0, 2, 4]},
        "child_1": 4,
        "child_2": 40.0,
    }
    grand = plain(found["grandchild"].array(1, 2))[0]
    assert grand["grandchild_1"] == 3 and grand["Child"]["BaseA"]["base_a1"] == 1


def test_every_standard_container_comes_back_in_its_python_shape():
    found = ntuple("test_stl_containers_rntuple_v1-0-0-0")
    batch = plain(found.arrays())
    assert batch["string"] == ["one", "two", "three", "four", "five"]
    assert batch["vector_int32"][2] == [1, 2, 3]
    assert batch["array_float"][1] == [2.0, 2.0, 2.0]
    assert batch["vector_vector_int32"][2] == [[1], [2], [3]]
    assert batch["vector_string"][1] == ["one", "two"]
    assert batch["vector_vector_string"][1] == [["one"], ["two"]]


def test_variants_tuples_and_records_in_containers_come_back_as_python_values():
    batch = plain(ntuple("test_stl_containers_rntuple_v1-0-0-0").arrays())
    assert batch["variant_int32_string"] == [1, "two", "three", 4, 5]
    assert batch["vector_variant_int64_string"][2] == ["one", 2, 3]
    assert batch["tuple_int32_string"][3] == (4, "four")
    assert batch["pair_int32_string"][0] == (1, "one")
    assert batch["vector_tuple_int32_string"][1] == [(1, "one"), (2, "two")]
    assert batch["lorentz_vector"][4] == {"pt": 5.0, "eta": 5.0, "phi": 5.0, "mass": 5.0}
    assert len(batch["array_lv"][0]) == 3


def test_the_python_type_of_every_standard_container_is_spelled_out():
    types = ntuple("test_stl_containers_rntuple_v1-0-0-0").typenames()
    assert types["array_float"] == "float32[3]"
    assert types["variant_int32_string"] == "int32 | str"
    assert types["tuple_int32_string"] == "tuple[int32, str]"
    assert types["vector_vector_int32"] == "list[list[int32]]"
    assert types["lorentz_vector"] == "{pt: float32, eta: float32, phi: float32, mass: float32}"


def test_a_variant_that_holds_nothing_is_none_and_an_empty_record_an_empty_dict():
    found = ntuple("test_emptystruct_invalidvar_rntuple_v1-0-0-0")
    assert found["variant"].array() == [1, None, {"i": 2}]
    assert found["empty_struct"].array() == [{}, {}, {}]


def test_a_bool_is_one_bit_and_a_bitset_one_row_of_bits():
    assert ntuple("test_bit_rntuple_v1-0-0-0")["one_bit"].array().tolist() == [
        True, False, False, True, False, False, True, False, False, True,
    ]  # fmt: skip
    found = ntuple("test_atomic_bitset_rntuple_v1-0-0-0")
    assert found["atomic_int"].array().tolist() == [1, 2, 3]
    bits = found["bitset"].array()
    assert bits.shape == (3, 42) and found["bitset"].typename == "bool[42]"
    values = [sum(1 << i for i, bit in enumerate(row) if bit) for row in bits]
    first = 42 | 0b1010101010101010
    assert values == [42, first, first & 0b1100110011001100]


def test_truncated_floats_keep_their_top_bits_and_quantized_ones_land_close():
    found = ntuple("test_float_types_rntuple_v1-0-0-0")
    value = np.float32(1.23456789)
    bits = int(np.array([value]).view(np.uint32)[0])
    for kept in (10, 16, 24, 31):
        got = found[f"trunc{kept}"].array(0, 1)
        want = np.array([bits & ~((1 << (32 - kept)) - 1)], np.uint32).view(np.float32)
        assert got.tolist() == want.tolist()
    q8, q32 = found["quant8"].array(0, 1)[0], found["quant32"].array(0, 1)[0]
    assert abs(q8 - value) < 0.02 and abs(q32 - value) < 1e-6
    assert found["quant1"].array().tolist() == [3.0, 3.0, -2.0, -2.0]


def test_a_column_added_while_writing_reads_as_zeros_before_it_existed():
    found = ntuple("test_extension_columns_rntuple_v1-0-0-0")
    assert found.num_clusters >= 2
    batch = found.arrays()
    for entry in range(600):
        assert batch["int_field"][entry] == entry % 200
        assert batch["float_field"][entry] == (entry % 200 + 0.5 if entry >= 200 else 0.0)
        want = [entry % 200, entry % 200 + 1] if entry >= 400 else []
        assert batch["intvec_field"][entry].tolist() == want


def test_a_field_written_two_ways_reads_whichever_is_live_in_each_cluster():
    found = ntuple("test_multiple_representations_rntuple_v1-0-0-0")
    assert found.num_clusters == 3
    assert found["real"].array().tolist() == [1.0, 2.0, 3.0]


def test_ranges_are_read_across_clusters_and_cluster_groups():
    found = ntuple("test_multiple_cluster_groups_rntuple_v1-0-0-0")
    assert len(found) == 1000 and found.num_clusters >= 3
    batch = found.arrays(entry_start=440, entry_stop=760)
    assert batch["one"].tolist() == list(range(440, 760))
    assert batch["int_vector"][10].tolist() == [450, 451]
    assert found["int_vector"].array(-2).tolist() == [[998, 999], [999, 1000]]


def test_offsets_are_counted_from_the_start_of_each_cluster():
    found = ntuple("test_index_multicluster_rntuple_v1-0-0-0")
    rows = found["int_vector"].array().tolist()
    assert rows == [[e % 100, e % 100 + e // 100] for e in range(200)]


def test_a_hundred_million_entries_are_reached_without_reading_them_all():
    found = ntuple("test_int_multicluster_rntuple_v1-0-0-0")
    assert len(found) == 100_000_000
    field = found["one_integers"]
    assert field.array(49_999_999, 50_000_001).tolist() == [2, 1]
    assert field.array(-1).tolist() == [1]
    assert field.array(0, 1).tolist() == [2]
    assert field.array(10_000_000, 45_000_000).sum() == 2 * 35_000_000  # evicts the cache


def test_a_cardinality_field_counts_the_items_of_its_collection():
    found = ntuple("Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0", "Events")
    batch = found.arrays(["nMuon", "Muon_pt", "Muon_charge"])
    assert batch["nMuon"].dtype == np.uint32
    assert batch["nMuon"].tolist() == batch["Muon_pt"].lengths().tolist()
    assert batch["Muon_charge"].lengths().tolist() == batch["Muon_pt"].lengths().tolist()
    collection = found["_collection0"].array(0, 1)[0]
    assert collection[0]["Muon_pt"] == pytest.approx(float(batch["Muon_pt"][0][0]))


def test_a_nanoaod_of_nearly_a_thousand_fields_reads_the_way_cms_wrote_it():
    found = ntuple("cmsopendata2015_ttbar_19980_NANOAOD_RNTupleImporter_rntuple_v1-0-0-1", "Events")
    assert len(found.keys()) > 900
    assert found.writer.startswith("ROOT v6.37")
    types = found.typenames()
    assert (types["run"], types["event"], types["nMuon"]) == ("uint32", "uint64", "uint32")
    assert types["Muon_pt"] == "list[float32]"
    assert found.cxx_types()["Muon_pt"] == "ROOT::VecOps::RVec<float>"
    batch = found.arrays(["nMuon", "Muon_pt", "Muon_eta"])
    assert batch["nMuon"].tolist() == batch["Muon_pt"].lengths().tolist()
    assert batch["nMuon"].tolist() == batch["Muon_eta"].lengths().tolist()


def test_a_field_has_a_name_a_type_in_both_languages_and_a_length():
    found = ntuple("test_1jag_int_float_rntuple_v1-0-0-0")
    field = found["two_v_floats"]
    assert repr(field) == "<RField 'two_v_floats' of std::vector<float>>"
    assert (field.typename, field.cxx_type, len(field)) == (
        "list[float32]",
        "std::vector<float>",
        100,
    )
    assert repr(found) == "<RNTuple 'ntuple' with 2 fields and 100 entries>"
    assert list(found) == found.keys() == found.readable() == ["one_v_integers", "two_v_floats"]
    assert found.description == "" and found.writer.startswith("ROOT")
    assert found.show().splitlines()[1].split() == [
        "two_v_floats",
        "list[float32]",
        "std::vector<float>",
    ]
    record = ntuple("test_nested_structs_rntuple_v1-0-0-0")["my_struct.sub_struct"]
    assert "sub_sub_struct" in record.typename
    untyped = ntuple("Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0", "Events")
    assert repr(untyped["_collection0"]) == "<RField '_collection0' of an untyped record>"


def test_asking_for_a_field_that_is_not_there_names_the_ones_that_are():
    found = ntuple("test_int_float_rntuple_v1-0-0-0")
    with pytest.raises(KeyError, match="one_integers, two_floats"):
        found["nope"]


def test_arrays_and_iterate_hand_the_fields_to_any_library():
    pytest.importorskip("awkward")
    found = ntuple("test_1jag_int_float_rntuple_v1-0-0-0")
    table = found.arrays(library="ak")
    assert table.fields == ["one_v_integers", "two_v_floats"]
    assert table["one_v_integers"][11].tolist() == [90]
    batches = list(found.iterate(["one_v_integers"], step=30, entry_start=5, entry_stop=-5))
    assert [len(batch["one_v_integers"]) for batch in batches] == [30, 30, 30]
    with pytest.raises(ValueError, match="step"):
        next(found.iterate(step=0))


def test_an_empty_range_is_an_empty_value_of_the_right_shape():
    found = ntuple("test_stl_containers_rntuple_v1-0-0-0")
    empty = found.arrays(entry_start=3, entry_stop=3)
    assert empty["string"] == [] and empty["vector_variant_int64_string"] == []
    assert empty["array_float"].shape == (0, 3)
    assert len(empty["vector_int32"]) == 0 and isinstance(empty["vector_int32"], Jagged)
    assert ntuple("test_atomic_bitset_rntuple_v1-0-0-0")["bitset"].array(1, 1).shape == (0, 42)
    counts = ntuple("Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0", "Events")
    assert counts["nMuon"].array(5, 5).tolist() == []


# Damage, and what the reader says about it.


def damaged(name: str, at: int, flip: int = 0xFF) -> io.BytesIO:
    data = bytearray((DATA / f"{name}.root").read_bytes())
    data[at] ^= flip
    return io.BytesIO(bytes(data))


def locations(name: str, key: str = "ntuple") -> tuple[int, int, int]:
    """Where the anchor's payload, the header and the first page start."""
    with xrdroot.open_root(DATA / f"{name}.root") as f:
        found = f[key]
        store = found._store
        page = store.clusters(0)[0].columns[0].offsets[0]
        anchor = f._key(key)
        return anchor.seek_key + anchor.keylen, store.anchor.header.offset, page


def test_a_damaged_page_is_refused_when_checksums_are_checked(verified):
    _anchor, _header, page = locations("test_int_float_rntuple_v1-0-0-0")
    found = xrdroot.open_root(damaged("test_int_float_rntuple_v1-0-0-0", page + 3))["ntuple"]
    with pytest.raises(FormatError, match="page of RNTuple column 0 does not match"):
        found["one_integers"].array()


def test_a_damaged_header_or_anchor_is_refused_when_checksums_are_checked(verified):
    name = "rntviewer-testfile-uncomp-single-rntuple-v1-0-0-0"
    _anchor, header, _page = locations(name, "Contributors")
    with pytest.raises(FormatError, match="header envelope does not match"):
        xrdroot.open_root(damaged(name, header + 12))["Contributors"]
    anchor, _header, _page = locations("test_int_float_rntuple_v1-0-0-0")
    with pytest.raises(FormatError, match="anchor that does not match"):
        xrdroot.open_root(damaged("test_int_float_rntuple_v1-0-0-0", anchor + 7))["ntuple"]


def test_without_checksums_a_damaged_page_reads_as_the_damage(unverified):
    _anchor, _header, page = locations(
        "rntviewer-testfile-uncomp-single-rntuple-v1-0-0-0", "Contributors"
    )
    source = damaged("rntviewer-testfile-uncomp-single-rntuple-v1-0-0-0", page, 0x01)
    found = xrdroot.open_root(source)["Contributors"]
    assert found["lastName"].array(0, 1) == ["Blomer"]


def test_an_anchor_of_another_epoch_or_class_version_is_refused_by_name(unverified):
    anchor, _header, _page = locations("test_int_float_rntuple_v1-0-0-0")
    with pytest.raises(UnsupportedFeatureError, match="format epoch 3"):
        xrdroot.open_root(damaged("test_int_float_rntuple_v1-0-0-0", anchor + 7, 0x02))["ntuple"]
    with pytest.raises(UnsupportedFeatureError, match="class version 3"):
        xrdroot.open_root(damaged("test_int_float_rntuple_v1-0-0-0", anchor + 5, 0x01))["ntuple"]


def test_a_column_shorter_than_its_cluster_is_a_damaged_file():
    found = ntuple("test_int_float_rntuple_v1-0-0-0")
    pages = found._store.clusters(0)[0].columns[1]
    pages.starts = pages.starts - np.array([0, 3])
    with pytest.raises(FormatError, match="holds 7 elements"):
        found["two_floats"].array()


def test_nan_survives_the_trip():
    found = ntuple("cmsopendata2015_ttbar_19980_NANOAOD_RNTupleImporter_rntuple_v1-0-0-1", "Events")
    assert math.isnan(found["HTXS_Higgs_y"].array(0, 1)[0])
