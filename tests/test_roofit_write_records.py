"""Native streamer grammar and generated schemas prevent accidental binary layout drift."""

from __future__ import annotations

import pytest

from xrdroot.buffer import Buffer
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.roofit.io.records import put, record
from xrdroot.roofit.io.stream import Reader
from xrdroot.roofit.io.wstream import Writer


def test_schema_members_are_found_in_bases_and_unknown_members_are_rejected():
    made = record("RooGaussian")
    put(made, "_unit", "GeV")
    assert made.get("_unit") == "GeV"
    with pytest.raises(KeyError, match="misspelled"):
        put(made, "misspelled", 1)
    with pytest.raises(UnsupportedFeatureError, match="workspace class Unknown"):
        record("Unknown")


def test_native_stl_pair_descriptors_retain_their_string_and_numeric_members():
    made = record("pair<string,int>")
    assert made.m == {"first": "", "second": 0}


def test_a_standard_string_member_uses_roots_version_ten_record_and_utf8_byte_length():
    writer = Writer(0)
    writer.container("string", "héllo")
    # ROOT streams a std::string member as a TString in a counted version-ten record.
    expected = bytes.fromhex("40000009000a0668c3a96c6c6f")
    assert bytes(writer.buf.data) == expected
    reader = Reader(Buffer(expected), {})
    assert reader.container("string") == "héllo"
    assert reader.buf.pos == len(expected)


@pytest.mark.parametrize("version,extra", [(1, "fit"), (3, "_sharedProp"), (4, "_sharedProp"),
                                            (10, "_sharedProp")])
def test_genuine_versioned_real_variable_custom_streamer_fields_are_accepted(version, extra):
    from xrdroot.roofit.io.schema import validate
    from xrdroot.roofit.io.stream import Streamed

    value = Streamed("RooRealVar")
    value.version = version
    value.m[extra] = None
    validate(value)
    value.m["unknown_native_definition"] = 1
    with pytest.raises(UnsupportedFeatureError, match="unknown_native_definition"):
        validate(value)


@pytest.mark.parametrize("classname,version,fields", [
    ("RooRefArray", 1, ["TObject", "fName", "fLowerBound", "pid", "uids"]),
    ("RooLinkedList", 1, ["TObject", "items"]),
    ("RooLinkedList", 3, ["TObject", "items", "_name"]),
    ("RooWorkspace::CodeRepo", 1, ["files", "relations", "extras"]),
    ("RooWorkspace::CodeRepo", 2, ["files", "relations", "extras"]),
    ("RooAbsBinning", 1, ["TNamed"]),
    ("RooAbsBinning", 2, ["TNamed"]),
    ("TList", 3, ["items", "options"]),
    ("TList", 5, ["TObject", "fName", "items", "options"]),
    ("THashList", 5, ["TObject", "fName", "items", "options"]),
])
def test_supported_custom_decoded_shapes_do_not_bypass_unknown_member_refusals(
    classname, version, fields
):
    from xrdroot.roofit.io.schema import validate
    from xrdroot.roofit.io.stream import Streamed

    value = Streamed(classname)
    value.version = version
    value.m.update(dict.fromkeys(fields))
    validate(value)
    value.m["unknown_native_definition"] = 1
    with pytest.raises(UnsupportedFeatureError, match="unknown_native_definition"):
        validate(value)


@pytest.mark.parametrize("member", ["_binning", "_sharedProp"])
def test_nested_persisted_binning_and_shared_property_members_cannot_bypass_validation(member):
    from xrdroot.roofit.io.schema import validate

    value = record("RooRealVar")
    if member == "_binning":
        value.m[member] = record("RooUniformBinning")
    value.m[member].m["unknown_nested_definition"] = 1
    with pytest.raises(UnsupportedFeatureError, match="unknown_nested_definition"):
        validate(value)


def test_a_custom_field_from_a_different_wire_version_is_not_silently_accepted():
    from xrdroot.roofit.io.schema import validate
    from xrdroot.roofit.io.stream import Streamed

    value = Streamed("RooRealVar")
    value.version = 2
    value.m["_sharedProp"] = None
    with pytest.raises(UnsupportedFeatureError, match="_sharedProp"):
        validate(value)


@pytest.mark.parametrize("version", [1, 2])
def test_binning_printable_bases_remain_checked_instead_of_hiding_nested_members(version):
    from xrdroot.roofit.io.schema import validate
    from xrdroot.roofit.io.stream import base

    made = record("RooAbsBinning")
    made.version = version
    validate(made)
    made.m["RooPrintable"].m["new_native_definition"] = 1
    with pytest.raises(UnsupportedFeatureError, match="new_native_definition"):
        validate(made)
    made.m["RooPrintable"] = base("RooPrintable")
    validate(made)


@pytest.mark.parametrize("version", [0, 3])
def test_unknown_custom_binning_versions_are_refused_before_any_fields_are_discarded(version):
    from xrdroot.roofit.io.schema import validate

    made = record("RooAbsBinning")
    made.version = version
    with pytest.raises(UnsupportedFeatureError, match="RooAbsBinning version"):
        validate(made)


def test_version_one_code_repositories_cannot_hide_nonempty_newer_extra_headers():
    from xrdroot.roofit.io.schema import validate

    made = record("RooWorkspace::CodeRepo")
    made.version = 1
    made.m["extras"] = ["new-definition.h"]
    with pytest.raises(UnsupportedFeatureError, match="extras"):
        validate(made)


def test_linked_list_record_defaults_match_the_custom_wire_shape_and_round_trip_exactly():
    from xrdroot.roofit.io import custom  # noqa: F401 - register custom native readers
    from xrdroot.roofit.io.schema import validate

    made = record("RooLinkedList")
    assert set(made.m) == {"TObject", "items", "_name"}
    made.m["_name"] = "saved"
    validate(made)
    writer = Writer(0)
    writer.object(made, "RooLinkedList")
    expected = bytes.fromhex("0003 0001 00000000 00000000 00000000") + b"\x05saved"
    assert bytes(writer.buf.data) == expected
    reader = Reader(Buffer(expected), {})
    decoded = reader.object("RooLinkedList")
    assert decoded.version == 3
    assert set(decoded.m) == {"TObject", "items", "_name"}
    assert decoded.get("items") == [] and decoded.get("_name") == "saved"
    assert decoded.get("TObject") == {"fUniqueID": 0, "fBits": 0}
    assert reader.buf.pos == len(expected)
    validate(decoded)
    repeated = Writer(0)
    repeated.object(decoded, "RooLinkedList")
    assert bytes(repeated.buf.data) == expected


@pytest.mark.parametrize("member", ["_hashThresh", "_size", "unknown_wire_definition"])
def test_declared_bookkeeping_or_unknown_named_wire_fields_still_cannot_bypass_the_schema(member):
    from xrdroot.roofit.io.schema import validate

    made = record("RooLinkedList")
    made.m[member] = 1
    with pytest.raises(UnsupportedFeatureError, match=member):
        validate(made)
