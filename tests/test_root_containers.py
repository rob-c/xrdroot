"""The C++ side of a ROOT file: split members, containers and packed floats.

Everything asserted against a file here is a value go-hep reads out of the
same bytes, which is the only way to be sure a reader is right rather than
merely self-consistent. The crafted records are for the shapes the corpus has
no example of - a streamer element from 2001, a container written field by
field - where a refusal has to be provoked to be checked.
"""

from __future__ import annotations

import io
import pathlib
import struct
from typing import Any

import numpy as np
import pytest

from support import plain
from xrdroot import FormatError, UnsupportedFeatureError, create, open_root
from xrdroot.buffer import BYTE_COUNT_MASK, Buffer
from xrdroot.cxx import Mapping, Pair, Prim, Seq, Str, parse, py_name
from xrdroot.file import Source
from xrdroot.interp import (
    KINDS,
    OFFSET_L,
    Flat,
    Refused,
    Rows,
    Values,
    _fields,
    _form,
    _packed,
    _Plain,
    _plan,
    _range,
    build,
)
from xrdroot.merging.infos import read_entries
from xrdroot.objects import (
    BranchRecord,
    LeafRecord,
    read_branch_element,
    read_derived_branch,
)
from xrdroot.streamers import (
    Member,
    read_element,
    read_info,
    read_layouts,
    read_streamers,
    reconcile,
)
from xrdroot.tree import TTree

DATA = pathlib.Path(__file__).parent / "data"


def opened(name: str):
    return open_root(str(DATA / f"{name}.root"))


@pytest.fixture
def containers():
    with opened("std-containers-split00") as handle:
        yield handle["tree"]


@pytest.fixture
def event():
    with opened("small-evnt-tree-fullsplit") as handle:
        yield handle["tree"]


# -- crafted bytes --------------------------------------------------------


def tstring(text: str) -> bytes:
    raw = text.encode()
    return bytes([len(raw)]) + raw


def record(version: int, body: bytes = b"") -> bytes:
    return struct.pack(">IH", BYTE_COUNT_MASK | (2 + len(body)), version) + body


def named_bytes(name: str, title: str = "") -> bytes:
    return record(1, struct.pack(">HII", 1, 0, 0) + tstring(name) + tstring(title))


def element_bytes(version: int, name: str, typename: str, stype: int = 3) -> bytes:
    """A ``TStreamerElement`` of the given version, under a subclass record."""
    body = named_bytes(name, "") + struct.pack(">iiii", stype, 4, 1, 1)
    body += struct.pack(">ii", 1, 0) if version == 1 else struct.pack(">5i", 0, 0, 0, 0, 0)
    return record(4, record(version, body + tstring(typename)))


def branch_bytes() -> bytes:
    """A ``TBranch`` with no baskets in it, which is the part every branch has."""
    body = named_bytes("b", "b/I") + record(1)
    body += struct.pack(">iiii", 1, 4096, 0, 0)
    body += struct.pack(">q", 0)
    body += struct.pack(">iii", 0, 0, 0)
    body += struct.pack(">qqq", 0, 0, 0)
    empty = record(3, struct.pack(">HII", 1, 0, 0) + tstring("") + struct.pack(">ii", 0, 0))
    return record(10, body + empty * 3 + b"\x01\x01\x01" + tstring(""))


def branch_element_bytes(version: int, classname: str, ftype: int = 0) -> bytes:
    """A ``TBranchElement`` of a version old enough to say things differently."""
    body = branch_bytes() + tstring(classname)
    if version > 1:
        body += tstring("") + tstring("") + struct.pack(">I", 0)
    body += struct.pack(">I" if version < 10 else ">H", 0)
    body += struct.pack(">ii", -1, ftype)  # which member this is, and what it holds
    return record(version, body)


class Nothing:
    """A source that describes no classes at all, which some files do not."""

    def streamers(self) -> dict[str, dict[str, Member]]:
        return {}


def column_of(typename: str, *, member: str = "", ltype: int = -1, source=None):
    """What :func:`build` makes of a branch declared to hold ``typename``."""
    record_ = BranchRecord()
    record_.name = record_.title = "b"
    record_.classname = typename
    leaf = LeafRecord("TLeafElement")
    leaf.name = member or "b"
    leaf.ltype = ltype
    record_.leaves = [leaf]
    return build(record_, leaf, source or Nothing())


# -- C++ type names -------------------------------------------------------


def test_every_spelling_of_a_fundamental_type_is_one_type():
    assert parse("unsigned long long int").typename == "uint64"
    assert parse("Float_t").typename == "float32"
    assert parse("std::vector<Double_t>*").item.typename == "float64"
    assert parse("Version_t").itemsize == 2


def test_both_kinds_of_string_are_strings_written_differently():
    assert parse("string").record is True
    assert parse("basic_string<char>").record is True
    assert parse("TString").record is False


def test_a_container_is_the_type_it_holds():
    assert py_name(parse("vector<vector<int> >".replace(" ", ""))) == "list[list[int32]]"
    assert py_name(parse("set<TString>")) == "list[str]"
    assert py_name(parse("unordered_map<string,deque<float> >")) == "dict[str, list[float32]]"
    assert py_name(parse("short")) == "int16"


def test_a_name_this_reader_has_no_idea_about_is_no_type_at_all():
    assert parse("TLorentzVector") is None
    assert parse("vector<TLorentzVector>") is None
    assert py_name(parse("multimap<int,int>")) == "list[tuple[int32, int32]]"  # duplicates kept
    assert parse("map<int>") is None
    assert parse("map<int,TLorentzVector>") is None
    assert parse("map<TLorentzVector,int>") is None
    assert parse("pair<int>") is None
    assert parse("pair<int,TLorentzVector>") is None


def test_a_pair_is_the_tuple_it_obviously_is():
    node = parse("pair<int,double>")
    assert isinstance(node, Pair)
    assert repr(node) == "<Pair of <Prim int32> and <Prim float64>>"
    assert py_name(parse("vector<pair<int,double> >")) == "list[tuple[int32, float64]]"


def test_a_pair_anywhere_but_under_its_own_container_is_refused():
    assert "a pair standing on its own" in column_of("pair<int,int>").reason
    assert "below the top of its container" in column_of("vector<vector<pair<int,int> > >").reason
    assert "a map inside another container" in column_of("vector<map<int,int> >").reason
    assert "a map inside another container" in column_of(
        "vector<pair<int,map<int,int> > >"
    ).reason
    assert "a pair keyed by a container" in column_of(
        "vector<pair<vector<int>,int> >"
    ).reason


def test_a_pair_may_hold_a_container_of_its_own_which_reads_a_column_of_them():
    """A map whose values are containers - ``unordered_multimap<string,vector<float>>`` - is
    written member-wise: every key, then every value, each value a count and its items inside
    the one record the column's items are wrapped in."""
    from xrdroot import interp

    column = column_of("vector<pair<int,vector<int> > >")
    keys = struct.pack(">ii", 1, 2)
    values = record(0, struct.pack(">i2i", 2, 15, 25) + struct.pack(">i", 0))
    body = struct.pack(">hi", 1, 2) + keys + values
    read = column.value(Buffer(record(interp.MEMBER_WISE | 1, body)), 0)
    assert [(key, list(value)) for key, value in read] == [(1, [15, 25]), (2, [])]


def test_the_members_of_a_class_that_cannot_be_walked_are_no_fields_at_all():
    """``None`` from :func:`_fields` is what refuses field-by-field data."""
    assert _fields("TArrayD", Nothing(), ()) is None  # streams itself instead
    described = Layout(Widget=declared("w", 999, "Mystery"))
    assert _fields("Widget", described, ()) is None
    assert _fields("Widget", Nothing(), ()) is None


def test_the_pieces_of_a_type_say_what_they_are():
    assert repr(parse("float")) == "<Prim float32>"
    assert repr(parse("string")) == "<Str>"
    assert repr(parse("vector<int>")) == "<Seq of <Prim int32>>"
    assert repr(parse("map<int,int>")) == "<Mapping <Prim int32> to <Prim int32>>"
    assert isinstance(parse("list<int>"), Seq)
    assert isinstance(parse("map<int,int>"), Mapping)
    assert isinstance(parse("TString"), Str)
    assert isinstance(parse("bool"), Prim)
    assert repr(parse("ROOT::VecOps::RVec<float>")) == "<Seq of <Prim float32>>"


# -- the classes a file describes -----------------------------------------


def test_a_file_says_what_its_own_classes_are_made_of():
    with opened("small-evnt-tree-fullsplit") as handle:
        handle["tree"]  # a split branch is what makes the streamers be read
        classes = handle._source.streamers()
        assert classes["Event"]["StlVecF64"].typename == "vector<double>"
        assert classes["Event"]["StdStr"].typename == "string"
        assert classes["Event"]["SliceF64"].typename == "double*"
        assert repr(classes["Event"]["StdStr"]) == "<Member string StdStr>"
        assert handle._source.streamers() is classes  # read once, then kept


def test_a_file_with_no_class_descriptions_describes_no_classes():
    source = Source(io.BytesIO(b""), "crafted", owned=True)
    assert read_streamers(source) == {}
    assert source.streamers() == {}


def test_a_streamer_element_from_2001_counted_its_own_dimensions():
    member = read_element(Buffer(element_bytes(1, "fN", "int")))
    assert (member.name, member.typename, member.stype, member.length) == ("fN", "int", 3, 1)


def test_a_class_written_with_no_member_list_has_no_members():
    body = named_bytes("Empty", "") + struct.pack(">Ii", 0, 1) + struct.pack(">I", 0)
    assert read_info(Buffer(record(9, body))) == ("Empty", {})


def test_a_record_read_by_the_wrong_layout_reports_the_bytes_it_took():
    """``TBufferFile::CheckByteCount``: what an object used against what its count said."""
    buf = Buffer(record(3, struct.pack(">ii", 1, 2)))
    start = buf.pos
    assert buf.header()[1] == start + 14  # the count word, the version and two numbers
    buf.i32()
    assert buf.check_count(start, start + 14) == (start + 10, start + 14)
    buf.i32()
    assert buf.check_count(start, start + 14) is None  # everything the count held was read


def test_a_record_without_a_byte_count_has_no_length_to_be_checked_against():
    buf = Buffer(struct.pack(">Hi", 3, 5))
    _version, end = buf.header()
    buf.i32()
    assert end is None
    assert buf.check_count(0, end) is None


def test_a_class_a_file_records_twice_keeps_both_layouts(tmp_path):
    """A merged file records one class per version, and every object names its own."""
    with opened("small-evnt-tree-fullsplit") as handle:
        entries = read_entries(handle._source)
        once = read_layouts(handle._source)
    (name, version), entry = next((pair, e) for pair, e in entries.items() if pair[0] == "Event")
    twice = dict(entries)
    twice[(name, version + 5)] = entry  # the same class, recorded a second time
    with create(tmp_path / "twice.root") as out:
        out._carried.update(twice)
        out["note"] = "twice"
    with open_root(str(tmp_path / "twice.root")) as again:
        assert read_layouts(again._source)[name] == once[name] * 2
        assert len(read_streamers(again._source)[name]) == len(once[name][0][2])


def test_a_layout_of_one_name_keeps_its_checksum_and_version():
    with opened("small-evnt-tree-fullsplit") as handle:
        layouts = read_layouts(handle._source)
        members = handle._source.streamers()
    checksum, version, elements = layouts["Event"][0]
    assert checksum == 0x42F2461B and version == 1
    assert [element[1] for element in elements] == list(members["Event"])


#: ``ParticleImplVec`` declared ``ClassDefOverride(ParticleImplVec,10)`` over a ``Particle``
#: whose own version moved from 11 to 12: the layout in memory, and the one a v1 file has.
IN_MEMORY = (
    0x8E42FEE8,
    10,
    (("TStreamerBase", "Particle", "", 0, 0, 0, 0, (0,) * 5, "BASE", (12,)),),
)
ON_FILE = (
    0xB5AD127C,
    10,
    (("TStreamerBase", "Particle", "", 0, 0, 0, 0, (0,) * 5, "BASE", (11,)),),
)


def test_a_class_whose_version_kept_but_whose_layout_moved_reads_as_declared():
    """ROOT's ``CreateReadMemberWiseActions`` warning, word for word from the ``.ref``."""
    layout, warning = reconcile("ParticleImplVec", IN_MEMORY, [ON_FILE], "mixedBase_v1.root")
    assert layout == IN_MEMORY
    assert warning == (
        "   The StreamerInfo of class ParticleImplVec read from file mixedBase_v1.root\n"
        "   has the same version (=10) as the active class but a different checksum.\n"
        "   You should update the version to ClassDef(ParticleImplVec,11).\n"
        "   The objects on this file might not be readable because:\n"
        "   The in-memory layout version 10 for class 'ParticleImplVec' has a base class "
        "(Particle) with version 12 but the on-file layout version 10 recorded the version "
        "number 11 for this base class (Particle)."
    )


def test_a_layout_that_differs_with_no_base_class_to_blame_still_warns_and_reads_declared():
    mine = (0x11111111, 4, ())
    theirs = (0x22222222, 4, ())
    layout, warning = reconcile("Tdata", mine, [theirs], "data1.root")
    assert layout == mine
    assert "might not be readable" not in warning  # no base class's version moved
    assert reconcile("Tdata", mine, [mine], "data1.root") == (mine, "")


def test_a_file_that_describes_another_version_of_a_class_is_left_to_the_record():
    """No layout at the version in memory means no disagreement to settle: the version the
    record itself names picks one from ``on_file``."""
    assert reconcile("Tdata", IN_MEMORY, [], "data1.root") == (None, "")
    other = (0x33333333, 9, ())
    assert reconcile("Tdata", IN_MEMORY, [other], "data1.root") == (None, "")


def test_a_class_whose_checksum_differs_over_a_base_class_nobody_moved_says_so_only_once():
    mine = (
        0x44444444,
        7,
        (
            ("TStreamerBase", "TObject", "", 0, 0, 0, 0, (0,) * 5, "BASE", (1,)),
            ("TStreamerBase", "Holder", "", 0, 0, 0, 0, (0,) * 5, "BASE", (9,)),
        ),
    )
    theirs = (0x55555555, 7, (("TStreamerBase", "Holder", "", 0, 0, 0, 0, (0,) * 5, "BASE", (9,)),))
    layout, warning = reconcile("GenParticle", mine, [theirs], "v1.root")
    assert layout == mine
    assert warning.splitlines() == [
        "   The StreamerInfo of class GenParticle read from file v1.root",
        "   has the same version (=7) as the active class but a different checksum.",
        "   You should update the version to ClassDef(GenParticle,8).",
    ]


def test_a_branch_element_from_before_the_version_shrank_still_names_its_class():
    assert read_branch_element(Buffer(branch_element_bytes(9, "Event"))).classname == "Event"
    assert read_branch_element(Buffer(branch_element_bytes(1, "Event"))).classname == "Event"
    assert not read_branch_element(Buffer(branch_element_bytes(10, "Event"))).streamed
    assert read_branch_element(Buffer(branch_element_bytes(10, "P3", -1))).streamed


def test_a_branch_of_a_kind_this_reader_cannot_follow_still_appears_by_name():
    """A ``TBranchObject`` and its two cousins: named and refused, not missing."""
    branch = read_derived_branch(Buffer(record(1, branch_bytes() + tstring("Event"))))
    assert branch.name == "b"
    assert branch.classname == ""  # nothing said what it holds, so nothing is guessed


# -- containers, against what go-hep reads --------------------------------


def test_a_vector_of_numbers_is_rows_of_numbers(containers):
    assert containers["vec_i32"].is_jagged
    assert containers["vec_i32"].array(0, 2).tolist() == [[-1], [-1, -2]]
    assert containers["vec_u32"].array(1, 2).tolist() == [[1, 2]]
    assert containers["lst_i32"].array(1, 2).tolist() == [[-1, -2]]
    assert containers["deq_i32"].array(1, 2).tolist() == [[-1, -2]]
    assert containers["set_i32"].array(1, 2).tolist() == [[-2, -1]]  # a set is sorted


def test_a_container_of_containers_is_a_list_of_arrays(containers):
    assert containers["vec_vec_i32"].typename == "list[list[int32]]"
    assert [list(map(list, e)) for e in containers["vec_vec_i32"].array(0, 2)] == [
        [[-1]],
        [[-1], [-1, -2]],
    ]
    assert [list(map(list, e)) for e in containers["vec_set_i32"].array(1, 2)] == [[[-1], [-2, -1]]]


def test_a_container_of_strings_is_a_list_of_strings(containers):
    assert containers["vec_str"].array(1, 2) == [["one", "two"]]
    assert containers["vec_tstr"].array(1, 2) == [["one", "two"]]
    assert containers["vec_vec_str"].array(1, 2) == [[["one"], ["one", "two"]]]
    assert containers["uset_str"].array(0, 1) == [["one"]]


def test_a_top_level_string_branch_is_written_bare(containers):
    assert containers["str"].array(0, 2) == ["one", "two"]
    assert containers["tstr"].array(0, 2) == ["one", "two"]
    assert containers["str"].typename == "str"


def test_a_map_comes_back_as_a_dict_whichever_way_it_is_keyed(containers):
    assert containers["map_i32_i16"].array(1, 2) == [{-2: -2, -1: -1}]
    assert containers["map_u32_u16"].array(1, 2) == [{1: 1, 2: 2}]
    assert containers["map_str_i16"].array(1, 2) == [{"one": -1, "two": -2}]
    assert containers["map_str_str"].array(1, 2) == [{"one": "ONE", "two": "TWO"}]
    assert containers["umap_str_str"].array(1, 2) == [{"one": "ONE", "two": "TWO"}]


def test_a_tstring_in_a_map_is_bytes_where_a_std_string_is_a_record(containers):
    both = {"one": "ONE", "two": "TWO"}
    assert containers["map_str_tstr"].array(1, 2) == [both]
    assert containers["map_tstr_str"].array(1, 2) == [both]
    assert containers["map_tstr_tstr"].array(1, 2) == [both]


def test_a_map_of_containers_keeps_the_containers(containers):
    assert plain(containers["map_i32_vec_i16"].array(1, 2)) == [{-2: [-1, -2], -1: [-1]}]
    assert containers["map_str_vec_str"].array(1, 2) == [{"one": ["one"], "two": ["one", "two"]}]
    assert plain(containers["map_i32_set_i16"].array(1, 2)) == [{-2: [-2, -1], -1: [-1]}]
    nested = containers["map_i32_vec_vec_i16"].array(1, 2)[0]
    assert {key: [list(row) for row in value] for key, value in nested.items()} == {
        -2: [[-1], [-1, -2]],
        -1: [[-1]],
    }


def test_every_column_of_the_container_file_but_none_is_readable(containers):
    assert containers.unreadable == {}
    assert len(containers.readable()) == len(containers.keys()) == 40


# -- a class split all the way down ---------------------------------------


def test_a_split_member_is_a_column_of_its_own(event):
    assert event["I16"].array(1, 2).tolist() == [1]
    assert event["U64"].array(1, 2).tolist() == [1]
    assert event["ArrayF64[10]"].array(1, 2).tolist() == [[1.0] * 10]
    assert event["SliceF64"].array(1, 2).tolist() == [[1.0]]
    assert event["N"].array(1, 2).tolist() == [1]


def test_a_split_object_inside_an_object_is_its_members(event):
    assert [event[f"P3.P{axis}"].array(1, 2)[0] for axis in "xyz"] == [0, 1.0, 0]
    assert [event[f"P3.P{axis}"].array(0, 1)[0] for axis in "xyz"] == [-1, 0.0, -1]


def test_the_three_kinds_of_string_a_split_class_can_hold(event):
    assert event["Beg"].array(1, 2) == ["beg-001"]  # a TString member
    assert event["Str"].array(1, 2) == ["evt-001"]
    assert event["StdStr"].array(1, 2) == ["std-001"]  # a std::string member
    assert event["StlVecStr"].array(1, 2) == [["vec-001"]]
    assert event["End"].array(1, 2) == ["end-001"]


def test_a_vector_member_of_a_split_class_is_rows(event):
    assert event["StlVecF64"].array(1, 2).tolist() == [[1.0]]
    assert event["StlVecI16"].array(0, 2).lengths().tolist() == [0, 1]
    assert event["StlVecF32"].typename == "float32"


def test_a_vector_of_bool_is_a_byte_an_element_not_a_bit():
    with opened("stdvec-bool-fullsplit-6.10.08") as handle:
        tree = handle["tree"]
        assert tree["Bool"].array(0, 3).tolist() == [1, 0, 1]
        assert tree["ArrayBool[10]"].array(2, 3).tolist() == [[True] * 10]
        assert tree["StlVecBool"].array(0, 3).tolist() == [[], [0], [1, 1]]
        assert tree["SliceBool"].array(0, 3).tolist() == [[], [0], [1, 1]]


# -- refusing, by name ----------------------------------------------------


def test_a_class_with_no_layout_this_reader_knows_is_refused_by_its_name():
    column = column_of("TLorentzVector")
    assert isinstance(column, Refused)
    assert "TLorentzVector, which is a C++ type this reader does not decode" in column.reason
    assert repr(column) == "<Refused unreadable>"
    assert "an unnamed type" in column_of("").reason


def test_a_map_a_dict_cannot_be_is_refused_in_those_words():
    assert "keyed by a container" in column_of("map<vector<int>,int>").reason
    assert "a map inside another container" in column_of("vector<map<int,int> >").reason
    assert "a map inside another container" in column_of("map<int,vector<map<int,int> > >").reason


def test_a_member_the_file_never_described_cannot_be_guessed_at():
    column = column_of("Event", member="fMissing", ltype=300)
    assert "streamer information does not describe" in column.reason


def test_a_streamer_type_this_reader_will_not_decode_is_named_in_words():
    # A pointer is read as what it points at, when the file says what that is.
    refused = {stype: words for stype, words in KINDS.items() if stype not in (63, 64)}
    for stype, words in refused.items():
        reason = column_of("Event", ltype=stype).reason
        assert reason == f"{words}, which this reader does not decode"


def test_a_leaf_that_is_not_a_leaf_this_reader_knows_keeps_its_reason():
    leaf = LeafRecord("TLeafQuantum")
    branch = BranchRecord()
    branch.leaves = [leaf]
    assert "not a kind of leaf" in build(branch, leaf, Nothing()).reason


def test_a_container_written_field_by_field_is_refused_rather_than_guessed():
    column = column_of("vector<vector<int> >")
    assert isinstance(column, Values)
    with pytest.raises(UnsupportedFeatureError, match="field by field"):
        column.value(Buffer(record(0x4000 | 6)), 0)


def test_a_map_written_pair_by_pair_is_read_a_key_then_its_value():
    """How a map inside an object streamed by its class goes - a ``TFormula``'s names."""
    column = column_of("map<int,int>")
    body = struct.pack(">Iiiii", 2, 7, 70, 8, 80)
    assert column.value(Buffer(record(6, body)), 0) == {7: 70, 8: 80}


def test_a_map_ordered_by_a_comparator_of_its_own_is_the_same_map():
    assert py_name(parse("map<TString,int,TFormulaParamOrder>")) == "dict[str, int32]"
    assert parse("map<int>") is None


class Fake:
    """Just enough of a basket for a row to be measured inside it."""

    def __init__(self, data: bytes, end: int) -> None:
        self.data = data
        self._end = end

    def starts(self, low: int, high: int, offset: int):
        return np.zeros(high - low, np.int64)

    def ends(self, low: int, high: int):
        return np.full(high - low, self._end, np.int64)


def test_a_row_claiming_more_values_than_it_holds_is_a_format_error():
    column = column_of("vector<int>")
    assert isinstance(column, Rows)
    basket = Fake(struct.pack(">IHI", BYTE_COUNT_MASK | 10, 6, 99), 14)
    with pytest.raises(FormatError, match="says it holds 99 values"):
        column.spans(basket, 0, 1, 0)


# -- floats packed into fewer bytes ---------------------------------------


def test_a_title_with_no_range_in_it_asks_for_the_default_packing():
    assert _range("") == (0.0, 0.0, 0.0)
    assert _range("x[10]") == (0.0, 0.0, 0.0)  # an array, which is not a range
    assert _range("f[0,0,16]") == (0.0, 0.0, 0.0)  # no slash: not a range either
    assert _range("x/d[10]") == (0.0, 0.0, 0.0)


def test_a_title_that_gives_a_range_gives_a_scale_factor():
    xmin, xmax, factor = _range("x/d[-1,1]")
    assert (xmin, xmax) == (-1.0, 1.0)
    assert factor == pytest.approx(0xFFFFFFFF / 2)
    assert _range("x/d[0,2,8]")[2] == pytest.approx(128.0)
    assert _range("[-1,1]")[:2] == (-1.0, 1.0)  # a title that is nothing but a range


def test_a_title_that_gives_only_a_bit_count_keeps_it_where_root_keeps_it():
    assert _range("x/d[0,0,10]")[0] == pytest.approx(10.1)
    assert _range("x/d[0,0,20]") == (0.0, 0.0, 0.0)  # too many bits to be kept


def test_a_range_can_be_written_in_units_of_pi():
    assert _range("x/d[-pi,2pi]")[:2] == pytest.approx((-3.141592653589793, 6.283185307179586))
    assert _range("x/d[-pi/4,pi/2]")[:2] == pytest.approx((-0.7853981633974483, 1.5707963267948966))
    assert _range("x/d[0,twopi]")[1] == pytest.approx(6.283185307179586)
    assert _range("x/d[0,2*pi]")[1] == pytest.approx(6.283185307179586)


def test_a_range_this_reader_cannot_read_is_refused_rather_than_ignored():
    assert _range("x/d[1,2,3,4]") is None
    assert _range("x/d[low,high]") is None
    assert _range("x/d[0,1,99]") is None
    leaf = LeafRecord("TLeafD32")
    leaf.title = "x/d[low,high]"
    assert "not a spelling this reader can turn into numbers" in _packed(leaf).reason


def test_a_scaled_double_comes_back_across_the_range_it_was_written_in():
    leaf = LeafRecord("TLeafD32")
    leaf.title, leaf.length = "x/d[0,4,2]", 3
    column = _packed(leaf)
    assert isinstance(column, Flat)
    assert column.itemsize == 4
    assert list(column.decode(struct.pack(">3I", 0, 2, 4))) == [0.0, 2.0, 4.0]


def test_a_packed_slice_is_rows_of_doubles():
    leaf = LeafRecord("TLeafF16")
    leaf.title = "x/f[0,0,10]"
    leaf.count = LeafRecord("TLeafI")
    column = _packed(leaf)
    assert isinstance(column, Rows)
    assert column.itemsize == 3


# -- a packed float split out of a class ----------------------------------


class Described:
    """A source that describes one class, the way a real file's streamers do."""

    def __init__(self, **members: Member) -> None:
        self.classes = {"Event": members}

    def streamers(self) -> dict[str, dict[str, Member]]:
        return self.classes


def packed_member(title: str, ltype: int, *, length: int = 1, stype: int = 9):
    """The column a ``Double32_t`` or ``Float16_t`` member of ``Event`` becomes."""
    record_ = BranchRecord()
    record_.name = record_.classname = "Event"
    leaf = LeafRecord("TLeafElement")
    leaf.name, leaf.ltype, leaf.length = "D", ltype, length
    record_.leaves = [leaf]
    source = Described(D=Member("D", title, stype, "Double32_t", length))
    return build(record_, leaf, source)


def test_a_streamer_element_keeps_the_comment_its_range_is_written_in():
    member = read_element(Buffer(element_bytes(4, "D", "Double32_t")))
    assert (member.name, member.title, member.typename) == ("D", "", "Double32_t")


def test_a_split_double32_member_is_unpacked_by_the_range_the_class_declares():
    column = packed_member("[0,4,2]", 9)
    assert isinstance(column, Flat)
    assert (column.typename, column.itemsize, column.length) == ("float64", 4, 1)
    assert list(column.decode(struct.pack(">3I", 0, 2, 4))) == [0.0, 2.0, 4.0]


def test_a_split_double32_member_with_no_range_is_the_float_it_was_written_as():
    column = packed_member("", 9)
    assert isinstance(column, Flat)
    assert column.itemsize == 4
    assert list(column.decode(struct.pack(">2f", 1.5, -2.5))) == [1.5, -2.5]


def test_a_split_float16_member_with_no_range_keeps_only_its_top_bits():
    column = packed_member("", 19, stype=19)
    assert isinstance(column, Flat)
    assert column.itemsize == 3  # an exponent byte and two of mantissa
    assert list(column.decode(struct.pack(">BH", 128, 0))) == [2.0]


def test_a_fixed_array_of_packed_floats_is_as_wide_as_the_class_says():
    column = packed_member("[0,4,2]", 9 + 20, length=10)
    assert isinstance(column, Flat)
    assert column.length == 10


def test_a_counted_run_of_packed_floats_is_rows_behind_one_marker_byte():
    column = packed_member("[0,4,2]", 9 + 40)
    assert isinstance(column, Rows)
    assert (column.header, column.counted, column.itemsize) == (1, False, 4)


def test_a_packed_member_of_a_class_the_file_does_not_describe_is_refused():
    reason = column_of("Event", member="D", ltype=9).reason
    assert "the range it was squeezed into is not knowable" in reason


def test_a_packed_member_whose_range_makes_no_sense_is_refused_by_its_spelling():
    reason = packed_member("[a,b]", 9).reason
    assert "not a spelling this reader can turn into numbers" in reason


# -- a split object, put back together ------------------------------------


def test_a_split_object_is_a_branch_that_gives_back_the_object(event):
    group = event["evt"]
    assert repr(group) == "<Group 'evt' of 39 members>"
    assert event.groups() == ["evt", "P3"]
    # What the C++ that wrote this file put in entry 1, member for member.
    row = group.array(1, 2)[0]
    assert row["Beg"] == "beg-001"
    assert (row["I16"], row["U64"], row["F64"]) == (1, 1, 1.0)
    assert row["Str"] == "evt-001"
    assert list(row["ArrayF32[10]"]) == [1.0] * 10
    assert row["StlVecI16"].tolist() == [1]


def test_an_object_inside_an_object_is_a_dictionary_inside_a_dictionary(event):
    assert event["evt"].array(1, 2)[0]["P3"] == {"Px": 0, "Py": 1.0, "Pz": 0}
    assert event["P3"].array(2, 3) == [{"Px": 1, "Py": 2.0, "Pz": 1}]


def test_the_members_are_the_columns_and_the_object_is_not_read_twice(event):
    assert "evt" in event.keys() and "evt" not in event.readable()
    assert event.unreadable == {}
    assert event.typenames()["evt"] == "dict"
    assert len(event["evt"]) == len(event) == 100
    assert event["evt"].array(5, 2) == []


def test_a_member_this_reader_will_not_decode_is_named_rather_than_dropped(event):
    event.branches["I16"].column = Refused("a shape no file has written")
    group = event["evt"]
    assert group.unreadable == {"I16": "a shape no file has written"}
    assert "I16" not in group.array(0, 1)[0]


def test_a_branch_holding_data_keeps_its_children_as_columns_beside_it():
    """A branch can have baskets of its own and branches under it as well.

    Only the ones holding nothing are the split objects, so this one stays a
    column and its child becomes a second column rather than a member of it.
    """
    parent, child = BranchRecord(), BranchRecord()
    parent.name = "top"
    parent.basket_seek = [0]
    for record_, name in ((parent, "top"), (child, "top.sub")):
        record_.name = name
        leaf = LeafRecord("TLeafI")
        leaf.name = name
        record_.leaves = [leaf]
    parent.branches = [child]
    tree = TTree("t", "", 0, [parent], Nothing())
    assert tree.keys() == ["top", "top.sub"]
    assert tree.groups() == []


# -- a whole object, written into the entry -------------------------------


class Layout:
    """A source describing exactly the classes a test needs it to."""

    def __init__(self, **classes: dict[str, Member]) -> None:
        self.classes = classes

    def streamers(self) -> dict[str, dict[str, Member]]:
        return self.classes


def declared(name: str, stype: int, typename: str = "", **rest) -> dict[str, Member]:
    """One class of one member, as a file's streamer information gives it."""
    kept = {"title": "", "length": 1, "count": ""} | rest
    return {name: Member(name, kept["title"], stype, typename, kept["length"], kept["count"])}


@pytest.fixture
def nosplit():
    with opened("small-evnt-tree-nosplit") as handle:
        yield handle["tree"]


def test_an_unsplit_object_is_a_dictionary_of_everything_the_class_declares(nosplit):
    # What the C++ that wrote this file put in entry 1, member for member.
    row = nosplit["evt"].array(1, 2)[0]
    _assert_unsplit_scalars(row)
    _assert_unsplit_collections(row)
    assert nosplit.typenames()["evt"] == "dict"
    assert nosplit.readable() == ["evt"] and nosplit.unreadable == {}


def _assert_unsplit_scalars(row):
    assert row["Beg"] == "beg-001"
    assert (row["I16"], row["U64"], row["F64"]) == (1, 1, 1.0)
    assert row["Str"] == "evt-001"
    assert row["P3"] == {"Px": 0, "Py": 1.0, "Pz": 0}


def _assert_unsplit_collections(row):
    assert row["ArrayF32"].tolist() == [1.0] * 10
    assert (row["N"], row["SliceI16"].tolist()) == (1, [1])
    assert (row["StdStr"], row["StlVecStr"]) == ("std-001", ["vec-001"])
    assert row["End"] == "end-001"


def test_the_same_events_written_split_and_unsplit_read_back_the_same(nosplit, event):
    """The two files hold the same 100 events, written the two ways ROOT can.

    A split file names a fixed-size array's branch ``ArrayF32[10]`` where the
    class calls the member ``ArrayF32``, so the names are compared without the
    size C++ never had in them either.
    """

    def plain(row):
        return {
            name.partition("[")[0]: value.tolist() if hasattr(value, "tolist") else value
            for name, value in row.items()
        }

    unsplit, split = nosplit["evt"].array(), event["evt"].array()
    assert len(unsplit) == len(split) == 100
    assert [plain(row) for row in unsplit] == [plain(row) for row in split]


def test_an_unsplit_tree_of_maps_reads_them_including_the_empty_one():
    with opened("std-map-split0") as handle:
        tree = handle["tree"]
        empty = {name: {} for name in ("mi32", "msi32", "mss", "msvs", "msvi32")}
        assert tree["evt"].array(0, 1) == [empty]
        row = tree["evt"].array(2, 3)[0]
        assert row["mi32"] == {0: 0, 1: 1}
        assert row["mss"] == {"key-000": "val-000", "key-001": "val-001"}
        assert row["msvs"]["key-001"] == ["val-001", "val-002", "val-003"]


def test_a_member_whose_class_the_file_does_not_describe_is_refused_by_name():
    source = Layout(Event=declared("p", 62, "P3"))
    assert "a member of type P3, which this file's streamer information does not describe" in (
        column_of("Event", source=source).reason
    )


def test_a_class_written_inside_itself_is_refused_rather_than_read_forever():
    source = Layout(Event=declared("inner", 62, "Event"))
    assert "Event, which is written inside itself" in column_of("Event", source=source).reason


def test_a_counted_member_written_before_its_count_is_refused():
    source = Layout(Event=declared("SliceI32", 43, "int*", count="N"))
    assert "holds as many values as N but is written before it" in (
        column_of("Event", source=source).reason
    )
    unnamed = Layout(Event=declared("SliceI32", 43, "int*"))
    assert "as a member with no name" in column_of("Event", source=unnamed).reason


def test_a_member_of_a_kind_this_reader_does_not_decode_is_named_in_the_refusal():
    source = Layout(Event=declared("obj", 500, "Widget"))
    assert "'obj', which is a class with a streamer of its own that this reader does not" in (
        column_of("Event", source=source).reason
    )


def test_a_container_no_dict_can_hold_is_refused_inside_an_unsplit_object():
    source = Layout(Event=declared("m", 500, "map<vector<int>,int>"))
    assert "keyed by a container" in column_of("Event", source=source).reason


def test_a_packed_float_inside_an_unsplit_object_is_unpacked_by_its_range():
    source = Layout(Event=declared("d", 9, "Double32_t", title="[0,4,2]"))
    column = column_of("Event", source=source)
    assert isinstance(column, Values)
    assert column.value(Buffer(struct.pack(">I", 2)), 0) == {"d": 2.0}


def test_a_packed_float_whose_range_makes_no_sense_is_refused_inside_one_too():
    source = Layout(Event=declared("d", 9, "Double32_t", title="[1,2,3,4]"))
    assert "'d', a packed float whose range is written '[1,2,3,4]'" in (
        column_of("Event", source=source).reason
    )


def test_a_fixed_array_of_packed_floats_inside_an_unsplit_object_is_a_row():
    # ``Double32_t fVertex[3]; //[-30,30,16]``: three steps across the range, read
    # by one struct and one division, in the shape the declaration gives them.
    source = Layout(Event=declared("v", 9 + 20, "Double32_t", title="[-30,30,16]", length=3))
    row = column_of("Event", source=source).value(Buffer(struct.pack(">3I", 0, 2048, 4095)), 0)
    assert row["v"].shape == (3,)
    steps = np.frombuffer(struct.pack(">3I", 0, 2048, 4095), ">u4")
    assert list(row["v"]) == list(steps * (60 / 65536) - 30)


def test_a_fixed_array_of_packed_floats_with_no_recipe_is_the_floats_they_were_written_as():
    # A ``Double32_t[3]`` whose declaration asks for nothing: ROOT wrote three floats,
    # which is what a reader that cannot widen them any other way gives back as doubles.
    source = Layout(Event=declared("v", 9 + 20, "Double32_t", length=3))
    row = column_of("Event", source=source).value(Buffer(struct.pack(">3f", 1.5, -2.5, 0.5)), 0)
    assert list(row["v"]) == [1.5, -2.5, 0.5]


def _single_value(buf: Buffer, _row: dict[str, object]) -> object:
    """A member read by its own step rather than gathered into a struct with the ones by it."""
    return buf.i32()


def test_a_run_of_plain_members_is_read_by_one_struct_and_keeps_its_order():
    """A class of numbers reads its numbers a struct at a time, member for member in order.

    Reading a run in one go is what makes a whole object of a few hundred numbers
    affordable; what it must not do is read one member's numbers as another's, or
    put them back under their names in a different order.
    """
    steps = [
        ("i", _Plain("i", "int32", 1, ())),
        ("f", _Plain("f", "float32", 1, ())),
        ("other", _single_value),
        ("m", _Plain("d", "float64", 4, (2, 2))),
        ("d", _Plain("d", "float64", 1, ())),
    ]
    plan = _plan(steps)
    assert len(plan) == 3  # the two numbers before the step that is not one, then each side
    written = struct.pack(">ifi4dd", -3, 1.5, 7, 1, 2, 3, 4, 8.25)
    buf, row = Buffer(written), {}
    for step in plan:
        step(buf, row)
    assert list(row) == ["i", "f", "other", "m", "d"], "the members keep their places"
    assert (row["i"], row["f"], row["other"], row["d"]) == (-3, 1.5, 7, 8.25)
    assert list(np.asarray(row["m"]).reshape(-1)) == [1.0, 2.0, 3.0, 4.0]
    assert np.asarray(row["m"]).shape == (2, 2), "a declaration's shape is kept"
    assert _form("i") is _form("i")  # built once, however many members are read with it


def test_a_bool_column_is_read_as_true_and_false_rather_than_one_and_zero():
    assert list(Buffer(struct.pack(">3?", True, False, True)).run(3, "bool")) == [
        True,
        False,
        True,
    ]


def test_a_member_declared_as_an_array_of_one_value_is_an_array_of_one_value():
    """``Float_t v[1]`` is an array of one value as ``Float_t v[3]`` is an array of three.

    The shape is what a formula indexes a member by, and the one dimension is what makes
    a ``char[1]`` the string ``TLeafC`` reads; a run of single numbers hands back bare
    values, so a declaration of one value keeps a read of its own - and what it gives
    must not depend on the members declared beside it.
    """
    source = Layout(
        Event={
            **declared("n", 3, "int32"),
            **declared("v", OFFSET_L + 5, "Float_t", length=1),
            **declared("c", OFFSET_L + 1, "char", length=1),
            **declared("b", OFFSET_L + 18, "Bool_t", length=1),
        }
    )
    row = column_of("Event", source=source).value(Buffer(struct.pack(">ifbb", 7, 2.5, 97, 1)), 0)
    assert row["n"] == 7, "a member of one number is still the number"
    assert (row["v"].shape, row["v"].dtype, float(row["v"][0])) == ((1,), np.float32, 2.5)
    assert (row["c"].shape, row["c"].dtype) == ((1,), np.int8), "twenty chars of one is a string"
    assert (row["b"].shape, row["b"].dtype, bool(row["b"][0])) == ((1,), np.bool_, True)


def test_a_declaration_of_one_value_in_two_dimensions_keeps_both_beside_a_number():
    """``Double_t m[1][1]`` is a matrix of one cell alone and a matrix of one cell next to
    a plain number: which member a run gathers must not decide another's shape."""
    matrix = {"m": Member("m", "", OFFSET_L + 8, "float64", 1, "", dims=(1, 1))}
    beside = {**declared("a", 3, "int32"), **matrix}
    alone = column_of("Event", source=Layout(Event=matrix)).value(Buffer(struct.pack(">d", 5.0)), 0)
    with_number = column_of("Event", source=Layout(Event=beside)).value(
        Buffer(struct.pack(">id", 1, 5.0)), 0
    )
    assert alone["m"].shape == with_number["m"].shape == (1, 1)
    assert alone["m"].tolist() == with_number["m"].tolist() == [[5.0]]
    assert with_number["a"] == 1


def test_a_class_inside_one_with_a_version_of_its_own_carries_no_checksum():
    source = Layout(Event=declared("p", 62, "P3"), P3=declared("x", 3, "int"))
    column = column_of("Event", source=source)
    written = struct.pack(">IHi", BYTE_COUNT_MASK | 6, 5, 42)
    assert column.value(Buffer(written), 0) == {"p": {"x": 42}}


def test_a_split_member_of_a_type_this_reader_does_not_know_is_still_refused():
    """A member of a split class is refused by type, not read member by member.

    Only the branch holding a whole object has the bytes of one; a member that
    is a class of its own was split into branches beneath it, and if it was
    not, the file says so with a type this reader has no reader for.
    """
    source = Layout(Event=declared("vtx", 500, "TLorentzVector"))
    column = column_of("Event", member="vtx", ltype=300, source=source)
    assert "TLorentzVector, which is a C++ type this reader does not decode" in column.reason


def test_a_member_held_by_value_is_read_with_the_record_around_it():
    """ROOT splits ``tref_test_pid::fRef`` as the whole ``TRef`` every object holds, its
    ``fStreamerType`` 61 saying the object is written into the entry where it stands - the
    record that says where it ends, then its members - rather than pointed at.

    The roottest ``root/io/tree`` split collections of that class are read only this way.
    """
    source = Layout(Event=declared("fRef", 61, "TRef"), TRef=declared("fUniqueID", 3, "int"))
    column = column_of("Event", member="fRef", ltype=61, source=source)
    assert isinstance(column, Values) and column.typename == "object"
    written = struct.pack(">IHi", BYTE_COUNT_MASK | 6, 5, 42)
    assert column.value(Buffer(written), 0) == {"fUniqueID": 42}


def test_a_branch_this_reader_cannot_decode_is_named_in_the_tree_not_dropped():
    """A refused column is still a key: it is listed, with the reason, not hidden."""
    record_ = BranchRecord()
    record_.name = "obj"
    leaf = LeafRecord("TLeafObject")
    leaf.name = "obj"
    record_.leaves = [leaf]
    tree = TTree("t", "", 0, [record_], Nothing())
    assert tree.keys() == ["obj"] and tree.readable() == []
    assert tree.unreadable["obj"] == tree["obj"].column.reason


# -- a class with a base class --------------------------------------------


def test_a_class_that_streamed_itself_reads_the_record_it_put_in_front():
    """`TLorentzVector` writes itself, so the entry is its record and not its members.

    The values are the ones go-hep reads out of the same ten entries: a
    momentum of ``(i, i+1, i+2)`` and an energy of ``i+3``.
    """
    with opened("tlv-split99") as handle:
        rows = handle["tree"]["p4"].array()
        assert len(rows) == 10
        assert rows[0]["fP"]["fX"] == 0.0 and rows[0]["fE"] == 3.0
        assert [row["fP"]["fZ"] for row in rows[:3]] == [2.0, 3.0, 4.0]
        assert rows[0]["TObject"] == {"fUniqueID": 0, "fBits": 0x03000000}


def test_the_older_branch_that_names_its_class_holds_the_same_objects():
    """A `TBranchObject` writes the class name in front of every entry.

    It is the same ten `TLorentzVector`s as the file above, written the way
    ROOT wrote objects before `TBranchElement` existed.
    """
    with opened("tlv-split00") as first, opened("tlv-split99") as second:
        assert first["tree"]["p4"].array() == second["tree"]["p4"].array()


def test_a_branch_that_says_one_class_and_holds_another_stops_rather_than_reads():
    source = Layout(P3=declared("x", 3, "int"))
    record_ = BranchRecord()
    record_.name, record_.classname = "p", "P3"
    leaf = LeafRecord("TLeafObject")
    leaf.name = "p"
    record_.leaves = [leaf]
    column = build(record_, leaf, source)
    written = bytes([2]) + b"P4\x00" + record(1, struct.pack(">i", 7))
    with pytest.raises(FormatError, match="says it holds 'P4' instead"):
        column.value(Buffer(written), 0)


def test_a_base_class_keeps_its_own_members_under_its_own_name():
    """`D2` declares an `I32` of its own and inherits another one.

    Nesting the base under its name is what keeps both: flattening them into
    one dictionary would quietly drop whichever was written first.
    """
    with opened("tbase") as handle:
        tree = handle["tree"]
        assert tree["d1"].array() == [
            {"Base": {"I32": 1}, "D32": 2},
            {"Base": {"I32": 2}, "D32": 3},
        ]
        assert tree["d2"].array(0, 1) == [{"Base": {"I32": 3}, "I32": 4}]


def test_the_rvec_an_rdataframe_writes_reads_back_as_the_vector_it_is():
    """`RVec` is `RDataFrame`'s vector, and is written exactly like one.

    The numbers are the ones go-hep dumps out of the same file.
    """
    with opened("rvec") as handle:
        tree = handle["events"]
        assert tree.unreadable == {}
        momenta = tree["MC_px"].array(0, 1)
        assert momenta.lengths() == [192]
        # float(), because round() of a NumPy scalar keeps its own dtype on NumPy 1 and
        # returns a Python float on NumPy 2 - the same number, rounded differently.
        assert [round(float(value), 5) for value in momenta[0][6:9]] == [0.0, 30.39946, -30.39946]
        assert tree["MC_pdg"].array(0, 1)[0][:5].tolist() == [11.0, -11.0, 11.0, -11.0, 23.0]
        assert round(float(tree["EVT_thrust_x"].array(0, 1)[0]), 5) == -5.89043


def test_a_bitset_is_a_byte_a_bit_with_the_lowest_bit_first():
    """The file holds `bitset<8>("00010001")`, written the way a vector is.

    C++ writes a `bitset` most significant bit first, so the row is that
    string backwards - which is `bs[0]`, `bs[1]`, ... in the order C++ asks
    for them.
    """
    with opened("std-bitset") as handle:
        rows = handle["tree"]["evt"].array(0, 2)
    assert rows[0]["Bs8"].tolist() == [1, 0, 0, 0, 1, 0, 0, 0]
    assert [bits.tolist() for bits in rows[0]["VecBs8"]] == [[0, 1, 1, 1, 0, 1, 1, 1]]
    assert rows[1]["Bs8"].tolist() == [1, 0, 0, 1, 1, 0, 0, 1]
    assert len(rows[1]["VecBs8"]) == 2


def test_a_bitset_of_something_that_is_not_a_width_is_not_a_bitset():
    assert parse("bitset<8>") is not None and parse("bitset<n>") is None


def test_a_tnamed_base_gives_back_the_name_and_title_it_carries():
    source = Layout(Event={"TNamed": Member("TNamed", "", 67, "BASE", 0)})
    column = column_of("Event", source=source)
    assert column.value(Buffer(named_bytes("a name", "a title")), 0) == {
        "TNamed": {"fName": "a name", "fTitle": "a title", "fUniqueID": 0, "fBits": 0}
    }


def shaped_bytes(dimensions: int, sizes: tuple[int, ...], length: int) -> bytes:
    """A ``TStreamerElement`` of a fixed array in ``dimensions`` dimensions."""
    body = named_bytes("m", "") + struct.pack(">iiii", 3, 4, length, dimensions)
    body += struct.pack(">5i", *(sizes + (0,) * (5 - len(sizes))))
    return record(4, record(4, body + tstring("Double32_t")))


def test_a_fixed_array_keeps_the_shape_the_file_declares():
    """ROOT writes how many dimensions a member has and each one's maximum index, so
    `m[4][4]` is a (4, 4) member and `m[16]` is a flat one, not the same thing."""
    shaped = read_element(Buffer(shaped_bytes(2, (4, 4), 16)))
    assert (shaped.length, shaped.dims) == (16, (4, 4))
    assert read_element(Buffer(shaped_bytes(1, (16,), 16))).dims == ()
    # A description that does not add up says nothing about a shape.
    assert read_element(Buffer(shaped_bytes(2, (4, 5), 16))).dims == ()


def test_a_class_whose_layout_moved_says_so_once_per_file(monkeypatch, capsys):
    """``TStreamerInfo::CreateReadMemberWiseActions``, said where ROOT says it: while the
    collection that needs the actions is being read, and only the once."""
    from xrdroot import interp

    source = Source(io.BytesIO(b""), "mixedBase_v1.root", owned=True)
    monkeypatch.setattr(interp, "read_layouts", lambda _source: {"ParticleImplVec": [ON_FILE]})
    interp._divergence(source, "Tdata")()  # a class this program does not declare
    assert capsys.readouterr().out == ""
    try:
        interp.declare("ParticleImplVec", IN_MEMORY)
        report = interp._divergence(source, "ParticleImplVec")
        report()
        report()
    finally:
        interp.DECLARED.clear()
    said = capsys.readouterr().out
    assert said.startswith("Warning in <CreateReadMemberWiseActions>:    The StreamerInfo")
    assert said.count("Warning in <CreateReadMemberWiseActions>:") == 1


def test_a_class_whose_layout_did_not_move_says_nothing_and_is_not_asked_again(
    monkeypatch, capsys
):
    """A class the file records at the same checksum as memory has nothing to report, and
    the question is not asked again for the next collection of it."""
    from xrdroot import interp

    source = Source(io.BytesIO(b""), "mixedBase_v2.root", owned=True)
    asked: list[str] = []

    def looked(_source: Source) -> dict[str, list[tuple[int, int, tuple[tuple[Any, ...], ...]]]]:
        asked.append("read")
        return {"ParticleImplVec": [IN_MEMORY]}

    monkeypatch.setattr(interp, "read_layouts", looked)
    try:
        interp.declare("ParticleImplVec", IN_MEMORY)
        report = interp._divergence(source, "ParticleImplVec")
        report()
        monkeypatch.setattr(
            interp, "read_layouts", lambda _source: {"ParticleImplVec": [ON_FILE]}
        )
        report()
    finally:
        interp.DECLARED.clear()
    assert capsys.readouterr().out == ""
    assert asked == ["read"]


def test_a_record_a_changed_class_overran_is_reported_and_read_on_from_its_end(capsys):
    """``TBufferFile::CheckByteCount``: the bytes a member-wise record took against its own
    count - said, and then carried on from the end the record named, as ROOT does."""
    from xrdroot import interp

    source = Source(io.BytesIO(b""), "mixedBase_v1.root", owned=True)
    buf = Buffer(record(1, b"\0" * 20) + b"\0" * 12)
    start = buf.pos
    _version, end = buf.header()
    assert end is not None
    buf.pos = end + 12  # three items, each a member longer than the file recorded
    interp._byte_count("vector<ParticleImplVec>", source)(buf, start, end)
    # Both counts start past the record's byte count word, as ROOT's own message does: a
    # record whose word says 48 is reported as "56 instead of 44" (mixedBase_v1).
    assert capsys.readouterr().out.splitlines() == [
        "Error in <TBufferFile::CheckByteCount>: object of class vector<ParticleImplVec> "
        f"read too many bytes: {end + 8} instead of {end - 4}",
        "Warning in <TBufferFile::CheckByteCount>: vector<ParticleImplVec>::Streamer() not in "
        "sync with data on file mixedBase_v1.root, fix Streamer()",
    ]
    assert buf.pos == end  # every object after the record is read from where it says


def test_a_record_a_changed_class_ended_exactly_as_counted_says_nothing(capsys):
    """The bytes taken are the record's own count: nothing to report, and the record is
    left where the walk ended it."""
    from xrdroot import interp

    source = Source(io.BytesIO(b""), "mixedBase_v2.root", owned=True)
    buf = Buffer(record(1, b"\0" * 20))
    start = buf.pos
    _version, end = buf.header()
    assert end is not None
    buf.pos = end
    interp._byte_count("vector<ParticleImplVec>", source)(buf, start, end)
    assert capsys.readouterr().out == ""
    assert buf.pos == end


def test_a_member_wise_record_that_ended_short_is_still_refused():
    """A record read by a layout that describes somebody else's bytes is not forgiven: only
    a class the program declares has an account of why it did not end where it said."""
    from xrdroot import interp

    source = Source(io.BytesIO(b""), "mixedBase_v1.root", owned=True)
    buf = Buffer(record(1, b"\0" * 20))
    start = buf.pos
    _version, end = buf.header()
    assert end is not None
    buf.pos = end - 4
    with pytest.raises(FormatError, match="ended 4 bytes"):
        interp._byte_count("vector<ParticleImplVec>", source)(buf, start, end)


def test_a_member_wise_container_is_read_by_its_columns_and_checked_afterwards():
    """The one shape ``_objects`` reads that has a record of its own to be checked against."""
    from xrdroot import interp

    said: list[str] = []
    buf = Buffer(record(interp.MEMBER_WISE, struct.pack(">hi", 1, 0)))
    warning = lambda: said.append("warning")  # noqa: E731 - a report of one line
    counted = lambda _buf, _start, _end: said.append("counted")  # noqa: E731
    read = interp._objects(lambda _buf: {}, [], (warning, counted))
    assert read(buf) == []
    assert said == ["warning", "counted"]


def test_the_items_of_a_member_wise_collection_are_read_as_the_class_is_today(monkeypatch):
    """A file that records the class at the version in memory under another checksum: the
    items go straight into objects the program has made, so they are read by the layout in
    memory - bases and all - and the record then ends where it did not say."""
    from xrdroot import interp

    source = Source(io.BytesIO(b""), "mixedBase_v1.root", owned=True)
    monkeypatch.setattr(interp, "read_layouts", lambda _source: {"ParticleImplVec": [ON_FILE]})
    assert interp._read_by_memory(source, "Tdata") is source  # nothing declared
    try:
        interp.declare("ParticleImplVec", IN_MEMORY)
        laid_out = interp._read_by_memory(source, "ParticleImplVec")
        assert set(laid_out.streamers()["ParticleImplVec"]) == {"Particle"}
        assert laid_out.name == "mixedBase_v1.root"
        # The file's own classes are still the file's, and a class it records at another
        # version is read by that version, as a record of it anywhere else is.
        assert "Tdata" not in laid_out.streamers()  # only the declared names are laid out
        monkeypatch.setattr(
            interp, "read_layouts", lambda _source: {"ParticleImplVec": [(0x12345678, 9, ())]}
        )
        assert interp._read_by_memory(source, "ParticleImplVec") is source
    finally:
        interp.DECLARED.clear()
