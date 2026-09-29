"""The workspace reader's byte walker and makers, on records crafted by hand.

ROOT writes the shapes below rarely or never in the files a test can carry - an old
``RooRealVar`` streamer version, a ``TList`` of version three, a map written member-wise, a
class reference pointing nowhere - so each is spelled out byte by byte as ROOT's streamers
lay it out, and what the reader makes of it, or the refusal it gives, is checked. The
makers are then given records naming the variants no file here holds: a variable with no
binning or a range binning, a category with no states, a code repository with classes of
its author's.
"""

from __future__ import annotations

import struct
from types import SimpleNamespace
from typing import Any

import pytest

from xrdroot.buffer import Buffer
from xrdroot.errors import FormatError, UnsupportedFeatureError
from xrdroot.roofit.io import custom, data, models, variables, workspace
from xrdroot.roofit.io.build import Builder, dress, name_of, title_of
from xrdroot.roofit.io.stream import Reader, Streamed, base

BYTE_COUNT, NEW_CLASS, CLASS_MASK = 0x40000000, 0xFFFFFFFF, 0x80000000


def record(version: int, body: bytes = b"") -> bytes:
    """A record: its byte count, its version, its body."""
    return struct.pack(">IH", BYTE_COUNT | (len(body) + 2), version) + body


def bare(version: int, body: bytes = b"") -> bytes:
    """A record written with a version and no byte count, as ``RooLinkedList`` writes one."""
    return struct.pack(">H", version) + body


def text(value: str) -> bytes:
    return bytes([len(value)]) + value.encode()


def tobject() -> bytes:
    return struct.pack(">HII", 1, 0, 0)


def i32(value: int) -> bytes:
    return struct.pack(">i", value)


def u32(value: int) -> bytes:
    return struct.pack(">I", value)


def f64(*values: float) -> bytes:
    return struct.pack(f">{len(values)}d", *values)


def new_object(cls: str, body: bytes, slack: int = 0) -> bytes:
    """A pointer to an object written here: byte count, new-class tag, name, the object."""
    rest = u32(NEW_CLASS) + cls.encode() + b"\0" + body
    return u32(BYTE_COUNT | (len(rest) + slack)) + rest


def member(name: str, stype: int, typename: str = "", length: int = 0,
           count: str = "") -> Any:  # fmt: skip
    return SimpleNamespace(name=name, stype=stype, typename=typename, length=length, count=count)


def reader(raw: bytes, infos: Any = None) -> Reader:
    return Reader(Buffer(raw), infos or {})


def test_a_record_shows_its_class_and_name_and_finds_members_in_its_bases() -> None:
    """``get`` looks through the bases' members; ``repr`` names the class and the object."""
    made = Streamed("RooThing")
    inner = base("TNamed")
    inner.m["fName"] = "thing"
    made.m["TNamed"] = inner
    assert (made.get("fName"), made.get("absent", 7), repr(made)) == (
        "thing", 7, "<Streamed RooThing 'thing'>")  # fmt: skip


def test_a_pointer_is_an_object_a_null_or_one_read_before() -> None:
    """A new object is registered under its offset, and a later reference is the same one."""
    first = new_object("Empty", record(1))
    again = u32(BYTE_COUNT | 10) + u32(CLASS_MASK | 6) + record(1)  # the class, by its offset
    same = u32(2)  # the object at offset 0, plus kMapOffset
    walker = reader(first + again + same + u32(0), {"Empty": {}})
    one, two = walker.pointer(), walker.pointer()
    assert (one.cls, two.cls, two is not one) == ("Empty", "Empty", True)
    assert walker.pointer() is one
    assert walker.pointer() is None


def test_a_class_reference_pointing_nowhere_is_a_format_error() -> None:
    with pytest.raises(FormatError, match="points nowhere"):
        reader(u32(BYTE_COUNT | 4) + u32(CLASS_MASK | 1234)).pointer()


def test_an_object_ending_short_of_its_byte_count_is_a_format_error() -> None:
    """The object's own record ends before the pointer's byte count says it does."""
    with pytest.raises(FormatError, match="ended 2 bytes short of its end"):
        reader(new_object("Empty", record(1), slack=2) + b"\0\0", {"Empty": {}}).pointer()


def test_a_class_the_file_does_not_describe_is_refused_by_name() -> None:
    with pytest.raises(UnsupportedFeatureError, match="does not describe the class Mystery"):
        reader(record(1)).object("Mystery")


def test_a_member_of_a_kind_the_reader_does_not_decode_is_refused() -> None:
    infos = {"K": {"odd": member("odd", 99, "Weird")}}
    with pytest.raises(UnsupportedFeatureError, match="as a kind \\(99\\)"):
        reader(record(1), infos).object("K")


def container(typename: str, raw: bytes) -> Any:
    infos = {"K": {"c": member("c", 500, typename)}}
    return reader(record(1, raw), infos).object("K").get("c")


def test_a_vector_of_objects_is_read_object_by_object() -> None:
    """``vector<Empty>``: its count, then each element's own record."""
    found = reader(record(1, u32(2) + record(1) + record(1)), {"Empty": {}})
    assert [one.cls for one in found.container("vector<Empty>")] == ["Empty", "Empty"]


def test_a_vector_of_objects_written_member_wise_is_refused() -> None:
    with pytest.raises(UnsupportedFeatureError, match="written field by field"):
        container("vector<Empty>", record(0x4001, u32(1)))


def test_a_container_that_is_not_a_sequence_or_map_is_refused() -> None:
    with pytest.raises(UnsupportedFeatureError, match="pair<int,int> is not a container"):
        container("pair<int,int>", record(1))


def test_a_map_is_read_pair_by_pair_or_member_wise_block_by_block() -> None:
    """``map<string,int>`` element-wise; member-wise, the keys in a record, then the values."""
    pairs = record(1, u32(2) + text("a") + i32(1) + text("b") + i32(2))
    assert container("map<string,int>", pairs) == {"a": 1, "b": 2}
    keys = record(1, text("x") + text("y"))
    wise = record(0x4001, struct.pack(">h", 1) + u32(2) + keys + i32(3) + i32(4))
    assert container("map<string,int>", wise) == {"x": 3, "y": 4}
    empty = record(0x4001, struct.pack(">h", 0) + u32(0) + u32(0))
    assert container("map<string,int>", empty) == {}


def test_the_code_repository_of_an_old_workspace_has_no_extra_headers() -> None:
    """Version one of ``CodeRepo`` writes its files and relations, and no third map."""
    raw = record(1, i32(1) + b"".join(text(t) for t in "abcd") + i32(1)
                 + b"".join(text(t) for t in "efg"))  # fmt: skip
    made = reader(raw).object("RooWorkspace::CodeRepo")
    assert (made.get("files"), made.get("relations"), made.get("extras")) == (
        [("a", "b", "c", "d")], [("e", "f", "g")], [])  # fmt: skip


def test_an_old_linked_list_keeps_no_name_and_a_middle_one_does() -> None:
    """``RooLinkedList`` of version one has no name; of versions two and three, one."""
    assert "_name" not in reader(bare(1, tobject() + i32(0))).object("RooLinkedList").m
    named = reader(bare(3, tobject() + i32(0) + text("cset"))).object("RooLinkedList")
    assert named.get("_name") == "cset"


def test_an_old_binning_has_a_bare_tobject_for_its_name() -> None:
    made = reader(record(1, tobject() + record(1))).object("RooAbsBinning")
    assert made.get("TNamed") == {"fUniqueID": 0, "fBits": 0}


def test_a_list_of_version_three_has_no_name_and_no_options() -> None:
    """``TList`` before version four: a count and the objects, with no options."""
    made = reader(record(3, i32(1) + u32(0))).object("TList")
    assert (made.get("items"), made.get("options"), "fName" in made.m) == ([None], [""], False)


@pytest.mark.parametrize("version", [1, 3])
def test_old_real_variables_stream_their_fit_values_or_a_shared_pointer(version: int) -> None:
    """``RooRealVar`` version one has the fit's numbers; version three its properties' pointer."""
    body = record(1) + (f64(1.0, 2.0) + i32(3) if version == 1 else b"") + f64(0.5, -0.1, 0.2)
    body += u32(0) + u32(0) if version == 3 else b""
    made = reader(record(version, body), {"RooAbsRealLValue": {}}).object("RooRealVar")
    assert (made.get("_error"), made.get("_asymErrLo"), made.get("_asymErrHi")) == (0.5, -0.1, 0.2)
    assert made.get("fit") == ((1.0, 2.0, 3) if version == 1 else None)
    assert made.get("_sharedProp") is None and made.get("_binning") is None


def made_of(cls: str, name: str = "", **members: Any) -> Streamed:
    """A record of ``cls`` named ``name``, with ``members``."""
    made = Streamed(cls)
    made.m["TNamed"] = {"fName": name, "fTitle": f"{name} title"}
    made.m.update(members)
    return made


def real(name: str, value: float, binning: Any = None) -> Streamed:
    return made_of("RooRealVar", name, _value=value, _binning=binning, _error=-1.0,
                   _asymErrLo=0.0, _asymErrHi=0.0, _sharedProp=None)  # fmt: skip


def test_a_variable_with_no_binning_is_unbounded_and_one_with_a_range_binning_is_ranged() -> None:
    """No binning is ``(-inf, inf)``; a ``RooRangeBinning`` of ``1e30`` ends is unbounded too."""
    free = Builder().node(real("a", 1.5))
    assert (free.getVal(), free.getMin(), free.getMax(), free.GetTitle()) == (
        1.5, float("-inf"), float("inf"), "a title")  # fmt: skip
    ranged = made_of("RooRangeBinning", "r", _range=[-1e30, 4.0])
    found = Builder().node(real("b", 2.0, ranged))
    assert (found.getMin(), found.getMax()) == (float("-inf"), 4.0)
    assert variables.binning(None) is None


def test_a_binning_class_the_engine_does_not_read_is_refused_by_name() -> None:
    with pytest.raises(UnsupportedFeatureError, match="binned by a RooParamBinning"):
        variables.binning(made_of("RooParamBinning"))


def test_a_category_with_no_states_is_made_empty() -> None:
    made = Builder().node(made_of("RooCategory", "c", _stateNames={}, _insertionOrder=[]))
    assert (made.GetName(), made.numTypes()) == ("c", 0)


def test_a_dataset_with_no_vector_store_is_refused() -> None:
    with pytest.raises(UnsupportedFeatureError, match="'d' is kept in no store"):
        Builder().node(made_of("RooDataSet", "d", _dstore=None))


def test_a_node_is_dressed_with_the_attributes_it_was_written_with() -> None:
    """Boolean and string attributes, a forced numeric integral and a unit, as written."""
    node = Builder().node(real("x", 0.0))
    dress(node, made_of("", _boolAttrib=["tag"], _stringAttrib={"k": "v"}, _forceNumInt=True,
                        _unit="cm"))  # fmt: skip
    assert (node.getAttribute("tag"), node.getStringAttribute("k"), node.getUnit()) == (
        True, "v", "cm")  # fmt: skip
    assert (name_of(None), name_of(Streamed("X")), title_of(Streamed("X"))) == ("", "", "")


def test_a_class_with_no_maker_is_refused_naming_the_object() -> None:
    with pytest.raises(UnsupportedFeatureError, match="a RooMomentMorph \\('mm'\\)"):
        Builder().node(made_of("RooMomentMorph", "mm"))


def workspace_record(**members: Any) -> Streamed:
    empty = {"_list": [], "items": []}
    fields: dict[str, Any] = {"_classes": None, "_allOwnedNodes": empty, "_dataList": empty,
                              "_embeddedDataList": empty, "_snapshots": empty, "_namedSets": {},
                              "_genObjects": empty}  # fmt: skip
    fields.update(members)
    return made_of("RooWorkspace", "w", **fields)


def test_a_workspace_keeps_its_embedded_data_and_wrapped_generic_objects() -> None:
    """A dataset of a density's, and a ``RooTObjWrap``'s object, which has no ``ReplaceWS``."""
    bins = made_of("RooUniformBinning", "", _xlo=0.0, _xhi=2.0, _nbins=2)
    hist = made_of("RooDataHist", "h", _vars={"_list": [real("x", 0.5, bins)]}, _wgt=[1.0, 3.0],
                   _sumw2=[])  # fmt: skip
    wrapped = made_of("RooTObjWrap", "", _list={"items": [made_of("RooConstVar", "k", _value=2.0)]})
    made = Builder().node(workspace_record(_embeddedDataList={"items": [hist]},
                                           _genObjects={"items": [wrapped]}))  # fmt: skip
    assert made.embeddedData("h").sumEntries() == 4.0
    assert made.obj("k").getVal() == 2.0


def test_reading_a_workspace_leaves_a_generic_object_with_no_workspace_of_its_own(
        monkeypatch: Any) -> None:  # fmt: skip
    """Only a generic object that points at its workspace - a ``ModelConfig`` - is re-pointed."""
    wrapped = made_of("RooConstVar", "k", _value=2.0)
    monkeypatch.setattr(Reader, "object", lambda self, cls: workspace_record(
        _genObjects={"items": [wrapped]}))  # fmt: skip
    assert workspace.read_workspace(Buffer(b""), {}).obj("k").getVal() == 2.0


def test_a_workspace_carrying_code_for_classes_the_engine_lacks_is_refused() -> None:
    """Its code repository's classes the engine has are allowed; others are named."""
    ok = made_of("", files=[("RooGaussian", "", "", "")], relations=[("RooGaussian", "", "")])
    workspace._check_code(ok)
    code = made_of("", files=[("MyPdf", "", "", "")], relations=[("MyOther", "", "")])
    with pytest.raises(UnsupportedFeatureError, match="C\\+\\+ code of MyOther, MyPdf"):
        Builder().node(workspace_record(_classes=code))


def test_the_readers_by_hand_are_installed_where_the_reader_looks() -> None:
    custom.install()
    assert data.__all__ == [] and variables.__all__ == [] and "RooGaussian" in models.SIMPLE
