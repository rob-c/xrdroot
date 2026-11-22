"""``TBufferJSON``: ROOT's JSON of an array of numbers, written and read back.

The text asserted here is ROOT 6.40.04's own, as ``roottest``'s
``root/io/json/ArrayCompress.ref`` records it and as the oracle prints it for
the float table below, so a difference is this library's and not the test's.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.cppstd.bitset import Bitset
from xrdroot.cppstd.buffers import CharBuffer
from xrdroot.cppstd.cells import Cell
from xrdroot.cppstd.containers import std
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pycxx.runtime.namespace import DECLARED
from xrdroot.pyroot.core.jsonbuffer import _float_text, _scalar

#: What ROOT's oracle prints for each value, as ``TArrayD`` and as ``TArrayF`` hold it.
FLOATS = [
    (0.1, "0.1", "0.1"),
    (1 / 3, "0.333333333333333", "0.3333333"),
    (1e-7, "1e-7", "1e-7"),
    (1e20, "100000000000000000000", "1e20"),
    (1e25, "1e25", "1e25"),
    (1e15, "1000000000000000", "999999986991104"),
    (1234567.5, "1234567.5", "1234568"),
    (0.0001234, "1.234e-4", "1.234e-4"),
    (0.00375, "0.00375", "0.00375"),
    (0.000375, "3.75e-4", "3.75e-4"),
    (99.99, "99.99", "99.99"),
    (1.1e-10, "1.1e-10", "1.1e-10"),
    (6.02e23, "601999999999999995805696", "6.02e23"),
    (1.7976931348623157e308, "1.79769313486232e308", "2e308"),
    (5e-324, "4.94065645841247e-324", "0"),
    (12345678901234567890.0, "12345678901234567168", "1.234568e19"),
    (999999999999999.9, "1e15", "999999986991104"),
    (-1.5, "-1.5", "-1.5"),
    (-0.0, "-0", "-0"),
]


def _float32(value: float) -> float:
    """``value`` as a ``Float_t`` holds it: the nearest float32, or an infinity for too much."""
    if abs(value) > 3.4028235e38:
        return math.copysign(math.inf, value)
    return float(np.float32(value))


def _json(obj, compact=0):
    return str(ROOT.TBufferJSON.ToJSON(obj, compact))


def test_root_writes_a_short_array_as_a_plain_json_array_of_its_values():
    """Below six values ROOT writes the numbers themselves, however it is asked to compress."""
    assert _json(ROOT.TArrayI(5, [0, 0, 3, 4, 5]), 23) == "[0,0,3,4,5]"
    assert _json(ROOT.TArrayI(5, [0, 0, 3, 4, 5]), 0) == "[0, 0, 3, 4, 5]"


def _filled(count=100, values=()):
    """A ``TArrayI`` of ``count`` zeros with ``values`` - (index, value) pairs - put in it."""
    arr = ROOT.TArrayI(count)
    for index, value in values:
        arr[index] = value
    return arr


def test_a_compressed_array_names_its_type_and_the_one_run_it_holds():
    """``ArrayCompress.ref``: ROOT's own JSON of the same array, less its leading zero."""
    arr = _filled(values=((n, n) for n in range(100)))
    assert _json(arr, 3) == "[" + ",".join(str(n) for n in range(100)) + "]"
    assert _json(arr, 23) == (
        '{"$arr":"Int32","len":100,"p":1,"v":[' + ",".join(str(n) for n in range(1, 100)) + "]}")


def test_an_array_of_zeros_writes_its_type_and_its_length_alone():
    assert _json(_filled(), 3) == "[" + ",".join("0" for _ in range(100)) + "]"
    assert _json(_filled(), 23) == '{"$arr":"Int32","len":100}'


def test_each_run_of_a_repeated_value_is_written_once_with_how_often_it_repeats():
    values = [(n + 17, 7) for n in range(10)] + [(n + 56, 11) for n in range(10)]
    assert _json(_filled(values=values), 23) == (
        '{"$arr":"Int32","len":100,"p":17,"v":7,"n":10,"p1":56,"v1":11,"n1":10}')


def test_a_run_of_different_values_is_written_as_their_array_without_a_position():
    values = [(n + 10, n) for n in range(10)] + [(n + 20, 55) for n in range(10)]
    values += [(n + 30, n) for n in range(10)]
    assert _json(_filled(values=values), 23) == (
        '{"$arr":"Int32","len":100,"p":11,"v":[1,2,3,4,5,6,7,8,9],"v1":55,"n1":10,'
        '"p2":31,"v2":[1,2,3,4,5,6,7,8,9]}'
    )


def test_a_run_between_two_runs_of_repeats_needs_no_position_of_its_own():
    values = [(n + 10, 11) for n in range(10)] + [(n + 20, n + 7) for n in range(10)]
    values += [(n + 30, 22) for n in range(10)]
    assert _json(_filled(values=values), 23) == (
        '{"$arr":"Int32","len":100,"p":10,"v":11,"n":10,"v1":[7,8,9,10,11,12,13,14,15,16],'
        '"v2":22,"n2":10}'
    )


def test_the_first_compression_keeps_the_zeros_at_the_ends_out():
    """``kZeroSuppression`` writes one run from the first value to the last, and no runs more."""
    arr = ROOT.TArrayI(10, [0, 0, 0, 4, 0, 6, 0, 0, 0, 0])
    assert _json(arr, 10) == '{"$arr":"Int32", "len":10, "p":3, "v":[4, 0, 6]}'
    assert _json(arr, 13) == '{"$arr":"Int32","len":10,"p":3,"v":[4,0,6]}'


def test_a_gap_of_more_than_nine_zeros_ends_the_run_it_falls_in():
    """A run of ten zeros is what the array itself already holds, so ROOT writes no ``v`` for it."""
    arr = ROOT.TArrayI(20)
    arr[0] = 5
    arr[11] = 7
    assert _json(arr, 23) == '{"$arr":"Int32","len":20,"v":5,"p1":11,"v1":7}'


def test_each_array_class_names_the_element_type_root_gives_it():
    for made, typname in [(ROOT.TArrayS, "Int16"), (ROOT.TArrayI, "Int32"),
                          (ROOT.TArrayL, "Int64"), (ROOT.TArrayF, "Float32"),
                          (ROOT.TArrayD, "Float64")]:
        assert _json(made(6, [1, 2, 3, 4, 5, 6]), 23) == (
            f'{{"$arr":"{typname}","len":6,"v":[1,2,3,4,5,6]}}')
    assert _json(ROOT.TArrayC(6, [65, 66, 67, 68, 69, 70]), 23) == '"ABCDEF"'


def test_a_char_array_is_the_string_of_its_characters_at_every_level():
    """ROOT writes a ``char`` array as one JSON string, which is what a ``TArrayC`` is."""
    arr = ROOT.TArrayC(3, [ord("a"), ord("b"), ord("c")])
    assert _json(arr, 0) == '"abc"'
    assert _json(arr, 3) == '"abc"'
    assert list(ROOT.TBufferJSON.FromJSON["TArrayC"]('"abc"')) == [97, 98, 99]


def test_a_float_is_written_in_the_text_root_writes_it_in():
    """``TBufferText::ConvertDouble``: whole digits below 1e25, ``%.14e`` shortened above."""
    for value, as_double, as_float in FLOATS:
        assert _float_text(value, "Float64", False) == as_double
        assert _float_text(_float32(value), "Float32", False) == as_float


def test_an_infinity_and_a_nan_are_root_s_stand_ins_unless_the_flag_asks_for_strings():
    array = ROOT.TArrayD(8, [0, 0, 0, float("inf"), 0, float("nan"), 0, 0])
    assert _json(array, 1000) == '[0, 0, 0, "inf", 0, "nan", 0, 0]'
    assert _json(array, 0) == "[0, 0, 0, 2e308, 0, null, 0, 0]"
    assert _json(ROOT.TArrayD(8, [0, 0, 0, float("-inf"), 0, 0, 0, 0]), 0) == (
        "[0, 0, 0, -2e308, 0, 0, 0, 0]")
    tiny = ROOT.TArrayF(8, [0, 0, 0, float("inf"), 0, float("nan"), 0, 0])
    assert _json(tiny, 1000) == '[0, 0, 0, "inff", 0, "nanf", 0, 0]'
    assert _json(tiny, 1023) == '{"$arr":"Float32","len":8,"p":3,"v":["inff",0,"nanf"]}'


def test_the_low_digit_spaces_out_and_the_tens_digit_compresses():
    """``kMapAsObject`` and above leave only their remainder, as ROOT's ``SetCompact`` does."""
    arr = ROOT.TArrayI(6, [1, 2, 3, 4, 5, 6])
    assert _json(arr, 0) == "[1, 2, 3, 4, 5, 6]"
    assert _json(arr, 3) == "[1,2,3,4,5,6]"
    assert _json(arr, 8) == "[1,2,3,4,5,6]"
    buffer = ROOT.TBufferJSON()
    buffer.SetCompact(-7)
    assert str(buffer.StoreObject(arr)) == "[1, 2, 3, 4, 5, 6]"
    buffer.SetCompact(23)
    assert str(buffer.StoreObject(arr)) == '{"$arr":"Int32","len":6,"v":[1,2,3,4,5,6]}'


@pytest.mark.parametrize("level", [10, 13, 20, 23, 1013, 1023])
def test_an_array_read_back_is_the_array_the_json_names(level):
    arr = ROOT.TArrayD(7, [0, 0, 2.5, 2.5, 2.5, 0, 0])
    read = ROOT.TBufferJSON.ConvertFromJSON(_json(arr, level))
    assert isinstance(read, ROOT.TArrayD)
    assert list(read) == list(arr)


def test_from_json_fills_the_pointer_it_is_handed_and_the_array_it_is_handed():
    text = '{"$arr":"Int32","len":6,"p":2,"v":[3,4,5]}'
    cell = type("Cell", (), {"value": None})()
    assert list(ROOT.TBufferJSON.FromJSON(cell, text)) == [0, 0, 3, 4, 5, 0]
    assert isinstance(cell.value, ROOT.TArrayI)
    held = ROOT.TArrayI(2)
    assert ROOT.TBufferJSON.FromJSON(held, text) is held
    assert list(held) == [0, 0, 3, 4, 5, 0]
    other = ROOT.TArrayD(0)
    assert list(ROOT.TBufferJSON.FromJSON(other, text)) == [0, 0, 3, 4, 5, 0]
    assert isinstance(ROOT.TBufferJSON.FromJSON(ROOT.TArrayI(0), text), ROOT.TArrayI)


#: The numbers ``roottest``'s ``runArrayCompress`` puts in a ``TArrayI(100)``, in the order it
#: makes them: the ramp it starts from, and then - each after an ``arr.Reset(0)`` - what it
#: writes into an array of zeros.
def _array_compress_states() -> list[np.ndarray[Any, Any]]:
    ramp = np.arange(100)
    many = np.zeros(100, dtype=int)
    many[17:27] = 7
    many[56:66] = 11
    inside = np.zeros(100, dtype=int)
    inside[10:20] = np.arange(10)
    inside[20:30] = 55
    inside[30:40] = np.arange(10)
    outside = np.zeros(100, dtype=int)
    outside[10:20] = 11
    outside[20:30] = np.arange(10) + 7
    outside[30:40] = 22
    return [ramp, np.zeros(100, dtype=int), many, inside, outside]


#: What ROOT writes for each state above at ``kSameSuppression``, as ``ArrayCompress.ref``
#: has it: the run of equal values ROOT leaves out, and the stretches it writes on either
#: side of a gap of zeros.
COMPRESSED = [
    '{"$arr":"Int32","len":100,"p":1,"v":[' + ",".join(str(n) for n in range(1, 100)) + "]}",
    '{"$arr":"Int32","len":100}',
    '{"$arr":"Int32","len":100,"p":17,"v":7,"n":10,"p1":56,"v1":11,"n1":10}',
    '{"$arr":"Int32","len":100,"p":11,"v":[1,2,3,4,5,6,7,8,9],"v1":55,"n1":10,'
    '"p2":31,"v2":[1,2,3,4,5,6,7,8,9]}',
    '{"$arr":"Int32","len":100,"p":10,"v":11,"n":10,"v1":[7,8,9,10,11,12,13,14,15,16],'
    '"v2":22,"n2":10}',
]


def test_the_compressed_arrays_of_root_s_own_array_test_are_written_and_read_back():
    """``runArrayCompress``: each of its five arrays at ``kNoSpaces`` and at
    ``kSameSuppression``, and the compressed text read back as the array it holds.

    The macro itself is refused, because ``FromJSON(arr2, json)`` writes through a pointer
    the translation has no cell for; what it would print once that cell is there is this.
    """
    for state, compressed in zip(_array_compress_states(), COMPRESSED, strict=True):
        arr = ROOT.TArrayI(100)
        for index, value in enumerate(state):
            arr[index] = int(value)
        assert _json(arr, 3) == "[" + ",".join(str(int(each)) for each in state) + "]"
        assert _json(arr, 23) == compressed
        read = ROOT.TBufferJSON.ConvertFromJSON(compressed)
        assert isinstance(read, ROOT.TArrayI)
        assert list(read) == list(state)


def test_a_cell_typed_by_the_pointer_reads_a_json_array_that_names_no_class():
    """``TBufferJSON::FromJSON(T *&obj, json)`` takes the class from the pointer it fills, which
    is how a ``TArrayI`` written at ``kNoSpaces`` - a JSON array that names no class - is read
    back. The cell a translated pointer arrives in carries that type, and this is the reader's
    half of it: with the type, both of ROOT's spellings of the same array are read.
    """
    cell = Cell(None, "TArrayI")
    read = ROOT.TBufferJSON.FromJSON(cell, "[0,1,2]")
    assert isinstance(read, ROOT.TArrayI)
    assert cell.value is read
    assert list(read) == [0, 1, 2]
    compressed = Cell(None, "TArrayI")
    assert list(ROOT.TBufferJSON.FromJSON(compressed, '{"$arr":"Int32","len":3,"v":[1,2]}')) \
        == [1, 2, 0]


def test_a_run_longer_than_the_array_says_nothing_about_the_array_s_rest():
    """ROOT reads the array's own ``len``; a ``v`` of more values writes past it and is cut."""
    read = ROOT.TBufferJSON.ConvertFromJSON
    assert list(read('{"$arr":"Int16","len":2,"v":[1,2,3,4]}')) == [1, 2]
    assert list(read('{"$arr":"Int16","len":4,"v":7,"n":2}')) == [7, 7, 0, 0]
    assert list(read('{"$arr":"Int16","len":3,"p":1}')) == [0, 0, 0]


def test_an_object_root_does_not_write_as_an_array_is_refused_by_its_name():
    with pytest.raises(UnsupportedFeatureError) as refused:
        _json(ROOT.TH1D("h", "a histogram", 10, 0, 1))
    assert "does not yet write the JSON of a TH1D" in str(refused.value)
    assert "TArrayI" in str(refused.value)


def test_one_data_member_is_written_as_the_value_root_writes_for_it():
    """``ToJSON(obj, level, member)`` writes that member's JSON and nothing round it."""
    marker = ROOT.TPolyMarker3D(2)
    marker.SetPoint(0, 1.5, -2.25, 3)
    marker.SetPoint(1, 0.1, 1e10, 1e-10)
    assert str(ROOT.TBufferJSON.ToJSON(marker, 0, "fN")) == "2"
    assert str(ROOT.TBufferJSON.ToJSON(marker, 0, "fOption")) == '""'
    assert str(ROOT.TBufferJSON.ToJSON(marker, 3, "fP")) == "[1.5,-2.25,3,0.1,10000000000,1e-10]"


def test_a_member_a_class_does_not_have_is_refused_by_its_name():
    with pytest.raises(UnsupportedFeatureError) as refused:
        ROOT.TBufferJSON.ToJSON(ROOT.TArrayI(6), 0, "fN")
    assert "'fN'" in str(refused.value)
    with pytest.raises(UnsupportedFeatureError) as unknown:
        ROOT.TBufferJSON.ToJSON(ROOT.TObjString("a"), 0, "fNope")
    assert "'fNope'" in str(unknown.value) and "fString" in str(unknown.value)


def test_base_64_is_the_bytes_of_the_values_little_endian_from_the_first_there_is():
    arr = ROOT.TArrayI(6, [0, 0, 300, -2, 0, 0])
    assert _json(arr, 30) == '{"$arr":"Int32", "len":6, "o":8, "b":"LAEAAP7///8="}'
    assert _json(arr, 33) == '{"$arr":"Int32","len":6,"o":8,"b":"LAEAAP7///8="}'
    assert list(ROOT.TBufferJSON.ConvertFromJSON(_json(arr, 33))) == [0, 0, 300, -2, 0, 0]
    whole = ROOT.TArrayI(6, [1, 2, 3, 4, 5, 6])
    assert _json(whole, 33) == '{"$arr":"Int32","len":6,"b":"AQAAAAIAAAADAAAABAAAAAUAAAAGAAAA"}'
    zeros = ROOT.TArrayI(6)
    assert _json(zeros, 33) == '{"$arr":"Int32","len":6,"b":""}'
    assert _json(ROOT.TArrayI(0), 33) == "[]"


def test_a_member_of_declared_lengths_is_the_nested_arrays_root_writes_for_it():
    """``int fTest2[2][2]`` is an array of arrays and ``char fChar[2][3][4]`` an array of the
    strings its rows are, as ``BasicTypes.ref`` has them for ``TJsonEx2``."""
    made = _declared_class("TNested", (("fTest2", "int", "", (2, 2)),
                                      ("fChar", "char", "", (2, 3, 4))))()  # fmt: skip
    made.fTest2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    made.fChar = "012312342345123423453456"
    assert _json(made) == (
        '{\n  "_typename" : "TNested",\n  "fTest2" : [[1, 2], [3, 4]],\n'
        '  "fChar" : [["0123", "1234", "2345"], ["1234", "2345", "3456"]]\n}'
    )


def test_an_object_is_read_back_from_its_typename_and_written_the_same_again(monkeypatch):
    """What ``roottest`` calls ``store/read/store``: the object ``ConvertFromJSONAny`` makes
    from a JSON object writes that JSON out again, member for member."""
    made = _reading_class(monkeypatch, "TNested", (
        ("fInt", "int", "", ()),
        ("fTest2", "int", "", (2, 2)),
        ("fChar", "char", "", (2, 3, 4)),
    ))()  # fmt: skip
    made.fInt, made.fTest2 = 123456, np.array([[1, 2], [3, 4]], dtype=np.int32)
    made.fChar = "012312342345123423453456"
    text = _json(made)
    read = ROOT.TBufferJSON.ConvertFromJSON(text)
    assert type(read) is type(made)
    assert (read.fInt, read.fTest2.tolist()) == (123456, [[1, 2], [3, 4]])
    assert _json(read) == text


def test_convert_from_json_any_hands_back_the_class_the_json_was_read_as(monkeypatch):
    """``ConvertFromJSONAny(json, &cl)`` fills the class cell, which is what a macro writes
    the object back through, and returns the object itself."""
    text = '{"_typename":"TReadBack","fInt":7}'
    _reading_class(monkeypatch, "TReadBack", (("fInt", "int", "", ()),))
    cell = type("Cell", (), {"value": None})()
    read = ROOT.TBufferJSON.ConvertFromJSONAny(text, cell)
    assert read.fInt == 7
    assert cell.value.GetName() == "TReadBack"
    assert str(ROOT.TBufferJSON.ConvertToJSON(read, cell.value, 3)) == text


def test_an_object_of_a_class_nothing_declares_is_refused_by_its_name():
    with pytest.raises(UnsupportedFeatureError) as refused:
        ROOT.TBufferJSON.ConvertFromJSON('{"_typename":"TUnknown","fName":"a"}')
    assert "does not know the class TUnknown" in str(refused.value)
    assert "FromJSON<T>(json)" in str(refused.value)
    assert "TBox" in str(refused.value)  # a class its own table describes is read back


#: What ROOT 6.40.04's oracle prints for ``TBufferJSON::ToJSON(new TBox(11,22,33,44))`` at
#: ``kNoCompress``: ``TObject``'s two members, then ``TAttLine``'s three, ``TAttFill``'s two
#: and the corners, which is the order ``roottest``'s ``RootClasses.ref`` writes them in.
BOX_JSON = """{
  "_typename" : "TBox",
  "fUniqueID" : 0,
  "fBits" : 0,
  "fLineColor" : 1,
  "fLineStyle" : 1,
  "fLineWidth" : 1,
  "fFillColor" : 19,
  "fFillStyle" : 1001,
  "fX1" : 11,
  "fY1" : 22,
  "fX2" : 33,
  "fY2" : 44
}"""


def test_a_box_is_written_as_the_members_its_streamer_writes_in_their_order():
    assert _json(ROOT.TBox(11, 22, 33, 44)) == BOX_JSON


def test_a_class_the_table_describes_reads_back_as_the_namespace_s_class():
    """``TBufferJSON::ConvertFromJSONAny`` of ROOT's own ``TBox`` JSON: the table says what
    a box's members are, so the JSON is read into the ``TBox`` the namespace has, and what
    is written from that object again is the same JSON - ``RootClasses``' store/read/store."""
    cell = type("Cell", (), {"value": None})()
    read = ROOT.TBufferJSON.ConvertFromJSONAny(BOX_JSON, cell)
    assert isinstance(read, ROOT.TBox)
    assert cell.value.GetName() == "TBox"
    assert _json(read) == BOX_JSON


def test_a_member_read_back_is_held_where_the_writer_reads_it():
    """A drawn object keeps its members in a dict by name, not in attributes: the corners
    come back into that dict, which is what ``IsInside`` and the writer both read."""
    read = ROOT.TBufferJSON.ConvertFromJSONAny(BOX_JSON)
    assert read.members["fX1"] == 11
    assert read.IsInside(20, 30)


def test_a_to_object_read_back_keeps_its_members_where_a_core_class_holds_them():
    """``TObject``'s ``fUniqueID`` is the object's ``_unique_id``, so a JSON of one is read
    back into the attribute the writer reads rather than an attribute named ``fUniqueID``."""
    read = ROOT.TBufferJSON.ConvertFromJSONAny(
        '{"_typename":"TObject","fUniqueID":7,"fBits":0}')
    assert isinstance(read, ROOT.TObject)
    assert read.GetUniqueID() == 7
    assert _json(read) == '{\n  "_typename" : "TObject",\n  "fUniqueID" : 7,\n  "fBits" : 0\n}'


def test_a_name_the_table_describes_that_is_no_class_is_left_to_the_refusal(monkeypatch):
    """The namespace holds an object by that name rather than the class - a ``gROOT``
    object, say - and the JSON is then refused as any class this cannot read back."""
    monkeypatch.setattr(ROOT, "TObjString", "not a class")
    with pytest.raises(UnsupportedFeatureError) as refused:
        ROOT.TBufferJSON.ConvertFromJSONAny('{"_typename":"TObjString","fString":"a"}')
    assert "does not know the class TObjString" in str(refused.value)


def test_a_bare_json_array_is_read_when_the_cell_knows_the_pointer_s_c_type():
    """``TBufferJSON::FromJSON(T *&obj, ...)``: C++ deduces ``T``, so the JSON need not name it."""
    cell = Cell(None, "TArrayI")
    read = ROOT.TBufferJSON.FromJSON(cell, "[0, 0, 3, 4, 5]")
    assert list(read) == [0, 0, 3, 4, 5]
    assert list(cell.value) == [0, 0, 3, 4, 5]
    with pytest.raises(UnsupportedFeatureError) as refused:
        ROOT.TBufferJSON.FromJSON(Cell(None, "double"), "[1]")
    assert "bare JSON array" in str(refused.value)


def test_a_json_this_library_cannot_read_back_says_what_to_write_instead():
    for text, why in [("[1, 2, 3]", "bare JSON array"),
                      ("not json", "TBufferJSON::FromJSON reads"),
                      ('{"$arr":"Int128","len":2,"v":[1]}', "Int128"),
                      ("7", "bare JSON array")]:
        with pytest.raises(UnsupportedFeatureError) as refused:
            ROOT.TBufferJSON.ConvertFromJSON(text)
        assert why in str(refused.value)


def test_a_buffer_is_opened_for_writing_since_reading_is_from_json_s_own_call():
    with pytest.raises(UnsupportedFeatureError) as refused:
        ROOT.TBufferJSON(0)
    assert "opened for reading" in str(refused.value)
    assert str(ROOT.TBufferJSON(1).StoreObject(ROOT.TArrayI(0))) == "[]"


def test_the_json_of_an_object_is_the_json_of_a_write():
    """``ToJSON`` builds the buffer ROOT builds, so ``ConvertToJSON`` and it agree."""
    arr = ROOT.TArrayI(100, list(range(100)))
    assert ROOT.TBufferJSON.ConvertToJSON(arr, None, 23) == ROOT.TBufferJSON.ToJSON(arr, 23)
    assert ROOT.TBufferJSON.ConvertToJSON(arr, 23) == ROOT.TBufferJSON.ToJSON(arr)


def _pair_map():
    """The map ``runMap.C`` makes: ten keys named ``fieldN``, each holding ``7N``."""
    data = std.map["std::string", "int"]()
    for n in range(3):
        data["field" + str(n)] = n * 7
    return data


def test_an_object_is_the_members_its_streamer_writes_in_the_order_it_writes_them():
    """``PolyMarker.ref``: ``fLastPoint`` is not there, and ``fName`` comes after ``fOption``.

    ROOT's own text for this marker says ``"fBits" : 1`` and
    ``"fName" : "TPolyMarker3D"``: ``TPolyMarker3D``'s constructor calls
    ``SetBit(kCanDelete)`` and names the marker after its class, and neither has this
    library's object, so both come from :data:`CLASS_BITS` and :data:`DEFAULTS`.
    """
    marker = ROOT.TPolyMarker3D(3)
    for n in range(3):
        marker.SetPoint(n, n * 2, n * 3, n * 4)
    assert _json(marker) == """{
  "_typename" : "TPolyMarker3D",
  "fUniqueID" : 0,
  "fBits" : 1,
  "fMarkerColor" : 1,
  "fMarkerStyle" : 1,
  "fMarkerSize" : 1,
  "fN" : 3,
  "fP" : [0, 0, 0, 2, 3, 4, 4, 6, 8],
  "fOption" : "",
  "fName" : "TPolyMarker3D"
}"""
    assert _json(marker, 3) == (
        '{"_typename":"TPolyMarker3D","fUniqueID":0,"fBits":1,"fMarkerColor":1,'
        '"fMarkerStyle":1,"fMarkerSize":1,"fN":3,"fP":[0,0,0,2,3,4,4,6,8],"fOption":"",'
        '"fName":"TPolyMarker3D"}')


def test_a_name_the_object_is_given_is_the_name_that_is_written():
    """The default is what the constructor sets; ``SetName`` still decides what goes out."""
    marker = ROOT.TPolyMarker3D(1)
    marker.SetName("other")
    assert _json(marker, 3).endswith('"fOption":"","fName":"other"}')


def test_the_heap_bits_of_a_member_the_object_holds_are_added_to_its_class_s():
    """A drawn object keeps ``fBits`` itself, with the heap's two bits, which ROOT masks."""
    marker = ROOT.TPolyMarker3D(1)
    marker.members["fBits"] = 0x03000003
    assert '"fBits":3' in _json(marker, 3)


def test_a_polyline_writes_its_line_attributes_and_no_name():
    """Its streamer writes ``fN``, ``fP`` and ``fOption``, and ``TPolyLine3D`` has no name."""
    line = ROOT.TPolyLine3D(2)
    line.SetPoint(0, 1, 2, 3)
    line.SetPoint(1, 4, 5, 6)
    assert _json(line, 2) == (
        '{"_typename" : "TPolyLine3D","fUniqueID" : 0,"fBits" : 0,"fLineColor" : 1,'
        '"fLineStyle" : 1,"fLineWidth" : 1,"fN" : 2,"fP" : [1, 2, 3, 4, 5, 6],'
        '"fOption" : ""}')


def test_a_member_an_object_does_not_hold_is_zero_rather_than_an_error():
    """A drawn object keeps only what it was given; a member it never set is its type's zero."""
    assert _json(ROOT.TObjString("hi"), 3) == (
        '{"_typename":"TObjString","fUniqueID":0,"fBits":0,"fString":"hi"}')
    assert _json(ROOT.TObject(), 3) == '{"_typename":"TObject","fUniqueID":0,"fBits":0}'
    assert _json(ROOT.TNamed("a", "b"), 1) == (
        '{\n"_typename" : "TNamed",\n"fUniqueID" : 0,\n"fBits" : 0,\n"fName" : "a",\n'
        '"fTitle" : "b"\n}')


def test_a_map_is_a_pair_per_entry_unless_the_level_asks_for_an_object():
    """``Map.ref``: ``$pair`` objects at ``kNoCompress``, and a JSON object at 5 and 108."""
    data = _pair_map()
    assert _json(data, 0) == (
        '[{"$pair" : "pair<string,int>", "first" : "field0", "second" : 0}, '
        '{"$pair" : "pair<string,int>", "first" : "field1", "second" : 7}, '
        '{"$pair" : "pair<string,int>", "first" : "field2", "second" : 14}]')
    assert _json(data, 5) == (
        '{"_typename": "map<string,int>", "field0": 0, "field1": 7, "field2": 14}')
    assert _json(data, 3) == (
        '[{"$pair":"pair<string,int>","first":"field0","second":0},'
        '{"$pair":"pair<string,int>","first":"field1","second":7},'
        '{"$pair":"pair<string,int>","first":"field2","second":14}]')
    assert _json(data, 108) == '{"field0":0,"field1":7,"field2":14}'
    assert _json(data, 103) == (
        '[{"$pair":1,"first":"field0","second":0},{"$pair":1,"first":"field1","second":7},'
        '{"$pair":1,"first":"field2","second":14}]')


@pytest.mark.parametrize("level", [0, 3, 5, 8, 103, 105, 108])
def test_a_map_read_back_is_the_map_root_wrote(level):
    data = _pair_map()
    read = ROOT.TBufferJSON.FromJSON["std::map<std::string,int>"](_json(data, level))
    assert read.size() == data.size()
    assert read.at("field2") == 14


def test_a_value_json_has_no_number_for_is_written_as_root_s_own_stand_in():
    assert _float_text(float("inf"), "Float64", True) == '"inf"'
    assert _float_text(float("-inf"), "Float32", True) == '"-inff"'
    assert _float_text(float("nan"), "Float64", True) == '"nan"'
    assert _float_text(float("nan"), "Float32", False) == "null"
    with pytest.raises(UnsupportedFeatureError) as refused:
        _json({"a": 1})
    assert "does not yet write the JSON of a dict" in str(refused.value)


def test_a_string_is_written_with_the_escapes_javascript_reads_and_root_writes():
    named = ROOT.TNamed('a"b\\c/d\te\nf', "x")
    assert _json(named, 3) == (
        '{"_typename":"TNamed","fUniqueID":0,"fBits":0,"fName":"a\\"b\\\\c\\/d\\te\\nf",'
        '"fTitle":"x"}')


def test_a_number_a_flag_or_nothing_at_all_is_written_as_json_has_it():
    """The leaves a member or a ``$pair`` value can be, written as their own JSON."""
    assert _json(True) == "true"
    assert _json(False) == "false"
    assert _json(None) == "null"
    assert _scalar(True, "Bool", False) == "true"
    assert _scalar(False, "Bool", False) == "false"


def test_a_number_written_on_its_own_is_the_text_root_gives_that_type():
    assert _json(7) == "7"
    assert _json(np.int32(7)) == "7"
    assert _json(0.5) == "0.5"
    assert _json(np.float32(0.5)) == "0.5"
    assert _json("a/b") == '"a\\/b"'


def test_a_pointer_read_into_nothing_says_to_read_what_from_json_returns():
    """A translated ``T *p`` that is ``nullptr`` has no cell for ``FromJSON`` to fill."""
    with pytest.raises(UnsupportedFeatureError) as refused:
        ROOT.TBufferJSON.FromJSON(None, '{"$arr":"Int32","len":2,"v":[1]}')
    assert "nowhere to write" in str(refused.value)


def test_the_numbers_a_drawn_object_keeps_are_written_as_the_array_they_are():
    """A member array is a NumPy one here, and goes out as the array of its own type."""
    assert _json(np.array([1.5, 2.5, 3.0], dtype=np.float32)) == "[1.5, 2.5, 3]"
    marker = ROOT.TPolyMarker3D(2)
    marker.SetPoint(0, 0.5, 1.5, 2.5)
    marker.SetPoint(1, 3.5, 4.5, 5.5)
    assert str(ROOT.TBufferJSON.ToJSON(marker, 23)) == (
        '{"_typename":"TPolyMarker3D","fUniqueID":0,"fBits":1,"fMarkerColor":1,'
        '"fMarkerStyle":1,"fMarkerSize":1,"fN":2,"fP":{"$arr":"Float32","len":6,'
        '"v":[0.5,1.5,2.5,3.5,4.5,5.5]},"fOption":"","fName":"TPolyMarker3D"}'
    )


def _declared_class(name, members, bases=(), base=None):
    """A class the way the translation declares one: ``_cxx_layout_`` holding its name, the
    names of its bases, and a ``(name, type, qualifiers, dims)`` per member."""
    made = type(name, (base or object,), {"_cxx_layout_": (name, bases, members)})
    for member in members:
        setattr(made, member[0], None)
    return made


def _reading_class(monkeypatch, name, members):
    """A declared class put into the interpreter's declaration table, which is where a reader
    looks up the ``_typename`` of the JSON it is handed - and taken out of it again after."""
    made = _declared_class(name, members)
    monkeypatch.setitem(DECLARED, name, made)
    return made
    return made


#: A class as ``test_classes.h`` declares it: scalars and strings of its own, on top of a base
#: whose members its streamer writes first.
BASE = _declared_class("TBase", (("fBool", "bool", "", ()), ("fChar", "char", "", ()),
                                ("fShort", "short", "", ()), ("fInt", "int", "", ()),
                                ("fLong", "long", "", ()), ("fFloat", "float", "", ()),
                                ("fDouble", "double", "", ())), )  # fmt: skip
OWN = _declared_class("TOwn", (("fStr1", "char", "", (100,)), ("fStr2", "char*", "", ()),
                              ("fDummy2", "int", "", ()), ("fStr3", "TString", "", ()),
                              ("fPtr", "std::string*", "", ())),
                      bases=("TBase",), base=BASE)  # fmt: skip


def test_a_class_a_macro_declared_is_written_from_its_own_declaration():
    """ROOT writes the members a class declares, base's first, and each as the type it is
    declared with: ``String.ref``'s ``TJsonEx4``, whose ``char[100]`` is one JSON string up to
    its first NUL and whose ``char`` is the code point of one character."""
    made = OWN()
    made.fBool, made.fChar, made.fShort, made.fInt = True, 67, 123, 123456
    made.fLong, made.fFloat, made.fDouble = 7654321, 1.2, 3.8
    made.fStr1, made.fStr2, made.fDummy2, made.fStr3 = "Value of string 1\0\0", "", 1234567, ""
    assert _json(made) == (
        '{\n  "_typename" : "TOwn",\n  "fBool" : true,\n  "fChar" : 67,\n'
        '  "fShort" : 123,\n  "fInt" : 123456,\n  "fLong" : 7654321,\n'
        '  "fFloat" : 1.2,\n  "fDouble" : 3.8,\n  "fStr1" : "Value of string 1",\n'
        '  "fStr2" : "",\n  "fDummy2" : 1234567,\n  "fStr3" : "",\n  "fPtr" : null\n}'
    )


def test_a_member_of_a_declared_class_is_written_on_its_own_as_root_writes_it():
    """``ToJSON(obj, level, member)`` writes the member by name, text as text."""
    made = OWN()
    made.fStr3 = "--- normal string value ---"
    assert str(ROOT.TBufferJSON.ToJSON(made, 0, "fStr3")) == '"--- normal string value ---"'
    with pytest.raises(UnsupportedFeatureError) as refused:
        ROOT.TBufferJSON.ToJSON(made, 0, "fMissing")
    assert "has no member 'fMissing'" in str(refused.value)


def test_a_char_pointer_with_no_string_at_it_is_null_where_root_writes_one_on_its_own():
    """``String.ref``: ``TJsonEx4``'s ``const char* fStr2`` is a null pointer, and ROOT
    writes ``null`` for it as a member of its own - the pointer itself - while the whole
    object has ``""`` there, the text it streams a null ``char*`` as. A null ``char*`` is
    the empty text once translated, so what tells them apart is that the member is written
    from the pointer rather than as a member of the object."""
    made = OWN()
    made.fStr1, made.fStr2 = "a char array\0", ""
    assert str(ROOT.TBufferJSON.ToJSON(made, 0, "fStr2")) == "null"
    assert str(ROOT.TBufferJSON.ToJSON(made, 0, "fPtr")) == "null"
    made.fStr2 = "a string after all"
    assert str(ROOT.TBufferJSON.ToJSON(made, 0, "fStr2")) == '"a string after all"'
    made.fStr2 = ""
    assert '"fStr2" : ""' in _json(made)


def test_a_member_of_a_type_no_writer_takes_refuses_the_object_by_name():
    """A pair, which ROOT writes as a class of its own, is not written as a number: the
    object is refused by the member's own name rather than written with it left out."""
    kinds = _declared_class("TKinds", (("fPair", "std::pair<int, int>", "", ()),))
    with pytest.raises(UnsupportedFeatureError) as refused:
        ROOT.TBufferJSON.ToJSON(kinds())
    assert "TKinds::fPair, which is declared as a std::pair<int, int>" in str(refused.value)


def _vector(*values):
    """A ``std::vector<double>`` holding ``values``, as a translated member arrives."""
    made = std.vector("double")()
    for value in values:
        made.push_back(value)
    return made


def test_an_empty_container_is_written_as_the_empty_json_root_writes_for_it():
    """``STL0.ref``: every container ``TJsonEx7`` declares, written as the ``[]`` a class's
    constructor leaves it - and a map whose declaration says ``/ JSON_object`` as one JSON
    object that names itself and holds no key."""
    kinds = _declared_class("TEmpty", (
        ("fVect", "std::vector<double>", "", ()),
        ("fVectPtr", "std::vector<double>*", "", ()),
        ("fList", "std::list<double>", "", ()),
        ("fSet", "std::set<double>", "", ()),
        ("fMap", "std::map<int, double>", "", ()),
        ("fMapObj", "std::map<std::string, int>", "/ JSON_object", ()),
    ))()
    for name, value in (("fVect", _vector()), ("fList", std.list("double")()),
                        ("fSet", std.set("double")()),
                        ("fMap", std.map("int", "double")()),
                        ("fMapObj", std.map("std::string", "int")())):
        setattr(kinds, name, value)
    kinds.fVectPtr = None
    assert _json(kinds) == (
        '{\n  "_typename" : "TEmpty",\n  "fVect" : [],\n  "fVectPtr" : null,\n'
        '  "fList" : [],\n  "fSet" : [],\n  "fMap" : [],\n'
        '  "fMapObj" : {"_typename": "map<string,int>"}\n}'
    )


def test_a_multimap_is_written_as_one_pair_object_per_entry():
    """``std::multimap``'s values are ``std::pair``s rather than the entries a map holds, and
    ``STL.ref`` writes each the way it writes a map's: a ``$pair`` naming the types it
    holds, in key order and with a key repeated as often as it was inserted."""
    kinds = _declared_class("TMulti", (("fMultimap", "std::multimap<int, double>", "", ()),))()
    kinds.fMultimap = std.multimap("int", "double")()
    kinds.fMultimap.insert((0, 0.0))
    kinds.fMultimap.insert((0, 1000.0))
    assert _json(kinds) == (
        '{\n  "_typename" : "TMulti",\n  "fMultimap" : ['
        '{"$pair" : "pair<int,double>", "first" : 0, "second" : 0}, '
        '{"$pair" : "pair<int,double>", "first" : 0, "second" : 1000}]\n}'
    )


def test_a_pair_written_on_its_own_is_the_pair_object_a_map_holds():
    """A ``std::pair`` reaches the writer as itself when it is a container's element rather
    than a map's entry, so it is written as the one JSON object ``_map`` writes one as."""
    made = std.pair("int", "double")(7, 2.5)
    assert str(ROOT.TBufferJSON.ToJSON(made)) == (
        '{"$pair" : "pair<int,double>", "first" : 7, "second" : 2.5}'
    )


def test_a_container_is_written_as_the_values_it_holds():
    """A sequence is one JSON array of its values and a map a ``$pair`` per entry, as
    ``STL.ref`` has them for every container ``TJsonEx7`` fills."""
    kinds = _declared_class("THeld", (("fVect", "std::vector<double>", "", ()),
                                      ("fMap", "std::map<int, double>", "", ())))()  # fmt: skip
    kinds.fVect = _vector(3.0, 6.0)
    kinds.fMap = std.map("int", "double")()
    kinds.fMap[1] = 5.0
    assert _json(kinds) == (
        '{\n  "_typename" : "THeld",\n  "fVect" : [3, 6],\n'
        '  "fMap" : [{"$pair" : "pair<int,double>", "first" : 1, "second" : 5}]\n}'
    )


def test_a_string_array_is_written_as_its_strings_and_a_pointer_array_as_nulls():
    """``std::string[N]`` is one JSON array of the strings - a constructor's unset entries as
    the empty string - and ``std::string *[N]`` an array of the ``null`` pointers, which is
    ``STL0.ref``'s ``fStrArr`` and ``fStrPtrArr``."""
    kinds = _declared_class("TStrings", (("fStrArr", "std::string", "", (3,)),
                                         ("fStrPtrArr", "std::string*", "", (3,))))()  # fmt: skip
    kinds.fStrArr = [None, "b", None]
    kinds.fStrPtrArr = ["c", None, None]
    assert _json(kinds) == (
        '{\n  "_typename" : "TStrings",\n  "fStrArr" : ["", "b", ""],\n'
        '  "fStrPtrArr" : ["c", null, null]\n}'
    )


def test_a_bitset_is_written_as_a_boolean_per_bit_in_the_order_the_bits_are_numbered():
    """``bitset<16>`` is what ``STL0.ref`` writes as sixteen ``false`` and ``STL.ref`` as
    ``k % 2`` set over them."""
    kinds = _declared_class("TBits", (("fBitsSet16", "std::bitset<4>", "", ()),))()
    kinds.fBitsSet16 = Bitset(4)
    kinds.fBitsSet16[1] = True
    assert _json(kinds) == (
        '{\n  "_typename" : "TBits",\n  "fBitsSet16" : [false, true, false, false]\n}'
    )


def test_an_array_of_containers_is_written_as_an_array_of_them():
    """``vector<double> fVectDoubleArr[3]``: one JSON array per slot, which is
    ``STL0.ref``'s ``[[], [], []]``, and ``null`` where the slot is a pointer to none."""
    kinds = _declared_class("TArrayOf", (("fArr", "std::vector<double>", "", (3,)),
                                         ("fPtrArr", "std::vector<double>*", "", (3,))))()
    kinds.fArr = [_vector(), _vector(3.0), _vector()]
    kinds.fPtrArr = [None, None, None]
    assert _json(kinds) == (
        '{\n  "_typename" : "TArrayOf",\n  "fArr" : [[], [3], []],\n'
        '  "fPtrArr" : [null, null, null]\n}'
    )


def test_a_container_a_class_inherits_is_written_under_the_name_root_gives_it():
    """``TJsonEx8 : std::vector<int>`` holds no member of its own: the object is the vector,
    and ROOT writes its values first, as ``fVector`` (``STL0.ref``)."""
    base = type(std.vector("int")())
    inherited = _declared_class("TJsonEx8", (("fInt", "int", "", ()),),
                                bases=("std::vector",), base=base)  # fmt: skip
    made = inherited()
    made.push_back(7)
    assert _json(made) == (
        '{\n  "_typename" : "TJsonEx8",\n  "fVector" : [7],\n  "fInt" : 0\n}'
    )


def test_an_empty_container_reads_back_as_the_one_its_constructor_made(monkeypatch):
    """What ``STL0`` calls ``store/read/store``: a container's JSON says the values it
    holds, so an empty one leaves the container the class's constructor made - and a
    bitset's bits are read back, in the order ROOT numbers them."""
    containers = _reading_class(monkeypatch, "TContainers", (
        ("fVect", "std::vector<double>", "", ()),
        ("fBitsSet16", "std::bitset<4>", "", ()),
    ))  # fmt: skip

    def __init__(self) -> None:
        self.fVect, self.fBitsSet16 = _vector(), Bitset(4)

    containers.__init__ = __init__
    made = containers()
    made.fBitsSet16 = Bitset(4)
    made.fBitsSet16[2] = True
    text = _json(made)
    read = ROOT.TBufferJSON.ConvertFromJSON(text)
    assert list(read.fVect) == []
    assert [read.fBitsSet16[at] for at in range(4)] == [False, False, True, False]
    assert _json(read) == text


def test_a_member_that_holds_values_is_filled_with_them_when_it_is_read_back(monkeypatch):
    """``STL1``'s ``store/read/store``: the JSON says which values a container holds, so they
    are put into the container the class's constructor made - the one the writer then reads
    them back from - rather than into an object whose containers stay as empty as they were."""
    containers = _reading_class(monkeypatch, "TFilled", (
        ("fVect", "std::vector<double>", "", ()),
        ("fList", "std::list<double>", "", ()),
        ("fSet", "std::set<double>", "", ()),
        ("fMap", "std::map<int, double>", "", ()),
        ("fMulti", "std::multimap<int, double>", "", ()),
    ))  # fmt: skip

    def __init__(self) -> None:
        self.fVect, self.fList = _vector(), std.list("double")()
        self.fSet, self.fMap = std.set("double")(), std.map("int", "double")()
        self.fMulti = std.multimap("int", "double")()

    containers.__init__ = __init__
    text = (
        '{"_typename":"TFilled","fVect":[3,6],"fList":[8],"fSet":[2,3],'
        '"fMap":[{"$pair":"pair<int,double>","first":1,"second":5}],'
        '"fMulti":[{"$pair":"pair<int,double>","first":0,"second":0},'
        '{"$pair":"pair<int,double>","first":0,"second":1000}]}'
    )
    read = ROOT.TBufferJSON.ConvertFromJSON(text)
    assert (list(read.fVect), list(read.fList), list(read.fSet)) == ([3.0, 6.0], [8.0], [2.0, 3.0])
    assert read.fMap.items() == [(1, 5.0)]
    assert read.fMulti.items() == [(0, 0.0), (0, 1000.0)]
    assert _json(read, 3) == text


def test_a_container_of_objects_holds_the_objects_the_json_names(monkeypatch):
    """A ``vector<TJsonEx1>`` of pointers: each element is the object its ``_typename`` names,
    and a ``$ref`` is the very object read before it, so the two links ``STL1``'s
    ``fVectEx1Ptr`` has to one object are still one object after the read."""
    _reading_class(monkeypatch, "TElem", (("fBool", "bool", "", ()),))
    members = (("fVect", "std::vector<TElem*>", "", ()),)
    made = _reading_class(monkeypatch, "TVect", members)
    made.__init__ = lambda self: setattr(self, "fVect", std.vector("TElem*")())
    text = (
        '{"_typename":"TVect","fVect":[{"_typename":"TElem","fBool":true},'
        '{"$ref":1},null]}'
    )
    read = ROOT.TBufferJSON.ConvertFromJSON(text)
    assert read.fVect[0] is read.fVect[1]
    assert (read.fVect[0].fBool, read.fVect[2]) == (True, None)
    assert _json(read, 3) == text


def test_a_container_of_containers_is_filled_at_every_depth(monkeypatch):
    """A ``vector<vector<int>>`` is one JSON array per element, and a ``set<TJsonEx1>`` a map
    holds is built the same way - ``STL1``'s ``TJsonEx13`` has both."""
    nested = _reading_class(monkeypatch, "TNested", (
        ("vvvv", "std::vector<std::vector<int> >", "", ()),
        ("map1", "std::map<int, std::set<int> >", "", ()),
    ))  # fmt: skip

    def __init__(self) -> None:
        self.vvvv = std.vector("std::vector<int>")()
        self.map1 = std.map("int", "std::set<int>")()

    nested.__init__ = __init__
    text = (
        '{"_typename":"TNested","vvvv":[[0,1],[2]],'
        '"map1":[{"$pair":"pair<int,set<int> >","first":1,"second":[7]}]}'
    )
    read = ROOT.TBufferJSON.ConvertFromJSON(text)
    assert [list(row) for row in read.vvvv] == [[0, 1], [2]]
    assert [list(held) for _, held in read.map1.items()] == [[7]]
    assert _json(read, 3) == text


def test_a_pointer_to_a_container_is_made_of_what_the_declaration_spells(monkeypatch):
    """``STL1``'s ``fVectPtrDouble``: the constructor leaves the pointer empty and the JSON
    holds values, so the container is made of the type the declaration gives it."""
    made = _reading_class(monkeypatch, "TPointed", (
        ("fPtr", "std::vector<double>*", "", ()),
        ("fRows", "std::vector<double>*", "", (2,)),
    ))  # fmt: skip
    made.__init__ = lambda self: setattr(self, "fPtr", None) or setattr(
        self, "fRows", [None, None])
    text = '{"_typename":"TPointed","fPtr":[3],"fRows":[[1],[2]]}'
    read = ROOT.TBufferJSON.ConvertFromJSON(text)
    assert (list(read.fPtr), [list(row) for row in read.fRows]) == ([3.0], [[1.0], [2.0]])
    assert _json(read, 3) == text


def test_a_string_array_reads_back_as_its_strings_and_its_pointer_as_null(monkeypatch):
    """A ``std::string[N]`` member is read back as the strings the JSON holds, and a
    ``std::string *`` with ``null`` as the pointer with nothing there - which is what makes
    ``STL0``'s ``store/read/store`` the same JSON twice."""
    _reading_class(monkeypatch, "TTextArrays", (("fStrArr", "std::string", "", (2,)),
                                                ("fStrPtrArr", "std::string*", "", (2,)),
                                                ("fStrPtr", "std::string*", "", ())))  # fmt: skip
    read = ROOT.TBufferJSON.ConvertFromJSON(
        '{"_typename":"TTextArrays","fStrArr":["a",""],'
        '"fStrPtrArr":["b",null],"fStrPtr":null}'
    )
    assert (read.fStrArr, read.fStrPtrArr, read.fStrPtr) == (["a", ""], ["b", None], None)


def test_a_member_called_f_vector_that_is_no_container_is_written_as_its_number():
    """ROOT names a container a class inherits ``fVector``, and a class of its own may hold a
    member of that name: only a container member is the inherited one."""
    numbers = _declared_class("TNumber", (("fVector", "int", "", ()),))()  # fmt: skip
    assert _json(numbers) == '{\n  "_typename" : "TNumber",\n  "fVector" : 0\n}'


def test_a_bitset_read_from_json_that_is_no_array_of_bits_is_refused(monkeypatch):
    bits = (("fBitsSet16", "std::bitset<4>", "", ()),)
    _reading_class(monkeypatch, "TBitsOnly", bits)
    with pytest.raises(UnsupportedFeatureError) as refused:
        ROOT.TBufferJSON.ConvertFromJSON('{"_typename":"TBitsOnly","fBitsSet16":3}')
    assert "TBitsOnly::fBitsSet16 holds" in str(refused.value)
    assert "one boolean per bit" in str(refused.value)


def test_a_pointer_whose_comment_declares_its_length_is_the_array_root_writes(monkeypatch):
    """``bool *fBool; // [fSize]`` is a dynamic array, which ROOT's streamer writes as the
    values it points at - ``BasicTypes.ref``'s ``TJsonEx3`` - and not as an address."""
    dynamic = _reading_class(monkeypatch, "TDynamic", (
        ("fSize", "int", "", ()),
        ("fBool", "bool*", "// [fSize]", ()),
        ("fChar", "char*", "// [fSize]", ()),
        ("fInt", "int*", "// [fSize]", ()),
    ))  # fmt: skip
    monkeypatch.setitem(DECLARED, "TDynamic", dynamic)
    made = dynamic()
    made.fSize, made.fBool, made.fChar = 7, [False] * 7, "1234567"
    made.fInt = [0, 5, 10, 15, 20, 25, 30]
    text = _json(made)
    assert text == (
        '{\n  "_typename" : "TDynamic",\n  "fSize" : 7,\n'
        '  "fBool" : [false, false, false, false, false, false, false],\n'
        '  "fChar" : "1234567",\n'
        '  "fInt" : [0, 5, 10, 15, 20, 25, 30]\n}'
    )
    read = ROOT.TBufferJSON.ConvertFromJSON(text)
    assert (read.fChar, read.fInt.tolist()) == ("1234567", [0, 5, 10, 15, 20, 25, 30])
    assert _json(read) == text


def test_a_pointer_with_no_length_in_its_comment_is_still_no_array(monkeypatch):
    """The pointer is an array only where its comment says how long: one whose comment says
    nothing is a pointer into storage the streamer does not describe, and is refused."""
    plain = _declared_class("TPlain", (("fPtr", "bool*", "", ()),))
    monkeypatch.setitem(DECLARED, "TPlain", plain)
    with pytest.raises(UnsupportedFeatureError) as refused:
        ROOT.TBufferJSON.ToJSON(plain())
    assert "TPlain::fPtr, which is declared as a bool*" in str(refused.value)


def test_a_class_whose_declaration_is_not_a_layout_is_no_declared_class():
    """A ``_cxx_layout_`` that is not a name, its bases and its members says nothing about the
    class, so the object is refused as a class with no table rather than half written."""
    odd = type("TOdd", (object,), {"_cxx_layout_": ("TOdd",)})
    with pytest.raises(UnsupportedFeatureError) as refused:
        ROOT.TBufferJSON.ToJSON(odd())
    assert "does not yet write the JSON of a TOdd" in str(refused.value)


def test_a_base_the_object_does_not_inherit_writes_no_members():
    """A layout that names a base the class does not have contributes nothing, which is what
    keeps a member of a base ROOT's streamer never writes out of the JSON."""
    lost = _declared_class("TLost", (("fInt", "int", "", ()),), bases=("TGone",))
    assert _json(lost()) == '{\n  "_typename" : "TLost",\n  "fInt" : 0\n}'


def test_a_text_member_and_a_buffer_of_characters_are_read_back_as_they_are_held(monkeypatch):
    """A ``std::string`` member reads back as text and a ``char[N]`` one as the characters
    the buffer holds, which the writer reads as its code points - ``char[3]`` here."""
    made = _reading_class(monkeypatch, "TText", (("fName", "std::string", "", ()),
                                                  ("fChar", "char", "", (3,))))()  # fmt: skip
    made.fName, made.fChar = "abc", np.array([49, 50, 51], dtype=np.int8).view(CharBuffer)
    text = _json(made)
    read = ROOT.TBufferJSON.ConvertFromJSON(text)
    assert (read.fName, read.fChar) == ("abc", "123")
    assert _json(read) == text
    #: A member the JSON leaves out keeps what the class's constructor gave it.
    assert ROOT.TBufferJSON.ConvertFromJSON('{"_typename":"TText","fName":"abc"}').fChar is None


def test_a_char_array_is_written_as_one_string_per_row_of_its_own(monkeypatch):
    """ROOT writes a ``char[N][M]`` as a string per row, and each row ends at its own first
    NUL: ``test_classes.h``'s ``TJsonEx2::fChar`` is ``[["0123", "1234", "2345"], ...]``,
    and a row of two characters is that row rather than the start of the one after it."""
    made = _reading_class(monkeypatch, "TCharRows", (("fChar", "char", "", (2, 3, 4)),))()
    made.fChar = np.zeros((2, 3, 4), dtype=np.int8)
    for at, row in enumerate(("0123", "1234", "2", "3456", "4567", "5678")):
        made.fChar[at // 3][at % 3][: len(row)] = [ord(char) for char in row]
    assert _json(made) == (
        '{\n  "_typename" : "TCharRows",\n'
        '  "fChar" : [["0123", "1234", "2"], ["3456", "4567", "5678"]]\n}'
    )


def test_a_char_array_reads_back_with_each_row_where_his_declaration_puts_it(monkeypatch):
    """What makes a ``char[N][M]`` member's store/read/store one JSON twice: the rows come
    back at the width the declaration gives them, each padded with the NULs that end it,
    rather than as one run of characters the writer would cut into rows again."""
    made = _reading_class(monkeypatch, "TCharBack", (("fChar", "char", "", (2, 3, 4)),))()
    made.fChar = np.zeros((2, 3, 4), dtype=np.int8)
    for at, row in enumerate(("0123", "1234", "2", "3456", "4567", "5678")):
        made.fChar[at // 3][at % 3][: len(row)] = [ord(char) for char in row]
    text = _json(made)
    read = ROOT.TBufferJSON.ConvertFromJSON(text)
    assert read.fChar.shape == (2, 3, 4)
    assert read.fChar[0][2].tolist() == [50, 0, 0, 0]
    assert _json(read) == text


def test_a_char_array_written_as_its_code_points_reads_back_as_its_characters(monkeypatch):
    """A ``char[N]`` whose JSON is the numbers of its characters - which ROOT writes when the
    member is not text - reads back as the characters they are."""
    _reading_class(monkeypatch, "TCodes", (("fChar", "char", "", (3,)),))
    read = ROOT.TBufferJSON.ConvertFromJSON('{"_typename":"TCodes","fChar":[49,50,51]}')
    assert read.fChar == "123"


def test_an_array_of_a_length_the_declaration_leaves_to_run_time_stays_flat(monkeypatch):
    """``int fDyn[n]`` is a length only the object knows, so the array is read as the flat
    one ROOT writes from it rather than as the nested arrays a fixed one is."""
    made = _reading_class(monkeypatch, "TDyn", (("fDyn", "int", "", (None,)),))()
    made.fDyn = np.array([1, 2, 3], dtype=np.int32)
    text = _json(made)
    read = ROOT.TBufferJSON.ConvertFromJSON(text)
    assert read.fDyn.tolist() == [1, 2, 3]
    assert _json(read) == text


def test_convert_from_json_any_needs_no_class_cell_to_read_an_object(monkeypatch):
    _reading_class(monkeypatch, "TNoCell", (("fInt", "int", "", ()),))
    assert ROOT.TBufferJSON.ConvertFromJSONAny('{"_typename":"TNoCell","fInt":3}').fInt == 3


def test_a_t_string_array_is_written_at_the_depth_the_object_holds_it():
    """``TString fStr[2]`` is one JSON array of its strings, and ``TString *fStr2[3]`` - a
    pointer another member counts, as ``StreamerLoop.ref``'s ``fStr2`` - is an array of one
    array per slot, each the ``null`` pointer where the object holds none."""
    kinds = _declared_class("TStringArrays", (("fStr", "TString", "", (2,)),
                                              ("fStr2", "TString*", "", (2,))))()  # fmt: skip
    kinds.fStr = [ROOT.TString("a"), ROOT.TString("")]
    kinds.fStr2 = [[ROOT.TString("b")], [None]]
    assert _json(kinds) == (
        '{\n  "_typename" : "TStringArrays",\n  "fStr" : ["a", ""],\n'
        '  "fStr2" : [["b"], [null]]\n}'
    )


def test_a_t_string_array_reads_back_with_the_empty_pointer_as_none(monkeypatch):
    """What makes ``StreamerLoop``'s store/read/store one JSON twice: the reader keeps the
    depth the JSON wrote and the ``null`` that stands for a pointer with nothing there."""
    _reading_class(monkeypatch, "TStringBack", (("fStr", "TString", "", (2,)),
                                                ("fStr2", "TString*", "", (2,))))  # fmt: skip
    read = ROOT.TBufferJSON.ConvertFromJSON(
        '{"_typename":"TStringBack","fStr":["a",""],"fStr2":[["b"],[null]]}')
    assert read.fStr == ["a", ""]
    assert read.fStr2 == [["b"], [None]]


def test_a_member_of_a_root_class_is_written_as_one_json_object_of_its_own():
    """ROOT streams a member of a class type, so ``TNamed fName`` is a JSON object inside the
    object, its own members indented a level deeper - ``Objects.ref``'s ``TJsonEx10`` - and a
    pointer that holds nothing is the ``null`` ROOT writes for it."""
    kinds = _declared_class("THolds", (("fName0", "TNamed", "", ()),
                                       ("fObj0", "TObject", "", ()),
                                       ("fNull", "TNamed*", "", ()),
                                       ("fArr", "TNamed*", "[fCnt]", ())))  # fmt: skip
    made = kinds()
    named = ROOT.TNamed("SimpleName", "SimpleTitle")
    also = ROOT.TNamed("SimpleName", "SimpleTitle")
    obj = ROOT.TObject()
    obj.SetUniqueID(10000)
    made.fName0, made.fObj0, made.fNull, made.fArr = named, obj, None, [also, None]
    assert _json(made) == (
        '{\n  "_typename" : "THolds",\n  "fName0" : {\n    "_typename" : "TNamed",\n'
        '    "fUniqueID" : 0,\n    "fBits" : 0,\n    "fName" : "SimpleName",\n'
        '    "fTitle" : "SimpleTitle"\n  },\n  "fObj0" : {\n    "_typename" : "TObject",\n'
        '    "fUniqueID" : 10000,\n    "fBits" : 0\n  },\n  "fNull" : null,\n'
        '  "fArr" : [{\n    "_typename" : "TNamed",\n    "fUniqueID" : 0,\n'
        '    "fBits" : 0,\n    "fName" : "SimpleName",\n    "fTitle" : "SimpleTitle"\n'
        '  }, null]\n}'
    )
    made.fArr = None  # an array of objects its pointer holds none of is one ``null``
    assert _json(made).endswith('  "fArr" : null\n}')


def test_an_array_of_objects_reads_back_as_one_object_per_element(monkeypatch):
    """A member of a class type is read back into an object of the class its ``_typename``
    names, and an array of them into one per element, with the ``null`` a pointer left."""
    _reading_class(monkeypatch, "TBackHolds", (("fArr", "TNamed*", "[fCnt]", ()),
                                                ("fObj", "TNamed", "", ())))  # fmt: skip
    read = ROOT.TBufferJSON.ConvertFromJSON(
        '{"_typename":"TBackHolds","fObj":null,"fArr":[{"_typename":"TNamed",'
        '"fUniqueID":0,"fBits":0,"fName":"a","fTitle":"t"},null]}'
    )
    assert read.fObj is None
    assert read.fArr[0].GetName() == "a"
    assert read.fArr[1] is None


#: What ROOT's oracle prints for a ``TList`` of one ``TNamed`` at ``kNoSpaces``, ``opt`` and
#: all: a link added without an option is a ``null`` in it, since a list keeps its options
#: per link rather than with the object.
LIST_JSON = ('{"_typename":"TList","name":"TList","arr":[{"_typename":"TNamed","fUniqueID":0,'
             '"fBits":0,"fName":"name0","fTitle":"title0"}],"opt":[null]}')


def test_a_collection_is_the_objects_it_holds_under_the_name_it_holds_them_by():
    """``TCollection``'s streamer writes the objects, and a ``TList``'s links' options beside
    them, as ``RootClasses`` and ROOT's own ``TBufferJSONTests`` show them."""
    held = ROOT.TList()
    held.Add(ROOT.TNamed("name0", "title0"))
    assert _json(held, 3) == LIST_JSON
    empty = ROOT.TList()
    assert _json(empty, 3) == '{"_typename":"TList","name":"TList","arr":[],"opt":[]}'
    clones = ROOT.TClonesArray("TNamed", 0)
    assert _json(clones, 3) == '{"_typename":"TClonesArray","name":"TClonesArray","arr":[]}'


def test_a_collection_reads_back_as_the_collection_it_names_with_its_objects():
    """``TBufferJSON::FromJSON<TList>`` of ROOT's own JSON: the objects are made again from
    their own ``_typename``, and what is written from that list is the JSON it came from."""
    read = ROOT.TBufferJSON.FromJSON["TList"](LIST_JSON)
    assert isinstance(read, ROOT.TList)
    assert read.GetName() == "TList"
    assert [each.GetName() for each in read] == ["name0"]
    assert _json(read, 3) == LIST_JSON
    assert _json(ROOT.TBufferJSON.ConvertFromJSON(LIST_JSON), 3) == LIST_JSON


def test_a_collection_whose_links_carry_an_option_is_refused_rather_than_read_short():
    """A collection here keeps no option per link, so reading one back would lose them."""
    text = '{"_typename":"TList","name":"TList","arr":[{"_typename":"TNamed","fUniqueID":0,' \
           '"fBits":0,"fName":"name0","fTitle":"title0"}],"opt":["option0"]}'
    with pytest.raises(UnsupportedFeatureError) as refused:
        ROOT.TBufferJSON.FromJSON["TList"](text)
    assert "whose links carry an option" in str(refused.value)


def test_from_json_of_a_template_reads_into_the_pointer_it_is_handed():
    """``FromJSON<T>(obj, json)``: ROOT has the reader as an overload filling the pointer it
    is given, which is the cell a translated ``T *p`` arrives in."""
    cell = Cell(None, "TNamed")
    read = ROOT.TBufferJSON.FromJSON["TNamed"](
        cell, '{"_typename":"TNamed","fUniqueID":0,"fBits":0,"fName":"a","fTitle":"t"}')
    assert read.GetName() == "a"
    assert cell.value is read
    with pytest.raises(UnsupportedFeatureError) as refused:
        ROOT.TBufferJSON.FromJSON["TNamed"](None, '{"_typename":"TNamed"}')
    assert "arrived as its value rather than as a cell holding it" in str(refused.value)


def test_a_collection_json_without_a_name_reads_back_under_the_one_it_was_made_with():
    """ROOT always writes the name a collection is held under; a JSON that leaves it out
    still names its class, so the collection is made under the name its constructor gives."""
    read = ROOT.TBufferJSON.FromJSON["TList"]('{"_typename":"TList","arr":[]}')
    assert read.GetName() == "TList"
    assert len(read) == 0


def test_a_member_of_a_class_a_macro_declared_is_written_as_its_own_object(monkeypatch):
    """``Objects.ref``: ``TJsonEx5``'s ``TJsonEx1 fObj1`` is the JSON of that object's own
    members, one level in - and a declared class is written this way however it is spelled,
    ``TJsonEx1``, ``TJsonEx1*``, and refused only when nothing can write it whole."""
    inner_kind = _reading_class(monkeypatch, "TInner", (("fInt", "int", "", ()),))
    _reading_class(monkeypatch, "TOuter", (("fObj", "TInner", "", ()),
                                          ("fPtr", "TInner*", "", ()),
                                          ("fPair", "std::pair<int, int>", "", ())))  # fmt: skip
    inner = inner_kind()
    inner.fInt = 7
    outer = DECLARED["TOuter"]()
    outer.fObj, outer.fPtr, outer.fPair = inner, None, None
    member = _reading_class(monkeypatch, "TOuter2", (("fObj", "TInner", "", ()),))()
    member.fObj = inner
    assert _json(member) == (
        '{\n  "_typename" : "TOuter2",\n  "fObj" : {\n    "_typename" : "TInner",\n'
        '    "fInt" : 7\n  }\n}'
    )
    with pytest.raises(UnsupportedFeatureError) as refused:
        _json(outer)
    assert "does not yet write TOuter::fPair" in str(refused.value)


def test_an_array_of_a_declared_class_arrives_nested_and_is_written_as_its_rows(monkeypatch):
    """A translated ``TJsonEx6::fSafePtr33[3][3]`` arrives as lists of lists rather than one
    flat run, and ROOT writes it as the arrays of objects its lengths describe."""
    deep = _reading_class(monkeypatch, "TDeep", (("fInt", "int", "", ()),))
    first, second = deep(), deep()
    first.fInt = second.fInt = 3
    kinds = _reading_class(monkeypatch, "TGrid", (("fPtr", "TDeep*", "", (2, 2)),))()
    kinds.fPtr = [[first, None], [None, second]]
    assert _json(kinds) == (
        '{\n  "_typename" : "TGrid",\n  "fPtr" : [[{\n    "_typename" : "TDeep",\n'
        '    "fInt" : 3\n  }, null], [null, {\n    "_typename" : "TDeep",\n'
        '    "fInt" : 3\n  }]]\n}'
    )


def test_an_object_one_member_already_points_at_is_written_as_that_object_number(monkeypatch):
    """``Objects.ref``: ``TJsonEx5``'s ``fPtr3`` is ``fPtr1``, and ROOT writes it as
    ``{"$ref":3}`` - the third object of the export - rather than a second copy of it.

    The members are pointers because that is what ROOT streams as a reference: two members
    holding the same *value* would be two copies of it in C++, and a copy is written whole."""
    kind = _reading_class(monkeypatch, "TShared", (("fInt", "int", "", ()),))
    members = (("fOne", "TShared*", "", ()), ("fTwo", "TShared*", "", ()),
               ("fOther", "TShared*", "", ()))  # fmt: skip
    kinds = _reading_class(monkeypatch, "THoldsTwice", members)
    shared = kind()
    shared.fInt = 5
    kinds = kinds()
    kinds.fOne, kinds.fTwo, kinds.fOther = shared, shared, kind()
    assert _json(kinds, 3) == (
        '{"_typename":"THoldsTwice","fOne":{"_typename":"TShared","fInt":5},'
        '"fTwo":{"$ref":1},"fOther":{"_typename":"TShared","fInt":0}}'
    )


def test_a_reference_reads_back_as_the_object_it_names_rather_than_a_copy(monkeypatch):
    """What ``store/read/store`` asks of a ``$ref``: the object read from the number is the
    one the member before it already points at, so writing it again writes the same number."""
    _reading_class(monkeypatch, "TShared2", (("fInt", "int", "", ()),))
    _reading_class(monkeypatch, "TBackTwice", (("fOne", "TShared2*", "", ()),
                                               ("fTwo", "TShared2*", "", ())))  # fmt: skip
    text = ('{"_typename":"TBackTwice","fOne":{"_typename":"TShared2","fInt":9},'
            '"fTwo":{"$ref":1}}')
    read = ROOT.TBufferJSON.ConvertFromJSON(text)
    assert read.fOne is read.fTwo
    assert _json(read, 3) == text


def test_a_reference_to_an_object_the_json_never_wrote_is_refused(monkeypatch):
    """A ``$ref`` past the objects the text holds cannot be resolved, and says so."""
    _reading_class(monkeypatch, "TBackTwice3", (("fOne", "TNamed*", "[fCnt]", ()),))
    with pytest.raises(UnsupportedFeatureError) as refused:
        ROOT.TBufferJSON.ConvertFromJSON('{"_typename":"TBackTwice3","fOne":[{"$ref":4}]}')
    assert "reference to object 4 of this JSON" in str(refused.value)


def test_a_nested_string_array_reads_back_as_the_rows_the_json_holds(monkeypatch):
    """A ``std::string[2][3][4]`` is written as nested arrays of strings, so reading it back
    gives those rows again rather than one flat list of them - ``Objects.ref``'s
    ``fStringArr234``."""
    _reading_class(monkeypatch, "TBackStrings", (("fStr", "std::string", "", (2, 2)),))
    text = '{"_typename":"TBackStrings","fStr":[["a","b"],["c","d"]]}'
    read = ROOT.TBufferJSON.ConvertFromJSON(text)
    assert read.fStr == [["a", "b"], ["c", "d"]]


def test_a_member_that_is_an_array_is_written_as_the_text_root_gives_it(monkeypatch):
    """``Objects.ref``: ``ToJSON(&ex6, 0, "fObj1")`` prints ``<not supported>``, because ROOT
    writes one member on its own only as the one value it is - and one that is a value is
    written at the depth the whole object would have had it at."""
    inner = _reading_class(monkeypatch, "TOne", (("fInt", "int", "", ()),))
    kinds = _reading_class(monkeypatch, "TMixed", (("fArr", "TOne", "", (2,)),
                                                   ("fMap", "std::map<int, int>", "", ()),
                                                   ("fObj", "TOne", "", ())))  # fmt: skip
    made = kinds()
    made.fArr = [inner(), inner()]
    made.fMap = std.map("int", "int")()
    made.fObj = inner()
    made.fObj.fInt = 4
    assert str(ROOT.TBufferJSON.ToJSON(made, 0, "fArr")) == "<not supported>"
    assert str(ROOT.TBufferJSON.ToJSON(made, 0, "fMap")) == "<not supported>"
    assert str(ROOT.TBufferJSON.ToJSON(made, 0, "fObj")) == (
        '{\n  "_typename" : "TOne",\n  "fInt" : 4\n}'
    )


def test_a_class_that_inherits_its_container_is_filled_where_the_object_is_one(monkeypatch):
    """``TJsonEx8 : std::vector<int>`` has no member of its own holding the base's values: the
    object is the vector, so that is what the JSON's ``fVector`` is read into - and a ``null``
    there leaves it as the constructor made it."""
    inherited = _declared_class("TInherits", (("fInt", "int", "", ()),),
                                bases=("std::vector",), base=type(std.vector("int")()))  # fmt: skip
    monkeypatch.setitem(DECLARED, "TInherits", inherited)
    read = ROOT.TBufferJSON.ConvertFromJSON(
        '{"_typename":"TInherits","fVector":[46,69],"fInt":12345}'
    )
    assert (list(read), read.fInt) == ([46, 69], 12345)
    empty = ROOT.TBufferJSON.ConvertFromJSON('{"_typename":"TInherits","fVector":null}')
    assert list(empty) == []


def test_a_two_dimensional_array_of_containers_is_filled_row_by_row(monkeypatch):
    """``vector<double> f[2][2]`` is two rows of two containers, which is the depth the
    declaration's lengths give the JSON - and a row that is ``null`` is a pointer with
    nothing there."""
    members = (("fRows", "std::vector<double>", "", (2, 2)),)
    rows = _reading_class(monkeypatch, "TRows", members)
    rows.__init__ = lambda self: setattr(
        self, "fRows", [[_vector(), _vector()], [_vector(), None]])
    read = ROOT.TBufferJSON.ConvertFromJSON(
        '{"_typename":"TRows","fRows":[[[1],[2]],[[3],null]]}'
    )
    assert (list(read.fRows[0][0]), list(read.fRows[1][0])) == ([1.0], [3.0])
    assert read.fRows[1][1] is None


def test_an_array_of_objects_is_read_back_as_the_objects_root_describes(monkeypatch):
    """``TBase fArr[2]``: one object per element, at the depth the declaration gives, so an
    array of them is read back the way an array of text is."""
    _reading_class(monkeypatch, "TElement", (("fBool", "bool", "", ()),))
    holder = _reading_class(monkeypatch, "TArray", (("fArr", "TElement", "", (2,)),))  # fmt: skip
    holder.__init__ = lambda self: setattr(self, "fArr", [None, None])
    text = '{"_typename":"TArray","fArr":[{"_typename":"TElement","fBool":true},null]}'
    read = ROOT.TBufferJSON.ConvertFromJSON(text)
    assert (read.fArr[0].fBool, read.fArr[1]) == (True, None)
    assert _json(read, 3) == text
    assert _json(ROOT.TBufferJSON.ConvertFromJSON('{"_typename":"TArray","fArr":null}'), 3) == (
        '{"_typename":"TArray","fArr":null}'
    )


def test_a_pointer_whose_length_a_comment_gives_is_written_at_the_depth_it_is_held(monkeypatch):
    """``TString *fStr; //[fCnt]`` and ``TNamed **fNm6[3]; //[fCnt]``: the declaration gives
    the length and the comment the one inside it, so the JSON is an array per row - nested as
    deeply as the object holds it - and ``null`` where a pointer has nothing."""
    rows = _reading_class(monkeypatch, "TLengths", (
        ("fStr", "TString**", "", (3,)),
        ("fArr", "TArrayI**", "", (3,)),
        ("fChars", "TArrayC**", "", (3,)),
        ("fObj", "TElement**", "", (3,)),
    ))  # fmt: skip
    for name in ("fStr", "fArr", "fChars", "fObj"):
        setattr(rows, name, None)
    rows._cxx_layout_ = ("TLengths", (), (
        ("fStr", "TString**", "[fCnt]", (3,)),
        ("fArr", "TArrayI**", "[fCnt]", (3,)),
        ("fChars", "TArrayC**", "[fCnt]", (3,)),
        ("fObj", "TElement**", "[fCnt]", (3,)),
    ))
    _reading_class(monkeypatch, "TElement", (("fBool", "bool", "", ()),))
    text = (
        '{"_typename":"TLengths","fStr":[[null],["a"]],'
        '"fArr":[[null],[{"$arr":"Int32","len":2,"v":[0,3]}]],"fChars":[[null],["ab"]],'
        '"fObj":[[null],[{"_typename":"TElement","fBool":true}]]}'
    )
    read = ROOT.TBufferJSON.ConvertFromJSON(text)
    assert (read.fStr, list(read.fArr[1][0])) == ([[None], ["a"]], [0, 3])
    assert list(read.fChars[1][0]) == [ord("a"), ord("b")]
    assert read.fObj[1][0].fBool is True
    with pytest.raises(UnsupportedFeatureError) as refused:
        ROOT.TBufferJSON.ConvertFromJSON(
            '{"_typename":"TLengths","fObj":[[5]]}'
        )
    assert "cannot read this JSON as TElement" in str(refused.value)
    plain = text.replace('{"$arr":"Int32","len":2,"v":[0,3]}', "[0,3]")
    assert _json(ROOT.TBufferJSON.ConvertFromJSON(plain), 3) == plain
    both = ROOT.TBufferJSON.ConvertFromJSON(plain).fArr
    assert list(both[1][0]) == [0, 3]


def test_the_name_a_map_holds_is_the_one_its_declaration_gives():
    """ROOT names the types a ``$pair`` holds as the declaration spells them, which the values
    no longer say - a ``TString`` key is a ``string`` by the time it reaches the writer - and
    a nested template keeps the space ROOT writes between two ``>``."""
    from xrdroot.pyroot.core.jsonbuffer import _pair_name

    assert _pair_name("std::map<TString, int>") == "pair<TString,int>"
    assert _pair_name("std::map<int, std::set<TJsonEx1> >") == "pair<int,set<TJsonEx1> >"
    assert _pair_name("std::vector<int>") is None


#: What ROOT's oracle prints for a ``TMap`` of two pairs, keyed by ``TObjString`` and valued
#: by ``TNamed``, at ``kNoCompress`` and at ``kNoSpaces`` - ``RootClasses.ref``'s own layout
#: for one, which opens the pair on the line the ``arr`` of the map does and keeps a space
#: after each comma inside it.
MAP_INDENTED = "\n".join([
    "{",
    '  "_typename" : "TMap",',
    '  "name" : "TMap",',
    '  "arr" : [{',
    '  "$pair" : "TPair", ',
    '  "first" : {',
    '    "_typename" : "TObjString",',
    '    "fUniqueID" : 0,',
    '    "fBits" : 0,',
    '    "fString" : "Str0"',
    '  }, ',
    '  "second" : {',
    '    "_typename" : "TNamed",',
    '    "fUniqueID" : 0,',
    '    "fBits" : 0,',
    '    "fName" : "Name0",',
    '    "fTitle" : "Title0"',
    '  }',
    '  }, {',
    '  "$pair" : "TPair", ',
    '  "first" : {',
    '    "_typename" : "TObjString",',
    '    "fUniqueID" : 0,',
    '    "fBits" : 0,',
    '    "fString" : "Str1"',
    '  }, ',
    '  "second" : {',
    '    "_typename" : "TNamed",',
    '    "fUniqueID" : 0,',
    '    "fBits" : 0,',
    '    "fName" : "Name1",',
    '    "fTitle" : "Title1"',
    '  }',
    '  }]',
    "}",
])

MAP_NO_SPACES = (
    '{"_typename":"TMap","name":"TMap","arr":['
    '{"$pair":"TPair","first":{"_typename":"TObjString","fUniqueID":0,"fBits":0,'
    '"fString":"Str0"},"second":{"_typename":"TNamed","fUniqueID":0,"fBits":0,'
    '"fName":"Name0","fTitle":"Title0"}},'
    '{"$pair":"TPair","first":{"_typename":"TObjString","fUniqueID":0,"fBits":0,'
    '"fString":"Str1"},"second":{"_typename":"TNamed","fUniqueID":0,"fBits":0,'
    '"fName":"Name1","fTitle":"Title1"}}]}'
)


def _map_of_two_pairs():
    """The ``TMap`` ``RootClasses`` writes, of the two pairs whose JSON is above."""
    made = ROOT.TMap()
    for n in range(2):
        made.Add(ROOT.TObjString(f"Str{n}"), ROOT.TNamed(f"Name{n}", f"Title{n}"))
    return made


def test_a_tmap_is_written_as_the_pairs_it_holds_rather_than_as_a_member_of_it():
    """``TMap`` is written as a collection - the name it holds under and an ``arr`` of pairs -
    and each pair over three lines of its own, indented as the ``arr`` is, which is what
    ``RootClasses.ref``'s ``TMap representation`` shows; the one-line level writes the pair
    on one line. The ``TPair`` name stays even where the level skips every other type."""
    assert _json(_map_of_two_pairs(), 0) == MAP_INDENTED
    assert _json(_map_of_two_pairs(), 3) == MAP_NO_SPACES
    untyped = _json(_map_of_two_pairs(), 100)
    assert untyped.startswith('{\n  "name" : "TMap",\n  "arr" : [{\n  "$pair" : "TPair", ')
    assert "_typename" not in untyped


def test_a_tmap_reads_back_as_the_pairs_the_json_holds_in_the_order_it_holds_them():
    """``ConvertFromJSONAny`` of a ``TMap``'s JSON: the map ROOT made, pair for pair and in
    the order the JSON has them, so writing it again is the text it was read from."""
    read = ROOT.TBufferJSON.ConvertFromJSONAny(MAP_INDENTED)
    assert isinstance(read, ROOT.TMap)
    assert [(pair.Key().GetName(), pair.Value().GetName()) for pair in read] == [
        ("Str0", "Name0"), ("Str1", "Name1")]  # fmt: skip
    assert _json(read, 0) == MAP_INDENTED
    assert _json(ROOT.TBufferJSON.ConvertFromJSONAny(MAP_NO_SPACES), 0) == MAP_INDENTED


def test_a_map_named_nothing_with_no_pairs_is_read_back_empty():
    """A ``TMap``'s JSON with no ``name`` and no ``arr`` is the empty map ROOT writes for one
    with no pairs, which keeps the name its constructor gives it."""
    read = ROOT.TBufferJSON.ConvertFromJSONAny('{"_typename":"TMap"}')
    assert isinstance(read, ROOT.TMap)
    assert (read.GetName(), len(list(read))) == ("TMap", 0)
