"""``TBufferJSON``: ROOT's JSON of an array of numbers, written and read back.

The text asserted here is ROOT 6.40.04's own, as ``roottest``'s
``root/io/json/ArrayCompress.ref`` records it and as the oracle prints it for
the float table below, so a difference is this library's and not the test's.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.cppstd.cells import Cell
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.core.jsonbuffer import _float_text

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
    for made, typname in [(ROOT.TArrayC, "Int8"), (ROOT.TArrayS, "Int16"),
                          (ROOT.TArrayL, "Int64"), (ROOT.TArrayF, "Float32"),
                          (ROOT.TArrayD, "Float64")]:
        assert _json(made(6, [1, 2, 3, 4, 5, 6]), 23) == (
            f'{{"$arr":"{typname}","len":6,"v":[1,2,3,4,5,6]}}')


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


def test_a_run_longer_than_the_array_says_nothing_about_the_array_s_rest():
    """ROOT reads the array's own ``len``; a ``v`` of more values writes past it and is cut."""
    read = ROOT.TBufferJSON.ConvertFromJSON
    assert list(read('{"$arr":"Int16","len":2,"v":[1,2,3,4]}')) == [1, 2]
    assert list(read('{"$arr":"Int16","len":4,"v":7,"n":2}')) == [7, 7, 0, 0]
    assert list(read('{"$arr":"Int16","len":3,"p":1}')) == [0, 0, 0]


def test_an_object_root_does_not_write_as_an_array_is_refused_by_its_name():
    with pytest.raises(UnsupportedFeatureError) as refused:
        _json(ROOT.TH1D("h", "a histogram", 10, 0, 1))
    assert "not yet the JSON of a TH1D" in str(refused.value)
    assert "TArrayI" in str(refused.value)


def test_writing_one_data_member_or_one_base_64_array_is_refused():
    with pytest.raises(UnsupportedFeatureError) as refused:
        ROOT.TBufferJSON.ToJSON(ROOT.TArrayI(6), 0, "fN")
    assert "'fN'" in str(refused.value)
    with pytest.raises(UnsupportedFeatureError) as coded:
        _json(ROOT.TArrayI(6, [1, 2, 3, 4, 5, 6]), 33)
    assert "kBase64" in str(coded.value)
    with pytest.raises(UnsupportedFeatureError) as reading:
        ROOT.TBufferJSON.ConvertFromJSON('{"$arr":"Int32","len":1,"b":"AAAA"}')
    assert "base64" in str(reading.value)


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
