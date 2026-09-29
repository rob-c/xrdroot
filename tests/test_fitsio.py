"""The FITS reader: cards read as CFITSIO reads them, units found, images and tables read."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fitssupport import ascii_table, binary_table, card, image_unit, primary, unit
from xrdroot.errors import FormatError, UnsupportedFeatureError
from xrdroot.fitsio import Record, image, open_unit, read_units, record
from xrdroot.fitsio.cards import header, number, text


def test_a_card_is_its_keyword_its_value_as_written_and_its_comment() -> None:
    assert record(card("ORIGIN", "'ST-DADS '", "Institution")) == Record(
        "ORIGIN", "'ST-DADS '", "Institution")
    assert record(card("NAXIS1", 2064, "")) == Record("NAXIS1", "2064", "")
    assert record(card("OBJ", "'it''s / here'")) == Record("OBJ", "'it''s / here'", "")
    assert record(card("CPLX", "(1.5, 2)", "complex")) == Record("CPLX", "(1.5, 2)", "complex")
    assert record(card("OPEN", "'no end")) == Record("OPEN", "'no end", "")
    assert record("JUNK    =    3 extra".ljust(80)) == Record("JUNK", "3", "extra")
    assert record("CLOSE   =   4/tight".ljust(80)) == Record("CLOSE", "4", "tight")


def test_a_card_without_a_value_is_all_comment_past_column_eight() -> None:
    assert record(card("COMMENT", comment="  said here")) == Record("COMMENT", "", "  said here")
    assert record(card("", comment="         / GROUP")) == Record("", "", "         / GROUP")
    assert record("NOEQUALS  value".ljust(80)) == Record("NOEQUALS", "", "  value")


def test_values_are_read_as_strings_and_numbers() -> None:
    assert (text("'O''Neil  '"), text("PLAIN"), number("1.5D3")) == ("O'Neil", "PLAIN", 1500.0)


def test_a_header_needs_its_end_and_leaves_out_the_blank_cards_before_it() -> None:
    data = unit([card("SIMPLE", True), card("", comment="  a note"), " " * 80, " " * 80])
    records, at = header(data, 0)
    assert ([r.keyword for r in records], at) == (["SIMPLE", ""], 2880)
    with pytest.raises(FormatError, match="no END card"):
        header(b" " * 2880, 0)


def test_units_are_found_by_their_sizes_and_an_image_is_scaled() -> None:
    pixels = np.arange(6, dtype=np.int16).reshape(2, 3)
    extra = [card("BSCALE", 2.0), card("BZERO", 1.0), card("EXTNAME", "'PIX'")]
    data = unit(primary()) + image_unit(pixels, 16, first=False, extra=extra) + b"\0" * 2880
    units = read_units(data)
    assert [(u.index, u.kind, u.sizes) for u in units] == [(0, "IMAGE", []), (1, "IMAGE", [3, 2])]
    assert image(units[1]).tolist() == [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
    assert (units[1].string("EXTNAME"), units[1].value("NONE"), image(units[0]).size) == (
        "PIX", None, 0)


def _write(tmp_path: Any, *units: bytes) -> str:
    path = tmp_path / "t.fits"
    path.write_bytes(unit(primary()) + b"".join(units))
    return str(path)


def _be(values: list[Any], kind: str) -> bytes:
    return np.asarray(values, dtype=kind).tobytes()


def test_a_binary_table_is_read_column_by_column(tmp_path: Any) -> None:
    columns = [("flag", "1L", b"TF"), ("byte", "B", bytes([7, 8])),
               ("short", "1I", _be([-1, 2], ">i2")), ("long", "J", _be([5, 6], ">i4")),
               ("wide", "K", _be([1, 2], ">i8")), ("name", "3A", b"ab\0cd "), ("none", "0A", b""),
               ("real", "E", _be([0.5, 1.5], ">f4")),
               ("pair", "2D", _be([1, 2, 3, 4], ">f8")), ("empty", "0E", b"")]  # fmt: skip
    extra = [card("TSCAL2", 2.0), card("TZERO2", 1.0)]
    opened = open_unit(_write(tmp_path, binary_table(columns, 2, extra=extra)) + "[1]")
    kinds = [(c.name, c.kind, c.typecode) for c in opened.columns]
    assert kinds == [("flag", "REAL NUMBER", 14), ("byte", "REAL NUMBER", 11),
                     ("short", "REAL NUMBER", 21), ("long", "REAL NUMBER", 41), ("wide", "", 81),
                     ("name", "STRING", 16), ("none", "STRING", 16), ("real", "REAL NUMBER", 42),
                     ("pair", "FIXED-LENGTH ARRAY", 82), ("empty", "REAL NUMBER", 42)]  # fmt: skip
    cells = [c.cells for c in opened.columns]
    assert [list(cells[i]) for i in (0, 1, 2, 3, 5, 6, 7, 9)] == [
        [1.0, 0.0], [15.0, 17.0], [-1.0, 2.0], [5.0, 6.0], ["ab", "cd"], ["-", "-"],
        [0.5, 1.5], [0.0, 0.0]]  # fmt: skip
    assert (cells[8].tolist(), cells[4], opened.rows) == ([[1, 2], [3, 4]], None, 2)


def test_variable_length_arrays_are_read_from_the_heap(tmp_path: Any) -> None:
    heap = _be([1, 2, 3], ">i2") + _be([0.25], ">f4")
    columns = [("short", "PI(3)", _be([3, 0, 0, 6], ">i4")),
               ("real", "1PE(1)", _be([1, 6, 0, 10], ">i4")),
               ("big", "QD(1)", _be([0, 0, 0, 0], ">i8")),
               ("bytes", "PB(2)", _be([2, 0, 1, 0], ">i4"))]  # fmt: skip
    opened = open_unit(_write(tmp_path, binary_table(columns, 2, heap)) + "[1]")
    assert [(c.kind, c.typecode) for c in opened.columns] == [
        ("VARIABLE-LENGTH ARRAY", -21), ("VARIABLE-LENGTH ARRAY", -42),
        ("VARIABLE-LENGTH ARRAY", -82), ("VARIABLE-LENGTH ARRAY", -11)]  # fmt: skip
    cells = [[list(cell) for cell in c.cells] for c in opened.columns]
    assert cells == [[[1, 2, 3], []], [[0.25], []], [[], []], [[0, 0], [0]]]


def test_a_binary_table_with_bits_or_an_unknown_form_is_refused(tmp_path: Any) -> None:
    bits = _write(tmp_path, binary_table([("bits", "8X", b"\1\2")], 2))
    with pytest.raises(UnsupportedFeatureError, match="holds bits"):
        open_unit(bits + "[1]")
    odd = _write(tmp_path, binary_table([("odd", "1Z", b"\1\2")], 2))
    with pytest.raises(UnsupportedFeatureError, match="TFORM xrdroot does not know"):
        open_unit(odd + "[1]")


def test_an_ascii_table_is_read_field_by_field(tmp_path: Any) -> None:
    columns = [("NAME", "A4", 1), ("N", "I3", 5), ("F", "F5.2", 8), ("E", "E9.2", 13)]
    rows = ["abc  12  150 1.5D+01", "xy     1.25  2.0E-1"]
    path = _write(tmp_path, ascii_table(columns, rows, "TAB"))
    opened = open_unit(path + "[tab]")
    assert [(c.name, c.kind, c.typecode) for c in opened.columns] == [
        ("NAME", "STRING", 16), ("N", "REAL NUMBER", 41), ("F", "REAL NUMBER", 42),
        ("E", "REAL NUMBER", 42)]  # fmt: skip
    assert [list(c.cells) for c in opened.columns] == [
        ["abc", "xy"], [12.0, 0.0], [1.5, 1.25], [15.0, 0.2]]  # fmt: skip
    assert opened.unit.kind == "ASCII TABLE"


def test_a_unit_is_chosen_by_number_or_name_and_refused_when_missing(tmp_path: Any) -> None:
    path = _write(tmp_path, ascii_table([("A", "A1", 1)], ["x"], "ONE"))
    assert (open_unit(path).unit.index, open_unit(path + "[1]").unit.index) == (0, 1)
    with pytest.raises(OSError, match="no unit 2: it has 2"):
        open_unit(path + "[2]")
    with pytest.raises(OSError, match="no unit named 'TWO'"):
        open_unit(path + "[TWO]")


def test_a_row_filter_keeps_the_rows_it_matches_and_counts_them(tmp_path: Any) -> None:
    columns = [("NAME", "A2", 1), ("X", "I2", 3), ("Y", "I2", 5)]
    rows = ["a  1 4", "b  2 3", "c  3 2", "d  4 1"]
    path = _write(tmp_path, ascii_table(columns, rows)) + "[1]"
    kept = open_unit(path + "[X > 1 && y .lt. 3 || NAME == 'a']")
    assert (kept.columns[0].cells, kept.rows) == (["a", "c", "d"], 3)
    assert kept.unit.value("NAXIS2") == "3"
    assert open_unit(path + "[!(x*2 - -1 >= 7) .and. .not. (y / 2 != 2)]").rows == 1
    assert open_unit(path + "[x + 1 <= 2.5e0]").columns[1].cells.tolist() == [1.0]
    assert open_unit(path + "[1]").rows == 4


def test_a_row_filter_xrdroot_cannot_read_is_refused(tmp_path: Any) -> None:
    path = _write(tmp_path, ascii_table([("X", "I2", 1)], [" 1", " 2"])) + "[1]"
    refusals = {"[x > 1 ?]": "from '\\?' on", "[z > 1]": "names 'z'", "[x >]": "ends where",
                "[x > 1 1]": "past '1'", "[col X]": "column and binning"}
    for spec, reason in refusals.items():
        with pytest.raises(UnsupportedFeatureError, match=reason):
            open_unit(path + spec)
