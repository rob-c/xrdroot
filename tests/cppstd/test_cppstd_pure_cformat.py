"""``sscanf``: the fields of a text read as C reads them, and the count it gives back."""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.cppstd import cformat, cstdio
from xrdroot.cppstd.buffers import CharBuffer, text_into
from xrdroot.cppstd.cells import AttrRef, Cell


class RefEntry:
    """A ``RefEntry``-like object whose members a scan writes through ``&d.field``."""

    def __init__(self) -> None:
        self.ps1ref, self.ps1err = 0, 0


def test_the_stress_graphics_reference_line_is_scanned_as_its_own_reader_reads_it() -> None:
    """The line ``ReadRefFile`` reads: ``%s`` into a ``char name[100]`` and the ten numbers of
    the entry, two of them through ``&d.field``."""
    line = ("          tline       529        50     12827        50     23848      3700"
            "      3937       100       569        60\n")
    entry, numbers = RefEntry(), [Cell(0) for _ in range(8)]
    name = CharBuffer(100)
    stored = cstdio.sscanf(line, "%s %d %d %d %d %d %d %d %d %d %d", name,
                           AttrRef(entry, "ps1ref"), AttrRef(entry, "ps1err"),
                           *numbers)  # fmt: skip
    assert stored == 11
    assert (str(name), entry.ps1ref, entry.ps1err) == ("tline", 529, 50)
    assert [cell.value for cell in numbers] == [12827, 50, 23848, 3700, 3937, 100, 569, 60]


def test_the_count_is_what_c_counts_and_eof_is_what_it_gives_at_the_end() -> None:
    """``sscanf``'s answers for an empty text, a text that ends, and one that does not match."""
    assert cstdio.sscanf("", "%d", Cell(0)) == cformat.EOF
    assert cstdio.sscanf("   ", "%d", Cell(0)) == cformat.EOF
    assert cstdio.sscanf("", "%c", Cell(0)) == cformat.EOF
    assert cstdio.sscanf("x", "%d", Cell(0)) == 0
    assert cstdio.sscanf("12abc", "%d%d", Cell(0), Cell(0)) == 1
    assert cstdio.sscanf("nomatch", "%d") == 0


def test_a_whole_number_reaches_its_cell_and_a_real_its_own() -> None:
    count, total, value = Cell(0, "int"), Cell(0, "long"), Cell(0.0, "double")
    assert cstdio.sscanf("7 12 3.25", "%d %ld %lf", count, total, value) == 3
    assert (count.value, total.value, value.value) == (7, 12, 3.25)


def test_a_character_in_the_format_matches_the_text_or_ends_the_scan() -> None:
    """C's count: 0 where the format's character is not the text's, ``EOF`` where the text
    ran out with nothing stored - the input failure C gives before the first conversion."""
    assert cstdio.sscanf("x=3", "x=%d", Cell(0)) == 1
    assert cstdio.sscanf("x=3", "y=%d", Cell(0)) == 0
    assert cstdio.sscanf("x", "x") == 0
    assert cstdio.sscanf("x", "xy") == cformat.EOF
    assert cstdio.sscanf("50%", "%d%%", Cell(0)) == 1
    assert cstdio.sscanf("%", "%%") == 0


def test_a_whitespace_in_the_format_matches_any_amount_of_the_texts_own() -> None:
    padded = Cell(0)
    assert cstdio.sscanf(" \t\n 42  ", "  %d  ", padded) == 1 and padded.value == 42


def test_a_width_holds_a_conversion_to_that_many_characters() -> None:
    short, real, word = Cell(0), Cell(0.0), CharBuffer(8)
    assert cstdio.sscanf("12345", "%2d", short) == 1 and short.value == 12
    assert cstdio.sscanf("1.5e3rest", "%3lf", real) == 1 and real.value == 1.5
    assert cstdio.sscanf("words", "%2s", word) == 1 and str(word) == "wo"


def test_a_field_left_unread_stays_where_the_next_conversion_starts() -> None:
    number, word = Cell(0), CharBuffer(8)
    assert cstdio.sscanf("12 and words", "%d %s", number, word) == 2
    assert (number.value, str(word)) == (12, "and")


def test_an_integer_is_read_in_the_base_its_conversion_names() -> None:
    hexadecimal, octal, unsigned, signed = Cell(0), Cell(0), Cell(0), Cell(0)
    assert cstdio.sscanf("0x1f 017 +9 -3", "%x %o %u %d", hexadecimal, octal, unsigned,
                         signed) == 4  # fmt: skip
    assert (hexadecimal.value, octal.value, unsigned.value, signed.value) == (31, 15, 9, -3)


def test_percent_i_reads_its_base_from_the_field_itself() -> None:
    """``%i``'s three bases, and ``"09"`` read as C reads it: the ``0`` of octal, then a ``9``
    left unread where the next conversion starts."""
    hexadecimal, octal, decimal, zero, nine = (Cell(0) for _ in range(5))
    assert cstdio.sscanf("0x1f 012 12", "%i %i %i", hexadecimal, octal, decimal) == 3
    assert (hexadecimal.value, octal.value, decimal.value) == (31, 10, 12)
    assert cstdio.sscanf("09", "%i%i", zero, nine) == 2
    assert (zero.value, nine.value) == (0, 9)


def test_a_real_number_is_read_in_each_shape_c_gives_it() -> None:
    values = [Cell(0.0) for _ in range(5)]
    assert cstdio.sscanf("1.5 .5 1e5 -inf 1.", "%lf %lf %lf %lf %lf", *values) == 5
    assert [value.value for value in values] == [1.5, 0.5, 1e5, float("-inf"), 1.0]
    assert cstdio.sscanf("nan", "%g", Cell(0.0)) == 1


def test_a_suppressed_conversion_reads_a_field_and_keeps_nothing() -> None:
    word = CharBuffer(8)
    assert cstdio.sscanf("  hello world  ", "%*s %s", word) == 1 and str(word) == "world"
    assert cstdio.sscanf("7 8", "%*d") == 0


def test_a_character_is_stored_as_the_code_a_char_variable_holds() -> None:
    character, characters = Cell(0, "char"), CharBuffer(4)
    assert cstdio.sscanf("xy", "%c", character) == 1 and character.value == ord("x")
    assert cstdio.sscanf("xy", "%2c", characters) == 1
    assert [int(characters[0]), int(characters[1]), int(characters[2])] == [ord("x"), ord("y"), 0]


def test_a_scan_into_an_array_element_writes_the_element_a_pointer_points_at() -> None:
    numbers = np.zeros(4, dtype=np.int32)
    assert cstdio.sscanf("5 6", "%d %d", numbers[1:], numbers[3:]) == 2
    assert list(numbers[1:4]) == [5, 0, 6]


def test_the_text_scanned_may_be_the_character_buffer_a_fgets_filled_or_bytes() -> None:
    line, number = CharBuffer(8), Cell(0)
    text_into(line, "42")
    assert cstdio.sscanf(line, "%d", number) == 1 and number.value == 42
    assert cstdio.sscanf(line[0:], "%d", number) == 1
    assert cstdio.sscanf(b"43", "%d", number) == 1 and number.value == 43


def test_the_text_scanned_may_be_bytes_read_as_latin_1() -> None:
    number, word = Cell(0), CharBuffer(4)
    assert cstdio.sscanf(b"42 ok", "%d %s", number, word) == 2
    assert (number.value, str(word)) == (42, "ok")


def test_text_scanned_into_a_char_array_the_translation_holds_as_text_is_refused() -> None:
    with pytest.raises(TypeError) as refused:
        cstdio.sscanf("tline", "%s", "")
    assert "%s" in str(refused.value) and "new char[n]" in str(refused.value)


def test_a_format_whose_conversions_outnumber_its_targets_is_refused() -> None:
    with pytest.raises(TypeError) as refused:
        cstdio.sscanf("1 2", "%d %d", Cell(0))
    assert "more conversions than the values" in str(refused.value)


def test_a_conversion_c_does_not_read_is_refused_by_name() -> None:
    with pytest.raises(ValueError) as refused:
        cstdio.sscanf("1", "%n", Cell(0))
    assert "%n" in str(refused.value) and "%d" in str(refused.value)
