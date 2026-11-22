"""``std::span``: a window onto elements someone else owns, written through where it points.

C++'s span holds an address and a length; here it holds whatever answers ``x[0]`` for
reading and writing - a :class:`~xrdroot.cppstd.cells.Cell` for the address of one number,
or the sequence itself for many. What matters is that writing through the span writes the
caller's own variable, which is what ``TMath::KNNDensity`` does to its result.
"""

from __future__ import annotations

import pytest

from xrdroot.cppstd.cells import Cell
from xrdroot.cppstd.span import span


def test_a_span_of_one_cell_writes_the_variable_behind_it():
    output = Cell(0.0, "double")
    view = span(output, 1)
    view[0] = 2.5
    assert output.value == 2.5
    assert view[0] == 2.5


def test_a_span_of_a_sequence_takes_its_length_from_the_count_it_was_given():
    values = [1.0, 2.0, 3.0]
    assert span(values).size() == 3
    assert span(values, 2).size() == 2
    assert span(values, 2)[1] == 2.0


def test_a_span_made_from_nothing_is_empty_as_cpp_declares_it():
    empty = span()
    assert empty.size() == 0
    assert empty.empty()
    assert list(empty) == []


def test_a_span_indexes_iterates_and_counts_what_it_was_given():
    view = span([4.0, 5.0])
    assert len(view) == 2
    assert not view.empty()
    assert list(view) == [4.0, 5.0]


def test_the_element_type_is_forgotten_so_one_class_serves_both_spellings():
    assert span[float] is span
    assert span["std::string"] is span


def test_a_span_says_how_many_elements_it_has_rather_than_what_is_in_them():
    assert repr(span([1.0, 2.0])) == "span(2 elements)"


def test_the_count_is_read_as_an_integer_whatever_spelling_the_macro_gave_it():
    view = span([1.0, 2.0, 3.0], float(2))
    assert view.size() == 2


@pytest.mark.parametrize("count", [2, 0])
def test_writing_through_a_span_of_a_sequence_writes_the_sequence(count: int) -> None:
    values = [1.0, 2.0, 3.0]
    view = span(values, count)
    for index in range(count):
        view[index] = 9.0
    assert values == [9.0] * count + [1.0, 2.0, 3.0][count:]


def test_a_char_pointer_of_a_character_is_its_text_and_of_an_object_the_object():
    """``(char *)x``: C++'s ``char *`` is an address, so an object cast to one is still the
    object - ROOT's ``(char *)f.IsA()->New()`` - while a character is the text it spells."""
    from xrdroot.cppstd.strings import char_pointer

    class Held:
        pass

    held = Held()
    assert char_pointer(65) == "A"
    assert char_pointer(None) == ""
    assert char_pointer(held)[0:] is held  # an object's address, which indexes to the object
    assert char_pointer("text") == "text"
    assert char_pointer([1, 2]) == [1, 2]


def test_an_objects_address_answers_the_object_moved_by_its_offset():
    """``p + n`` of ``(char *)obj``: indexing the address moves through the object, and only an
    offset of 0 - a base class at the object's start - names the object itself."""
    from xrdroot.cppstd.strings import char_pointer

    class Held:
        pass

    held = Held()
    address = char_pointer(held)
    assert address[0:] is held
    with pytest.raises(ValueError, match="into the middle"):
        address[8:]
    with pytest.raises(TypeError, match="not written through its address"):
        address[0:] = held
