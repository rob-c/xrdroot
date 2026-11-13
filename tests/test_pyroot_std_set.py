"""``ROOT.std.set``, C strings as an element type, and a vector's ``+=``."""

from __future__ import annotations

import xrdroot.pyroot as ROOT


def test_a_set_keeps_each_element_once_in_order_and_is_true_when_it_holds_any():
    numbers = ROOT.std.set["int"]([3, 1, 3])
    assert list(numbers) == [1, 3] and len(numbers) == numbers.size() == 2 and numbers
    assert 3 in numbers and numbers.count(2) == 0 and repr(numbers) == "<set<int> [1, 3]>"
    assert numbers.erase(3) == 1 and numbers.erase(3) == 0
    numbers.clear()
    assert numbers.empty() and not numbers and ROOT.std.set(int)().empty()


def test_a_set_of_char_keeps_the_letters_it_is_given():
    letters = ROOT.std.set["char"](["b", "a", "b"])
    assert set(letters) == {"a", "b"}
    numbers = ROOT.std.set["char"]()
    numbers.insert(99)
    assert 99 in numbers


def test_a_c_string_is_a_string_to_the_containers():
    names = ROOT.std.vector["const char*"]()
    names.push_back("x")
    assert list(names) == ["x"] and ROOT.std.set["const char *"](["a"]).size() == 1


def test_adding_values_to_a_vector_puts_each_at_its_end():
    numbers = ROOT.std.vector["int"]([1])
    numbers += [2, 3]
    rows = ROOT.std.vector[ROOT.std.vector[int]]([[1]])
    rows += [[2, 3]]
    assert list(numbers) == [1, 2, 3] and [list(row) for row in rows] == [[1], [2, 3]]
