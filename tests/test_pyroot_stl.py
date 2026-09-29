"""``ROOT.std``: the containers a PyROOT script hands a tree, as cppyy spells them.

The spellings are the ones the macro translator writes - ``std.vector['float']()``
for ``std::vector<float> v;`` - and the ones PyROOT scripts use, ``std.vector('double')``
and ``std.vector[float]`` among them, which all have to be one class per type.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.pyroot import stl
from xrdroot.pyroot.stl import NPOS, std


def test_every_spelling_of_one_element_type_is_the_one_class():
    assert std.vector["float"] is std.vector("float") is std.vector["Float_t"]
    assert std.vector[float] is std.vector["double"] is std.vector[np.float64]
    assert std.vector[int] is std.vector["int"]
    assert std.vector[np.dtype("int16")] is std.vector["short"]
    assert std.vector[std.string] is std.vector["std::string"] is std.vector[str]
    assert std.vector["std::vector<int>"] is std.vector[std.vector["int"]]
    assert std.vector["float"].__cpp_name__ == "vector<float>"
    assert repr(std.vector).startswith("<std.vector")


def test_a_vector_of_numbers_grows_as_it_is_pushed_and_keeps_numpy_storage():
    v = std.vector["float"]()
    assert (v.empty(), v.size()) == (True, 0)
    for value in (1.5, 2.5, 3.5):
        v.push_back(value)
    v.emplace_back(4.5)
    assert (v.size(), len(v), v.capacity() >= 4, list(v)) == (4, 4, True, [1.5, 2.5, 3.5, 4.5])
    assert (v[0], v[-1], v.at(1), v.front(), v.back()) == (1.5, 4.5, 2.5, 1.5, 4.5)
    assert v.data().dtype == np.float32


def test_a_vector_of_numbers_is_changed_compared_and_seen_as_an_array():
    v = std.vector["float"]([1.5, 2.5, 3.5, 4.5])
    v[1] = 9.0
    assert v[1:3].tolist() == [9.0, 3.5]
    v.pop_back()
    assert (v == [1.5, 9.0, 3.5], v != "abc", v == [1.5]) == (True, True, False)
    assert np.asarray(v, dtype=float).tolist() == [1.5, 9.0, 3.5]
    assert "vector<float>" in repr(v)


def test_a_vector_is_made_sized_filled_or_from_values_and_resized_as_cplusplus_does():
    assert list(std.vector("double")(3, 0.5)) == [0.5, 0.5, 0.5]
    assert list(std.vector["int"](2)) == [0, 0]
    made = std.vector["int"]([1, 2, 3])
    made.resize(5, 7)
    assert list(made) == [1, 2, 3, 7, 7]
    made.resize(2)
    assert list(made) == [1, 2]
    made.reserve(100)
    assert made.capacity() >= 100 and made.size() == 2
    made.assign(np.arange(4))
    assert list(made) == [0, 1, 2, 3]
    made.clear()
    assert made.size() == 0


def test_an_index_outside_a_vector_is_refused_rather_than_read_past_its_end():
    v = std.vector["int"]([1])
    with pytest.raises(IndexError, match="outside"):
        v[3]
    with pytest.raises(IndexError, match="outside"):
        std.vector["int"]().pop_back()


def test_a_vector_of_strings_or_vectors_keeps_its_elements_as_objects():
    words = std.vector["string"](["a", "b"])
    words.push_back("c")
    words.emplace_back(std.string("d"))
    assert [str(word) for word in words] == ["a", "b", "c", "d"]
    assert words.size() == 4 and not words.empty() and words.at(0) == "a"
    assert words.front() == "a" and words.back() == "d" and words[1] == "b"
    words[1] = "z"
    words.pop_back()
    assert words == ["a", "z", "c"] and words != 3 and "vector<string>" in repr(words)
    assert words.data()[0] == "a"
    nested = std.vector["vector<int>"](2)
    nested[0].push_back(4)
    assert [list(row) for row in nested] == [[4], []]
    nested.resize(3, [1, 2])
    assert list(nested[2]) == [1, 2]
    nested.resize(1)
    assert len(nested) == 1
    nested.clear()
    assert nested.empty()


def test_a_string_finds_and_cuts_as_cplusplus_does():
    s = std.string("hello")
    s += " world"
    assert (str(s), s.size(), s.length(), len(s)) == ("hello world", 11, 11, 11)
    assert (s.find("world"), s.find("nope"), s.rfind("o"), s.rfind("q")) == (6, NPOS, 7, NPOS)
    assert (s.substr(6), s.substr(0, 5), s.c_str(), s.data()) == (
        "world",
        "hello",
        "hello world",
        "hello world",
    )
    assert (s[0], list(s)[:2]) == ("h", ["h", "e"])
    assert (s.compare("hello world"), s.compare("a"), s.compare("z")) == (0, 1, -1)


def test_a_string_is_joined_compared_and_changed_in_place():
    s = std.string("hello world")
    assert (s + "!", "> " + s) == ("hello world!", "> hello world")
    assert (s < "z", s != 3, hash(s) == hash("hello world"), repr(s)) == (
        True,
        True,
        True,
        "'hello world'",
    )
    s.append("?")
    s.assign("x")
    assert (s == std.string(b"x"), s.empty()) == (True, False)
    s.clear()
    assert s.empty()


def test_a_map_keeps_its_keys_in_order_and_makes_what_operator_brackets_does_not_find():
    m = std.map["std::string", "int"]()
    m["b"] += 2
    m["a"] = 1
    assert m.keys() == ["a", "b"] and m.values() == [1, 2] and m.size() == len(m) == 2
    assert [(str(key), value) for key, value in m] == [("a", 1), ("b", 2)]
    assert "a" in m and m.count("a") == 1 and m.count("q") == 0 and m.at("b") == 2
    m.insert(std.pair["string", "int"]("c", 3))
    m.insert(("a", 99))
    assert m["a"] == 1 and m["c"] == 3
    assert m.erase("c") == 1 and m.erase("c") == 0
    with pytest.raises(IndexError, match="not a key"):
        m.at("zzz")
    assert "map<string,int>" in repr(m) and not m.empty()
    m.clear()
    assert m.empty()
    assert std.map["int", "double"]({2: 1.5, 1: 0.5}).items() == [(1, 0.5), (2, 1.5)]
    assert std.map["int", "double"]([(1, 2.0)])[1] == 2.0


def test_a_pair_is_first_and_second_and_unpacks_as_two():
    p = std.pair["int", "double"](1, 2)
    assert (p.first, p.second) == (1, 2.0) and p[0] == 1 and len(p) == 2
    first, second = p
    assert (first, second) == (1, 2.0) and p == (1, 2.0) and p != 5
    assert "pair<int,double>" in repr(p)
    empty = std.pair["string", "vector<int>"]()
    assert empty.first == "" and list(empty.second) == []


def test_what_std_does_not_have_is_refused_by_name():
    with pytest.raises(AttributeError, match=r"ROOT has std\.deque; xrdroot\.pyroot does not yet"):
        std.deque  # noqa: B018
    assert repr(std) == "<namespace std>"
    with pytest.raises(TypeError, match="template argument"):
        std.vector["int", "int"]
    with pytest.raises(TypeError, match="not a type the std stand-ins know"):
        std.vector["deque<int>"]
    with pytest.raises(TypeError, match="not a type the std stand-ins know"):
        std.vector["vector<int"]
    with pytest.raises(TypeError, match="not a type an STL container can hold"):
        std.vector[3.5]
    with pytest.raises(TypeError, match="not a type an STL container can hold"):
        std.vector[dict]


def test_template_arguments_split_at_the_top_level_commas_only():
    assert stl._split_arguments("map<int,int>,float") == ["map<int,int>", "float"]
    assert stl._spelled(" std::map < std::string , int > ") == "map<string,int>"
    assert std.map["std::string", "std::vector<float>"].__cpp_name__ == "map<string,vector<float>>"


def test_a_vector_of_vectors_keeps_a_vector_it_is_given_as_it_is():
    inner = std.vector["int"]([1])
    outer = std.vector["vector<int>"]()
    outer.push_back(inner)
    assert outer[0] is inner and std.vector["int"]([1]) != 5
    assert stl.is_vector(outer) and stl.is_vector(inner) and not stl.is_vector([1])


class _Interval:
    def __init__(self, low: float) -> None:
        self.low = low

    def GetMin(self) -> float:
        return self.low


def test_a_vector_of_pointers_to_a_class_keeps_the_objects_pushed_into_it():
    kind = std.vector["TMVA::Interval*"]
    assert kind is std.vector["TMVA::Interval*"]
    v = kind()
    first = _Interval(1.0)
    v.push_back(first)
    v.push_back(_Interval(2.0))
    assert v.back().low == 2.0
    assert v.begin().GetMin() == 1.0
    assert v.begin().__deref__() is first


def test_a_vector_iterator_walks_to_end_and_measures_its_distance():
    v = std.vector["double"]([1.5, 2.5, 3.5])
    it = v.begin()
    assert float(it) == 1.5
    it += 1
    assert float(it + 1) == 3.5
    assert (v.end() - v.begin(), float(v.end() - 1)) == (3, 3.5)
    assert it != v.end() and it < v.end() and v.end() == v.begin() + 3
    assert (it == 1) is False and (it != 1) is True
    with pytest.raises(AttributeError):
        getattr(it, "__missing__")  # noqa: B009


def test_a_vector_of_strings_is_walked_by_the_same_iterators():
    v = std.vector["std::string"](["a", "b"])
    assert str(v.back()) == "b"
    assert v.end() - v.begin() == 2


def test_a_map_iterator_reads_and_assigns_its_entries_and_finds_keys():
    m = std.map["int", "double"]({2: 1.5, 1: 0.5})
    it = m.begin()
    assert (it.first, it.second) == (1, 0.5)
    it.second = 4.0
    assert m[1] == 4.0
    it += 1
    assert (it + 1) == m.end() and it != m.end()
    assert m.find(2) == it and m.find(9) == m.end()
    assert (it == 2) is False and (it != 2) is True
