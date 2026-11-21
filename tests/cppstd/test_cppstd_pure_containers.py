"""Container contracts need neither a ROOT resolver nor interpreter fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xrdroot.cppstd import containers, references, stdlib


def test_standard_template_spellings_share_one_concrete_numeric_class() -> None:
    vector = containers.std.vector
    assert vector["float"] is vector("float") is vector[np.dtype("float32")]
    assert vector[int] is vector["int"]
    assert vector[float] is vector["double"]
    assert vector["std::vector<int>"] is vector[vector["int"]]


def test_numeric_vectors_reserve_storage_without_changing_their_logical_values() -> None:
    value = containers.std.vector["int"]([1, 2, 3])
    value.reserve(20)
    storage = value._data
    source = type(value)([7, 8, 9, 10])
    assert references.assign_vector(value, source) is value
    assert value._data is storage
    assert value.capacity() >= 20 and list(value) == [7, 8, 9, 10]
    source[0] = 99
    assert value.front() == 7


def test_nested_vectors_copy_occurrences_and_preserve_references_on_element_assignment() -> None:
    leaf = containers.std.vector["int"]
    source = leaf([1, 2])
    values = containers.std.vector[leaf]([source, source])
    held, sibling = values[0], values[1]
    assert held is not source and sibling is not source and held is not sibling
    source[0] = 10
    assert [list(item) for item in values] == [[1, 2], [1, 2]]
    values[0] = leaf([7, 8, 9])
    assert values[0] is held and values[1] is sibling
    assert list(held) == [7, 8, 9] and list(sibling) == [1, 2]


def test_map_insertion_preserves_order_and_existing_values() -> None:
    std = containers.std
    mapping = std.map["string", "int"]()
    mapping["b"] += 2
    mapping["a"] = 1
    mapping.insert(std.pair["string", "int"]("a", 99))
    assert mapping.keys() == ["a", "b"] and mapping.values() == [1, 2]
    assert [(str(key), value) for key, value in mapping] == [("a", 1), ("b", 2)]
    with pytest.raises(IndexError, match="not a key"):
        mapping.at("missing")


def test_string_search_and_mutation_keep_the_standard_sentinels() -> None:
    value = containers.std.string("hello world")
    assert value.find("world") == 6 and value.rfind("o") == 7
    assert value.find("missing") == containers.NPOS
    assert str(value.substr(6)) == "world"
    value.assign("x")
    value.append("yz")
    assert str(value) == "xyz" and value.compare("xyz") == 0


def test_opaque_construction_uses_only_the_explicit_local_resolver() -> None:
    class Value:
        def __init__(self, left: int, right: int) -> None:
            self.pair = left, right

    seen = []

    def resolver(name: str) -> type:
        seen.append(name)
        return Value

    containers.set_type_resolver(resolver)
    values = containers.std.vector["CppstdPureP2ValueFixture"]()
    values.emplace_back(7, 9)
    assert values[0].pair == (7, 9)
    assert seen == ["CppstdPureP2ValueFixture"]


def test_numeric_limits_keep_the_declared_integer_width_and_float_precision() -> None:
    limits = stdlib.numeric_limits
    assert limits["int"].min() == -(2**31) and limits["int"].max() == 2**31 - 1
    assert limits["unsigned short"].min() == 0 and limits["unsigned short"].max() == 65535
    assert limits["double"].epsilon() == 2.220446049250313e-16
    assert limits["bool"].min() is False and limits["bool"].max() is True


def test_generic_file_streams_write_and_close_without_loading_a_reader(tmp_path: Path) -> None:
    target = tmp_path / "values.txt"
    output = containers.std.ofstream(target)
    assert output.is_open()
    output << "answer " << 42
    output.close()
    assert not output.is_open() and target.read_text() == "answer 42"
    assert stdlib.remove(target) == 0 and stdlib.remove(target) == -1


def test_lists_and_deques_grow_and_shrink_at_both_ends() -> None:
    held = containers.std.list["int"]([2, 3])
    held.push_front(1)
    held.pop_back()
    assert list(held) == [1, 2] and type(held).__cpp_name__ == "list<int>"
    held.pop_front()
    queue = containers.std.deque["std::string"]()
    queue.push_back("b")
    queue.push_front("a")
    assert list(held) == [2] and list(queue) == ["a", "b"]


def test_a_multimap_keeps_every_pair_in_key_order() -> None:
    held = containers.std.multimap["int", "double"]([(2, 1.0), (1, 5.0)])
    held.insert((2, 3.0))
    assert held.items() == [(1, 5.0), (2, 1.0), (2, 3.0)]
    assert (held.size(), len(held), held.count(2), held.empty()) == (3, 3, 2, False)
    assert held[0].second == 5.0 and held.begin() != held.end()
    held.clear()
    assert held.empty() and list(held) == []


def test_unordered_containers_are_their_ordered_kin_by_another_name() -> None:
    table = containers.std.unordered_map["int", "int"]()
    table[2] = 4
    kinds = containers.std.unordered_set["int"]()
    kinds.insert(3)
    assert type(table).__cpp_name__ == "unordered_map<int,int>" and dict(table.items()) == {2: 4}
    assert type(kinds).__cpp_name__ == "unordered_set<int>" and list(kinds) == [3]


def test_a_vector_is_made_from_a_range_and_inserted_into() -> None:
    from xrdroot.cppstd.containers import std
    from xrdroot.cppstd.objects import Pair

    numbers = std.vector["double"]([1.0, 2.0, 3.0])
    part = std.vector["double"](numbers.begin() + 1, numbers.end())
    assert list(part) == [2.0, 3.0]
    at = part.insert(part.begin(), 2, 0.5)
    part.insert(at + 1, 9.0)
    part.insert(len(part), numbers.begin(), numbers.begin() + 1)
    assert list(part) == [0.5, 9.0, 0.5, 2.0, 3.0, 1.0]
    pairs = std.vector["std::pair<double,double>"]([Pair(1.0, 2.0)])
    more = std.vector["std::pair<double,double>"](pairs.begin(), pairs.end())
    more.insert(more.end(), pairs.begin(), pairs.end())
    assert [p.first for p in more] == [1.0, 1.0]


def test_a_braced_list_deduces_its_container_element_type_as_cpp17_does() -> None:
    std = containers.std
    assert type(std.vector([1, 2, 3])) is std.vector["int"]
    assert type(std.vector([1.5, 2])) is std.vector["double"]
    assert type(std.vector([True, False])) is std.vector["bool"]
    assert list(std.vector(["a", "b"])) == ["a", "b"]
    made = std.array([1, 2, 3])
    assert type(made) is std.vector["int"] and list(made) == [1, 2, 3]


def test_the_stream_names_std_has_are_the_streams_own() -> None:
    from xrdroot.cppstd import streams

    assert containers.std.ostringstream is streams.ostringstream
    assert containers.std.setw is streams.setw
    with pytest.raises(AttributeError, match=r"ROOT has std\.nothing"):
        containers.std.nothing  # noqa: B018


#: Every template of the family whose C++ declarations name ``::iterator`` and
#: ``::const_iterator`` - the names the translator emits for a declared iterator.
CONTAINER_TEMPLATES = ("vector", "list", "deque", "set", "multiset", "map", "multimap",
                       "forward_list", "unordered_map", "unordered_set",
                       "unordered_multimap", "unordered_multiset")  # fmt: skip


def test_every_container_template_names_the_iterator_a_declaration_makes() -> None:
    """``std::list<std::pair<TBranch*,TClonesArray*> >::iterator iBranch;`` of a macro: the
    translator drops the template arguments of a nested name, so every template answers the
    one class, and its ``const_iterator`` is the same one."""
    for name in CONTAINER_TEMPLATES:
        template = containers.TEMPLATES[name]
        assert template.const_iterator is template.iterator
        assert repr(template.iterator()) == "<iterator, no container given>"


def test_a_declared_iterator_is_walked_from_begin_to_end_as_cpp_walks_one() -> None:
    """The loop of ``readauto``: assign ``begin()``, compare against ``end()``, ``iBranch->first``
    and ``iBranch++`` - over the containers a macro declares one of."""
    declared = containers.TEMPLATES["list"].iterator
    branches = containers.std.list["std::pair<int,double>"]()
    branches.push_back(containers.std.make_pair(1, 2.0))
    branches.push_back(containers.std.make_pair(3, 4.0))
    walked = []
    it = declared()
    it = branches.begin()
    while it != branches.end():
        walked.append((it.first, it.second))
        it += 1
    assert walked == [(1, 2.0), (3, 4.0)]

    numbers = containers.std.vector["int"]([7, 8])
    held = []
    at = numbers.begin()
    while at != numbers.end():
        held.append((int(at), float(at)))
        at = at + 1
    assert held == [(7, 7.0), (8, 8.0)]

    table = containers.std.map["std::string","int"]()
    table["a"] = 1
    pairs = []
    key = containers.std.map.iterator()
    key = table.find("a")
    while key != table.end():
        pairs.append((key.first, key.second))
        key += 1
    assert pairs == [("a", 1)]


def test_an_iterator_no_container_gave_out_reads_nothing_and_steps_nowhere() -> None:
    """An unassigned one is compared and stepped before ``begin()`` replaces it; reading
    through it refuses with what to do instead of failing on an index."""
    unassigned = containers.std.deque.iterator()
    assert unassigned == containers.std.deque.iterator()
    assert unassigned != containers.std.vector["int"]([1]).begin()
    assert unassigned.__iadd__(1) is unassigned and unassigned + 1 is unassigned
    with pytest.raises(TypeError, match="assign it a container's begin"):
        unassigned.__deref__()
    with pytest.raises(TypeError, match="assign it a container's begin"):
        unassigned.second  # noqa: B018
    assert not hasattr(unassigned, "__wrapped__")


def test_a_sets_iterator_steps_the_elements_in_the_order_iteration_gives() -> None:
    """``std::set<float>::const_iterator i = s.begin(); i != s.end(); ++i``: a set is indexed
    by no position, so its iterator counts off the elements in the order iteration gives."""
    values = containers.std.set["double"]()
    for value in (3.0, 1.0, 2.0):
        values.insert(value)
    walked = []
    at = values.begin()
    while at != values.end():
        walked.append(float(at))
        at += 1
    assert walked == [1.0, 2.0, 3.0]
    assert values.end() - values.begin() == 3
    assert float(values.begin() + 2) == 3.0 and float(values.end() - 1) == 3.0
    assert at == values.end()

    names = containers.std.set["std::string"]()
    names.insert("b")
    assert names.begin().c_str() == "b"

    repeats = containers.std.multiset["int"]()
    repeats.insert(1)
    repeats.insert(2)
    repeats.insert(1)
    seen = []
    each = repeats.begin()
    while each != repeats.end():
        seen.append(int(each))
        each += 1
    assert seen == [1, 1, 2]
