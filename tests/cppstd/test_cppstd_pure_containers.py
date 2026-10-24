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
