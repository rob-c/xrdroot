"""Whole numeric assignment writes a held C++ object without changing foreign stores."""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pytest

from xrdroot import pyroot as ROOT
from xrdroot.cint import translate
from xrdroot.cint.execute import run_source


def _calls(source: str) -> list[ast.Call]:
    python = translate(source, "f.C")
    tree = ast.parse(python)
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "assign_vector"]


@pytest.mark.parametrize(("cpp", "maker"), [
    ("std::vector<unsigned long>", ROOT.std.vector),
    ("ROOT::VecOps::RVec<unsigned long>", ROOT.RVec),
])
@pytest.mark.parametrize("route", ["target = source;", "V& alias = target; alias = source;",
                                   "V* pointer = &target; *pointer = source;"])
def test_numeric_assignment_preserves_the_target_and_its_existing_aliases(
    cpp: str, maker: Any, route: str,
) -> None:
    source = f"using V = {cpp}; V& f(V& target, const V& source) {{ {route} return target; }}"
    words = np.array([0, 2**64 - 1, 2**63, 1], dtype="uint64")
    kind = maker["unsigned long"]
    given, target = kind(words), kind([9, 8])
    target.reserve(16)
    storage = target._data
    returned = run_source(source, "f.C", (target, given), root=ROOT)
    assert returned is target and type(returned) is kind and target._data is storage
    assert target.capacity() == len(storage) and target.size() == len(words)
    assert target.data().tobytes() == given.data().tobytes() == words.tobytes()
    assert not np.shares_memory(target.data(), given.data())
    target[0] = 23
    given[1] = 37
    assert int(given[0]) == 0 and int(target[1]) == 2**64 - 1
    assert _calls(source)


@pytest.mark.parametrize(("cpp", "maker"), [
    ("std::vector<float>", ROOT.std.vector), ("ROOT::RVec<float>", ROOT.RVec),
])
def test_pointer_assignment_preserves_float_payload_words_without_scalar_conversion(
    cpp: str, maker: Any,
) -> None:
    source = (f"{cpp}& f({cpp}& target, const {cpp}& source) {{ "
              "auto* p = &target; *p = source; return *p; }")
    words = np.array([0x80000000, 0x7F800043, 0x7FC00042, 1], dtype="uint32")
    given, target = maker["float"](words.view("float32")), maker["float"]([1, 2, 3, 4])
    returned = run_source(source, "f.C", (target, given), root=ROOT)
    assert returned is target
    assert target.data().view("uint32").tobytes() == words.tobytes()
    assert given.data().view("uint32").tobytes() == words.tobytes()
    target.data().view("uint32")[0] = 0x3F800000
    assert given.data().view("uint32").tobytes() == words.tobytes()


@pytest.mark.parametrize(("cpp", "maker"), [
    ("std::vector<double>", ROOT.std.vector), ("ROOT::RVec<double>", ROOT.RVec),
])
def test_pointer_self_assignment_keeps_reserved_storage_and_all_initialized_words(
    cpp: str, maker: Any,
) -> None:
    source = f"{cpp}& f({cpp}& target) {{ {cpp}* p = &target; *p = *p; return *p; }}"
    value = maker["double"]([1, 2, 3])
    value.reserve(17)
    value._data[3:] = 9
    storage, words = value._data, value._data.tobytes()
    assert run_source(source, "f.C", (value,), root=ROOT) is value
    assert value._data is storage and value._data.tobytes() == words and value.size() == 3
    python = translate(source, "f.C")
    assert "assign_vector" in python and "value_copy" not in python


@pytest.mark.parametrize(("cpp", "maker"), [
    ("std::vector<double>", ROOT.std.vector), ("ROOT::RVec<double>", ROOT.RVec),
])
def test_chained_assignment_returns_the_left_target_and_independent_values(
    cpp: str, maker: Any,
) -> None:
    source = (f"{cpp}& f({cpp}& target, {cpp}& peer, const {cpp}& source) {{ "
              "return target = peer = source; }")
    target, peer, given = (maker["double"](values) for values in ([1], [2], [3, 4]))
    assert run_source(source, "f.C", (target, peer, given), root=ROOT) is target
    assert list(target) == list(peer) == list(given) == [3, 4]
    assert not np.shares_memory(target.data(), peer.data())
    assert not np.shares_memory(target.data(), given.data())
    assert not np.shares_memory(peer.data(), given.data())
    peer[0] = 8
    given[1] = 9
    assert list(target) == [3, 4] and list(peer) == [8, 4] and list(given) == [3, 9]


@pytest.mark.parametrize("cpp", ["std::vector<long double>", "ROOT::RVec<long double>"])
def test_long_double_assignment_keeps_the_existing_factory_dtype_and_numeric_values(
    cpp: str,
) -> None:
    maker = ROOT.std.vector if cpp.startswith("std") else ROOT.RVec
    kind = maker["long double"]
    given, target = kind([1, 2]), kind([3, 4])
    source = f"{cpp}& f({cpp}& target, const {cpp}& source) {{ target = source; return target; }}"
    assert run_source(source, "f.C", (target, given), root=ROOT) is target
    assert target.dtype == given.dtype and list(target) == list(given) == [1, 2]
    assert not np.shares_memory(target.data(), given.data())


@pytest.mark.parametrize("cpp", ["std::vector<bool>", "ROOT::RVec<bool>",
                                  "std::vector<std::vector<double>>",
                                  "std::vector<double, std::allocator<double>>"])
def test_noneligible_cpp_container_types_keep_the_original_assignment_route(cpp: str) -> None:
    assert not _calls(f"void f({cpp}& target, const {cpp}& source) {{ target = source; }}")


def test_a_macros_own_class_keeps_its_existing_class_assignment_path() -> None:
    source = ("struct Value { int member; }; "
              "void f(Value& target, const Value& source) { target = source; }")
    assert not _calls(source)


@pytest.mark.parametrize("rhs", ["{1., 2.}", "std::vector<double>{1., 2.}"])
def test_initializer_lists_and_fresh_temporaries_keep_the_existing_store_route(rhs: str) -> None:
    source = f"void f(std::vector<double>& target) {{ target = {rhs}; }}"
    assert not _calls(source)


class _Rows(list[Any]):
    def __init__(self, value: Any, events: list[str]) -> None:
        super().__init__([value])
        self.events = events

    def __getitem__(self, index: Any) -> Any:
        self.events.append("index")
        return super().__getitem__(index)


class _Holder:
    def __init__(self, value: Any, events: list[str]) -> None:
        self.held, self.events = value, events

    @property
    def value(self) -> Any:
        self.events.append("member")
        return self.held


@pytest.mark.parametrize("member", [False, True], ids=["index", "member"])
def test_vector_reference_aliases_capture_index_or_member_objects_only_once(member: bool) -> None:
    events: list[str] = []
    target, given = ROOT.std.vector["double"]([1]), ROOT.std.vector["double"]([3, 4])
    if member:
        declaration = "struct Holder { std::vector<double> value; };"
        parameter, binding = "Holder& owner", "owner.value"
        owner: Any = _Holder(target, events)
    else:
        declaration = ""
        parameter, binding = "std::vector<std::vector<double>>& owner", "owner[0]"
        owner = _Rows(target, events)
    source = (declaration + f"std::vector<double>& f({parameter}, "
              "const std::vector<double>& source) { "
              f"std::vector<double>& first = {binding}; "
              "std::vector<double>& alias = first; alias = source; return alias; }")
    assert run_source(source, "f.C", (owner, given), root=ROOT) is target
    assert events == (["member"] if member else ["index"])
    assert list(target) == list(given) == [3, 4]
    target[0] = 7
    assert list(given) == [3, 4] and list(target) == [7, 4]
    assert _calls(source)


@pytest.mark.parametrize(("declaration", "rhs", "copy"), [
    ("", "source", True),
    ("std::vector<double>& get_ref();", "get_ref()", False),
    ("", "(peer = source)", False),
])
def test_pointer_fallback_uses_the_original_rhs_copy_decision(
    declaration: str, rhs: str, copy: bool,
) -> None:
    source = (declaration + "void f(std::vector<double>* target, std::vector<double>& peer, "
              f"const std::vector<double>& source) {{ *target = {rhs}; }}")
    calls = [call for call in _calls(source)
             if isinstance(call.args[0], ast.Name) and call.args[0].id == "target"]
    assert len(calls) == 1
    flags = [arg.value for arg in calls[0].args[2:] if isinstance(arg, ast.Constant)]
    assert flags == ([True] if copy else [True, False])


class _Source:
    def __init__(self, events: list[str], copied: bool = False) -> None:
        self.events, self.copied = events, copied

    def __deepcopy__(self, memo: dict[int, Any]) -> _Source:
        self.events.append("copy")
        return _Source(self.events, True)


class _Target:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.got: Any = None

    @property
    def value(self) -> None:
        self.events.append("value")

    def _assign(self, value: Any) -> None:
        self.events.append("assign")
        self.got = value


def test_declined_typed_pointer_keeps_rhs_copy_before_target_callbacks() -> None:
    events: list[str] = []
    given, target = _Source(events), _Target(events)
    source = ("void f(std::vector<double>* target, const std::vector<double>& source) "
              "{ *target = source; }")
    run_source(source, "f.C", (target, given), root=ROOT)
    assert events == ["copy", "value", "assign"]
    assert isinstance(target.got, _Source) and target.got.copied and target.got is not given


def test_declined_reference_return_keeps_target_callbacks_without_an_extra_copy() -> None:
    events: list[str] = []
    given, target = _Source(events), _Target(events)
    source = ("std::vector<double>& get_ref(std::vector<double>& source) { return source; } "
              "void f(std::vector<double>* target, std::vector<double>& source) "
              "{ *target = get_ref(source); }")
    run_source(source, "f.C", (target, given), root=ROOT)
    assert events == ["value", "assign"] and target.got is given


@pytest.mark.parametrize(("cpp", "maker"), [
    ("std::vector<double>", ROOT.std.vector), ("ROOT::RVec<double>", ROOT.RVec),
])
def test_assigning_a_vector_pointer_rebinds_the_pointer_without_copying_either_vector(
    cpp: str, maker: Any,
) -> None:
    source = f"{cpp}* f({cpp}* target, {cpp}* source) {{ target = source; return target; }}"
    target, given = maker["double"]([1, 2]), maker["double"]([3, 4])
    target_storage, given_storage = target._data, given._data
    target_bits, given_bits = target.data().tobytes(), given.data().tobytes()
    assert run_source(source, "f.C", (target, given), root=ROOT) is given
    assert target._data is target_storage and given._data is given_storage
    assert target.data().tobytes() == target_bits and given.data().tobytes() == given_bits
    assert not _calls(source)
    assert "value_copy" not in translate(source, "f.C")


def test_a_library_namespace_value_keeps_its_existing_copy_assignment_route() -> None:
    source = ("ROOT::Math::XYZVector f(ROOT::Math::XYZVector target, "
              "const ROOT::Math::XYZVector& source) { target = source; return target; }")
    target, given = ROOT.Math.XYZVector(1, 2, 3), ROOT.Math.XYZVector(4, 5, 6)
    result = run_source(source, "f.C", (target, given), root=ROOT)
    assert type(result) is type(given) and result is not given and result is not target
    assert (result.X(), result.Y(), result.Z()) == (4, 5, 6)
    assert (target.X(), target.Y(), target.Z()) == (1, 2, 3)
    result.SetXYZ(7, 8, 9)
    assert (given.X(), given.Y(), given.Z()) == (4, 5, 6)
    assert not _calls(source) and "value_copy(source)" in translate(source, "f.C")


@pytest.mark.parametrize(("alias", "word", "bits"), [
    ("RVecD", "uint64", [0x8000000000000000, 0x7FF0000000000043, 0x7FF8000000000042]),
    ("RVecF", "uint32", [0x80000000, 0x7F800043, 0x7FC00042]),
])
def test_root_numeric_vector_aliases_assign_into_the_existing_object_without_scalar_promotion(
    alias: str, word: str, bits: list[int],
) -> None:
    kind = getattr(ROOT, alias)
    words = np.array(bits, dtype=word)
    given, target = kind(words.view(kind.dtype)), kind([1, 2])
    target.reserve(16)
    storage, capacity = target._data, target.capacity()
    source = (f"ROOT::{alias}& f(ROOT::{alias}& target, const ROOT::{alias}& source) "
              "{ target = source; return target; }")
    assert run_source(source, "f.C", (target, given), root=ROOT) is target
    assert target._data is storage and target.capacity() == capacity and target.size() == 3
    assert (
        target.data().view(word).tobytes() == given.data().view(word).tobytes()
        == words.tobytes()
    )
    assert not np.shares_memory(target.data(), given.data())
    target.data().view(word)[0] = 1
    assert given.data().view(word).tobytes() == words.tobytes()
    assert _calls(source)
