"""Generic containers remain usable before ROOT explicitly installs its adapters."""

from __future__ import annotations

import operator
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from xrdroot.cppstd import containers, stdlib


def test_root_standard_factories_reexport_the_generic_instances() -> None:
    from xrdroot.pyroot import stl

    assert stl.std is containers.std
    assert stl.TEMPLATES is containers.TEMPLATES
    assert stl.NUMBERS is containers.NUMBERS
    assert stl.string is containers.string
    assert stl.element_type is containers.element_type
    assert stl.std.vector["Float_t"] is containers.std.vector["float"]
    assert stl.std.vector["TString"] is containers.std.vector["string"]
    assert stl.std.numeric_limits is stdlib.numeric_limits
    assert stl.std.numeric_limits["Int_t"].min() == -(2**31)


def test_root_vector_templates_and_emplacement_are_explicit_adapters() -> None:
    from xrdroot.pyroot import stl
    from xrdroot.pyroot.rdf.rvec import RVec

    assert stl.std.vector["ROOT::VecOps::RVec<float>"].element is RVec["float"]
    values = stl.std.vector["ROOT::Math::XYZTVector"]()
    values.emplace_back(1, 2, 3, 4)
    assert (values[0].X(), values[0].Y(), values[0].Z(), values[0].T()) == (1, 2, 3, 4)


def test_opaque_element_construction_uses_the_explicit_resolver(monkeypatch: pytest.MonkeyPatch):
    class Value:
        def __init__(self, left, right):
            self.pair = (left, right)

    monkeypatch.setattr(containers, "TYPE_RESOLVER", containers.TYPE_RESOLVER)
    seen = []

    def resolver(name):
        seen.append(name)
        return Value

    # The resolver is the one ROOT installs when pyroot is imported: put it back, or every
    # later test in this process finds no adapter at all.
    previous = containers.TYPE_RESOLVER
    containers.set_type_resolver(resolver)
    try:
        values = containers.std.vector["CppstdValueFixture"]()
        values.emplace_back(7, 9)
        assert values[0].pair == (7, 9)
        assert seen == ["CppstdValueFixture"]
        assert containers._resolved_class(np.dtype("int32")) is None
        containers.set_type_resolver(None)
        assert containers._resolved_class(containers._opaque("NoAdapterFixture")) is None
    finally:
        containers.set_type_resolver(previous)


def test_standard_containers_and_traits_import_without_root_or_the_interpreter() -> None:
    program = """
import sys

class Boundary:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('xrdclient'):
            raise AssertionError(fullname)
        if fullname.startswith('xrdroot.') and not fullname.startswith('xrdroot.cppstd'):
            raise AssertionError(fullname)

sys.meta_path.insert(0, Boundary())
from xrdroot.cppstd import containers, stdlib
std = containers.std
assert containers.TYPE_RESOLVER is None
assert 'Float_t' not in containers.NUMBERS
assert 'Int_t' not in stdlib._KINDS
assert set(containers.TEMPLATES) == {'vector', 'map', 'pair', 'set', 'list', 'deque', 'multimap',
                                     'multiset', 'forward_list', 'complex', 'atomic',
                                     'unordered_map', 'unordered_set',
                                     'unordered_multimap', 'unordered_multiset'}
values = std.vector['int']([1, 2, 3])
copy = std.vector['vector<int>']([values])
values[0] = 9
assert list(copy[0]) == [1, 2, 3]
mapping = std.map['string', 'int']()
mapping['answer'] = 42
assert mapping['answer'] == 42
assert std.numeric_limits['int'].min() == -(2**31)
assert std.ofstream.__module__ == 'xrdroot.cppstd.streams'
assert not any(name.startswith('xrdclient') for name in sys.modules)
assert not any(name.startswith('xrdroot.') and not name.startswith('xrdroot.cppstd')
               for name in sys.modules)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", program], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_legacy_imports_share_the_implementation_module_and_lookup_globals(monkeypatch):
    from xrdroot.pyroot import stl

    assert stl is containers
    assert vars(stl) is containers.cpp_name.__globals__
    replacement = dict(stl.CPP_NAMES, int32="legacy_integer")
    monkeypatch.setattr(stl, "CPP_NAMES", replacement)
    assert containers.CPP_NAMES is replacement
    assert containers.cpp_name(np.dtype("int32")) == "legacy_integer"
    replacement = dict(stl._DTYPE_NAMES)
    replacement[np.dtype("int32").num] = "float32"
    monkeypatch.setitem(vars(stl), "_DTYPE_NAMES", replacement)
    assert containers._DTYPE_NAMES is replacement
    assert stl.cpp_name(np.dtype("int32")) == "float"


def test_a_complex_keeps_its_parts_at_the_width_it_was_asked_for() -> None:
    """``std::complex<T>``: one class per argument however it is spelled, its parts at the
    template's width, and the reading of them C++ gives."""
    C, F = containers.std.complex["double"], containers.std.complex["float"]
    z = C(1.0, 2.0)
    assert (z.real(), z.imag(), str(z), repr(z)) == (
        1.0, 2.0, "(1,2)", "<complex<double> (1,2)>")
    assert containers.std.complex["float"] is F is containers.std.complex[np.float32]
    assert containers.std.complex[float] is C  # Python's float is a C++ double
    assert isinstance(F(1, 2).real(), float) and F.part is np.float32
    assert (F(0.1, 0).real(), C(0.1, 0).real()) == (np.float32(0.1).item(), 0.1)
    held = C()
    assert (held.real(3.5), held.imag(2.5), held) == (None, None, C(3.5, 2.5))
    assert (C(z), C(2.0), complex(z)) == (C(1, 2), C(2, 0), 1 + 2j)
    assert abs(z) == pytest.approx(2.23606797749979)


def test_a_complex_adds_subtracts_multiplies_and_divides_as_cpp_does() -> None:
    C = containers.std.complex["double"]
    z = C(1.0, 2.0)
    assert (z + C(1, -2), z - 1.0, 1.0 - z, -z, z.__pos__()) == (
        C(2, 0), C(0, 2), C(0, -2), C(-1, -2), z)
    assert (z * z, z / C(1, -1), z / 2, z * 2, 2 * z) == (
        C(-3, 4), C(-0.5, 1.5), C(0.5, 1), C(2, 4), C(2, 4))
    assert z == C(1, 2) and z != 1.0 and bool(z) is True


def test_the_exponential_and_logarithm_of_a_complex_are_the_standards() -> None:
    """``exp``, ``log``, ``sqrt`` and ``pow`` of a complex, which C++ finds by its type rather
    than in ``<cmath>``."""
    import math

    from xrdroot.cppstd import cmath as cm

    z = containers.std.complex["double"](1.0, 2.0)
    unit = containers.std.complex["double"](0.0, math.pi / 2)
    assert cm.exp(z).real() == pytest.approx(math.exp(1) * math.cos(2))
    assert cm.exp(z).imag() == pytest.approx(math.exp(1) * math.sin(2))
    assert cm.exp(unit).real() == pytest.approx(0.0, abs=1e-15)
    assert cm.exp(unit).imag() == pytest.approx(1.0)
    assert cm.log(z).real() == pytest.approx(math.log(math.hypot(1, 2)))
    assert cm.log(z).imag() == pytest.approx(math.atan2(2, 1))
    assert cm.log10(z) == cm.log(z) / math.log(10.0)
    assert cm.sqrt(z) * cm.sqrt(z) == z
    # ``pow`` goes through ``log`` and ``exp``, so its answer is the nearest double.
    assert abs(cm.pow(z, 2.0) - z * z) < 1e-12


def test_the_trigonometric_functions_of_a_complex_are_the_standards() -> None:
    import math

    from xrdroot.cppstd import cmath as cm

    z = containers.std.complex["double"](1.0, 2.0)
    unit = containers.std.complex["double"](0.0, math.pi / 2)
    # sin(iy) = i sinh(y) and cos(iy) = cosh(y), as ``<complex>`` defines them.
    assert cm.sin(unit).real() == pytest.approx(0.0)
    assert cm.sin(unit).imag() == pytest.approx(math.sinh(math.pi / 2))
    assert cm.cos(unit).imag() == pytest.approx(0.0)
    assert cm.cos(unit).real() == pytest.approx(math.cosh(math.pi / 2))
    assert cm.sin(z).real() == pytest.approx(math.sin(1) * math.cosh(2))
    assert cm.sinh(z).real() == pytest.approx(math.sinh(1) * math.cos(2))
    assert cm.tan(z) == cm.sin(z) / cm.cos(z)
    assert cm.tanh(z) == cm.sinh(z) / cm.cosh(z)
    assert abs(cm.pow(z, z) - cm.exp(z * cm.log(z))) < 1e-12  # exp(w log z), to the double


def test_the_scalar_questions_about_a_complex_are_the_standards() -> None:
    from xrdroot.cppstd import cmath as cm

    z = containers.std.complex["double"](1.0, 2.0)
    assert cm.cabs(z) == pytest.approx(2.23606797749979)
    assert stdlib.complex_math("norm", z) == 5.0
    assert stdlib.complex_math("arg", z) == pytest.approx(1.1071487177940904)
    assert stdlib.complex_math("conj", z) == containers.std.complex["double"](1, -2)
    with pytest.raises(AttributeError, match="no std::acos of a std::complex"):
        stdlib.complex_math("acos", z)
    assert cm.sqrt(-1.0) != cm.sqrt(-1.0)  # nan, as C's is
    assert cm.pow(2.0, 3.0) == 8.0


def test_an_atomic_is_read_and_written_whole_as_cpps_is() -> None:
    counted = stdlib.atomic_type("int")(0)
    counted += 1
    counted += 1
    assert (int(counted), counted.load(), counted.value) == (2, 2, 2)
    assert counted.fetch_add(2) == 2 and int(counted) == 4
    assert counted.fetch_sub(1) == 4 and int(counted) == 3
    assert counted.exchange(9) == 3 and int(counted) == 9
    counted.store(1)
    counted.value = 2
    assert (int(counted), float(counted), bool(counted)) == (2, 2.0, True)


def test_an_atomic_answers_arithmetic_and_ordering_as_the_value_it_holds() -> None:
    counted = stdlib.atomic_type("int")(2)
    assert (counted + 1, 1 + counted, counted - 1, 5 - counted) == (3, 3, 1, 3)
    assert (counted * 2, counted / 2) == (4, 1.0)
    assert (counted == 2, counted < 3, counted > 2, counted <= 2, counted >= 2) == (
        True, True, False, True, True)
    assert stdlib.atomic_type("atomic_int") is stdlib.atomic_type("int")
    assert stdlib.atomic_type("int") is stdlib.atomic_type(np.int32)
    assert containers.std.atomic_bool(True).value is True
    assert str(containers.std.atomic_int(1)) == "<atomic<int> 1>"


def test_a_forward_list_adds_at_its_front_and_iterates_from_it() -> None:
    assert containers.std.forward_list["int"] is containers.std.forward_list[np.int32]
    forwards = containers.std.forward_list["int"]()
    forwards.emplace_front(1)
    forwards.push_front(2)
    assert list(forwards) == [2, 1]  # what was added last comes first, as C++ gives it
    forwards.pop_front()
    assert list(forwards) == [1]


def _a_numbered_thing() -> _Numbered:
    """An object of an element type none of the containers here knows: what a set of a
    macro's own class holds, which the adapter resolves to the object itself."""
    return _Numbered()


class _Numbered:
    """One member, so that "the copy kept the value it was inserted with" can be read."""

    def __init__(self) -> None:
        self.fInt = 0

    def __lt__(self, other: _Numbered) -> bool:
        return self.fInt < other.fInt


def test_a_multiset_keeps_an_element_as_often_as_it_is_inserted() -> None:
    multiset = containers.std.multiset["int"]([3, 1, 1])
    assert (list(multiset), len(multiset), multiset.count(1), bool(multiset)) == (
        [1, 1, 3], 3, 2, True)
    assert (2 in multiset, multiset.insert(2), list(multiset)) == (False, None, [1, 1, 2, 3])
    assert multiset.erase(1) == 2 and list(multiset) == [2, 3]
    assert multiset.erase(9) == 0 and multiset.size() == 2 and multiset.empty() is False
    multiset.clear()
    assert (multiset.empty(), multiset.size(), list(multiset)) == (True, 0, [])


def test_make_pair_deduces_the_types_cpp_deduces() -> None:
    made = containers.std.make_pair(1, "x")
    assert (made.first, made.second) == (1, "x")
    assert made.__cpp_name__ == "pair<int,string>", "C++ deduces int for a literal 1"
    assert containers.std.make_pair(2**40, "x").__cpp_name__ == "pair<long,string>"
    assert containers.std.make_pair(1.5, True).__cpp_name__ == "pair<double,bool>"
    assert containers.std.make_pair(containers.std.multiset["int"](), "s").first is not None


def test_a_map_emplace_takes_a_pair_or_a_key_and_value_or_two_tuples() -> None:
    """``std::map::emplace`` in each form a macro writes it: a key already there keeps its
    value, and ``piecewise_construct`` builds from the two ``forward_as_tuple``s."""
    mapping = containers.std.map["int", "int"]()
    mapping.emplace(1, 2)
    mapping.emplace(containers.std.make_pair(3, 4))
    mapping.emplace(1, 9)  # a key already there is left alone, as C++'s emplace leaves it
    assert dict(mapping.items()) == {1: 2, 3: 4}
    built = containers.std.map["float", "int"]()
    forwarded = containers.std.forward_as_tuple(1.5)
    built.emplace(containers.std.piecewise_construct, forwarded, containers.std.forward_as_tuple(7))
    assert dict(built.items()) == {1.5: 7}
    assert (len(forwarded), forwarded[0], list(forwarded), repr(forwarded)) == (
        1, 1.5, [1.5], "<tuple (1.5,)>")
    assert repr(containers.std.piecewise_construct) == "<piecewise_construct tag>"


def test_a_complex_is_only_of_a_floating_type_as_cpp_makes_it() -> None:
    """``std::complex<T>`` of anything but a floating type is refused by name, and each
    spelling of a width is the one class of it."""
    with pytest.raises(TypeError, match=r"std::complex<string> is not a complex this kit has"):
        stdlib.complex_type(containers.std.string)
    assert stdlib.complex_type("long double").part is np.float64
    assert (stdlib._cxx_name("double"), stdlib._cxx_name(np.int64), stdlib._cxx_name(int)) == (
        "double", "long", "int")
    tiny = stdlib.atomic_type("short")(1)
    assert (tiny.value_type is int, int(tiny)) == (True, 1)
    assert containers.std.atomic["long"](2).load() == 2


def test_a_complex_and_an_atomic_are_added_to_in_place_as_cpp_adds() -> None:
    """``z += w`` and ``n += 1`` keep the object - which is what a translated ``++n`` and a
    ``z -= w`` rely on - and an atomic spelled ``atomic<int>`` is the ``atomic_int`` class."""
    z = containers.std.complex["double"](1.0, 2.0)
    z += containers.std.complex["double"](1, -2)
    z -= 1.0
    assert z == containers.std.complex["double"](1, 0)
    counted = stdlib.atomic_type("int")(1)
    counted += 2
    counted -= 1
    assert int(counted) == 2
    assert stdlib.atomic_type("atomic<int>") is stdlib.atomic_type("int")
    assert stdlib.atomic_type("short").__name__ == "atomic<short>"
    assert operator.index(counted) == 2  # ``a[i]`` of an atomic is its value, as C++'s is
    assert stdlib.complex_type("double") is stdlib.complex_type("double")
    assert repr(containers.std.multiset["int"]([1])) == "<multiset<int> [1]>"


def test_a_list_sorts_deduplicates_reverses_and_removes_in_place() -> None:
    """The four things only a ``std::list`` does to itself, which is what
    ``roottest/root/meta/loadAllLibs`` runs over the libraries it finds."""
    held = containers.std.list["string"](["b", "a", "a", "c"])
    held.sort()
    assert list(held) == ["a", "a", "b", "c"]
    held.unique()
    assert list(held) == ["a", "b", "c"]
    held.reverse()
    assert list(held) == ["c", "b", "a"]
    held.remove("b")
    assert list(held) == ["c", "a"]
    empty = containers.std.list["int"]()
    empty.unique()
    assert list(empty) == []


def test_a_std_array_takes_a_member_declared_with_its_inner_braces() -> None:
    """C++ elides the braces of a one-dimensional aggregate: ``std::array<int,3> a {{0,0,0}}``
    hands over ``{0,0,0}``, which a member declared that way in a macro does -
    roottest's ``io/stdarray`` declares one."""
    assert list(containers.std.array["int", 3]([[1, 2, 3]])) == [1, 2, 3]
    assert list(containers.std.array["int", 3]([1, 2, 3])) == [1, 2, 3]
    assert containers._unbraced(([1, 2], [3, 4])) == [[1, 2], [3, 4]]  # a real nesting stays


def test_a_class_name_is_the_cpp_name_its_layout_carries() -> None:
    """A class a macro declares is named by its layout's C++ name, which a container of its
    objects is named with - and one that says its own name is taken at its word."""

    class Declared:
        _cxx_layout_ = ("ns::Thing<int>", (), ())

    assert containers.cpp_name(Declared) == "ns::Thing<int>"
    assert containers.cpp_name(containers.string) == "string"
    assert containers.cpp_name(np.dtype("int32")) == "int"


def test_a_template_name_no_container_here_is_declared_is_resolved_by_the_installed_adapter():
    """``std::vector<ArrayHolderT<int>>``: the element is a class a macro declared, which the
    adapter ROOT installs finds by its C++ name - and nothing else is invented."""
    from xrdroot.cppstd import containers as plain

    class Holder:
        pass

    previous = plain.TYPE_RESOLVER
    plain.set_type_resolver(lambda name: Holder if name == "ArrayHolderT<int>" else None)
    try:
        assert plain.element_type("ArrayHolderT<int>") is Holder
        with pytest.raises(TypeError, match="is not a type the std stand-ins know"):
            plain.element_type("ArrayHolderT<float>")
    finally:
        plain.set_type_resolver(previous)  # ROOT's, which pyroot installed


def test_a_std_array_is_an_element_type_of_a_vector_and_of_itself() -> None:
    """``std::array<int,3>`` where an element type is asked for: a ``vector<std::array<int,3>>``
    holds what ``std.array['int', 3]`` makes, and a ``std::array<std::array<int,3>,3>`` holds
    three of them - the declarations roottest's ``io/stdarray`` writes out and reads back.

    The size is not a type, so only the element type is resolved; a vector of the two arrays is
    the one class, since a ``std::array`` here is a vector of its size that never grows.
    """
    made = containers.std.array["int", 3]([[1, 2, 3]])
    assert isinstance(made, containers.element_type("std::array<int,3>"))  # one class, not two
    inner = containers.std.array["std::array<int,3>", 3]([[[[1, 2, 3]], [[1, 2, 3]]]])
    assert [list(each) for each in inner[:1]] == [[1, 2, 3]]
    assert len(inner) == 3  # the size comes from the declaration, not from the values given
    held = containers.std.vector["std::array<int,3>"]()
    held.push_back([1, 2, 3])
    assert list(held[0]) == [1, 2, 3]
    with pytest.raises(TypeError, match="std::array takes an element type and a size"):
        containers.element_type("std::array<int>")
    with pytest.raises(TypeError, match="std::array takes an element type and a size"):
        containers.element_type("std::array<int,N>")


def test_a_brace_enclosed_value_builds_the_element_from_its_values() -> None:
    """``a[0] = {1,2,3}`` of an element with a three-argument constructor: C++ list-initialises
    the element, so the braced values are the constructor's arguments - as a pair's are."""

    class Holder:
        def __init__(self, *values):
            self.values = values

    assert containers._converted(Holder, [1, 2, 3]).values == (1, 2, 3)
    already = Holder(4, 5, 6)
    assert containers._converted(Holder, already) is already
    pair = containers._element(containers.std.pair["int", "int"], (7, 8))
    assert (pair.first, pair.second) == (7, 8)


def test_a_hashed_multi_container_keeps_what_it_was_given_as_cpp_names_it() -> None:
    """``std::unordered_multimap`` and ``std::unordered_multiset``, which a macro's braced
    list fills: the multimap takes a key as often as it is inserted, and both are spelled
    with the name C++ gives them."""
    spreads = containers.std.unordered_multimap["std::string", "std::vector<float>"](
        [["one", [1, 2, 3]], ["one", [4, 5, 6]]]
    )
    assert spreads.__cpp_name__ == "unordered_multimap<string,vector<float>>"
    assert (len(spreads), [key for key, _ in spreads.items()]) == (2, ["one", "one"])
    assert [list(value) for _, value in spreads.items()] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert containers.std.unordered_multimap["int", "int"]().size() == 0
    counted = containers.std.unordered_multiset["int"]([3, 1, 1])
    assert counted.__cpp_name__ == "unordered_multiset<int>"
    assert (list(counted), counted.count(1)) == ([1, 1, 3], 2)


def test_a_set_keeps_a_copy_of_the_object_it_is_given() -> None:
    """C++'s ``insert`` copies, which ``TJsonEx13::Init`` leans on: it inserts one
    ``TJsonEx1`` four times, changing it in between, so the set holds four elements.
    A Python set given four times the same object would hold one."""
    containers._opaque("TJsonEx1")
    thing = _a_numbered_thing()
    held = containers.std.set["TJsonEx1"]()
    for number in range(3):
        thing.fInt = number
        held.insert(thing)
    assert [each.fInt for each in held] == [0, 1, 2]


def test_a_multiset_keeps_a_copy_of_the_object_it_is_given() -> None:
    """``multiset::insert`` copies as ``set``'s does, and counts the copies it holds."""
    thing = _a_numbered_thing()
    held = containers.std.multiset["TJsonEx1"]()
    held.insert(thing)
    thing.fInt = 7
    held.insert(thing)
    assert [each.fInt for each in held] == [0, 7]


def test_a_map_keeps_copies_of_its_keys_and_values() -> None:
    """``m[k] = v`` copies both in C++: the key stays as it was when it was put in, and a
    value that is a container is a container of its own."""
    thing = _a_numbered_thing()
    held = containers.std.map["TJsonEx1", "int"]()
    for number in range(2):
        thing.fInt = number
        held[thing] = number
    assert [(key.fInt, one) for key, one in held.items()] == [(0, 0), (1, 1)]
    nested = containers.std.map["int", "std::set<int>"]()
    growing = containers.std.set["int"]()
    nested[1] = growing
    growing.insert(2)  # the map kept a set of its own, so this is not its element
    assert [list(one) for one in nested.values()] == [[]]


def test_a_set_of_pointers_holds_the_object_itself() -> None:
    """A ``std::set<TBox*>`` copies the pointer, which here is the object, so what the
    caller goes on doing to it is seen through the set - ROOT's own behaviour for one."""
    thing = _a_numbered_thing()
    held = containers.std.set["TJsonEx1*"]()
    held.insert(thing)
    thing.fInt = 4
    assert [each.fInt for each in held] == [4]


def test_a_set_of_numbers_and_strings_holds_a_value_that_cannot_be_changed() -> None:
    """A number and a ``std::string`` are already values: nothing to copy."""
    held = containers.std.set["int"]()
    held.insert(3)
    text = containers.std.string("a")
    words = containers.std.set["std::string"]()
    words.insert(text)
    assert (list(held), [str(each) for each in words]) == ([3], ["a"])


def test_two_sets_are_equal_when_they_hold_the_same_elements() -> None:
    """``a == b``, which ``EXPECT_EQ`` and a macro's ``==`` both mean: what is held, and how
    often - never the order a hashed container is free not to keep."""
    numbers = containers.std.unordered_set["double"]([1, 3, 5, 6, 8])
    assert numbers == containers.std.unordered_set["double"]([8, 6, 5, 3, 1])
    assert numbers != containers.std.unordered_set["double"]([1, 3])
    assert numbers != 5  # a set is not equal to something that is not one
    assert containers.std.multiset["int"]([1, 1, 2]) == containers.std.multiset["int"]([2, 1, 1])
    assert containers.std.multiset["int"]([1, 1, 2]) != containers.std.multiset["int"]([1, 2])
    assert containers.std.vector["int"]([1]) == containers.std.vector["int"]([1])


def test_two_maps_are_equal_when_they_hold_the_same_pairs() -> None:
    """A map's ``==`` by its keys and their values, and a multimap's by its pairs however
    the hashed order it does not promise happened to put them in."""
    eggs = containers.std.unordered_map["int", "std::string"]([[2, "two"]])
    assert eggs == containers.std.unordered_map["int", "std::string"]([[2, "two"]])
    assert eggs != containers.std.unordered_map["int", "std::string"]([[4, "four"]])
    assert eggs != 3
    pairs = containers.std.unordered_multimap["std::string", "int"](
        [["one", 1], ["one", 2], ["two", 3]]
    )
    assert pairs == containers.std.unordered_multimap["std::string", "int"](
        [["two", 3], ["one", 2], ["one", 1]]
    )
    assert pairs != containers.std.unordered_multimap["std::string", "int"]([["one", 1]])
    assert pairs != containers.std.unordered_multimap["std::string", "int"](
        [["one", 1], ["one", 2], ["two", 4]]
    )
    assert pairs != 3  # and it is not equal to something that is not a multimap
