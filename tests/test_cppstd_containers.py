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

    containers.set_type_resolver(resolver)
    values = containers.std.vector["CppstdValueFixture"]()
    values.emplace_back(7, 9)
    assert values[0].pair == (7, 9)
    assert seen == ["CppstdValueFixture"]
    assert containers._resolved_class(np.dtype("int32")) is None
    containers.set_type_resolver(None)
    assert containers._resolved_class(containers._opaque("NoAdapterFixture")) is None


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
                                     'unordered_map', 'unordered_set'}
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
