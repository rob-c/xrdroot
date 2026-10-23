"""Generic containers remain usable before ROOT explicitly installs its adapters."""

from __future__ import annotations

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
assert set(containers.TEMPLATES) == {'vector', 'map', 'pair'}
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
