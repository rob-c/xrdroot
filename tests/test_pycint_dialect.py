"""ROOT's dialect of the translator, and the names the translator had before it was split.

:mod:`xrdroot.cint` is kept so that every import of it keeps working: each of
its names is :mod:`xrdroot.pycint`'s very object. A translation kept from
before imports ``xrdroot.cint.runtime``, which is the ROOT dialect's runtime.
"""

from __future__ import annotations

import pytest

import xrdroot.cint as former
import xrdroot.cint.execute as former_execute
import xrdroot.cint.runtime as former_runtime
import xrdroot.pycint as pycint
from xrdroot import plugins
from xrdroot.pycint import execute, runtime
from xrdroot.pycint.dialect import ROOT
from xrdroot.pycxx import CXX, CxxError, translate
from xrdroot.pycxx.ctype import canonical
from xrdroot.pycxx.dialect import using


@pytest.mark.parametrize("name", former.__all__)
def test_every_former_name_is_the_interpreters_own_object(name: str) -> None:
    assert getattr(former, name) is getattr(pycint, name)


@pytest.mark.parametrize("name", former_execute.__all__)
def test_every_former_runner_is_the_interpreters_own(name: str) -> None:
    assert getattr(former_execute, name) is getattr(execute, name)


def test_a_kept_translation_imports_the_ROOT_dialects_runtime() -> None:
    assert former_runtime.__all__ == runtime.__all__
    assert all(getattr(former_runtime, name) is getattr(runtime, name) for name in runtime.__all__)
    assert "CXX" not in runtime.__all__ and "ROOT" in runtime.__all__


def test_the_translators_errors_are_the_cint_errors() -> None:
    assert pycint.CintError is CxxError and issubclass(pycint.Refusal, CxxError)


def test_roots_dialect_is_what_the_namespace_reaches_through_its_provider() -> None:
    assert plugins.require("interpreter.dialect")() is ROOT
    assert plugins.require("interpreter.run_source") is execute.run_source


def test_roots_typedefs_are_known_in_its_dialect_alone() -> None:
    assert canonical("Double32_t") == "Double32_t"
    with using(ROOT):
        assert canonical("Double32_t") == "double"
    assert ROOT.typedefs["Long64_t"] == "long long" and not CXX.typedefs


def test_a_strings_data_is_the_string_in_roots_dialect() -> None:
    python = pycint.translate('void f() { std::string s = "a"; const char *c = s.Data(); }')
    assert "c = s\n" in python
    with pytest.raises(pycint.Refusal, match=r"std::string::Data\(\)"):
        translate('void f() { std::string s = "a"; const char *c = s.Data(); }')
