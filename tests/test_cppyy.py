"""``import cppyy`` over xrdroot's translator: ``cppdef``, ``gbl``, pointers and the low level."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

import xrdroot.cppyy as cppyy
import xrdroot.pyroot as ROOT
from xrdroot.cppyy import ll, py
from xrdroot.pyroot.core import troot


@pytest.fixture(autouse=True)
def _fresh_line_processor():
    """``ProcessLine`` finds the translator afresh after each test, as it would in a new run."""
    yield
    troot.set_line_processor(None)


def test_cpp_defined_is_found_on_the_global_namespace_as_on_roots():
    assert cppyy.cppdef("struct CppyyPoint { int x = 2; int twice() { return 2 * x; } };")
    point = cppyy.gbl.CppyyPoint()
    assert point.twice() == 4 and cppyy.gbl.CppyyPoint is ROOT.CppyyPoint
    assert cppyy.gbl.std.vector["int"] is ROOT.std.vector["int"] and cppyy.gbl.TH1F is ROOT.TH1F
    assert repr(cppyy.gbl) == "<namespace cppyy.gbl at 0x0>"


def test_a_name_cpp_has_not_declared_is_an_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute 'NoSuchThing'"):
        cppyy.gbl.NoSuchThing  # noqa: B018
    with pytest.raises(AttributeError):
        cppyy.gbl.__wrapped__  # noqa: B018


def test_the_global_namespace_is_a_module_names_are_imported_from():
    assert importlib.import_module("xrdroot.cppyy.gbl") is cppyy.gbl
    assert sys.modules["xrdroot.cppyy.gbl"].std is ROOT.std


def test_cpp_that_will_not_translate_is_a_syntax_error():
    with pytest.raises(SyntaxError, match="Failed to parse"):
        cppyy.cppdef("this is not C++ at all {")
    with pytest.raises(SyntaxError, match="Failed to parse"):
        cppyy.cppexec("int x = ;")


def test_statements_run_at_once(capsys):
    assert cppyy.cppexec('printf("ran\\n");')
    assert capsys.readouterr().out == "ran\n"


def test_a_header_is_included_from_a_directory_added_to_the_path(tmp_path, monkeypatch):
    monkeypatch.setattr(cppyy, "INCLUDE_PATHS", [])
    monkeypatch.setattr(ROOT.gSystem, "_include_path", ROOT.gSystem.GetIncludePath())
    (tmp_path / "cppyy_header.h").write_text("int cppyy_from_header() { return 7; }\n")
    cppyy.add_include_path(str(tmp_path))
    assert cppyy.include("cppyy_header.h") and cppyy.c_include is cppyy.include
    assert cppyy.gbl.cppyy_from_header() == 7
    (tmp_path / "cppyy_broken.h").write_text("this is not C++ at all {" + chr(10))
    with pytest.raises(ImportError, match=r"cppyy_broken\.h"):
        cppyy.include("cppyy_broken.h")


def test_a_library_that_is_not_there_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(ROOT.gSystem, "_dynamic_path", ROOT.gSystem.GetDynamicPath())
    cppyy.add_library_path(str(tmp_path))
    with pytest.raises(RuntimeError, match="libCppyyNothing"):
        cppyy.load_library("libCppyyNothing")
    assert cppyy.load_reflection_info("libCore")


def test_an_address_binds_back_to_its_object_and_a_null_one_to_a_false_object():
    owner = ROOT.TNamed("n", "t")
    address = cppyy.addressof(owner)
    assert cppyy.bind_object(address, "TNamed") is owner
    assert cppyy.bind_object(owner, ROOT.TNamed) is owner
    null = cppyy.bind_object(cppyy.nullptr, "TObject")
    assert not null and isinstance(null, ROOT.TObject) and repr(null).endswith("at 0x0>")
    assert null == cppyy.nullptr and not (null != 0) and cppyy.addressof(null) == 0
    with pytest.raises(TypeError, match="not compared with None"):
        _ = null == None  # noqa: E711
    with pytest.raises(TypeError, match="not compared with None"):
        _ = null != None  # noqa: E711
    assert type(cppyy.bind_object(0, "TObject")) is type(null)


def test_an_address_nothing_was_taken_of_is_refused():
    with pytest.raises(TypeError, match=r"no object is known at address 0x1"):
        cppyy.bind_object(1, "TObject")


def test_a_member_address_is_a_cell_and_objects_without_weak_references_are_kept():
    owner = ROOT.TNamed("n", "t")
    assert cppyy.addressof(owner, "_name").value == "n"
    plain = [1]
    assert cppyy.bind_object(cppyy.addressof(plain), "TObject") is plain


def test_nullptr_is_one_false_object_printed_as_cpp_spells_it():
    assert cppyy.nullptr is type(cppyy.nullptr)() and not cppyy.nullptr
    assert repr(cppyy.nullptr) == "nullptr" and ll.nullptr is cppyy.nullptr


def test_a_class_in_a_namespace_is_bound_by_its_qualified_name():
    cppyy.cppdef("namespace CppyyNS { struct Inner {}; }")
    assert type(cppyy.bind_object(0, "CppyyNS::Inner")).__mro__[1] is cppyy.gbl.CppyyNS.Inner


def test_sizes_are_those_of_cpps_fundamental_types():
    assert (cppyy.sizeof("int"), cppyy.sizeof("double"), cppyy.sizeof("char")) == (4, 8, 1)
    assert cppyy.sizeof(ROOT.TObject) == 8 and cppyy.sizeof("std::string") == 8
    assert cppyy.sizeof("NotAType") == 8
    cppyy.set_debug()


def test_a_low_level_cast_converts_numbers_as_cpp_does_and_leaves_objects_be():
    assert ll.cast["int"](3.7) == 3 and ll.static_cast["unsigned char"](300) == 44
    owner = ROOT.TObject()
    assert ll.reinterpret_cast[ROOT.TObject](owner) is owner
    assert ll.dynamic_cast["TObject"](cppyy.addressof(owner)) is owner


def test_low_level_arrays_and_signals_are_what_python_has():
    made = ll.array_new("int", 3)
    assert made.dtype == np.int32 and list(made) == [0, 0, 0]
    assert ll.array_new("NotAType", 1).dtype == object
    assert ll.array_new("vector<int", 1).dtype == object  # no type C++ could spell
    ll.array_delete(made)
    assert ll.set_signals_as_exception(True)
    with ll.signals_as_exception():
        pass
    assert issubclass(ll.SegmentationViolation, ll.FatalError)
    assert issubclass(ll.BusError, ll.FatalError) and issubclass(ll.AbortSignal, ll.FatalError)
    assert issubclass(ll.IllegalInstruction, ll.FatalError)


def test_a_pythonization_runs_once_on_each_class_looked_up_after_it_is_added():
    cppyy.cppdef("struct CppyyPythonized {}; struct CppyyOther {};")
    seen = []

    def callback(klass, name):
        seen.append(name)
        klass.added = True

    py.add_pythonization(callback)
    try:
        assert cppyy.gbl.CppyyPythonized.added and cppyy.gbl.CppyyPythonized.added
        other = cppyy.gbl.CppyyOther
        assert py.pythonized(other, "CppyyOther", "Elsewhere") is other
    finally:
        assert py.remove_pythonization(callback)
    assert not py.remove_pythonization(callback)
    assert seen == ["CppyyPythonized", "CppyyOther"]
    py.pin_type(cppyy.gbl.CppyyOther)


def test_a_composed_method_is_given_the_original_to_call():
    cppyy.cppdef("struct CppyyComposed { int value() { return 2; } };")
    klass = cppyy.gbl.CppyyComposed
    py.compose_method(klass, "value", lambda self, original: 10 * original(self))
    assert klass().value() == 20
