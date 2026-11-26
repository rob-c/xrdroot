"""``TPython`` and ``std::any``: the C++ side running Python, and the value it gets back."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT


@pytest.fixture(autouse=True)
def one_namespace_per_test():
    """Each test starts from the session's namespace as a fresh process has it."""
    kept = dict(ROOT.TPython.namespace)
    yield
    ROOT.TPython.namespace.clear()
    ROOT.TPython.namespace.update(kept)


def test_an_exec_sets_a_name_the_next_exec_sees():
    ROOT.TPython.Exec("made_here = 41")
    ROOT.TPython.Exec("made_here += 1")
    assert ROOT.TPython.Eval("made_here") == 42


def test_an_exec_that_leaves_anyresult_fills_the_any_the_cpp_side_handed():
    """ROOT's ``TPython::Exec(cmd, &out)``: the Python value arrives as the any's."""
    written = ROOT.std.any()
    ROOT.TPython.Exec("_anyresult = ROOT.std.make_any['int'](100)", written)
    assert written.has_value() and ROOT.std.any_cast["int"](written) == 100
    assert written.kind == "int"


def test_an_exec_that_leaves_no_anyresult_leaves_the_any_as_it_was():
    written = ROOT.std.make_any["int"](7)
    ROOT.TPython.Exec("nothing_left_behind = 1", written)
    ROOT.TPython.Exec("nothing_left_behind = 2")  # no any to write into: nothing to do
    assert ROOT.std.any_cast["int"](written) == 7


def test_a_plain_python_value_arrives_with_the_type_it_has():
    written = ROOT.std.any()
    ROOT.TPython.Exec("_anyresult = 'text'", written)
    assert ROOT.std.any_cast["string"](written) == "text"
    assert written.type().__cpp_name__ == "string"
    other = ROOT.std.any()
    ROOT.TPython.Exec("_anyresult = 1.5", other)
    assert ROOT.std.any_cast["double"](other) == 1.5


def test_a_macro_is_a_file_run_in_the_same_namespace(tmp_path):
    macro = tmp_path / "with'quote.py"
    macro.write_text("tpython_loadmacro_marker = 11\n", encoding="utf-8")
    ROOT.TPython.Exec("tpython_loadmacro_marker = 0")
    ROOT.TPython.LoadMacro(macro)
    assert ROOT.TPython.Eval("tpython_loadmacro_marker") == 11
    other = tmp_path / "second.py"
    other.write_text("tpython_loadmacro_marker = 22\n", encoding="utf-8")
    ROOT.TPython.ExecScript(other)
    assert ROOT.TPython.Eval("tpython_loadmacro_marker") == 22


def test_the_session_namespace_has_root_in_it_as_roots_python_does():
    assert ROOT.TPython.Eval("ROOT") is ROOT


def test_an_any_holds_one_value_with_the_type_it_was_made_as():
    made = ROOT.std.make_any["int"](100)
    assert made.has_value() and bool(made) and made.value() == 100
    assert made.type().__cpp_name__ == "int" and repr(made) == "<std::any int 100>"
    made.reset()
    assert not made.has_value() and not made and made.type().__cpp_name__ == "void"
    assert repr(made) == "<std::any empty>"
    with pytest.raises(RuntimeError, match="holds nothing"):
        made.value()


def test_a_cast_of_a_value_that_is_not_an_any_converts_what_it_is_given():
    """``std::any_cast<int>(a)`` converts, so one made here from a number reads back as one."""
    assert ROOT.std.any_cast["int"](ROOT.std.make_any["double"](2.5)) == 2
    assert ROOT.std.any_cast["string"]("text") == "text"
    held = ROOT.std.make_any["vector<int]"]([1, 2])
    assert ROOT.std.any_cast["vector<int]"](held) == [1, 2]  # a type with no converter


def test_the_kind_a_value_carries_is_its_cpp_type_spelled_out():
    """``std::any::type()``'s name for each Python kind - bool before int, since bool is one."""
    from xrdroot.cppstd.stdlib import AnyValue, kind_of

    assert [kind_of(one) for one in (True, 3, 2.5, "t", [1])] == [
        "bool",
        "int",
        "double",
        "string",
        "list",
    ]
    assert kind_of(AnyValue(3, "int")) == "int" and kind_of(AnyValue()) == "void"
    made = AnyValue()
    made.set([1, 2])
    assert made.kind == "list" and made.held == [1, 2]  # set with no kind: the value's own
