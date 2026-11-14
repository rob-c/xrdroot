"""Namespaces a ``Declare`` or a ``ProcessLine`` declares are objects, as a library's are."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.pycint.library import loading
from xrdroot.pycxx.runtime.namespace import DECLARED
from xrdroot.pyroot.core import troot


@pytest.fixture(autouse=True)
def _fresh_line_processor():
    """``ProcessLine`` finds the translator afresh after each test, as it would in a new run."""
    yield
    troot.set_line_processor(None)


def test_a_declared_namespace_holds_its_classes_functions_and_inner_namespaces():
    ROOT.gInterpreter.Declare(
        "namespace DeclNS { struct A { int get() { return 3; } };"
        " namespace In { int g() { return 2; } } }"
    )
    assert ROOT.DeclNS.A().get() == 3 and ROOT.DeclNS.In.g() == 2
    assert repr(ROOT.DeclNS) == "<namespace DeclNS>"


def test_a_line_declaring_a_namespace_adds_to_it_and_later_cpp_finds_it(capsys):
    ROOT.gInterpreter.Declare("namespace LineNS { int one() { return 1; } }")
    ROOT.gInterpreter.ProcessLine("namespace LineNS { class Q {}; }")
    assert ROOT.LineNS.one() == 1 and isinstance(ROOT.LineNS.Q(), ROOT.LineNS.Q)
    ROOT.gInterpreter.ProcessLine('printf("%d\\n", LineNS::one());')
    assert capsys.readouterr().out == "1\n"


def test_a_line_declaring_a_class_declares_it_at_the_top_as_cling_does():
    ROOT.gInterpreter.ProcessLine("class TopLineClass { };")
    assert DECLARED["TopLineClass"].__name__ == "TopLineClass"


def test_a_line_declaring_a_class_then_using_it_runs_as_a_block(capsys):
    ROOT.gInterpreter.ProcessLine(
        'struct BlockLine { int x = 4; }; BlockLine b; b.x = 5; printf("%d\\n", b.x);'
    )
    assert capsys.readouterr().out == "5\n"


def test_std_and_names_already_meaning_something_else_are_not_made_namespaces():
    DECLARED["NotASpace"] = 5
    try:
        assert loading.declare_namespaces("namespace NotASpace { int z; }", {})[0] == {}
        assert loading.declare_namespaces("namespace std { int z; }", {})[0] == {}
        assert loading.declare_namespaces("int nothing;", {}) == ({}, set())
    finally:
        del DECLARED["NotASpace"]


@pytest.mark.parametrize("source", ["namespace Empty {}"])
def test_an_empty_namespace_is_an_empty_object(source):
    made, hidden = loading.declare_namespaces(source, {})
    assert list(made) == ["Empty"] and isinstance(made["Empty"], loading.CxxNamespace)
    assert hidden == set()


def test_a_namespaces_class_named_as_one_at_the_top_leaves_the_top_ones_name_be():
    ROOT.gInterpreter.ProcessLine("class Collides { public: int top() { return 1; } };")
    ROOT.gInterpreter.ProcessLine(
        "namespace Outer { class Collides { public: int in() { return 2; } }; }"
    )
    ROOT.gInterpreter.Declare("namespace Outer { namespace Deep { class Collides {}; } }")
    assert ROOT.Collides().top() == 1 and ROOT.Outer.Collides().in_() == 2
    assert ROOT.Outer.Collides.__cpp_name__ == "Outer::Collides"
    assert ROOT.Outer.Deep.Collides.__cpp_name__ == "Outer::Deep::Collides"


def test_a_macro_run_declares_its_namespaces_too(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "spaced.C").write_text(
        "namespace MacroNS { int seven() { return 7; } }\n"
        'void spaced() { printf("%d\\n", MacroNS::seven()); }\n'
    )
    ROOT.gROOT.ProcessLine(".x spaced.C")
    assert capsys.readouterr().out == "7\n" and ROOT.MacroNS.seven() == 7


def test_a_line_declaring_one_variable_makes_it_a_global_of_the_session(capsys):
    ROOT.gInterpreter.ProcessLine("int session_count = 3;")
    assert ROOT.session_count == 3
    ROOT.gInterpreter.ProcessLine('printf("%d\\n", session_count + 1);')
    assert capsys.readouterr().out == "4\n"


def test_a_variable_declared_into_a_namespace_given_is_set_there():
    from types import SimpleNamespace

    from xrdroot.pycint.execute import process_line

    space = SimpleNamespace()
    process_line("double given_here = 2.5;", root=ROOT)
    process_line("int other_here = 1;", root=space)
    assert ROOT.given_here == 2.5 and space.other_here == 1


def test_a_line_shaped_like_a_declaration_that_is_none_runs_as_a_statement(capsys):
    ROOT.gInterpreter.ProcessLine("int shown_here = 5;")
    ROOT.gInterpreter.ProcessLine("std::cout << shown_here;")
    assert capsys.readouterr().out == "5"
