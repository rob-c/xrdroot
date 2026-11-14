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
        assert loading.declare_namespaces("namespace NotASpace { int z; }") == {}
        assert loading.declare_namespaces("namespace std { int z; }") == {}
        assert loading.declare_namespaces("int nothing;") == {}
    finally:
        del DECLARED["NotASpace"]


@pytest.mark.parametrize("source", ["namespace Empty {}"])
def test_an_empty_namespace_is_an_empty_object(source):
    made = loading.declare_namespaces(source)
    assert list(made) == ["Empty"] and isinstance(made["Empty"], loading.CxxNamespace)
