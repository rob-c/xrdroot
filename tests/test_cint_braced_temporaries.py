"""``T{a, b}``: a temporary built from a braced list - a container's items, a class's arguments."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from cintfake import fake
from xrdroot.pycint import translate
from xrdroot.pycint.execute import run_source


def expression(source: str) -> str:
    text = translate("void t(int b) {\nauto v = " + source + ";\n}\n", "t.C")
    compile(text, "t.C", "exec")
    return text


def test_a_braced_container_temporary_holds_its_items_rather_than_sizing_itself():
    assert "([b, b * 10])" in expression("ROOT::VecOps::RVec<int>{ b, b * 10 }")
    assert "([1.5])" in expression("std::vector<double>{1.5}")


def test_a_braced_class_temporary_still_takes_its_items_as_constructor_arguments():
    assert "TVector2(1, b)" in expression("TVector2{1, b}")


def test_a_braced_rvec_in_a_define_holds_its_items_per_entry():
    frame = ROOT.RDataFrame(3).Define("b", "(int) rdfentry_")
    found = frame.Define("v", "ROOT::VecOps::RVec<int>{ b, b * 10 }").AsNumpy(["v"])["v"]
    assert [list(v) for v in found] == [[0, 0], [1, 10], [2, 20]]


def test_a_std_string_joined_to_an_integer_writes_it_in_decimal_as_tstring_does():
    text = expression('std::string("a[") + b + "]"')
    assert "str('a[') + str(b) + ']'" in text
    assert "+ str(" not in expression('std::string("a") + \'c\'')
    assert "+ str(" not in expression("std::string(\"a\") + true")


def test_a_braced_hashed_container_holds_its_items_rather_than_taking_them_as_arguments():
    """``std::unordered_set<double> s{1, 3, 5}`` is C++'s ``initializer_list`` constructor,
    which every container has however it orders its elements - so the items go in as a
    list, and six of them are six elements and not six arguments."""
    assert "([1.5])" in expression("std::unordered_set<double>{1.5}")
    assert "'int']([b, b])" in expression("std::unordered_multiset<int>{b, b}")
    assert "([['one', [1, 2, 3]]])" in expression(
        'std::unordered_multimap<std::string, std::vector<float>>{{"one", {1, 2, 3}}}'
    )
    assert "'float']([1.5])" in expression("std::multiset<float>{1.5}")
    assert "'int']([b])" in expression("std::forward_list<int>{b}")


def test_a_braced_list_handed_to_a_container_parameter_is_that_container():
    """``void take(std::vector<double> v)`` called with ``{1.0, 2.0}`` is C++'s
    ``initializer_list`` constructor for the parameter, so what arrives is a vector."""
    text = translate(
        'void take(std::vector<double> v) { printf("%zu\\n", v.size()); }\n'
        "void t() { take({1.0, 2.0}); }\n",
        "t.C",
    )
    assert "if isinstance(v, list): v = ROOT.std.vector['double'](v)" in text


def test_a_lambda_converts_a_braced_list_the_same_way_its_parameter_is_declared(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = (
        "void t() {\n"
        "  auto count = [](std::vector<double> v) { return (int)v.size(); };\n"
        '  printf("%d %d\\n", count({1.0, 2.0}), count(std::vector<double>{1.0, 2.0, 3.0}));\n'
        "}\n"
    )
    run_source(source, "t.C", root=fake())
    assert capsys.readouterr().out == "2 3\n"
