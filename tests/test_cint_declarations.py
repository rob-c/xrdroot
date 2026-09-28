"""Declared variables: what each starts as, and the declarations refused by name."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.cint import Refusal, translate
from xrdroot.cint.execute import run_source


def body(source: str) -> str:
    text = translate("void t() {\n" + source + "\n}\n", "t.C")
    compile(text, "t.C", "exec")
    return text


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ('std::string name = "n"; auto *s = name;', "s = name"),
        ("int &r = unknown; r = 2;", "ROOT.unknown = 2"),
        ("std::vector<double> v; double &e = v[f()];", "e = v[ROOT.f()]"),
        ("std::function<double(double)> f;", "f = None"),
        ("std::string s{};", "s = ''"),
        ("std::vector<int> v{1, 2};", "v = ROOT.std.vector['int']([1, 2])"),
        ("std::string s(3, 'x');", "s = chr(ord('x')) * 3"),
        ("int x = {}; double d = {2};", "x = 0"),
        ("double d = {2};", "d = 2.0"),
        ("std::vector<int> v = {1, 2};", "v = ROOT.std.vector['int']([1, 2])"),
        ("TLorentzVector v[2];", "v = array('TLorentzVector', 2, make=ROOT.TLorentzVector)"),
        (
            'char names[2][8] = {"a", "b"}; char more[3][4];',
            "names = array('char*', 2, ['a', 'b'])",
        ),
        ("char more[3][4];", "more = array('char*', 3, [])"),
        ('char s[] = "abc"; int n = sizeof(s);', "s = 'abc'"),
        ("double a[3]; void *p = &a;", "a = array('double', 3)"),
        ("Point p = {1, 2};", "p = ROOT.Point(1, 2)"),
    ],
)
def test_a_declaration_starts_its_variable_as_cpp_does(source: str, fragment: str) -> None:
    assert fragment in body(source)


@pytest.mark.parametrize(
    ("source", "why"),
    [
        ("auto [a, b];", "a structured binding with nothing to unpack"),
        ("auto x;", "auto x with nothing to take its type from"),
        ("int a[];", r"the array a\[\] with no size to give it"),
        ("char s[4] = {'a', 'b'};", "the character array s initialised one char at a time"),
    ],
)
def test_declarations_with_no_python_that_does_the_same_are_refused(source: str, why: str) -> None:
    with pytest.raises(Refusal, match=why):
        body(source)


def test_a_template_parameter_type_cannot_be_built_from_nothing() -> None:
    with pytest.raises(Refusal, match="a variable of the template parameter type T"):
        translate("template <typename T> T zero() { T x; return x; }", "t.C")


def test_a_static_local_keeps_its_value_between_calls(capsys: pytest.CaptureFixture[str]) -> None:
    source = """
    int next() { static int n = 10; return n++; }
    void t() { next(); next(); printf("%d\\n", next()); }
    """
    run_source(source, "t.C", root=fake())
    assert capsys.readouterr().out == "12\n"
