"""Operators a macro defines outside a class: ``a == b`` calls the one their types choose."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.pycint import Refusal, translate
from xrdroot.pycint.execute import run_source
from xrdroot.pycxx.ctype import CType
from xrdroot.pycxx.errors import Where
from xrdroot.pycxx.freeops import FreeOperator, choose
from xrdroot.pycxx.nodes import Function, Param

VECTORS = """
#include <iostream>
struct V { double x, y; V(double a = 0, double b = 0) : x(a), y(b) {} };
struct W : public V { W(double a) : V(a, a) {} };
V operator+(const V &a, const V &b) { return V(a.x + b.x, a.y + b.y); }
V operator*(double k, const V &v) { return V(k * v.x, k * v.y); }
V operator*(const V &v, int k) { return V(v.x, 100 * k * v.y); }
V operator-(const V &v) { return V(-v.x, -v.y); }
bool operator!(const V &v) { return v.x == 0 && v.y == 0; }
bool operator==(const V &a, const V &b) { return a.x == b.x && a.y == b.y; }
"""


def output(capsys: pytest.CaptureFixture[str], source: str) -> str:
    run_source(source, "t.C", root=fake())
    return capsys.readouterr().out


def test_arithmetic_on_a_macros_objects_calls_the_operators_it_defines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = VECTORS + """
    void t() {
      V a(1, 2), b(3, 4);
      V c = a + b, d = 2.0 * a, e = a * 2, f = -a;
      W w(5);
      V g = a + w;
      std::cout << c.x << " " << c.y << " " << d.y << " " << e.y << " " << f.x << " " << g.y;
      std::cout << " " << (a + b + b).x << " " << (1 + 2) << " " << -3 << std::endl;
    }
    """
    text = translate(source, "t.C")
    assert "c = operator_add(a, b)" in text
    assert "d = operator_mul__1(2.0, a)" in text
    assert "e = operator_mul__2(a, 2)" in text
    assert "operator_add(operator_add(a, b), b).x" in text
    assert output(capsys, source) == "4 6 4 400 -1 7 7 3 -3\n"


def test_comparisons_and_negation_are_the_macros_where_an_operand_is_its_object(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = VECTORS + """
    void t() {
      V a(1, 2), b(3, 4);
      std::cout << (a == a) << (a == b) << !a << !V() << (1 == 1) << !0 << std::endl;
      if (!V()) std::cout << "zero" << std::endl;
      if (!(a == b)) std::cout << "differ" << std::endl;
    }
    """
    assert output(capsys, source) == "100111\nzero\ndiffer\n"


def test_an_operator_in_a_namespace_is_called_for_the_namespaces_class(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    namespace geo {
      struct P { int id; };
      bool operator==(const P &a, const P &b) { return a.id == b.id; }
      bool operator<(const P &a, const P &b) { return a.id < b.id; }
    }
    void t() {
      geo::P p{1}, q{2};
      printf("%d %d %d\\n", p == q, p < q, q < p);
    }
    """
    assert output(capsys, source) == "0 1 0\n"


def test_a_stream_extraction_writes_through_the_reference_it_is_given(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    struct Buf { int n = 0; };
    Buf &operator<<(Buf &b, int k) { b.n += k; return b; }
    Buf &operator>>(Buf &b, int &k) { k = b.n; return b; }
    void t() {
      Buf buf; int got = 0;
      buf << 3 << 4;
      buf >> got;
      printf("%d %d\\n", got, 7 >> 1);
    }
    """
    text = translate(source, "t.C")
    assert "got = Cell(0, 'int')" in text
    assert "operator_rshift(buf, got)" in text
    assert output(capsys, source) == "7 3\n"


def test_a_template_operator_takes_any_pointer_and_a_named_template_is_preferred(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    template <class T> class Array {};
    class Buffer {};
    template <class Tmpl> Buffer &operator>>(Buffer &buf, Tmpl *&obj) {
      printf("Generic operator>>\\n"); return buf;
    }
    template <class T> Buffer &operator>>(Buffer &buf, Array<T> *&obj) {
      printf("Array operator>>\\n"); return buf;
    }
    void t() { Buffer b; int *i; Array<int> *a; b >> i; b >> a; }
    """
    assert output(capsys, source) == "Generic operator>>\nArray operator>>\n"


def test_an_operator_returning_auto_leaves_its_result_untyped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    struct N { int v; };
    auto operator+(const N &a, const N &b) { return a.v + b.v; }
    void t() { N a{2}, b{3}; int s = (a + b) / 2; printf("%d\\n", s); }
    """
    assert "s = int(div(operator_add(a, b), 2))" in translate(source, "t.C")
    assert output(capsys, source) == "2\n"


def test_an_enums_operator_is_called_for_the_enum_and_not_for_a_number(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    enum Flag { kA = 1, kB = 2 };
    int operator|(Flag a, Flag b) { return 10 * a + b; }
    void t() { Flag a = kA, b = kB; printf("%d %d\\n", a | b, 1 | 2); }
    """
    assert output(capsys, source) == "12 3\n"


def test_a_number_is_converted_and_a_string_matched_for_an_operators_parameters(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    struct S { int n; };
    int operator+(const S &s, double d) { return s.n + 100; }
    int operator-(const S &s, const char *t) { return s.n - 1; }
    void t() { S s{1}; S *p = &s; printf("%d %d %d\\n", s + 1, s - "x", p == nullptr); }
    """
    assert output(capsys, source) == "101 0 0\n"


def test_an_operator_with_no_python_function_is_refused_where_it_is_defined() -> None:
    with pytest.raises(Refusal, match="the operator && defined outside a class"):
        translate("struct V {};\nbool operator&&(V a, V b) { return true; }", "t.C")
    with pytest.raises(Refusal, match="the operator \\+ defined outside a class"):
        translate("struct V {};\nV operator+(V a, V b, V c) { return a; }", "t.C")


def _operator(*params: CType, template: list[str] | None = None) -> FreeOperator:
    where = Where("t.C", 1)
    func = Function(where, "operator==", CType("bool"), [Param(where, "p", p) for p in params],
                    None, template=template)
    return FreeOperator(func, "operator_eq")


A, A_PTR, INT, INT_PTR = CType("A"), CType("A", pointer=1), CType("int"), CType("int", pointer=1)
#: Operators to choose from: two objects, two numbers, a template's pointer, a stream's pointer.
EXACT = _operator(A, A)
NUMBERS = _operator(INT, INT)
GENERIC = _operator(CType("T", pointer=1), A, template=["T"])
POINTER = _operator(CType("ostream"), A_PTR)


@pytest.mark.parametrize(
    ("candidates", "operands", "chosen"),
    [
        ([EXACT], [A, A], EXACT),
        ([EXACT], [A], None),
        ([EXACT], [CType("B"), A], None),
        ([EXACT], [None, None], None),
        ([EXACT], [A_PTR, A], None),
        ([NUMBERS], [INT, INT], None),
        ([GENERIC], [INT_PTR, A], GENERIC),
        ([GENERIC], [None, A], GENERIC),
        ([GENERIC], [INT, A], None),
        ([POINTER], [None, A_PTR], POINTER),
        ([POINTER], [CType("ostream"), None], None),
        ([EXACT, GENERIC], [INT_PTR, A], GENERIC),
    ],
)
def test_overload_resolution_needs_an_object_and_prefers_the_closest_fit(
    candidates: list[FreeOperator], operands: list[CType | None], chosen: FreeOperator | None
) -> None:
    assert choose(candidates, operands, {}) is chosen


def test_a_derived_class_fits_its_base_after_an_exact_match() -> None:
    source = """
    struct A {}; struct B : public A {}; struct C : public B {}; struct D {};
    int operator==(const A &x, const A &y) { return 1; }
    int operator==(const B &x, const B &y) { return 2; }
    void t() { A a; B b; C c; D d; printf("%d %d %d %d\\n", a == a, b == b, c == c, a == c); }
    """
    assert run_source(source, "t.C", root=fake()) is None
    text = translate(source, "t.C")
    assert "operator_eq__1(a, a), operator_eq__2(b, b), operator_eq__2(c, c)" in text
    assert "operator_eq__1(a, c)" in text
