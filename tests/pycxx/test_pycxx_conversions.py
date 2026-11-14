"""Values made as C++ makes them: a class converted to a number - ``operator unsigned
int()``, ``operator double()`` - and a lambda's capture of a value, ``[n = expr]``.

The expected output is what ROOT 6.40.04's cling prints for the same program.
"""

from __future__ import annotations

import pytest

from xrdroot.cppstd.objects import UNGIVEN, convertible
from xrdroot.pycxx import Refusal, run_source, translate
from xrdroot.pycxx.operators import conversion

PROGRAM = r"""
#include <cstdio>
typedef unsigned int id_type;
struct Item { id_type id; Item(int i) : id(i) {} operator id_type() const { return id; } };
struct S { short v; operator short() const { return v; } };
struct F { operator long double() const { return 2.5; } };
int conv() {
   Item it(7);
   unsigned int u = it;
   S s{4};
   short x = s;
   int y = s + 1;
   int z = 10 - s;
   double d = F();
   double e = F() * 2;
   bool lt = s < 5;
   Item *p = &it;
   Item *q = &it;
   printf("%u %d %d %d %g %g %d %d\n", u, x, y, z, d, e, lt, p == q);
   return 0;
}
"""


def test_a_class_converting_to_a_number_is_that_number_in_arithmetic(capsys) -> None:
    run_source(PROGRAM, "conv.C")
    assert capsys.readouterr().out == "7 4 5 6 2.5 5 1 1\n"


def test_a_conversion_is_to_a_number_or_none() -> None:
    assert (conversion("operator  unsigned int"), conversion("operator long double")) == (
        "__int__", "__float__")  # fmt: skip
    assert conversion("operator bool") is None and conversion("operator TString") is None
    assert conversion("size") is None
    with pytest.raises(Refusal, match="the operator B defined for a class"):
        translate("struct B {};\nstruct T { operator B() const { return B(); } };", "t.C")


def test_an_operator_of_the_class_itself_is_kept() -> None:
    class Own:
        def __int__(self) -> int:
            return 3

        def __add__(self, other: object) -> str:
            return "own"

        def __neg__(self) -> str:
            return "minus"

        def __radd__(self, other: object) -> str:
            return "right"

    convertible(Own)
    one = Own()
    assert (one + 1, 1 + one, -one, one * 2, 7 - one, float(one), +one) == (
        "own", "right", "minus", 6, 4, 3.0, 3)  # fmt: skip
    assert one.__index__() == 3 and [0, 1, 2, 3][one] == 3


def test_a_lambda_capture_that_names_a_value_holds_it_from_when_it_was_made(capsys) -> None:
    source = r"""
    #include <cstdio>
    #include <vector>
    int lam() {
       int base = 10;
       auto f = [n = base * 2, m = 3](int x) { return n + m + x; };
       base = 0;
       std::vector<int> v{1, 2, 3};
       auto g = [s = v.size()]() { return (int)s; };
       v.push_back(4);
       printf("%d %d %d\n", f(1), g(), (int)v.size());
       return 0;
    }
    """
    run_source(source, "lam.C")
    assert capsys.readouterr().out == "24 3 4\n"


def test_a_default_cpp_works_out_at_each_call_is_worked_out_at_each_call(capsys) -> None:
    source = r"""
    #include <cstdio>
    #include <vector>
    int g = 0;
    const int kTen = 10;
    struct B {
       static int fg;
       static int Next() { return ++fg; }
       void F(int a = ++g, int b = B::Next(), int c = -kTen) { printf("%d %d %d\n", a, b, c); }
    };
    int B::fg = 0;
    int grow(std::vector<int> v = {}) { v.push_back(1); return (int)v.size(); }
    int calls() {
       B o;
       o.F();
       o.F();
       o.F(7);
       auto f = [](int n = ++g) { return n; };
       int h = B::fg++;
       printf("%d %d %d %d %d\n", grow(), grow(), f(), h, B::fg);
       return 0;
    }
    """
    run_source(source, "calls.C")
    assert capsys.readouterr().out == "1 1 -10\n2 2 -10\n7 3 -10\n1 1 3 3 4\n"
    assert repr(UNGIVEN) == "UNGIVEN"  # what a deferred default reads as in a traceback


def test_decltype_of_a_variable_declares_another_of_its_type(capsys) -> None:
    source = r"""
    #include <cstdio>
    #include <map>
    #include <vector>
    int dt() {
       std::vector<int> m0{1, 2};
       decltype(m0) m4;
       m4.push_back(3);
       m4.push_back(4);
       std::map<int, double> mm;
       decltype(mm) m5;
       m5[1] = 2.5;
       int i = 3;
       decltype(i) j = 7;
       decltype(i + 1) k = 8;
       printf("%d %d %g %d %d\n", (int)m4.size(), m4[1], m5[1], j, k);
       return 0;
    }
    """
    run_source(source, "dt.C")
    assert capsys.readouterr().out == "2 4 2.5 7 8\n"


def test_a_local_class_declared_with_its_variable_destroys_it_as_the_scope_ends(
    capsys,
) -> None:
    source = r"""
    #include <cstdio>
    #include <string>
    int raii() {
       struct Raii {
          std::string f;
          Raii(const char *n) : f(n) {}
          ~Raii() { printf("bye %s\n", f.c_str()); }
       } guard("x");
       printf("hi %s\n", guard.f.c_str());
       return 0;
    }
    """
    run_source(source, "raii.C")
    assert capsys.readouterr().out == "hi x\nbye x\n"


def test_a_destructor_called_by_hand_runs_its_body(capsys) -> None:
    source = r"""
    #include <cstdio>
    struct A { int v; A(int x) : v(x) {} ~A() { printf("~A %d\n", v); } };
    struct P { int v; };
    int dd() {
       A *a = new A(3);
       a->~A();
       P *p = new P;
       p->~P();
       printf("end\n");
       return 0;
    }
    """
    run_source(source, "dd.C")
    assert capsys.readouterr().out == "~A 3\nend\n"
    with pytest.raises(Refusal, match="calling the destructor ~A by hand"):
        translate("struct A { ~A() {} void f() { this->~A(); } };", "t.C")
    with pytest.raises(Refusal, match="calling the destructor ~B by hand"):
        translate("struct A { ~A() {} };\nvoid f(A *a) { a->~B(); }", "t.C")
