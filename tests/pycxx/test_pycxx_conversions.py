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


def test_a_string_read_through_a_pointer_to_it_is_the_string(capsys) -> None:
    source = r"""
    #include <iostream>
    #include <string>
    bool byAddress(std::string *s) { std::cout << "--" << *s << "--\n"; return *s == "x y"; }
    int strings() {
       std::string mine("x y");
       std::string *p = &mine;
       return byAddress(&mine) && byAddress(p) ? 0 : 1;
    }
    """
    assert run_source(source, "strings.C") == 0
    assert capsys.readouterr().out == "--x y--\n--x y--\n"


def test_a_using_directive_or_declaration_finds_the_hosts_names_there(capsys) -> None:
    """``using namespace Lib::Hist;`` and ``using Lib::Axis;`` of a library's namespaces:
    the names the program does not declare looked for there too; ``std`` and the
    program's own namespaces are not the host's business."""
    from types import SimpleNamespace

    from xrdroot.pycxx.runtime import CXX as HOST
    from xrdroot.pycxx.runtime.namespace import use_namespace

    lib = SimpleNamespace(Lib=SimpleNamespace(Hist=SimpleNamespace(convert=lambda x: x * 2),
                                              Axis=lambda: 5))  # fmt: skip
    source = r"""
    #include <cstdio>
    namespace mine { int one() { return 1; } }
    using namespace std;
    using namespace mine;
    using namespace Lib::Hist;
    using Lib::Axis;
    using typename Lib::Axis;
    using Lib::Axis::operator();
    int uses() { printf("%d %d %d\n", convert(21), Axis(), one()); return 0; }
    """
    python = translate(source, "uses.C")
    assert "use_namespace(CXX, 'Lib.Hist')" in python and "'std'" not in python
    with HOST.bind(lib):
        run_source(source, "uses.C")
    assert capsys.readouterr().out == "42 5 1\n"
    use_namespace(object(), "Ignored")


def test_a_name_in_a_method_of_a_class_derived_from_a_library_is_its_bases_first(capsys):
    """``par`` and ``SetParameters`` of a library base, unqualified in a method of a class
    derived from it, are the object's; a name the base has not is the library's."""
    from types import SimpleNamespace

    from xrdroot.pycxx.runtime import CXX as HOST

    class Base:
        def __init__(self, n: int) -> None:
            self.par = [0.0] * n

        def SetParameters(self, values: list[float]) -> None:
            self.par = list(values)

    lib = SimpleNamespace(Base=Base, scale=10)
    source = r"""
    #include <cstdio>
    #include <vector>
    struct Model : public Base {
       Model() : Base(1) { std::vector<double> p{2.5}; SetParameters(p); }
       double value() const { return par[0] * scale; }
       static int twice(int n) { return 2 * n; }
    };
    int model() { Model m; printf("%g %d\n", m.value(), Model::twice(3)); return 0; }
    """
    with HOST.bind(lib):
        run_source(source, "model.C")
    assert capsys.readouterr().out == "25 6\n"


def test_a_string_built_from_a_streams_iterators_is_the_rest_of_it(tmp_path, capsys) -> None:
    (tmp_path / "text.txt").write_text("one\ntwo\n")
    source = r"""
    #include <cstdio>
    #include <fstream>
    #include <iterator>
    #include <string>
    int whole(const char *name) {
       std::ifstream f(name);
       std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
       std::ifstream g(name);
       std::string first;
       std::getline(g, first);
       std::string rest(std::istreambuf_iterator<char>(g), {});
       printf("%d [%s] [%s]\n", (int)s.size(), first.c_str(), rest.c_str());
       return 0;
    }
    """
    run_source(source, "whole.C", (str(tmp_path / "text.txt"),))
    assert capsys.readouterr().out == "8 [one] [two\n]\n"


def test_a_function_template_written_generically_takes_its_arguments_as_a_subscript() -> None:
    """``ROOT.f["ULong64_t"]``, as PyROOT names an instance, is the generic function; the
    function itself is still called and named as it was."""
    module = run_source("template <typename T>\nT same(T v) { return v; }\n", "same.C",
                        call=False)  # fmt: skip
    same = module["same"]
    assert same["unsigned long"](5) == 5 and same(7) == 7
    assert same.__module__ == "__cint__" and same.__name__ == "same"
