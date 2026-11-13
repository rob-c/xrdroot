"""C++ the translator reads now: braced temporaries of any type, member templates called with
their arguments, ``typename`` in an expression, ``alignof``, unnamed structs with variables,
contextual keywords as names, and characters of a string written in place.

Expected outputs are what ROOT 6.40.04's ``root.exe`` printed for the same C++.
"""

from __future__ import annotations

import pytest

from xrdroot.cppstd.strings import with_char
from xrdroot.pycxx import Refusal, run_source, translate

#: A string's characters written one at a time, as C and C++ write them.
CHARACTERS = r"""
#include <cstdio>
#include <string>
#include <cstring>
struct Holder { char fChar[8]; };
std::string named = "global";
void characters() {
   char buf[6] = "abc";
   buf[1] = 'X';
   printf("%s %d\n", buf, (int)strlen(buf));
   buf[3] = 'd';
   buf[4] = 0;
   printf("%s %d\n", buf, (int)strlen(buf));
   buf[2] = '\0';
   printf("%s %d\n", buf, (int)strlen(buf));
   std::string s = "hello";
   s[0] = 'j';
   s[4]++;
   named[0] = 'G';
   printf("%s %s\n", s.c_str(), named.c_str());
   Holder h;
   for (int n = 0; n < 3; ++n) h.fChar[n] = 'a' + n;
   h.fChar[3] = 0;
   printf("%s\n", h.fChar);
}
"""


def test_a_character_written_into_a_string_is_written_as_cpp_writes_it(capsys) -> None:
    run_source(CHARACTERS, "characters.C")
    assert capsys.readouterr().out == "aXc 3\naXcd 4\naX 2\njellp Global\nabc\n"


def test_a_character_written_past_the_end_of_a_string_is_refused() -> None:
    assert with_char("ab", 1, 0, False) == "a\0" and with_char("ab", 5, 0, True) == "ab"
    with pytest.raises(IndexError, match="past its end"):
        with_char("ab", 3, ord("x"), True)
    with pytest.raises(IndexError, match="past its end"):
        with_char("ab", 2, ord("x"), False)


def test_a_character_of_a_string_handed_in_is_written_through_its_cell(capsys) -> None:
    source = r"""
    void upper(std::string &s) { s[0] = 'A'; }
    void keep(char *p) { }
    void handed() { std::string s = "abc"; upper(s); printf("%s\n", s.c_str()); }
    """
    run_source(source, "handed.C")
    assert capsys.readouterr().out == "Abc\n"


#: Constructs of the grammar, each once.
GRAMMAR = r"""
#include <cstdio>
#include <cinttypes>
#include <cstddef>
#include <type_traits>
struct Point { int x, y; int sum() const { return x + y; } };
struct Maker {
   template <class T> T make(int v) const { return T{v, v + 1}; }
};
struct Traits { typedef double value_type; };
template <class C> double half(int n) { return typename C::value_type(n) / 2; }
static enum Level { kLow, kHigh } level = kHigh;
int grammar() {
   int final = 1, override = 2;
   struct Guard { int g; } const guard{4};
   Point p = Point{3, 4};
   Maker m;
   struct { int a; double b; } anon;
   anon.a = 5;
   std::byte b = std::byte{7};
   uint64_t big = 12345678901234ULL;
   printf("%d %d %d %d %g\n", final + override, p.sum(), m.make<Point>(2).sum(), Point{1, 1}.x,
          half<Traits>(3));
   printf("%d %d %d %zu %zu %" PRIu64 "\n", level, guard.g, anon.a, alignof(double),
          alignof(Point), big);
   for ([[maybe_unused]] auto unused : {1, 2}) {}
   return int(b);
}
"""


def test_the_grammar_reads_as_cpp_does(capsys) -> None:
    run_source(GRAMMAR, "grammar.C")
    assert capsys.readouterr().out == "3 7 5 1 1.5\n1 4 5 8 4 12345678901234\n"


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("double sumit(const std::vector<double>&);", ""),
        ("template <class T> class Inner; template <> class Inner<int>;", ""),
        ("template <class T> struct P { template <class U> friend class Q; };", "class P:"),
        ("void t() { auto d = std::any_cast<Thing*>(held)->x; }", "CXX.std.any_cast['Thing*']"),
        ("void t() { MnMigrad m(f, MnStrategy{2}); }", "CXX.MnStrategy(2)"),
        ("void t() { auto v = a.Get<std::vector<int>>(1); }", "a.Get['std::vector<int>'](1)"),
        ("template <class A> void t(int x) { A::Show(typename A::Kind{x}); }", "A.Kind(x)"),
    ],
)
def test_declarations_and_expressions_once_refused_are_read(source: str, fragment: str) -> None:
    assert fragment in translate(source, "t.C")


#: The largest and least of a range, a local class declared ahead, and an attribute.
EXTREMES = r"""
#include <cstdio>
#include <algorithm>
#include <vector>
#include <variant>
#include <optional>
struct Tool {
   template <class T> long width(T x) const;
   template <class T> long kind() const { return sizeof(T) * 2; }
};
template <class T> long Tool::width(T x) const { return sizeof(T) + long(x); }
static int __attribute__((used)) kSeven = 7;
int extremes() {
   class Later;
   std::vector<double> v{2.5, 7.25, -1, 7.25};
   int a[5] = {4, 9, 1, 9, 3};
   Tool t;
   printf("%g %g %d %d %d\n", *std::max_element(v.begin(), v.end()),
          *std::min_element(v.begin() + 1, v.end()), *std::max_element(a, a + 5),
          *std::min_element(a, a + 3), kSeven);
   printf("%zu %zu %zu %ld %ld\n", sizeof(std::variant<char, short>), sizeof(std::optional<char>),
          sizeof(std::variant<int, double>), t.width(2.0), t.kind<short>());
   return 0;
}
"""


def test_extremes_member_templates_and_the_standards_variants(capsys) -> None:
    run_source(EXTREMES, "extremes.C")
    assert capsys.readouterr().out == "7.25 -1 9 1 7\n8 2 16 10 4\n"


def test_a_number_made_shared_is_read_and_written_through_its_pointer(capsys) -> None:
    source = r"""
    #include <memory>
    void shared() {
       auto p = std::make_shared<int>(5);
       auto q = std::make_shared<unsigned int>();
       auto r = std::make_unique<double>(2.5);
       *q = 9;
       printf("%d %u %g\n", *p, *q, *r);
    }
    """
    run_source(source, "shared.C")
    assert capsys.readouterr().out == "5 9 2.5\n"


def test_an_extreme_by_a_comparison_of_its_own_is_refused() -> None:
    source = "bool less(int a, int b); void f(int *a) { int m = *std::max_element(a, a, less); }"
    with pytest.raises(Refusal, match="std::max_element with a comparison of its own"):
        translate(source, "t.C")


def test_a_member_template_whose_body_needs_its_arguments_refuses_its_use_where_unknown() -> None:
    source = "struct S { template <class T> T make() { T x; return x; } }; void f(S s) {}"
    assert "uninstantiated('the member template make', " in translate(source, "t.C")
