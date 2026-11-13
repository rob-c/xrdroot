"""C++ the translator reads now: braced temporaries of any type, member templates called with
their arguments, ``typename`` in an expression, ``alignof``, unnamed structs with variables,
contextual keywords as names, and characters of a string written in place.

Expected outputs are what ROOT 6.40.04's ``root.exe`` printed for the same C++.
"""

from __future__ import annotations

import pytest

from xrdroot.cppstd.strings import with_char
from xrdroot.pycxx import run_source, translate

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
