"""``std::numeric_limits`` and ``sizeof`` as C++ has them on LP64 macOS on ARM64.

The expected values are what ROOT 6.40.04's ``root.exe`` printed for the same C++.
"""

from __future__ import annotations

import pytest

from xrdroot.pycxx import Refusal, run_source, translate

#: Every member of ``numeric_limits`` asked of an integer, a character and a floating type.
LIMITS = r"""
#include <cstdio>
#include <limits>
using namespace std;
int limits() {
   printf("%.17g %.17g %d %d %d\n", std::numeric_limits<float>::denorm_min(),
          numeric_limits<double>::epsilon(), std::numeric_limits<unsigned int>::digits,
          numeric_limits<long>::digits10, std::numeric_limits<float>::max_digits10);
   bool s = std::numeric_limits<char>::is_signed;
   auto m = std::numeric_limits<unsigned int>::lowest();
   printf("%d %u %d %d\n", s, m, std::numeric_limits<int>::max() / 2,
          std::numeric_limits<wchar_t>::digits);
   printf("%g %g %d %d %g\n", std::numeric_limits<long double>::max(),
          std::numeric_limits<double>::infinity(), std::numeric_limits<float>::min_exponent,
          std::numeric_limits<bool>::digits, std::numeric_limits<float>::round_error());
   return 0;
}
"""


def test_numeric_limits_are_the_constants_cpp_has(capsys) -> None:
    run_source(LIMITS, "limits.C")
    assert capsys.readouterr().out == (
        "1.4012984643248171e-45 2.2204460492503131e-16 32 18 9\n"
        "1 0 1073741823 31\n"
        "1.79769e+308 inf -125 1 0.5\n"
    )
    python = translate(LIMITS, "limits.C")
    assert "1.401298464324817e-45" in python and "float('inf')" in python


def test_numeric_limits_of_what_is_no_arithmetic_type_is_refused() -> None:
    with pytest.raises(Refusal, match=r"std::numeric_limits<std::string>::digits is not"):
        translate("void t() { int d = std::numeric_limits<std::string>::digits; }", "t.C")
    with pytest.raises(Refusal, match=r"std::numeric_limits<int\*>::max\(\) is not"):
        translate("void t() { auto d = std::numeric_limits<int*>::max(); }", "t.C")
    with pytest.raises(Refusal, match=r"std::numeric_limits<int>::nothing is not"):
        translate("void t() { auto d = std::numeric_limits<int>::nothing(); }", "t.C")


#: The sizes of numbers, the standard's classes, and classes laid out as the Itanium ABI does.
SIZES = r"""
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <array>
#include <complex>
#include <functional>
#include <bitset>
#include <atomic>
struct A { char c; double d; int i; };
struct B { char c; };
struct E {};
struct V { virtual ~V() {} int x; };
struct W : V { char y; };
struct D : A { short s; };
struct P { int *p; char c[3]; };
enum Colour { kBlue2 };
enum class Small : char { a };
struct N { A a[2]; B b; Small s; };
struct F : E { int z; };
union U { char c[5]; int i; };
struct S { std::string name; std::vector<double> v; std::pair<int, double> pr; };
struct R { int &r; };
class H { int h; char c; public: H() : h(0), c(0) {} };
struct HD : H { char d; };
class K { int k; char c; };
struct KD : K { char d; };
struct M2 { int m; private: char c; };
struct MD : M2 { char d; };
struct RF { int &r; char c; };
struct RD : RF { char d; };
struct BB : B { int b; };
struct BD : BB { char d; };
struct EB : E { int e; char f; };
struct ED : EB { char d; };
struct CM { H h; };
struct CD : CM { char d; };
struct XH : H { char x; };
struct YH : XH { char y; };
struct alignas(16) Q { char q; };
struct alignas(double) Q2 { char q; };
struct Empty2 : E {};
const int kN = 3;
int sizes() {
   double m[kN][kN + 1];
   int arr[] = {1, 2, 3, 4, 5};
   A a;
   printf("%zu %zu %zu %zu %zu %zu %zu %zu %zu %zu\n", sizeof(A), sizeof(B), sizeof(E), sizeof(V),
          sizeof(W), sizeof(D), sizeof(P), sizeof(Colour), sizeof(Small), sizeof(N));
   printf("%zu %zu %zu %zu %zu %zu %zu\n", sizeof(F), sizeof(U), sizeof(S), sizeof(m),
          sizeof(arr) / sizeof(arr[0]), sizeof(a.d), sizeof a);
   printf("%zu %zu %zu %zu %zu\n", sizeof(std::array<short, 3>), sizeof(std::complex<float>),
          sizeof(long double), sizeof(wchar_t), sizeof(std::map<int, A>));
   printf("%zu %zu %zu %zu %zu %zu %zu\n", sizeof(R), sizeof(HD), sizeof(Q), sizeof(Q2),
          sizeof(Empty2), sizeof(std::function<void()>), sizeof(std::bitset<70>));
   printf("%zu %zu %zu %zu %zu\n", sizeof(std::atomic<short>), sizeof(int *[4]), sizeof(A &),
          sizeof(std::tuple<char, int, char>), sizeof(KD));
   printf("%zu %zu %zu %zu %zu %zu\n", sizeof(MD), sizeof(RD), sizeof(BD), sizeof(ED), sizeof(CD),
          sizeof(YH));
   return 0;
}
"""


def test_sizeof_is_the_size_clang_gives_on_arm64(capsys) -> None:
    run_source(SIZES, "sizes.C")
    assert capsys.readouterr().out == (
        "24 1 1 16 16 32 16 4 1 56\n"
        "4 8 64 96 5 8 24\n"
        "6 8 8 4 24\n"
        "8 8 16 8 1 32 16\n"
        "2 32 24 12 12\n"
        "8 16 12 12 12 8\n"
    )


@pytest.mark.parametrize(
    "declaration",
    [
        "struct L : Library { int x; }; unsigned long n = sizeof(L);",
        "struct M { Library m; }; unsigned long n = sizeof(M);",
        "union M { Library m; }; unsigned long n = sizeof(M);",
        "unsigned long n = sizeof(std::array<int, k>);",
        "unsigned long n = sizeof(std::bitset<k>);",
        "unsigned long n = sizeof(std::complex<Library>);",
        "unsigned long n = sizeof(std::pair<Library, int>);",
        "struct alignas(Library) M { int m; }; unsigned long n = sizeof(M);",
        "struct alignas(k) M { int m; }; unsigned long n = sizeof(M);",
        "void f(int k) { int a[k]; unsigned long n = sizeof(a); }",
        "const double kD = 2; int a[kD]; unsigned long n = sizeof(a);",
        "int a[2 / 0]; unsigned long n = sizeof(a);",
        "int a[2 - x]; unsigned long n = sizeof(a);",
        "int a[ns::k]; unsigned long n = sizeof(a);",
        "struct G { virtual ~G() {} }; struct M : G { Library m; }; unsigned long n = sizeof(M);",
    ],
)
def test_sizeof_of_what_has_no_known_size_is_refused(declaration: str) -> None:
    with pytest.raises(Refusal, match="sizeof of a type whose size this translator does not know"):
        translate(declaration, "t.C")


def test_constant_array_sizes_are_worked_out(capsys) -> None:
    source = r"""
    const int kRows = 2;
    int grid[kRows * 3 - 1][-(-4) / 2];
    void sized() { printf("%zu %zu\n", sizeof(grid), sizeof(grid[0])); }
    """
    run_source(source, "sized.C")
    assert capsys.readouterr().out == "40 8\n"
