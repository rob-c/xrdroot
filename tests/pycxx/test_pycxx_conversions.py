"""A class's conversion to a number - ``operator unsigned int()``, ``operator double()``.

The expected output is what ROOT 6.40.04's cling prints for the same program.
"""

from __future__ import annotations

import pytest

from xrdroot.cppstd.objects import convertible
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
