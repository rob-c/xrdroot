"""``std::numeric_limits`` of every arithmetic type, bit for bit as C++ has it on LP64 ARM64."""

from __future__ import annotations

import math

import pytest

from xrdroot.cppstd import stdlib

#: What ``root.exe`` (ROOT 6.40.04, clang on macOS ARM64) printed for each type: ``min max
#: lowest epsilon round_error infinity denorm_min quiet_NaN`` with ``%.17g``, then ``digits
#: digits10 max_digits10 is_signed is_integer is_exact radix min_exponent max_exponent
#: min_exponent10 max_exponent10 has_infinity is_modulo is_iec559``.
ORACLE = """\
bool 0 1 0 0 0 0 0 0 1 0 0 0 1 1 2 0 0 0 0 0 0 0
char -128 127 -128 0 0 0 0 0 7 2 0 1 1 1 2 0 0 0 0 0 0 0
signed char -128 127 -128 0 0 0 0 0 7 2 0 1 1 1 2 0 0 0 0 0 0 0
unsigned char 0 255 0 0 0 0 0 0 8 2 0 0 1 1 2 0 0 0 0 0 1 0
wchar_t -2147483648 2147483647 -2147483648 0 0 0 0 0 31 9 0 1 1 1 2 0 0 0 0 0 0 0
char16_t 0 65535 0 0 0 0 0 0 16 4 0 0 1 1 2 0 0 0 0 0 1 0
char32_t 0 4294967295 0 0 0 0 0 0 32 9 0 0 1 1 2 0 0 0 0 0 1 0
short -32768 32767 -32768 0 0 0 0 0 15 4 0 1 1 1 2 0 0 0 0 0 0 0
unsigned short 0 65535 0 0 0 0 0 0 16 4 0 0 1 1 2 0 0 0 0 0 1 0
int -2147483648 2147483647 -2147483648 0 0 0 0 0 31 9 0 1 1 1 2 0 0 0 0 0 0 0
unsigned int 0 4294967295 0 0 0 0 0 0 32 9 0 0 1 1 2 0 0 0 0 0 1 0
long -9.2233720368547758e+18 9.2233720368547758e+18 -9.2233720368547758e+18 0 0 0 0 0 63 18 0 1 1 1 2 0 0 0 0 0 0 0
unsigned long 0 1.8446744073709552e+19 0 0 0 0 0 0 64 19 0 0 1 1 2 0 0 0 0 0 1 0
long long -9.2233720368547758e+18 9.2233720368547758e+18 -9.2233720368547758e+18 0 0 0 0 0 63 18 0 1 1 1 2 0 0 0 0 0 0 0
unsigned long long 0 1.8446744073709552e+19 0 0 0 0 0 0 64 19 0 0 1 1 2 0 0 0 0 0 1 0
float 1.1754943508222875e-38 3.4028234663852886e+38 -3.4028234663852886e+38 1.1920928955078125e-07 0.5 inf 1.4012984643248171e-45 nan 24 6 9 1 0 0 2 -125 128 -37 38 1 0 1
double 2.2250738585072014e-308 1.7976931348623157e+308 -1.7976931348623157e+308 2.2204460492503131e-16 0.5 inf 4.9406564584124654e-324 nan 53 15 17 1 0 0 2 -1021 1024 -307 308 1 0 1
long double 2.2250738585072014e-308 1.7976931348623157e+308 -1.7976931348623157e+308 2.2204460492503131e-16 0.5 inf 4.9406564584124654e-324 nan 53 15 17 1 0 0 2 -1021 1024 -307 308 1 0 1
"""  # noqa: E501

#: The members printed with ``%.17g``, and those printed with ``%d``.
VALUES = "min max lowest epsilon round_error infinity denorm_min quiet_NaN".split()
NUMBERS = """digits digits10 max_digits10 is_signed is_integer is_exact radix min_exponent
max_exponent min_exponent10 max_exponent10 has_infinity is_modulo is_iec559""".split()


def _printed(name: str) -> str:
    limits = stdlib.numeric_limits[name]
    values = [f"{float(getattr(limits, member)()):.17g}" for member in VALUES]
    numbers = [str(int(getattr(limits, member))) for member in NUMBERS]
    return " ".join([name, *values, *numbers])


@pytest.mark.parametrize("line", ORACLE.splitlines())
def test_every_member_of_numeric_limits_prints_as_cpps_does(line: str) -> None:
    name = line.rsplit(" ", 22)[0]
    assert _printed(name) == line


def test_the_limits_are_of_the_types_cpp_gives_them() -> None:
    assert stdlib.limit("float", "max") == 3.4028234663852886e38
    assert stdlib.limit("int", "infinity") == 0 and stdlib.limit("bool", "epsilon") is False
    assert stdlib.limit("double", "is_iec559") is True
    assert math.isnan(stdlib.limit("double", "signaling_NaN"))
    assert stdlib.limit("std::string", "max") is None
    assert stdlib.limit("int", "nothing") is None
    assert stdlib.numeric_limits["unsigned"].max() == 2**32 - 1
