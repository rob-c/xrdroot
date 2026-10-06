"""The runtime translated macros run on: C's printf, C++'s streams, C's arithmetic."""

from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot.cint import runtime as rt


@pytest.mark.parametrize(
    ("fmt", "args", "text"),
    [
        ("%d|%5d|%-5d|%05d|%+d", (7, 7, 7, 7, 7), "7|    7|7    |00007|+7"),
        ("%i %ld %lld %hd %hhd", (1, 2, 3, 70000, 300), "1 2 3 4464 44"),
        ("%u %lu", (-1, -1), "4294967295 18446744073709551615"),
        ("%x %X %o %#x", (-1, 255, 8, 255), "ffffffff FF 10 0xff"),
        ("%f %.2f %10.3f %e %E", (3.14159, 3.14159, 3.14159, 12345.678, 0.5), None),
        ("%g %g %g %g %G", (100000.0, 1e6, 1e-5, 0.0001, 1e-10), "100000 1e+06 1e-05 0.0001 1E-10"),
        ("%s %c %c %%", ("hi", 65, "xyz"), "hi A x %"),
        ("%*d|%-*.*f", (4, 1, 8, 2, 3.14159), "   1|3.14    "),
        ("%'d %zu", (1234, 5), "1234 5"),
        ("%s", (None,), "(null)"),
        ("%a", (1.5,), "0x1.8000000000000p+0"),
        ("%A", (1.5,), "0X1.8000000000000P+0"),
        ("%.0f %.0e", (2.5, 12345.0), "2 1e+04"),
        ("%f %f", (float("inf"), float("nan")), "inf nan"),
    ],
)
def test_cformat_writes_what_cs_printf_writes(fmt: str, args: tuple, text: str | None) -> None:
    expected = text if text is not None else "3.141590 3.14      3.142 1.234568e+04 5.000000E-01"
    assert rt.cformat(fmt, *args) == expected


def test_a_pointer_is_printed_in_hexadecimal_and_a_null_one_as_nil() -> None:
    assert rt.cformat("%p", object()).startswith("0x")
    assert rt.cformat("%p", 0) == "(nil)"


def test_too_few_arguments_and_n_are_refused_in_sentences() -> None:
    with pytest.raises(TypeError, match="wants more than the 1 argument it"):
        rt.cformat("%d %d", 1)
    with pytest.raises(TypeError, match="the 0 arguments"):
        rt.cformat("%d")
    with pytest.raises(ValueError, match="%n writes a count"):
        rt.cformat("%n", 1)


def test_the_printf_family_writes_where_c_and_root_write(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert rt.printf("%d\n", 3) == 2
    rt.Printf("x=%g", 0.5)
    assert rt.sprintf("%03d", 7) == "007"
    assert rt.Form("h%d", 2) == "h2"
    rt.puts("line")
    rt.putchar(65)
    rt.putchar("b")
    rt.fprintf("stdout", "%s", "out")
    rt.fprintf("stderr", "%s", "err")
    rt.Info("where", "n=%d", 1)
    rt.Warning("", "careful")
    rt.Error("f", "bad %s", "x")
    with pytest.raises(SystemExit):
        rt.Fatal("f", "fatal")
    captured = capsys.readouterr()
    assert captured.out == "3\nx=0.5\nline\nAbout"
    assert captured.err == (
        "errInfo in <where>: n=1\nWarning: careful\nError in <f>: bad x\nFatal in <f>: fatal\n"
    )


def same(got: object, expected: object) -> bool:
    """Equal - with ``nan`` equal to itself, which is what a C function returning it means."""
    if isinstance(expected, float) and math.isnan(expected):
        return isinstance(got, float) and math.isnan(got)
    return got == expected and type(got) is type(expected)


@pytest.mark.parametrize(
    ("items", "text"),
    [
        (["x = ", 1.0 / 3, " ", True, " ", 42], "x = 0.333333 1 42"),
        ([rt.boolalpha, False, rt.noboolalpha, " ", False], "false 0"),
        ([rt.fixed, rt.setprecision(2), 3.14159, " ", rt.scientific, 1234.5], "3.14 1.23e+03"),
        ([rt.setw(6), 7, "|", rt.left, rt.setw(4), 1, "|"], "     7|1   |"),
        ([rt.setfill("*"), rt.setw(3), 5, rt.setfill(ord("-")), rt.setw(2), 1], "**5-1"),
        (
            [rt.hex, 255, " ", -1, rt.oct, " ", 8, " ", -1, rt.dec, " ", 9],
            "ff ffffffff 10 37777777777 9",
        ),
        ([rt.showpos, 3, " ", 2.5, rt.noshowpos, " ", 3], "+3 +2.5 3"),
        ([None, " ", np.float32(0.1), " ", np.int32(3), rt.flush], "0 0.1 3"),
        ([rt.setprecision(0), 2.5, " ", rt.defaultfloat, 1e-5], "2 1e-05"),
    ],
)
def test_a_stream_formats_as_iostreams_do(items: list[object], text: str) -> None:
    out = rt.ostringstream()
    for item in items:
        out << item
    assert out.str() == text


def test_endl_ends_the_line(capsys: pytest.CaptureFixture[str]) -> None:
    out = rt.ostream()
    out << "a" << rt.endl << "b" << rt.endl
    assert capsys.readouterr().out == "a\nb\n"


def test_a_stream_keeps_cpps_member_functions() -> None:
    out = rt.ostringstream("a")
    assert out.precision(3) == 6
    assert out.precision() == 3
    out.setf(rt.fixed, "ignored")
    out << 1.0
    out.unsetf(rt.fixed)
    out << " " << 1.0
    out.put(ord("!")).put("?").flush()
    assert out.good()
    assert out.str() == "a1.000 1!?"
    assert out.str("new") == ""
    assert out.str() == "new"
    assert repr(rt.endl) == "std::endl"


def test_cout_writes_to_whatever_stdout_is_now(capsys: pytest.CaptureFixture[str]) -> None:
    rt.cout << "to out" << rt.endl
    rt.cerr << "to err" << rt.endl
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("to out\n", "to err\n")


@pytest.mark.parametrize(
    ("a", "b", "quotient", "rest"),
    [(7, 2, 3, 1), (-7, 2, -3, -1), (7, -2, -3, 1), (-7, -2, 3, -1), (np.int32(9), 4, 2, 1)],
)
def test_integer_division_truncates_toward_zero_as_c_does(
    a: int, b: int, quotient: int, rest: int
) -> None:
    assert (rt.idiv(a, b), rt.imod(a, b), rt.div(a, b), rt.mod(a, b)) == (quotient, rest) * 2


def test_division_of_unknown_types_is_true_division_unless_both_are_integers() -> None:
    assert rt.div(7.0, 2) == 3.5
    assert rt.mod(-7.5, 2) == -1.5
    with pytest.raises(ZeroDivisionError, match="undefined"):
        rt.idiv(1, 0)


@pytest.mark.parametrize(
    ("got", "expected"),
    [
        (lambda: rt.to_int(3.9), 3),
        (lambda: rt.to_int(-3.9), -3),
        (lambda: rt.to_int("A"), 65),
        (lambda: rt.to_int(np.int64(5)), 5),
        (lambda: rt.f32(0.1), float(np.float32(0.1))),
        (lambda: rt.u8(257), 1),
        (lambda: rt.u16(-1), 65535),
        (lambda: rt.u32(-1), 4294967295),
        (lambda: rt.u64(-1), 2**64 - 1),
        (lambda: rt.i8(200), -56),
        (lambda: rt.i16(40000), -25536),
        (lambda: rt.i32(2**31), -(2**31)),
        (lambda: rt.i64(2**63), -(2**63)),
        (lambda: rt.comma(1, 2, 3), 3),
    ],
)
def test_stores_convert_as_c_converts(got: object, expected: object) -> None:
    assert same(got(), expected)  # type: ignore[operator]


def test_an_infinity_stored_into_an_integer_is_refused() -> None:
    with pytest.raises(OverflowError, match="does not fit"):
        rt.to_int(float("inf"))


def test_exit_and_assert_stop_the_macro() -> None:
    with pytest.raises(SystemExit) as stopped:
        rt.c_exit(3)
    assert stopped.value.code == 3
    rt.cassert(True)
    with pytest.raises(AssertionError, match="assert in the macro failed"):
        rt.cassert(0, "why")


@pytest.mark.parametrize(
    ("got", "expected"),
    [
        (lambda: rt.sqrt(-1), math.nan),
        (lambda: rt.log(0), -math.inf),
        (lambda: rt.log(-1), math.nan),
        (lambda: rt.log10(math.nan), math.nan),
        (lambda: rt.log2(8), 3.0),
        (lambda: rt.exp(1000), math.inf),
        (lambda: rt.pow(0, -1), math.inf),
        (lambda: rt.pow(-8, 1 / 3), math.nan),
        (lambda: rt.pow(10, 400), math.inf),
        (lambda: rt.pow(2, 3), 8.0),
        (lambda: round(rt.cbrt(-8), 12), -2.0),
        (lambda: rt.floor(2.5), 2.0),
        (lambda: rt.ceil(2.5), 3.0),
        (lambda: rt.trunc(-2.5), -2.0),
        (lambda: rt.floor(math.inf), math.inf),
        (lambda: rt.cround(2.5), 3.0),
        (lambda: rt.cround(-2.5), -3.0),
        (lambda: rt.cround(math.inf), math.inf),
        (lambda: rt.cabs(-3), 3),
        (lambda: rt.cabs(-2.5), 2.5),
        (lambda: list(rt.cabs(np.array([-1.5, 2.0]))), [1.5, 2.0]),
        (lambda: type(rt.cabs(np.float64(-2.0))), float),
        (lambda: rt.isnan(math.nan), True),
        (lambda: rt.isinf(math.inf), True),
        (lambda: rt.isfinite(1.0), True),
        (lambda: rt.fmin(1, 2), 1.0),
        (lambda: rt.fmax(1, 2), 2.0),
        (lambda: rt.sqrt.__name__, "sqrt"),
        (lambda: rt.M_PI, math.pi),
    ],
)
def test_cmath_answers_nan_and_inf_where_python_raises(got: object, expected: object) -> None:
    assert same(got(), expected)  # type: ignore[operator]


@pytest.mark.parametrize(
    ("got", "expected"),
    [
        (lambda: rt.strlen("abc"), 3),
        (lambda: rt.strcmp("a", "b"), -1),
        (lambda: rt.strcmp("b", "a"), 1),
        (lambda: rt.strcmp("a", "a"), 0),
        (lambda: rt.strncmp("abc", "abd", 2), 0),
        (lambda: rt.strcasecmp("ABC", "abc"), 0),
        (lambda: rt.strstr("hello", "ll"), "llo"),
        (lambda: rt.strstr("hello", "z"), None),
        (lambda: rt.strchr("hello", ord("e")), "ello"),
        (lambda: rt.atoi("  42abc"), 42),
        (lambda: rt.atoi("x"), 0),
        (lambda: rt.atol("-7"), -7),
        (lambda: rt.atof("2.5e1x"), 25.0),
        (lambda: rt.atof("nope"), 0.0),
        (lambda: rt.stoi("12"), 12),
        (lambda: rt.stod("1.5"), 1.5),
        (lambda: rt.to_string(3), "3"),
        (lambda: rt.to_string(2.5), "2.500000"),
        (lambda: rt.to_string(True), "1"),
        (lambda: rt.char_at("ab", 1), 98),
        (lambda: rt.char_at("ab", 2), 0),
        (lambda: rt.find("hello", "l"), 2),
        (lambda: rt.find("hello", "z"), rt.npos),
        (lambda: rt.rfind("hello", "l"), 3),
        (lambda: rt.rfind("hello", "z"), rt.npos),
        (lambda: rt.substr("hello", 1, 3), "ell"),
        (lambda: rt.substr("hello", 2), "llo"),
        (lambda: rt.cstr(None), ""),
        (lambda: rt.cstr(65), "A"),
        (lambda: rt.cstr(1.5), "1.5"),
    ],
)
def test_string_functions_read_c_strings_as_c_does(got: object, expected: object) -> None:
    assert same(got(), expected)  # type: ignore[operator]


@pytest.mark.parametrize(
    ("call", "why"),
    [
        (lambda: rt.stoi("x"), "std::stoi"),
        (lambda: rt.stod("x"), "std::stod"),
        (lambda: rt.substr("ab", 5), "past the end"),
    ],
)
def test_string_functions_that_throw_in_cpp_raise(call: object, why: str) -> None:
    with pytest.raises((ValueError, IndexError), match=why):
        call()  # type: ignore[operator]
