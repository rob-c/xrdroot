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


def test_the_printf_family_writes_where_c_and_root_write(capsys: pytest.CaptureFixture[str]) -> None:
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


def test_cout_formats_numbers_as_iostreams_do(capsys: pytest.CaptureFixture[str]) -> None:
    out = rt.ostream()
    out << "x = " << 1.0 / 3 << " " << True << " " << 42 << rt.endl
    out << rt.boolalpha << False << rt.noboolalpha << " " << False << rt.endl
    out << rt.fixed << rt.setprecision(2) << 3.14159 << " " << rt.scientific << 1234.5 << rt.endl
    out << rt.defaultfloat << rt.setw(6) << 7 << "|" << rt.left << rt.setw(4) << 1 << "|"
    out << rt.right << rt.setfill("*") << rt.setw(3) << 5 << rt.setfill(ord("-")) << rt.endl
    out << rt.hex << 255 << " " << -1 << rt.oct << " " << 8 << " " << -1 << rt.dec << rt.endl
    out << rt.showpos << 3 << " " << 2.5 << rt.noshowpos << " " << None << " " << out.pad("a")
    out << np.float32(0.1) << " " << np.int32(3) << " " << rt.flush << rt.endl
    captured = capsys.readouterr().out.splitlines()
    assert captured == [
        "x = 0.333333 1 42",
        "false 0",
        "3.14 1.23e+03",
        "     7|1   |**5",
        "ff ffffffff 10 37777777777",
        "+3 +2.5 0 a0.1 3 ",
    ]


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
    assert rt.idiv(a, b) == quotient
    assert rt.imod(a, b) == rest
    assert rt.div(a, b) == quotient
    assert rt.mod(a, b) == rest


def test_division_of_unknown_types_is_true_division_unless_both_are_integers() -> None:
    assert rt.div(7.0, 2) == 3.5
    assert rt.mod(-7.5, 2) == -1.5
    with pytest.raises(ZeroDivisionError, match="undefined"):
        rt.idiv(1, 0)


def test_stores_convert_as_c_converts() -> None:
    assert rt.to_int(3.9) == 3
    assert rt.to_int(-3.9) == -3
    assert rt.to_int("A") == 65
    assert rt.to_int(np.int64(5)) == 5
    with pytest.raises(OverflowError, match="does not fit"):
        rt.to_int(float("inf"))
    assert rt.f32(0.1) == float(np.float32(0.1))
    assert rt.u8(257) == 1
    assert rt.u16(-1) == 65535
    assert rt.u32(-1) == 4294967295
    assert rt.u64(-1) == 2**64 - 1
    assert rt.i8(200) == -56
    assert rt.i16(40000) == -25536
    assert rt.i32(2**31) == -(2**31)
    assert rt.i64(2**63) == -(2**63)
    assert rt.comma(1, 2, 3) == 3


def test_exit_and_assert_stop_the_macro() -> None:
    with pytest.raises(SystemExit) as stopped:
        rt.c_exit(3)
    assert stopped.value.code == 3
    rt.cassert(True)
    with pytest.raises(AssertionError, match="assert in the macro failed"):
        rt.cassert(0, "why")


def test_cmath_answers_nan_and_inf_where_python_raises() -> None:
    assert math.isnan(rt.sqrt(-1))
    assert rt.log(0) == -math.inf
    assert math.isnan(rt.log(-1))
    assert math.isnan(rt.log10(float("nan")))
    assert rt.log2(8) == 3
    assert rt.exp(1000) == math.inf
    assert rt.pow(0, -1) == math.inf
    assert math.isnan(rt.pow(-8, 1 / 3))
    assert rt.pow(10, 400) == math.inf
    assert rt.pow(2, 3) == 8.0
    assert rt.cbrt(-8) == pytest.approx(-2)
    assert rt.floor(2.5) == 2.0 and rt.ceil(2.5) == 3.0 and rt.trunc(-2.5) == -2.0
    assert rt.floor(math.inf) == math.inf
    assert rt.cround(2.5) == 3.0 and rt.cround(-2.5) == -3.0 and rt.cround(math.inf) == math.inf
    assert rt.cabs(-3) == 3 and isinstance(rt.cabs(-3), int)
    assert rt.cabs(-2.5) == 2.5
    assert rt.isnan(math.nan) and rt.isinf(math.inf) and rt.isfinite(1.0)
    assert rt.fmin(1, 2) == 1 and rt.fmax(1, 2) == 2
    assert rt.sqrt.__name__ == "sqrt"
    assert rt.M_PI == math.pi


def test_string_functions_read_c_strings_as_c_does() -> None:
    assert rt.strlen("abc") == 3
    assert rt.strcmp("a", "b") < 0 < rt.strcmp("b", "a")
    assert rt.strcmp("a", "a") == 0
    assert rt.strncmp("abc", "abd", 2) == 0
    assert rt.strcasecmp("ABC", "abc") == 0
    assert rt.strstr("hello", "ll") == "llo"
    assert rt.strstr("hello", "z") is None
    assert rt.strchr("hello", ord("e")) == "ello"
    assert rt.atoi("  42abc") == 42 and rt.atoi("x") == 0
    assert rt.atol("-7") == -7
    assert rt.atof("2.5e1x") == 25.0 and rt.atof("nope") == 0.0
    assert rt.stoi("12") == 12 and rt.stod("1.5") == 1.5
    with pytest.raises(ValueError, match="std::stoi"):
        rt.stoi("x")
    with pytest.raises(ValueError, match="std::stod"):
        rt.stod("x")
    assert rt.to_string(3) == "3" and rt.to_string(2.5) == "2.500000"
    assert rt.to_string(True) == "1"
    assert rt.char_at("ab", 1) == 98 and rt.char_at("ab", 2) == 0
    assert rt.find("hello", "l") == 2 and rt.find("hello", "z") == rt.npos
    assert rt.rfind("hello", "l") == 3 and rt.rfind("hello", "z") == rt.npos
    assert rt.substr("hello", 1, 3) == "ell" and rt.substr("hello", 2) == "llo"
    with pytest.raises(IndexError, match="past the end"):
        rt.substr("ab", 5)
    assert rt.cstr(None) == "" and rt.cstr(65) == "A" and rt.cstr(1.5) == "1.5"
