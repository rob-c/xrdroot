"""C formatting stays independent while ROOT's message functions keep their behavior."""

from __future__ import annotations

import importlib

import pytest

from cppstdsupport import isolated


def test_c_stdio_exports_are_shared_with_the_runtime_adapter() -> None:
    from xrdroot.cppstd import printf

    legacy = importlib.import_module("xrdroot.cint.runtime.printf")
    for name in printf.__all__:
        assert getattr(legacy, name) is getattr(printf, name), name
    assert legacy.SPEC is printf.SPEC
    assert legacy.BITS is printf.BITS
    assert legacy.CONVERTERS is printf.CONVERTERS
    assert legacy.Printf_ is legacy.Printf
    for name in ("Printf", "Form", "Info", "Warning", "Error", "Fatal"):
        assert getattr(legacy, name).__module__ == legacy.__name__


def test_root_diagnostics_keep_their_channels_and_newline_conventions(capsys) -> None:
    legacy = importlib.import_module("xrdroot.cint.runtime.printf")
    legacy.Printf("x=%d", 7)
    assert legacy.Form("%03d", 7) == "007"
    legacy.Info("test", "%s", "message")
    legacy.Warning("", "%d", 1)
    legacy.Error("test", "%s", "error")
    with pytest.raises(SystemExit) as error:
        legacy.Fatal("test", "%d", 2)
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == "x=7\n"
    assert captured.err == (
        "Info in <test>: message\nWarning: 1\n"
        "Error in <test>: error\nFatal in <test>: 2\n"
    )


def test_c_stdio_formats_and_writes_without_loading_root_or_the_interpreter() -> None:
    isolated("""
import io
from xrdroot.cppstd import printf
assert printf.cformat('%hhd %hhu %lld %zu', 255, -1, 2**40, -1) == (
    '-1 255 1099511627776 18446744073709551615'
)
assert printf.cformat('%s %c %p', None, 65, None) == '(null) A (nil)'
assert printf.sprintf('%*.*f', -7, 2, 1.25) == '1.25   '
assert printf.cformat('%a %A', 1.25, 1.25) == '0x1.4000000000000p+0 0X1.4000000000000P+0'
target = io.StringIO()
assert printf.fprintf(target, '%04d', 7) == 4
assert target.getvalue() == '0007'
try:
    printf.cformat('%d')
except TypeError as error:
    assert 'wants more' in str(error)
else:
    raise AssertionError('missing arguments must fail')
try:
    printf.cformat('%n', 0)
except ValueError as error:
    assert '%n writes a count' in str(error)
else:
    raise AssertionError('count-pointer writes must fail')
""")
