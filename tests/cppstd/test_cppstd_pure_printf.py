"""C stdio formatting keeps widths, argument counts and explicit output destinations."""

from __future__ import annotations

import io

import pytest

from xrdroot.cppstd import printf


@pytest.mark.parametrize(
    ("fmt", "arguments", "expected"),
    [
        ("%hhd %hhu %lld %zu", (255, -1, 2**40, -1), "-1 255 1099511627776 18446744073709551615"),
        ("%s %c %p", (None, 65, None), "(null) A (nil)"),
        ("%*.*f", (-7, 2, 1.25), "1.25   "),
        ("%a %A", (1.25, 1.25), "0x1.4000000000000p+0 0X1.4000000000000P+0"),
        ("%#x %% %03d", (42, 7), "0x2a % 007"),
    ],
)
def test_c_formatting_preserves_integer_widths_flags_and_argument_order(fmt, arguments, expected):
    assert printf.cformat(fmt, *arguments) == expected
    assert printf.sprintf(fmt, *arguments) == expected


def test_formatted_writes_report_counts_and_use_the_requested_channel(capsys) -> None:
    target = io.StringIO()
    assert printf.fprintf(target, "%04d", 7) == 4
    assert target.getvalue() == "0007"
    assert printf.printf("x=%d", 2) == 3
    assert printf.puts(" line") == 1 and printf.putchar(65) == 1
    assert printf.fprintf("stderr", "%s", "error") == 5
    captured = capsys.readouterr()
    assert captured.out == "x=2 line\nA" and captured.err == "error"


def test_missing_format_arguments_and_count_pointer_writes_raise_specific_errors() -> None:
    with pytest.raises(TypeError, match="wants more than the 0 arguments"):
        printf.cformat("%d")
    with pytest.raises(ValueError, match="%n writes a count through a pointer"):
        printf.cformat("%n", 0)
