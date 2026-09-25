"""``printf`` and its family, printing exactly what C's would.

Python's ``%`` formatting is C's, nearly: the flags, widths, precisions and
``%d %e %f %g %x %o %s %c`` agree character for character, ``%g`` included.
What differs is handled here: the length modifiers (``%ld``, ``%lld``,
``%hd``, ``%zu``) are dropped after they have said how wide the argument is,
``%u``, ``%x`` and ``%o`` show a negative number as C does - its two's
complement at that width - ``%s`` of a null pointer is ``(null)``, and the
thousands flag ``'`` is ignored as the C locale does.
"""

from __future__ import annotations

import re
import sys
from typing import Any

__all__ = ["cformat", "printf", "Printf", "sprintf", "Form", "fprintf", "puts", "putchar",
           "Info", "Warning", "Error", "Fatal", "Printf_"]

#: One conversion specification.
SPEC = re.compile(
    r"%(?P<flags>[-+ #0']*)(?P<width>\*|\d+)?(?:\.(?P<prec>\*|\d*))?"
    r"(?P<length>hh|h|ll|l|L|q|j|z|t|I64|I32)?(?P<conv>[diouxXeEfFgGaAcspn%])"
)

#: How many bits an integer argument has, by its length modifier.
BITS = {"hh": 8, "h": 16, None: 32, "l": 64, "ll": 64, "q": 64, "j": 64, "z": 64, "t": 64,
        "L": 64, "I64": 64, "I32": 32}


def _signed(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def _integer(value: Any) -> int:
    if isinstance(value, str):
        return ord(value[:1] or "\0")
    return int(value)


class _Arguments:
    def __init__(self, fmt: str, args: tuple[Any, ...]) -> None:
        self.fmt = fmt
        self.args = args
        self.at = 0

    def next(self) -> Any:
        if self.at >= len(self.args):
            raise TypeError(
                f"the format {self.fmt!r} wants more than the {len(self.args)} "
                f"argument{'s' if len(self.args) != 1 else ''} it was given"
            )
        value = self.args[self.at]
        self.at += 1
        return value


def _value(conv: str, length: str | None, value: Any) -> Any:
    bits = BITS[length]
    if conv in "di":
        return _signed(_integer(value), bits) if length in ("hh", "h") else _integer(value)
    if conv in "ouxX":
        return _integer(value) & ((1 << bits) - 1)
    if conv in "eEfFgGaA":
        return float(value)
    if conv == "c":
        return chr(_integer(value) & 0xFF) if not isinstance(value, str) else value[:1]
    if conv == "p":
        return id(value)
    return "(null)" if value is None else str(value)


def _conversion(found: re.Match[str], arguments: _Arguments) -> str:
    conv = found["conv"]
    if conv == "%":
        return "%"
    if conv == "n":
        raise ValueError("%n writes a count through a pointer, which printf here does not do")
    width = found["width"]
    if width == "*":
        width = str(int(arguments.next()))
    prec = found["prec"]
    if prec == "*":
        prec = str(int(arguments.next()))
    flags = found["flags"].replace("'", "")
    if width and width.startswith("-"):
        flags, width = flags + "-", width[1:]
    spec = "%" + flags + (width or "") + ("" if prec is None else "." + (prec or "0"))
    value = _value(conv, found["length"], arguments.next())
    if conv == "p":
        return (spec + "x") % value if value else "(nil)"
    if conv in "aA":
        text = float.hex(value)
        return text.upper() if conv == "A" else text
    return (spec + {"i": "d", "u": "d"}.get(conv, conv)) % value


def cformat(fmt: Any, *args: Any) -> str:
    """What C's ``sprintf(buffer, fmt, args...)`` would leave in the buffer."""
    arguments = _Arguments(str(fmt), args)
    return SPEC.sub(lambda found: _conversion(found, arguments), str(fmt))


def printf(fmt: Any, *args: Any) -> int:
    """C's ``printf``: the formatted text on standard output, and how many characters it was."""
    text = cformat(fmt, *args)
    sys.stdout.write(text)
    return len(text)


def fprintf(stream: Any, fmt: Any, *args: Any) -> int:
    """C's ``fprintf`` to ``stdout``/``stderr`` (or any object with a ``write``)."""
    text = cformat(fmt, *args)
    target = {"stdout": sys.stdout, "stderr": sys.stderr}.get(stream, stream)
    target.write(text)
    return len(text)


def puts(text: Any) -> int:
    sys.stdout.write(f"{text}\n")
    return 1


def putchar(char: Any) -> int:
    sys.stdout.write(chr(char) if isinstance(char, int) else str(char)[:1])
    return 1


def Printf(fmt: Any, *args: Any) -> None:
    """ROOT's ``Printf``: ``printf`` with a newline after."""
    sys.stdout.write(cformat(fmt, *args) + "\n")


#: ``Printf`` under a name no macro's own ``Printf`` method can hide.
Printf_ = Printf


def sprintf(fmt: Any, *args: Any) -> str:
    """The text ``sprintf`` writes; the translation assigns it to the buffer named."""
    return cformat(fmt, *args)


def Form(fmt: Any, *args: Any) -> str:
    """ROOT's ``Form``: ``sprintf`` into a buffer of ROOT's, handed back."""
    return cformat(fmt, *args)


def _message(level: str, location: Any, fmt: Any, args: tuple[Any, ...]) -> None:
    where = f" in <{location}>" if location else ""
    sys.stderr.write(f"{level}{where}: {cformat(fmt, *args)}\n")


def Info(location: Any, fmt: Any, *args: Any) -> None:
    """ROOT's ``Info(where, fmt, ...)``: ``Info in <where>: message`` on standard error."""
    _message("Info", location, fmt, args)


def Warning(location: Any, fmt: Any, *args: Any) -> None:  # noqa: A001
    _message("Warning", location, fmt, args)


def Error(location: Any, fmt: Any, *args: Any) -> None:
    _message("Error", location, fmt, args)


def Fatal(location: Any, fmt: Any, *args: Any) -> None:
    _message("Fatal", location, fmt, args)
    raise SystemExit(1)
