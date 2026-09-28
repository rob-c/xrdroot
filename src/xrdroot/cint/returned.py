"""What ``root -q file.C`` does with the value the macro's function returns.

ROOT runs the macro as cling's ``.x file.C``, and cling prints the value of
what it ran - ``(int) 3``, ``(TCanvas *) 0x7fd1...``, ``(double) 2.7000000``
- on a line of its own after everything the macro printed. Quitting, ROOT's
TRint then exits with that value taken as an integer: ``0`` to ``255`` as it
is, anything else - a negative number, ``300``, a pointer's address, an
object's - as ``255``; a double is truncated first, a null pointer is ``0``
and a ``void`` function gives ``0``. That is why CI expects ``hsimple.C``,
which returns its ``TFile *``, to exit 255.

The spellings and formats here are ROOT 6.40's own, from macros returning
each kind of value run under ``root -b -q``: ``(long long) 7`` for a
``Long64_t``, ``(bool) true``, ``(char) 'A'``, ``(float) 2.50000f``, a double
to eight significant digits with its point kept (``(double) 0.10000000``,
``(double) 1.0000000e+20``), ``(const char *) "abc"``, ``(std::string)
"abc"``, ``(TString) "abc"[3]`` and ``(TObject *) nullptr``.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable
from typing import Any

from .ctype import INTEGRAL, CType

__all__ = ["spelling", "shown", "status"]

#: The largest exit status ROOT hands on as it is; any other value exits as this.
LARGEST = 255

#: The character types, whose values cling prints quoted.
CHARACTERS = frozenset({"char", "signed char", "unsigned char"})

#: The string classes cling prints as quoted text, as it spells them.
STRING_CLASSES = {"std::string": "std::string", "string": "std::string", "TString": "TString"}


def spelling(returns: CType) -> str | None:
    """How cling names a function's return type: ``TCanvas *``, ``int`` - None for ``void``.

    Only the types this module knows cling's printing of are named: a
    ``long double``, a class held by value or reference, ``auto`` and the
    like are not, and nothing is printed for them.
    """
    if returns.reference or returns.dims or returns.is_void or returns.args:
        return None
    if returns.pointer:
        const = "const " if returns.const else ""
        return f"{const}{returns.name} {'*' * returns.pointer}"
    if returns.name in INTEGRAL or returns.name in ("float", "double"):
        return returns.name
    return STRING_CLASSES.get(returns.name)


def _pointer(value: Any) -> str:
    if value is None:
        return "nullptr"
    if isinstance(value, str):
        return _quoted(value)
    return f"0x{id(value):x}"


def _quoted(text: str) -> str:
    return '"' + text + '"'


def _character(value: Any) -> str:
    return "'" + (value if isinstance(value, str) else chr(int(value))) + "'"


def _integer(value: Any) -> str:
    return str(int(value))


def _bool(value: Any) -> str:
    return "true" if value else "false"


def _float(value: Any) -> str:
    return f"{float(value):#.6g}f"


def _double(value: Any) -> str:
    return f"{float(value):#.8g}"


def _tstring(value: Any) -> str:
    text = str(value)
    return f"{_quoted(text)}[{len(text)}]"


#: How cling writes a value of each type named by :func:`spelling`, pointers aside.
FORMATS: dict[str, Callable[[Any], str]] = {
    **dict.fromkeys(INTEGRAL, _integer),
    **dict.fromkeys(CHARACTERS, _character),
    "bool": _bool,
    "float": _float,
    "double": _double,
    "std::string": lambda value: _quoted(str(value)),
    "TString": _tstring,
}


def shown(value: Any, spelled: str | None) -> str | None:
    """The line cling prints for ``value`` returned as ``spelled`` - None when it prints none.

    A function declared to return a number that falls off its end without
    one leaves C++'s value undefined; nothing is made up for it.
    """
    if spelled is None or (value is None and not spelled.endswith("*")):
        return None
    written = _pointer(value) if spelled.endswith("*") else FORMATS[spelled](value)
    return f"({spelled}) {written}"


def status(value: Any, spelled: str | None) -> int:
    """The exit status ``root -q`` gives after a macro returned ``value`` as ``spelled``."""
    if value is None:
        return 0
    if isinstance(value, str) and spelled in CHARACTERS:
        value = ord(value)
    if not isinstance(value, numbers.Real) or (spelled or "").endswith("*"):
        # A pointer, a string or an object: its address, which no status holds.
        return LARGEST
    if not math.isfinite(float(value)):
        return LARGEST
    whole = int(float(value))
    return whole if 0 <= whole <= LARGEST else LARGEST
