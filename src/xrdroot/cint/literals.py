"""What a C++ number literal is worth, and which C++ type it has.

``10`` is an ``int``, ``10u`` an ``unsigned int``, ``10L`` a ``long``,
``0x1F`` hexadecimal, ``017`` octal, ``0b101`` binary and ``1'000`` a
thousand; ``1.``, ``.5``, ``1e-3`` are ``double`` and ``1.5f`` a ``float``.
The type matters because the translator does C's arithmetic: an ``int``
divided by an ``int`` truncates, a ``float`` stored is rounded to 32 bits.
"""

from __future__ import annotations

import re

from .errors import Refusal, Where

__all__ = ["number", "user_literal", "SUFFIXES"]

#: The user-defined literal suffixes whose ``operator""`` the library defines, and whose
#: value the runtime's ``user_literal`` makes: ROOT 7's pad lengths and ``std::chrono``'s.
SUFFIXES = frozenset({"_normal", "_px", "_user", "ns", "us", "ms", "s", "min", "h"})

#: A number followed by a suffix that is no C++ number's own.
SUFFIXED = re.compile(r"^(?P<number>[\d.']+(?:[eE][+-]?\d+)?)(?P<suffix>[A-Za-z_]\w*)$")

#: An integer literal: its digits in some base, then its suffix.
INTEGER = re.compile(
    r"(?P<digits>0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]*|[1-9][0-9]*)(?P<suffix>[uUlLzZ]*)$"
)

#: A floating literal, decimal or hexadecimal, then its suffix.
FLOATING = re.compile(
    r"(?P<digits>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|0[xX][0-9a-fA-F.]+[pP][+-]?\d+)"
    r"(?P<suffix>[fFlL]?)$"
)


def _integer_type(suffix: str) -> str:
    lower = suffix.lower()
    unsigned = "u" in lower
    count = lower.count("l")
    base = {0: "int", 1: "long", 2: "long long"}[min(count, 2)]
    return f"unsigned {base}" if unsigned else base


def _integer(found: re.Match[str]) -> tuple[int, str]:
    digits = found["digits"]
    lower = digits.lower()
    if lower.startswith("0x"):
        value = int(digits[2:], 16)
    elif lower.startswith("0b"):
        value = int(digits[2:], 2)
    elif len(digits) > 1 and digits.startswith("0"):
        value = int(digits, 8)
    else:
        value = int(digits)
    return value, _integer_type(found["suffix"])


def _floating(found: re.Match[str]) -> tuple[float, str]:
    digits = found["digits"]
    value = float.fromhex(digits) if digits.lower().startswith("0x") else float(digits)
    kind = "float" if found["suffix"] in ("f", "F") else "double"
    return value, kind


def number(text: str, where: Where) -> tuple[int | float, str]:
    """The value of the number literal ``text``, and the name of its C++ type."""
    plain = text.replace("'", "")
    integer = INTEGER.match(plain)
    if integer is not None:
        return _integer(integer)
    floating = FLOATING.match(plain)
    if floating is not None:
        return _floating(floating)
    if re.match(r"^[\d.']+(?:[eE][+-]?\d+)?[A-Za-z_]\w*$", text):
        why = f'{text} is a user-defined literal, which calls an operator"" of its own'
        raise Refusal(why, where)
    raise Refusal(f"{text} is not a number literal C++ has", where)


def user_literal(text: str) -> tuple[str, str] | None:
    """``0.1_normal`` as ``("0.1", "_normal")`` when the suffix is one of :data:`SUFFIXES`."""
    found = SUFFIXED.match(text)
    if found is None or found["suffix"] not in SUFFIXES:
        return None
    return found["number"], found["suffix"]
