"""``std::numeric_limits<T>``'s members, written as the constants they are.

Every member of ``numeric_limits`` of an arithmetic type is known when the
program is translated, so the translation writes the number - bit for bit
the one C++ gives on LP64, from :mod:`xrdroot.cppstd.stdlib` - rather than
a call: ``numeric_limits<float>::epsilon()`` is ``1.1920928955078125e-07``.
"""

from __future__ import annotations

import math
from typing import Any

from xrdroot.cppstd.stdlib import FUNCTIONS, limit

from .ctype import CType

__all__ = ["FUNCTIONS", "limit_text", "limit_type", "is_limit"]

#: The members of ``numeric_limits`` that are ``bool`` constants; the other constants are ``int``.
FLAGS = frozenset(
    """is_signed is_integer is_exact has_infinity has_quiet_NaN has_signaling_NaN is_iec559
    is_modulo is_specialized is_bounded""".split()
)


def is_limit(parts: list[str], targs: list[Any] | None) -> bool:
    """Is ``parts`` a member of ``std::numeric_limits<T>`` - ``numeric_limits<int>::max``?"""
    return len(parts) >= 2 and parts[-2] == "numeric_limits" and bool(targs)


def limit_text(kind: Any, member: str) -> str | None:
    """The Python constant ``numeric_limits<kind>::member`` is, or ``None`` when ``kind`` is no
    arithmetic type this knows, or ``member`` no member of it."""
    if not isinstance(kind, CType) or kind.pointer or kind.dims:
        return None
    value = limit(kind.name, member)
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return f"float({str(value)!r})"
    return repr(value)


def limit_type(kind: Any, member: str) -> CType | None:
    """The C++ type of ``numeric_limits<kind>::member``: ``kind`` for a function, ``bool`` for
    a flag, ``int`` for the other constants."""
    if member in FUNCTIONS:
        return kind.value() if isinstance(kind, CType) else None
    return CType("bool" if member in FLAGS else "int")
