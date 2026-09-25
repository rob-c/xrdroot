"""C's arithmetic where Python's differs: division, remainder, conversions and wrapping.

``7 / 2`` is ``3`` in C and ``3.5`` in Python, and ``-7 / 2`` is ``-3`` in C
where Python's ``//`` gives ``-4``: C truncates toward zero. ``-7 % 2`` is
``-1`` in C, ``1`` in Python. An ``unsigned int`` wraps at 2^32, an ``int``
stored from ``3.9`` is ``3``, a ``float`` keeps 24 bits of mantissa. The
translator writes these helpers where the types say C and Python part ways,
and :func:`div` where it cannot tell: that one decides by what it is given.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "idiv",
    "imod",
    "div",
    "mod",
    "to_int",
    "f32",
    "wrap",
    "u8",
    "u16",
    "u32",
    "u64",
    "i8",
    "i16",
    "i32",
    "i64",
    "comma",
    "c_exit",
]


def _integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer))


def idiv(a: Any, b: Any) -> int:
    """``a / b`` for two integers, truncated toward zero as C does."""
    a, b = int(a), int(b)
    if b == 0:
        raise ZeroDivisionError("integer division by zero, which C++ leaves undefined")
    quotient = abs(a) // abs(b)
    return quotient if (a < 0) == (b < 0) else -quotient


def imod(a: Any, b: Any) -> int:
    """``a % b`` for two integers, with the sign of ``a`` as C gives it."""
    a, b = int(a), int(b)
    return a - b * idiv(a, b)


def div(a: Any, b: Any) -> Any:
    """``a / b`` when the types were not known: integral if both are, as C would."""
    if _integer(a) and _integer(b):
        return idiv(a, b)
    return a / b


def mod(a: Any, b: Any) -> Any:
    """``a % b`` when the types were not known."""
    if _integer(a) and _integer(b):
        return imod(a, b)
    return math.fmod(a, b)


def to_int(value: Any) -> int:
    """A value stored into an integer: truncated toward zero, as C converts."""
    if isinstance(value, str):
        return ord(value[:1] or "\0")
    number = float(value) if not _integer(value) else value
    if isinstance(number, float) and not math.isfinite(number):
        raise OverflowError(f"{number} does not fit in an integer, which C++ leaves undefined")
    return int(number)


def f32(value: Any) -> float:
    """A value stored into a ``float``: rounded to single precision, handed back as a float."""
    return float(np.float32(value))


def wrap(value: Any, bits: int, signed: bool) -> int:
    """``value`` reduced to an integer ``bits`` wide, two's complement if ``signed``."""
    number = to_int(value) & ((1 << bits) - 1)
    if signed and number >> (bits - 1):
        return number - (1 << bits)
    return number


def u8(value: Any) -> int:
    return wrap(value, 8, False)


def u16(value: Any) -> int:
    return wrap(value, 16, False)


def u32(value: Any) -> int:
    return wrap(value, 32, False)


def u64(value: Any) -> int:
    return wrap(value, 64, False)


def i8(value: Any) -> int:
    return wrap(value, 8, True)


def i16(value: Any) -> int:
    return wrap(value, 16, True)


def i32(value: Any) -> int:
    return wrap(value, 32, True)


def i64(value: Any) -> int:
    return wrap(value, 64, True)


def comma(*values: Any) -> Any:
    """C's comma operator: everything evaluated, left to right, and the last one's value."""
    return values[-1]


def c_exit(code: Any = 0) -> Any:
    """C's ``exit(code)``: the macro stops, with that status."""
    raise SystemExit(int(code))
