"""Values as GoogleTest prints and compares them: ``PrintToString``, equality, and ULPs.

A failure says what each side was - ``Which is: 3`` - in GoogleTest's
spelling: ``true``, ``NULL``, ``"text"`` quoted and escaped, ``{ 1, 2 }``
for a container, and a floating-point value with six significant digits
when that is exact, else all seventeen. ``FLOAT_EQ`` and ``DOUBLE_EQ`` hold
within four units in the last place, as GoogleTest's ``AlmostEquals``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy

__all__ = ["show", "equal", "almost_equal", "ulps", "text_of"]

#: How many of a container's elements GoogleTest prints before ``...``.
SHOWN = 32

#: The escapes GoogleTest writes inside a quoted string.
ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\t": "\\t", "\r": "\\r"}


def show(value: Any) -> str:
    """``value`` as GoogleTest's ``PrintToString`` writes it."""
    if isinstance(value, (bool, numpy.bool_)):
        return "true" if value else "false"
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return '"' + "".join(ESCAPES.get(char, char) for char in value) + '"'
    if isinstance(value, (float, numpy.floating)):
        return _floating(value)
    if isinstance(value, (int, numpy.integer)):
        return str(int(value))
    return _compound(value)


def _floating(value: Any) -> str:
    """Six significant digits if they give the value back exactly, else every digit there is."""
    single = isinstance(value, numpy.float32)
    number = float(value)
    if not math.isfinite(number):
        return str(number)
    six = f"{number:.6g}"
    exact = numpy.float32(six) == value if single else float(six) == number
    return six if exact else f"{number:.{9 if single else 17}g}"


def _compound(value: Any) -> str:
    """A tuple as ``(a, b)``, a container as ``{ a, b }``, anything else as Python writes it."""
    if isinstance(value, tuple):
        return "(" + ", ".join(show(item) for item in value) + ")"
    if not hasattr(value, "__len__") or not hasattr(value, "__iter__"):
        return str(value)
    items = [show(item) for _, item in zip(range(SHOWN + 1), value, strict=False)]
    if not items:
        return "{}"
    more = ", ..." if len(items) > SHOWN else ""
    return "{ " + ", ".join(items[:SHOWN]) + more + " }"


def text_of(value: Any) -> str | None:
    """A C string's text - ``None`` for a null pointer - as ``EXPECT_STREQ`` compares it."""
    return None if value is None else str(value)


def equal(left: Any, right: Any) -> bool:
    """``left == right`` as C++ means it, for ``EXPECT_EQ``.

    A character the translation made a one-letter string equals its code, as
    a ``char`` does; a comparison made element by element - of two arrays -
    is equality when every element is equal.
    """
    if left is right:
        return True
    left, right = _character(left, right), _character(right, left)
    outcome = left == right
    if isinstance(outcome, (bool, numpy.bool_)):
        return bool(outcome)
    return bool(numpy.all(outcome))


def _character(value: Any, other: Any) -> Any:
    """A one-letter string as its code, when it is compared with an integer."""
    if isinstance(value, str) and len(value) == 1 and isinstance(other, int):
        return ord(value)
    return value


def ulps(left: float, right: float, single: bool) -> int:
    """How many representable values apart ``left`` and ``right`` are, as GoogleTest counts."""
    return abs(_biased(left, single) - _biased(right, single))


def _biased(value: float, single: bool) -> int:
    """The bits of ``value`` in GoogleTest's biased order, where adjacent values differ by one."""
    kind, unsigned = (numpy.float32, numpy.uint32) if single else (numpy.float64, numpy.uint64)
    with numpy.errstate(over="ignore"):  # a double past a float's range is a float's infinity
        bits = int(numpy.array(value, dtype=kind).view(unsigned))
    sign = 1 << (31 if single else 63)
    return sign - (bits & (sign - 1)) if bits & sign else sign | bits


def almost_equal(left: Any, right: Any, single: bool) -> bool:
    """``FLOAT_EQ`` or ``DOUBLE_EQ``: within four units in the last place, and neither NaN."""
    first, second = float(left), float(right)
    if math.isnan(first) or math.isnan(second):
        return False
    return ulps(first, second, single) <= 4
