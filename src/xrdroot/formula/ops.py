"""The arithmetic of an expression, on whole arrays, with ROOT's idea of each operator.

ROOT evaluates a ``TTree::Draw`` formula in ``double``: ``3/2`` is ``1.5``
and an ``int`` branch divided by another gives a fraction, not C's truncated
quotient. The operators that only mean something on integers - ``%``, the
bitwise ``&``, ``|``, ``<<``, ``>>`` and ``~`` - convert their operands to
64-bit integers first, truncating toward zero as C's conversion does.
Comparisons and ``&&``, ``||`` and ``!`` give booleans, which count as ``1``
and ``0`` in arithmetic after them.

Strings compare with ``==`` and ``!=`` and do nothing else; anything more
is refused with the column's name rather than compared as something it is
not.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError

__all__ = ["binary", "unary", "real", "truth", "integer", "is_text", "as_index", "cast"]

Array = Any


def is_text(values: Array) -> bool:
    return bool(values.dtype.kind in "US")


def _no_text(values: Array, what: str) -> None:
    if is_text(values):
        raise UnsupportedFeatureError(
            f"{what} was given strings; a string column compares with == and != against "
            f"a string in quotes, and is not a number to do anything else with"
        )


def real(values: Array, what: str = "arithmetic") -> Array:
    """The values as ``double``, as ROOT evaluates a formula."""
    _no_text(values, what)
    return values.astype(np.float64, copy=False)


def truth(values: Array, what: str = "a condition") -> Array:
    """Nonzero is true, as C tests a number."""
    _no_text(values, what)
    return values if values.dtype == np.bool_ else values != 0


def integer(values: Array, what: str = "an integer operator") -> Array:
    """The values as 64-bit integers, truncated toward zero as C converts a ``double``."""
    _no_text(values, what)
    if values.dtype.kind == "f":
        return np.trunc(np.nan_to_num(values)).astype(np.int64)
    return values.astype(np.int64, copy=False)


def _compare(ufunc: Callable[..., Array], op: str) -> Callable[[Array, Array], Array]:
    def apply(left: Array, right: Array) -> Array:
        if is_text(left) or is_text(right):
            if op not in ("==", "!=") or not (is_text(left) and is_text(right)):
                raise UnsupportedFeatureError(
                    f"{op} was given a string; a string compares with == and != against "
                    f"another string, and with nothing else"
                )
            return ufunc(left, right)
        return ufunc(real(left), real(right))

    return apply


def _arithmetic(ufunc: Callable[..., Array], op: str) -> Callable[[Array, Array], Array]:
    return lambda left, right: ufunc(real(left, op), real(right, op))


def _integral(ufunc: Callable[..., Array], op: str) -> Callable[[Array, Array], Array]:
    return lambda left, right: ufunc(integer(left, op), integer(right, op))


def _logical(ufunc: Callable[..., Array], op: str) -> Callable[[Array, Array], Array]:
    return lambda left, right: ufunc(truth(left, op), truth(right, op))


def _shift(ufunc: Callable[..., Array]) -> Callable[[Array, Array], Array]:
    """A shift by a count C would call undefined - negative, or 64 and more - gives zero."""

    def apply(left: Array, right: Array) -> Array:
        count = integer(right)
        fits = (count >= 0) & (count < 64)
        return np.where(fits, ufunc(integer(left), np.where(fits, count, 0)), 0)

    return apply


#: What each binary operator does to two arrays of values.
BINARY: dict[str, Callable[[Array, Array], Array]] = {
    "+": _arithmetic(np.add, "+"),
    "-": _arithmetic(np.subtract, "-"),
    "*": _arithmetic(np.multiply, "*"),
    "/": _arithmetic(np.true_divide, "/"),
    "^": _arithmetic(np.power, "^"),
    "%": _integral(np.fmod, "%"),
    "&": _integral(np.bitwise_and, "&"),
    "|": _integral(np.bitwise_or, "|"),
    "<<": _shift(np.left_shift),
    ">>": _shift(np.right_shift),
    "==": _compare(np.equal, "=="),
    "!=": _compare(np.not_equal, "!="),
    "<": _compare(np.less, "<"),
    "<=": _compare(np.less_equal, "<="),
    ">": _compare(np.greater, ">"),
    ">=": _compare(np.greater_equal, ">="),
    "&&": _logical(np.logical_and, "&&"),
    "||": _logical(np.logical_or, "||"),
}

#: What each unary operator does to an array of values.
UNARY: dict[str, Callable[[Array], Array]] = {
    "-": lambda values: -real(values, "-"),
    "+": lambda values: real(values, "+"),
    "!": lambda values: ~truth(values, "!"),
    "~": lambda values: ~integer(values, "~"),
}


def binary(op: str, left: Array, right: Array) -> Array:
    return BINARY[op](left, right)


def unary(op: str, values: Array) -> Array:
    return UNARY[op](values)


def as_index(values: Array) -> tuple[Array, Array | None]:
    """Values used as an index: truncated to integers as ``(Int_t)`` does, NaN never valid."""
    if values.dtype.kind != "f":
        return integer(values, "an index"), None
    finite = np.isfinite(values)
    whole = np.trunc(np.where(finite, values, -1.0)).astype(np.int64)
    return whole, None if bool(finite.all()) else finite


def cast(values: Array, dtype: Any, what: str) -> tuple[Array, Array | None]:
    """A C++ conversion, and where it had no value to give: a NaN made an ``int``.

    ``bool`` is nonzero, a floating type is the number rounded to it, and an
    integer type truncates toward zero and wraps as C++ narrowing does.
    """
    if dtype is np.bool_:
        return truth(values, what), None
    numbers = real(values, what)
    if np.dtype(dtype).kind == "f":
        return numbers.astype(dtype), None
    finite = np.isfinite(numbers)
    whole = np.trunc(np.where(finite, numbers, 0.0)).astype(np.int64).astype(dtype)
    return whole, None if bool(finite.all()) else finite
