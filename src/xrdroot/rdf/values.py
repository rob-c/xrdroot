"""What an ``RDataFrame`` expression works on: a batch of values, and C++'s arithmetic on it.

A column in an expression is one of three things for a batch of entries: a
number per entry, a collection per entry - an ``RVec`` - or a constant. A
:class:`Value` holds any of them the same way: ``data``, and for a
collection the ``offsets`` that cut ``data`` into one row per entry. Two
values meet element by element, as ``RVec`` arithmetic does; a number per
entry is repeated to its entry's elements, and two collections must be the
same size in every entry, as ROOT insists.

The arithmetic is C++'s rather than ``TTree::Draw``'s, because an
``RDataFrame`` expression is C++: integers stay integers and divide as C
divides them, ``float`` stays ``float``, the small integer types and ``bool``
are promoted to ``int`` first, and ``^`` is exclusive or. Comparing
collections gives ``RVec<int>``, as ``ROOT::VecOps`` does, and comparing
numbers gives ``bool``.

Where C++ is undefined - an integer divided by zero, an index past the end
of a collection - nothing is made up: :class:`Undefined` is raised with the
row it happened in, and the expression says which entry that was.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from ..tree import Jagged
from . import kernels

__all__ = [
    "Value",
    "Mismatch",
    "Undefined",
    "from_column",
    "to_column",
    "align",
    "common",
    "promoted",
    "binary",
    "unary",
    "truth",
    "subset",
    "per_entry",
    "undefined_where",
]

Array = Any


class Mismatch(ValueError):
    """Two collections of different sizes met in one entry: which row of the batch."""

    def __init__(self, what: str, row: int, sizes: tuple[int, int]) -> None:
        super().__init__(what, row, sizes)
        self.what = what
        self.row = row
        self.sizes = sizes


class Undefined(ValueError):
    """Something C++ leaves undefined happened, in this row of the batch."""

    def __init__(self, what: str, row: int) -> None:
        super().__init__(what, row)
        self.what = what
        self.row = row


def undefined_where(bad: Array, what: str) -> None:
    """Raise :class:`Undefined` for the first row ``bad`` marks, if it marks any."""
    if np.any(bad):
        raise Undefined(what, int(np.flatnonzero(bad)[0]))


class Value:
    """A batch of one expression's values: numbers, collections, or a group of collections."""

    __slots__ = ("data", "offsets", "items")

    def __init__(
        self, data: Array, offsets: Array | None = None, items: tuple[Value, ...] | None = None
    ) -> None:
        self.data = data
        #: Where each entry's elements start and stop, for a collection; else ``None``.
        self.offsets = offsets
        #: The members of what ``Combinations`` gives - several collections - else ``None``.
        self.items = items

    @property
    def dtype(self) -> Any:
        return np.asarray(self.data).dtype


def subset(value: Value, rows: Array) -> Value:
    """The same value for only the rows given, in the order given."""
    if value.items is not None:
        return Value(value.data, items=tuple(subset(item, rows) for item in value.items))
    if value.offsets is not None:
        content, offsets = kernels.compact(Jagged(value.data, value.offsets).take(rows))
        return Value(content, offsets)
    return Value(value.data[rows])  # a column is never a constant: one value per row


def _fixed_rows(values: Array) -> Value:
    """A column of fixed-size arrays, which ROOT reads as an ``RVec`` of them all."""
    width = int(np.prod(values.shape[1:]))
    offsets = np.arange(len(values) + 1, dtype=np.int64) * width
    return Value(values.reshape(-1), offsets)


def _jagged_rows(values: Jagged) -> Value:
    content, offsets = kernels.compact(values)
    if content.ndim > 1:  # x[n][3]: ROOT flattens the fixed part into the RVec too
        width = int(np.prod(content.shape[1:]))
        return Value(content.reshape(-1), offsets * width)
    return Value(content, offsets)


def _refused(name: str, what: str) -> UnsupportedFeatureError:
    return UnsupportedFeatureError(
        f"{name!r} holds {what}, which a string expression does not compute with; give "
        f"Define or Filter a Python callable, which is handed the column as it is read"
    )


def _listed(name: str, values: list[Any]) -> Value:
    if all(isinstance(value, str) for value in values):
        return Value(np.asarray(values, dtype=str))
    raise _refused(name, "Python objects - nested collections, maps or classes")


def from_column(name: str, values: Any) -> Value:
    """A column, as reading or defining it gave it, as a value to compute with."""
    if isinstance(values, Jagged):
        return _jagged_rows(values)
    if isinstance(values, tuple):
        return Value(np.zeros(0), items=tuple(from_column(name, item) for item in values))
    if isinstance(values, list):
        return _listed(name, values)
    values = np.asarray(values)
    if values.dtype == object:
        raise _refused(name, "Python objects")
    if values.ndim > 1:
        return _fixed_rows(values)
    return Value(values)


def to_column(value: Value, count: int) -> Any:
    """A value as a column: an array of one per entry, a ``Jagged``, or a tuple of them."""
    if value.items is not None:
        return tuple(to_column(item, count) for item in value.items)
    if value.offsets is not None:
        return Jagged(value.data, value.offsets)
    if np.ndim(value.data) == 0:
        return np.full(count, value.data, dtype=np.asarray(value.data).dtype)
    return value.data


def _first_difference(first: Array, second: Array) -> int:
    return int(np.flatnonzero(np.diff(first) != np.diff(second))[0])


def _shared_offsets(values: Sequence[Value], what: str) -> Array | None:
    offsets = None
    for value in values:
        if value.offsets is None:
            continue
        if offsets is None:
            offsets = value.offsets
        elif not np.array_equal(offsets, value.offsets):
            row = _first_difference(offsets, value.offsets)
            sizes = (int(np.diff(offsets)[row]), int(np.diff(value.offsets)[row]))
            raise Mismatch(what, row, sizes)
    return offsets


def _spread(value: Value, offsets: Array | None) -> Array:
    data = value.data
    if value.offsets is not None or offsets is None or np.ndim(data) == 0:
        return data
    return np.repeat(data, np.diff(offsets))


def align(values: Sequence[Value], what: str) -> tuple[list[Array], Array | None]:
    """Values lined up element for element: their arrays, and the offsets they share."""
    for value in values:
        if value.items is not None:
            raise UnsupportedFeatureError(
                f"{what} was given what Combinations gives - several collections at once - "
                f"rather than one; take one of them with [0] or [1] first"
            )
    offsets = _shared_offsets(values, what)
    return [np.asarray(_spread(value, offsets)) for value in values], offsets


def per_entry(elements: Array, offsets: Array | None) -> Array:
    """A mark on elements as a mark on the entries holding any of them."""
    if offsets is None:
        return elements
    return kernels.reduced(np.logical_or, elements, offsets, False)[0]


# -- C++'s types ---------------------------------------------------------------


def promoted(dtype: Any) -> Any:
    """C++'s integral promotion: ``bool``, ``char`` and ``short`` become ``int``."""
    dtype = np.dtype(dtype)
    if dtype.kind == "b" or (dtype.kind in "iu" and dtype.itemsize < 4):
        return np.dtype(np.int32)
    return dtype


def no_text(dtype: Any, what: str) -> None:
    if np.dtype(dtype).kind in "US":
        raise UnsupportedFeatureError(
            f"{what} was given a string; a string compares with == and != against another "
            f"string, and is not a number to do anything else with"
        )


def common(first: Any, second: Any, what: str = "arithmetic") -> Any:
    """C++'s usual arithmetic conversions: the type two operands meet in.

    The wider floating type if either is floating; otherwise the wider of
    two integers, an unsigned one winning a tie, as C++ has it.
    """
    no_text(first, what)
    no_text(second, what)
    a, b = promoted(first), promoted(second)
    floats = [dtype for dtype in (a, b) if dtype.kind == "f"]
    if floats:
        return max(floats, key=lambda dtype: dtype.itemsize)
    if a.kind == b.kind:
        return a if a.itemsize >= b.itemsize else b
    signed, unsigned = (a, b) if a.kind == "i" else (b, a)
    return unsigned if unsigned.itemsize >= signed.itemsize else signed


def truth(data: Array, what: str = "a condition") -> Array:
    """Nonzero is true, as C++ converts a number to ``bool``."""
    no_text(np.asarray(data).dtype, what)
    return np.asarray(data) != 0


# -- the operators ---------------------------------------------------------------


def _integers(first: Array, second: Array, op: str) -> Any:
    kind = common(first.dtype, second.dtype, op)
    if kind.kind == "f":
        raise UnsupportedFeatureError(
            f"C++'s {op} is for integers, and was given a floating-point number; cast it "
            f"to an integer type first" + ("; fmod is the remainder of two" if op == "%" else "")
        )
    return kind


def _divide(a: Array, b: Array, kind: Any) -> tuple[Array, Array | None]:
    """``/``: true division of floating types, and C's truncating division of integers."""
    a, b = a.astype(kind), b.astype(kind)
    if kind.kind == "f":
        return np.true_divide(a, b), None
    zero = b == 0
    safe = np.where(zero, 1, b).astype(kind)
    floor = a // safe
    truncated = np.where((a % safe != 0) & ((a < 0) != (safe < 0)), floor + 1, floor)
    return truncated.astype(kind), zero


def _remainder(a: Array, b: Array, kind: Any) -> tuple[Array, Array | None]:
    """``%``: the remainder with the sign of the dividend, as C's is."""
    a, b = a.astype(kind), b.astype(kind)
    zero = b == 0
    return np.fmod(a, np.where(zero, 1, b).astype(kind)), zero


def _arithmetic(ufunc: Any) -> Any:
    def apply(a: Array, b: Array, op: str) -> tuple[Array, Array | None]:
        kind = common(a.dtype, b.dtype, op)
        return ufunc(a.astype(kind), b.astype(kind)), None

    return apply


def _dividing(kernel: Any, integral: bool) -> Any:
    def apply(a: Array, b: Array, op: str) -> tuple[Array, Array | None]:
        kind = _integers(a, b, op) if integral else common(a.dtype, b.dtype, op)
        return kernel(a, b, kind)  # type: ignore[no-any-return]

    return apply


def _bitwise(ufunc: Any) -> Any:
    def apply(a: Array, b: Array, op: str) -> tuple[Array, Array | None]:
        kind = _integers(a, b, op)
        return ufunc(a.astype(kind), b.astype(kind)), None

    return apply


def _shift(ufunc: Any) -> Any:
    """A shift: of the promoted left operand, zero for a count C++ would call undefined."""

    def apply(a: Array, b: Array, op: str) -> tuple[Array, Array | None]:
        _integers(a, b, op)
        kind = promoted(a.dtype)
        count = b.astype(np.int64)
        fits = (count >= 0) & (count < kind.itemsize * 8)
        shifted = ufunc(a.astype(kind), np.where(fits, count, 0).astype(kind))
        return np.where(fits, shifted, 0).astype(kind), None

    return apply


def _compare(ufunc: Any) -> Any:
    def apply(a: Array, b: Array, op: str) -> tuple[Array, Array | None]:
        text = [side.dtype.kind in "US" for side in (a, b)]
        if any(text) and (op not in ("==", "!=") or not all(text)):
            raise UnsupportedFeatureError(
                f"{op} was given a string; a string compares with == and != against "
                f"another string, and with nothing else"
            )
        return (a == b if op == "==" else a != b) if any(text) else ufunc(a, b), None

    return apply


def _logical(ufunc: Any) -> Any:
    def apply(a: Array, b: Array, op: str) -> tuple[Array, Array | None]:
        return ufunc(truth(a, op), truth(b, op)), None

    return apply


#: What each binary operator does to two aligned arrays, and where C++ is undefined.
BINARY = {
    "+": _arithmetic(np.add),
    "-": _arithmetic(np.subtract),
    "*": _arithmetic(np.multiply),
    "/": _dividing(_divide, integral=False),
    "%": _dividing(_remainder, integral=True),
    "&": _bitwise(np.bitwise_and),
    "|": _bitwise(np.bitwise_or),
    "^": _bitwise(np.bitwise_xor),
    "<<": _shift(np.left_shift),
    ">>": _shift(np.right_shift),
    "==": _compare(np.equal),
    "!=": _compare(np.not_equal),
    "<": _compare(np.less),
    "<=": _compare(np.less_equal),
    ">": _compare(np.greater),
    ">=": _compare(np.greater_equal),
    "&&": _logical(np.logical_and),
    "||": _logical(np.logical_or),
}

#: The operators whose answer is true or false: ``bool`` per entry, ``int`` in an ``RVec``.
TESTS = ("==", "!=", "<", "<=", ">", ">=", "&&", "||")


def binary(op: str, left: Value, right: Value) -> Value:
    """``left op right``, element by element, with C++'s types."""
    (a, b), offsets = align((left, right), f"the operator {op}")
    with np.errstate(all="ignore"):
        data, undefined = BINARY[op](a, b, op)
    if undefined is not None:
        undefined_where(per_entry(undefined, offsets), f"an integer {op} by zero")
    if op in TESTS and offsets is not None:
        data = data.astype(np.int32)  # ROOT::VecOps compares into an RVec<int>
    return Value(data, offsets)


def _negate(data: Array) -> Array:
    no_text(data.dtype, "-")
    return -data.astype(promoted(data.dtype))


def _plus(data: Array) -> Array:
    no_text(data.dtype, "+")
    return data.astype(promoted(data.dtype))


def _invert(data: Array) -> Array:
    kind = promoted(data.dtype)
    if kind.kind not in "iu":
        raise UnsupportedFeatureError("C++'s ~ is for integers, and was given something else")
    return ~data.astype(kind)


def _not(data: Array) -> Array:
    return ~truth(data, "!")


#: What each unary operator does to an array.
UNARY = {"-": _negate, "+": _plus, "~": _invert, "!": _not}


def unary(op: str, operand: Value) -> Value:
    """``op operand``, element by element."""
    (data,), offsets = align((operand,), f"the operator {op}")
    out = UNARY[op](data)
    if op == "!" and offsets is not None:
        out = out.astype(np.int32)
    return Value(out, offsets)
