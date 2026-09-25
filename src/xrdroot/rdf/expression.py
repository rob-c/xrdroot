"""A compiled ``RDataFrame`` expression, and its value for a batch of entries at once.

    >>> e = Expression("Sum(jet_pt > 30) >= 2", ["jet_pt", "nJet"])   # doctest: +SKIP
    >>> e.columns
    ('jet_pt',)

Evaluating walks the parsed tree once per batch: every operator and function
works on the whole batch's values, so nothing loops over entries in Python.
What C++ evaluates conditionally is evaluated conditionally here too. The
right side of ``&&`` and ``||``, and each branch of ``? :``, is evaluated
only for the entries that reach it - so ``nJet > 0 && jet_pt[0] > 30`` never
looks at the first jet of an entry that has none, and two collections added
together in a branch that only runs where they are the same size are never
added where they are not.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from ..tree import Jagged
from . import functions, kernels
from .language import (
    Binary,
    Call,
    Cast,
    Column,
    Index,
    Literal,
    Method,
    Node,
    Ternary,
    Unary,
    columns_of,
    parse,
)
from .values import (
    Mismatch,
    Undefined,
    Value,
    align,
    binary,
    common,
    subset,
    truth,
    unary,
    undefined_where,
)

__all__ = ["Expression", "Scope"]

Array = Any


class Scope:
    """The columns of a batch, for the rows an expression is being evaluated on.

    ``fetch`` gives a column's value for every row of the batch, and is asked
    once per column; ``entries`` are the entry numbers of the rows, which is
    what a refusal names.
    """

    __slots__ = ("fetch", "count", "entries", "_seen")

    def __init__(self, fetch: Callable[[str], Value], count: int, entries: Array) -> None:
        self.fetch = fetch
        self.count = count
        self.entries = entries
        self._seen: dict[str, Value] = {}

    def column(self, name: str) -> Value:
        if name not in self._seen:
            self._seen[name] = self.fetch(name)
        return self._seen[name]

    def subset(self, rows: Array) -> Scope:
        """The same columns for only the rows given."""
        return Scope(lambda name: subset(self.column(name), rows), len(rows), self.entries[rows])


def _spread(data: Array, count: int) -> Array:
    return np.broadcast_to(np.asarray(data), (count,)) if np.ndim(data) == 0 else data


def _merged_numbers(a: Value, b: Value, where: tuple[Array, Array], count: int) -> Value:
    yes, no = where
    text = [value.dtype.kind in "US" for value in (a, b)]
    kind = np.result_type(a.dtype, b.dtype) if all(text) else common(a.dtype, b.dtype, "? :")
    out = np.empty(count, kind)
    out[yes] = _spread(a.data, len(yes))
    out[no] = _spread(b.data, len(no))
    return Value(out)


def _merged_rows(a: Value, b: Value, where: tuple[Array, Array]) -> Value:
    kind = common(a.dtype, b.dtype, "? :")
    joined = Jagged.join(
        [Jagged(a.data.astype(kind), a.offsets), Jagged(b.data.astype(kind), b.offsets)]
    )
    order = np.argsort(np.concatenate(where), kind="stable")
    return Value(*kernels.compact(joined.take(order)))


class Evaluator:
    """One walk of an expression's tree over one scope."""

    def __init__(self, scope: Scope, text: str) -> None:
        self.scope = scope
        self.text = text

    def top(self, node: Node) -> Value:
        """The value of the whole expression, with what went wrong told by entry."""
        try:
            return self.value(node)
        except Mismatch as why:
            entry = int(self.scope.entries[why.row])
            raise ValueError(
                f"{self.text!r}: {why.what} was given collections of different sizes in "
                f"entry {entry}, {why.sizes[0]} and {why.sizes[1]} elements; RVec "
                f"arithmetic pairs the elements of two collections one for one"
            ) from None
        except Undefined as why:
            entry = int(self.scope.entries[why.row])
            raise ValueError(
                f"{self.text!r} computes {why.what} in entry {entry}, which C++ leaves "
                f"undefined; guard it, as with 'v.size() > 0 && v[0] > 30'"
            ) from None

    def value(self, node: Node) -> Value:
        return EVALUATE[type(node)](self, node)

    def on(self, node: Node, rows: Array) -> Value:
        """The value of ``node`` for only the rows given, as a branch of C++ evaluates it."""
        if len(rows) == self.scope.count:
            return self.value(node)
        return Evaluator(self.scope.subset(rows), self.text).top(node)

    # -- the kinds of node ---------------------------------------------------------

    def literal(self, node: Node) -> Value:
        assert isinstance(node, Literal)
        return Value(np.asarray(node.value))

    def column(self, node: Node) -> Value:
        assert isinstance(node, Column)
        return self.scope.column(node.name)

    def unary(self, node: Node) -> Value:
        assert isinstance(node, Unary)
        return unary(node.op, self.value(node.operand))

    def binary(self, node: Node) -> Value:
        assert isinstance(node, Binary)
        left = self.value(node.left)
        if node.op in ("&&", "||") and left.offsets is None and left.items is None:
            return self._short_circuit(node, left)
        return binary(node.op, left, self.value(node.right))

    def _short_circuit(self, node: Binary, left: Value) -> Value:
        """``&&`` and ``||`` of numbers per entry: the right side only where it decides."""
        decided = node.op == "||"
        test = _spread(truth(left.data, node.op), self.scope.count)
        rows = np.flatnonzero(test != decided)
        right = self.on(node.right, rows)
        if right.offsets is not None:  # a number and an RVec: an RVec's operator, both sides
            return binary(node.op, left, self.value(node.right))
        out = np.full(self.scope.count, decided, np.bool_)
        out[rows] = _spread(truth(right.data, node.op), len(rows))
        return Value(out)

    def ternary(self, node: Node) -> Value:
        assert isinstance(node, Ternary)
        condition = self.value(node.condition)
        if condition.offsets is not None:
            raise UnsupportedFeatureError(
                f"{self.text!r} tests a collection with ? :, which C++ does not do; "
                f"Where(condition, a, b) chooses element by element"
            )
        test = _spread(truth(condition.data, "? :"), self.scope.count)
        where = np.flatnonzero(test), np.flatnonzero(~test)
        a, b = self.on(node.then, where[0]), self.on(node.otherwise, where[1])
        if (a.offsets is None) != (b.offsets is None) or a.items or b.items:
            raise UnsupportedFeatureError(
                f"{self.text!r} gives a collection on one side of ? : and a number on the "
                f"other, which C++ refuses: both sides have to be the same kind of thing"
            )
        if a.offsets is None:
            return _merged_numbers(a, b, where, self.scope.count)
        return _merged_rows(a, b, where)

    def cast(self, node: Node) -> Value:
        assert isinstance(node, Cast)
        return functions.cast(self.value(node.operand), node.dtype)

    def call(self, node: Node) -> Value:
        assert isinstance(node, Call)
        args = [self.value(arg) for arg in node.args]
        return functions.call(node.name, args, self.text)

    def index(self, node: Node) -> Value:
        assert isinstance(node, Index)
        target = self.value(node.target)
        if target.items is not None:
            return self._member(target, node.index)
        index = self.value(node.index)
        if target.offsets is None:
            raise UnsupportedFeatureError(
                f"{self.text!r} indexes a number per entry, which is not a collection"
            )
        if index.offsets is not None:
            return self._masked(target, index)
        return self._element(target, index.data, "an index past the end of a collection")

    def _member(self, target: Value, index: Node) -> Value:
        assert target.items is not None
        if not isinstance(index, Literal) or np.asarray(index.value).dtype.kind not in "iu":
            raise UnsupportedFeatureError(
                f"{self.text!r} takes a member of what Combinations gives by a number that "
                f"is not written out, such as [0] or [1]"
            )
        position = int(index.value)
        if not 0 <= position < len(target.items):
            raise IndexError(
                f"{self.text!r} asks for member {position} of {len(target.items)} collections"
            )
        return target.items[position]

    def _masked(self, target: Value, mask: Value) -> Value:
        assert target.offsets is not None
        if not np.array_equal(target.offsets, mask.offsets):
            align((target, mask), "a mask")  # raises, naming the entry
        return Value(*kernels.masked(np.asarray(target.data), target.offsets, mask.data))

    def _element(self, target: Value, index: Array, what: str) -> Value:
        """One element of each entry's collection, at a position per entry."""
        assert target.offsets is not None
        data = np.asarray(index)
        if data.dtype.kind not in "biu":
            raise UnsupportedFeatureError(
                f"{self.text!r} indexes a collection with {data.dtype}; an index is an integer"
            )
        where = np.arange(self.scope.count + 1, dtype=np.int64)
        local = _spread(data, self.scope.count)
        picked, _, bad = kernels.taken(np.asarray(target.data), target.offsets, local, where)
        undefined_where(bad, what)
        return Value(picked)

    def method(self, node: Node) -> Value:
        assert isinstance(node, Method)
        target = self.value(node.target)
        name, args = node.name, [self.value(arg) for arg in node.args]
        if target.dtype.kind in "US" and target.offsets is None:
            return self._text_method(target, name, args)
        if target.offsets is None or target.items is not None:
            raise UnsupportedFeatureError(
                f"{self.text!r} calls {name}() on a number per entry, which has no methods"
            )
        return self._rvec_method(target, name, args)

    def _arity(self, name: str, args: list[Value], count: int) -> None:
        if len(args) != count:
            raise UnsupportedFeatureError(
                f"{self.text!r} calls {name}() with {len(args)} arguments rather than {count}"
            )

    def _rvec_method(self, target: Value, name: str, args: list[Value]) -> Value:
        assert target.offsets is not None
        lengths = np.diff(target.offsets)
        if name in ("size", "empty"):
            self._arity(name, args, 0)
            return Value(functions.as_size(lengths) if name == "size" else lengths == 0)
        if name in ("front", "back"):
            self._arity(name, args, 0)
            where = np.zeros(len(lengths), np.int64) if name == "front" else lengths - 1
            return self._element(target, where, f"{name}() of an empty collection")
        if name == "at":
            self._arity(name, args, 1)
            return self._element(target, args[0].data, "at() past the end of a collection")
        raise UnsupportedFeatureError(
            f"{self.text!r} calls {name}(), and the methods of an RVec an expression here "
            f"may call are size(), empty(), front(), back() and at(i)"
        )

    def _text_method(self, target: Value, name: str, args: list[Value]) -> Value:
        if name not in ("size", "length", "empty"):
            raise UnsupportedFeatureError(
                f"{self.text!r} calls {name}() on a string; the ones a string here has are "
                f"size(), length() and empty()"
            )
        self._arity(name, args, 0)
        lengths = functions.as_size(np.char.str_len(np.asarray(target.data)))
        return Value(lengths == 0 if name == "empty" else lengths)


#: How each kind of node is evaluated.
EVALUATE: dict[type, Callable[[Evaluator, Node], Value]] = {
    Literal: Evaluator.literal,
    Column: Evaluator.column,
    Unary: Evaluator.unary,
    Binary: Evaluator.binary,
    Ternary: Evaluator.ternary,
    Cast: Evaluator.cast,
    Call: Evaluator.call,
    Index: Evaluator.index,
    Method: Evaluator.method,
}


class Expression:
    """One compiled expression: its text, the columns it reads, and its value for a batch."""

    __slots__ = ("text", "root", "columns")

    def __init__(self, text: str, names: Collection[str]) -> None:
        self.text = text
        self.root = parse(text, names, functions.TABLE)
        #: The columns evaluating it reads, by the names they have in the frame.
        self.columns = columns_of(self.root)

    def __repr__(self) -> str:
        return f"<Expression {self.text!r}>"

    def evaluate(self, scope: Scope) -> Value:
        """The expression's value for every row of ``scope``."""
        return Evaluator(scope, self.text).top(self.root)
