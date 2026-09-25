"""An expression's value for every entry of a batch at once, and where it has none.

Every node evaluates to a :class:`Value`: an array, a mask of where that
array really has a value, and whether it is one value per entry or one per
item of the loop. Values per entry meet values per item by being repeated
to them - ``data[space.entry]`` - so a number per entry goes with every
element of the loop, as ROOT has it.

A value can be missing because an index ran past the end of its collection:
``pt[3]`` of an entry with two. ROOT leaves such an entry, or element, out of
what it draws, and so does this - every operator's result is missing where
either operand is - with the one exception ROOT makes for exactly this:
``Alt$(primary, alternate)`` gives the alternate where the primary is
missing. A ternary takes the missing-ness of the branch it picks.

The special functions that reduce a collection - ``Sum$``, ``Length$``,
``Min$``, ``Max$``, ``MinIf$`` and ``MaxIf$`` - run a loop of their own over
their argument and give one value per entry, so ``pt > Sum$(pt)/Length$(pt)``
compares each element with its entry's mean.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from .binding import Specs, column, explicit, loops, varies
from .columns import Layout
from .functions import CASTS, FUNCTIONS, STRING_FUNCTIONS
from .loops import Cursor, Scope, Space, both
from .nodes import (
    Alt,
    Binary,
    Call,
    Cast,
    Node,
    Number,
    Reduce,
    Ref,
    Size,
    Special,
    Ternary,
    Text,
    Unary,
    children,
)
from .ops import as_index, binary, cast, is_text, real, truth, unary

__all__ = ["Value", "Numbers", "Evaluator", "lift"]

Array = Any


class Value:
    """What one node evaluates to: values, where they are real, and what they line up with."""

    __slots__ = ("data", "valid", "items")

    def __init__(self, data: Array, valid: Array | None = None, items: bool = False) -> None:
        self.data = data
        #: ``None`` when every value is real.
        self.valid = valid
        #: One per item of the loop, rather than one per entry (or one for all).
        self.items = items


class Numbers:
    """The entry numbers ``Entry$``, ``LocalEntry$`` and ``Entries$`` are made of."""

    __slots__ = ("entry", "local", "total")

    def __init__(self, entry: Array, local: Array, total: int) -> None:
        self.entry = entry
        self.local = local
        self.total = total


class Member:
    """A branch in one loop, and how it takes part in it."""

    __slots__ = ("node", "weak", "in_index", "late")

    def __init__(self, node: Node, weak: bool, in_index: bool, late: bool) -> None:
        self.node = node
        #: Inside ``Alt$``'s first argument, so it does not cut the loop short.
        self.weak = weak
        #: Inside the index of another branch.
        self.in_index = in_index
        #: Indexed by something that differs element to element, so read at the last level.
        self.late = late


def lift(value: Value, space: Space) -> Value:
    """A value per entry repeated to every item of the loop; one per item as it is."""
    if value.items or not space.looped:
        return value
    data = value.data if not np.ndim(value.data) else value.data[space.entry]
    valid = None if value.valid is None else value.valid[space.entry]
    return Value(data, valid, True)


#: A reducer: the elements, which of them count (``None`` for all), and each entry's run.
Reducer = Callable[[Array, Array, Array], Array]


def _counted(keep: Array | None, offsets: Array) -> Array:
    """How many elements of each entry count."""
    if keep is None:
        return np.diff(offsets)
    running = np.zeros(len(keep) + 1, np.int64)
    np.cumsum(keep, out=running[1:])
    return running[offsets[1:]] - running[offsets[:-1]]


def _runs(ufunc: Any, numbers: Array, offsets: Array) -> Array:
    """``ufunc`` over each entry's run of elements, and zero for an entry with none."""
    out = np.zeros(len(offsets) - 1, np.float64)
    if len(numbers):
        filled = np.diff(offsets) > 0
        out[filled] = ufunc.reduceat(numbers, offsets[:-1][filled])
    return out


def _sum(data: Array, keep: Array | None, offsets: Array) -> Array:
    numbers = real(data, "Sum$")
    return _runs(np.add, numbers if keep is None else np.where(keep, numbers, 0.0), offsets)


def _length(data: Array, keep: Array | None, offsets: Array) -> Array:
    return _counted(keep, offsets).astype(np.int64)


def _extreme(ufunc: Any, name: str, empty: float) -> Reducer:
    """``Min$`` or ``Max$`` of each entry's elements, and zero for an entry with none."""

    def apply(data: Array, keep: Array | None, offsets: Array) -> Array:
        numbers = real(data, name)
        if keep is not None:
            numbers = np.where(keep, numbers, empty)
        out = _runs(ufunc, numbers, offsets)
        out[_counted(keep, offsets) == 0] = 0.0
        return out

    return apply


#: How each reducing special function makes one value per entry of its elements.
REDUCERS: dict[str, Reducer] = {
    "Sum$": _sum,
    "Length$": _length,
    "Min$": _extreme(np.minimum, "Min$", np.inf),
    "Max$": _extreme(np.maximum, "Max$", -np.inf),
    "MinIf$": _extreme(np.minimum, "MinIf$", np.inf),
    "MaxIf$": _extreme(np.maximum, "MaxIf$", -np.inf),
}


class Evaluator:
    """Evaluates one bound expression over one batch of columns."""

    __slots__ = ("layouts", "specs", "entries", "numbers")

    def __init__(
        self, layouts: Mapping[str, Layout], specs: Specs, entries: int, numbers: Numbers
    ) -> None:
        self.layouts = layouts
        self.specs = specs
        self.entries = entries
        self.numbers = numbers

    def top(self, root: Node) -> tuple[Space, Value]:
        """The loop the whole expression runs, and its value at the last level of it."""
        scope = self.scope((root,))
        return scope.space, lift(self.value(root, scope), scope.space)

    # -- loops ----------------------------------------------------------------

    def scope(self, roots: tuple[Node, ...]) -> Scope:
        """Run the loop the branches under ``roots`` make, as far down as it goes."""
        members: list[Member] = []
        for root in roots:
            self._collect(root, False, False, members)
        self._refuse_crossed(members)
        scope = Scope(self.entries)
        cursors = [
            Cursor(m.node, self.layouts[column(m.node)], self.specs[id(m.node)], m.weak)
            for m in members
            if not m.late
        ]
        scope.run(cursors, self._index)
        return scope

    def _late(self, node: Node) -> bool:
        return any(varies(index, self.specs) for index in explicit(node) if index is not None)

    def _collect(self, node: Node, weak: bool, in_index: bool, out: list[Member]) -> None:
        if isinstance(node, Reduce):
            return  # it loops on its own
        if isinstance(node, Alt):
            self._collect(node.primary, True, in_index, out)
            self._collect(node.alternate, weak, in_index, out)
            return
        if not isinstance(node, (Ref, Size)):
            for child in children(node):
                self._collect(child, weak, in_index, out)
            return
        late = self._late(node)
        out.append(Member(node, weak, in_index, late))
        for index in explicit(node) if late else ():
            assert index is not None  # a branch indexed element by element never also loops
            self._collect(index, weak, True, out)

    def _refuse_crossed(self, members: list[Member]) -> None:
        """Refuse a loop inside an index beside a loop outside one, which ROOT nests."""
        indexed = [m for m in members if m.late and self._indexed_by_loop(m.node)]
        free = [m for m in members if self._loops_freely(m)]
        if indexed and free:
            raise UnsupportedFeatureError(
                f"{column(indexed[0].node)!r} is indexed by a collection while "
                f"{column(free[0].node)!r} is looped over outside any index; ROOT runs those "
                f"as two loops, one inside the other, which this does not do - reduce one of "
                f"them with Sum$, Max$ or an explicit index"
            )

    def _indexed_by_loop(self, node: Node) -> bool:
        return any(loops(index, self.specs) for index in explicit(node) if index is not None)

    def _loops_freely(self, member: Member) -> bool:
        """Does a branch loop on its own account, rather than inside another's index?"""
        if member.late or member.in_index:
            return False
        return None in self.specs[id(member.node)]

    def _index(self, spec: Node) -> tuple[Array, Array | None]:
        """An index expression's value per entry, in a loop of its own that never loops."""
        scope = self.scope((spec,))
        value = self.value(spec, scope)
        index, ok = as_index(value.data)
        return index, both(ok, value.valid)

    # -- values ---------------------------------------------------------------

    def value(self, node: Node, scope: Scope) -> Value:
        return VALUES[type(node)](self, node, scope)

    def _aligned(self, scope: Scope, *nodes: Node) -> list[Value]:
        values = [self.value(node, scope) for node in nodes]
        if any(value.items for value in values):
            values = [lift(value, scope.space) for value in values]
        return values

    def _number(self, node: Node, scope: Scope) -> Value:
        assert isinstance(node, (Number, Text))
        return Value(np.asarray(node.value))

    def _branch(self, node: Node, scope: Scope) -> Value:
        if self._late(node):
            return self._late_branch(node, scope)
        data, valid, items = scope.value(node)
        return Value(data, valid, items)

    def _late_branch(self, node: Node, scope: Scope) -> Value:
        """A branch indexed element by element, read at the last level of the loop."""
        space = scope.space
        cursor = Cursor(node, self.layouts[column(node)], self.specs[id(node)], False)
        cursor.move(space.entry)
        for spec in cursor.specs:
            assert spec is not None
            written = lift(self.value(spec, scope), space)
            index, ok = as_index(written.data)
            cursor.step(index, both(ok, written.valid))
        data, valid = cursor.result()
        return Value(data, valid, space.looped)

    def _special(self, node: Node, scope: Scope) -> Value:
        assert isinstance(node, Special)
        return SPECIALS[node.name](self, scope.space)

    def _unary(self, node: Node, scope: Scope) -> Value:
        assert isinstance(node, Unary)
        operand = self.value(node.operand, scope)
        return Value(unary(node.op, operand.data), operand.valid, operand.items)

    def _binary(self, node: Node, scope: Scope) -> Value:
        assert isinstance(node, Binary)
        left, right = self._aligned(scope, node.left, node.right)
        data = binary(node.op, left.data, right.data)
        return Value(data, both(left.valid, right.valid), left.items or right.items)

    def _ternary(self, node: Node, scope: Scope) -> Value:
        assert isinstance(node, Ternary)
        condition, then, otherwise = self._aligned(scope, node.condition, node.then, node.otherwise)
        if is_text(then.data) != is_text(otherwise.data):
            raise UnsupportedFeatureError(
                "a ternary gives a string on one side and a number on the other, and one "
                "column cannot hold both"
            )
        chosen = truth(condition.data, "a ternary's condition")
        data = np.where(chosen, then.data, otherwise.data)
        valid = None
        if then.valid is not None or otherwise.valid is not None:
            valid = np.where(
                chosen,
                True if then.valid is None else then.valid,
                True if otherwise.valid is None else otherwise.valid,
            )
        items = condition.items or then.items or otherwise.items
        return Value(data, both(condition.valid, valid), items)

    def _cast(self, node: Node, scope: Scope) -> Value:
        assert isinstance(node, Cast)
        operand = self.value(node.operand, scope)
        data, ok = cast(operand.data, CASTS[node.type], f"the cast to {node.type}")
        return Value(data, both(operand.valid, ok), operand.items)

    def _call(self, node: Node, scope: Scope) -> Value:
        assert isinstance(node, Call)
        args = self._aligned(scope, *node.args)
        valid = None
        for arg in args:
            valid = both(valid, arg.valid)
        if node.name in STRING_FUNCTIONS:
            data = STRING_FUNCTIONS[node.name].apply(*(self._text(node, arg) for arg in args))
        else:
            data = FUNCTIONS[node.name].apply(*(real(arg.data, node.name) for arg in args))
        return Value(np.asarray(data), valid, any(arg.items for arg in args))

    def _text(self, node: Call, arg: Value) -> Array:
        if not is_text(arg.data):
            raise UnsupportedFeatureError(
                f"{node.name} compares strings, and was given numbers; its arguments are "
                f"a string column and a string in quotes"
            )
        return arg.data

    def _reduce(self, node: Node, scope: Scope) -> Value:
        assert isinstance(node, Reduce)
        inner = self.scope(node.args)
        values = [lift(self.value(arg, inner), inner.space) for arg in node.args]
        shape = (len(inner.space),)
        data = np.broadcast_to(values[0].data, shape)
        keep = None if values[0].valid is None else np.broadcast_to(values[0].valid, shape)
        if len(values) == 2:
            condition = np.broadcast_to(truth(values[1].data, node.kind), shape)
            keep = both(keep, both(values[1].valid, condition))
        return Value(REDUCERS[node.kind](data, keep, inner.space.offsets()))

    def _alt(self, node: Node, scope: Scope) -> Value:
        assert isinstance(node, Alt)
        primary, alternate = self._aligned(scope, node.primary, node.alternate)
        if primary.valid is None:
            return primary
        data = np.where(primary.valid, primary.data, alternate.data)
        valid = None if alternate.valid is None else primary.valid | alternate.valid
        return Value(data, valid, primary.items or alternate.items)


def _entry(evaluator: Evaluator, space: Space) -> Value:
    return Value(evaluator.numbers.entry)


def _local_entry(evaluator: Evaluator, space: Space) -> Value:
    return Value(evaluator.numbers.local)


def _entries(evaluator: Evaluator, space: Space) -> Value:
    return Value(np.asarray(evaluator.numbers.total, dtype=np.int64))


def _iteration(evaluator: Evaluator, space: Space) -> Value:
    """Which element of its entry each item is: ``0`` to ``Length$ - 1``."""
    if not space.looped:
        return Value(np.asarray(0, dtype=np.int64))
    starts = space.offsets()[:-1][space.entry]
    return Value(np.arange(len(space), dtype=np.int64) - starts, None, True)


def _count(evaluator: Evaluator, space: Space) -> Value:
    """Bare ``Length$``: how many values the expression has for each entry."""
    if not space.looped:
        return Value(np.asarray(1, dtype=np.int64))
    return Value(np.diff(space.offsets()))


#: What each of ROOT's special names is, for one space of the loop.
SPECIALS: dict[str, Callable[[Evaluator, Space], Value]] = {
    "Entry$": _entry,
    "LocalEntry$": _local_entry,
    "Entries$": _entries,
    "Iteration$": _iteration,
    "Length$": _count,
}

#: How each kind of node is evaluated.
VALUES: dict[type, Callable[[Evaluator, Any, Scope], Value]] = {
    Number: Evaluator._number,
    Text: Evaluator._number,
    Ref: Evaluator._branch,
    Size: Evaluator._branch,
    Special: Evaluator._special,
    Unary: Evaluator._unary,
    Binary: Evaluator._binary,
    Ternary: Evaluator._ternary,
    Cast: Evaluator._cast,
    Call: Evaluator._call,
    Reduce: Evaluator._reduce,
    Alt: Evaluator._alt,
}
