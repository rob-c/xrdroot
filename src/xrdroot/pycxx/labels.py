"""``goto`` and its labels, turned into the loop a jump back to one is.

``again: ... if (bad) goto again;`` is how C++ writes "do this until it comes
out right", and it is a ``while True:`` in Python: the label heads it, the
jump back is its ``continue``, and the end of the block the label leads into
breaks out of it, where falling off that end left the label. Anything else a
``goto`` can do - jump forward to a label, or into or out of a loop or a
``switch`` - has no statement in Python that goes there, and is refused.

The statements a jump became a loop around are the ones written after the
label, so a ``break`` or ``continue`` already there would be taken by that
loop, and a ``goto`` written inside a nested loop would continue the wrong
one: both are refused rather than translated into something else.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from .errors import Refusal, Where
from .nodes import (
    Block,
    Break,
    Case,
    Continue,
    DoWhile,
    For,
    Goto,
    Label,
    Lambda,
    Literal,
    Node,
    RangeFor,
    Stmt,
    Switch,
    While,
)
from .program import children

__all__ = ["lower"]

#: The statements a jump does not cross, and whose own ``break`` and ``continue`` go where
#: that statement's Python does: a jump inside one of these is not a jump to this label's loop.
OWNING = (While, DoWhile, For, RangeFor, Switch, Case, Lambda)


def lower(body: Stmt, refuse: Callable[[str, Node], Refusal]) -> None:
    """Turn every ``goto`` in ``body`` that jumps back to a label into a loop, in place.

    ``refuse`` is the emitter's, so that what is left is refused in the words, and at the
    line, the rest of the translator refuses in - the emitter refusing the ``goto`` itself.
    """
    _blocks(body, refuse)


def _blocks(stmt: Stmt, refuse: Callable[[str, Node], Refusal]) -> None:
    """Every block of statements in ``stmt``, the outer ones first, made into what they are."""
    if isinstance(stmt, Block):
        stmt.body = _headed(stmt.body, refuse)
    for child in children(stmt):
        if isinstance(child, Stmt):
            _blocks(child, refuse)


def _headed(body: list[Stmt], refuse: Callable[[str, Node], Refusal]) -> list[Stmt]:
    """``body`` with the first label a ``goto`` jumps back to heading a ``while True:`` that
    runs to the end of the block, as C++ runs from that label to there."""
    for index, stmt in enumerate(body):
        if not isinstance(stmt, Label):
            continue
        rest = [stmt.body, *body[index + 1 :]]
        jumps, crossed = _jumps_to(rest, stmt.name)
        if not jumps:
            continue
        if crossed:
            raise refuse(
                f"a goto back to {stmt.name}, which becomes a loop a break or a continue "
                "already in it would leave instead",
                stmt,
            )
        for jump in jumps:
            jump.lowered = True
        return [*body[:index], _retry(stmt.where, rest)]
    return body


def _jumps_to(body: list[Stmt], name: str) -> tuple[list[Goto], bool]:
    """The ``goto``\\ s in ``body`` that jump to ``name``, and whether one of them is inside a
    loop, a ``switch`` or a lambda, or a ``break``, ``continue`` or ``case`` stands in the way."""
    jumps: list[Goto] = []
    crossed = False
    for node in body:
        for found in _within(node):
            if isinstance(found, Goto) and found.name == name and not found.lowered:
                jumps.append(found)
            elif isinstance(found, (Break, Continue, Case)):
                crossed = True
    return jumps, crossed


def _within(node: Node) -> Iterator[Node]:
    """``node`` and what is written inside it, up to the statements that own their jumps."""
    yield node
    if isinstance(node, OWNING):
        return
    for child in children(node):
        yield from _within(child)


def _retry(where: Where, body: list[Stmt]) -> While:
    """The ``while True:`` a label and the statements it leads into become: the block ends by
    leaving the loop, which is what reaching its end did before the label was a loop."""
    cond = Literal(where, "bool", True, "bool")
    return While(where, cond, Block(where, [*body, Break(where)]))
