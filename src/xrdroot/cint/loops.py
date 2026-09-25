"""The facts about loops and switches the statement emitter decides its Python by.

Can ``for (int i = a; i < b; ++i)`` be ``for i in range(a, b)``? Only if the
body never changes ``i`` or anything ``b`` is computed from. Does a
``do``-``while`` body ``continue``? Then the condition must still be tested.
Does a ``switch`` fall through from one case into the next? Then it is not
an ``if``/``elif`` chain. These questions are answered here, over the tree,
before any Python is written.
"""

from __future__ import annotations

from collections.abc import Iterator

from .nodes import (
    Assign,
    Binary,
    Break,
    Call,
    Case,
    Continue,
    DoWhile,
    ExprStmt,
    For,
    Lambda,
    Literal,
    Member,
    Name,
    Node,
    RangeFor,
    Return,
    Stmt,
    Switch,
    Throw,
    Unary,
    While,
)
from .program import children

__all__ = [
    "jumps",
    "assigned_names",
    "stable",
    "cases",
    "ends_in_jump",
    "READ_ONLY",
    "is_literal_step",
    "bound_of",
]

#: Methods whose call leaves the object as it was, so a loop bound computed with them is fixed.
READ_ONLY = frozenset(
    """size length empty at GetN GetNbinsX GetNbinsY GetNbinsZ GetEntries GetEntriesFast
    GetSize GetLast GetNpar GetNdim GetNrows GetNcols GetNoElements GetXmin GetXmax
    GetBinContent GetNumberOfPoints GetListOfKeys numEntries getSize GetNumberOfBins
    GetNofBins""".split()
)

#: The statements a loop body is not looked into for its own ``break`` and ``continue``.
LOOPS = (For, While, DoWhile, RangeFor)


def _own(node: Node, stop: tuple[type, ...]) -> Iterator[Node]:
    """``node``'s descendants, not looking inside ``stop`` nodes or lambdas."""
    for child in children(node):
        yield child
        if not isinstance(child, (*stop, Lambda)):
            yield from _own(child, stop)


def jumps(body: Node, kind: type, through_switch: bool = False) -> bool:
    """Does ``body`` hold a ``break`` or ``continue`` (``kind``) that jumps out of it?"""
    stop: tuple[type, ...] = LOOPS if through_switch else (*LOOPS, Switch)
    return any(isinstance(node, kind) for node in _own(body, stop))


def assigned_names(body: Node) -> set[str]:
    """Every plain name ``body`` assigns, increments, or takes the address of."""
    names: set[str] = set()
    for node in _own(body, ()):
        target = None
        if isinstance(node, Assign):
            target = node.target
        elif isinstance(node, Unary) and node.op in ("++", "--", "&"):
            target = node.operand
        if isinstance(target, Name):
            names.add(target.last)
    return names


def _changed_receivers(body: Node) -> set[str]:
    """The objects ``body`` calls a method of that might change them."""
    found: set[str] = set()
    for node in _own(body, ()):
        if isinstance(node, Call) and isinstance(node.func, Member):
            receiver = node.func.obj
            if isinstance(receiver, Name) and node.func.name not in READ_ONLY:
                found.add(receiver.last)
    return found


def stable(bound: Node, body: Node) -> bool:
    """Is a loop bound worth the same every time round - so ``range`` may take it once?"""
    changed = assigned_names(body)
    receivers = _changed_receivers(body)
    for node in [bound, *_own(bound, ())]:
        if isinstance(node, Name) and node.last in changed:
            return False
        if isinstance(node, Call) and not _stable_call(node, receivers):
            return False
    return True


def _stable_call(call: Call, receivers: set[str]) -> bool:
    func = call.func
    if not isinstance(func, Member) or func.name not in READ_ONLY:
        return False
    return not (isinstance(func.obj, Name) and func.obj.last in receivers)


def cases(body: list[Stmt]) -> list[tuple[list[Case], list[Stmt]]]:
    """A switch body as its groups: the labels of each, and the statements after them."""
    groups: list[tuple[list[Case], list[Stmt]]] = []
    for stmt in body:
        if isinstance(stmt, Case):
            if groups and not groups[-1][1]:
                groups[-1][0].append(stmt)
            else:
                groups.append(([stmt], []))
        elif groups:
            groups[-1][1].append(stmt)
    return groups


def ends_in_jump(stmts: list[Stmt]) -> bool:
    """Does a case's statement list end by leaving - so it cannot fall into the next case?"""
    if not stmts:
        return False
    last = stmts[-1]
    if isinstance(last, (Break, Continue, Return)):
        return True
    return isinstance(last, ExprStmt) and isinstance(last.expr, Throw)


def is_literal_step(node: Node, name: str) -> int | None:
    """The step of ``++i``, ``i--``, ``i += 2``: how much each time round changes ``i``."""
    if isinstance(node, Unary) and node.op in ("++", "--") and _is(node.operand, name):
        return 1 if node.op == "++" else -1
    if isinstance(node, Assign) and node.op in ("+=", "-=") and _is(node.target, name):
        value = node.value
        if isinstance(value, Literal) and value.kind == "int" and value.value > 0:
            return value.value if node.op == "+=" else -value.value
    return None


def _is(node: Node, name: str) -> bool:
    return isinstance(node, Name) and node.parts == [name]


def bound_of(cond: Node, name: str) -> tuple[str, Node] | None:
    """``i < n``: the comparison and the bound, when ``cond`` compares ``i`` with something."""
    if isinstance(cond, Binary) and cond.op in ("<", "<=", ">", ">=", "!=") and _is(
        cond.left, name
    ):
        return cond.op, cond.right
    return None

