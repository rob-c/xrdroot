"""The facts about loops and switches the statement emitter decides its Python by.

Can ``for (int i = a; i < b; ++i)`` be ``for i in range(a, b)``? Only if the
body never changes ``i`` or anything ``b`` is computed from. Does a
``do``-``while`` body ``continue``? Then the condition must still be tested.
Does a ``switch`` fall through from one case into the next? Then it is not
an ``if``/``elif`` chain. These questions are answered here, over the tree,
before any Python is written.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from .nodes import (
    Assign,
    Binary,
    Block,
    Break,
    Call,
    Case,
    Continue,
    DeclStmt,
    DoWhile,
    Expr,
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
    "without_break",
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
    if isinstance(last, Block):
        return ends_in_jump(last.body)
    return isinstance(last, ExprStmt) and isinstance(last.expr, Throw)


def without_break(stmts: list[Stmt]) -> list[Stmt]:
    """A case's statements without the ``break`` ending them, even one inside a closing block."""
    if not stmts:
        return stmts
    last = stmts[-1]
    if isinstance(last, Break):
        return stmts[:-1]
    if isinstance(last, Block):
        return [*stmts[:-1], Block(last.where, without_break(last.body))]
    return stmts


def is_literal_step(node: Node, name: str) -> int | None:
    """The step of ``++i``, ``i--``, ``i += 2``: how much each time round changes ``i``."""
    if isinstance(node, Unary) and node.op in ("++", "--") and _is(node.operand, name):
        return 1 if node.op == "++" else -1
    if isinstance(node, Assign) and node.op in ("+=", "-=") and _is(node.target, name):
        return _literal_step(node)
    return None


def _literal_step(node: Assign) -> int | None:
    value = node.value
    if not isinstance(value, Literal) or value.kind != "int" or value.value <= 0:
        return None
    step = int(value.value)
    return step if node.op == "+=" else -step


def _is(node: Node, name: str) -> bool:
    return isinstance(node, Name) and node.parts == [name]


def bound_of(cond: Node, name: str) -> tuple[str, Expr] | None:
    """``i < n``: the comparison and the bound, when ``cond`` compares ``i`` with something."""
    if (
        isinstance(cond, Binary)
        and cond.op in ("<", "<=", ">", ">=", "!=")
        and _is(cond.left, name)
    ):
        return cond.op, cond.right
    return None


# -- a loop filling a histogram from a random draw ---------------------------------------

#: ``TRandom``'s draws that take ``n`` for an array of them, the stream's next ``n`` in order.
DRAWS = frozenset(
    "Rndm Uniform Gaus Exp Landau Integer Poisson PoissonD BreitWigner Binomial".split()
)
#: The generators, and the global one, whose draws those are.
RANDOMS = frozenset(
    """TRandom TRandom1 TRandom2 TRandom3 TRandomMixMax TRandomMixMax17 TRandomMixMax256
    TRandomRanlux48 TRandomRanluxpp""".split()
)
#: The draws that are whole numbers.
INTEGRAL_DRAWS = frozenset(("Integer", "Poisson", "Binomial"))
#: The histograms whose ``Fill`` of arrays is ``Fill`` of each in turn, to the last bit.
HISTOGRAMS = ("TH1", "TH2", "TH3")
#: What arithmetic on an array of draws stays plain Python arithmetic on.
ARITHMETIC = frozenset("+-*/")

Typeof = Callable[[Any], Any]


@dataclass(eq=False)
class VectorDraw:
    """A fill loop's draw: the call, and the two variables ``Rannor`` draws into, if it is
    that - which keep the last turn's pair once the loop is done, as they would."""

    call: Call
    cells: tuple[str, ...]


def vector_fill(body: Stmt, counter: str, typeof: Typeof) -> VectorDraw | None:
    """The one random draw a loop body fills histograms from, when the body is nothing but
    that: floating temporaries declared from the draw and arithmetic on it, and a ``Fill``
    of each of one or more histograms with them, every other value the same on every
    turn - or ``Rannor(a, b)`` first, and the temporaries and the fills from ``a`` and
    ``b``. ``n`` draws at once and one ``Fill`` of the arrays per histogram are then the
    loop to the last bit, and the loop is written that way; any other body is ``None``,
    and written as the loop.
    """
    stmts = body.body if isinstance(body, Block) else [body]
    plan = _Plan(counter, typeof)
    pair = plan.rannor(stmts[0]) if stmts else None
    if not plan.scanned(stmts[1:] if pair is not None else stmts):
        return None
    if pair is not None:
        return VectorDraw(pair, _cells_of(pair)) if not plan.draws else None
    return VectorDraw(plan.draws[0], ()) if len(plan.draws) == 1 else None


def _fill_of(stmt: Stmt, typeof: Typeof) -> Call | None:
    """The ``Fill`` a statement is, if it is one."""
    if not isinstance(stmt, ExprStmt) or not isinstance(stmt.expr, Call):
        return None
    return stmt.expr if _is_fill(stmt.expr, typeof) else None


def _cells_of(call: Call) -> tuple[str, ...]:
    """The names ``Rannor(a, b)`` draws into."""
    return tuple(arg.last for arg in call.args if isinstance(arg, Name))


def _is_fill(expr: Call, typeof: Typeof) -> bool:
    """``h->Fill(...)`` of a histogram the macro declared as one."""
    if not isinstance(expr.func, Member):
        return False
    if expr.func.name != "Fill" or not isinstance(expr.func.obj, Name) or not expr.args:
        return False
    owner = typeof(expr.func.obj)
    return owner is not None and not owner.dims and owner.name.startswith(HISTOGRAMS)


def _temporary(stmt: Stmt) -> Any:
    """``double x = ...;`` - one floating value, declared with what it starts as."""
    if not isinstance(stmt, DeclStmt) or len(stmt.decls) != 1:
        return None
    decl = stmt.decls[0]
    ctype = decl.ctype
    if decl.style != "=" or decl.init is None or decl.static or decl.binding is not None:
        return None
    return decl if ctype is not None and ctype.scalar and ctype.floating else None


def _promoted(left: str | None, right: str | None) -> str | None:
    """What C computes an operation on the two kinds in: floating if either is."""
    if "float" in (left, right):
        return "float"
    return None if None in (left, right) else "int"


class _Plan:
    """What a fill loop's expressions may hold, and the draw and the temporaries met."""

    def __init__(self, counter: str, typeof: Typeof) -> None:
        self.counter, self.typeof = counter, typeof
        self.draws: list[Call] = []
        #: The temporaries declared so far - floating, every one - and those holding arrays.
        self.temps: set[str] = set()
        self.vectors: set[str] = set()

    def temporary(self, stmt: Stmt) -> bool:
        """Is ``stmt`` a floating temporary declared from plain values? Then it is kept, as
        holding an array when the draw is among them."""
        decl = _temporary(stmt)
        if decl is None or not self.plain(decl.init):
            return False
        self.temps.add(decl.name)
        if self.vectored(decl.init):
            self.vectors.add(decl.name)
        return True

    def rannor(self, stmt: Stmt) -> Call | None:
        """``r.Rannor(a, b);`` into two floating variables of the macro's, the draw a body
        may begin with: ``a`` and ``b`` hold arrays from then on."""
        if not isinstance(stmt, ExprStmt) or not isinstance(stmt.expr, Call):
            return None
        call = stmt.expr
        func = call.func
        if not isinstance(func, Member) or func.name != "Rannor" or not self._generator(func):
            return None
        names = _cells_of(call)
        if not self._pair(names, call.args):
            return None
        self.temps.update(names)
        self.vectors.update(names)
        return call

    def _pair(self, names: tuple[str, ...], args: list[Expr]) -> bool:
        """Two different floating variables of the macro's, neither the counter."""
        if len(names) != 2 or len(args) != 2 or len(set(names)) != 2 or self.counter in names:
            return False
        return all(self.kind(arg) == "float" for arg in args)

    def scanned(self, stmts: list[Stmt]) -> bool:
        """Are ``stmts`` fills and temporaries, each fill of plain values with the draw among
        them and of a histogram of its own, and at least one of them a fill?"""
        fills: list[Call] = []
        for stmt in stmts:
            fill = _fill_of(stmt, self.typeof)
            if fill is not None:
                fills.append(fill)
            elif not self.temporary(stmt):
                return False
        return bool(fills) and self.distinct(fills) and all(map(self.fills, fills))

    @staticmethod
    def distinct(fills: list[Call]) -> bool:
        """Does each ``Fill`` fill a histogram of its own? Twice into one, the entries would
        be summed in another order than the loop's."""
        receivers = [fill.func.obj.last for fill in fills if isinstance(fill.func, Member)
                     and isinstance(fill.func.obj, Name)]  # fmt: skip
        return len(set(receivers)) == len(fills)

    def fills(self, fill: Call) -> bool:
        """Is the ``Fill`` of plain values, one of them from the draw?"""
        if not all(self.plain(arg) for arg in fill.args):
            return False
        return any(self.vectored(arg) for arg in fill.args)

    def plain(self, expr: Expr) -> bool:
        """Numbers, names but the counter's, arithmetic, and the one draw of a generator."""
        if isinstance(expr, Literal):
            return expr.kind in ("int", "float", "char")
        if isinstance(expr, Name):
            found = self.typeof(expr)
            return expr.last != self.counter and (found is None or found.scalar)
        if isinstance(expr, Unary):
            return expr.op in ("-", "+") and self.plain(expr.operand)
        if isinstance(expr, Binary):
            return self._arithmetic(expr)
        return isinstance(expr, Call) and self._draw(expr)

    def _arithmetic(self, expr: Binary) -> bool:
        """``+ - * /`` of plain values; with an array among them, floating in C - so that
        NumPy's arithmetic is C's - and a division the emitter can see is floating, so that
        it writes ``/`` rather than the helper that tells integers apart at run time."""
        if expr.op not in ARITHMETIC or not self.plain(expr.left) or not self.plain(expr.right):
            return False
        if not self.vectored(expr):
            return True
        if self.kind(expr) != "float":
            return False
        return expr.op != "/" or self.seen_floating(expr.left) or self.seen_floating(expr.right)

    def kind(self, expr: Expr) -> str | None:
        """``float`` or ``int``, as C computes ``expr``, a temporary's being its declaration's
        and a draw's its distribution's; ``None`` for a name whose type is not known."""
        if isinstance(expr, Literal):
            return "float" if expr.kind == "float" else "int"
        if isinstance(expr, Name):
            return self._kind_of_name(expr)
        if isinstance(expr, Unary):
            return self.kind(expr.operand)
        if isinstance(expr, Binary):
            return _promoted(self.kind(expr.left), self.kind(expr.right))
        assert isinstance(expr, Call) and isinstance(expr.func, Member)
        return "int" if expr.func.name in INTEGRAL_DRAWS else "float"

    def _kind_of_name(self, expr: Name) -> str | None:
        if expr.last in self.temps:
            return "float"
        found = self.typeof(expr)
        return None if found is None else ("float" if found.floating else "int")

    def seen_floating(self, expr: Expr) -> bool:
        """Will the emitter, writing ``expr`` after the temporaries are declared, know it is
        floating? A draw it does not know the type of; the rest it does."""
        if isinstance(expr, Literal):
            return expr.kind == "float"
        if isinstance(expr, Name):
            return expr.last in self.temps or self.kind(expr) == "float"
        if isinstance(expr, Unary):
            return self.seen_floating(expr.operand)
        if isinstance(expr, Binary):
            return self.seen_floating(expr.left) or self.seen_floating(expr.right)
        return False

    def _draw(self, expr: Call) -> bool:
        """``r.Gaus(...)``, ``gRandom->Rndm()``: a draw from a generator, of fixed arguments."""
        func = expr.func
        if not isinstance(func, Member) or func.name not in DRAWS or not self._generator(func):
            return False
        if not all(self.plain(arg) and not self.vectored(arg) for arg in expr.args):
            return False
        self.draws.append(expr)
        return True

    def _generator(self, func: Member) -> bool:
        """Is the method's object a ``TRandom`` the macro declared, or ``gRandom``?"""
        if not isinstance(func.obj, Name):
            return False
        owner = self.typeof(func.obj)
        if owner is None:
            return func.obj.last == "gRandom"
        return owner.name in RANDOMS and not owner.dims

    def vectored(self, expr: Expr) -> bool:
        """Does ``expr`` take the draw's value - an array, once the loop is written as one?"""
        for node in (expr, *_own(expr, ())):
            if isinstance(node, Call) or (isinstance(node, Name) and node.last in self.vectors):
                return True
        return False
