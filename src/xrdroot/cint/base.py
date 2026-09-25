"""The state every part of the emitter shares: the program, the output, the scopes.

The emitter is one class built in layers - type inference, expressions,
calls, statements, declarations - each a module of its own; this is the
bottom layer, with the scope chain, the naming rules and the way to refuse.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .ctype import CType
from .errors import Refusal, Where
from .nodes import Node
from .program import Program
from .runtime import __all__ as RUNTIME_NAMES
from .symbols import Scope, Symbol, python_name
from .writer import Writer

__all__ = ["EmitterBase", "Out", "P"]

#: A piece of Python and the precedence of its outermost operator.
Out = tuple[str, int]


class P:
    """Python's operator precedences, loosest first: what decides the brackets written."""

    WALRUS = 0
    TERNARY = 1
    OR = 2
    AND = 3
    NOT = 4
    COMPARE = 5
    BITOR = 6
    XOR = 7
    BITAND = 8
    SHIFT = 9
    ADD = 10
    MUL = 11
    UNARY = 12
    POWER = 13
    POSTFIX = 15
    ATOM = 16


#: Names a macro's own identifiers must not take, since the translation's Python uses them.
TAKEN = frozenset(RUNTIME_NAMES)


class EmitterBase:
    """The emitter's shared state and its smallest helpers."""

    def __init__(self, program: Program, file: str) -> None:
        self.program = program
        self.file = file
        self.out = Writer()
        self.scope = Scope("module")
        self._counter = 0

    # -- refusing ---------------------------------------------------------------

    @staticmethod
    def refuse(why: str, node: Node | Where) -> Refusal:
        """A refusal of ``why``, placed at the C++ line of ``node``."""
        where = node if isinstance(node, Where) else node.where
        return Refusal(f"{why} is not something this translator turns into Python", where)

    # -- scopes -------------------------------------------------------------------

    @contextmanager
    def scoped(self, kind: str = "block", **extra: Any) -> Iterator[Scope]:
        outer = self.scope
        self.scope = Scope(kind, outer, function=extra.get("function", outer.function),
                           klass=extra.get("klass", outer.klass))
        try:
            yield self.scope
        finally:
            self.scope = outer

    def declare(self, name: str, kind: str, ctype: CType | None = None, **extra: Any) -> Symbol:
        """A new symbol for ``name`` here, renamed if it would shadow one of the same function's."""
        py = python_name(name, TAKEN)
        if self.scope.visible(name) or self._clashes(py):
            py = self.fresh(py)
        symbol = Symbol(name, kind, py, ctype, **extra)
        return self.scope.add(symbol)

    def _clashes(self, py: str) -> bool:
        if self.scope.kind not in ("block", "function"):
            return False
        scope: Scope | None = self.scope
        while scope is not None and scope.kind in ("block", "function"):
            if any(s.py == py and s.kind in ("local", "param") for s in scope.symbols.values()):
                return True
            scope = scope.parent if scope.kind != "function" else None
        return False

    def fresh(self, stem: str) -> str:
        """A Python name no C++ name can have: ``stem_1``, ``stem_2``..."""
        self._counter += 1
        return f"{stem.rstrip('_')}_{self._counter}"

    def lookup(self, name: str) -> Symbol | None:
        return self.scope.lookup(name)

    # -- ROOT's names ---------------------------------------------------------------

    @staticmethod
    def root(parts: list[str]) -> str:
        """``TMath::Pi`` as ``ROOT.TMath.Pi``, ``ROOT::Math::X`` as ``ROOT.Math.X``."""
        rest = parts[1:] if parts[0] == "ROOT" and len(parts) > 1 else parts
        return "ROOT." + ".".join(rest)
