"""The syntax tree of a macro: what the parser builds and the emitter walks.

Every node knows where in the macro it was written (:class:`Where`), which is
what a refusal, and a runtime error mapped back through the source map, names.
Expressions carry a ``ctype`` slot the type pass fills in; statements carry
nothing but their parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ctype import CType
from .errors import Where

__all__ = [
    "Node",
    "Expr",
    "Name",
    "Literal",
    "Unary",
    "Binary",
    "Assign",
    "Ternary",
    "Call",
    "Member",
    "Index",
    "Cast",
    "SizeOf",
    "New",
    "Delete",
    "Lambda",
    "Capture",
    "InitList",
    "Comma",
    "This",
    "Throw",
    "Stmt",
    "Block",
    "ExprStmt",
    "VarDecl",
    "DeclStmt",
    "If",
    "While",
    "DoWhile",
    "For",
    "RangeFor",
    "Switch",
    "Case",
    "Break",
    "Continue",
    "Return",
    "Try",
    "Handler",
    "Empty",
    "Param",
    "Function",
    "ClassDecl",
    "Base",
    "EnumDecl",
    "Namespace",
    "Typedef",
    "Unit",
    "Decl",
]


@dataclass(eq=False)
class Node:
    where: Where


# -- expressions -------------------------------------------------------------


@dataclass(eq=False)
class Expr(Node):
    pass


@dataclass(eq=False)
class Name(Expr):
    """A name, qualified or not: ``x``, ``TMath::Pi``, ``std::vector<int>``."""

    parts: list[str]
    targs: list[Any] | None = None
    rooted: bool = False

    @property
    def text(self) -> str:
        return "::".join(self.parts)

    @property
    def last(self) -> str:
        return self.parts[-1]


@dataclass(eq=False)
class Literal(Expr):
    """A number, string, character, ``true``/``false`` or ``nullptr``."""

    kind: str
    value: Any
    ctype: str


@dataclass(eq=False)
class Unary(Expr):
    op: str
    operand: Expr
    postfix: bool = False


@dataclass(eq=False)
class Binary(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(eq=False)
class Assign(Expr):
    op: str
    target: Expr
    value: Expr


@dataclass(eq=False)
class Ternary(Expr):
    cond: Expr
    yes: Expr
    no: Expr


@dataclass(eq=False)
class Call(Expr):
    func: Expr
    args: list[Expr]


@dataclass(eq=False)
class Member(Expr):
    obj: Expr
    name: str
    arrow: bool
    targs: list[Any] | None = None


@dataclass(eq=False)
class Index(Expr):
    obj: Expr
    index: Expr


@dataclass(eq=False)
class Cast(Expr):
    """``(T)x``, ``static_cast<T>(x)`` and friends, and ``T(x)`` for a built-in ``T``."""

    ctype: CType
    operand: Expr
    kind: str


@dataclass(eq=False)
class SizeOf(Expr):
    ctype: CType | None
    operand: Expr | None


@dataclass(eq=False)
class New(Expr):
    ctype: CType
    args: list[Expr] | None
    count: Expr | None = None
    braces: bool = False


@dataclass(eq=False)
class Delete(Expr):
    operand: Expr
    array: bool


@dataclass(eq=False)
class Capture(Node):
    """One capture of a lambda: a name, or ``&``/``=`` for everything, by reference or not."""

    name: str
    byref: bool


@dataclass(eq=False)
class Lambda(Expr):
    captures: list[Capture]
    params: list[Param]
    body: Block
    returns: CType | None = None


@dataclass(eq=False)
class InitList(Expr):
    """``{a, b}``, or ``T{a, b}`` when a type is named before it."""

    items: list[Expr]
    ctype: CType | None = None


@dataclass(eq=False)
class Comma(Expr):
    items: list[Expr]


@dataclass(eq=False)
class This(Expr):
    pass


@dataclass(eq=False)
class Throw(Expr):
    operand: Expr | None


# -- statements --------------------------------------------------------------


@dataclass(eq=False)
class Stmt(Node):
    pass


@dataclass(eq=False)
class Block(Stmt):
    body: list[Stmt]


@dataclass(eq=False)
class ExprStmt(Stmt):
    expr: Expr


@dataclass(eq=False)
class VarDecl(Stmt):
    """One declared variable: ``int x = 1``, ``TH1F h("h", ...)``, ``double a[3] = {...}``.

    ``style`` says how it was initialised: ``=`` (copy), ``()`` (constructor
    arguments in ``args``), ``{}`` (braces, in ``args``) or ``None`` (default).
    """

    name: str
    ctype: CType
    init: Expr | None = None
    style: str | None = None
    args: list[Expr] = field(default_factory=list)
    static: bool = False
    #: The names of a structured binding, ``auto [a, b] = ...``.
    binding: list[str] | None = None


@dataclass(eq=False)
class DeclStmt(Stmt):
    decls: list[VarDecl]


@dataclass(eq=False)
class If(Stmt):
    cond: Expr | VarDecl
    then: Stmt
    orelse: Stmt | None = None
    init: Stmt | None = None


@dataclass(eq=False)
class While(Stmt):
    cond: Expr | VarDecl
    body: Stmt


@dataclass(eq=False)
class DoWhile(Stmt):
    body: Stmt
    cond: Expr


@dataclass(eq=False)
class For(Stmt):
    init: Stmt | None
    cond: Expr | None
    step: Expr | None
    body: Stmt


@dataclass(eq=False)
class RangeFor(Stmt):
    decl: VarDecl
    iterable: Expr
    body: Stmt


@dataclass(eq=False)
class Switch(Stmt):
    cond: Expr
    body: Block


@dataclass(eq=False)
class Case(Stmt):
    """A ``case value:`` label, or ``default:`` when ``value`` is ``None``."""

    value: Expr | None


@dataclass(eq=False)
class Break(Stmt):
    pass


@dataclass(eq=False)
class Continue(Stmt):
    pass


@dataclass(eq=False)
class Return(Stmt):
    value: Expr | None


@dataclass(eq=False)
class Handler(Node):
    decl: VarDecl | None
    body: Block


@dataclass(eq=False)
class Try(Stmt):
    body: Block
    handlers: list[Handler]


@dataclass(eq=False)
class Empty(Stmt):
    pass


# -- declarations ------------------------------------------------------------


@dataclass(eq=False)
class Param(Node):
    name: str | None
    ctype: CType
    default: Expr | None = None


@dataclass(eq=False)
class Function(Stmt):
    """A function, method, constructor, destructor or operator, declared or defined.

    ``scope`` is the class (or namespace) prefix of an out-of-class definition,
    ``Foo`` in ``void Foo::Bar()``; ``kind`` is ``function``, ``constructor``,
    ``destructor`` or ``operator``.
    """

    name: str
    returns: CType
    params: list[Param]
    body: Block | None
    scope: list[str] = field(default_factory=list)
    kind: str = "function"
    const: bool = False
    static: bool = False
    virtual: bool = False
    variadic: bool = False
    inits: list[tuple[str, list[Expr]]] = field(default_factory=list)
    template: list[str] | None = None
    #: ``= 0``, ``= default``, ``= delete``: declared, and never to be given a body.
    special: str | None = None
    #: The template's value parameters, ``N`` of ``template <unsigned N>``, as parameters.
    values: list[Param] = field(default_factory=list)


@dataclass(eq=False)
class Base(Node):
    ctype: CType
    access: str


@dataclass(eq=False)
class ClassDecl(Stmt):
    name: str
    kind: str
    bases: list[Base]
    members: list[Stmt]
    template: list[str] | None = None
    #: Variables declared after the closing brace, ``struct P {...} p;``.
    declarators: list[VarDecl] = field(default_factory=list)


@dataclass(eq=False)
class EnumDecl(Stmt):
    name: str | None
    items: list[tuple[str, Expr | None]]
    scoped: bool = False
    declarators: list[VarDecl] = field(default_factory=list)


@dataclass(eq=False)
class Namespace(Stmt):
    name: str | None
    body: list[Stmt]


@dataclass(eq=False)
class Typedef(Stmt):
    name: str
    ctype: CType


Decl = Stmt


@dataclass(eq=False)
class Unit(Node):
    """A whole macro: its declarations, and its brace-only body if it is an unnamed macro."""

    decls: list[Stmt]
    unnamed: Block | None = None
