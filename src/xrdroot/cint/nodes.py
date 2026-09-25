"""The syntax tree of a macro: what the parser builds and the emitter walks.

Every node knows where in the macro it was written (:class:`Where`), which is
what a refusal, and a runtime error mapped back through the source map, names.
Expressions carry a ``ctype`` slot the type pass fills in; statements carry
nothing but their parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

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
    targs: Optional[list[Any]] = None
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
    targs: Optional[list[Any]] = None


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
    ctype: Optional[CType]
    operand: Optional[Expr]


@dataclass(eq=False)
class New(Expr):
    ctype: CType
    args: Optional[list[Expr]]
    count: Optional[Expr] = None
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
    returns: Optional[CType] = None


@dataclass(eq=False)
class InitList(Expr):
    """``{a, b}``, or ``T{a, b}`` when a type is named before it."""

    items: list[Expr]
    ctype: Optional[CType] = None


@dataclass(eq=False)
class Comma(Expr):
    items: list[Expr]


@dataclass(eq=False)
class This(Expr):
    pass


@dataclass(eq=False)
class Throw(Expr):
    operand: Optional[Expr]


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
    init: Optional[Expr] = None
    style: Optional[str] = None
    args: list[Expr] = field(default_factory=list)
    static: bool = False
    #: The names of a structured binding, ``auto [a, b] = ...``.
    binding: Optional[list[str]] = None


@dataclass(eq=False)
class DeclStmt(Stmt):
    decls: list[VarDecl]


@dataclass(eq=False)
class If(Stmt):
    cond: Union[Expr, VarDecl]
    then: Stmt
    orelse: Optional[Stmt] = None
    init: Optional[Stmt] = None


@dataclass(eq=False)
class While(Stmt):
    cond: Union[Expr, VarDecl]
    body: Stmt


@dataclass(eq=False)
class DoWhile(Stmt):
    body: Stmt
    cond: Expr


@dataclass(eq=False)
class For(Stmt):
    init: Optional[Stmt]
    cond: Optional[Expr]
    step: Optional[Expr]
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

    value: Optional[Expr]


@dataclass(eq=False)
class Break(Stmt):
    pass


@dataclass(eq=False)
class Continue(Stmt):
    pass


@dataclass(eq=False)
class Return(Stmt):
    value: Optional[Expr]


@dataclass(eq=False)
class Handler(Node):
    decl: Optional[VarDecl]
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
    name: Optional[str]
    ctype: CType
    default: Optional[Expr] = None


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
    body: Optional[Block]
    scope: list[str] = field(default_factory=list)
    kind: str = "function"
    const: bool = False
    static: bool = False
    virtual: bool = False
    variadic: bool = False
    inits: list[tuple[str, list[Expr]]] = field(default_factory=list)
    template: Optional[list[str]] = None
    #: ``= 0``, ``= default``, ``= delete``: declared, and never to be given a body.
    special: Optional[str] = None


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
    template: Optional[list[str]] = None
    #: Declared with ``struct Foo;`` and nothing more.
    forward: bool = False
    #: Variables declared after the closing brace, ``struct P {...} p;``.
    declarators: list[VarDecl] = field(default_factory=list)


@dataclass(eq=False)
class EnumDecl(Stmt):
    name: Optional[str]
    items: list[tuple[str, Optional[Expr]]]
    scoped: bool = False
    declarators: list[VarDecl] = field(default_factory=list)


@dataclass(eq=False)
class Namespace(Stmt):
    name: Optional[str]
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
    unnamed: Optional[Block] = None
