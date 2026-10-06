"""The whole macro at a glance, before a line of Python is written.

Python needs to know some things about a name before its first use that C++
lets a macro say later: which functions are overloaded, which methods are
defined outside their class, and - most of all - which variables have their
address taken. A variable handed to ``SetBranchAddress(&x)`` or to
``TRandom::Rannor(px, py)`` must be a :class:`~xrdroot.cint.runtime.Cell`
from its declaration on, so that every use of it reads the cell. This pass
finds them all.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .ctype import CType
from .nodes import (
    Call,
    ClassDecl,
    DeclStmt,
    EnumDecl,
    Expr,
    Function,
    Lambda,
    Member,
    Name,
    Namespace,
    New,
    Node,
    Param,
    Stmt,
    Typedef,
    Unary,
    Unit,
    VarDecl,
)

__all__ = ["Program", "ClassInfo", "OUT_PARAMETERS", "children", "walk", "by_reference"]

#: ROOT methods that write through references, by name and number of arguments: which ones.
OUT_PARAMETERS: dict[tuple[str, int], tuple[int, ...]] = {
    ("Rannor", 2): (0, 1),
    ("Sphere", 4): (0, 1, 2),
    ("Circle", 3): (0, 1),
    ("GetRandom2", 2): (0, 1),
    ("GetRandom2", 3): (0, 1),
    ("GetRandom3", 3): (0, 1, 2),
    ("GetRandom3", 4): (0, 1, 2),
    ("GetPoint", 3): (1, 2),
    ("GetPoint", 4): (1, 2, 3),
    ("GetRange", 2): (0, 1),
    ("GetRange", 4): (0, 1, 2, 3),
    ("GetRange", 6): (0, 1, 2, 3, 4, 5),
    ("GetParLimits", 3): (1, 2),
    ("GetParameter", 3): (1, 2),
    ("GetBinXYZ", 4): (1, 2, 3),
    ("GetObject", 2): (1,),
    ("HLS2RGB", 6): (3, 4, 5),
    ("RGB2HLS", 6): (3, 4, 5),
    ("HSV2RGB", 6): (3, 4, 5),
    ("RGB2HSV", 6): (3, 4, 5),
    ("GetRGB", 3): (0, 1, 2),
    ("GetHLS", 3): (0, 1, 2),
    ("AbsPixeltoXY", 4): (2, 3),
    ("GetPadPar", 4): (0, 1, 2, 3),
    ("GetMinimumAndMaximum", 2): (0, 1),
    ("mnstat", 6): (0, 1, 2, 3, 4, 5),
    ("GetStats", 4): (0, 1, 2, 3),
    ("GetSigma", 2): (0, 1),
    ("GetSigmaX", 2): (0, 1),
    ("GetSigmaY", 2): (0, 1),
    ("GetRo", 2): (0, 1),
    ("GetBackgroundParameters", 6): (0, 1, 2, 3, 4, 5),
    ("GetTailParameters", 6): (0, 1, 2, 3, 4, 5),
    ("GetTailParameters", 16): tuple(range(16)),
    ("GetAngles", 3): (0, 1, 2),
    ("ShiftToNext", 1): (0,),
    ("ShiftToNext", 2): (0,),
    ("GetAngles", 6): (0, 1, 2, 3, 4, 5),
    ("Det", 2): (0, 1),
    ("Determinant", 2): (0, 1),
    ("Solve", 2): (1,),
}


def children(node: Any) -> Iterator[Node]:
    """The nodes directly inside ``node``, in the order they were written."""
    for item in dataclasses.fields(node):
        yield from _nodes(getattr(node, item.name))


def _nodes(value: Any) -> Iterator[Node]:
    if isinstance(value, Node):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _nodes(item)


def walk(node: Node) -> Iterator[Node]:
    """``node`` and every node inside it, depth first."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(list(children(current))))


@dataclass(eq=False)
class ClassInfo:
    """One class of the macro's: its members, split the ways the translation needs them."""

    decl: ClassDecl
    fields: dict[str, VarDecl] = field(default_factory=dict)
    statics: dict[str, VarDecl] = field(default_factory=dict)
    methods: dict[str, list[Function]] = field(default_factory=dict)
    constants: dict[str, str] = field(default_factory=dict)
    types: list[Stmt] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.decl.name

    def add_method(self, func: Function) -> None:
        found = self.methods.setdefault(func.name, [])
        for index, other in enumerate(found):
            if _same_signature(other, func):
                if func.body is not None:
                    found[index] = _merged(other, func)
                return
        found.append(func)


def _same_signature(one: Function, other: Function) -> bool:
    if len(one.params) != len(other.params):
        return False
    return all(_same_type(a.ctype, b.ctype) for a, b in zip(one.params, other.params, strict=False))


def _same_type(one: CType, other: CType) -> bool:
    return (one.name, one.pointer, one.reference) == (other.name, other.pointer, other.reference)


def _merged(declared: Function, defined: Function) -> Function:
    """A definition, with the defaults and qualifiers its declaration had and it may not."""
    for mine, theirs in zip(defined.params, declared.params, strict=False):
        if mine.default is None:
            mine.default = theirs.default
        if not mine.name:
            mine.name = theirs.name
    defined.static = defined.static or declared.static
    defined.virtual = defined.virtual or declared.virtual
    return defined


class Program:
    """A macro's classes, functions, globals and enumerators, and its variables held in cells."""

    def __init__(self, unit: Unit) -> None:
        self.unit = unit
        self.classes: dict[str, ClassInfo] = {}
        self.functions: dict[str, list[Function]] = {}
        self.globals: dict[str, VarDecl] = {}
        #: Every enumerator, and the enum (or class) it belongs to.
        self.constants: dict[str, str] = {}
        self.enums: dict[str, EnumDecl] = {}
        self.aliases: dict[str, CType] = {}
        #: The declarations at namespace scope, namespaces flattened, in the order written.
        self.order: list[Stmt] = []
        #: For each function (by ``id``), the names in it whose address is taken.
        self.cells: dict[int, set[str]] = {}
        #: The globals whose address is taken somewhere.
        self.global_cells: set[str] = set()
        #: The out-of-class definitions of static members, written in their class instead.
        self.class_statics: set[int] = set()
        self._flatten(unit.decls)
        self._attach()
        self._locals()
        self._escapes()

    def _locals(self) -> None:
        """Classes and enums declared inside functions: known by name, as they are in C++."""
        found = [node for _, _, body in list(self.bodies()) for node in walk(body)]
        for node in found:
            if isinstance(node, ClassDecl):
                self._class(node)
            elif isinstance(node, EnumDecl) and node.name:
                self.enums[node.name] = node

    # -- collecting ---------------------------------------------------------

    def _flatten(self, decls: list[Stmt]) -> None:
        for decl in decls:
            if isinstance(decl, Namespace):
                self._flatten(decl.body)
                continue
            self.order.append(decl)
            self._collect(decl)

    def _collect(self, decl: Stmt) -> None:
        if isinstance(decl, ClassDecl):
            self._class(decl)
        elif isinstance(decl, EnumDecl):
            self._enum(decl, None)
        elif isinstance(decl, DeclStmt):
            for var in decl.decls:
                self._global(var)
        elif isinstance(decl, Typedef):
            self.aliases[decl.name] = decl.ctype

    def _global(self, var: VarDecl) -> None:
        """A variable at namespace scope - or ``int Foo::n = 3;``, a static member's definition."""
        if "::" not in var.name:
            self.globals[var.name] = var
            return
        parts = var.name.split("::")
        var.name = parts[-1]
        owner = self.classes.get(parts[-2])
        if owner is None:
            self.globals[var.name] = var
            return
        owner.statics[var.name] = var
        self.class_statics.add(id(var))

    def _class(self, decl: ClassDecl) -> None:
        info = self.classes.setdefault(decl.name, ClassInfo(decl))
        info.decl = decl
        for member in decl.members:
            self._member(info, member)
        for var in decl.declarators:
            self.globals[var.name] = var

    def _member(self, info: ClassInfo, member: Stmt) -> None:
        if isinstance(member, Function):
            info.add_method(member)
        elif isinstance(member, DeclStmt):
            for var in member.decls:
                (info.statics if var.static else info.fields)[var.name] = var
        elif isinstance(member, EnumDecl):
            self._enum(member, info)
        elif isinstance(member, ClassDecl):
            info.types.append(member)
            self._class(member)

    def _enum(self, decl: EnumDecl, owner: ClassInfo | None) -> None:
        if decl.name:
            self.enums[decl.name] = decl
        if not decl.scoped:
            self._enumerators(decl, owner)
        for var in decl.declarators:
            self.globals[var.name] = var

    def _enumerators(self, decl: EnumDecl, owner: ClassInfo | None) -> None:
        """An unscoped enum's names, which belong to the scope around it: a class, or the macro."""
        home = owner.name if owner is not None else ""
        for item, _ in decl.items:
            if owner is not None:
                owner.constants[item] = owner.name
            self.constants.setdefault(item, home)

    def _attach(self) -> None:
        """Put each function where it belongs: a method in its class, the rest by name."""
        for decl in self.order:
            if not isinstance(decl, Function):
                continue
            owner = self.classes.get(decl.scope[-1]) if decl.scope else None
            if owner is not None:
                owner.add_method(decl)
            elif decl.body is not None or decl.name not in self.functions:
                self._function(decl)

    def _function(self, func: Function) -> None:
        found = self.functions.setdefault(func.name, [])
        for index, other in enumerate(found):
            if _same_signature(other, func):
                found[index] = _merged(other, func) if func.body is not None else other
                return
        found.append(func)

    def overloads(self, name: str) -> list[Function]:
        """The definitions of free function ``name`` - prototypes dropped when one is defined."""
        found = self.functions.get(name, [])
        defined = [func for func in found if func.body is not None]
        return defined or found

    # -- escapes -------------------------------------------------------------

    def bodies(self) -> Iterator[tuple[Node, list[Param], Node]]:
        """Every function body, with its parameters: free functions, methods, the unnamed block."""
        for func in [*_all(self.functions), *self.methods()]:
            if func.body is not None:
                yield func, func.params, func.body
        if self.unit.unnamed is not None:
            yield self.unit, [], self.unit.unnamed

    def methods(self) -> list[Function]:
        """Every method of every class of the macro's."""
        return [func for info in self.classes.values() for func in _all(info.methods)]

    def _escapes(self) -> None:
        for owner, params, body in self.bodies():
            names = set(self._escaping(body))
            local = {param.name for param in params} | _declared(body)
            self.cells[id(owner)] = names & local
            self.global_cells |= names - local

    def cell_names(self, owner: Node) -> set[str]:
        return self.cells.get(id(owner), set())

    def _escaping(self, body: Node) -> Iterator[str]:
        for node in walk(body):
            if isinstance(node, Unary) and node.op == "&" and isinstance(node.operand, Name):
                yield node.operand.last
            elif isinstance(node, Call):
                yield from _written(node.args, self.reference_positions(node))
            elif isinstance(node, (New, VarDecl)) and node.args:
                positions = self.constructor_positions(node.ctype.name, len(node.args))
                yield from _written(node.args, positions)

    def constructor_positions(self, name: str, count: int) -> tuple[int, ...]:
        """Which arguments of a constructor of the macro's class ``name`` it takes by reference."""
        info = self.classes.get(name)
        if info is None:
            return ()
        return self._user_positions(info.methods.get(name, []), count)

    def reference_positions(self, call: Call) -> tuple[int, ...]:
        func = call.func
        if isinstance(func, Member):
            known = OUT_PARAMETERS.get((func.name, len(call.args)))
            if known is not None:
                return known
            return self._user_positions(self._methods(func.name), len(call.args))
        if isinstance(func, Name) and len(func.parts) == 1:
            if func.last in self.classes:
                return self.constructor_positions(func.last, len(call.args))
            return self._user_positions(self.overloads(func.last), len(call.args))
        return ()

    def _methods(self, name: str) -> list[Function]:
        return [f for info in self.classes.values() for f in info.methods.get(name, [])]

    @staticmethod
    def _user_positions(funcs: list[Function], count: int) -> tuple[int, ...]:
        for func in funcs:
            if len(func.params) >= count:
                return tuple(i for i, p in enumerate(func.params) if by_reference(p.ctype))
        return ()


def _written(args: list[Expr], positions: tuple[int, ...]) -> Iterator[str]:
    """The variables handed to a call where it takes them by reference - they live in cells."""
    for index in positions:
        if index < len(args):
            arg = args[index]
            if isinstance(arg, Name) and len(arg.parts) == 1:
                yield arg.last


def _all(functions: dict[str, list[Function]]) -> list[Function]:
    return [func for funcs in functions.values() for func in funcs]


def by_reference(ctype: CType) -> bool:
    """A parameter a caller's variable is written through: a non-const reference to a value."""
    if not ctype.reference or ctype.const:
        return False
    return ctype.scalar or ctype.is_string or ctype.is_pointer


def _declared(body: Node) -> set[str]:
    names: set[str] = set()
    for node in walk(body):
        if isinstance(node, VarDecl):
            names.add(node.name)
        elif isinstance(node, Lambda):
            names.update(param.name for param in node.params if param.name)
    return names
