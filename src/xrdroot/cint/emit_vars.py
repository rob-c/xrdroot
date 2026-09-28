"""Declared variables as Python assignments, with the value C++ would have started them at.

``int n;`` is ``n = 0`` (C++ leaves it indeterminate; zero is the value no
macro can be relying on otherwise), ``double x[3] = {1}`` is an array padded
with zeros, ``TH1F h("h", ...)`` is ``h = ROOT.TH1F("h", ...)``, ``TString s =
"a"`` is ``ROOT.TString('a')``, ``TLorentzVector v = w`` is a copy of ``w``.
A variable whose address is taken is put in a :class:`Cell`; a reference
bound to an element or a member is not a variable at all, but a second name
for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from .ctype import CType
from .emit_expr import zero
from .emit_names import type_text
from .emit_stmt import StmtEmitter
from .nodes import (
    Call,
    Expr,
    Index,
    InitList,
    Lambda,
    Literal,
    Member,
    Name,
    New,
    This,
    Unary,
    VarDecl,
)
from .program import walk
from .symbols import Symbol

__all__ = ["VariableEmitter", "Context"]

#: The standard containers, which are built from a Python list of what a braced list holds.
CONTAINERS = ("vector", "list", "deque", "set", "array", "map", "unordered_map", "RVec")

#: The ROOT string class, which a C string stored into one is converted to.
STRING_CLASSES = frozenset({"TString"})


@dataclass(eq=False)
class Context:
    """A function (or lambda) being written: the names it must declare ``global``/``nonlocal``."""

    owner: Any
    returns: CType | None
    kind: str = "function"
    globals: set[str] = field(default_factory=set)
    nonlocals: set[str] = field(default_factory=set)
    declared: set[int] = field(default_factory=set)
    cells: set[str] = field(default_factory=set)


class VariableEmitter(StmtEmitter):
    """The emitter's layer for declared variables."""

    contexts: list[Context]
    #: Template parameter names in scope, which a variable cannot be default-built from.
    template_names: set[str]

    # -- the function context ----------------------------------------------------

    def declare(self, name: str, kind: str, ctype: CType | None = None, **extra: Any) -> Symbol:
        symbol = super().declare(name, kind, ctype, **extra)
        self.contexts[-1].declared.add(id(symbol))
        return symbol

    def assigned(self, symbol: Any) -> None:
        if symbol is None or not self.contexts:
            return
        context = self.contexts[-1]
        if symbol.kind == "global":
            context.globals.add(symbol.py)
        elif context.kind == "lambda" and symbol.kind in ("local", "param"):
            if id(symbol) not in context.declared and not symbol.cell:
                context.nonlocals.add(symbol.py)

    def returns(self) -> Any:
        return self.contexts[-1].returns if self.contexts else None

    def cell_names(self) -> set[str]:
        return self.contexts[-1].cells if self.contexts else set()

    # -- declarations ------------------------------------------------------------

    def local_variable(self, decl: VarDecl) -> None:
        if decl.binding is not None:
            self._binding(decl)
            return
        if decl.static and self.contexts:
            self.static_local(decl)
            return
        ctype = sized(self.declared_type(decl), decl)
        alias = self.alias_of(decl, ctype)
        if alias is not None:
            self.declare(decl.name, "local", ctype, alias=alias)
            return
        destructs = self._destructible(decl, ctype)
        value = self.initial(decl, ctype)
        cell = decl.name in self.cell_names() and addressable(ctype)
        symbol = self.declare(decl.name, "local", ctype, cell=cell)
        self.write_variable(symbol, value, decl)
        if destructs:
            assert self.destructed is not None
            self.destructed.append(symbol.py)

    def _destructible(self, decl: VarDecl, ctype: CType | None) -> bool:
        """Does ``decl`` hold an object whose destructor does something as its scope ends?

        In a row of statements the rest of them go in a ``try`` that runs it
        (see :meth:`statements`); anywhere else - a ``for``'s first clause, an
        ``if``'s condition - it is refused.
        """
        if ctype is None or ctype.pointer or ctype.reference or ctype.dims:
            return False
        info = self.program.classes.get(ctype.name)
        if info is None:
            return False
        destructors = info.methods.get("~" + info.name, [])
        if not any(func.body is not None for func in destructors):
            return False
        if self.destructed is None:
            why = f"the local {info.name} {decl.name} declared in a condition or a for's first part"
            raise self.refuse(f"{why}, whose destructor C++ runs as that statement ends", decl)
        return True

    def write_variable(self, symbol: Symbol, value: str, decl: VarDecl) -> None:
        if symbol.cell:
            ctype = symbol.ctype
            kind = f", {ctype.name!r}" if ctype is not None and ctype.scalar else ""
            value = f"Cell({value}{kind})"
        self.out.line(f"{symbol.py} = {value}", decl.where)

    def _binding(self, decl: VarDecl) -> None:
        assert decl.binding is not None
        if decl.init is None:
            raise self.refuse("a structured binding with nothing to unpack", decl)
        value = self.value(decl.init)
        names = [self.declare(name, "local").py for name in decl.binding]
        self.out.line(f"{', '.join(names)} = {value}", decl.where)

    def declared_type(self, decl: VarDecl) -> CType | None:
        """The type a declaration gives its name: as written, or, for ``auto``, as initialised."""
        ctype = decl.ctype
        if not ctype.is_auto:
            return ctype
        init = decl.init if decl.init is not None else (decl.args[0] if decl.args else None)
        found = self.typeof(init) if init is not None else None
        if found is None:
            return None
        if ctype.pointer and not found.pointer:
            return found
        return found.value() if not found.dims else found

    def alias_of(self, decl: VarDecl, ctype: CType | None) -> Expr | None:
        """The element or member a reference names, when it is a second name for one."""
        init = decl.init
        if not decl.ctype.reference or init is None or not _value_like(ctype, init):
            return None
        if isinstance(init, Name):
            return init
        return init if isinstance(init, (Index, Member)) and self._fixed(init) else None

    def _fixed(self, init: Expr) -> bool:
        for node in walk(init):
            if isinstance(node, (Call, Unary)) or (
                isinstance(node, Name) and node.last in self.cell_names()
            ):
                return False
        return True

    # -- initial values ------------------------------------------------------------

    def initial(self, decl: VarDecl, ctype: CType | None) -> str:
        """The Python for what ``decl`` starts as, converted as C++ would convert it."""
        if ctype is None:
            source = decl.init if decl.init is not None else (decl.args[0] if decl.args else None)
            if source is None:
                raise self.refuse(f"auto {decl.name} with nothing to take its type from", decl)
            return self.value(source)
        if ctype.dims:
            return self.array_initial(decl, ctype)
        if decl.style in ("()", "{}"):
            return self.constructed(ctype, decl.args, decl)
        if decl.init is not None:
            return self.assigned_initial(ctype, decl.init)
        return self.default(ctype, decl)

    def default(self, ctype: CType, decl: VarDecl) -> str:
        """What a variable declared with no initialiser holds: zero, ``''``, ``None``, ``T()``."""
        if ctype.scalar or ctype.pointer or ctype.is_smart:
            return zero(ctype)
        if ctype.is_string:
            return "''"
        if ctype.name in self.template_names:
            raise self.refuse(f"a variable of the template parameter type {ctype.name}", decl)
        if ctype.callable:
            return "None"
        return f"{self.class_expr(ctype)}()"

    def constructed(self, ctype: CType, args: list[Expr], decl: VarDecl) -> str:
        """``T x(a, b)`` or ``T x{a, b}``: a scalar converted, an object built."""
        if not args:
            return self.default(ctype, decl)
        if ctype.scalar:
            return self.store(ctype, args[0])
        if ctype.pointer or ctype.is_smart or ctype.callable or ctype.is_string:
            return self._held(ctype, args)
        items = self.constructor_arguments(ctype, args)
        return self._built(ctype, items, decl.style == "{}")

    def _built(self, ctype: CType, items: str, braces: bool) -> str:
        """An object built from ``items``; a container from a braced list is built from a list."""
        if braces and _container(ctype):
            return self._container_of(ctype, items)
        return f"{self.class_expr(ctype)}({items})"

    def _container_of(self, ctype: CType, items: str) -> str:
        """A container from a braced list; a plain list when C++ deduced its type from one."""
        if not ctype.args:
            return f"[{items}]"
        return f"{self.class_expr(ctype)}([{items}])"

    def _held(self, ctype: CType, args: list[Expr]) -> str:
        """A pointer, callable or string built from arguments: what the first one is."""
        if not ctype.is_string:
            return self.value(args[0])
        return f"cstr({self.value(args[0])})" if len(args) == 1 else self._string_of(args)

    def _string_of(self, args: list[Expr]) -> str:
        count, char = (self.value(arg) for arg in args[:2])
        return f"chr({char}) * {count}"

    def assigned_initial(self, ctype: CType, init: Expr) -> str:
        """``T x = init``: converted, wrapped in its class, or copied, as C++'s rules say."""
        if isinstance(init, InitList) and init.ctype is None:
            return self._braced_initial(ctype, init)
        if self._converted_by_constructor(ctype, init):
            return f"{self.class_expr(ctype)}({self.value(init)})"
        return self.store(ctype, init)

    def _braced_initial(self, ctype: CType, init: InitList) -> str:
        """``T x = {a, b}``: the first item for a number, the class built from them otherwise."""
        if ctype.scalar:
            return self.store(ctype, init.items[0]) if init.items else zero(ctype)
        items = ", ".join(self.value(item) for item in init.items)
        if _container(ctype):
            return self._container_of(ctype, items)
        return f"{self.class_expr(ctype)}({items})"

    def _converted_by_constructor(self, ctype: CType, init: Expr) -> bool:
        """Does C++ build a ``ctype`` from ``init`` - ``TString s = "a"`` - rather than copy one?"""
        found = self.typeof(init)
        if ctype.name in STRING_CLASSES and not ctype.pointer:
            return found is None or found.name not in STRING_CLASSES
        if not _by_value(ctype):
            return False
        return isinstance(init, Literal) or (
            found is not None and (found.arithmetic or found.is_string)
        )

    def array_initial(self, decl: VarDecl, ctype: CType) -> str:
        """``double a[3] = {1, 2}``: an array of zeros, the given values first."""
        init = (
            decl.init
            if decl.init is not None
            else (InitList(decl.where, decl.args) if decl.args else None)
        )
        element = CType(ctype.name, ctype.args, ctype.pointer, False, ctype.const)
        if element.name in ("char", "signed char", "unsigned char") and not element.pointer:
            return self._char_array(decl, ctype, init)
        dims = self._dimensions(decl, ctype, init)
        values = f", {self.value(init)}" if init is not None else ""
        if element.scalar:
            return f"array({element.name!r}, {dims}{values})"
        if element.pointer or element.is_string:
            return f"array({type_text(element)!r}, {dims}{values})"
        return f"array({type_text(element)!r}, {dims}{values}, make={self.class_expr(element)})"

    def _dimensions(self, decl: VarDecl, ctype: CType, init: Expr | None) -> str:
        dims = []
        for dim in ctype.dims:
            if dim is None:
                # A size an initialiser gives was given it by ``sized``; this has none.
                raise self.refuse(f"the array {decl.name}[] with no size to give it", decl)
            dims.append(str(dim) if isinstance(dim, int) else self.value(dim))
        return dims[0] if len(dims) == 1 else f"({', '.join(dims)})"

    def _char_array(self, decl: VarDecl, ctype: CType, init: Expr | None) -> str:
        """``char name[20]``: a C string, which is a Python ``str`` here."""
        if len(ctype.dims) > 1:
            items = self.value(init) if init is not None else "[]"
            dims = self._dimensions(decl, ctype, init)
            return f"array('char*', {dims.split(',')[0].strip('(')}, {items})"
        if init is None:
            return "''"
        if isinstance(init, InitList):
            return repr(_characters(init, decl, self.refuse))
        return self.value(init)

    def static_local(self, decl: VarDecl) -> None:
        """A ``static`` local: one :class:`Static` for the program's life, made at module level.

        C++ initialises a static local the first time control reaches its
        declaration, and never again; so does the translation, unless the
        value is a constant, which C++ (and the translation) sets before
        anything runs.
        """
        ctype = sized(self.declared_type(decl), decl)
        kind = f", {ctype.name!r}" if ctype is not None and ctype.scalar else ""
        symbol = self.declare(decl.name, "local", ctype, cell=True)
        symbol.py = self.fresh(f"{self._static_owner()}_{decl.name}")
        if _constant_static(decl, ctype):
            value = self.initial(decl, ctype)
            self.module_statics.append((f"{symbol.py} = Static({value}{kind})", decl.where))
            return
        initial = zero(ctype) if ctype is not None and ctype.scalar else "None"
        holder = f"{symbol.py} = Static({initial}{kind}, ready=False)"
        self.module_statics.append((holder, decl.where))
        value = self.initial(decl, ctype)
        self.out.line(f"if not {symbol.py}.ready:", decl.where)
        with self.out.indented():
            self.out.line(f"{symbol.py}.value = {value}", decl.where)
            self.out.line(f"{symbol.py}.ready = True", decl.where)

    def _static_owner(self) -> str:
        """The name a static local's holder is filed under: its class and function."""
        owner = getattr(self.contexts[-1].owner, "name", "lambda")
        stem = f"{self.scope.klass}_{owner}" if self.scope.klass else owner
        return re.sub(r"\W", "_", stem).strip("_") or "static"

    #: The module-level lines the static locals of the declaration being written become.
    module_statics: list[tuple[str, Any]]


def _constant_static(decl: VarDecl, ctype: CType | None) -> bool:
    """Is a static local's first value a constant, which C++ sets before anything runs?"""
    if ctype is None or not (ctype.pointer or ctype.arithmetic or ctype.is_string):
        return False
    return not any(isinstance(node, (Name, Call, New, Lambda, This)) for node in walk(decl))


def _characters(init: InitList, decl: VarDecl, refuse: Any) -> str:
    """``char s[8] = {'a', 'b', 0}``: the C string the characters spell, to the first NUL."""
    if not all(isinstance(item, Literal) and item.kind in ("char", "int") for item in init.items):
        why = f"the character array {decl.name} initialised one char at a time from variables"
        raise refuse(why, decl)
    text = "".join(chr(int(item.value)) for item in init.items)  # type: ignore[attr-defined]
    return text.split("\0", 1)[0]


def _value_like(ctype: CType | None, init: Expr) -> bool:
    """Is what a reference is bound to a number or string - not an object, already shared?"""
    if ctype is None:
        return not isinstance(init, Name)
    return ctype.scalar or ctype.is_string


def _by_value(ctype: CType) -> bool:
    """An object held by value: not a pointer, smart pointer or reference to one."""
    held = ctype.pointer or ctype.is_smart or ctype.reference or ctype.callable
    return ctype.is_class and not held


def addressable(ctype: CType | None) -> bool:
    """Does ``&x`` need a cell - is ``x`` a value, not an object whose address is itself?"""
    if ctype is None or ctype.dims:
        return False
    return ctype.scalar or ctype.is_string or ctype.pointer > 0 or ctype.is_smart


def _container(ctype: CType) -> bool:
    return ctype.name.split("::")[-1] in CONTAINERS


def sized(ctype: CType | None, decl: VarDecl) -> CType | None:
    """``int a[] = {1, 2, 3}`` has the size its initialiser gives it, which ``sizeof`` needs."""
    if ctype is None or not ctype.dims or ctype.dims[0] is not None:
        return ctype
    init = decl.init
    if isinstance(init, Literal) and init.kind == "str":
        return replace(ctype, dims=[len(init.value) + 1, *ctype.dims[1:]])
    if not isinstance(init, InitList):
        return ctype
    return replace(ctype, dims=[len(init.items), *ctype.dims[1:]])
