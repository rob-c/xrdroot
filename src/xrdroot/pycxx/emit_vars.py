"""Declared variables as Python assignments, with the value C++ would have started them at.

``int n;`` is ``n = 0`` (C++ leaves it indeterminate; zero is the value no
program can be relying on otherwise), ``double x[3] = {1}`` is an array padded
with zeros, ``Point p(1, 2)`` is ``p = Point(1, 2)``, a dialect's string class
built from ``"a"`` is built from ``'a'``, ``Point q = p`` is a copy of ``p``.
A variable whose address is taken is put in a :class:`Cell`; a reference
bound to an element or a member is not a variable at all, but a second name
for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .ctype import CType
from .dialect import active
from .emit_expr import _is_null, is_container, zero
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
    Return,
    This,
    Unary,
    VarDecl,
)
from .nrvo import Candidate
from .program import sized as sized_type
from .program import walk
from .symbols import Symbol

__all__ = ["VariableEmitter", "Context"]

#: The classes whose objects do something as their scope ends: the standard's lock guards
#: give their mutex back. A dialect adds its own (:attr:`~.dialect.Dialect.guards`).
GUARDS = frozenset({"std::lock_guard", "std::unique_lock", "std::scoped_lock"})


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
    nrvo_candidate: Candidate | None = None
    nrvo_symbol: Symbol | None = None


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

    def return_local(self, node: Return) -> bool:
        """An eligible tail keeps the exact local symbol, not a same-spelled name."""
        if not self.contexts or not self._selected_return(self.contexts[-1], node):
            return False
        self.out.line(f"return {self.value(node.value)}", node.where)
        return True

    def _selected_return(self, context: Context, node: Return) -> bool:
        selected = context.nrvo_candidate
        if selected is None or node is not selected.tail:
            return False
        assert isinstance(node.value, Name)
        symbol = self.symbol(node.value)
        return symbol is not None and symbol is context.nrvo_symbol and symbol.alias is None

    def _remember_return(self, decl: VarDecl, symbol: Symbol, destructs: bool) -> None:
        context = self.contexts[-1]
        if context.nrvo_candidate is not None and decl is context.nrvo_candidate.declaration:
            context.nrvo_symbol = symbol
        if destructs:
            context.nrvo_candidate = None

    # -- declarations ------------------------------------------------------------

    def local_variable(self, decl: VarDecl) -> None:
        if decl.binding is not None:
            self._binding(decl)
            return
        if decl.static and self.contexts:
            self.static_local(decl)
            return
        ctype = sized(self.declared_type(decl), decl)
        if self._borrowed_variable(decl, ctype):
            return
        alias = self.alias_of(decl, ctype)
        if alias is not None:
            self.declare(decl.name, "local", ctype, alias=alias)
            return
        destructs = self._destructible(decl, ctype)
        value = self.initial(decl, ctype)
        cell = decl.name in self.cell_names() and addressable(ctype)
        symbol = self.declare(decl.name, "local", ctype, cell=cell)
        self.write_variable(symbol, value, decl)
        self._remember_return(decl, symbol, destructs)
        if destructs:
            assert self.destructed is not None
            self.destructed.append(symbol.py)

    def _borrowed_variable(self, decl: VarDecl, ctype: CType | None) -> bool:
        if not decl.ctype.reference or ctype is None or not ctype.scalar or decl.init is None:
            return False
        reference = self.borrowed_reference(decl.init)
        if reference is None:
            return False
        symbol = self.declare(decl.name, "local", ctype, cell=True)
        self.out.line(f"{symbol.py} = {reference}", decl.where)
        return True

    def _destructible(self, decl: VarDecl, ctype: CType | None) -> bool:
        """Does ``decl`` hold an object whose destructor does something as its scope ends?

        In a row of statements the rest of them go in a ``try`` that runs it
        (see :meth:`statements`); anywhere else - a ``for``'s first clause, an
        ``if``'s condition - it is refused.
        """
        if ctype is None or ctype.pointer or ctype.reference or ctype.dims:
            return False
        guarded = ctype.name in GUARDS or ctype.name in active().guards
        if not (guarded or self._destructs(ctype.name)):
            return False
        if self.destructed is None:
            where = "declared in a condition or a for's first part"
            why = f"the local {ctype.name} {decl.name} {where}, whose destructor C++ runs"
            raise self.refuse(f"{why} as that statement ends", decl)
        return True

    def _destructs(self, name: str) -> bool:
        """Does the macro's class ``name`` have a destructor with a body?"""
        info = self.program.classes.get(name)
        destructors = info.methods.get("~" + info.name, []) if info is not None else []
        return any(func.body is not None for func in destructors)

    def write_variable(self, symbol: Symbol, value: str, decl: VarDecl) -> None:
        if symbol.cell:
            ctype = symbol.ctype
            kind = f", {ctype.name!r}" if ctype is not None and ctype.scalar else ""
            if _mentions(value, symbol.py):
                # C++'s own initialiser may take the address of the variable it initialises -
                # `static int x = (f(&x), 0)` - so the cell is made first and filled after.
                self.out.line(f"{symbol.py} = Cell(0{kind})", decl.where)
                self.out.line(f"{symbol.py}.value = {value}", decl.where)
                return
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
        init = _initialiser(decl)
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
        if self._binds_an_object(ctype, args):
            return self.store(ctype, args[0])
        items = self.constructor_arguments(ctype, args)
        return self._built(ctype, items, decl.style == "{}")

    def _binds_an_object(self, ctype: CType, args: list[Expr]) -> bool:
        """Is this a reference taking the object it already refers to - nothing to build?"""
        return (ctype.reference and not ctype.dims and len(args) == 1
                and self._same_object(ctype, args[0]))  # fmt: skip

    def _same_object(self, ctype: CType, arg: Expr) -> bool:
        """Is ``arg`` already the object a reference to ``ctype`` binds - ``const type_info&
        ti(typeid(T))``, which builds nothing?"""
        found = self.typeof(arg)
        if found is None or bool(found.pointer) != bool(ctype.pointer):
            return False
        return found.name.split("::")[-1] == ctype.name.split("::")[-1]

    def _built(self, ctype: CType, items: str, braces: bool) -> str:
        """An object built from ``items``; a container from a braced list is built from a list."""
        if braces and is_container(ctype):
            return self._container_of(ctype, items)
        return f"{self.class_expr(ctype)}({items})"

    def _container_of(self, ctype: CType, items: str) -> str:
        """A container from a braced list; a plain list when C++ deduced its type from one."""
        if not ctype.args and ctype.name.split("::")[-1] not in active().typed_containers:
            return f"[{items}]"
        return f"{self.class_expr(ctype)}([{items}])"

    def _held(self, ctype: CType, args: list[Expr]) -> str:
        """A pointer, callable or string built from arguments: what the first one is."""
        if not ctype.is_string:
            if ctype.is_object_pointer and _is_null(args[0]):
                return "None"
            return self.value(args[0])
        return f"cstr({self.value(args[0])})" if len(args) == 1 else self._string_of(args)

    def _string_of(self, args: list[Expr]) -> str:
        """``std::string(n, c)``, ``n`` of ``c``; ``std::string(istreambuf_iterator<char>(f),
        {})``, the rest of the stream ``f``."""
        stream = _streambuf(args[0])
        if stream is not None:
            return f"stream_rest({self.value(stream)})"
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
        if is_container(ctype):
            return self._container_of(ctype, items)
        return f"{self.class_expr(ctype)}({items})"

    def _converted_by_constructor(self, ctype: CType, init: Expr) -> bool:
        """Does C++ build a ``ctype`` from ``init`` - a string class from ``"a"`` - rather than
        copy one?"""
        found = self.typeof(init)
        strings = active().string_classes
        if ctype.name in strings and not ctype.pointer:
            return found is None or found.name not in strings
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
        if element.name == "char" and not element.pointer:  # text; uint8_t's are bytes
            return self._char_array(decl, ctype, init)
        dims = self._dimensions(decl, ctype, init)
        values = f", {self.value(init)}" if init is not None else ""
        if element.scalar:
            return f"array({element.name!r}, {dims}{values})"
        if element.pointer or element.is_string:
            return f"array({self.spelled(element)!r}, {dims}{values})"
        made = self.class_expr(element)
        return f"array({self.spelled(element)!r}, {dims}{values}, make={made})"

    def _dimensions(self, decl: VarDecl, ctype: CType, init: Expr | None) -> str:
        dims = []
        for dim in ctype.dims:
            if dim is None:
                # A size an initialiser gives was given it by ``sized``; this has none.
                raise self.refuse(f"the array {decl.name}[] with no size to give it", decl)
            dims.append(str(dim) if isinstance(dim, int) else self.value(dim))
        return dims[0] if len(dims) == 1 else f"({', '.join(dims)})"

    def _char_array(self, decl: VarDecl, ctype: CType, init: Expr | None) -> str:
        """``char name[20]``: a C string, which is a Python ``str`` here - or, handed to a
        library to write into (:attr:`~.program.Program.buffers`), a buffer of its bytes."""
        if decl.name in self.program.buffers and len(ctype.dims) == 1:
            made = f"array('char', {self._dimensions(decl, ctype, init)})"
            return made if init is None else f"text_into({made}, {self.value(init)})"
        if len(ctype.dims) > 1:
            # A ``char[N][M]`` is a buffer of the characters, so ``name[i]`` is its row and
            # ``name[i][j]`` a byte of it, which a write goes into in place.
            items = f", {self.value(init)}" if init is not None else ""
            return f"char_array({self._dimensions(decl, ctype, init)}{items})"
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


def sized(ctype: CType | None, decl: VarDecl) -> CType | None:
    """``int a[] = {1, 2, 3}`` has the size its initialiser gives it, which ``sizeof`` needs."""
    return None if ctype is None else sized_type(ctype, decl)


def _mentions(python: str, name: str) -> bool:
    """Does the Python text ``python`` read ``name`` - as a name, not inside a string?"""
    return re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", python) is not None


def _initialiser(decl: VarDecl) -> Expr | None:
    """What an ``auto`` declaration's type is that of: its initialiser - or, for
    ``decltype(m) n;``, the variable ``m``."""
    ctype = decl.ctype
    if ctype.name == "decltype" and ctype.args:
        return Name(decl.where, [str(ctype.args[0])])
    return decl.init if decl.init is not None else (decl.args[0] if decl.args else None)


def _streambuf(node: Expr) -> Expr | None:
    """The stream ``std::istreambuf_iterator<char>(f)`` iterates over, if ``node`` is one."""
    if not (isinstance(node, Call) and isinstance(node.func, Name) and len(node.args) == 1):
        return None
    return node.args[0] if node.func.last == "istreambuf_iterator" else None
