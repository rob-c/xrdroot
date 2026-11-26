"""What a name means where it is used: the scopes of a translation and their symbols.

A C++ name can be a local, a parameter, a member of the class whose method
is being translated, a global, a function, a class, an enumerator - or,
when the program declares nothing by that name, one of its library's. Each becomes
different Python (``x``, ``x.value``, ``self.x``, ``Foo.x``, ``CXX.x``),
and a :class:`Scope` chain answers which.
"""

from __future__ import annotations

import builtins
import keyword
from dataclasses import dataclass, field
from typing import Any

from .ctype import CType
from .dialect import active

__all__ = ["Symbol", "Scope", "python_name", "member_name", "RESERVED"]

#: Names the Python a translation writes relies on, which a macro's own names must not hide.
RESERVED = frozenset(
    set(keyword.kwlist)
    | {
        "self",
        "int",
        "float",
        "bool",
        "str",
        "len",
        "range",
        "print",
        "isinstance",
        "min",
        "max",
        "abs",
        "chr",
        "ord",
        "super",
        "object",
        "list",
        "sum",
        "enumerate",
        "iter",
        "next",
        "type",
        "tuple",
        "dict",
        "set",
        "id",
        "None",
        "True",
        "False",
        "staticmethod",
        "Exception",
        "RuntimeError",
        "SystemExit",
        "zip",
        "map",
        "filter",
        "sorted",
        "reversed",
        "round",
        "pow",
        "hex",
        "oct",
        "divmod",
        "open",
        "input",
        "format",
        "vars",
        "exec",
        "eval",
        "compile",
        "globals",
        "locals",
        "hash",
        "all",
        "any",
        "match",
        "case",
        "_",
    }
)


def python_name(name: str, taken: frozenset[str] = frozenset()) -> str:
    """A C++ identifier as a Python one that hides nothing the translation needs."""
    clean = name.replace("$", "_")
    if clean in RESERVED or clean in taken or clean == active().host or hasattr(builtins, clean):
        return clean + "_"
    return clean


def member_name(name: str) -> str:
    """A C++ member's name as a Python attribute: itself, unless it is one of Python's keywords.

    A member may be called ``set`` or ``sum`` - attributes hide nothing - but
    not ``print``'s neighbours ``lambda`` or ``import``, which Python's grammar owns.
    """
    return name + "_" if keyword.iskeyword(name) else name


@dataclass(eq=False)
class Symbol:
    """One declared name: what it is, its C++ type, and the Python that stands for it.

    ``kind`` is ``local``, ``param``, ``global``, ``field``, ``static``,
    ``method``, ``member``, ``function``, ``class``, ``enum`` or ``constant``. A
    symbol with ``cell`` set lives in a :class:`~xrdroot.pycxx.runtime.Cell`; one
    with ``alias`` set is a C++ reference, and every use of it is the
    expression it was bound to. A ``member`` is one of the macro's own member
    functions named through its class, ``Base.Init``: the class's own function,
    which the call hands ``self`` to rather than one an object binds.
    """

    name: str
    kind: str
    py: str
    ctype: CType | None = None
    cell: bool = False
    alias: Any = None
    owner: str | None = None
    #: A ``char *`` holding an object's **address** rather than text - ``char *ele = (char *)obj``
    #: - so moving it by bytes is arithmetic. A ``str`` has no address to move back, which is why
    #: every other ``char *`` is still read as the string it is.
    address: bool = False


@dataclass(eq=False)
class Scope:
    """A block, function, class or the module: its symbols, and the scope around it."""

    kind: str
    parent: Scope | None = None
    symbols: dict[str, Symbol] = field(default_factory=dict)
    #: The function this scope belongs to, when it is inside one.
    function: Any = None
    #: The class whose method this is, when it is one.
    klass: str | None = None

    def lookup(self, name: str) -> Symbol | None:
        scope: Scope | None = self
        while scope is not None:
            found = scope.symbols.get(name)
            if found is not None:
                return found
            scope = scope.parent
        return None

    def global_scope(self) -> Scope:
        """The module's own scope, which a name written from ``::`` is looked up in."""
        scope = self
        while scope.parent is not None:
            scope = scope.parent
        return scope

    def global_lookup(self, name: str) -> Symbol | None:
        """``::name``: the name as the file's own scope has it, skipping every class and block.

        A leading ``::`` is C++'s global namespace, so a member of the class being translated
        of that name is not what it means: ``::Print(h.fSingle)`` inside ``Holder::Print`` is
        the file's free ``Print``, and reading it as ``self.Print`` calls the method again.
        """
        return self.global_scope().symbols.get(name)

    def add(self, symbol: Symbol) -> Symbol:
        self.symbols[symbol.name] = symbol
        return symbol

    def visible(self, name: str) -> bool:
        """Is ``name`` declared in an enclosing scope of this function, so a new one shadows?"""
        scope: Scope | None = self
        while scope is not None and scope.kind == "block":
            if name in scope.symbols:
                return True
            scope = scope.parent
        return scope is not None and scope.kind == "function" and name in scope.symbols
