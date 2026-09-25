"""A whole macro as a Python module: the top layer of the emitter.

The module starts with the one import every translation has, declares
every name the macro declares at namespace scope - so a function may call
one defined below it, as C++ allows - and then writes the declarations in
the order the macro has them. An unnamed macro's block becomes a function
named after the file, so that every macro is run the same way: by calling
the function of its file's name. At the bottom, ``if __name__ ==
"__main__"`` calls it, so a kept translation runs as a script.
"""

from __future__ import annotations

import re
from typing import Any

from .ctype import CType
from .emit_classes import ClassEmitter
from .emit_vars import addressable
from .nodes import ClassDecl, DeclStmt, EnumDecl, Function, Namespace, Stmt, VarDecl
from .program import Program
from .symbols import Symbol, python_name
from .writer import Writer

__all__ = ["Translator", "HEADER", "entry_name"]

#: The first lines of every translation.
HEADER = [
    "# Translated from {name} by xrdroot.cint. ROOT's names are ROOT.<name>; the C++",
    "# semantics Python lacks (integer division, printf, cells for &x) are the runtime's.",
    "from xrdroot.cint.runtime import *  # noqa: F403",
]


def entry_name(stem: str) -> str:
    """The Python name of the function a macro file runs: its file's name, made an identifier."""
    clean = re.sub(r"\W", "_", stem) or "macro"
    return clean if not clean[0].isdigit() else "_" + clean


class Translator(ClassEmitter):
    """Everything the emitter does, from the top: :meth:`translate` a whole program."""

    def __init__(self, program: Program, file: str, stem: str) -> None:
        super().__init__(program, file)
        self.stem = stem
        self.contexts = []
        self.loops = []
        self.template_names = set()
        self.namespaces = _namespaces(program.unit.decls)
        self.scope_where = program.unit.where
        self._emitted: set[str] = set()
        #: The Python name of the function running the macro runs, once known.
        self.entry: str | None = None

    def function_named(self, symbol: Symbol) -> list[Any]:
        if symbol.owner is not None:
            info = self.program.classes.get(symbol.owner)
            return info.methods.get(symbol.name, []) if info is not None else []
        return self.program.overloads(symbol.name)

    # -- the module ----------------------------------------------------------------

    def translate(self) -> Writer:
        for line in HEADER:
            self.out.line(line.format(name=self.file.rsplit("/", 1)[-1]))
        self.predeclare()
        for decl in self.program.order:
            self.top(decl)
        self._unnamed()
        self._footer()
        return self.out

    def predeclare(self) -> None:
        """Every name the macro declares at namespace scope, before any is used."""
        add = self.scope.add
        scoped = [name for name, enum in self.program.enums.items() if enum.scoped]
        for name in [*self.program.classes, *scoped]:
            add(Symbol(name, "class", python_name(name)))
        for name, home in self.program.constants.items():
            py = f"{home}.{name}" if home else python_name(name)
            add(Symbol(name, "constant", py, CType("int")))
        for name in self.program.functions:
            add(Symbol(name, "function", python_name(name)))
        self._predeclare_globals()

    def _predeclare_globals(self) -> None:
        add = self.scope.add
        cells = self.program.global_cells
        for name, var in self.program.globals.items():
            cell = name in cells and addressable(var.ctype)
            add(Symbol(name, "global", python_name(name), var.ctype, cell=cell))

    def top(self, decl: object) -> None:
        if isinstance(decl, ClassDecl):
            self.class_def(decl)
        elif isinstance(decl, EnumDecl):
            self.enum_def(decl)
        elif isinstance(decl, DeclStmt):
            self._globals(decl)
        elif isinstance(decl, Function):
            self._free_function(decl)

    def declarators(self, decls: list[VarDecl]) -> None:
        if self.scope.kind == "module":
            self._globals(DeclStmt(self.scope_where, decls))
        else:
            for var in decls:
                self.local_variable(var)

    def _globals(self, decl: DeclStmt) -> None:
        if self.out.lines and self.out.lines[-1].startswith(" "):
            self.out.blank(2)
        for var in decl.decls:
            if id(var) in self.program.class_statics:
                continue
            symbol = self.lookup(var.name)
            assert symbol is not None
            ctype = self.declared_type(var)
            symbol.ctype = ctype
            self.write_variable(symbol, self.initial(var, ctype), var)

    def _free_function(self, decl: Function) -> None:
        if decl.scope and decl.scope[-1] in self.program.classes:
            return
        if decl.name in self._emitted or decl.body is None:
            return
        if decl.kind == "operator":
            raise self.refuse(f"the operator {decl.name[8:]} defined outside a class", decl)
        self._emitted.add(decl.name)
        symbol = self.lookup(decl.name)
        py = symbol.py if symbol is not None else python_name(decl.name)
        self.out.blank(2)
        self.overloaded(decl.name, py, self.program.overloads(decl.name), self.function)
        self.out.blank()

    def _unnamed(self) -> None:
        block = self.program.unit.unnamed
        if block is None:
            return
        name = entry_name(self.stem)
        func = Function(block.where, name, CType("void"), [], block)
        self.program.cells[id(func)] = self.program.cell_names(self.program.unit)
        self.out.blank(2)
        self.function(func, name)
        self.entry = name

    def _footer(self) -> None:
        if self.entry is None:
            symbol = self.lookup(self.stem)
            if symbol is not None and symbol.kind == "function":
                self.entry = symbol.py
        if self.entry is None:
            return
        self.out.blank(2)
        self.out.line('if __name__ == "__main__":')
        with self.out.indented():
            self.out.line(f"{self.entry}()")


def _namespaces(decls: list[Stmt]) -> set[str]:
    found: set[str] = set()
    for decl in decls:
        if isinstance(decl, Namespace):
            if decl.name:
                found.update(decl.name.split("::"))
            found |= _namespaces(decl.body)
    return found
