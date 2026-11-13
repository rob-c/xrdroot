"""Calls of function templates, each made a call of the instance for its template arguments.

``twice(2.5)`` of ``template <class T> T twice(T x)`` calls ``twice<double>``,
an instance written as a function of its own, ``twice__double``, the first
time a call needs it: its arguments given in the call's ``<...>``, or deduced
from the types of the call's arguments as C++ deduces them, or the template's
defaults. It is written at module level, before the declaration that first
calls it, with any class instance it is the first to need before it. A call
whose template arguments cannot all be worked out - an argument of a type
the translator does not know - calls the template itself, written once for
every such use (:meth:`~.emit_classes.ClassEmitter.generic`).
"""

from __future__ import annotations

from typing import Any

from .ctype import CType
from .emit_classes import ClassEmitter
from .instantiate import deduce_call
from .nodes import Call, Function, Member, Name
from .substitute import substituted
from .symbols import Scope, Symbol, python_name
from .writer import Writer

__all__ = ["TemplateEmitter"]


class TemplateEmitter(ClassEmitter):
    """The emitter's layer for instances of function templates."""

    #: How deep the declaration being written is: a class's nested types are written too.
    _depth: int
    #: The instances asked for while the declaration at the top is written, to go before it,
    #: each with the class it is a member template of, if it is one.
    _pending: list[tuple[Function, str | None]]
    #: Whether instances are being written: what they ask for is written with them.
    splicing = False

    def _template_of(self, symbol: Symbol | None, node: Call) -> Function | None:
        """The function template a call of ``symbol`` calls: the one function of its name,
        a template with a body - not a set of overloads, which only the run chooses between."""
        if symbol is None or symbol.kind not in ("function", "static", "method"):
            return None
        funcs = self.function_named(symbol)
        found: Function | None = funcs[0] if len(funcs) == 1 else None
        return found if found is not None and found.template is not None and found.body else None

    def _bindings(
        self, func: Name | Member, node: Call, template: Function
    ) -> dict[str, Any] | None:
        """The template's arguments for this call, if they can all be worked out here."""
        given = [self.typeof(arg) for arg in node.args]
        known = self.program.templates.arguments
        found = deduce_call(template, list(func.targs or []), given, known)
        if found is None or not all(self._concrete(value) for value in found.values()):
            return None
        return found

    def _concrete(self, value: Any) -> bool:
        """An argument that names no parameter of a template being written generically."""
        if isinstance(value, CType):
            names = {value.name, value.name.split("::")[0]}
            return not (names & self.template_names) and all(
                self._concrete(arg) for arg in value.args
            )
        return not isinstance(value, Name) or value.parts[0] not in self.template_names

    def function_instance(self, func: Name, node: Call, symbol: Symbol | None) -> str | None:
        """The Python name of the instance a call of a function template calls - asked for
        now, to be written before the declaration that calls it - or ``None``."""
        template = self._template_of(symbol, node)
        bindings = self._bindings(func, node, template) if template is not None else None
        if template is None or bindings is None:
            return None
        owner = symbol.owner if symbol is not None else None
        found = self.program.templates.function(template, bindings, owner)
        if found is None:
            return None
        key, made = found
        if made is not None:
            self._made(key, made, owner)
        return key

    def _made(self, key: str, made: Function, owner: str | None) -> None:
        """A new instance, to be written before the declaration being written: its name, and
        those of the class instances it made, known from now on."""
        self.program.analyse(made)
        root = self._root()
        root.add(Symbol(key, "function", key))
        for name in self.program.classes:
            if name not in self.class_symbols:
                self.class_symbols[name] = root.add(Symbol(name, "class", python_name(name)))
        self._pending.append((made, owner))

    def method_instance(self, owner: CType | None, func: Member, node: Call) -> str | None:
        """The Python function a call of a member template calls - its instance for the call's
        arguments, written at module level and handed the object - or ``None``."""
        info = self.program.classes.get(owner.name) if owner is not None else None
        template = self._member_template(info, func.name)
        bindings = self._bindings(func, node, template) if template is not None else None
        if info is None or template is None or bindings is None:
            return None
        found = self.program.templates.function(template, bindings, info.name)
        if found is None:
            return None
        key, made = found
        if made is not None:
            self._made(key, made, info.name)
        return key

    @staticmethod
    def _member_template(info: Any, name: str) -> Function | None:
        """The member template ``name`` of the class ``info``, if it is one, with a body."""
        funcs = info.methods.get(name, []) if info is not None else []
        found: Function | None = funcs[0] if len(funcs) == 1 else None
        return found if found is not None and found.template is not None and found.body else None

    def instance_returns(self, func: Name, node: Call, symbol: Symbol | None) -> CType | None:
        """What a call of a function template returns: its return type, the arguments in."""
        template = self._template_of(symbol, node)
        bindings = self._bindings(func, node, template) if template is not None else None
        if template is None or bindings is None or template.returns.is_auto:
            return None
        return substituted(template.returns, bindings)

    def _root(self) -> Scope:
        scope = self.scope
        while scope.parent is not None:
            scope = scope.parent
        return scope

    # -- writing them ------------------------------------------------------------

    def splice_instances(self, mark: int) -> None:
        """The instances asked for while the declaration from ``mark`` was written, put
        before it - class instances first, as a class may be needed as the module is loaded
        and a function only once it is called."""
        end = mark
        self.splicing = True
        try:
            while self._pending or self._unwritten_classes():
                classes = self._unwritten_classes()
                block = self._apart(lambda chosen=classes: self._next_instances(chosen))
                self.out.splice(mark if classes else end, block)
                mark += len(block.lines) if classes else 0
                end += len(block.lines)
        finally:
            self.splicing = False

    def _next_instances(self, classes: list[str]) -> None:
        """The class instances not written yet - or else the next function instance."""
        self._emitted.update(classes)
        for key in classes:
            self.top(self.program.classes[key].decl)
        if not classes:
            self._instance_function(*self._pending.pop(0))

    def _instance_function(self, func: Function, owner: str | None) -> None:
        """A function instance at module level - a member template's taking the object as
        ``self``, its class's members in scope."""
        self.out.blank(2)
        if owner is None:
            self.function(func, func.name)
        else:
            with self.scoped("class", klass=owner) as scope:
                self._members(self.program.classes[owner], scope.symbols)
                self.function(func, func.name, method=not func.static)
        self.out.blank()

    def _unwritten_classes(self) -> list[str]:
        """Class instances made while a function instance was looked over, not written yet."""
        templates = self.program.templates
        return [
            key
            for key in templates.made
            if key not in self._emitted and templates.points[key] > len(self.program.order)
        ]

    def _apart(self, write: Any) -> Writer:
        """What ``write`` writes, at module level, into a writer of its own."""
        saved = (self.out, self.scope, self.contexts, self.loops, self.template_names,
                 self.module_statics, self.destructed, self.vector_names)  # fmt: skip
        self.out, self.scope, self.contexts, self.loops = Writer(), self._root(), [], []
        self.template_names, self.module_statics, self.destructed = set(), [], None
        self.vector_names = {}
        try:
            write()
            for offset, (line, where) in enumerate(self.module_statics):
                self.out.insert(offset, line, where)
            return self.out
        finally:
            (self.out, self.scope, self.contexts, self.loops, self.template_names,
             self.module_statics, self.destructed, self.vector_names) = saved  # fmt: skip

    #: The names written so far of the free functions and class instances.
    _emitted: set[str]
