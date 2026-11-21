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

That stand-in is the template's alone. A template beside other functions of
its name - ROOT's ``generate<T>(T *&)`` beside ``generate(void *, TClass *)``
- is the one a call names only where no other function of the name takes the
call: ``generate(p)`` is the template, ``generate(void *, TClass *)`` taking
two arguments where the call passes one, and the others are written as they
are. Written once for every set of arguments, the template has no
``typeid(T)`` to write and refuses its own uses, but a call that is the other
function's still reaches it.

What binds a call to a function of a name is the arguments', not only their
number: ``test_value(in, ref)`` of ``test_value(const std::string &, const
char *)`` passes a ``char *`` where the template ``test_value(T, V)`` would
take it as ``V`` and ask for ``std::numeric_limits<char *>`` - no such thing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .ctype import CType
from .emit_classes import ClassEmitter
from .emit_funcs import kind_of
from .instantiate import deduce_call
from .nodes import Call, Function, Member, Name
from .substitute import substituted
from .symbols import Scope, Symbol, python_name
from .writer import Writer

__all__ = ["TemplateEmitter"]


def _takes(func: Function, count: int) -> bool:
    """Can ``func`` be called with ``count`` arguments: as many as its parameters, or fewer
    where the ones left out have defaults."""
    fewest = sum(1 for param in func.params if param.default is None)
    most = 255 if func.variadic else len(func.params)
    return fewest <= count <= most


def _binds(func: Function, given: list[CType | None]) -> bool:
    """Can arguments of the types ``given`` be passed to the parameters of ``func``, each to
    the parameter it is the argument of - one whose type the translator does not know may be."""
    return all(
        arg is None or _passes(param.ctype, arg)
        for param, arg in zip(func.params, given, strict=False)
    )


def _fits(func: Function, given: list[CType | None]) -> bool:
    """Does a call of the arguments ``given`` reach ``func`` at all."""
    return _takes(func, len(given)) and _binds(func, given)


def _fitting(
    funcs: list[Function], given: list[CType | None], templated: bool
) -> list[Function]:
    """The functions of ``funcs`` a call of the arguments ``given`` reaches: the templates with
    a body, or the ones that are no templates, since a body is what makes a template one here."""
    return [func for func in funcs if _is_template(func) is templated and _fits(func, given)]


def _is_template(func: Function) -> bool:
    """Is ``func`` a function template with a body: the kind a call of its name instantiates."""
    return func.template is not None and bool(func.body)


def _sole_template(func: Function) -> Function | None:
    """The one function of a name, if that is a template with a body."""
    return func if _is_template(func) else None


def _passes(param: CType, arg: CType) -> bool:
    """Does an argument of type ``arg`` pass for a parameter of type ``param``: what the run's
    overload dispatcher compares, and a number of either kind for a number, as C++ converts
    one, and a C string for a string class, which it converts to."""
    wanted, came = kind_of(param), kind_of(arg)
    if "None" in (wanted, came):
        return True
    numbers = (kind_of(CType("int")), kind_of(CType("double")))
    return wanted == came or (wanted in numbers and came in numbers)


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
        """The function template a call of ``symbol`` calls: the one template of its name that
        takes the call as it is - ``generate(p)`` of ``generate<T>(T *&)`` beside
        ``generate(void *, TClass *)`` is the template, ``T`` deduced from ``p`` - a template
        with a body, since only the run chooses between a set's ordinary functions."""
        if symbol is None or symbol.kind not in ("function", "static", "method"):
            return None
        given = [self.typeof(arg) for arg in node.args]
        return self._template_taking(self.function_named(symbol), given)

    @staticmethod
    def _template_taking(funcs: list[Function], given: list[CType | None]) -> Function | None:
        """The one template among ``funcs`` a call of the arguments ``given`` names: a
        template with a body - a set of two templates of the same arity is the run's - where
        none of the name's ordinary functions takes the same arguments, which C++ prefers to
        a template, or ``test_value(in, ref)`` of ``test_value(const std::string &, const
        char *)`` would be the template, a ``V`` no ``numeric_limits`` is written for."""
        if len(funcs) == 1:
            return _sole_template(funcs[0])
        fitting = _fitting(funcs, given, True)
        if len(fitting) != 1 or _fitting(funcs, given, False):
            return None
        return fitting[0]

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

    def function_instance(
        self, func: Name, node: Call, symbol: Symbol | None
    ) -> tuple[str, Function] | None:
        """The Python name of the instance a call of a function template calls - asked for
        now, to be written before the declaration that calls it - and the instance, or
        ``None``."""
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
        return key, self.program.templates.functions[key]

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

    def overloaded(
        self,
        name: str,
        py: str,
        funcs: list[Function],
        emit: Callable[[Function, str], None],
        method: bool = False,
    ) -> None:
        """An overload set whose template cannot be written for arguments the translator does
        not know - one with a ``typeid(T)`` in it, say - keeps its other functions callable:
        the template refuses its own uses, typed by its own arity, and a call of the
        arguments of another function of the set still reaches that function."""
        super().overloaded(name, py, funcs, self._templated(emit, method), method)

    def _templated(
        self, emit: Callable[[Function, str], None], member: bool
    ) -> Callable[[Function, str], None]:
        """``emit``, with a template of the set standing in for itself where it cannot be
        written once for every set of arguments."""

        def one(func: Function, py: str) -> None:
            if func.template is None:
                emit(func, py)
                return
            self.generic(
                lambda: emit(func, py), lambda why: self._stand_in(func, py, why, member)
            )

        return one

    def _stand_in(self, func: Function, py: str, why: str, member: bool) -> None:
        """A template of an overload set, as the function refusing a use whose arguments were
        not worked out - the set's other functions written as they are."""
        what = f"the {'member' if member else 'function'} template {func.name}"
        self.out.line(f"def {py}(*args, **kwargs):", func.where)
        with self.out.indented():
            self.out.line(f"uninstantiated({what!r}, {why!r})", func.where)

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
