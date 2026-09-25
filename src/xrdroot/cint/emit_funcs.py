"""Functions, methods and lambdas as ``def``s; overload sets as one name that dispatches.

A parameter taken by non-const reference to a number is a cell: the caller
hands one over, and the function reads and writes ``.value``. A parameter
whose address the function takes is put in a cell on entry. A global the
function assigns is declared ``global``, and a lambda capturing by
reference and assigning is given ``nonlocal``. Overloads are written as
``f__1``, ``f__2``... and ``f`` becomes an :class:`Overloaded` over them,
choosing by argument count and then by the Python types of the arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import Out, P
from .ctype import CType
from .emit_vars import Context, VariableEmitter
from .nodes import Function, Lambda, Param
from .program import by_reference

__all__ = ["FunctionEmitter", "kind_of"]

#: The Python the overload dispatcher checks an argument of each kind of parameter against.
KINDS = {"integral": "INTEGRAL", "floating": "REAL", "string": "str"}


def kind_of(ctype: CType) -> str:
    """What Python type an argument for a ``ctype`` parameter must be, to pick an overload."""
    if ctype.integral and not ctype.reference:
        return KINDS["integral"]
    if ctype.floating:
        return KINDS["floating"]
    if ctype.is_string:
        return KINDS["string"]
    return "None"


class FunctionEmitter(VariableEmitter):
    """The emitter's layer for functions."""

    def parameters(self, params: list[Param], method: bool) -> list[str]:
        """The Python parameter list, declaring each parameter in the scope just opened."""
        items = ["self"] if method else []
        cells = self.cell_names()
        for index, param in enumerate(params):
            name = param.name or f"arg{index}"
            default = ""
            if param.default is not None:
                default = "=" + self.store(param.ctype.value(), param.default)
            cell = by_reference(param.ctype) or name in cells
            symbol = self.declare(name, "param", param.ctype.value(), cell=cell)
            items.append(symbol.py + default)
        return items

    def function(
        self,
        func: Function,
        py: str,
        method: bool = False,
        decorator: str | None = None,
        prologue: Callable[[], None] | None = None,
    ) -> None:
        """``def py(...):`` and the body of ``func``, in a scope of its own."""
        context = Context(func, func.returns if func.kind != "constructor" else None)
        context.cells = self.program.cell_names(func)
        context.start = self.out.mark()
        self.contexts.append(context)
        try:
            with self.scoped("function", function=func):
                self._function_body(func, py, method, decorator, prologue)
        finally:
            self.contexts.pop()
        for offset, (line, where) in enumerate(context.statics):
            self.out.insert(context.start + offset, line, where)

    def _function_body(
        self,
        func: Function,
        py: str,
        method: bool,
        decorator: str | None,
        prologue: Callable[[], None] | None,
    ) -> None:
        context = self.contexts[-1]
        template = set(func.template or [])
        self.template_names |= template
        params = self.parameters(func.params, method)
        if func.variadic:
            params.append("*varargs")
        if decorator:
            self.out.line(decorator, func.where)
        self.out.line(f"def {py}({', '.join(params)}):", func.where)
        with self.out.indented():
            head = self.out.mark()
            self._wrap_parameters(func)
            if prologue is not None:
                prologue()
            if func.body is not None:
                for stmt in func.body.body:
                    self.statement(stmt)
            self._declare_scopes(context, head, func)
        self.template_names -= template

    def _wrap_parameters(self, func: Function) -> None:
        """A parameter passed by value whose address is taken goes into a cell on entry."""
        for param in func.params:
            symbol = self.lookup(param.name or "")
            if symbol is not None and symbol.cell and not by_reference(param.ctype):
                self.out.line(f"{symbol.py} = Cell({symbol.py})", param.where)

    def _declare_scopes(self, context: Context, head: int, node: Any) -> None:
        if context.nonlocals:
            names = ", ".join(sorted(context.nonlocals))
            self.out.insert(head, f"nonlocal {names}", node.where)
        if context.globals:
            self.out.insert(head, f"global {', '.join(sorted(context.globals))}", node.where)

    # -- overloads ---------------------------------------------------------------

    def overloaded(
        self,
        name: str,
        py: str,
        funcs: list[Function],
        emit: Callable[[Function, str], None],
        method: bool = False,
    ) -> None:
        """Every function of an overload set, and the dispatcher that chooses between them."""
        if len(funcs) == 1:
            emit(funcs[0], py)
            return
        candidates = []
        ordered = sorted(funcs, key=_specificity)
        for index, func in enumerate(ordered, start=1):
            each = f"{py}__{index}"
            emit(func, each)
            self.out.blank()
            candidates.append(self._candidate(func, each, method))
        self.out.line(f"{py} = Overloaded({name!r}, {', '.join(candidates)})", funcs[0].where)

    @staticmethod
    def _candidate(func: Function, py: str, method: bool) -> str:
        """``(f__1, fewest, most, kinds)``, counting ``self`` for a method."""
        own = 1 if method else 0
        fewest = own + sum(1 for param in func.params if param.default is None)
        most = 255 if func.variadic else own + len(func.params)
        kinds = [kind_of(param.ctype) for param in func.params]
        if method:
            kinds.insert(0, "None")
        listed = ", ".join(kinds)
        return f"({py}, {fewest}, {most}, ({listed}{',' if len(kinds) == 1 else ''}))"

    # -- lambdas -----------------------------------------------------------------

    def lambda_(self, node: Lambda) -> Out:
        """A lambda as a ``def`` written just before the statement that uses it."""
        py = self.fresh("lambda")
        outer_cells = self.cell_names()
        context = Context(node, node.returns, kind="lambda", cells=set(outer_cells))
        self.contexts.append(context)
        try:
            with self.scoped("function", function=node):
                params = self.parameters(node.params, False)
                params.extend(self._captured(node))
                self.out.line(f"def {py}({', '.join(params)}):", node.where)
                with self.out.indented():
                    head = self.out.mark()
                    self.loops, saved = [], self.loops
                    try:
                        for stmt in node.body.body:
                            self.statement(stmt)
                    finally:
                        self.loops = saved
                    self._declare_scopes(context, head, node)
        finally:
            self.contexts.pop()
        return py, P.ATOM

    def _captured(self, node: Lambda) -> list[str]:
        """``[x]`` captures ``x`` by value: bound now, as a default, not looked up later."""
        found = []
        for capture in node.captures:
            if capture.byref or capture.name in ("=", "&", "this"):
                continue
            symbol = self.lookup(capture.name)
            if symbol is not None and symbol.kind in ("local", "param") and not symbol.cell:
                found.append(f"{symbol.py}={symbol.py}")
        return found


def _specificity(func: Function) -> tuple[int, ...]:
    """Overloads with integer parameters are tried before ones taking any number."""
    return tuple(0 if kind_of(p.ctype) == "INTEGRAL" else 1 for p in func.params)
