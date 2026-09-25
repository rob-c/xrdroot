"""C++ statements as Python statements, one C++ line's worth at a time.

Blocks become indentation, ``if``/``while`` stay themselves, a counted
``for`` whose bound cannot change becomes ``for i in range(...)`` and any
other ``for`` a ``while`` with its step written at the end and before each
``continue``. A ``do``-``while`` is a ``while True`` that tests at the
bottom, a ``switch`` an ``if``/``elif`` chain - or, when a case falls into
the next, a one-pass loop that keeps falling. Each line written remembers
the C++ line it came from.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .base import P
from .ctype import CType
from .emit_calls import CallEmitter
from .loops import assigned_names, bound_of, cases, ends_in_jump, is_literal_step, jumps, stable
from .nodes import (
    Assign,
    Binary,
    Block,
    Break,
    Call,
    Case,
    ClassDecl,
    Comma,
    Continue,
    DeclStmt,
    Delete,
    DoWhile,
    Empty,
    EnumDecl,
    Expr,
    ExprStmt,
    For,
    Function,
    If,
    Index,
    InitList,
    Member,
    RangeFor,
    Switch,
    Literal,
    Name,
    Return,
    Stmt,
    Throw,
    Try,
    Typedef,
    Unary,
    VarDecl,
    While,
)

__all__ = ["StmtEmitter", "Loop"]


@dataclass
class Loop:
    """A loop or switch being written: what ``continue`` must do first, and what ``break`` means."""

    kind: str
    step: list[Expr] = field(default_factory=list)


class StmtEmitter(CallEmitter):
    """The emitter's layer for statements."""

    loops: list[Loop]

    def local_variable(self, decl: VarDecl) -> None:
        raise NotImplementedError

    def local_class_statement(self, decl: ClassDecl) -> None:
        raise NotImplementedError

    def local_enum_statement(self, decl: EnumDecl) -> None:
        raise NotImplementedError

    def returns(self) -> Any:
        """The C++ type the function being written returns, or ``None``."""
        raise NotImplementedError

    def statement(self, node: Stmt) -> None:
        emit = self._STATEMENTS.get(type(node))
        if emit is None:
            raise self.refuse(f"the statement {type(node).__name__}", node)
        emit(self, node)

    def body(self, node: Stmt) -> None:
        """A statement as the indented body of the line just written."""
        with self.out.indented(), self.scoped():
            self.statement(node)

    def _block_statement(self, node: Block) -> None:
        with self.scoped():
            for stmt in node.body:
                self.statement(stmt)

    def _declarations(self, node: DeclStmt) -> None:
        for decl in node.decls:
            self.local_variable(decl)

    def _nothing(self, node: Stmt) -> None:
        pass

    def _class_statement(self, node: ClassDecl) -> None:
        self.local_class(node)

    def _enum_statement(self, node: EnumDecl) -> None:
        self.local_enum(node)

    def _function_statement(self, node: Function) -> None:
        if node.body is not None:
            raise self.refuse("a function defined inside another function", node)

    # -- expression statements -------------------------------------------------

    def _expression(self, node: ExprStmt) -> None:
        self.expression_statement(node.expr)

    def expression_statement(self, expr: Expr) -> None:
        """``expr;`` - with assignment, ``++``, ``delete`` and friends as the statements they are."""
        special = self._EXPRESSIONS.get(type(expr))
        if special is not None and special(self, expr):
            return
        self.out.line(self.value(expr), expr.where)

    def _assign_statement(self, expr: Assign) -> bool:
        target = expr.target
        if isinstance(target, Unary) and target.op == "*":
            found = self.typeof(target.operand)
            if found is None or found.is_object_pointer:
                raise self.refuse("assigning a whole object through a pointer to it", expr)
        if isinstance(target, Call):
            raise self.refuse("assigning to what a call returns by reference, f(i) = v", expr)
        if isinstance(target, Index):
            owner = self.typeof(target.obj)
            if owner is not None and owner.is_string and not owner.dims:
                raise self.refuse("changing one character of a string in place", expr)
        lhs = self.value(target)
        if isinstance(target, Name):
            symbol = self.symbol(target)
            if symbol is not None:
                self.assigned(symbol)
        if expr.op != "=" and self._in_place(expr):
            self.out.line(f"{lhs} {expr.op} {self.value(expr.value)}", expr.where)
            return True
        self.out.line(f"{lhs} = {self.assigned_value(expr)}", expr.where)
        return True

    def _in_place(self, expr: Assign) -> bool:
        """Can ``x op= v`` be written as Python's own ``op=`` - no C conversion to make?"""
        if expr.op in ("/=", "%=", ">>=", "<<="):
            return False
        target = self.typeof(expr.target)
        if target is None or not target.scalar:
            return target is None or not target.scalar
        if target.name == "float" or target.unsigned or target.name in ("short", "char"):
            return False
        if target.integral:
            value = self.typeof(expr.value)
            return value is not None and value.integral
        return True

    def _increment_statement(self, expr: Unary) -> bool:
        if expr.op not in ("++", "--"):
            return False
        op = "+=" if expr.op == "++" else "-="
        one = Literal(expr.where, "int", 1, "int")
        return self._assign_statement(Assign(expr.where, op, expr.operand, one))

    def _delete_statement(self, expr: Delete) -> bool:
        found = self.typeof(expr.operand)
        if expr.array or (found is not None and found.element().scalar):
            self.out.line(f"{self.value(expr.operand)} = None", expr.where)
            return True
        self.out.line(f"delete({self.value(expr.operand)})", expr.where)
        return True

    def _comma_statement(self, expr: Comma) -> bool:
        for item in expr.items:
            self.expression_statement(item)
        return True

    def _throw_statement(self, expr: Throw) -> bool:
        if expr.operand is None:
            self.out.line("raise", expr.where)
            return True
        value = self.value(expr.operand)
        if value.split("(")[0] not in ("RuntimeError", "ValueError", "IndexError", "Exception"):
            value = f"CppException({value})"
        self.out.line(f"raise {value}", expr.where)
        return True

    def _read_statement(self, expr: Binary) -> bool:
        if expr.op != ">>" or not self._streams(expr):
            return False
        targets: list[Expr] = []
        root: Expr = expr
        while isinstance(root, Binary) and root.op == ">>":
            targets.insert(0, root.right)
            root = root.left
        stream = self.value(root)
        for target in targets:
            kind = self.typeof(target)
            name = kind.name if kind is not None else "double"
            self.out.line(f"{self.value(target)} = {stream}.extract({name!r})", expr.where)
        return True

    def _call_statement(self, expr: Call) -> bool:
        return self.call_statement(expr)

    def call_statement(self, expr: Call) -> bool:
        raise NotImplementedError

    _EXPRESSIONS: dict[type, Callable[[StmtEmitter, Any], bool]] = {
        Assign: _assign_statement,
        Unary: _increment_statement,
        Delete: _delete_statement,
        Comma: _comma_statement,
        Throw: _throw_statement,
        Binary: _read_statement,
        Call: _call_statement,
    }

    # -- branches ----------------------------------------------------------------

    def _if(self, node: If) -> None:
        with self.scoped():
            if node.init is not None:
                self.statement(node.init)
            test = self.test(node.cond)
            self.out.line(f"if {test}:", node.where)
            self.body(node.then)
            self._orelse(node.orelse)

    def _orelse(self, orelse: Stmt | None) -> None:
        while isinstance(orelse, If) and orelse.init is None and not isinstance(
            orelse.cond, VarDecl
        ):
            mark = self.out.mark()
            test = self.condition(orelse.cond)
            if self.out.mark() != mark:
                self.out.truncate(mark)
                break
            self.out.line(f"elif {test}:", orelse.where)
            self.body(orelse.then)
            orelse = orelse.orelse
        if orelse is not None:
            self.out.line("else:", orelse.where)
            self.body(orelse)

    def test(self, cond: Expr | VarDecl) -> str:
        """What ``if (...)`` tests; a declaration in the condition is declared first."""
        if isinstance(cond, VarDecl):
            self.local_variable(cond)
            return self.condition(Name(cond.where, [cond.name]))
        return self.condition(cond)

    # -- loops ---------------------------------------------------------------------

    def loop_body(self, body: Stmt, loop: Loop) -> None:
        """A loop's body, indented, with ``break`` and ``continue`` meaning this loop."""
        with self.out.indented():
            self.loop_statements(body, loop)

    def loop_statements(self, body: Stmt, loop: Loop) -> None:
        """A loop's body at the indentation already open."""
        self.loops.append(loop)
        try:
            with self.scoped():
                self.statement(body)
        finally:
            self.loops.pop()

    def _while(self, node: While) -> None:
        if isinstance(node.cond, VarDecl):
            self.out.line("while True:", node.where)
            with self.out.indented(), self.scoped():
                test = self.test(node.cond)
                self.out.line(f"if not ({test}):", node.where)
                with self.out.indented():
                    self.out.line("break", node.where)
                self.loop_statements(node.body, Loop("loop"))
            return
        self.out.line(f"while {self.condition(node.cond)}:", node.where)
        self.loop_body(node.body, Loop("loop"))

    def _do(self, node: DoWhile) -> None:
        if jumps(node.body, Continue):
            first = self.fresh("first")
            self.out.line(f"{first} = True", node.where)
            test = self._truth_at(node.cond, P.OR + 1)
            self.out.line(f"while {first} or {test}:", node.where)
            with self.out.indented():
                self.out.line(f"{first} = False", node.where)
                self.loop_statements(node.body, Loop("loop"))
            return
        self.out.line("while True:", node.where)
        with self.out.indented():
            self.loop_statements(node.body, Loop("loop"))
            self.out.line(f"if not ({self.condition(node.cond)}):", node.cond.where)
            with self.out.indented():
                self.out.line("break", node.cond.where)

    def _for(self, node: For) -> None:
        with self.scoped():
            if self.counted(node):
                return
            if node.init is not None:
                self.statement(node.init)
            test = "True" if node.cond is None else self.condition(node.cond)
            self.out.line(f"while {test}:", node.where)
            steps = _items(node.step)
            with self.out.indented():
                self.loop_statements(node.body, Loop("loop", steps))
                for step in steps:
                    self.expression_statement(step)

    def counted(self, node: For) -> bool:
        """``for (int i = a; i < b; ++i)`` as ``for i in range(a, b)``, when that is the same loop."""
        found = _counter(node)
        if found is None:
            return False
        decl, op, bound, step = found
        if not self._countable(decl, bound, node):
            return False
        start = self.store(decl.ctype, decl.init) if decl.init is not None else "0"
        stop = self._stop(op, bound, step)
        if stop is None:
            return False
        symbol = self.declare(decl.name, "local", decl.ctype)
        stride = "" if step == 1 else f", {step}"
        self.out.line(f"for {symbol.py} in range({start}, {stop}{stride}):", node.where)
        self.loop_body(node.body, Loop("loop"))
        return True

    def _countable(self, decl: VarDecl, bound: Expr, node: For) -> bool:
        if not decl.ctype.integral or decl.name in self.cell_names():
            return False
        if decl.name in assigned_names(node.body):
            return False
        found = self.typeof(bound)
        if found is not None and not found.integral:
            return False
        return stable(bound, node.body)

    def _stop(self, op: str, bound: Expr, step: int) -> str | None:
        upward = step > 0
        if op == "<" and upward or op == ">" and not upward:
            return self.value(bound)
        if op == "!=" and abs(step) == 1:
            return self.value(bound)
        if op == "<=" and upward:
            return f"{self.at(bound, P.ADD)} + 1"
        if op == ">=" and not upward:
            return f"{self.at(bound, P.ADD)} - 1"
        return None

    def cell_names(self) -> set[str]:
        raise NotImplementedError

    def _break(self, node: Break) -> None:
        if self.loops and self.loops[-1].kind == "switch":
            raise self.refuse("a break in the middle of a case", node)
        self.out.line("break", node.where)

    def _continue(self, node: Continue) -> None:
        for loop in reversed(self.loops):
            if loop.kind == "loop":
                for step in loop.step:
                    self.expression_statement(step)
                break
            if loop.kind == "falling":
                raise self.refuse("a continue inside a switch whose cases fall through", node)
        self.out.line("continue", node.where)

    def _return(self, node: Return) -> None:
        if node.value is None:
            self.out.line("return", node.where)
            return
        returns = self.returns()
        if returns is None or returns.is_void:
            self.out.line(f"return {self.value(node.value)}", node.where)
            return
        self.out.line(f"return {self.store(returns.value(), node.value)}", node.where)

    # -- switch ----------------------------------------------------------------------

    def _switch(self, node: Switch) -> None:
        groups = cases(node.body.body)
        if node.body.body and not isinstance(node.body.body[0], Case):
            raise self.refuse("a statement in a switch before its first case", node)
        subject = self._subject(node)
        falls = any(not ends_in_jump(stmts) for _, stmts in groups[:-1])
        breaks = any(jumps(Block(node.where, stmts[:-1]), Break) for _, stmts in groups)
        if falls or breaks:
            self._falling_switch(node, subject, groups)
        else:
            self._chained_switch(subject, groups)

    def _subject(self, node: Switch) -> str:
        value = self.value(node.cond)
        if isinstance(node.cond, Name) and self.symbol(node.cond) is not None:
            return value
        subject = self.fresh("switch")
        self.out.line(f"{subject} = {value}", node.where)
        return subject

    def _labels(self, subject: str, labels: list[Case], others: list[Expr]) -> str:
        values = [self.value(case.value) for case in labels if case.value is not None]
        if any(case.value is None for case in labels):
            rest = [self.value(value) for value in others]
            return f"{subject} not in ({', '.join(rest)},)" if rest else "True"
        if len(values) == 1:
            return f"{subject} == {values[0]}"
        return f"{subject} in ({', '.join(values)})"

    def _chained_switch(self, subject: str, groups: list[tuple[list[Case], list[Stmt]]]) -> None:
        ordered = sorted(groups, key=lambda group: any(c.value is None for c in group[0]))
        self.loops.append(Loop("switch"))
        try:
            for index, (labels, stmts) in enumerate(ordered):
                default = any(case.value is None for case in labels)
                word = "if" if index == 0 else "elif"
                line = "else:" if default and index else f"{word} {self._labels(subject, labels, [])}:"
                if default and not index:
                    line = "if True:"
                self.out.line(line, labels[0].where)
                body = stmts[:-1] if stmts and isinstance(stmts[-1], Break) else stmts
                with self.out.indented(), self.scoped():
                    for stmt in body:
                        self.statement(stmt)
        finally:
            self.loops.pop()

    def _falling_switch(
        self, node: Switch, subject: str, groups: list[tuple[list[Case], list[Stmt]]]
    ) -> None:
        fall = self.fresh("fall")
        every = [case.value for labels, _ in groups for case in labels if case.value is not None]
        self.out.line(f"{fall} = False", node.where)
        self.out.line("while True:", node.where)
        self.loops.append(Loop("falling"))
        try:
            with self.out.indented():
                for labels, stmts in groups:
                    test = self._labels(subject, labels, every)
                    self.out.line(f"if {fall} or {test}:", labels[0].where)
                    with self.out.indented(), self.scoped():
                        self.out.line(f"{fall} = True", labels[0].where)
                        for stmt in stmts:
                            self.statement(stmt)
                self.out.line("break", node.where)
        finally:
            self.loops.pop()

    def _case(self, node: Case) -> None:
        raise self.refuse("a case label outside the top of a switch body", node)

    # -- the rest ---------------------------------------------------------------------

    def _try(self, node: Try) -> None:
        handlers = node.handlers
        if len(handlers) > 1 and handlers[-1].decl is not None:
            raise self.refuse("a try with several catch clauses for different types", node)
        self.out.line("try:", node.where)
        self.body(node.body)
        handler = handlers[0]
        with self.scoped():
            decl = handler.decl
            if decl is not None and decl.name:
                symbol = self.declare(decl.name, "local", decl.ctype)
                self.out.line(f"except Exception as {symbol.py}:", handler.where)
            else:
                self.out.line("except Exception:", handler.where)
            self.body(handler.body)

    def _range_for(self, node: RangeFor) -> None:
        with self.scoped():
            iterable, element = self._iterable(node.iterable)
            decl = node.decl
            if decl.binding is not None:
                names = [self.declare(name, "local").py for name in decl.binding]
                self.out.line(f"for {', '.join(names)} in {iterable}:", node.where)
            elif self._writes_elements(decl, node):
                self._indexed_for(node, iterable, element)
                return
            else:
                ctype = element if decl.ctype.is_auto else decl.ctype
                symbol = self.declare(decl.name, "local", ctype)
                self.out.line(f"for {symbol.py} in {iterable}:", node.where)
            self.loop_body(node.body, Loop("loop"))

    def _iterable(self, node: Expr) -> tuple[str, Any]:
        found = self.typeof(node)
        text = self.value(node)
        if isinstance(node, InitList):
            return text, None
        if found is not None and (found.is_array or found.is_pointer):
            return text, found.element()
        if found is not None and found.args and "map" not in found.name:
            first = found.args[0]
            return text, first if isinstance(first, CType) else None
        return f"iterate({text})", None

    def _writes_elements(self, decl: VarDecl, node: RangeFor) -> bool:
        if not decl.ctype.reference or decl.ctype.const:
            return False
        return decl.name in assigned_names(node.body)

    def _indexed_for(self, node: RangeFor, iterable: str, element: Any) -> None:
        index = self.fresh("i")
        container = self.fresh("items")
        self.out.line(f"{container} = {iterable}", node.where)
        self.out.line(f"for {index} in range(len({container})):", node.where)
        alias = Index(node.where, Name(node.where, [container]), Name(node.where, [index]))
        self.declare(container, "local", CType("auto", dims=[None]) if element is None else
                     CType(element.name, dims=[None]))
        self.declare(index, "local", CType("int"))
        self.declare(node.decl.name, "local", element, alias=alias)
        self.loop_body(node.body, Loop("loop"))

    # -- calls that are assignments ------------------------------------------------

    def call_statement(self, expr: Call) -> bool:
        """``sprintf(buf, ...)``, ``std::swap(a, b)``, ``s.Form(...)``: calls that assign."""
        func = expr.func
        if isinstance(func, Name):
            writer = self._WRITERS.get(func.last)
            if writer is not None and (len(func.parts) == 1 or func.parts[0] == "std"):
                writer(self, expr)
                return True
            return False
        if isinstance(func, Member):
            return self._member_statement(func, expr)
        return False

    def _buffer(self, expr: Call, count: int) -> str:
        """The buffer a C string function writes into, as something Python can assign."""
        if len(expr.args) < count:
            raise self.refuse(f"{expr.func.text}() with too few arguments", expr)  # type: ignore[attr-defined]
        target = expr.args[0]
        if isinstance(target, Name):
            symbol = self.symbol(target)
            if symbol is not None and symbol.kind == "param" and not symbol.cell:
                raise self.refuse("writing a string into a buffer the caller passed", expr)
            if symbol is not None:
                self.assigned(symbol)
        return self.value(target)

    def _sprintf(self, expr: Call) -> None:
        buffer = self._buffer(expr, 2)
        args = ", ".join(self.value(arg) for arg in expr.args[1:])
        self.out.line(f"{buffer} = cformat({args})", expr.where)

    def _snprintf(self, expr: Call) -> None:
        buffer = self._buffer(expr, 3)
        size = self.at(expr.args[1], P.ADD)
        args = ", ".join(self.value(arg) for arg in expr.args[2:])
        self.out.line(f"{buffer} = cformat({args})[: {size} - 1]", expr.where)

    def _strcpy(self, expr: Call) -> None:
        buffer = self._buffer(expr, 2)
        text = f"cstr({self.value(expr.args[1])})"
        if len(expr.args) > 2:
            text += f"[: {self.value(expr.args[2])}]"
        self.out.line(f"{buffer} = {text}", expr.where)

    def _strcat(self, expr: Call) -> None:
        buffer = self._buffer(expr, 2)
        text = f"cstr({self.value(expr.args[1])})"
        if len(expr.args) > 2:
            text += f"[: {self.value(expr.args[2])}]"
        self.out.line(f"{buffer} += {text}", expr.where)

    def _swap(self, expr: Call) -> None:
        if len(expr.args) != 2:
            raise self.refuse("std::swap of other than two things", expr)
        for arg in expr.args:
            if isinstance(arg, Name) and self.symbol(arg) is not None:
                self.assigned(self.symbol(arg))
        a, b = (self.value(arg) for arg in expr.args)
        self.out.line(f"{a}, {b} = {b}, {a}", expr.where)

    _WRITERS: dict[str, Callable[[StmtEmitter, Call], None]] = {
        "sprintf": _sprintf,
        "snprintf": _snprintf,
        "strcpy": _strcpy,
        "strncpy": _strcpy,
        "strcat": _strcat,
        "strncat": _strcat,
        "swap": _swap,
    }

    def _member_statement(self, func: Member, expr: Call) -> bool:
        owner = self.typeof(func.obj)
        if owner is None:
            return False
        target = self.value(func.obj)
        args = [self.value(arg) for arg in expr.args]
        line = None
        if owner.name == "TString" and not owner.pointer and func.name == "Form":
            line = f"{target} = ROOT.TString(cformat({', '.join(args)}))"
        elif owner.is_smart and func.name == "reset":
            line = f"{target} = {args[0] if args else 'None'}"
        elif owner.is_string and not owner.dims:
            line = _string_statement(target, func.name, args)
        if line is None:
            return False
        if isinstance(func.obj, Name) and self.symbol(func.obj) is not None:
            self.assigned(self.symbol(func.obj))
        self.out.line(line, expr.where)
        return True

    _STATEMENTS: dict[type, Callable[[StmtEmitter, Any], None]] = {
        Block: _block_statement,
        ExprStmt: _expression,
        DeclStmt: _declarations,
        Empty: _nothing,
        Typedef: _nothing,
        ClassDecl: _class_statement,
        EnumDecl: _enum_statement,
        Function: _function_statement,
        If: _if,
        While: _while,
        DoWhile: _do,
        For: _for,
        RangeFor: _range_for,
        Switch: _switch,
        Case: _case,
        Break: _break,
        Continue: _continue,
        Return: _return,
        Try: _try,
    }


def _string_statement(target: str, name: str, args: list[str]) -> str | None:
    """A ``std::string`` member that changes the string, as the assignment it is in Python."""
    if name in ("append", "operator+="):
        return f"{target} += cstr({args[0]})"
    if name == "push_back":
        return f"{target} += cstr({args[0]})"
    if name == "clear":
        return f"{target} = ''"
    return None


def _items(step: Expr | None) -> list[Expr]:
    if step is None:
        return []
    if isinstance(step, Comma):
        return list(step.items)
    return [step]


def _counter(node: For) -> tuple[VarDecl, str, Expr, int] | None:
    """The loop variable, comparison, bound and step of a counted ``for``, if it is one."""
    init = node.init
    if not isinstance(init, DeclStmt) or len(init.decls) != 1 or node.cond is None:
        return None
    decl = init.decls[0]
    bound = bound_of(node.cond, decl.name)
    step = is_literal_step(node.step, decl.name) if node.step is not None else None
    if bound is None or step is None or decl.style not in ("=", None, "{}"):
        return None
    if decl.style == "{}":
        return None
    return decl, bound[0], bound[1], step
