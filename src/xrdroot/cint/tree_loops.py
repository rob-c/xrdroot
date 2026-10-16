"""Only a plain ntuple read followed by one histogram fill can become a batch."""
from __future__ import annotations

from dataclasses import dataclass

from .nodes import Block, Call, ExprStmt, Index, Literal, Member, Name, Stmt


@dataclass
class NtupleFill:
    """A candidate whose concrete objects are still checked before batching."""

    tree: Name
    histogram: Name
    column: int


def _method(statement: Stmt, name: str) -> Call | None:
    if not isinstance(statement, ExprStmt) or not isinstance(statement.expr, Call):
        return None
    call = statement.expr
    if not isinstance(call.func, Member) or call.func.name != name:
        return None
    return call if isinstance(call.func.obj, Name) else None


def ntuple_fill(body: Stmt, counter: str) -> NtupleFill | None:
    statements = body.body if isinstance(body, Block) else [body]
    if len(statements) != 2:
        return None
    pair = _calls(statements)
    if pair is None:
        return None
    return _plan(*pair, counter)


def _plan(read: Call, fill: Call, counter: str) -> NtupleFill | None:
    index = read.args[0]
    if not isinstance(index, Name) or index.parts != [counter]:
        return None
    assert isinstance(read.func, Member) and isinstance(read.func.obj, Name)
    assert isinstance(fill.func, Member) and isinstance(fill.func.obj, Name)
    column = _column(fill.args[0], read.func.obj)
    return None if column is None else NtupleFill(read.func.obj, fill.func.obj, column)


def _column(expression: object, tree: Name) -> int | None:
    if not isinstance(expression, Index) or not isinstance(expression.index, Literal):
        return None
    column = expression.index
    if column.kind != 'int' or column.value < 0:
        return None
    call = expression.obj
    method = _getter(call)
    if method is None or not isinstance(method.obj, Name):
        return None
    return int(column.value) if method.obj.parts == tree.parts else None


def _calls(statements: list[Stmt]) -> tuple[Call, Call] | None:
    read, fill = _method(statements[0], 'GetEntry'), _method(statements[1], 'Fill')
    if read is None or fill is None:
        return None
    return (read, fill) if len(read.args) == len(fill.args) == 1 else None


def _getter(call: object) -> Member | None:
    if not isinstance(call, Call) or call.args or not isinstance(call.func, Member):
        return None
    return call.func if call.func.name == 'GetArgs' else None
