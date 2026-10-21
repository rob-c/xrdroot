"""Choose a small, statically certified subset of optional named return-value elision."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ctype import CType
from .nodes import (
    Assign,
    ClassDecl,
    DeclStmt,
    Expr,
    Function,
    Lambda,
    Name,
    Node,
    Return,
    Try,
    Unary,
    VarDecl,
)
from .program import Program, walk

#: Only these parser-resolved native families participate, without custom allocator arguments.
_CONTAINERS = {"std::vector": "vector", "RVec": "rvec", "ROOT::RVec": "rvec",
               "ROOT::VecOps::RVec": "rvec"}


@dataclass(frozen=True)
class Candidate:
    """The exact declaration and return are retained through symbol emission."""

    declaration: VarDecl
    tail: Return


def candidate(function: Function, program: Program, method: bool = False) -> Candidate | None:
    """Unmodelled types, storage and control flow keep the ordinary copying path."""
    if not _function(function, method):
        return None
    assert function.body is not None
    selected = _tail(function)
    if selected is None:
        return None
    tail, name = selected
    nodes = list(walk(function.body))
    if not _body(nodes, tail) or not _binding(nodes, name.last, function, program):
        return None
    declaration = _declaration(function, name.last)
    if declaration is None or not _fresh(declaration):
        return None
    if not _same_type(function.returns, declaration.ctype, program):
        return None
    return Candidate(declaration, tail)


def _function(function: Function, method: bool) -> bool:
    return not (method or function.kind != "function" or function.template
                or function.values or function.constant_evaluation or function.body is None)


def _tail(function: Function) -> tuple[Return, Name] | None:
    assert function.body is not None
    if not function.body.body:
        return None
    tail = function.body.body[-1]
    if not isinstance(tail, Return) or not isinstance(tail.value, Name):
        return None
    name = tail.value
    if len(name.parts) != 1 or name.rooted or name.targs is not None:
        return None
    return tail, name


def _body(nodes: list[Node], tail: Return) -> bool:
    if any(isinstance(node, (Try, Lambda, ClassDecl, Function)) for node in nodes):
        return False
    returns = [node for node in nodes if isinstance(node, Return)]
    return len(returns) == 1 and returns[0] is tail


def _binding(nodes: list[Node], name: str, function: Function, program: Program) -> bool:
    if not _unique_binding(nodes, name, function, program):
        return False
    if any(_rebound(node, name) for node in nodes):
        return False
    references = _references(nodes, function)
    return not any(_alias_write(node, references) for node in nodes)


def _unique_binding(nodes: list[Node], name: str, function: Function, program: Program) -> bool:
    declarations = [node for node in nodes if isinstance(node, VarDecl) and node.name == name]
    if len(declarations) != 1 or name in program.globals:
        return False
    return not any(param.name == name for param in function.params)


def _references(nodes: list[Node], function: Function) -> set[str]:
    references = {node.name for node in nodes if isinstance(node, VarDecl)
                  and node.ctype.reference}
    references.update(param.name for param in function.params
                      if param.ctype.reference and param.name is not None)
    return references


def _declaration(function: Function, name: str) -> VarDecl | None:
    assert function.body is not None
    declarations = [decl for stmt in function.body.body if isinstance(stmt, DeclStmt)
                    for decl in stmt.decls if decl.name == name]
    return declarations[0] if len(declarations) == 1 else None


def _fresh(declaration: VarDecl) -> bool:
    return (not declaration.static and not declaration.storage
            and declaration.binding is None and declaration.init is None
            and declaration.style in (None, "()", "{}"))


def _named(node: Expr, name: str) -> bool:
    return isinstance(node, Name) and node.parts == [name] and not node.rooted


def _rebound(node: Node, name: str) -> bool:
    if isinstance(node, Assign):
        return _named(node.target, name)
    if isinstance(node, Unary) and node.op in ("++", "--"):
        return _named(node.operand, name)
    return False


def _alias_write(node: Node, references: set[str]) -> bool:
    """Whole-object alias assignment is outside the first certified binding scope."""
    if not isinstance(node, Assign):
        return False
    if isinstance(node.target, Unary):
        return node.target.op == "*"
    return isinstance(node.target, Name) and node.target.last in references


def _qualified(ctype: CType) -> bool:
    return bool(ctype.pointer or ctype.reference or ctype.dims or ctype.const
                or ctype.volatile or ctype.enum)


def _same_type(returned: CType, declared: CType, program: Program) -> bool:
    native = _native_type(returned, program)
    return (native is not None and native[0] != "scalar"
            and _native_type(declared, program) == native)


def _native_type(ctype: CType, program: Program) -> tuple[Any, ...] | None:
    if _qualified(ctype):
        return None
    if ctype.arithmetic:
        return None if ctype.is_bool else ("scalar", ctype.name)
    family = _CONTAINERS.get(ctype.name)
    if family is None or len(ctype.args) != 1:
        return None
    if ctype.name.rsplit("::", 1)[-1] in program.classes:
        return None
    element = ctype.args[0]
    if not isinstance(element, CType):
        return None
    nested = _native_type(element, program)
    return (family, nested) if nested is not None else None
