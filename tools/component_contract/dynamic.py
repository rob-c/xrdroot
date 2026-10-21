"""Expand only exact reviewed dynamic sites into ordinary checked imports."""

from __future__ import annotations

import ast
from dataclasses import replace
from typing import TYPE_CHECKING

from .forwarders import forwarded, validated
from .model import ContractError, Dynamic, Policy
from .tables import facade_targets

if TYPE_CHECKING:
    from .imports import Import, Module


def _key(ref: Import) -> tuple[str, str, str]:
    return ref.module, ref.scope, ref.statement


def _targets(bound: Dynamic, known: dict[str, Module], package: str) -> list[str]:
    result = set(bound.targets)
    for prefix in bound.prefixes:
        if prefix != package and not prefix.startswith(package + "."):
            raise ContractError("Dynamic external imports require finite literal targets.")
        expanded = {n for n in known if n == prefix or n.startswith(prefix + ".")}
        if not expanded:
            raise ContractError(f"Dynamic prefix {prefix} matches no source module.")
        result.update(expanded)
    return sorted(result)


def expand(policy: Policy, refs: list[Import], known: dict[str, Module]) -> list[Import]:
    bindings = _binding_table(policy)
    used: set[tuple[str, str, str]] = set()
    result = []
    valid_forwarder = validated(policy, known)
    for ref in refs:
        result.extend(_expanded_ref(policy, ref, refs, known, bindings, used, valid_forwarder))
    if set(bindings) != used:
        raise ContractError("Unused dynamic import declarations must be removed or reviewed again.")
    return result


def _special(
    policy: Policy, ref: Import, refs: list[Import], known: dict[str, Module]
) -> list[Import] | None:
    if ref.kind != "dynamic":
        return None
    if _registry_site(policy, ref):
        return _registry_dispatch(ref, refs)
    return _facade_dispatch(policy, ref, known)


def _registry_site(policy: Policy, ref: Import) -> bool:
    return (
        ref.module == policy.provider_registry
        and ref.scope == "lookup"
        and ref.statement == "importlib.import_module(module)"
    )


def _registry_dispatch(ref: Import, refs: list[Import]) -> list[Import] | None:
    declarations = [r for r in refs if r.kind == "declaration"]
    return [
        replace(r, line=ref.line, scope=ref.scope, statement=ref.statement, kind="dispatch")
        for r in declarations
    ] or None


def _facade_dispatch(policy: Policy, ref: Import, known: dict[str, Module]) -> list[Import] | None:
    calls = {"_import_module(f'.{module}', __name__)", "_import_module(f'.{name}', __name__)"}
    if ref.module == policy.package and ref.scope == "__getattr__" and ref.statement in calls:
        targets = facade_targets(known[ref.module].source, policy.package)
        return [replace(ref, target=n, kind="import") for n in targets]
    return None


def constants(module: Module) -> dict[str, str]:
    result = {"__name__": module.name, "__package__": module.package}
    tree = ast.parse(module.source)
    values: dict[str, list[str | None]] = {}
    for node in tree.body:
        for name, value in _assignments(node):
            values.setdefault(name, []).append(value)
    result.update({n: v[0] for n, v in values.items() if len(v) == 1 and v[0] is not None})
    return result


def _assignments(node: ast.AST) -> list[tuple[str, str | None]]:
    if isinstance(node, ast.Assign):
        return [(t.id, _literal(node.value)) for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [(node.target.id, _literal(node.value))]
    return []


def _literal(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def string(node: ast.AST | None, values: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return values.get(node.id)
    if isinstance(node, ast.JoinedStr):
        parts = [formatted(n, values) for n in node.values]
        return "".join(p for p in parts if p is not None) if None not in parts else None
    return _literal(node)


def formatted(node: ast.AST, values: dict[str, str]) -> str | None:
    if isinstance(node, ast.FormattedValue):
        if node.conversion not in (-1, 115) or node.format_spec is not None:
            return None
        return string(node.value, values)
    return _literal(node)


class _Bindings(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()
        self.outer: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_arg(self, node: ast.arg) -> None:
        self.names.add(node.arg)

    def visit_alias(self, node: ast.alias) -> None:
        self.names.add(node.asname or node.name.split(".")[0])

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)
        for expression in (*node.decorator_list, *node.args.defaults, *node.args.kw_defaults):
            if expression is not None:
                self.visit(expression)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for expression in (*node.args.defaults, *node.args.kw_defaults):
            if expression is not None:
                self.visit(expression)

    def visit_Global(self, node: ast.Global) -> None:
        self.outer.update(node.names)

    visit_Nonlocal = visit_Global

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.visit(node.iter)
        for condition in node.ifs:
            self.visit(condition)


def local_bindings(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> set[str]:
    visitor = _Bindings()
    if not isinstance(node, ast.ClassDef):
        visitor.visit(node.args)
    for statement in node.body:
        visitor.visit(statement)
    return visitor.names - visitor.outer


def _binding_table(policy: Policy) -> dict[tuple[str, str, str], Dynamic]:
    bindings = {(d.caller, d.scope, d.statement): d for d in policy.dynamics}
    if len(bindings) != len(policy.dynamics):
        raise ContractError("Duplicate dynamic import declarations are not permitted.")
    return bindings


def _expanded_ref(policy: Policy, ref: Import, refs: list[Import], known: dict[str, Module],
                  bindings: dict[tuple[str, str, str], Dynamic],
                  used: set[tuple[str, str, str]], valid_forwarder: bool) -> list[Import]:
    if forwarded(policy, ref, valid_forwarder):
        return [replace(ref, kind="forwarder")]
    special = _special(policy, ref, refs, known)
    if special is not None:
        return special
    binding = bindings.get(_key(ref)) if ref.kind == "dynamic" else None
    if binding is None:
        return [ref]
    used.add(_key(ref))
    return [replace(ref, target=n, kind="import")
            for n in _targets(binding, known, policy.package)]
