"""Recognize one reviewed registry wrapper without granting arbitrary point forwarding."""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from .model import ContractError

if TYPE_CHECKING:
    from .imports import Import, Module
    from .model import Policy


_REQUIRE = """
found = lookup(point, key)
if found is None:
    raise UnsupportedFeatureError(f"No provider is registered for {point!r} with key {key!r}.")
return found
"""


def _signature(node: ast.FunctionDef) -> bool:
    expected = ast.parse("def forward(point, key='default'): pass").body[0]
    assert isinstance(expected, ast.FunctionDef)
    args = node.args
    plain = ast.arguments(
        posonlyargs=args.posonlyargs,
        args=[ast.arg(arg=arg.arg) for arg in args.args],
        vararg=args.vararg,
        kwonlyargs=args.kwonlyargs,
        kw_defaults=args.kw_defaults,
        kwarg=args.kwarg,
        defaults=args.defaults,
    )
    return not node.decorator_list and ast.dump(plain) == ast.dump(expected.args)


def _body(node: ast.FunctionDef) -> str:
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1 or not _signature(found[0]):
        raise ContractError("Registry forwarding needs one declared lookup/require signature.")
    return found[0]


class _ClassGlobals(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Global(self, node: ast.Global) -> None:
        self.names.update(node.names)

    def visit_FunctionDef(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> None:
        pass  # Each nested definition has its own global declarations.

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef


class _ModuleBindings(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: Counter[str] = Counter()
        self.wildcard = False
        self.module_scope = True
        self.globals: set[str] = set()

    def bound(self, name: str | None) -> None:
        if name in {"lookup", "require"} and (self.module_scope or name in self.globals):
            self.names[str(name)] += 1

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound(node.id)

    def visit_Import(self, node: ast.Import | ast.ImportFrom) -> None:
        for name in node.names:
            if name.name == "*" and self.module_scope:
                self.wildcard = True
            self.bound(name.asname or name.name.split(".")[0])

    visit_ImportFrom = visit_Import

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.bound(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit(node.args)  # Defaults are eager; the lambda body is deferred.

    @contextmanager
    def class_scope(self, node: ast.ClassDef) -> Iterator[None]:
        saved = self.module_scope, self.globals
        declared = _ClassGlobals()
        declared.visit(ast.Module(body=node.body, type_ignores=[]))
        self.module_scope, self.globals = False, declared.names
        try:
            yield
        finally:
            self.module_scope, self.globals = saved

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound(node.name)
        for expression in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(expression)
        with self.class_scope(node):
            for statement in node.body:
                self.visit(statement)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.bound(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs | ast.MatchStar) -> None:
        self.bound(node.name)
        self.generic_visit(node)

    visit_MatchStar = visit_MatchAs

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        self.bound(node.rest)
        self.generic_visit(node)

    def _comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        # Iteration targets belong to the comprehension; walrus stores do not.
        for generator in node.generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        values = (node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,)
        for value in values:
            self.visit(value)

    visit_ListComp = _comprehension
    visit_SetComp = _comprehension
    visit_DictComp = _comprehension
    visit_GeneratorExp = _comprehension


def _no_rebinding(tree: ast.Module) -> bool:
    bindings = _ModuleBindings()
    bindings.visit(tree)
    return not bindings.wildcard and bindings.names == {"lookup": 1, "require": 1}


def validated(policy: Policy, known: dict[str, Module]) -> bool:
    tree = _required_registry(policy, known)
    if tree is None:
        return False
    if policy.provider_registry + ".lookup" not in policy.provider_functions:
        raise ContractError("Registry require must forward to the declared lookup function.")
    require = _function(tree, "require")
    _function(tree, "lookup")
    expected = ast.dump(ast.parse(_REQUIRE))
    if _body(require) != expected or not _no_rebinding(tree):
        raise ContractError(
            "Registry require must preserve the reviewed point/key forwarding body."
        )
    return True


def forwarded(policy: Policy, ref: Import, valid: bool) -> bool:
    return valid and (
        ref.kind,
        ref.module,
        ref.scope,
        ref.statement,
        ref.provider,
        ref.point,
        ref.target,
    ) == (
        "provider",
        policy.provider_registry,
        "require",
        "lookup(point, key)",
        policy.provider_registry + ".lookup",
        "<dynamic>",
        "*",
    )


def _required_registry(policy: Policy, known: dict[str, Module]) -> ast.Module | None:
    source = known.get(policy.provider_registry)
    if source is None or policy.provider_registry + ".require" not in policy.provider_functions:
        return None
    tree = ast.parse(source.source)
    if not any(isinstance(node, ast.FunctionDef) and node.name == "require" for node in tree.body):
        return None
    return tree
