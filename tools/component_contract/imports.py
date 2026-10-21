"""Walk source imports, including function-local and explicit dynamic dependencies."""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from .dynamic import constants, local_bindings, string
from .model import ContractError, Policy
from .registrations import callable_target


@dataclass(frozen=True)
class Module:
    name: str
    path: str
    package: str
    source: str


@dataclass(frozen=True)
class Import:
    module: str
    path: str
    line: int
    scope: str
    target: str
    statement: str
    lazy: bool
    kind: str = "import"
    point: str = ""
    provider: str = ""
    key: str = ""


def modules(policy: Policy) -> dict[str, Module]:
    base = policy.root / policy.source
    result = {}
    for path in sorted(base.rglob("*.py")):
        relative = path.relative_to(base).with_suffix("")
        parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        name = ".".join((policy.package, *parts))
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        result[name] = Module(
            name,
            path.relative_to(policy.root).as_posix(),
            package,
            path.read_bytes().decode("utf-8"),
        )
        policy.owner(name)
    if policy.package not in result:
        raise ContractError(f"The source root {base} has no package initializer.")
    return result


def _literal(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _guarded(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return False
    return any(
        isinstance(n, ast.Name) and n.id in {"ImportError", "ModuleNotFoundError"}
        for n in ast.walk(handler.type)
    )


def _provider_key(key: ast.AST | None) -> tuple[str | None, str | None]:
    if not isinstance(key, ast.Tuple) or len(key.elts) != 2:
        return None, None
    return _literal(key.elts[0]), _literal(key.elts[1])


def _provider_parts(entry: ast.AST) -> list[str | None]:
    if isinstance(entry, ast.Dict):
        values = {_literal(k): _literal(v) for k, v in zip(entry.keys, entry.values, strict=True)}
        return [values.get(n) for n in ("component", "module", "attribute")]
    if isinstance(entry, ast.Tuple) and len(entry.elts) == 3:
        return [_literal(p) for p in entry.elts]
    return []


class Imports(ast.NodeVisitor):
    def __init__(self, module: Module, known: dict[str, Module], policy: Policy):
        self.module, self.known, self.policy = module, known, policy
        self.aliases: dict[str, str] = {}
        self.scope: list[str] = []
        self.depth = 0
        self.guards = 0
        self.found: list[Import] = []
        self.constants = constants(module)
        self.masked: list[set[str]] = []
        self.lexical_aliases = self.aliases
        self.lexical_masks = self.masked

    def record(
        self,
        node: ast.AST,
        target: str,
        kind: str = "import",
        point: str = "",
        provider: str = "",
        key: str = "",
    ) -> None:
        self.found.append(
            Import(
                self.module.name,
                self.module.path,
                node.lineno,
                ".".join(self.scope),
                target,
                ast.unparse(node),
                self.depth > 0 or self.guards > 0,
                kind,
                point,
                provider,
                key,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            self.record(node, item.name)
            self.aliases[item.asname or item.name.split(".")[0]] = (
                item.name if item.asname else item.name.split(".")[0]
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = (
            importlib.util.resolve_name("." * node.level + (node.module or ""), self.module.package)
            if node.level
            else node.module or ""
        )
        for item in node.names:
            candidate = base + "." + item.name
            target = candidate if candidate in self.known else base
            self.record(node, target)
            self.aliases[item.asname or item.name] = candidate

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Defaults, annotations and decorator expressions run in the enclosing scope.
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        with self.enter(node.name, local_bindings(node), deferred=True):
            for statement in node.body:
                self.visit(statement)
        self.aliases[node.name] = self.module.name + "." + ".".join((*self.scope, node.name))

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(expression)
        with self.enter(node.name, local_bindings(node), deferred=False):
            for statement in node.body:
                self.visit(statement)
        self.aliases[node.name] = self.module.name + "." + ".".join((*self.scope, node.name))

    @contextmanager
    def enter(self, name: str, bindings: set[str], *, deferred: bool) -> Iterator[None]:
        saved = self.aliases, self.lexical_aliases, self.masked, self.lexical_masks, self.depth
        self.aliases = self.lexical_aliases.copy()
        self.aliases.update(dict.fromkeys(bindings, ""))
        self.masked = [*self.lexical_masks, bindings]
        self.scope.append(name)
        if deferred:
            self.depth += 1
            self.lexical_aliases, self.lexical_masks = self.aliases, self.masked
        try:
            yield
        finally:
            self.aliases, self.lexical_aliases, self.masked, self.lexical_masks, self.depth = saved
            self.scope.pop()

    def visit_Try(self, node: ast.Try) -> None:
        guarded = any(_guarded(h) for h in node.handlers)
        self.guards += int(guarded)
        for statement in node.body:
            self.visit(statement)
        self.guards -= int(guarded)
        for child in (*node.handlers, *node.orelse, *node.finalbody):
            self.visit(child)

    def visit_If(self, node: ast.If) -> None:
        name = self.qualified(node.test)
        guarded = name in {"typing.TYPE_CHECKING", "TYPE_CHECKING"}
        self.visit(node.test)
        self.guards += int(guarded)
        for statement in node.body:
            self.visit(statement)
        self.guards -= int(guarded)
        for statement in node.orelse:
            self.visit(statement)

    def qualified(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return self.qualified(node.value) + "." + node.attr
        return ""

    def visit_Call(self, node: ast.Call) -> None:
        name = self.qualified(node.func)
        if name in {"__import__", "importlib.import_module", "importlib.util.find_spec"}:
            self.dynamic(node)
        if name in self.policy.provider_functions:
            self.provider(node, name)
        if name in self.policy.registration_functions:
            self.registration(node, name)
        self.generic_visit(node)

    def argument(self, node: ast.Call, position: int, keyword: str) -> ast.AST | None:
        if len(node.args) > position:
            return node.args[position]
        return next((k.value for k in node.keywords if k.arg == keyword), None)

    def dynamic(self, node: ast.Call) -> None:
        visible = {n: v for n, v in self.constants.items() if not any(n in s for s in self.masked)}
        target = string(self.argument(node, 0, "name"), visible)
        if target is None:
            self.record(node, "<dynamic>", "dynamic")
            return
        if target.startswith("."):
            package = self.argument(node, 1, "package")
            value = _literal(package)
            if isinstance(package, ast.Name) and package.id == "__package__":
                value = self.module.package
            target = importlib.util.resolve_name(target, value or "")
        self.record(node, target)

    def registration(self, node: ast.Call, callee: str) -> None:
        point = self.policy.registration_functions[callee]
        argument = self.argument(node, point.position, point.keyword)
        qualified = self.qualified(argument) if argument is not None else ""
        self.record(node, callable_target(qualified, self.known), "registration", callee)

    def provider(self, node: ast.Call, callee: str) -> None:
        point = _literal(self.argument(node, 0, "point"))
        argument = self.argument(node, 1, "key")
        key = "default" if argument is None else _literal(argument)
        self.record(node, key or "*", "provider", point or "<dynamic>", callee)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.declaration(node, node.targets, node.value)
        self.generic_visit(node)
        self.assign_aliases(node.targets, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.declaration(node, [node.target], node.value)
        self.generic_visit(node)
        self.assign_aliases([node.target], node.value)

    def assign_aliases(self, targets: list[ast.expr], value: ast.AST | None) -> None:
        qualified = self.qualified(value) if value is not None else ""
        for target in targets:
            if isinstance(target, ast.Name):
                self.aliases[target.id] = qualified

    def declaration(self, node: ast.AST, targets: list[ast.expr], value: ast.AST | None) -> None:
        if self.module.name != self.policy.provider_registry:
            return
        if not any(isinstance(t, ast.Name) and t.id == "PROVIDERS" for t in targets):
            return
        if not isinstance(value, ast.Dict):
            self.record(node, "<dynamic>", "declaration")
            return
        for key, entry in zip(value.keys, value.values, strict=True):
            self.provider_entry(node, key, entry)

    def provider_entry(self, node: ast.AST, key: ast.AST | None, entry: ast.AST) -> None:
        point, name = _provider_key(key)
        parts = _provider_parts(entry)
        if len(parts) != 3 or not all(parts) or point is None or name is None:
            self.record(node, "<dynamic>", "declaration")
            return
        self.record(node, f"{parts[1]}:{parts[2]}", "declaration", point, parts[0] or "", name)


def imports(module: Module, known: dict[str, Module], policy: Policy) -> list[Import]:
    try:
        visitor = Imports(module, known, policy)
        visitor.visit(ast.parse(module.source, filename=module.path))
    except (SyntaxError, ImportError, ValueError) as error:
        raise ContractError(f"Cannot inspect {module.path}: {error}") from error
    return visitor.found
