"""Read the facade's literal export origins without evaluating its initializer."""

from __future__ import annotations

import ast

from .model import ContractError


def _module_names(node: ast.DictComp) -> list[str]:
    value = ast.literal_eval(_origin_dictionary(node))
    if not all(isinstance(n, str) for n in value):
        raise ContractError("Facade export module names must be literal strings.")
    return list(value)


def _origin_dictionary(node: ast.DictComp) -> ast.Dict:
    if len(node.generators) != 2 or not isinstance(node.value, ast.Tuple):
        raise ContractError("The facade export table needs a reviewed literal form.")
    first = node.generators[0]
    if not isinstance(first.iter, ast.Call) or not isinstance(first.iter.func, ast.Attribute):
        raise ContractError("The facade export table must iterate a literal module dictionary.")
    call = first.iter.func
    if call.attr != "items" or not isinstance(call.value, ast.Dict):
        raise ContractError("The facade export table must iterate literal dictionary items.")
    return call.value


def _updated_names(node: ast.Call) -> list[str]:
    if len(node.args) != 1 or not isinstance(node.args[0], ast.DictComp):
        raise ContractError("Facade export updates must use a literal module tuple.")
    expression = node.args[0]
    if len(expression.generators) != 1:
        raise ContractError("Facade export updates must have one generator.")
    value = ast.literal_eval(expression.generators[0].iter)
    if not isinstance(value, tuple) or not all(isinstance(n, str) for n in value):
        raise ContractError("Facade submodule origins must be a literal string tuple.")
    return list(value)


def _literal_submodule_names(node: ast.AST) -> list[str]:
    if not isinstance(node, (ast.Set, ast.Tuple)):
        raise ContractError("Facade submodules must use a finite literal set.")
    names = [entry.value for entry in node.elts if isinstance(entry, ast.Constant)]
    if len(names) != len(node.elts) or not all(
        isinstance(n, str) and n.isidentifier() for n in names
    ):
        raise ContractError("Facade submodule names must be literal top-level identifiers.")
    return names


def _submodule_value(node: ast.AST) -> ast.AST:
    if not isinstance(node, ast.Call):
        return node
    if not isinstance(node.func, ast.Name) or node.func.id != "frozenset":
        raise ContractError("Facade submodules must use a finite literal set.")
    if len(node.args) != 1 or node.keywords:
        raise ContractError("Facade submodules need one finite literal set.")
    return node.args[0]


def _submodules(node: ast.AST) -> list[str]:
    # The previous derived set is bounded by the independently parsed export table.
    legacy = ast.parse("frozenset(module for module, _ in _EXPORTS.values())", mode="eval").body
    if ast.dump(node) == ast.dump(legacy):
        return []
    return _literal_submodule_names(_submodule_value(node))


def _submodule_declaration(node: ast.AST) -> ast.AST | None:
    if isinstance(node, ast.Assign):
        names = node.targets
    elif isinstance(node, ast.AnnAssign):
        names = [node.target]
    else:
        return None
    if any(isinstance(name, ast.Name) and name.id == "_SUBMODULES" for name in names):
        return node.value
    return None


def _export_modules(node: ast.AST) -> list[str]:
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        if node.target.id == "_EXPORTS" and isinstance(node.value, ast.DictComp):
            return _module_names(node.value)
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        call = node.value
        if isinstance(call.func, ast.Attribute) and ast.unparse(call.func) == "_EXPORTS.update":
            return _updated_names(call)
    return []


def facade_targets(source: str, package: str) -> list[str]:
    result = []
    submodules = []
    for node in ast.parse(source).body:
        result.extend(_export_modules(node))
        submodule_value = _submodule_declaration(node)
        if submodule_value is not None:
            submodules.extend(_submodules(submodule_value))
    if not result:
        raise ContractError("The lazy facade dynamic imports need a literal export-origin table.")
    return sorted({package + "." + name for name in [*result, *submodules]})
