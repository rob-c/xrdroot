"""Retain exact reviewed private imports visibly without weakening their dependencies."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .analysis import Violation
    from .model import Policy


@dataclass(frozen=True)
class PrivateImport:
    caller: str
    scope: str
    statement: str
    target: str
    reason: str


def parse_private(raw: list[dict[str, Any]]) -> tuple[PrivateImport, ...]:
    from .model import ContractError

    result = tuple(PrivateImport(**v) for v in raw)
    if len(result) != len(set(result)):
        raise ContractError("Duplicate reviewed private imports are not permitted.")
    for entry in result:
        _validate(entry)
    return result


def _validate(entry: PrivateImport) -> None:
    _validate_fields(entry)
    _validate_statement(entry.statement)


def _matches(policy: Policy, item: Violation, entry: PrivateImport) -> bool:
    from .imports import modules

    source = modules(policy).get(entry.caller)
    return source is not None and (
        item.path,
        item.scope,
        item.statement,
        item.target,
        item.rule,
    ) == (source.path, entry.scope, entry.statement, entry.target, "external-private")


def reviewed(
    policy: Policy, violations: list[Violation]
) -> tuple[list[Violation], tuple[Violation, ...]]:
    from .model import ContractError

    approved = []
    for entry in policy.private_imports:
        matches = [v for v in violations if _matches(policy, v, entry)]
        if len(matches) != 1 or matches[0] in approved:
            raise ContractError(
                f"Reviewed private import {entry.caller}:{entry.scope} must match "
                "exactly one current import; shrink stale declarations."
            )
        approved.extend(matches)
    return [v for v in violations if v not in approved], tuple(sorted(approved))


def _validate_fields(entry: PrivateImport) -> None:
    from .model import ContractError

    values = (entry.caller, entry.statement, entry.target, entry.reason)
    if not all(isinstance(v, str) and v.strip() for v in values):
        raise ContractError("Reviewed private imports need exact names and a source-review reason.")
    if (
        not isinstance(entry.scope, str)
        or not any(p.startswith("_") for p in entry.target.split(".")[1:])
    ):
        raise ContractError("A reviewed private target must name its concrete private module.")


def _validate_statement(statement: str) -> None:
    from .model import ContractError

    try:
        statements = ast.parse(statement).body
    except SyntaxError as error:
        raise ContractError("A reviewed private import must be valid Python source.") from error
    if len(statements) != 1 or not isinstance(statements[0], (ast.Import, ast.ImportFrom)):
        raise ContractError(
            "A reviewed private declaration must contain one exact import statement."
        )
    if ast.unparse(statements[0]) != statement:
        raise ContractError("Reviewed private import statements must use canonical AST spelling.")
