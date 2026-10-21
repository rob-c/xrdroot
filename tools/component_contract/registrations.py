"""Keep callable injection finite, source-owned and distinct from ordinary graph edges."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .imports import Import, Module
    from .model import Policy


@dataclass(frozen=True)
class RegistrationFunction:
    position: int
    keyword: str


@dataclass(frozen=True)
class Registration:
    caller: str
    scope: str
    statement: str
    callee: str
    callback: str
    reason: str


def parse_functions(raw: dict[str, Any]) -> dict[str, RegistrationFunction]:
    from .model import ContractError

    result = {name: RegistrationFunction(**value) for name, value in raw.items()}
    for name, value in result.items():
        if (
            type(value.position) is not int
            or value.position < 0
            or not isinstance(value.keyword, str)
            or not value.keyword.isidentifier()
        ):
            raise ContractError(f"Registration function {name} needs one exact callback argument.")
    return result


def parse_registrations(raw: list[dict[str, Any]]) -> tuple[Registration, ...]:
    from .model import ContractError

    result = tuple(Registration(**entry) for entry in raw)
    for entry in result:
        _validate_registration(entry)
    keys = {(e.caller, e.scope, e.statement, e.callee, e.callback) for e in result}
    if len(result) != len(keys):
        raise ContractError("Duplicate registration declarations are not permitted.")
    return result


def callable_target(qualified: str, known: dict[str, Module]) -> str:
    modules = [name for name in known if qualified.startswith(name + ".")]
    if not modules:
        return "<dynamic>"
    name = max(modules, key=len)
    return name + ":" + qualified[len(name) + 1 :]


def _definitions(body: list[ast.stmt], prefix: str = "") -> dict[str, ast.AST]:
    result = {}
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = prefix + node.name
            result[name] = node
            result.update(_definitions(node.body, name + "."))
    return result


def _concrete(target: str, known: dict[str, Module]) -> str:
    from .model import ContractError

    module, separator, attribute = target.partition(":")
    if not separator or module not in known:
        raise ContractError(f"Registration target {target} has no concrete source module.")
    if attribute not in _definitions(ast.parse(known[module].source).body):
        raise ContractError(f"Registration target {target} has no source callable definition.")
    return module


def validate(policy: Policy, known: dict[str, Module], refs: list[Import]) -> None:
    for callee, point in policy.registration_functions.items():
        _validate_function(policy, known, callee, point)
    for entry in policy.registrations:
        _validate_entry(policy, known, refs, entry)


def _validate_function(
    policy: Policy, known: dict[str, Module], callee: str, point: RegistrationFunction
) -> None:
    from .model import ContractError

    target = callable_target(callee, known)
    module = _concrete(target, known)
    if policy.owner(module) != "cppstd":
        raise ContractError("Generic registration functions must belong to cppstd.")
    node = _definitions(ast.parse(known[module].source).body)[target.partition(":")[2]]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ContractError("A registration function must have an explicit source signature.")
    parameters = [*node.args.posonlyargs, *node.args.args]
    if point.position >= len(parameters) or parameters[point.position].arg != point.keyword:
        raise ContractError(f"Registration function {callee} has a different callback signature.")


def _validate_entry(
    policy: Policy, known: dict[str, Module], refs: list[Import], entry: Registration
) -> None:
    from .model import ContractError

    if entry.callee not in policy.registration_functions or entry.caller not in known:
        raise ContractError("Registration declarations need a finite function and a source caller.")
    callback_module = _concrete(entry.callback, known)
    if policy.owner(callback_module) not in policy.closure(policy.owner(entry.caller)):
        raise ContractError("A registered callback must belong to its caller's downward closure.")
    matches = [r for r in refs if matches_entry(r, entry)]
    if len(matches) != 1:
        raise ContractError(
            f"Registration {entry.caller}:{entry.scope} must match exactly one "
            "current call; shrink stale declarations."
        )


def matches_entry(ref: Import, entry: Registration) -> bool:
    return (ref.kind, ref.module, ref.scope, ref.statement, ref.point, ref.target) == (
        "registration",
        entry.caller,
        entry.scope,
        entry.statement,
        entry.callee,
        entry.callback,
    )


def rules(policy: Policy, ref: Import) -> list[str]:
    return (
        []
        if any(matches_entry(ref, entry) for entry in policy.registrations)
        else ["registration-undeclared"]
    )


def _validate_registration(entry: Registration) -> None:
    from .model import ContractError

    if (
        not all(isinstance(value, str) for value in vars(entry).values())
        or not entry.reason.strip()
    ):
        raise ContractError(
            "Registrations need exact string declarations and a source-review reason."
        )
    try:
        parsed = ast.parse(entry.statement, mode="eval").body
    except SyntaxError as error:
        raise ContractError("Registration declarations need valid Python calls.") from error
    if not isinstance(parsed, ast.Call) or ast.unparse(parsed) != entry.statement:
        raise ContractError("Registration declarations need one canonical AST call.")
