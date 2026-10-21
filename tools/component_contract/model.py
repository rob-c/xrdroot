"""Describe the internal graph without importing any xrdroot implementation."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .modulealiases import ModuleAlias, parse_aliases, validate_aliases
from .privateimports import PrivateImport, parse_private
from .registrations import (
    Registration,
    RegistrationFunction,
    parse_functions,
    parse_registrations,
)
from .testscope import TestScope, parse_scopes, validate_scopes


class ContractError(ValueError):
    """A malformed policy cannot silently weaken the component boundary."""


@dataclass(frozen=True)
class Component:
    modules: tuple[str, ...]
    exact: tuple[str, ...]
    depends: tuple[str, ...]
    hard: tuple[str, ...]
    optional: tuple[str, ...]


@dataclass(frozen=True)
class Extension:
    caller: str
    point: str
    provider: str
    target: str


@dataclass(frozen=True)
class Dynamic:
    caller: str
    scope: str
    statement: str
    targets: tuple[str, ...]
    prefixes: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class Policy:
    root: Path
    package: str
    source: str
    baseline: str
    components: dict[str, Component]
    extensions: tuple[Extension, ...]
    provider_functions: tuple[str, ...]
    provider_registry: str
    stdlib: frozenset[str]
    stdlib_extra: tuple[str, ...]
    dynamics: tuple[Dynamic, ...]
    test_scopes: dict[str, TestScope]
    module_aliases: dict[str, ModuleAlias]
    private_imports: tuple[PrivateImport, ...]
    registration_functions: dict[str, RegistrationFunction]
    registrations: tuple[Registration, ...]

    def owner(self, module: str) -> str:
        choices = self._owners(module)
        if not choices:
            raise ContractError(f"No component owns {module}.")
        return max(choices)[1]

    def _owners(self, module: str) -> list[tuple[int, str]]:
        choices = []
        for name, item in self.components.items():
            prefixes = [p for p in item.modules if module == p or module.startswith(p + ".")]
            exact = [p for p in item.exact if module == p]
            choices.extend((len(p), name) for p in prefixes + exact)
        return choices

    def closure(self, name: str) -> frozenset[str]:
        if name not in self.components:
            raise ContractError(f"Unknown component {name!r}.")
        result = {name}
        pending = [name]
        while pending:
            for target in self.components[pending.pop()].depends:
                if target not in result:
                    result.add(target)
                    pending.append(target)
        return frozenset(result)

    def fingerprint(self) -> str:
        value = {
            "package": self.package,
            "source": self.source,
            "components": _record_map(self.components),
            "extensions": _record_list(self.extensions),
            "provider_functions": self.provider_functions,
            "provider_registry": self.provider_registry,
            "stdlib": self.stdlib_extra,
            "dynamics": _record_list(self.dynamics),
            "tests": _record_map(self.test_scopes),
            "module_aliases": _record_map(self.module_aliases),
            "private_imports": _record_list(self.private_imports),
            "registration_functions": _record_map(self.registration_functions),
            "registrations": _record_list(self.registrations),
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
        raise ContractError(f"{label} must be a list of nonempty strings.")
    if len(value) != len(set(value)):
        raise ContractError(f"{label} contains duplicates.")
    return tuple(value)


def _component(raw: dict[str, Any], groups: dict[str, Any]) -> Component:
    depends = []
    for entry in _strings(raw.get("depends", []), "depends"):
        depends.extend(_strings(groups[entry[1:]], entry) if entry.startswith("@") else [entry])
    return Component(
        _strings(raw.get("modules", []), "modules"),
        _strings(raw.get("exact", []), "exact"),
        tuple(dict.fromkeys(depends)),
        _strings(raw.get("hard", []), "hard"),
        _strings(raw.get("optional", []), "optional"),
    )


def _check_selectors(policy: Policy) -> None:
    seen: dict[str, str] = {}
    for name, item in policy.components.items():
        for selector in item.modules + item.exact:
            if selector in seen:
                raise ContractError(f"Ambiguous ownership of {selector}: {seen[selector]}, {name}.")
            if selector != policy.package and not selector.startswith(policy.package + "."):
                raise ContractError(f"Selector {selector} is outside {policy.package}.")
            seen[selector] = name


def _visit(policy: Policy, name: str, active: set[str], done: set[str]) -> None:
    if name in active:
        raise ContractError(f"The component graph contains a cycle at {name}.")
    if name in done:
        return
    if name not in policy.components:
        raise ContractError(f"Unknown dependency {name!r}.")
    active.add(name)
    for target in policy.components[name].depends:
        _visit(policy, target, active, done)
    active.remove(name)
    done.add(name)


def _check_extensions(policy: Policy) -> None:
    for edge in policy.extensions:
        _check_extension(policy, edge)
    if len(policy.extensions) != len(set(policy.extensions)):
        raise ContractError("Duplicate extension edges are not permitted.")


def _check_extension(policy: Policy, edge: Extension) -> None:
    target = edge.target.split(":")
    if len(target) != 2 or not all(p.isidentifier() for p in target[1].split(".")):
        raise ContractError(f"Extension target {edge.target!r} must name module:callable.")
    if policy.owner(target[0]) != edge.provider:
        raise ContractError(f"Extension target {edge.target} has the wrong provider.")
    if policy.owner(edge.caller) == "cppstd" and edge.provider != "cppstd":
        raise ContractError("cppstd cannot dispatch to a higher component.")


def policy_from(root: Path, raw: dict[str, Any]) -> Policy:
    if raw.get("schema") != 1:
        raise ContractError("The component policy must use schema 1.")
    groups = raw.get("groups", {})
    components = {n: _component(v, groups) for n, v in raw["ownership"].items()}
    extensions = tuple(Extension(**e) for e in raw.get("extensions", []))
    extra = _strings(raw.get("stdlib", []), "stdlib")
    stdlib = frozenset(sys.stdlib_module_names) | frozenset(extra)
    dynamics = tuple(_dynamic(d) for d in raw.get("dynamic", []))
    result = Policy(
        root,
        raw["package"],
        raw["source"],
        raw["baseline"],
        components,
        extensions,
        tuple(raw.get("provider_functions", [])),
        raw.get("provider_registry", "xrdroot.plugins"),
        stdlib,
        extra,
        dynamics,
        parse_scopes(raw.get("tests", {})),
        parse_aliases(raw.get("module_aliases", {})),
        parse_private(raw.get("private_imports", [])),
        parse_functions(raw.get("registration_functions", {})),
        parse_registrations(raw.get("registrations", [])),
    )
    _check_selectors(result)
    for name in components:
        _visit(result, name, set(), set())
    _check_extensions(result)
    validate_aliases(result)
    validate_scopes(result)
    return result


def _dynamic(raw: dict[str, Any]) -> Dynamic:
    result = Dynamic(
        raw["caller"],
        raw["scope"],
        raw["statement"],
        _strings(raw.get("targets", []), "dynamic targets"),
        _strings(raw.get("prefixes", []), "dynamic prefixes"),
        raw["reason"],
    )
    if not (result.targets or result.prefixes) or not result.reason.strip():
        raise ContractError("Dynamic imports need finite targets and a source-review reason.")
    return result


def load_policy(root: Path) -> Policy:
    """Use tomllib, or tomli supplied by the existing dev pytest dependency on 3.10."""
    parser = "tomllib" if sys.version_info >= (3, 11) else "tomli"
    try:
        module = importlib.import_module(parser)
    except ModuleNotFoundError as error:
        raise ContractError("Component tooling on Python 3.10 needs pip install tomli.") from error
    raw = module.loads((root / "pyproject.toml").read_text())
    return policy_from(root, raw["tool"]["xrdroot"]["components"])


def _record_map(values: dict[str, Any]) -> dict[str, Any]:
    return {name: asdict(value) for name, value in values.items()}


def _record_list(values: tuple[Any, ...]) -> list[Any]:
    return [asdict(value) for value in values]
