"""Declare focused test support and the exact fixture cutoff used by isolation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model import Policy


@dataclass(frozen=True)
class TestScope:
    components: tuple[str, ...]
    paths: tuple[str, ...]
    support: tuple[str, ...]
    confcutdir: str
    blocked: tuple[str, ...]


@dataclass(frozen=True)
class Selection:
    paths: tuple[str, ...] = ()
    confcutdir: str | None = None
    blocked: tuple[str, ...] = ()


def parse_scopes(raw: dict[str, Any]) -> dict[str, TestScope]:
    from .model import _strings

    return {
        name: TestScope(
            _strings(v["components"], "test components"),
            _strings(v["paths"], "test paths"),
            _strings(v.get("support", []), "test support"),
            v["confcutdir"],
            _strings(v.get("blocked", []), "blocked imports"),
        )
        for name, v in raw.items()
    }


def _inside(root: Path, value: str) -> Path:
    from .model import ContractError

    path = root / value
    if not path.resolve().is_relative_to(root.resolve()):
        raise ContractError(f"Test support {value!r} escapes the project root.")
    if not path.exists():
        raise ContractError(f"Declared test support {value!r} does not exist.")
    return path


def _files(policy: Policy, name: str, scope: TestScope) -> dict[str, Path]:
    from .model import ContractError

    cutoff = _inside(policy.root, scope.confcutdir)
    if not cutoff.is_dir() or not scope.paths or not scope.components:
        raise ContractError(f"Test scope {name} needs components, paths and a fixture directory.")
    files: set[Path] = set()
    for value in (*scope.paths, *scope.support):
        path = _inside(policy.root, value)
        if not path.resolve().is_relative_to(cutoff.resolve()):
            raise ContractError(f"Test path {value} lies outside its declared fixture cutoff.")
        files.update(path.rglob("*.py") if path.is_dir() else [path])
    return {
        "component_tests."
        + name
        + "."
        + p.relative_to(cutoff).with_suffix("").as_posix().replace("/", "."): p
        for p in files
    }


def _fixtures(policy: Policy, scope: TestScope, files: dict[str, Path]) -> None:
    from .model import ContractError

    expected = {
        p.relative_to(policy.root).as_posix() for p in files.values() if p.name == "conftest.py"
    }
    if not expected <= set(scope.support):
        raise ContractError("Every focused conftest must be listed as declared test support.")


def _permitted(policy: Policy, scope: TestScope, target: str) -> bool:
    if _blocked(scope, target):
        return False
    allowed = _closure(policy, scope.components)
    if target == policy.package:
        return True  # Only the same inert parent bootstrap permitted by runtime isolation.
    if target.startswith(policy.package + "."):
        return policy.owner(target) in allowed
    return target.split(".", 1)[0] in policy.stdlib | _externals(policy, allowed) | {"pytest"}


def _check_source(policy: Policy, scope: TestScope, name: str, files: dict[str, Path]) -> None:
    from .imports import Module, imports, modules
    from .model import ContractError

    known = modules(policy)
    test_modules = {
        n: Module(
            n,
            p.relative_to(policy.root).as_posix(),
            n.rpartition(".")[0],
            p.read_bytes().decode("utf-8"),
        )
        for n, p in files.items()
    }
    for module in test_modules.values():
        for ref in imports(module, {**known, **test_modules}, policy):
            if ref.target in test_modules:
                continue
            target = ref.point if ref.kind == "registration" else ref.target
            if ref.kind not in {"import", "registration"} or not _permitted(policy, scope, target):
                raise ContractError(
                    f"Test scope {name}: {ref.path}:{ref.line} imports "
                    f"{ref.target} outside its declared closure."
                )


def validate_scopes(policy: Policy) -> None:
    for name, scope in policy.test_scopes.items():
        for component in scope.components:
            policy.closure(component)
        files = _files(policy, name, scope)
        _fixtures(policy, scope, files)
        _check_source(policy, scope, name, files)


def select(policy: Policy, component: str, names: list[str]) -> Selection:
    if not names:
        return Selection()
    scopes = _selected_scopes(policy, component, names)
    cutoff = _cutoff(scopes)
    return Selection(_selected_paths(scopes, "paths"), cutoff, _selected_paths(scopes, "blocked"))


def _blocked(scope: TestScope, target: str) -> bool:
    return any(target == p or target.startswith(p + ".") for p in scope.blocked)


def _closure(policy: Policy, components: tuple[str, ...]) -> set[str]:
    return set().union(*(policy.closure(name) for name in components))


def _externals(policy: Policy, allowed: set[str]) -> set[str]:
    return {name for component in allowed
            for name in policy.components[component].hard + policy.components[component].optional}


def _selected_scopes(policy: Policy, component: str, names: list[str]) -> list[TestScope]:
    from .model import ContractError

    if any(name not in policy.test_scopes for name in names):
        raise ContractError("Isolation --tests accepts declared scope names, not arbitrary paths.")
    scopes = [policy.test_scopes[name] for name in names]
    if any(not set(scope.components) <= policy.closure(component) for scope in scopes):
        raise ContractError(
            "The selected test scope needs components outside this isolation boundary."
        )
    return scopes


def _cutoff(scopes: list[TestScope]) -> str:
    from .model import ContractError

    cutoffs = {scope.confcutdir for scope in scopes}
    if len(cutoffs) != 1:
        raise ContractError(
            "Run scopes with different fixture cutoffs in separate fresh processes."
        )
    return cutoffs.pop()


def _selected_paths(scopes: list[TestScope], field: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(path for scope in scopes for path in getattr(scope, field)))
