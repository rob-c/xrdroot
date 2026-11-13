"""Find every import that leaves its component by an undeclared edge."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .dynamic import expand
from .imports import Import, Module, imports, modules
from .model import ContractError, Extension, Policy
from .privateimports import reviewed
from .registrations import rules as registration_rules
from .registrations import validate as validate_registrations


@dataclass(frozen=True, order=True)
class Violation:
    path: str
    line: int
    component: str
    target: str
    rule: str
    scope: str
    statement: str



@dataclass(frozen=True)
class Report:
    modules: dict[str, Module]
    violations: tuple[Violation, ...]
    import_count: int
    reviewed_private: tuple[Violation, ...] = ()
    registrations: tuple[Import, ...] = ()
    forwarders: tuple[Import, ...] = ()



def _external(policy: Policy, ref: Import, component: str) -> list[str]:
    root = ref.target.split(".", 1)[0]
    if root in policy.stdlib or root == "__future__":
        return []
    item = policy.components[component]
    rules = []
    if root not in item.hard + item.optional:
        rules.append("external-undeclared")
    if root in item.optional and not ref.lazy:
        rules.append("optional-eager")
    if any(p.startswith("_") for p in ref.target.split(".")[1:]):
        rules.append("external-private")
    return rules


def _declaration(policy: Policy, ref: Import, known: dict[str, Module]) -> list[str]:
    if ":" not in ref.target:
        return ["provider-dynamic"]
    target = ref.target.split(":", 1)[0]
    if target not in known:
        return ["provider-missing"]
    if ref.provider != policy.owner(target):
        return ["provider-owner"]
    allowed = any(
        e.point == ref.point and e.provider == ref.provider and e.target == ref.target
        for e in policy.extensions
    )
    return [] if allowed else ["provider-undeclared"]


def _provider(policy: Policy, ref: Import, declarations: list[Import]) -> list[str]:
    targets = [
        r
        for r in declarations
        if r.point == ref.point and (ref.target == "*" or r.key == ref.target)
    ]
    if not targets or ref.point == "<dynamic>":
        return ["provider-point-undeclared"]
    allowed = all(
        Extension(ref.module, r.point, r.provider, r.target) in policy.extensions for r in targets
    )
    return [] if allowed else ["provider-edge"]


def _rules(
    policy: Policy, ref: Import, known: dict[str, Module], declarations: list[Import]
) -> list[str]:
    component = policy.owner(ref.module)
    if ref.kind == "forwarder":
        return []
    if ref.kind == "registration":
        return registration_rules(policy, ref)
    if ref.kind == "provider":
        return _provider(policy, ref, declarations)
    if ref.kind in {"declaration", "dispatch"}:
        return _declaration(policy, ref, known)
    if ref.kind == "dynamic":
        return ["dynamic-import"]
    return _import_rules(policy, ref, known, component)


def inspect(policy: Policy) -> Report:
    known = modules(policy)
    _extension_callers(policy, known)
    refs = _references(policy, known)
    expanded = expand(policy, refs, known)
    validate_registrations(policy, known, expanded)
    declarations = _of_kind(expanded, "declaration")
    found = _violations(policy, expanded, known, declarations)
    violations, approved = reviewed(policy, found)
    return Report(
        known, tuple(sorted(violations)), len(refs), approved,
        tuple(_of_kind(expanded, "registration")), tuple(_of_kind(expanded, "forwarder")),
    )


def graph(policy: Policy, report: Report, dot: bool = False) -> str:
    counts = Counter(policy.owner(n) for n in report.modules)
    if dot:
        rows = ["digraph components {"]
        rows.extend(f'  "{name}" [label="{name} ({counts[name]})"];' for name in policy.components)
        rows.extend(
            f'  "{name}" -> "{target}";'
            for name, item in policy.components.items()
            for target in item.depends
        )
        return "\n".join([*rows, "}"])
    return "\n".join(
        f"{name}: {counts[name]} modules; depends on {', '.join(item.depends) or 'none'}"
        for name, item in policy.components.items()
    )


def _import_rules(
    policy: Policy, ref: Import, known: dict[str, Module], component: str
) -> list[str]:
    if ref.target != policy.package and not ref.target.startswith(policy.package + "."):
        return _external(policy, ref, component)
    if ref.target not in known:
        return ["internal-missing"]
    owner = policy.owner(ref.target)
    return [] if owner == component or owner in policy.components[component].depends else ["edge"]


def _extension_callers(policy: Policy, known: dict[str, Module]) -> None:
    for edge in policy.extensions:
        if edge.caller not in known:
            raise ContractError(f"Extension caller {edge.caller} has no source module.")


def _references(policy: Policy, known: dict[str, Module]) -> list[Import]:
    return [ref for module in known.values() for ref in imports(module, known, policy)]


def _of_kind(refs: list[Import], kind: str) -> list[Import]:
    return [ref for ref in refs if ref.kind == kind]


def _violations(policy: Policy, refs: list[Import], known: dict[str, Module],
                declarations: list[Import]) -> list[Violation]:
    found = []
    for ref in refs:
        found.extend(
            Violation(ref.path, ref.line, policy.owner(ref.module), ref.target,
                      rule, ref.scope, ref.statement)
            for rule in _rules(policy, ref, known, declarations)
        )
    return found
