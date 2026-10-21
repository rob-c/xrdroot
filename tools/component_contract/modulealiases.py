"""Allow only reviewed downward compatibility names for one actual source module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model import Policy


@dataclass(frozen=True)
class ModuleAlias:
    target: str
    reason: str


def parse_aliases(raw: dict[str, Any]) -> dict[str, ModuleAlias]:
    from .model import ContractError

    result = {}
    for name, value in raw.items():
        if set(value) != {"target", "reason"} or not isinstance(value["reason"], str):
            raise ContractError(f"Module alias {name} needs an exact target and review reason.")
        if not isinstance(value["target"], str) or not value["reason"].strip():
            raise ContractError(f"Module alias {name} needs an exact target and review reason.")
        result[name] = ModuleAlias(**value)
    return result


def validate_aliases(policy: Policy) -> None:
    from .imports import modules
    from .model import ContractError

    if not policy.module_aliases:
        return
    known = modules(policy)
    for name, alias in policy.module_aliases.items():
        if name == policy.package or name not in known or alias.target not in known:
            raise ContractError(f"Module alias {name} must name two concrete source modules.")
        if alias.target == name or alias.target in policy.module_aliases:
            raise ContractError(f"Module alias {name} cannot chain or refer to itself.")
        if policy.owner(alias.target) not in policy.closure(policy.owner(name)):
            raise ContractError(
                f"Module alias {name} points outside its downward component closure."
            )


def verify_origin(module: Any, source: Path, name: str) -> None:
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or Path(origin).resolve() != source.resolve():
        raise ImportError(f"Component isolation loaded {name} from another source tree.")


def verify_alias(policy: Policy, name: str, module: Any, loaded: dict[str, Any]) -> None:
    from .imports import modules

    target = policy.module_aliases[name].target
    if module is not loaded.get(target):
        raise ImportError(f"Declared module alias {name} is not the loaded {target} module.")
    verify_origin(module, policy.root / modules(policy)[target].path, name)
