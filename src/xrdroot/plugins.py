"""Declared built-in extension targets, resolved only when a caller needs them.

The table records deliberate dispatch across component boundaries. Resolving a
target does not discover packages or retain objects whose module may be replaced.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast

from .errors import UnsupportedFeatureError

__all__ = ["PROVIDERS", "keys", "lookup", "require"]

#: Extension point and key against the owning component, module and attribute.
PROVIDERS: dict[tuple[str, str], tuple[str, str, str]] = {
    ("display.plot", "default"): ("graphics", "xrdroot.plot", "plot"),
    ("display.html", "default"): ("graphics", "xrdroot.plot.notebook", "figure"),
}


def keys(point: str) -> tuple[str, ...]:
    """The declared keys of an extension point, without loading its providers."""
    return tuple(key for name, key in PROVIDERS if name == point)


def lookup(point: str, key: str = "default") -> Callable[..., Any] | None:
    """Resolve a declared target, leaving absent entries to the caller's fallback."""
    target = PROVIDERS.get((point, key))
    if target is None:
        return None
    _, module, attribute = target
    return cast(Callable[..., Any], getattr(importlib.import_module(module), attribute))


def require(point: str, key: str = "default") -> Callable[..., Any]:
    """Resolve a required target and name the missing registration if it is absent."""
    found = lookup(point, key)
    if found is None:
        raise UnsupportedFeatureError(f"No provider is registered for {point!r} with key {key!r}.")
    return found
