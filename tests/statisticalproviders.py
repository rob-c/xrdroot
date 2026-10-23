"""Downward providers exercise statistical engines without loading their ROOT adapters."""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

from xrdroot import plugins


def install_provider(
    monkeypatch: pytest.MonkeyPatch,
    point: str,
    callback: Callable[..., Any],
    key: str = "default",
) -> None:
    """Keep lookup real while replacing one reviewed target with a test's plain callable."""
    name = f"_statistical_{point.replace('.', '_')}_{key}"
    module = ModuleType(name)
    module.provide = callback
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(plugins.PROVIDERS, (point, key), ("core", name, "provide"))
