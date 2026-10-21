"""Built-in extension lookup stays lazy and respects the current target object."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from xrdroot import plugins
from xrdroot.errors import UnsupportedFeatureError


def test_listing_and_missing_providers_do_not_import_their_modules(monkeypatch):
    def unexpected(name):
        raise AssertionError(f"Unexpected provider import: {name}")

    monkeypatch.setattr(plugins, "importlib", SimpleNamespace(import_module=unexpected))
    assert plugins.keys("display.plot") == ("default",)
    assert plugins.keys("unknown") == ()
    assert plugins.lookup("unknown") is None
    assert plugins.lookup("display.plot", "unknown") is None
    with pytest.raises(UnsupportedFeatureError, match="No provider is registered for 'unknown'"):
        plugins.require("unknown")


def test_a_provider_is_resolved_lazily_without_retaining_a_replaced_target(monkeypatch):
    calls = []
    first, second = object(), object()
    module = SimpleNamespace(draw=lambda: first)

    def loaded(name):
        calls.append(name)
        return module

    monkeypatch.setitem(
        plugins.PROVIDERS, ("drawing", "test"), ("graphics", "test.provider", "draw")
    )
    monkeypatch.setattr(plugins, "importlib", SimpleNamespace(import_module=loaded))
    assert plugins.keys("drawing") == ("test",)
    assert calls == []
    original = plugins.require("drawing", "test")
    assert original() is first
    module.draw = lambda: second
    assert plugins.lookup("drawing", "test")() is second
    assert original() is first
    assert calls == ["test.provider", "test.provider"]


def test_a_broken_declared_provider_keeps_its_import_error(monkeypatch):
    def missing(name):
        raise ImportError(f"Cannot import {name}")

    monkeypatch.setattr(plugins, "importlib", SimpleNamespace(import_module=missing))
    with pytest.raises(ImportError, match=r"Cannot import xrdroot.plot"):
        plugins.require("display.plot")
