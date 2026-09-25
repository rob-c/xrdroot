"""``ROOT``: where a translated macro finds ROOT's names - ``ROOT.TH1F``, ``ROOT.gRandom``.

Every name a macro uses that it did not declare itself is ROOT's, and the
translation spells it ``ROOT.<name>``. ``ROOT`` is this proxy, which looks
the name up in :mod:`xrdroot.pyroot` the first time it is wanted - so a
translation can be imported, read and kept without pyroot loading - or in
whatever namespace :meth:`RootProxy.bind` was handed, which is how a test
runs a macro against a small fake ROOT of its own. ``ROOT.std`` is the
standard library's containers, from :mod:`xrdroot.pyroot.stl` when the
namespace bound does not have its own.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

__all__ = ["RootProxy", "ROOT"]

#: The module ROOT's names come from when nothing else was bound.
DEFAULT = "xrdroot.pyroot"


class RootProxy:
    """ROOT's namespace, looked up late: bound to a module or to any object with attributes."""

    def __init__(self) -> None:
        self._bound: list[Any] = []

    def namespace(self) -> Any:
        """What names are looked up in now: the innermost :meth:`bind`, else pyroot."""
        if self._bound:
            return self._bound[-1]
        return importlib.import_module(DEFAULT)

    @contextmanager
    def bind(self, namespace: Any) -> Iterator[Any]:
        """Look ROOT's names up in ``namespace`` until the ``with`` block ends."""
        self._bound.append(namespace)
        try:
            yield namespace
        finally:
            self._bound.pop()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        namespace = self.namespace()
        try:
            return getattr(namespace, name)
        except AttributeError:
            if name != "std":
                raise
        return importlib.import_module(f"{DEFAULT}.stl").std

    def __repr__(self) -> str:
        return "<ROOT, the names a translated macro did not declare>"


ROOT = RootProxy()
