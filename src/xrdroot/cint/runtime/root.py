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
#: Namespaces whose names a macro uses unqualified - every RooFit tutorial says
#: ``using namespace RooFit``, TMVA's say ``using namespace TMVA`` - looked in when
#: ROOT itself has no such name. A macro without the ``using`` would not have
#: compiled, so looking there is never wrong. ``RooStats::HistFactory``'s functions
#: are called unqualified by argument-dependent lookup - ``MakeModelAndMeasurementFast(meas)``
#: of a ``HistFactory::Measurement`` - which finds them there too.
USED = ("RooFit", "RooStats", "RooStats.HistFactory", "TMVA", "TMVA.Experimental")


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
                return self._used(namespace, name)
        return importlib.import_module(f"{DEFAULT}.stl").std

    @staticmethod
    def _used(namespace: Any, name: str) -> Any:
        """``name`` from a namespace of :data:`USED`, or the refusal ROOT's own lookup gave."""
        for used in USED:
            found = getattr(_member(namespace, used), name, None)
            if found is not None:
                return found
        return getattr(namespace, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """``gErrorIgnoreLevel = kWarning``: a macro assigning one of ROOT's globals."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        setattr(self.namespace(), name, value)

    def __repr__(self) -> str:
        return "<ROOT, the names a translated macro did not declare>"


def _member(namespace: Any, dotted: str) -> Any:
    """``TMVA.Experimental`` of a namespace: each name in turn, or ``None`` where one is missing."""
    for name in dotted.split("."):
        namespace = getattr(namespace, name, None)
    return namespace


ROOT = RootProxy()
