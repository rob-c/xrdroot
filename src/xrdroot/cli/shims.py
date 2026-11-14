"""``import ROOT`` and ``import cppyy`` meaning xrdroot's, for a script run as PyROOT runs it.

A PyROOT script imports ``ROOT`` and, now and then, PyROOT's own modules
beneath it - ``ROOT._pythonization._th1``, ``ROOT._distrdf.Proxy`` - or
``cppyy``, the binding PyROOT is built on. Putting :mod:`xrdroot.pyroot` in
``sys.modules`` as ``ROOT`` is not enough for the first: Python would load
``ROOT._distrdf`` a second time from the same file, under the other name,
and a class from one copy would not be the class from the other.
:class:`Aliases` makes every ``ROOT.x.y`` the very module
``xrdroot.pyroot.x.y`` is.

``cppyy`` is aliased to :mod:`xrdroot.cppyy` only when no real cppyy is
installed: its finder is the last one Python asks, so an installed cppyy is
found first and wins. Nothing is imported until a script asks for ``cppyy``,
as PyROOT itself never imports it unasked.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from collections.abc import Sequence
from types import ModuleType
from typing import Any

__all__ = ["NAMESPACES", "Aliases", "install_root", "install_cppyy", "python_warnings"]

#: ROOT's namespaces a script imports names from, ``from ROOT.<namespace> import ...``.
NAMESPACES = ("Experimental", "VecOps", "RDF", "Math", "RooFit", "TMVA")


class Aliases(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """A finder that imports ``alias.rest`` as the module ``target.rest`` already is.

    The module is imported under its own name and the alias made to point at
    it, so both names are one module object.
    """

    def __init__(self, alias: str, target: str) -> None:
        self.alias = alias
        self.target = target
        #: Each module's own spec, which Python overwrites with the alias's as it loads it.
        self._specs: dict[str, Any] = {}

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != self.alias and not fullname.startswith(self.alias + "."):
            return None
        name = self.target + fullname[len(self.alias) :]
        try:
            found = importlib.util.find_spec(name)
        except ModuleNotFoundError:  # a parent that is a namespace object, not a module
            return _member(fullname)
        if found is None:
            return None
        return importlib.util.spec_from_loader(fullname, self, is_package=found.parent == name)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = importlib.import_module(self.target + spec.name[len(self.alias) :])
        self._specs[module.__name__] = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        """The module ran when it was imported under its own name; it keeps that name's spec,
        so that its relative imports still resolve from where it is."""
        module.__spec__ = self._specs.pop(module.__name__)


class _Member(importlib.abc.Loader):
    """Loads ``ROOT.RDF.Distributed``: the module a namespace object holds by that name."""

    def __init__(self, module: ModuleType) -> None:
        self.module = module
        self.spec = module.__spec__

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        return self.module

    def exec_module(self, module: ModuleType) -> None:
        module.__spec__ = self.spec


def _member(fullname: str) -> importlib.machinery.ModuleSpec | None:
    """The spec of a module a namespace in ``sys.modules`` holds as an attribute, if it does."""
    parent, _, name = fullname.rpartition(".")
    found = getattr(sys.modules.get(parent), name, None)
    if not isinstance(found, ModuleType):
        return None
    return importlib.util.spec_from_loader(fullname, _Member(found))


def _installed(alias: str) -> Aliases | None:
    for finder in sys.meta_path:
        if isinstance(finder, Aliases) and finder.alias == alias:
            return finder
    return None


def install_root() -> ModuleType:
    """``ROOT`` is :mod:`xrdroot.pyroot`, ``ROOT.RDF`` its namespace, ``ROOT.x`` its module."""
    root = sys.modules["ROOT"] = importlib.import_module("xrdroot.pyroot")
    for name in NAMESPACES:  # ``from ROOT.Experimental import RCanvas`` imports the namespace
        found = getattr(root, name, None)
        if found is not None:
            sys.modules[f"ROOT.{name}"] = found
            _as_package(found, f"ROOT.{name}")
    if _installed("ROOT") is None:  # first: ``ROOT.__path__`` would load a second copy
        sys.meta_path.insert(0, Aliases("ROOT", "xrdroot.pyroot"))
    return root


def _as_package(namespace: Any, name: str) -> None:
    """Give a namespace object a package's ``__path__`` and ``__spec__``, so that ``import
    ROOT.RDF.Distributed`` looks inside it for ``Distributed`` as Python looks in a package."""
    if isinstance(namespace, ModuleType) or "__path__" in getattr(namespace, "__dict__", {}):
        return  # a module is imported from as it is
    try:
        namespace.__path__ = []
        namespace.__name__ = name
        namespace.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    except (AttributeError, TypeError):  # an object that takes no new attributes
        pass


def install_cppyy() -> None:
    """``import cppyy`` gives :mod:`xrdroot.cppyy`, unless a real cppyy is installed."""
    if _installed("cppyy") is None:  # last: whatever else finds a cppyy wins
        sys.meta_path.append(Aliases("cppyy", "xrdroot.cppyy"))


def python_warnings(on: bool) -> bool:
    """ROOT's warnings as Python's ``RuntimeWarning`` and its classes answering Python as
    PyROOT's pythonizations do - or not; was it so before?

    Only a namespace already imported is told: the warnings are its messages.
    """
    messages = sys.modules.get("xrdroot.pyroot.core.messages")
    before = bool(getattr(messages, "PYROOT_SCRIPT", False))
    if messages is not None:
        setattr(messages, "PYROOT_SCRIPT", on)  # noqa: B010 - a module looked up by name
    return before
