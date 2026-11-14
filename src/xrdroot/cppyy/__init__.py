"""``import cppyy``: the binding PyROOT is built on, over xrdroot's C++ translator.

cppyy hands Python the C++ a program declares: ``cppyy.cppdef`` compiles
code, ``cppyy.gbl`` is C++'s global namespace - every class, function and
namespace declared, with ROOT's - and ``cppyy.include`` and
``cppyy.load_library`` bring in headers and libraries. Here the C++ is
translated into Python by :mod:`xrdroot.pycint`, in ROOT's dialect, as
``gInterpreter.Declare`` translates it, so ``cppyy.gbl.X`` and ``ROOT.X``
are the one object, as they are in PyROOT; a library is the ``.xrdlib``
``xrdroot build`` makes. Pointers are Python references: an address is a
number standing for its object while it lives, ``bind_object`` finds the
object again, and a null pointer bound to a class is a falsy object of it.

``xrdroot run`` and the suite's shim make ``import cppyy`` this module when
no real cppyy is installed (see :mod:`xrdroot.cli.shims`).
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path
from typing import Any

from . import ll, py
from .pointers import addressof, bind_object, nullptr

__all__ = [
    "gbl",
    "cppdef",
    "cppexec",
    "include",
    "c_include",
    "load_library",
    "load_reflection_info",
    "add_include_path",
    "add_library_path",
    "addressof",
    "bind_object",
    "nullptr",
    "sizeof",
    "set_debug",
    "ll",
    "py",
]

#: The cppyy release whose interface this follows.
__version__ = "3.5.0"


def _root() -> Any:
    """The ROOT namespace, which holds every name C++ has declared so far."""
    from .. import pyroot

    return pyroot


class _Global(types.ModuleType):
    """``cppyy.gbl``: C++'s global namespace - ``gbl.std.vector``, ``gbl.TH1F``, ``gbl.MyClass``.

    It is a module, so that ``from cppyy.gbl import std`` imports from it.
    """

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            found = getattr(_root(), name)
        except AttributeError:
            raise AttributeError(f"<namespace cppyy.gbl> has no attribute '{name}'") from None
        return py.pythonized(found, name)

    def __repr__(self) -> str:
        return "<namespace cppyy.gbl at 0x0>"


#: ``cppyy.gbl``.
gbl = _Global(f"{__name__}.gbl")
gbl.__spec__ = importlib.machinery.ModuleSpec(gbl.__name__, None, is_package=True)
gbl.__path__ = []
sys.modules[gbl.__name__] = gbl


def cppdef(code: str) -> bool:
    """``cppyy.cppdef``: C++ declarations, translated and run; ``SyntaxError`` if they will not."""
    try:
        _root().gInterpreter.Declare(str(code))
    except Exception as why:  # what the translator refuses, or the code raises as it runs
        raise SyntaxError(f"Failed to parse the given C++ code: {why}") from why
    return True


def cppexec(code: str) -> bool:
    """``cppyy.cppexec``: C++ statements, run at once."""
    try:
        _root().gInterpreter.ProcessLine(str(code))
    except Exception as why:
        raise SyntaxError(f"Failed to parse the given C++ code: {why}") from why
    return True


#: The directories ``include`` looks for a header in after the working directory.
INCLUDE_PATHS: list[Path] = []


def include(header: str) -> bool:
    """``cppyy.include``: a header, read as ``#include`` reads it - from the working directory
    or one added to the path, else as the library's own; ``ImportError`` if it will not read."""
    found = next((each / header for each in INCLUDE_PATHS if (each / header).is_file()), None)
    try:
        _root().gInterpreter.Declare(f'#include "{found or header}"')
    except Exception as why:
        raise ImportError(f"Failed to load header file \"{header}\": {why}") from why
    return True


c_include = include


def load_library(name: str) -> bool:
    """``cppyy.load_library``: a library ``xrdroot build`` made, loaded as ``gSystem->Load``."""
    if _root().gSystem.Load(str(name)) < 0:
        raise RuntimeError(
            f"Could not load library \"{name}\": not found, or not one xrdroot built"
        )
    return True


#: ``load_reflection_info``: a dictionary is a library here like any other.
load_reflection_info = load_library


def add_include_path(path: str) -> None:
    """``cppyy.add_include_path``: a directory headers are looked for in."""
    INCLUDE_PATHS.append(Path(path))
    _root().gSystem.AddIncludePath(f"-I{path}")


def add_library_path(path: str) -> None:
    """``cppyy.add_library_path``: a directory libraries are looked for in."""
    _root().gSystem.AddDynamicPath(str(path))


def sizeof(kind: Any) -> int:
    """``cppyy.sizeof``: the bytes a value of a fundamental type takes, as C++'s ``sizeof``."""
    from ..cppstd.containers import element_type

    try:
        return int(element_type(kind).itemsize if isinstance(kind, str) else 8)
    except (TypeError, AttributeError):
        return 8


def set_debug(enable: bool = True) -> None:
    """``cppyy.set_debug``: cling's debug output, which a translation has none of."""

