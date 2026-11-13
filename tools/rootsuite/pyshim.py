"""Run ROOT's Python unittest files under pytest with ``import ROOT`` meaning xrdroot.pyroot.

Usage (in the Python whose xrdroot is under test)::

    python pyshim.py --junitxml OUT.xml test_file.py [...]

The harness runs this file by path in a child process, so it imports
nothing of the harness: it puts :mod:`xrdroot.pyroot` in ``sys.modules`` as
``ROOT`` (and its namespaces as ``ROOT.RDF``...) before pytest imports a
test, then runs pytest with ``--noconftest``. ROOT's own ``conftest.py``
imports PyROOT internals (``ROOT._pythonization``); rather than fail every
test in its directory on that import, it is loaded here as a plugin when it
imports, and left out - with a line on standard error - when it does not,
so its fixtures appear the day xrdroot can provide what they need.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

#: ROOT's namespaces a test imports names from, ``from ROOT.RDF import ...``.
NAMESPACES = ("Experimental", "VecOps", "RDF", "Math", "RooFit", "TMVA")


def alias_root() -> ModuleType:
    """``import ROOT`` gives xrdroot.pyroot, and ``import ROOT.RDF`` its namespace."""
    root = importlib.import_module("xrdroot.pyroot")
    sys.modules["ROOT"] = root
    for name in NAMESPACES:
        found = getattr(root, name, None)
        if found is not None:
            sys.modules[f"ROOT.{name}"] = found
    return root


def conftests(files: list[str]) -> list[ModuleType]:
    """Each test directory's ``conftest.py`` that imports, ready to be a pytest plugin."""
    loaded = []
    for directory in sorted({Path(name).resolve().parent for name in files}):
        path = directory / "conftest.py"
        if path.is_file():
            module = _load(path)
            if module is not None:
                loaded.append(module)
    return loaded


def _load(path: Path) -> ModuleType | None:
    spec = importlib.util.spec_from_file_location(f"rootsuite_conftest_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as why:  # a conftest needing PyROOT internals is left out, said so
        sys.stderr.write(f"pyshim: {path} not loaded: {type(why).__name__}: {why}\n")
        return None
    return module


def main(argv: list[str]) -> int:
    here = Path(__file__).resolve().parent  # the harness's modules must not shadow a test's imports
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != here]
    alias_root()
    import pytest

    files = [arg for arg in argv if arg.endswith(".py")]
    args = [*argv, "--noconftest", "-p", "no:cacheprovider", "-q", "-rfE"]
    return int(pytest.main(args, plugins=conftests(files)))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
