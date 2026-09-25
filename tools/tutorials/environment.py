"""The ROOT installed here - the oracle - and the build options it was made with.

Nothing here imports ROOT: it is asked through ``root-config`` (its version,
its ``--features``, where its libraries are). The oracle is a prefix - an
installation with ``bin/root``, ``bin/root-config`` and, for a conda-forge
ROOT, ``bin/python`` with PyROOT in it - chosen by ``--oracle-prefix``,
``$XRDROOT_ORACLE_PREFIX``, or ``~/.local/root-oracle`` when that exists;
failing all three, whatever ``root-config`` is on ``PATH``. Its Python is the
prefix's own, else the ``pythonX.Y`` that ``root-config --python-version``
names, run with ROOT's library directory on ``PYTHONPATH``. ROOT is never a
dependency of xrdroot; without it the harness still runs xrdroot, and says
there was nothing to compare against.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Oracle", "find_oracle", "root_config", "cmake_variables", "python_for"]

#: CMake variables the tutorials' CMakeLists tests that ``--features`` spells differently:
#: each is ON when any of the features after it is.
DERIVED = {
    "TBB_FOUND": ("imt", "tbb"),
    "GRAPHVIZ_FOUND": ("gviz",),
    "BLAS_FOUND": ("tmva-cpu",),
    "tmva-pymva": ("tmva-pymva",),
}

#: Where the oracle ROOT is installed when nothing says otherwise.
DEFAULT_PREFIX = Path(os.environ.get("XRDROOT_ORACLE_PREFIX", "~/.local/root-oracle"))


@dataclass(frozen=True)
class Oracle:
    """A ROOT to run tutorials with: ``root``, and a Python that imports PyROOT."""

    root: str
    version: str
    features: frozenset[str] = frozenset()
    python: str | None = None
    pythonpath: str = ""
    rootsys: str = ""
    modules: dict[str, bool] = field(default_factory=dict, compare=False, hash=False)

    def has_module(self, name: str) -> bool:
        """Whether the oracle's Python imports a package - asked once, then remembered."""
        if name not in self.modules:
            self.modules[name] = self._imports(name)
        return self.modules[name]

    def _imports(self, name: str) -> bool:
        if self.python is None:
            return False
        env = dict(os.environ, PYTHONPATH=self.pythonpath)
        try:
            done = subprocess.run(
                [self.python, "-c", f"import {name}"], env=env, capture_output=True, timeout=120
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return done.returncode == 0


def _ask(command: Sequence[str]) -> str | None:
    try:
        done = subprocess.run(list(command), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def python_for(version: str, bindir: str = "") -> str | None:
    """The Python ROOT's bindings are for: the prefix's own, else ``pythonX.Y`` on the path."""
    own = Path(bindir) / "python" if bindir else None
    if own is not None and own.is_file():
        return str(own)
    return shutil.which(f"python{version}") if version else None


def root_config(prefix: Path | None = None) -> str:
    """The ``root-config`` of the oracle prefix: the one given, the default, or the path's."""
    for candidate in (prefix, DEFAULT_PREFIX):
        if candidate is not None:
            config = Path(candidate).expanduser() / "bin" / "root-config"
            if config.is_file():
                return str(config)
    return "root-config"


def find_oracle(config: str = "root-config", python: str | None = None) -> Oracle | None:
    """The ROOT a ``root-config`` belongs to, if it answers; ``None`` if there is none."""
    config = shutil.which(config) or config
    version = _ask([config, "--version"])
    if version is None:
        return None
    bindir = _ask([config, "--bindir"]) or str(Path(config).parent)
    features = frozenset((_ask([config, "--features"]) or "").split())
    if python is None:
        python = python_for(_ask([config, "--python-version"]) or "", bindir)
    return Oracle(
        root=str(Path(bindir) / "root"),
        version=version,
        features=features,
        python=python,
        pythonpath=_ask([config, "--libdir"]) or "",
        rootsys=_ask([config, "--prefix"]) or "",
    )


def cmake_variables(features: frozenset[str], source: Path, python: str = "") -> dict[str, str]:
    """The variables ROOT's configure step would have set, for the tutorials' CMakeLists."""
    variables = dict.fromkeys(features, "ON")
    for name, spelled in DERIVED.items():
        if any(feature in features for feature in spelled):
            variables[name] = "ON"
    variables.update(
        CMAKE_SOURCE_DIR=str(Path(source).parent),
        CMAKE_BINARY_DIR="<build>",
        CMAKE_CURRENT_BINARY_DIR="<build>/tutorials",
        CMAKE_SYSTEM_NAME=platform.system(),
        CMAKE_SIZEOF_VOID_P="8",
        Python3_EXECUTABLE=python or "python3",
        ROOT_root_CMD="root",
    )
    return variables


def targets(features: frozenset[str]) -> set[str]:
    """The CMake targets ``if(TARGET ...)`` finds: ROOT's GUI, in any build with graphics."""
    return {"Gui"} if features & {"x11", "cocoa", "asimage", "opengl"} else set()
