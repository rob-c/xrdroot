"""The ROOT installed here - the oracle - and the build options it was made with.

Nothing here imports ROOT: it is asked through ``root-config`` (its version,
its ``--features``, where its libraries are) and its Python bindings are
found as the Python ``root-config --python-version`` names, with ROOT's
library directory on ``PYTHONPATH`` - which is how Homebrew's ROOT is used
from Python. ROOT is never a dependency of xrdroot; without it the harness
still runs xrdroot, and says there was nothing to compare against.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Oracle", "find_oracle", "cmake_variables", "python_for"]

#: CMake variables the tutorials' CMakeLists tests that ``--features`` spells differently:
#: each is ON when any of the features after it is.
DERIVED = {
    "TBB_FOUND": ("imt", "tbb"),
    "GRAPHVIZ_FOUND": ("gviz",),
    "BLAS_FOUND": ("tmva-cpu",),
    "tmva-pymva": ("tmva-pymva",),
}

#: Where Homebrew keeps a versioned Python, on Intel and on Apple silicon.
BREW_PREFIXES = ("/usr/local/opt", "/opt/homebrew/opt")


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


def python_for(version: str) -> str | None:
    """The ``pythonX.Y`` ROOT's bindings were built for: on the path, or Homebrew's."""
    name = f"python{version}"
    found = shutil.which(name)
    if found:
        return found
    for prefix in BREW_PREFIXES:
        candidate = Path(prefix) / f"python@{version}" / "bin" / name
        if candidate.is_file():
            return str(candidate)
    return None


def find_oracle(root_config: str = "root-config", python: str | None = None) -> Oracle | None:
    """The installed ROOT, if ``root-config`` answers; ``None`` if there is none."""
    config = shutil.which(root_config) or root_config
    version = _ask([config, "--version"])
    if version is None:
        return None
    bindir = _ask([config, "--bindir"]) or str(Path(config).parent)
    features = frozenset((_ask([config, "--features"]) or "").split())
    if python is None:
        python = python_for(_ask([config, "--python-version"]) or "")
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
