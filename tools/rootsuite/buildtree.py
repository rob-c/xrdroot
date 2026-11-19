"""The build tree tests expect around them: ROOT's programs, ``tutorials/``, ``hsimple.root``.

ctest runs ROOT's tests from a build directory that holds more than the
tests: ``bin/`` with ``root.exe``, ``hadd``, ``rootls``...; ``tutorials/``,
copied from the sources; and ``tutorials/hsimple.root``, which the build
makes by running ``hsimple.C``. ``RootTestDriver.cmake`` puts ``bin/``
first on ``PATH`` and the tree in ``ROOTSYS``, so a shell script that calls
``root`` or a macro that opens ``$ROOTSYS/tutorials/hsimple.root`` finds
them. The working copy is given the same: ``bin/`` holds launchers for
xrdroot's stand-ins (or, for a subcommand this xrdroot lacks, a launcher
that says so the way xrdroot refuses), ``tutorials/`` is copied, and
``hsimple.root`` is made with ``xrdroot root`` as ROOT's build makes it.

Which subcommands an xrdroot has is asked of it - ``xrdroot --help`` -
since the harness never imports it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from .plan import TOOLS
from .process import Result, execute

__all__ = ["subcommands", "launchers", "tutorials", "hsimple", "DESCRIPTIONS"]

#: What each xrdroot subcommand the suite needs stands in for, for the refusals that name it.
DESCRIPTIONS = {
    "build": "ROOT's build steps (rootcling, genreflex, ACLiC and the C++ compiler)",
    "root": "ROOT's root.exe", "hadd": "ROOT's hadd", "ls": "ROOT's rootls",
    "rm": "ROOT's rootrm", "mkdir": "ROOT's rootmkdir", "mv": "ROOT's rootmv",
    "cp": "ROOT's rootcp", "print": "ROOT's rootprint", "run": "a macro or program run",
}  # fmt: skip

#: An entry of ``xrdroot --help``'s list of subcommands.
_ENTRY = re.compile(r"^ {4}([a-z][\w-]*)(?:\s|$)", re.M)

#: ROOT's dictionary generators, which a test may run from a shell of its own - inside a
#: Python script's ``subprocess``, say - so that ``bin/`` has to hold them too.
GENERATORS = ("genreflex", "rootcling", "rootcint")

#: A generator's stand-in: the translation of its words (``dictgen.dictionary_words``) is the
#: harness's, and it is made when the generator is run, so the shim asks the harness.
_GENERATOR = '''#!{python}
"""{tool}: xrdroot's dictionary generator, its words as ``xrdroot build dictionary``'s."""
import subprocess
import sys

sys.path.insert(0, {root!r})
from tools.rootsuite.dictgen import dictionary_words  # noqa: E402  (after the path)

sys.exit(subprocess.call([{python!r}, "-m", "xrdroot", "build",
                          *dictionary_words({tool!r}, sys.argv[1:])]))
'''


def subcommands(python: str, env: Mapping[str, str]) -> frozenset[str] | None:
    """The subcommands ``python -m xrdroot`` has, or ``None`` when it does not say."""
    try:
        done = subprocess.run([python, "-m", "xrdroot", "--help"], capture_output=True,
                              text=True, env=dict(env), timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    found = frozenset(_ENTRY.findall(done.stdout))
    return found or None


def missing(needed: frozenset[str], available: frozenset[str] | None) -> str:
    """The sentence saying which subcommand a test needs and this xrdroot lacks, or ``""``."""
    if available is None or needed <= available:
        return ""
    lacking = sorted(needed - available)[0]
    return (f"This xrdroot has no `xrdroot {lacking}` subcommand, which stands in for "
            f"{DESCRIPTIONS.get(lacking, 'a ROOT program')}.")


def _launcher(python: str, tool: str, sub: str, available: frozenset[str] | None) -> str:
    if available is None or sub in available:
        return f'#!/bin/sh\nexec "{python}" -m xrdroot {sub} "$@"\n'
    sentence = missing(frozenset({sub}), available).replace("'", "")
    return f"#!/bin/sh\necho 'xrdroot {sub}: {sentence}' >&2\nexit 2\n"


def launchers(work: Path, python: str, available: frozenset[str] | None) -> Path:
    """``bin/`` of the working copy: each ROOT program a script starting its stand-in."""
    directory = work / "bin"
    directory.mkdir(parents=True, exist_ok=True)
    for tool, sub in TOOLS.items():
        path = directory / tool
        path.write_text(_launcher(python, tool, sub, available), encoding="utf-8")
        path.chmod(0o755)
    _generators(directory, python, available)
    return directory


def _generators(directory: Path, python: str, available: frozenset[str] | None) -> None:
    """``bin/``'s ``genreflex`` and its like, for a test that runs one from a shell."""
    root = Path(__file__).resolve().parents[2]
    for tool in GENERATORS:
        path = directory / tool
        if available is not None and "build" not in available:
            path.write_text(_launcher(python, tool, "build", available), encoding="utf-8")
        else:
            path.write_text(_GENERATOR.format(python=python, tool=tool, root=str(root)),
                             encoding="utf-8")  # fmt: skip
        path.chmod(0o755)


def tutorials(root: Path, work: Path) -> Path | None:
    """``tutorials/`` copied into the working copy once, as ROOT's build copies it."""
    source, target = root / "tutorials", work / "tutorials"
    if not source.is_dir():
        return None
    if not (target / ".rootsuite-copied").exists():
        shutil.copytree(source, target, symlinks=True, dirs_exist_ok=True)
        (target / ".rootsuite-copied").touch()
    return target


def hsimple(directory: Path, python: str, env: Mapping[str, str]) -> Result | None:
    """``tutorials/hsimple.root``, made as ROOT's build makes it, unless it is there already."""
    if (directory / "hsimple.root").exists():
        return None
    argv = (python, "-m", "xrdroot", "root", "-q", "-b", "-n",
            "-e", f".L {directory / 'hsimple.C'}", "-e", "hsimple();", "-e", "return")
    return execute(argv, directory, env, 300.0)
