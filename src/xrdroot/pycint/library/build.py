"""Building a library: rootcling's dictionary, a library from sources, ACLiC's ``macro_C.so``.

Each is a translation written to ``<name>.xrdlib`` (:mod:`.manifest`). A
dictionary or a library is its files translated as one unit, as a unity
build compiles them; ACLiC's is the macro's own translation, with the
function named like the file as its entry. What a build cannot do it says as
rootcling says it, on standard error - ``file:line:1: error: ...`` and then
``Error: rootcling: compilation failure`` - and fails with a
:class:`BuildError`; what a LinkDef or selection file asks for and the
headers do not declare is ``Warning: Unused class rule: Foo``, as both
rootcling and genreflex warn and carry on.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from ...pycxx import cache
from ...pycxx.errors import CxxError
from ..dialect import ROOT as DIALECT
from .manifest import Library, write
from .selection import requested
from .translate import Declared, digest, translated_unit

__all__ = ["BuildError", "build_dictionary", "build_library", "build_aclic", "translator_version"]


class BuildError(CxxError):
    """A build that failed: the tool's last word. The compiler's errors before it are on
    standard error already."""


def translator_version() -> str:
    """The translator's version, as its cache keys a translation: a library written by
    another is translated again rather than trusted."""
    return cache.key("", "<library>", DIALECT)


def _missing(paths: Sequence[Path], tool: str) -> None:
    """Refuse, as rootcling does, files that are not there."""
    for path in paths:
        if not path.is_file():
            sys.stderr.write(f"fatal error: '{path}' file not found\n")
            raise BuildError(f"{tool}: compilation failure ({path})")


def _failed(why: CxxError, tool: str, what: str) -> BuildError:
    """A refusal as a compiler's error at the C++ line, then the tool's own last word."""
    where = why.where
    spot = f"{where.file}:{where.line}:1: " if where is not None else ""
    sys.stderr.write(f"{spot}error: {why.why}\n")
    return BuildError(f"{tool}: compilation failure ({what})")


def _unit(name: str, files: Sequence[Path], includes: Sequence[Path], tool: str) -> Library:
    """``files`` translated as one unit into the library ``name``."""
    _missing(files, tool)
    source = "".join(f'#include "{path.resolve()}"\n' for path in files)
    unity = str(Path.cwd() / name)
    directories = [*includes, *(path.parent for path in files), Path.cwd()]
    try:
        made, found = translated_unit(source, unity, list(dict.fromkeys(directories)))
    except CxxError as why:
        raise _failed(why, tool, name) from None
    inputs = [str(path.resolve()) for path in files]
    headers = _absolute(made.included)
    return Library(
        name, "library", made.python, unity, dict(made.source_map), sources=inputs,
        headers=headers, digest=digest(inputs + headers),
        translator=translator_version(), names=found.top, namespaces=found.namespaces,
        classes=found.classes,
    )  # fmt: skip


def _absolute(paths: list[str]) -> list[str]:
    """``paths`` absolute, each once: what a library's up-to-date check hashes."""
    return list(dict.fromkeys(str(Path(path).resolve()) for path in paths))


def _unused(rules: list[str], found: Declared) -> None:
    """``Warning: Unused class rule: Foo`` for each rule naming nothing declared - unless it
    is the standard library's or ROOT's own class."""
    for rule in rules:
        if not found.knows(rule) and not _elsewhere(rule):
            sys.stderr.write(f"Warning: Unused class rule: {rule}\n")


def _elsewhere(rule: str) -> bool:
    """Whether ``rule`` names a class of the standard library or of ROOT itself."""
    bare = rule.split("<", 1)[0].strip().removeprefix("::")
    if bare.startswith("std::"):
        return True
    from ... import pyroot

    head = bare.split("::")[0]
    return hasattr(pyroot, head)


def build_dictionary(
    name: str,
    files: Sequence[Path],
    *,
    linkdef: Path | None = None,
    selection: Path | None = None,
    includes: Sequence[Path] = (),
    out: Path | None = None,
    rootmap: bool = True,
    libraries: Sequence[str] = (),
) -> Path:
    """rootcling's (or genreflex's) dictionary of ``files`` as ``lib<name>.xrdlib``; its path."""
    tool = "genreflex" if selection is not None and linkdef is None else "rootcling"
    _missing([each for each in (linkdef, selection) if each is not None], tool)
    library = _unit(_library_name(name), files, includes, tool)
    _unused(requested(linkdef, selection), _found(library))
    library.kind, library.rootmap, library.libraries = "dictionary", rootmap, list(libraries)
    return write(library, out or Path.cwd())


def _found(library: Library) -> Declared:
    return Declared(library.classes, [], library.namespaces, library.names)


def build_library(
    name: str,
    files: Sequence[Path],
    *,
    includes: Sequence[Path] = (),
    out: Path | None = None,
    libraries: Sequence[str] = (),
) -> Path:
    """A library compiled from ``files``, as ``lib<name>.xrdlib``; its path.

    It carries no rootmap: as a library ROOT links without a dictionary, it is
    there once loaded, and found by no name before.
    """
    library = _unit(_library_name(name), files, includes, "xrdroot build library")
    library.rootmap, library.libraries = False, list(libraries)
    return write(library, out or Path.cwd())


def _library_name(name: str) -> str:
    """``Foo`` as ``libFoo``, the file a linker makes of it; ``libFoo`` stays."""
    return name if name.startswith("lib") else f"lib{name}"


def build_aclic(macro: Path, name: str, out: Path) -> Library:
    """ACLiC's library of ``macro``: its translation, as ``<out>/<name>.xrdlib``."""
    _missing([macro], "ACLiC")
    text = macro.read_text(encoding="utf-8", errors="replace")
    try:
        made, found = translated_unit(text, str(macro), [macro.parent])
    except CxxError as why:
        where = why.where
        spot = f"{where.file}:{where.line}:1: " if where is not None else ""
        sys.stderr.write(f"{spot}error: {why.why}\n")
        raise BuildError("Compilation failed!") from None
    inputs = [str(macro.resolve())]
    headers = _absolute(made.included)
    library = Library(
        name, "aclic", made.python, str(macro), dict(made.source_map), made.entry, made.unnamed,
        made.returns, sources=inputs, headers=headers, digest=digest(inputs + headers),
        translator=translator_version(),
        names=found.top, namespaces=found.namespaces, classes=found.classes, rootmap=False,
    )  # fmt: skip
    write(library, out)
    return library
