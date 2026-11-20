"""Building a library: rootcling's dictionary, a library from sources, ACLiC's ``macro_C.so``.

Each is a translation written to ``<name>.xrdlib`` (:mod:`.manifest`). A
dictionary or a library is its files translated as one unit, as a unity
build compiles them; ACLiC's is the macro's own translation, with the
function named like the file as its entry. What a build cannot do it says as
rootcling says a missing file - ``fatal error: 'x.h' file not found`` and then
``Error: rootcling: compilation failure`` - and fails with a
:class:`BuildError`; C++ the translator refuses is raised as the refusal it is
(``xrdroot build`` says it on one line and exits 2, a session asking ACLiC
says it as a compiler would, then ``Compilation failed!``); what a LinkDef or
selection file asks for and the headers do not declare is ``Warning: Unused
class rule: Foo``, as both rootcling and genreflex warn and carry on.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ...pycxx import cache
from ...pycxx.errors import CxxError
from ..dialect import LIBRARY as DIALECT
from . import checks
from .manifest import SUFFIX, Library, read, write
from .selection import requested
from .translate import Declared, digest, translated_unit

__all__ = [
    "BuildError",
    "Strictness",
    "build_dictionary",
    "build_library",
    "build_aclic",
    "explained",
    "translator_version",
]


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


def explained(why: CxxError) -> BuildError:
    """A translator's refusal said as a compiler's error at the C++ line, then ACLiC's last
    word: what a session that asked ACLiC for a library says."""
    where = why.where
    spot = f"{where.file}:{where.line}:1: " if where is not None else ""
    sys.stderr.write(f"{spot}error: {why.why}\n")
    return BuildError("Compilation failed!")


def _unit(
    name: str, files: Sequence[Path], includes: Sequence[Path]
) -> tuple[Library, Declared]:
    """``files`` translated as one unit into the library ``name``, and what they declare.
    What the translator refuses is raised as it refuses it: a build that cannot be done,
    not one that failed."""
    source = "".join(f'#include "{path.resolve()}"\n' for path in files)
    unity = str(Path.cwd() / name)
    directories = [*includes, *(path.parent for path in files), Path.cwd()]
    made, found = translated_unit(source, unity, list(dict.fromkeys(directories)))
    inputs = [str(path.resolve()) for path in files]
    headers = _absolute(made.included)
    return Library(
        name, "library", made.python, unity, dict(made.source_map), sources=inputs,
        headers=headers, digest=digest(inputs + headers),
        translator=translator_version(), names=found.top, namespaces=found.namespaces,
        classes=found.classes, enums=[note.as_stored() for note in found.enums],
    ), found  # fmt: skip


def _absolute(paths: list[str]) -> list[str]:
    """``paths`` absolute, each once: what a library's up-to-date check hashes."""
    return list(dict.fromkeys(str(Path(path).resolve()) for path in paths))


def _unused(rules: list[str], found: Declared) -> list[str]:
    """``Warning: Unused class rule: Foo`` for each rule naming nothing declared - unless it
    is the standard library's or ROOT's own class."""
    return [f"Warning: Unused class rule: {rule}" for rule in rules
            if not found.knows(rule) and not _elsewhere(rule)]  # fmt: skip


#: The standard library's class templates, which a rule may name without ``std::``.
STANDARD = frozenset(
    "vector list deque map multimap set multiset unordered_map unordered_multimap "
    "unordered_set unordered_multiset pair tuple array bitset string complex valarray "
    "shared_ptr unique_ptr weak_ptr atomic function optional variant".split()
)


def _elsewhere(rule: str) -> bool:
    """Whether ``rule`` names a class of the standard library or of ROOT itself."""
    bare = rule.split("<", 1)[0].strip().removeprefix("::")
    if bare.startswith("std::") or bare in STANDARD:
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
    strict: Strictness | None = None,
) -> Path:
    """rootcling's (or genreflex's) dictionary of ``files`` as ``lib<name>.xrdlib``; its path.

    What the generator would refuse is refused first, as it refuses it (:mod:`.checks`);
    ``strict`` asks for what only a generator run by hand does."""
    strict = strict or Strictness()
    genreflex = selection is not None and linkdef is None
    tool = "genreflex" if genreflex else "rootcling"
    _missing([*files, *(each for each in (linkdef, selection) if each is not None)], tool)
    if strict.selection_required:
        checks.unselected(files, linkdef, selection)
    parsed = checks.before(files, linkdef, selection, genreflex)
    place, named = out or Path.cwd(), _library_name(name)
    inputs, existing = _added(place, named, files)
    library, found = _unit(named, inputs, includes)
    warnings = _unused(requested(linkdef, selection), found)
    _warned(warnings + checks.io_targets(parsed, found.members), strict.fail_on_warnings)
    library.kind, library.rootmap = "dictionary", rootmap
    library.libraries = _joined(existing, libraries)
    return write(library, place)


@dataclass(frozen=True)
class Strictness:
    """What a generator run by hand checks that a build's dictionary step does not ask for."""

    #: rootcling's refusal of a dictionary with no LinkDef, selection or ``#pragma link``.
    selection_required: bool = False
    #: genreflex's ``--fail_on_warnings``.
    fail_on_warnings: bool = False


def _warned(warnings: list[str], fatal: bool) -> None:
    """The warnings said; under ``--fail_on_warnings``, the build stopped by them."""
    checks.said(warnings)
    if warnings and fatal:
        raise checks.GeneratorError(1)


def check_selection(
    files: Sequence[Path], *, linkdef: Path | None = None, selection: Path | None = None
) -> None:
    """genreflex's ``--selSyntaxOnly``: the selection parsed and what it says reported, with no
    dictionary written. genreflex refuses a duplicate rule only when it goes on to generate, so
    nothing is refused here."""
    tool = "genreflex" if selection is not None and linkdef is None else "rootcling"
    _missing([*files, *(each for each in (linkdef, selection) if each is not None)], tool)
    checks.before(files, linkdef, selection, tool == "genreflex", fatal_duplicates=False)


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
    there once loaded, and found by no name before. Built where a library of
    that name is already - the dictionary CMake compiles with its sources - it
    adds the sources to it, keeping what it was.
    """
    _missing(files, "xrdroot build library")
    place, named = out or Path.cwd(), _library_name(name)
    inputs, existing = _added(place, named, files)
    library, _ = _unit(named, inputs, includes)
    if existing is not None:
        library.kind, library.rootmap = existing.kind, existing.rootmap
    else:
        library.rootmap = False
    library.libraries = _joined(existing, libraries)
    return write(library, place)


def _added(place: Path, name: str, files: Sequence[Path]) -> tuple[list[Path], Library | None]:
    """The files a library is built from: those of the library of that name already there,
    if one is, and ``files`` added to them."""
    existing = read(place / f"{name}{SUFFIX}")
    before = [Path(each) for each in existing.resolved(existing.sources)] if existing else []
    return list(dict.fromkeys([*before, *(each.resolve() for each in files)])), existing


def _joined(existing: Library | None, libraries: Sequence[str]) -> list[str]:
    """The libraries to load first: the existing library's, and those given."""
    return list(dict.fromkeys([*(existing.libraries if existing else []), *libraries]))


def _library_name(name: str) -> str:
    """``Foo`` as ``libFoo``, the file a linker makes of it; ``libFoo`` stays."""
    return name if name.startswith("lib") else f"lib{name}"


def build_aclic(
    macro: Path,
    name: str,
    out: Path,
    defines: Path | None = None,
    includes: Sequence[Path] = (),
) -> Library:
    """ACLiC's library of ``macro``: its translation, as ``<out>/<name>.xrdlib``; what the
    translator refuses is raised as it refuses it. ``defines`` is a header of the
    ``#define`` s the compiler would have been given, read before the macro."""
    _missing([macro], "ACLiC")
    text = macro.read_text(encoding="utf-8", errors="replace")
    if defines is not None:
        text = f'#include "{defines.resolve()}"\n#include "{macro.resolve()}"\n'
    made, found = translated_unit(text, str(macro), [macro.parent, *includes])
    inputs = [str(macro.resolve())]
    headers = _absolute(made.included)
    library = Library(
        name, "aclic", made.python, str(macro), dict(made.source_map), made.entry, made.unnamed,
        made.returns, sources=inputs, headers=headers, digest=digest(inputs + headers),
        translator=translator_version(),
        names=found.top, namespaces=found.namespaces, classes=found.classes, rootmap=False,
        enums=[note.as_stored() for note in found.enums],
    )  # fmt: skip
    write(library, out)
    return library
