"""A built library on disk: ``libNAME.xrdlib``, the translation a compiler's ``.so`` stands for.

ROOT's build model makes shared libraries - a dictionary from rootcling, a
library from sources, ``macro_C.so`` from ACLiC - and loads them later by
name. A translated library is the same thing for xrdroot: one JSON file
holding the Python the sources translate into, what it needs to report an
error against the C++ (the source map), and what it provides - the names it
defines, the namespaces they sit in and the headers it read - so that loading
it, or finding the library that provides a name (ROOT's rootmap), needs no
translator run.

Paths of the files a library was built from are kept relative to the
library's own directory when they are under it, so a build directory moved
as a whole still works; the rest are absolute.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ...pycxx.translation import Translation

__all__ = ["FORMAT", "SUFFIX", "LIBRARY_SUFFIXES", "Library", "read", "write", "stem_of"]

#: What a library file says it is in its ``format`` field.
FORMAT = "xrdroot-library"
#: The version of the layout below; a different one is rebuilt, never misread.
VERSION = 2
#: The extension of a translated library, where a compiler would write ``.so``.
SUFFIX = ".xrdlib"
#: The extensions ROOT accepts after a library's name, every one meaning the same library.
LIBRARY_SUFFIXES = (".so", ".dylib", ".dll", ".sl", ".dl", ".a", SUFFIX)


@dataclass
class Library:
    """A translated library: its Python, and what it provides to the code loaded after it."""

    name: str
    #: ``dictionary`` (rootcling, genreflex), ``library`` (from sources) or ``aclic``.
    kind: str
    python: str
    #: The file the translation is said to be of: the macro, for ACLiC's.
    file: str
    source_map: dict[int, tuple[str, int]] = field(default_factory=dict)
    entry: str | None = None
    unnamed: bool = False
    returns: str | None = None
    #: The files it was built from, and every local file they read, as stored.
    sources: list[str] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)
    #: A hash of the text of everything it was built from: what ACLiC's "up to date" is.
    digest: str = ""
    translator: str = ""
    #: The names it defines at its top, and each namespace's: ``{"ns": ["A", {"inner": [...]}]}``.
    names: list[str] = field(default_factory=list)
    namespaces: dict[str, Any] = field(default_factory=dict)
    #: The classes it provides, qualified: what a rootmap lists, and autoloading looks up.
    classes: list[str] = field(default_factory=list)
    #: The enumerations it declares, as ROOT's interpreter knows them once it is loaded: each
    #: by its qualified name, with its constants' values (:class:`.translate.EnumNote`).
    enums: list[dict[str, Any]] = field(default_factory=list)
    #: What a dictionary was asked to select - a LinkDef's ``#pragma link`` lines, a genreflex
    #: selection file's classes - which are the classes it gives a dictionary to. A header's
    #: other classes are translated with it but have none, as ROOT's dictionary has none of
    #: them. Empty when the selection names classes by pattern, and so selects whatever of the
    #: header matches: then every class the dictionary was built from has one.
    selected: list[str] = field(default_factory=list)
    #: Whether autoloading finds it by the names it provides, as ROOT's rootmap file does.
    rootmap: bool = True
    #: The libraries to load before it.
    libraries: list[str] = field(default_factory=list)
    #: Where it was read from - not stored.
    path: Path | None = None

    def translation(self) -> Translation:
        """The library's Python as the translator hands a translation over."""
        return Translation(
            self.python, self.file, dict(self.source_map), self.entry,
            self.resolved(self.headers), self.unnamed, self.returns,
        )  # fmt: skip

    def resolved(self, stored: list[str]) -> list[str]:
        """Stored paths as absolute ones, those kept relative taken from the library's place."""
        base = self.path.parent if self.path is not None else Path.cwd()
        return [str((base / each).resolve()) if not os.path.isabs(each) else each
                for each in stored]  # fmt: skip


def stem_of(name: str) -> str:
    """``libFoo.so``, ``./libFoo`` and ``libFoo.xrdlib`` as the library's name, ``libFoo``."""
    base = os.path.basename(name)
    for suffix in LIBRARY_SUFFIXES:
        if base.endswith(suffix) and len(base) > len(suffix):
            return base[: -len(suffix)]
    return base


def _stored(path: str, base: Path) -> str:
    """``path`` relative to ``base`` when it is under it, else absolute."""
    absolute = Path(path).resolve()
    try:
        return str(absolute.relative_to(base.resolve()))
    except ValueError:
        return str(absolute)


def write(library: Library, directory: Path) -> Path:
    """Write ``library`` into ``directory`` as ``<name>.xrdlib``; where it went."""
    directory.mkdir(parents=True, exist_ok=True)
    where = directory / f"{library.name}{SUFFIX}"
    data = asdict(library)
    data.pop("path")
    data["sources"] = [_stored(each, directory) for each in library.sources]
    data["headers"] = [_stored(each, directory) for each in library.headers]
    data["source_map"] = {str(line): list(place) for line, place in library.source_map.items()}
    data = {"format": FORMAT, "version": VERSION, **data}
    temporary = where.with_name(f".{where.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=1), encoding="utf-8")
    temporary.replace(where)
    library.path = where
    return where


def read(where: Path) -> Library | None:
    """The library at ``where``, or ``None`` if it is not one this version reads."""
    try:
        data = json.loads(where.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        return None
    if data.get("version") != VERSION:
        return None
    data.pop("format")
    data.pop("version")
    data["source_map"] = {int(line): (place[0], place[1])
                          for line, place in data.get("source_map", {}).items()}  # fmt: skip
    try:
        library = Library(**data)
    except TypeError:
        return None
    library.path = where
    return library
