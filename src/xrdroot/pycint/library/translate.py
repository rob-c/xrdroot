"""Translating what a library is built from, and reading off what it declares.

A library's headers and sources are one translation unit here, as a unity
build makes them: each is ``#include``-d in turn, so a header the others
include is read once behind its guard and an out-of-line method finds its
class. The unit is translated in ROOT's dialect, and besides the Python the
declarations are read from the parsed unit - each class with its namespaces,
what a rootmap lists and a LinkDef selects.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...pycxx.dialect import using
from ...pycxx.emit_module import Translator
from ...pycxx.nodes import (
    ClassDecl,
    DeclStmt,
    EnumDecl,
    Function,
    Namespace,
    Stmt,
    Typedef,
    Unit,
)
from ...pycxx.parse_decl import parse
from ...pycxx.preprocessor import preprocess
from ...pycxx.program import Program
from ...pycxx.translation import Translation
from ..dialect import ROOT as DIALECT

__all__ = ["Declared", "translated_unit", "declared", "declared_in", "digest"]


@dataclass
class Declared:
    """What a unit declares, by C++ name: classes qualified, namespaces nested."""

    #: Every class, struct, union, enum and typedef, qualified: ``ns::Inner``.
    classes: list[str] = field(default_factory=list)
    #: The functions, variables and enumerators at namespace scope, qualified.
    others: list[str] = field(default_factory=list)
    #: Each namespace's members by name, nested: ``{"ns": {"A": None, "in": {...}}}``.
    namespaces: dict[str, Any] = field(default_factory=dict)
    #: The names at the top, outside any namespace.
    top: list[str] = field(default_factory=list)

    def knows(self, name: str) -> bool:
        """Whether ``name`` - ``ns::A``, ``A<int>`` - is a class or namespace declared here."""
        bare = name.split("<", 1)[0].strip().removeprefix("::")
        if bare in self.classes or bare in self.others:
            return True
        return self._namespace(bare) is not None

    def _namespace(self, qualified: str) -> Any:
        found: Any = self.namespaces
        for part in qualified.split("::"):
            if not isinstance(found, dict) or part not in found:
                return None
            found = found[part]
        return found


def translated_unit(
    source: str, file: str, directories: list[Path]
) -> tuple[Translation, Declared]:
    """``source``, as the file ``file``, translated in ROOT's dialect; and what it declares,
    read before the translator settles each name in its class."""
    with using(DIALECT):
        tokens, included = preprocess(source, file, directories)
        unit = parse(tokens)
        found = declared(unit)
        translator = Translator(Program(unit), file, Path(file).stem)
        out = translator.translate()
    made = Translation(out.text(), file, out.source_map(), translator.entry,
                       [str(path) for path in included], unit.unnamed is not None,
                       translator.entry_returns)  # fmt: skip
    return made, found


def declared_in(source: str) -> Declared:
    """What ``source``, a few lines of C++, declares: its namespaces' members, nested."""
    with using(DIALECT):
        tokens, _ = preprocess(source, "<namespaces>", [])
        return declared(parse(tokens))


def declared(unit: Unit) -> Declared:
    """What ``unit`` declares at namespace scope, read off its declarations."""
    found = Declared()
    _walk(unit.decls, [], found, found.namespaces)
    return found


def _walk(decls: list[Stmt], scope: list[str], found: Declared, members: dict[str, Any]) -> None:
    for decl in decls:
        if isinstance(decl, Namespace):
            _namespace(decl, scope, found, members)
            continue
        for name, is_class in _names(decl):
            _record(found, members, scope, name, is_class)
        if isinstance(decl, ClassDecl):
            nested = [member for member in decl.members if isinstance(member, ClassDecl)]
            _nested(nested, [*scope, decl.name], found)


def _namespace(decl: Namespace, scope: list[str], found: Declared, members: Any) -> None:
    if decl.name is None:  # an unnamed namespace's names are the enclosing one's
        _walk(decl.body, scope, found, members)
        return
    inner = members.setdefault(decl.name, {})
    if not scope:
        _add(found.top, decl.name)
    _walk(decl.body, [*scope, decl.name], found, inner)


def _nested(classes: list[ClassDecl], scope: list[str], found: Declared) -> None:
    """Classes declared inside a class: ``Outer::Inner``, known to a rootmap by that name."""
    for decl in classes:
        _add(found.classes, "::".join([*scope, decl.name]))
        _nested([member for member in decl.members if isinstance(member, ClassDecl)],
                [*scope, decl.name], found)  # fmt: skip


def _names(decl: Stmt) -> list[tuple[str, bool]]:
    """The names one declaration declares, each with whether it names a type."""
    if isinstance(decl, (ClassDecl, Typedef)):
        return [(decl.name, True)]
    if isinstance(decl, EnumDecl):
        return _enumerated(decl)
    if isinstance(decl, Function):
        return [] if decl.scope else [(decl.name, False)]
    if isinstance(decl, DeclStmt):
        # ``int A::n = 1;`` defines a class's static member, which is no name of its own.
        return [(each.name, False) for each in decl.decls if "::" not in each.name]
    return []


def _enumerated(decl: EnumDecl) -> list[tuple[str, bool]]:
    """An enum's name, and its enumerators unless it is scoped."""
    named = [(decl.name, True)] if decl.name else []
    return named + ([] if decl.scoped else [(item, False) for item, _ in decl.items])


def _record(found: Declared, members: Any, scope: list[str], name: str, is_class: bool) -> None:
    qualified = "::".join([*scope, name])
    _add(found.classes if is_class else found.others, qualified)
    if scope:
        members.setdefault(name, None)
    else:
        _add(found.top, name)


def _add(names: list[str], name: str) -> None:
    if name not in names:
        names.append(name)


def digest(paths: list[str]) -> str:
    """A hash of the text of every file in ``paths``: the same only if none has changed."""
    hashed = hashlib.sha256()
    for path in paths:
        hashed.update(path.encode("utf-8", "surrogatepass") + b"\0")
        try:
            hashed.update(Path(path).read_bytes())
        except OSError:
            hashed.update(b"\0missing\0")
    return hashed.hexdigest()

