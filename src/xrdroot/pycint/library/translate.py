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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...pycxx.ctype import SIZES
from ...pycxx.dialect import using
from ...pycxx.emit_module import Translator
from ...pycxx.nodes import (
    Binary,
    ClassDecl,
    DeclStmt,
    EnumDecl,
    Expr,
    Function,
    Literal,
    Name,
    Namespace,
    SizeOf,
    Stmt,
    Typedef,
    Unary,
    Unit,
)
from ...pycxx.parse_decl import parse
from ...pycxx.preprocessor import preprocess
from ...pycxx.program import Program
from ...pycxx.translation import Translation
from ..dialect import LIBRARY as DIALECT

__all__ = [
    "Declared",
    "EnumNote",
    "translated_unit",
    "declared",
    "declared_in",
    "enums_of",
    "with_extra_includes",
    "digest",
]


@dataclass
class EnumNote:
    """An enumeration a unit declares, as ROOT's interpreter knows one: its name as C++
    writes it, its constants and their values in the order written, whether it is scoped,
    and the integer type it was declared with.

    A dictionary carries these beside the classes it describes because a program asks the
    interpreter for an enumeration by name - ``TEnum::GetEnum("myNamespace::A")`` - and a
    header's enumerations are known as soon as its dictionary is loaded, before anything
    reads the header.
    """

    qualified: str
    items: list[tuple[str, int]] = field(default_factory=list)
    scoped: bool = False
    #: The integer type after ``:``, spelled as C++ spells it - ``short`` - or empty.
    underlying: str = ""

    def as_stored(self) -> dict[str, Any]:
        """The note as a library's file keeps it, which is plain JSON."""
        return {"name": self.qualified, "items": [list(each) for each in self.items],
                "scoped": self.scoped, "underlying": self.underlying}  # fmt: skip

    @staticmethod
    def of_stored(stored: dict[str, Any]) -> EnumNote:
        """The note a library's file kept."""
        return EnumNote(str(stored["name"]),
                        [(str(name), int(value)) for name, value in stored.get("items", ())],
                        bool(stored.get("scoped")), str(stored.get("underlying", "")))  # fmt: skip


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
    #: Each class's data members, by the class's qualified name.
    members: dict[str, list[str]] = field(default_factory=dict)
    #: Every enumeration declared, in the order met, each once by its qualified name.
    enums: list[EnumNote] = field(default_factory=list)

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


#: ``#pragma extra_include "header"``: a header rootcling reads into a dictionary and the
#: compiler never sees. ROOT defines ``__ROOTCLING__`` for that pass, which is how the pragma
#: is usually guarded; a library build here is the one pass a dictionary and a compile share,
#: so the pragma is read off the text itself.
EXTRA_INCLUDE = re.compile(r'^[ \t]*#[ \t]*pragma[ \t]+extra_include[ \t]*"([^"]+)"', re.M)


def with_extra_includes(source: str) -> str:
    """``source`` with every header a ``#pragma extra_include`` names included before it.

    rootcling parses those headers for the same reason it parses the file: their declarations
    are what the dictionary describes, and the session then has them - ``dictgen_base``'s
    ``myextra.h`` declares ``f()``, which the macro itself never includes.
    """
    named = EXTRA_INCLUDE.findall(source)
    if not named:
        return source
    return "".join(f'#include "{each}"\n' for each in named) + source


def translated_unit(
    source: str, file: str, directories: list[Path]
) -> tuple[Translation, Declared]:
    """``source``, as the file ``file``, translated in ROOT's dialect; and what it declares,
    read before the translator settles each name in its class."""
    with using(DIALECT):
        tokens, included = preprocess(with_extra_includes(source), file, directories)
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
    found.enums = enums_of(unit)
    return found


def _walk(decls: list[Stmt], scope: list[str], found: Declared, members: dict[str, Any]) -> None:
    for decl in decls:
        if isinstance(decl, Namespace):
            _namespace(decl, scope, found, members)
            continue
        for name, is_class in _names(decl):
            _record(found, members, scope, name, is_class)
        if isinstance(decl, ClassDecl):
            _nested(decl, [*scope, decl.name], found)


def _namespace(decl: Namespace, scope: list[str], found: Declared, members: Any) -> None:
    if decl.name is None:  # an unnamed namespace's names are the enclosing one's
        _walk(decl.body, scope, found, members)
        return
    inner = members.setdefault(decl.name, {})
    if not scope:
        _add(found.top, decl.name)
    _walk(decl.body, [*scope, decl.name], found, inner)


def _nested(owner: ClassDecl, scope: list[str], found: Declared) -> None:
    """A class's data members, and the classes, typedefs and enums declared inside it:
    ``Outer::Inner``, known to a rootmap and a LinkDef by that name."""
    found.members["::".join(scope)] = [
        each.name for member in owner.members if isinstance(member, DeclStmt)
        for each in member.decls
    ]  # fmt: skip
    for decl in owner.members:
        if isinstance(decl, (ClassDecl, Typedef, EnumDecl)) and decl.name:
            _add(found.classes, "::".join([*scope, decl.name]))
        if isinstance(decl, ClassDecl):
            _nested(decl, [*scope, decl.name], found)


def enums_of(unit: Unit) -> list[EnumNote]:
    """The enumerations ``unit`` declares, each once by its qualified name, in the order met.

    ROOT's interpreter knows an enumeration as soon as it has read the declaration - the line
    declaring it need not have run - so what a translated program declares is noted as it is
    parsed. This walk is the enumerations' alone, since every line a session runs is parsed
    again.
    """
    found = Declared()
    _enums(unit.decls, [], found, {})
    return found.enums


def _enums(decls: list[Any], scope: list[str], found: Declared, known: dict[str, int]) -> None:
    """Every enumeration ``decls`` declare, noting each with ``known`` - the constants of the
    enumerations declared before it, by name as C++ writes them - and adding its own."""
    for decl in decls:
        if isinstance(decl, Namespace):
            inside = scope if decl.name is None else [*scope, decl.name]
            _enums(decl.body, inside, found, known)
        elif isinstance(decl, EnumDecl):
            _note_enum(found, scope, decl.name, decl, known)
        elif isinstance(decl, ClassDecl) and decl.name:
            _enums(decl.members, [*scope, decl.name], found, known)
        elif isinstance(decl, DeclStmt):
            _enums(decl.decls, scope, found, known)


def _note_enum(found: Declared, scope: list[str], name: str | None, decl: EnumDecl,
               known: dict[str, int]) -> None:  # fmt: skip
    """Keep the enumeration ``name``, declared in ``scope``, as the interpreter knows it.

    An enumeration with no name of its own - ``enum {kA};`` - is one nothing looks up and so
    is not noted, and one already met - a header included again behind no guard - is kept
    once, as ROOT's table keeps one ``TEnum`` per enumeration. One none of whose constants
    this can work out is not noted either: ROOT's table would hold the constants, and an
    enumeration with none is a name a lookup finds but nothing about it is true.
    """
    if not name:
        return
    qualified = "::".join([*scope, name])
    if any(note.qualified == qualified for note in found.enums):
        return
    items = _valued(decl, known)
    if decl.items and not items:
        return
    underlying = decl.underlying.name if decl.underlying is not None else ""
    found.enums.append(EnumNote(qualified, items, decl.scoped, underlying))
    known.update(dict(items))
    known.update({f"{qualified}::{item}": value for item, value in items})


def _valued(decl: EnumDecl, known: dict[str, int]) -> list[tuple[str, int]]:
    """Each enumerator and its value, counted on from the last as C++ counts, as far as the
    declarations say: an enumerator given something only a compiler works out - a call, a
    value of a type nothing here declares - ends the list, since a wrong value in the table is
    worse than none.

    ``known`` holds the constants of the enumerations declared before this one, by the name
    C++ writes - ``A::kX`` - since an enumerator may be given another enumeration's, and the
    enumeration's own as they are counted.
    """
    values: list[tuple[str, int]] = []
    last = -1
    for name, value in decl.items:
        resolved = (last + 1 if value is None
                    else _number(value, known))  # fmt: skip
        if resolved is None:
            break
        known[name] = last = resolved
        values.append((name, resolved))
    return values


def _quotient(a: int, b: int) -> int:
    """C++'s integer division: truncated towards zero, which Python's ``//`` does not."""
    return abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)


#: The arithmetic and bit operations an enumerator's value may be written with, as C++'s
#: integers do them.
ENUMERATOR_OPERATIONS: dict[str, Any] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": _quotient,
    "%": lambda a, b: a - b * _quotient(a, b),
    "<<": lambda a, b: a << b,
    ">>": lambda a, b: a >> b,
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "^": lambda a, b: a ^ b,
}


def _number(value: Expr, known: dict[str, int]) -> int | None:
    """An enumerator's value when it is an integer this can work out: a number written out -
    ``3``, ``-1``, ``+81`` - ``true``, another enumeration's constant, ``sizeof`` of a type
    whose size is known, or arithmetic and bit operations on those.

    Anything else - a call, a float, a constant of a type this knows nothing of - is ``None``,
    which is where the list of values stops.
    """
    if isinstance(value, Literal):
        if value.kind == "bool":
            return int(bool(value.value))
        return int(value.value) if value.kind in ("int", "char") else None
    if isinstance(value, Unary):
        return _unary_number(value, known)
    if isinstance(value, SizeOf):
        return _size(value)
    if isinstance(value, Binary):
        return _binary_number(value, known)
    return _named_number(value, known)


#: The unary operations an enumerator's value may be written with, as C++'s integers do them:
#: ``~mask`` is the complement and ``!value`` an ``int``, which is how C++ types it.
UNARY_OPERATIONS: dict[str, Any] = {
    "-": lambda value: -value,
    "+": lambda value: value,
    "~": lambda value: -value - 1,
    "!": lambda value: int(not value),
}


def _unary_number(value: Unary, known: dict[str, int]) -> int | None:
    """``-kA``, ``+81``, ``~mask``, ``!value``: the operation, when its operand is known."""
    operation = UNARY_OPERATIONS.get(value.op)
    inner = _number(value.operand, known)
    if operation is None or inner is None:
        return None
    return int(operation(inner))


def _binary_number(value: Binary, known: dict[str, int]) -> int | None:
    """``1 << 14``, ``kA | kB``: C++'s integer operation, when both sides are known ones.

    One C++ would refuse outright - ``1 / 0`` - is no value either, and must not raise.
    """
    operation = ENUMERATOR_OPERATIONS.get(value.op)
    left, right = _number(value.left, known), _number(value.right, known)
    if operation is None or left is None or right is None:
        return None
    try:
        return int(operation(left, right))
    except (ArithmeticError, ValueError):
        return None


def _size(value: SizeOf) -> int | None:
    """``sizeof(int)`` or ``alignof(int)``, which are numbers C++ writes into an enumerator's
    value - for a fundamental type, whose size and alignment are the same number. A class's
    size is its own program's to know, and is no number here."""
    if value.ctype is None or value.operand is not None or value.ctype.pointer:
        return None
    found = SIZES.get(value.ctype.name)
    return None if found is None else found


def _named_number(value: Expr, known: dict[str, int]) -> int | None:
    """A constant of an enumeration declared before this one: ``kA``, ``A::kX``.

    ``known`` holds each of those both as C++ writes it there and by its own name, since C++
    finds an enumerator of an unscoped enumeration either way; a name of the program's that is
    no enumerator at all is none of them, and is where the list of values stops.
    """
    if not isinstance(value, Name):
        return None
    return known.get("::".join(value.parts))


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
