"""What rootcling and genreflex refuse to make a dictionary of, refused as they refuse it.

A dictionary built here is a translation, but the generators' own checks of
what they are asked to select are not about C++, and a build that ROOT's
tools stop must stop here too, with their words and their exit status:

- a selection file whose ``noStreamer`` or ``noInputOperator`` is not
  ``true`` or ``false`` (``Error: XML at line N: ...``, status 1);
- two ``<class>`` rules for one name or pattern
  (``SelectionRules::CheckDuplicates``, each pair shown, status 1);
- a header given to genreflex - which inlines it - with ``#pragma once``;
- a standard class whose I/O ROOT does not support, or need not be asked for
  (``std::thread``, ``std::ratio``, ``std::chrono::duration``,
  ``std::array``), and a union a LinkDef selects: one ``Error:`` line each,
  the status their count;
- rootcling run with nothing to select by - no LinkDef, no selection file,
  no ``#pragma link`` - when asked to behave so (:func:`unselected`).

The warnings - a rule naming nothing declared, an I/O rule's target that is
not a member of its class - are said as both tools say them, and fail the
build only under genreflex's ``--fail_on_warnings``.
"""

from __future__ import annotations

import fnmatch
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ...pycxx.errors import CxxError

__all__ = ["GeneratorError", "Rule", "rules", "before", "unselected", "io_targets", "said"]

#: A start, end or empty tag, its name, its attributes, and whether it closes itself.
TAG = re.compile(
    r"<(?P<end>/?)(?P<name>[\w:-]+)(?P<attributes>(?:\s+[\w:-]+\s*=\s*(?:\"[^\"]*\"|'[^']*'))*)"
    r"\s*(?P<empty>/?)>"
)

#: One attribute of a tag.
ATTRIBUTE = re.compile(r"""([\w:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")

#: What a selection file holds that is no rule: comments and character data.
SKIPPED = re.compile(r"<!--.*?-->|<!\[CDATA\[.*?\]\]>", re.S)

#: The attributes of a class rule that take ``true`` or ``false`` alone.
BOOLEANS = ("noStreamer", "noInputOperator")

#: The standard classes whose I/O ROOT does not support, as genreflex names them, in the
#: order it meets them: that of their headers, ``ratio`` declared before ``duration``.
UNSUPPORTED = ("ratio", "chrono::duration", "thread")

#: The standard classes whose I/O needs no selecting.
TRANSPARENT = ("array",)

#: The order genreflex says each standard class's error in.
ORDER = {name: rank for rank, name in enumerate((*TRANSPARENT, *UNSUPPORTED))}


class GeneratorError(CxxError):
    """What a dictionary generator stopped at, said already: ``status`` is its exit status."""

    def __init__(self, status: int) -> None:
        super().__init__("the dictionary generator stopped at what it said")
        self.status = status


@dataclass
class Rule:
    """A ``<class>`` rule: its line, its attributes, and its ``<field>`` rules' attributes."""

    line: int
    attributes: dict[str, str]
    fields: list[dict[str, str]] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str] | None:
        """What it selects by: its ``name`` or its ``pattern``."""
        for kind in ("name", "pattern"):
            if kind in self.attributes:
                return kind, self.attributes[kind]
        return None


@dataclass
class Parsed:
    """A selection file's class rules and I/O rules, in order."""

    classes: list[Rule] = field(default_factory=list)
    io: list[dict[str, str]] = field(default_factory=list)


def _blanked(found: re.Match[str]) -> str:
    """Skipped text, its newlines kept so that every rule after it keeps its line."""
    return "\n" * found.group().count("\n")


def rules(text: str) -> Parsed:
    """The rules of a selection file, read as leniently as genreflex reads it."""
    text = SKIPPED.sub(_blanked, text)
    parsed: Parsed = Parsed()
    current: Rule | None = None
    for tag in TAG.finditer(text):
        if tag["end"]:
            current = None if tag["name"] == "class" else current
        else:
            current = _tag(parsed, current, tag, text.count("\n", 0, tag.start()) + 1)
    return parsed


def _tag(parsed: Parsed, current: Rule | None, tag: re.Match[str], line: int) -> Rule | None:
    """One start tag of a selection file read into ``parsed``; the class rule open after it."""
    attributes = {key: one or two for key, one, two in ATTRIBUTE.findall(tag["attributes"])}
    name = tag["name"]
    if name == "class":
        rule = Rule(line, attributes)
        parsed.classes.append(rule)
        return None if tag["empty"] else rule
    if name == "field" and current is not None:
        current.fields.append(attributes)
    elif name in ("ioread", "read"):
        parsed.io.append(attributes)
    return current


def said(lines: Sequence[str]) -> None:
    """Lines on standard error, where both tools say what is wrong."""
    for line in lines:
        sys.stderr.write(line + "\n")


def before(
    files: Sequence[Path], linkdef: Path | None, selection: Path | None, genreflex: bool,
    fatal_duplicates: bool = True,
) -> Parsed:
    """Refuse what the generator would refuse before it compiles a line; the selection's
    rules, for the checks after. Only a generator that goes on to write a dictionary refuses
    the duplicates it finds, so ``--selSyntaxOnly`` passes ``fatal_duplicates=False``."""
    parsed = rules(_text(selection)) if selection is not None else Parsed()
    _booleans(parsed)
    _duplicates(parsed, fatal_duplicates)
    errors = _pragma_once(files) if genreflex else []
    errors += _standard([rule.key[1] for rule in parsed.classes if rule.key is not None
                         and rule.key[0] == "name"])  # fmt: skip
    errors += _unions(_text(linkdef)) if linkdef is not None else []
    if errors:
        said(errors)
        raise GeneratorError(len(errors))
    return parsed


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _booleans(parsed: Parsed) -> None:
    for rule in parsed.classes:
        for name in BOOLEANS:
            value = rule.attributes.get(name)
            if value is not None and value not in ("true", "false"):
                said([f"Error: XML at line {rule.line}: class attribute '{name}' must be 'true' "
                      f"or 'false' (it was {value})"])  # fmt: skip
                raise GeneratorError(1)


def _duplicates(parsed: Parsed, fatal: bool = True) -> None:
    """``SelectionRules::CheckDuplicates``: each pair of rules for one name or pattern, the
    last first. Identical ones are only warned about, while two that differ stop the build -
    unless ``fatal`` is off, as genreflex's ``--selSyntaxOnly`` reads a selection."""
    pairs = _pairs_of(parsed.classes)
    conflicting = fatal and any(_differ(rule, later) for rule, later in pairs)
    shown = [line for rule, later in pairs for line in _pair(rule, later)]
    if not conflicting:
        said(shown)
        return
    said([*shown, "Error in <SelectionRules::CheckDuplicates>: Duplicates in rules were found."])
    raise GeneratorError(1)


def _pairs_of(classes: list[Rule]) -> list[tuple[Rule, Rule]]:
    """Every pair of rules naming one key, the later rule first - the pairs a duplicate is."""
    found = [(rule, later) for index, rule in enumerate(classes)
             for later in classes[index + 1:]
             if rule.key is not None and rule.key == later.key]  # fmt: skip
    return found[::-1]


def _differ(rule: Rule, later: Rule) -> bool:
    """Whether two rules select differently: their ``ClassVersion`` aside, which genreflex
    keeps apart."""
    return (_selecting(rule), rule.fields) != (_selecting(later), later.fields)


def _selecting(rule: Rule) -> dict[str, str]:
    return {key: value for key, value in rule.attributes.items() if key != "ClassVersion"}


def _pair(rule: Rule, later: Rule) -> list[str]:
    """Two rules for one name, as the warning shows them."""
    stored = "Conflicting" if _differ(rule, later) else "Identical"
    return ["Warning in <SelectionRules::CheckDuplicates>: Duplicated rule found.", "Rule:",
            *_shown(rule), f"{stored} rule already stored:", *_shown(later)]  # fmt: skip


def _shown(rule: Rule) -> list[str]:
    """A rule as ``SelectionRules`` prints it: its line, its attributes, its field rules."""
    shown = [f"\t\tSelected (line {rule.line}): Yes", "\t\tAttributes: "]
    shown += [f"\t\t{key} = {value}" for key, value in sorted(_selecting(rule).items())]
    if not rule.fields:
        shown.append("\t\tNo field sel rules")
    for index, attributes in enumerate(rule.fields):
        shown += [f"\t\tField {index}:", "\t\tSelected: Yes"]
        shown += [f"\t\t\t{key} = {value}" for key, value in sorted(attributes.items())]
    return [*shown, "\t\tNo method sel rules"]


def _pragma_once(files: Sequence[Path]) -> list[str]:
    """genreflex inlines the headers it is given, and refuses one that says ``#pragma once``."""
    once = re.compile(r"^[ \t]*#[ \t]*pragma[ \t]+once\b", re.M)
    return [f"Error: #pragma once directive detected in header file {path.name} which was "
            "requested to be inlined." for path in files if once.search(_text(path))]  # fmt: skip


def _normalised(name: str) -> str:
    """A standard class's name as genreflex writes it: no ``std::``, no spaces after commas,
    ``> >``, a ``duration``'s default period put in."""
    text = re.sub(r"\bstd::", "", name).replace(", ", ",").strip()
    duration = re.fullmatch(r"chrono::duration<([^,<>]+)>", text)
    if duration is not None:
        text = f"chrono::duration<{duration[1]},ratio<1,1>>"
    return text.replace(">>", "> >")


def _standard(names: Sequence[str]) -> list[str]:
    errors = []
    standard = [name for name in names if name.strip().startswith("std::")]
    for name in sorted(standard, key=lambda name: ORDER.get(_normalised(name).split("<")[0], 0)):
        shown = _normalised(name)
        bare = shown.split("<", 1)[0]
        if bare in TRANSPARENT:
            errors.append(f"Error: It is not necessary to explicitly select class {shown}. I/O "
                          "is supported for it transparently.")  # fmt: skip
        elif bare in UNSUPPORTED:
            errors.append(f"Error: Class {shown} has been selected but currently the support "
                          f"for its I/O is not yet available. Note that {shown}, even if not "
                          "selected, will be available for interpreted code.")  # fmt: skip
    return errors


#: ``#pragma link C++ union Name;``: a union a LinkDef selects.
UNION = re.compile(r"^\s*#\s*pragma\s+link\s+C\+\+\s+union\s+(?P<name>[^;]+?)[-+!]*\s*;", re.M)


def _unions(text: str) -> list[str]:
    return [f"Error: Union {found['name'].strip()} has been selected for I/O. This is not "
            "supported. Interactive usage of unions is supported, as all C++ entities, without "
            "the need of dictionaries." for found in UNION.finditer(text)]  # fmt: skip


def unselected(files: Sequence[Path], linkdef: Path | None, selection: Path | None) -> None:
    """rootcling given nothing to select by - no LinkDef, no selection file, no ``#pragma link``
    in the headers - and no C++ module to make, which it refuses."""
    if linkdef is not None or selection is not None:
        return
    if any(re.search(r"#\s*pragma\s+link\b", _text(path)) for path in files):
        return
    said(["Error: No selection rules specified and creation of C++ module not requested: did "
          "you forget to specify a selection file or to request the creation of a C++ "
          "module?"])  # fmt: skip
    raise GeneratorError(1)


def io_targets(parsed: Parsed, members: dict[str, list[str]]) -> list[str]:
    """The warning for each I/O rule whose ``target`` - taken whole, as genreflex takes it -
    is no data member of its target class, when that class is declared here; and for the
    rules of each target class nothing selects, which go unused."""
    warnings = []
    unused: dict[str, int] = {}
    for rule in parsed.io:
        owner = rule.get("targetClass", "")
        if owner and not _selects(parsed, owner):
            unused[owner] = unused.get(owner, 0) + 1
        else:
            warnings += _missing_target(rule, owner, members)
    return warnings + [f"Warning: {count} rule{'s' if count > 1 else ''} for target class "
                       f"{owner} {'were' if count > 1 else 'was'} not used!"
                       for owner, count in unused.items()]  # fmt: skip


def _missing_target(rule: dict[str, str], owner: str, members: dict[str, list[str]]) -> list[str]:
    target = rule.get("target", "").strip()
    known = members.get(owner.removeprefix("::"))
    if not target or known is None or target in known:
        return []
    return [f"Warning: IO rule for class {owner} data member: {target} was specified as a "
            "target in the rule but doesn't seem to appear in target class"]  # fmt: skip


def _selects(parsed: Parsed, name: str) -> bool:
    """Whether a class rule selects the class ``name``: by its name, or by a pattern."""
    for rule in parsed.classes:
        key = rule.key
        if key is not None and (key[1] == name or (key[0] == "pattern"
                                and fnmatch.fnmatchcase(name, key[1]))):  # fmt: skip
            return True
    return False
