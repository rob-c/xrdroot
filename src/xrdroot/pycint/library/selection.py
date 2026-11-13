"""What a dictionary is asked to select: a LinkDef's ``#pragma link`` lines, a genreflex XML.

rootcling reads ``#pragma link C++ class Foo+;`` and genreflex ``<class
name="Foo"/>``; each names a class, struct, namespace, enum or typedef the
headers must declare. A rule naming nothing declared is not fatal to ROOT:
rootcling and genreflex both say ``Warning: Unused class rule: Foo`` and
build the dictionary anyway, so that is what is said here. Functions and
globals are not checked, as neither tool checks them; nor are patterns, the
standard library's classes, or ROOT's own.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ElementTree
from pathlib import Path

__all__ = ["requested", "linkdef_rules", "xml_rules"]

#: ``#pragma link C++ class Foo+;``: the kind of thing selected and its name.
LINK = re.compile(
    r"^\s*#\s*pragma\s+link\s+C\+\+\s+(?:options\s*=\s*\S+\s+)?"
    r"(?P<kind>class|struct|union|namespace|enum|typedef)\s+(?P<name>[^;]+?)\s*;",
    re.M,
)

#: The XML elements of a selection file that name a type to select.
SELECTING = frozenset({"class", "struct", "union", "namespace", "enum", "typedef"})

#: What a LinkDef rule may end with after the name: streamer and I/O options.
OPTIONS = "+-!"


def linkdef_rules(text: str) -> list[str]:
    """The types a LinkDef's ``#pragma link C++`` lines select, by name, options taken off."""
    return [found["name"].rstrip(OPTIONS).strip() for found in LINK.finditer(text)]


def xml_rules(text: str) -> list[str]:
    """The types a genreflex selection file selects by ``name``, outside ``<exclusion>``."""
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as why:
        raise ValueError(f"the selection file is not XML genreflex can read: {why}") from None
    names: list[str] = []
    for section in root.iter():
        if section.tag == "exclusion":
            continue
        names.extend(child.get("name", "") for child in section
                     if child.tag in SELECTING and child.get("name"))  # fmt: skip
    return names


def requested(linkdef: Path | None, selection: Path | None) -> list[str]:
    """Every type the LinkDef and the selection file ask for, in their order."""
    rules: list[str] = []
    if linkdef is not None:
        rules += linkdef_rules(linkdef.read_text(encoding="utf-8", errors="replace"))
    if selection is not None:
        rules += xml_rules(selection.read_text(encoding="utf-8", errors="replace"))
    return [rule for rule in rules if "*" not in rule]
