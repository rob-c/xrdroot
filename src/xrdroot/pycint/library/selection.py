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
from pathlib import Path

__all__ = ["requested", "linkdef_rules", "xml_rules"]

#: ``#pragma link C++ class Foo+;``: the kind of thing selected and its name.
LINK = re.compile(
    r"^\s*#\s*pragma\s+link\s+C\+\+\s+(?:options\s*=\s*\S+\s+)?"
    r"(?P<kind>class|struct|union|namespace|enum|typedef)\s+(?P<name>[^;]+?)\s*;",
    re.M,
)

#: What a LinkDef rule may end with after the name: streamer and I/O options.
OPTIONS = "+-!"


def linkdef_rules(text: str) -> list[str]:
    """The types a LinkDef's ``#pragma link C++`` lines select, by name, options taken off."""
    return [found["name"].rstrip(OPTIONS).strip() for found in LINK.finditer(text)]


#: What genreflex leaves out of a selection file: comments, and ``<exclusion>`` sections.
IGNORED = re.compile(r"<!--.*?-->|<exclusion\b.*?</exclusion\s*>", re.S)

#: An element naming a type: ``<class name="std::vector<TH1F>" />``. genreflex reads the
#: file leniently - a name keeps its ``<`` and ``>`` unescaped - and so does this.
ELEMENT = re.compile(
    r"<(?P<tag>class|struct|union|namespace|enum|typedef)\b(?P<attributes>(?:\s+[\w:-]+\s*=\s*"
    r'(?:"[^"]*"|\'[^\']*\'))*)\s*/?>'
)

#: An element's ``name`` attribute.
NAME = re.compile(r"""\bname\s*=\s*(?:"(?P<double>[^"]*)"|'(?P<single>[^']*)')""")


def xml_rules(text: str) -> list[str]:
    """The types a genreflex selection file selects by ``name``, outside ``<exclusion>``."""
    names: list[str] = []
    for element in ELEMENT.finditer(IGNORED.sub("", text)):
        found = NAME.search(element["attributes"])
        if found is not None:
            names.append(found["double"] if found["double"] is not None else found["single"])
    return names


def requested(linkdef: Path | None, selection: Path | None) -> list[str]:
    """Every type the LinkDef and the selection file ask for, in their order."""
    rules: list[str] = []
    if linkdef is not None:
        rules += linkdef_rules(linkdef.read_text(encoding="utf-8", errors="replace"))
    if selection is not None:
        rules += xml_rules(selection.read_text(encoding="utf-8", errors="replace"))
    return [rule for rule in rules if "*" not in rule]
