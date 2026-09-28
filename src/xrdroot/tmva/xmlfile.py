"""The XML of TMVA's weight files: written as TMVA lays it out, read with the standard library.

A weight file is ``<MethodSetup Method="BDT::BDT">`` holding the general
information, the options, the variables, spectators, classes and targets,
the transformations, the output densities and the method's own
``<Weights>``. :class:`Node` writes that the way ``TXMLEngine`` does - two
spaces of indent, attributes in the order given, ``<Tag/>`` when empty - so
a file written here reads like TMVA's own, and ElementTree reads either.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any
from xml.sax.saxutils import escape, quoteattr

__all__ = ["Node", "child", "children", "floats", "load", "number"]


def number(value: Any, digits: int = 16) -> str:
    """A number as ``gTools().AddAttr`` writes it: scientific, ``digits`` after the point."""
    return format(float(value), f".{digits}e")


class Node:
    """An element being written: a tag, its attributes in order, its children and its text."""

    def __init__(self, tag: str, **attributes: Any) -> None:
        self.tag = tag
        self.attributes: list[tuple[str, str]] = [
            (key.rstrip("_"), str(value)) for key, value in attributes.items()
        ]
        self.children: list[Node] = []
        self.text: str | None = None

    def set(self, key: str, value: Any) -> Node:
        self.attributes.append((key, str(value)))
        return self

    def add(self, tag: str, **attributes: Any) -> Node:
        """A child, appended and handed back to be filled."""
        made = Node(tag, **attributes)
        self.children.append(made)
        return made

    def lines(self, depth: int = 0) -> list[str]:
        """The element as ``TXMLEngine`` lays it out."""
        pad = "  " * depth
        attributes = "".join(f" {key}={quoteattr(value)}" for key, value in self.attributes)
        if not self.children and self.text is None:
            return [f"{pad}<{self.tag}{attributes}/>"]
        if not self.children:
            return [f"{pad}<{self.tag}{attributes}>{escape(self.text or '')}</{self.tag}>"]
        inner = [line for node in self.children for line in node.lines(depth + 1)]
        return [f"{pad}<{self.tag}{attributes}>", *inner, f"{pad}</{self.tag}>"]

    def block(self, values: Any, digits: int = 8, depth_hint: int = 0) -> Node:
        """Numbers as text - a histogram's bins, a matrix - as TMVA writes them, space-separated."""
        self.text = "\n" + " ".join(number(value, digits) for value in values) + " \n"
        return self

    def write(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as out:
            out.write('<?xml version="1.0"?>\n')
            out.write("\n".join(self.lines()) + "\n")


def load(path: str) -> ET.Element:
    """The root element of a weight file."""
    return ET.parse(path).getroot()


def child(node: ET.Element, tag: str) -> ET.Element | None:
    return node.find(tag)


def children(node: ET.Element, tag: str | None = None) -> list[ET.Element]:
    return [item for item in node if tag is None or item.tag == tag]


def floats(node: ET.Element) -> list[float]:
    """The numbers of an element's text, as ``std::istream`` reads them one after another."""
    return [float(token) for token in (node.text or "").split()]
