"""The XML documents ``TXMLEngine`` makes, reads and writes: elements, attributes, namespaces.

A node has a name, perhaps a namespace (a prefix and the reference it
stands for, declared on some node as ``xmlns:prefix="reference"``), its
attributes in the order they were given, perhaps text, and its children.
Written, a document is ``<?xml version="1.0"?>`` and its root, each child
two spaces further in, an element with only text on one line and one with
nothing in it closed at once (``<a x="1"/>``) - as ``TXMLEngine::SaveDoc``
lays one out. Read, a namespace declaration is the node's namespace, not
one of its attributes, and every prefixed node below it takes it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.dom import minidom
from xml.parsers.expat import ExpatError

__all__ = ["Node", "Namespace", "Document", "save", "parse"]


class Namespace:
    """A prefix and the reference it stands for."""

    def __init__(self, name: str, reference: str) -> None:
        self.name, self.reference = name, reference


class Node:
    """An element: its name, namespace, attributes, text and children."""

    def __init__(self, name: str, ns: Namespace | None = None, content: str | None = None) -> None:
        self.name, self.ns, self.content = name, ns, content
        self.attrs: list[tuple[str, str]] = []
        self.declared: list[Namespace] = []
        self.children: list[Node] = []
        self.parent: Node | None = None

    def add(self, child: Node, after: Node | None = None) -> None:
        """``child`` among this node's children: after ``after``, or at the end - moved there,
        if it was one of them already, as ``AddChildAfter`` relinks it."""
        if child in self.children:
            self.children.remove(child)
        child.parent = self
        at = self.children.index(after) + 1 if after in self.children else len(self.children)
        self.children.insert(at, child)

    def following(self) -> Node | None:
        """The next of its parent's children, or ``None``."""
        siblings = self.parent.children if self.parent is not None else []
        at = next((i for i, one in enumerate(siblings) if one is self), len(siblings))
        return siblings[at + 1] if at + 1 < len(siblings) else None

    def tag(self) -> str:
        """Its name as written: prefixed by its namespace - unless it declares it itself."""
        own = self.ns is None or any(ns is self.ns for ns in self.declared)
        return self.name if own else f"{self.ns.name}:{self.name}"  # type: ignore[union-attr]


class Document:
    """A document: its version and its root element."""

    def __init__(self, version: str = "1.0") -> None:
        self.version = version
        self.root: Node | None = None


def _written(node: Node, depth: int) -> list[str]:
    pad = "  " * depth
    attrs = [f' xmlns:{ns.name}="{ns.reference}"' for ns in node.declared]
    attrs += [f' {name}="{value}"' for name, value in node.attrs]
    head = f"{pad}<{node.tag()}{''.join(attrs)}"
    if not node.children:
        text = node.content
        return [f"{head}/>" if text is None else f"{head}>{text}</{node.tag()}>"]
    lines = [f"{head}>"]
    for child in node.children:
        lines += _written(child, depth + 1)
    return [*lines, f"{pad}</{node.tag()}>"]


def save(document: Document, path: Any) -> None:
    """``SaveDoc``: the document written as ``TXMLEngine`` lays one out."""
    lines = [f'<?xml version="{document.version}"?>']
    lines += _written(document.root, 0) if document.root is not None else []
    Path(path).write_text("\n".join(lines) + "\n")


def _node(element: Any, scope: dict[str, Namespace]) -> Node:
    """One element read, with the namespaces it and its ancestors declared."""
    scope = dict(scope)
    declared = [Namespace(name[6:], value) for name, value in element.attributes.items()
                if name.startswith("xmlns:")]  # fmt: skip
    scope.update({ns.name: ns for ns in declared})
    prefix, _, name = element.tagName.rpartition(":")
    node = Node(name, scope.get(prefix) if prefix else (declared[0] if declared else None))
    node.declared = declared
    node.attrs = [(k, v) for k, v in element.attributes.items() if not k.startswith("xmlns:")]
    texts = [c.data for c in element.childNodes if c.nodeType == c.TEXT_NODE and c.data.strip()]
    node.content = "".join(texts) if texts else None
    for child in element.childNodes:
        if child.nodeType == child.ELEMENT_NODE:
            node.add(_node(child, scope))
    return node


def parse(path: Any) -> Document | None:
    """``ParseFile``: the document in ``path``, or ``None`` for one that is not there or not XML."""
    try:
        read = minidom.parse(str(path))
    except (OSError, ExpatError):
        return None
    document = Document(read.version or "1.0")
    document.root = _node(read.documentElement, {})
    return document
