"""``TXMLEngine``: ROOT's small XML engine - nodes, attributes and namespaces by pointer.

Every ``XMLNodePointer_t``, ``XMLAttrPointer_t``, ``XMLNsPointer_t`` and
``XMLDocPointer_t`` is an object of :mod:`xrdroot.xmldoc`, and ``nullptr``
is ``None``, so a macro walks a document as it walks ROOT's: the first
child, the next sibling, the first attribute and the next. A document is
written as ``SaveDoc`` lays one out and read by Python's own parser.
"""

from __future__ import annotations

from typing import Any

from ..xmldoc import Document, Namespace, Node, parse, save
from .core.objects import TObject

__all__ = ["TXMLEngine"]


class _Attr:
    """One attribute of a node: which node, and which of its attributes."""

    def __init__(self, node: Node, index: int) -> None:
        self.node, self.index = node, index


def _attr(node: Node, index: int) -> _Attr | None:
    return _Attr(node, index) if index < len(node.attrs) else None


class TXMLEngine(TObject):
    """``TXMLEngine``: make, walk, change, read and write XML documents."""

    # -- documents -----------------------------------------------------------------------

    def NewDoc(self, version: str = "1.0") -> Document:
        return Document(str(version))

    def DocSetRootElement(self, doc: Document, node: Node) -> None:
        doc.root = node

    def DocGetRootElement(self, doc: Document) -> Node | None:
        return doc.root

    def SaveDoc(self, doc: Document, filename: str, layout: int = 1) -> None:
        save(doc, str(filename))

    def ParseFile(self, filename: str, maxbuf: int = 100000) -> Document | None:
        return parse(str(filename))

    def FreeDoc(self, doc: Any) -> None:
        """``FreeDoc``: Python frees it when nothing holds it."""

    def FreeNode(self, node: Any) -> None:
        self.UnlinkNode(node)

    # -- nodes ---------------------------------------------------------------------------

    def NewChild(self, parent: Node | None, ns: Namespace | None, name: str,
                 content: str | None = None) -> Node:  # fmt: skip
        node = Node(str(name), ns, None if content is None else str(content))
        if parent is not None:
            parent.add(node)
        return node

    def AddChild(self, parent: Node, child: Node) -> None:
        parent.add(child)

    def AddChildAfter(self, parent: Node, child: Node, afternode: Node | None) -> None:
        parent.add(child, afternode)

    def UnlinkNode(self, node: Node) -> None:
        if node.parent is not None:
            node.parent.children.remove(node)
            node.parent = None

    def GetChild(self, node: Node) -> Node | None:
        return node.children[0] if node.children else None

    def GetNext(self, node: Node) -> Node | None:
        return node.following()

    def GetParent(self, node: Node) -> Node | None:
        return node.parent

    def ShiftToNext(self, node: Any) -> None:
        """``ShiftToNext(xmlnode)``: the pointer moved to the next sibling, in the cell given."""
        node.value = node.value.following()

    def GetNodeName(self, node: Node) -> str:
        return node.name

    def GetNodeContent(self, node: Node) -> str | None:
        return node.content

    def SetNodeContent(self, node: Node, content: str) -> None:
        node.content = str(content)

    # -- namespaces ----------------------------------------------------------------------

    def NewNS(self, node: Node, reference: str, name: str | None = None) -> Namespace:
        """A namespace declared on ``node`` - named as the node is, unless told - and its own."""
        ns = Namespace(str(name) if name else node.name, str(reference))
        node.declared.append(ns)
        node.ns = ns
        return ns

    def GetNS(self, node: Node) -> Namespace | None:
        return node.ns

    def GetNSName(self, ns: Namespace) -> str:
        return ns.name

    def GetNSReference(self, ns: Namespace) -> str:
        return ns.reference

    # -- attributes ----------------------------------------------------------------------

    def NewAttr(self, node: Node, ns: Any, name: str, value: str) -> _Attr | None:
        node.attrs.append((str(name), str(value)))
        return _attr(node, len(node.attrs) - 1)

    def NewIntAttr(self, node: Node, name: str, value: int) -> _Attr | None:
        return self.NewAttr(node, None, name, str(int(value)))

    def GetFirstAttr(self, node: Node) -> _Attr | None:
        return _attr(node, 0)

    def GetNextAttr(self, attr: _Attr) -> _Attr | None:
        return _attr(attr.node, attr.index + 1)

    def GetAttrName(self, attr: _Attr) -> str:
        return attr.node.attrs[attr.index][0]

    def GetAttrValue(self, attr: _Attr) -> str:
        return attr.node.attrs[attr.index][1]

    def HasAttr(self, node: Node, name: str) -> bool:
        return any(key == name for key, _ in node.attrs)

    def GetAttr(self, node: Node, name: str) -> str | None:
        return next((value for key, value in node.attrs if key == name), None)

    def GetIntAttr(self, node: Node, name: str) -> int:
        found = self.GetAttr(node, name)
        return int(found) if found is not None else 0
