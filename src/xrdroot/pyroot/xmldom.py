"""``TDOMParser`` and the nodes it reads a document into: ROOT's older DOM view.

A document is read into nodes of three kinds - an element, the text between
elements, and a comment - each with its children and next sibling, an element's
attributes as a ``TList`` of ``TXMLAttr``, and the text of an element asked for
by :meth:`TXMLNode.GetText`. That is how ROOT's own tutorials walk a document -
``ParseContext(node->GetChildren())``, printing each node by its type - so what
they print is what this reads.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast
from xml.dom import Node as DomNode
from xml.dom import minidom
from xml.parsers import expat
from xml.parsers.expat import ExpatError

from .core.collections import TList
from .core.objects import TNamed, TObject

__all__ = ["TDOMParser", "TSAXParser", "TXMLAttr", "TXMLDocument", "TXMLNode"]


class TXMLAttr(TNamed):
    """One attribute of an element: its name, and the value written for it."""

    def __init__(self, name: str, value: str) -> None:
        super().__init__(name, value)

    def GetValue(self) -> str:
        """``GetValue``: what the document wrote for this attribute."""
        return TNamed.GetTitle(self)


class TXMLNode(TObject):
    """``TXMLNode``: one node of a document - an element, its text, or a comment."""

    #: ``TXMLNode::ENodeType``: an element, text, a comment, a CDATA block, a PI.
    kXMLElementNode, kXMLTextNode, kXMLCommentNode = 1, 2, 3
    kXMLCdataNode, kXMLPDataType = 4, 5

    #: Which of the document's kinds of node each of ROOT's types is.
    KINDS: ClassVar[dict[int, int]] = {
        DomNode.ELEMENT_NODE: kXMLElementNode,
        DomNode.TEXT_NODE: kXMLTextNode,
        DomNode.CDATA_SECTION_NODE: kXMLCdataNode,
        DomNode.COMMENT_NODE: kXMLCommentNode,
        DomNode.PROCESSING_INSTRUCTION_NODE: kXMLPDataType,
    }

    def __init__(
        self,
        node: Any,
        parent: TXMLNode | None = None,
        children: list[TXMLNode] | None = None,
    ) -> None:
        super().__init__()
        self._node, self._parent = node, parent
        self._children = children if children is not None else []
        self._attributes: TList | None = None

    def __repr__(self) -> str:
        return f"<TXMLNode {self.GetNodeName()!r}>"

    def GetNodeType(self) -> int:
        """``GetNodeType``: which of the five kinds of node this one is."""
        return TXMLNode.KINDS.get(self._node.nodeType, TXMLNode.kXMLElementNode)

    def GetNodeName(self) -> str | None:
        """``GetNodeName``: the tag of an element as the document wrote it - the prefix it
        was given included - and ``None`` for a node that holds text rather than a name."""
        if self.GetNodeType() != TXMLNode.kXMLElementNode:
            return None
        return str(self._node.tagName)

    def GetContent(self) -> str | None:
        """``GetContent``: what a text or comment node holds, and ``None`` for an element."""
        if self.GetNodeType() == TXMLNode.kXMLElementNode:
            return None
        return str(self._node.data)

    def GetText(self) -> str:
        """``GetText``: the text an element holds - its first text child's, as ROOT reads one."""
        for child in self._children:
            if child.GetNodeType() == TXMLNode.kXMLTextNode:
                return str(child._node.data)
        return ""

    def GetChildren(self) -> TXMLNode | None:
        """``GetChildren``: the first child of an element, text included, or ``None``."""
        return self._children[0] if self._children else None

    def GetNextNode(self) -> TXMLNode | None:
        """``GetNextNode``: the next node of the parent, or of the document."""
        if self._parent is None:
            return None
        siblings = self._parent._children
        at = next((index for index, one in enumerate(siblings) if one is self), -1)
        return siblings[at + 1] if 0 <= at < len(siblings) - 1 else None

    def GetParent(self) -> TXMLNode | None:
        """``GetParent``: the element this one is written in."""
        return self._parent

    def HasAttributes(self) -> bool:
        """``HasAttributes``: was this element written with any?"""
        return self.GetNodeType() == TXMLNode.kXMLElementNode and bool(self._node.attributes)

    def GetAttributes(self) -> TList:
        """``GetAttributes``: its attributes, in the order the document wrote them."""
        if self._attributes is None:
            self._attributes = TList()
            if self.GetNodeType() == TXMLNode.kXMLElementNode:
                for at in range(self._node.attributes.length):
                    attr = self._node.attributes.item(at)
                    self._attributes.Add(TXMLAttr(str(attr.name), str(attr.value)))
        return self._attributes

    def GetAttribute(self, name: str) -> TXMLAttr | None:
        """``GetAttribute(name)``: the attribute ``name``, or ``None``."""
        for attr in self.GetAttributes():
            if attr.GetName() == name:
                return cast(TXMLAttr, attr)
        return None


class TXMLDocument(TObject):
    """``TXMLDocument``: a document read, and the root element it starts at."""

    def __init__(self, root: TXMLNode | None) -> None:
        super().__init__()
        self._root = root

    def GetRootNode(self) -> TXMLNode | None:
        """``GetRootNode``: the document's root element."""
        return self._root

    def ReleaseRootNode(self) -> None:
        """``ReleaseRootNode``: the root no longer belongs to the document."""
        self._root = None


def _context(node: Any, parent: TXMLNode | None) -> TXMLNode:
    """``node`` and everything under it as the nodes a macro walks."""
    made = TXMLNode(node, parent)
    made._children = [
        _context(child, made)
        for child in node.childNodes
        if child.nodeType in TXMLNode.KINDS
    ]
    return made


class TDOMParser(TObject):
    """``TDOMParser``: read an XML file into a document a macro walks node by node."""

    #: What ``ParseFile`` answers for a file it could not read: ROOT's own codes are the
    #: parser's, and this is the one this reader has.
    kParseError = -1

    def __init__(self) -> None:
        super().__init__()
        self._document: TXMLDocument | None = None
        self._code = 0

    def SetValidate(self, validate: bool = True) -> None:
        """``SetValidate``: whether the document is held to the DTD it names. This reader does
        not validate against DTDs, so a document is read the same way either way."""
        self._validate = bool(validate)

    def SetNamespaceMode(self, mode: int = 0, *rest: Any) -> None:
        """``SetNamespaceMode``: how namespaces are reported. Every node here keeps the name
        the document wrote, prefix and all, whichever mode a macro asks for."""
        self._namespaces = int(mode)

    def SetStopOnError(self, stop: bool = True) -> None:
        """``SetStopOnError``: whether reading stops at the first error. A document either
        reads or it does not, so this is kept and nothing else."""
        self._stop = bool(stop)

    def ParseFile(self, path: Any) -> int:
        """``ParseFile``: the file read, ``0`` - or a negative code for one that was not."""
        try:
            read = minidom.parse(str(path))
        except (OSError, ExpatError, ValueError):
            self._document, self._code = None, TDOMParser.kParseError
            return self._code
        self._document = TXMLDocument(_context(read.documentElement, None))
        self._code = 0
        return 0

    def GetXMLDocument(self) -> TXMLDocument | None:
        """``GetXMLDocument``: the document read, or ``None`` if the read failed."""
        return self._document

    def GetParseCodeMessage(self, code: int) -> str:
        """``GetParseCodeMessage``: what a code means, in words."""
        if code == 0:
            return "Parsing OK"
        return "The document is not valid XML"


class _Handler:
    """What the document's events are passed to: the macro's own handler object, through the
    methods ROOT's ``TSAXParser`` calls on it. A handler that does not define one is left
    alone, as a macro is free not to write down every method ROOT may call."""

    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self._in_cdata: str | None = None

    def _say(self, name: str, *words: Any) -> None:
        told = getattr(self._handler, name, None)
        if told is not None:
            told(*words)

    def start_document(self) -> None:
        self._say("OnStartDocument")

    def end_document(self) -> None:
        self._say("OnEndDocument")

    def start_element(self, name: str, attrs: list[str]) -> None:
        written = TList()
        for at in range(0, len(attrs), 2):
            written.Add(TXMLAttr(attrs[at], attrs[at + 1]))
        self._say("OnStartElement", name, written)

    def end_element(self, name: str) -> None:
        self._say("OnEndElement", name)

    def characters(self, content: str) -> None:
        if self._in_cdata is None:
            self._say("OnCharacters", content)
        else:  # a CDATA block is told as one block, not character run by character run
            self._in_cdata += content

    def start_cdata(self) -> None:
        self._in_cdata = ""

    def end_cdata(self) -> None:
        text = self._in_cdata or ""
        self._in_cdata = None
        self._say("OnCdataBlock", text, len(text))

    def comment(self, text: str) -> None:
        self._say("OnComment", text)

    def fatal_error(self, text: str) -> None:
        self._say("OnFatalError", text)


class TSAXParser(TObject):
    """``TSAXParser``: read an XML file by telling a macro's handler what happens in it."""

    def __init__(self) -> None:
        super().__init__()
        self._handler: Any = None
        self._code = 0

    def ConnectToHandler(self, classname: str, handler: Any) -> None:
        """``ConnectToHandler``: ``handler`` hears what ``ParseFile`` reads next. ROOT asks
        ``classname`` for the type a handler is, to cast it; the object it is given is
        already the macro's own, so only the object matters here."""
        self._handler = handler

    def _reader(self) -> tuple[Any, _Handler]:
        """A reader that reports comments and CDATA blocks too, in document order, as the
        SAX reader ROOT calls on libxml2 does, and the bridge that passes them on."""
        reader = expat.ParserCreate()
        reader.ordered_attributes = True
        told = _Handler(self._handler)
        reader.StartElementHandler = told.start_element
        reader.EndElementHandler = told.end_element
        reader.CharacterDataHandler = told.characters
        reader.CommentHandler = told.comment
        reader.StartCdataSectionHandler = told.start_cdata
        reader.EndCdataSectionHandler = told.end_cdata
        return reader, told

    def ParseFile(self, path: Any) -> int:
        """``ParseFile``: the file read, ``0`` - or ``-1`` for one that was not, the handler
        told of a document that stopped being readable halfway."""
        reader, told = self._reader()
        try:
            opened = open(str(path), "rb")
        except OSError:
            self._code = -1
            return self._code
        with opened:
            told.start_document()
            try:
                reader.ParseFile(opened)
            except expat.ExpatError as bad:
                told.fatal_error(str(bad))
                self._code = -1
                return self._code
        told.end_document()
        self._code = 0
        return 0
