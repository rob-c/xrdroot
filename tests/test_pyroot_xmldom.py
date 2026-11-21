"""``TDOMParser`` and the nodes it reads a document into, as ROOT's tutorials walk them."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT

XML = """<?xml version="1.0"?>
<family>
  <!-- one of each -->
  <person id="3" city="Oslo">Ada<note>first</note></person>
  <person id="4">Bob<bare><inner>x</inner></bare></person>
</family>
"""


@pytest.fixture
def document(tmp_path):
    path = tmp_path / "people.xml"
    path.write_text(XML)
    parser = ROOT.TDOMParser()
    assert parser.ParseFile(path) == 0
    return parser.GetXMLDocument()


def children(node):
    """Every node under ``node``, as a macro walks them: the first child, then its siblings."""
    out = []
    child = node.GetChildren()
    while child is not None:
        out.append(child)
        child = child.GetNextNode()
    return out


def elements(node):
    return [one for one in children(node) if one.GetNodeType() == ROOT.TXMLNode.kXMLElementNode]


def root_text(document):
    """The root element's own text, which is the whitespace written before its first child."""
    return document.GetRootNode().GetText()


def iterated(collection):
    """The objects of a collection, in ROOT's own order."""
    out = []
    it = ROOT.TIter(collection)
    while (found := it.Next()) is not None:
        out.append(found)
    return out


def test_a_document_is_a_tree_of_elements_text_and_comments(document):
    """A node has a kind, a name for an element and content for anything else, and the
    children and next siblings a macro walks with ``GetChildren``/``GetNextNode``."""
    root = document.GetRootNode()
    assert (root.GetNodeName(), root.GetNodeType()) == ("family", ROOT.TXMLNode.kXMLElementNode)
    kinds = [one.GetNodeType() for one in children(root)]
    assert kinds == [
        ROOT.TXMLNode.kXMLTextNode,
        ROOT.TXMLNode.kXMLCommentNode,
        ROOT.TXMLNode.kXMLTextNode,
        ROOT.TXMLNode.kXMLElementNode,
        ROOT.TXMLNode.kXMLTextNode,
        ROOT.TXMLNode.kXMLElementNode,
        ROOT.TXMLNode.kXMLTextNode,
    ]
    comment = children(root)[1]
    assert (comment.GetNodeName(), comment.GetContent()) == (None, " one of each ")
    assert repr(comment) == "<TXMLNode None>"
    assert (comment.HasAttributes(), len(comment.GetAttributes())) == (False, 0)
    assert root.GetNextNode() is None  # the root is nobody's sibling
    first, second = elements(root)
    assert (first.GetParent() is root, second.GetParent() is root) == (True, True)
    assert first.GetNextNode().GetNodeType() == ROOT.TXMLNode.kXMLTextNode


def test_an_elements_text_and_attributes_are_what_the_file_wrote(document):
    """``GetText`` is the element's own text - its ``note`` child's is not - and the
    attributes come back as a ``TList`` of ``TXMLAttr`` in the order written."""
    first, second = elements(document.GetRootNode())
    assert first.GetText() == "Ada"
    assert first.GetAttribute("city").GetValue() == "Oslo"
    assert first.GetAttribute("missing") is None
    (ident, city) = iterated(first.GetAttributes())
    assert (ident.GetName(), ident.GetValue(), city.GetName()) == ("id", "3", "city")
    note = elements(first)[0]
    assert (note.GetNodeName(), note.GetText(), note.HasAttributes()) == ("note", "first", False)
    assert (first.GetContent(), root_text(document)) == (None, "\n  ")
    assert (second.GetText(), second.HasAttributes()) == ("Bob", True)
    assert len(second.GetAttributes()) == 1
    bare = elements(second)[0]  # an element with no text of its own
    assert (bare.GetText(), elements(bare)[0].GetText()) == ("", "x")


def test_a_file_that_is_not_xml_says_so_and_carries_no_document(tmp_path):
    path = tmp_path / "broken.xml"
    path.write_text("this is not xml at all <<<")
    parser = ROOT.TDOMParser()
    parser.SetValidate(False)
    parser.SetNamespaceMode(0)
    parser.SetStopOnError(True)
    assert parser.ParseFile(path) == ROOT.TDOMParser.kParseError
    assert parser.GetXMLDocument() is None
    said = parser.GetParseCodeMessage(ROOT.TDOMParser.kParseError)
    assert said == "The document is not valid XML"
    assert parser.GetParseCodeMessage(0) == "Parsing OK"
    assert parser.ParseFile(tmp_path / "missing.xml") == ROOT.TDOMParser.kParseError


def test_a_root_element_let_go_of_leaves_the_document_without_one(document):
    assert document.GetRootNode() is not None
    document.ReleaseRootNode()
    assert document.GetRootNode() is None


class Hearer:
    """A SAX handler that writes down what it hears, as a macro's own handler would."""

    def __init__(self, skip=()):
        self.heard = []
        self._skip = skip

    def __getattr__(self, name):
        if name in self._skip:
            raise AttributeError(name)
        return lambda *words: self.heard.append((name, *words))


def test_a_sax_parser_tells_its_handler_what_the_document_does(tmp_path):
    """``TSAXParser`` reports each event of the document to the handler it was connected to:
    an element with its attributes in the order written, the text between elements, a
    comment, and a CDATA block as one block."""
    path = tmp_path / "events.xml"
    path.write_text(
        '<?xml version="1.0"?>\n<root>\n<!-- said -->\n<one id="7" city="Oslo">text</one>'
        "<two><![CDATA[raw <text>]]><?pi value?></two></root>\n"
    )
    heard = Hearer()
    parser = ROOT.TSAXParser()
    parser.ConnectToHandler("Hearer", heard)
    assert parser.ParseFile(path) == 0
    events = [(name, words) for name, *words in heard.heard]
    assert [name for name, _ in events] == [
        "OnStartDocument",
        "OnStartElement",
        "OnCharacters",
        "OnComment",
        "OnCharacters",
        "OnStartElement",
        "OnCharacters",
        "OnEndElement",
        "OnStartElement",
        "OnCdataBlock",
        "OnEndElement",
        "OnEndElement",
        "OnEndDocument",
    ]
    root = events[1][1]
    assert (root[0], list(iterated(root[1]))) == ("root", [])  # written without attributes
    one = events[5][1]
    assert [(attr.GetName(), attr.GetValue()) for attr in iterated(one[1])] == [
        ("id", "7"),
        ("city", "Oslo"),
    ]
    assert events[6][1] == ["text"]
    assert events[9][1] == ["raw <text>", 10]


def test_a_sax_parser_on_a_file_it_cannot_read_reports_it_and_says_nothing(tmp_path):
    """A file that is not there is not read and the handler hears none of it - ROOT's own
    tutorials hand this parser a path that may not exist - while a document that stops being
    readable halfway is reported to the handler as the fatal error it is."""
    heard = Hearer()
    parser = ROOT.TSAXParser()
    parser.ConnectToHandler("Hearer", heard)
    assert parser.ParseFile(tmp_path / "missing.xml") == -1
    assert heard.heard == []
    broken = tmp_path / "broken.xml"
    broken.write_text("<root><unclosed></root>")
    assert parser.ParseFile(broken) == -1
    assert [name for name, *_ in heard.heard][-1] == "OnFatalError"


def test_a_handler_that_does_not_write_a_method_down_is_told_nothing(tmp_path):
    """A macro's handler is free to leave a method out, and then nothing happens for it."""
    path = tmp_path / "quiet.xml"
    path.write_text("<root><!-- said -->text</root>")
    quiet = Hearer(skip=("OnStartDocument", "OnEndDocument", "OnComment"))
    parser = ROOT.TSAXParser()
    parser.ConnectToHandler("Hearer", quiet)
    assert parser.ParseFile(path) == 0
    assert [name for name, *_ in quiet.heard] == [
        "OnStartElement",
        "OnCharacters",
        "OnEndElement",
    ]
