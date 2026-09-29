"""``TXMLEngine``: XML made, walked, changed, written and read as ROOT's engine does.

The documents are those of ROOT's ``xmlnewfile.C`` and ``xmlmodifyfile.C``;
what they write is what ROOT 6.40 wrote for them, byte for byte.
"""

from __future__ import annotations

from typing import Any

import xrdroot.pyroot as ROOT
from geomsupport import geometry_session  # noqa: F401
from pyrootsupport import expect

#: What ROOT 6.40's ``xmlnewfile.C`` wrote.
NEW_FILE = """<?xml version="1.0"?>
<main>
  <child1>Content of child1 node</child1>
  <child2 attr1="value1" attr2="value2"/>
  <child4 xmlns:child4="http://website/webpage">
    <child4:subchild1>subchild1 content</child4:subchild1>
  </child4>
</main>
"""


def made(xml: Any) -> Any:
    main = xml.NewChild(None, None, "main")
    xml.NewChild(main, None, "child1", "Content of child1 node")
    child2 = xml.NewChild(main, None, "child2")
    xml.NewAttr(child2, None, "attr1", "value1")
    xml.NewAttr(child2, None, "attr2", "value2")
    child4 = xml.NewChild(main, None, "child4")
    ns = xml.NewNS(child4, "http://website/webpage")
    xml.NewChild(child4, ns, "subchild1", "subchild1 content")
    doc = xml.NewDoc()
    xml.DocSetRootElement(doc, main)
    return doc


def test_a_document_is_written_as_root_lays_one_out(tmp_path: Any) -> None:
    xml = ROOT.TXMLEngine()
    doc = made(xml)
    xml.SaveDoc(doc, str(tmp_path / "new.xml"))
    xml.FreeDoc(doc)
    assert (tmp_path / "new.xml").read_text() == NEW_FILE


def test_a_document_read_back_is_walked_node_by_node_and_attribute_by_attribute(
    tmp_path: Any,
) -> None:
    (tmp_path / "new.xml").write_text(NEW_FILE)
    xml = ROOT.TXMLEngine()
    main = xml.DocGetRootElement(xml.ParseFile(str(tmp_path / "new.xml")))
    child1 = xml.GetChild(main)
    child2 = xml.GetNext(child1)
    first = xml.GetFirstAttr(child2)
    child4 = xml.GetNext(child2)
    sub = xml.GetChild(child4)
    expect((xml.GetNodeName(child1), "child1"), (xml.GetNodeContent(child1),
           "Content of child1 node"),
           ((xml.GetAttrName(first), xml.GetAttrValue(first)), ("attr1", "value1")),
           (xml.GetAttrValue(xml.GetNextAttr(first)), "value2"),
           (xml.GetNextAttr(xml.GetNextAttr(first)), None), (xml.GetNodeContent(child2), None),
           (xml.GetNSName(xml.GetNS(sub)), "child4"),
           (xml.GetNSReference(xml.GetNS(child4)), "http://website/webpage"),
           (xml.GetNS(child1), None), (xml.GetNext(child4), None), (xml.GetChild(child1), None),
           (xml.GetParent(sub), child4), (xml.ParseFile(str(tmp_path / "none.xml")),
           None))  # fmt: skip


def test_nodes_are_added_after_others_moved_and_unlinked(tmp_path: Any) -> None:
    xml = ROOT.TXMLEngine()
    doc = made(xml)
    main = xml.DocGetRootElement(doc)
    child1 = xml.GetChild(main)
    info = xml.NewChild(main, None, "info")
    xml.NewIntAttr(info, "num", 3)
    xml.AddChildAfter(main, info, child1)
    cell = type("Cell", (), {"value": info})()
    xml.ShiftToNext(cell)
    extra = xml.NewChild(None, None, "extra")
    xml.AddChild(main, extra)
    xml.SetNodeContent(extra, "text")
    xml.UnlinkNode(extra)
    xml.UnlinkNode(extra)
    xml.FreeNode(info)
    expect((xml.GetNodeName(cell.value), "child2"), (xml.HasAttr(info, "num"), True),
           (xml.GetIntAttr(info, "num"), 3), (xml.GetIntAttr(info, "none"), 0),
           (xml.GetAttr(info, "none"), None), (xml.GetNext(child1), cell.value),
           (xml.GetNodeContent(extra), "text"))  # fmt: skip
    empty = xml.NewDoc()
    xml.SaveDoc(empty, str(tmp_path / "empty.xml"))
    assert (tmp_path / "empty.xml").read_text() == '<?xml version="1.0"?>\n'
