"""ROOT's own bookkeeping objects written from their members, and read back by either side.

A ``TNamed``, a ``TObjString``, a bare ``TObject`` and a ``TList`` of them
have no xrdroot class of their own; :class:`~xrdroot.writer.Record` hands the
writer the class and its members, laid out by the class's streamer
information, and the pyroot classes of those names are written that way.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.core.records import recorded
from xrdroot.writer import Record

HEAP = 0x03000000


def _histogram() -> xrdroot.Histogram:
    return xrdroot.Histogram.new("h", np.arange(3.0), np.ones(2))


def _written(tmp_path, **objects):
    path = tmp_path / "records.root"
    with xrdroot.create(path) as out:
        for name, obj in objects.items():
            out.write(name, obj)
    return path


def test_a_tnamed_a_tobject_a_tobjstring_and_a_named_list_are_read_back_as_written(tmp_path):
    string = Record("TObjString", {"TObject": {}, "fString": "a"})
    path = _written(
        tmp_path,
        n=Record("TNamed", {"TObject": {}, "fName": "n", "fTitle": "the title"}, "the title"),
        o=Record("TObject", {"fUniqueID": 5, "fBits": HEAP}),
        s=Record("TObjString", {"TObject": {}, "fString": "hello"}),
        l=Record("TList", [string, Record("TObject", {})], "Doubly linked list", "mine"),
    )
    with xrdroot.open_root(path) as found:
        assert found["n"]["fName"] == "n" and found["n"]["fTitle"] == "the title"
        assert found["o"]["fUniqueID"] == 5
        assert found["s"]["fString"] == "hello"
        assert [entry.get("fString") for entry in found["l"]] == ["a", None]
        assert found.key("n").title == "the title" and found.key("l").title == "Doubly linked list"
        assert set(found.classnames().values()) >= {"TNamed", "TObject", "TObjString", "TList"}


def test_a_record_of_a_class_without_a_layout_or_only_part_of_others_is_refused():
    with pytest.raises(UnsupportedFeatureError, match="THnSparseD is not a class"):
        Record("THnSparseD", {})
    with pytest.raises(UnsupportedFeatureError, match="TCollection is not a class"):
        Record("TCollection", {})
    assert "TNamed" in repr(Record("TNamed", {}, "t"))


def test_a_list_holding_what_this_writer_cannot_write_is_refused_naming_it(tmp_path):
    with pytest.raises(UnsupportedFeatureError, match="holding object is not written"):
        _written(tmp_path, l=Record("TList", [object()]))


def test_roots_core_objects_from_a_script_become_records_and_anything_else_stays(tmp_path):
    named = ROOT.TNamed("n", "t")
    named.SetUniqueID(7)
    made = recorded(named)
    assert isinstance(made, Record) and made.classname == "TNamed"
    assert made.members["TObject"] == {"fUniqueID": 7, "fBits": HEAP}
    assert recorded(ROOT.TObjString("s")).members["fString"] == "s"
    assert recorded(ROOT.TObject()).classname == "TObject"
    listed = ROOT.TList()
    listed.SetName("l")
    listed.Add(ROOT.TObjString("x"))
    made = recorded(listed)
    assert made.name == "l" and made.title == "Doubly linked list"
    assert made.members[0].classname == "TObjString"

    class Mine(ROOT.TNamed):
        pass

    mine = Mine("m", "")
    assert recorded(mine) is mine
    hist = ROOT.TH1F("h", "h", 2, 0, 1)
    assert recorded(hist) is hist._xrd


def test_a_list_writes_one_key_with_ksinglekey_and_a_key_per_object_without(tmp_path):
    path = str(tmp_path / "lists.root")
    out = ROOT.TFile(path, "RECREATE")
    held = ROOT.TList()
    for text in ("uno", "due"):
        held.Add(ROOT.TObjString(text))
    assert held.Write("MyList", ROOT.TObject.kSingleKey) == 1
    assert held.Write() == 2
    gappy = ROOT.TObjArray()
    gappy.AddAt(ROOT.TObjString("tre"), 2)
    assert gappy.Write() == 1
    named = ROOT.TNamed("named", "a title")
    named.Write()
    out.Close()
    back = ROOT.TFile(path)
    assert [str(entry) for entry in back.Get("MyList")] == ["uno", "due"]
    assert str(back.Get("due")) == "due" and str(back.Get("tre")) == "tre"
    assert back.Get("named").GetTitle() == "a title"
    assert back.GetKey("MyList").GetTitle() == "Doubly linked list"
    back.Close()


def test_the_bits_and_write_options_are_members_of_tobject_as_in_root():
    assert ROOT.TObject.kSingleKey == 1 and ROOT.TObject.kOverwrite == 2
    assert ROOT.TObject.kIsReferenced == 1 << 4


def test_collections_are_titled_as_their_classes_are_in_root():
    assert ROOT.TObjArray().GetTitle() == "An array of objects"
    assert ROOT.THashList().GetTitle() == "Doubly linked list with hashtable for lookup"


def test_a_bare_tobject_and_a_list_of_a_class_without_a_wrapper_read_back_in_a_script(tmp_path):
    line = Record("TAttLine", {"fLineColor": 2, "fLineStyle": 1, "fLineWidth": 3})
    path = _written(
        tmp_path,
        o=Record("TObject", {"fUniqueID": 9, "fBits": HEAP | 1}),
        l=Record("TList", [line, Record("TNamed", {"fName": "x", "fTitle": "y"}), _histogram()]),
    )
    found = ROOT.TFile(str(path))
    plain = found.Get("o")
    assert plain.ClassName() == "TObject" and plain.GetUniqueID() == 9 and plain.TestBit(1)
    listed = list(found.Get("l"))
    assert listed[0].ClassName() == "TAttLine" and listed[1].GetName() == "x"
    assert listed[2].GetName() == "h" and listed[2].ClassName() == "TH1D"
    found.Close()


def test_a_named_base_a_reference_may_point_at_names_its_process():
    from xrdroot.writer import WBuffer, _write_tnamed

    buf = WBuffer()
    named = {"fName": "n", "fTitle": "t", "fBits": 1 << 4, "fUniqueID": 5, "pidf": 2}
    _write_tnamed(buf, named)
    expected = "40000012" "0001" "0001" "00000005" "00000010" "0002" "016e" "0174"
    assert bytes(buf.data).hex() == expected
