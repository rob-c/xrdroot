"""Objects streamed as their classes stream them, and entries of objects one at a time.

An object keeps its ``TObject`` unless its class is told to ignore it; a
clones array not bypassing its objects' streamers writes each whole; a tree
of objects fills an entry at a time as it fills a column at a time; and a
column handed what it cannot hold refuses it by name.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from xrdroot import create, open_root
from xrdroot.wbranch import Collection, Split, Vector, Whole
from xrdroot.wclasses import Layout, Member, checksum
from xrdroot.wcolumns import CountColumn, MemberColumn, MemberRows, StreamedColumn
from xrdroot.winfo import INFOS
from xrdroot.wobjects import stream, stream_clones
from xrdroot.writer import WBuffer

#: A line as a getter reads its members, by ROOT's names.
LINE = {"fX1": 1.0, "fY1": 2.0, "fX2": 3.0, "fY2": 4.0, "fLineColor": 2, "fLineStyle": 1,
        "fLineWidth": 1}  # fmt: skip


def test_an_object_streams_its_tobject_unless_its_class_is_told_to_ignore_it():
    buf = WBuffer()
    stream(buf, "TVector3", {"fX": 1.0, "fY": 2.0, "fZ": 3.0}.__getitem__)
    data = bytes(buf.data)
    assert data[4:6] == b"\x00\x03" and data[6:8] == b"\x00\x01"  # version 3, then TObject's 1
    assert len(data) == 6 + 10 + 24


def test_a_clones_array_not_bypassing_streams_each_object_whole():
    data = stream_clones("TLine", [LINE.__getitem__], bypass=False)
    assert b"TLines" in data and b"TLine;3" in data
    whole = WBuffer()
    stream(whole, "TLine", LINE.__getitem__)
    assert data.endswith(b"\x01" + bytes(whole.data))


def _written(specs, fill):
    buf = io.BytesIO()
    with create(buf, level=1) as out:
        tree = out.tree("t", specs)
        fill(tree)
        assert {"TBranchElement", "TLeafElement"} <= set(tree.classes)
    return open_root(io.BytesIO(buf.getvalue()))["t"]


def _hit() -> Layout:
    made = Layout("Hit", 1, 0, (Member("fE", 8, "double", "", 8), Member("fN", 3, "int", "", 4)))
    return made._replace(checksum=checksum("Hit", made.elements()))


def test_entries_given_one_at_a_time_land_as_a_column_of_them_would():
    specs = {"v": Vector("d"), "hits": Collection(_hit(), own=True), "hit": Split(_hit(), own=True)}

    def fill(tree):
        for i in range(3):
            row = {"v": np.arange(i, dtype=float), "hits": i, "hits\0hits.fE": [0.5] * i,
                   "hits\0hits.fN": [i] * i, "hit\0fE": 1.5 * i, "hit\0fN": i}  # fmt: skip
            tree.fill(**row)
        assert tree.columns["fE"] == "Double_t" and tree.columns["v"] == "object"

    back = _written(specs, fill)
    assert back["v"].array().tolist() == [[], [0.0], [0.0, 1.0]]
    assert back["hits.fN"].array().tolist() == [[], [1], [2, 2]]
    assert back["fE"].array().tolist() == [0.0, 1.5, 3.0]
    assert back["hits"].record.element[7] == 2  # fMaximum: the most hits an entry held


def test_a_column_given_what_it_cannot_hold_is_refused_by_name():
    with pytest.raises(ValueError, match="takes one number per entry"):
        MemberColumn("x", 8, "", 32000).pack_many(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="already streamed"):
        StreamedColumn("o", 32000).pack("not bytes")
    assert StreamedColumn("o", 32000).typename == "object"
    assert MemberRows("r", 5, "", 32000).pack([1.0, 2.0]) == bytes.fromhex("3f80000040000000")
    assert CountColumn("c", 32000).pack(7) == bytes.fromhex("00000007")
    with pytest.raises(ValueError, match="holds a NUL"):
        _written({"bad\0name": Vector("f")}, lambda tree: None)


def test_a_whole_object_names_every_class_it_holds_for_the_file_to_describe():
    clones = Whole("TClonesArray", object=True, holds=("TLine",))
    assert {"TBranchObject", "TLeafObject", "TClonesArray", "TLine"} <= set(clones.classes())
    assert "TVector3" in Whole("TLorentzVector", custom=True).classes()
    assert INFOS["TLine"][1] == 3
