"""Reading RNTuples by ROOT's names: entries loaded into a model, views, and what is printed."""

from __future__ import annotations

import pathlib

import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.ntuple.printing import json_text

DATA = pathlib.Path(__file__).parent / "data" / "rntuple"


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def _write(path: str = "r.root", name: str = "events") -> None:
    with xrdroot.create(path) as f:
        ntuple = f.rntuple(name, {"n": "std::int32_t", "pt": "std::vector<float>", "s": str,
                                  "ok": bool, "x": "double"})  # fmt: skip
        ntuple.extend({"n": [1, 2], "pt": [[0.5], [1.5, 2.5]], "s": ['a"b', "c\\d"],
                       "ok": [True, False], "x": [0.25, 1e-07]})  # fmt: skip


def test_a_reader_loads_each_entry_into_the_model_it_was_given() -> None:
    _write()
    model = ROOT.RNTupleModel.Create()
    n = model.MakeField["int"]("n")
    pt = model.MakeField["std::vector<float>"]("pt")
    reader = ROOT.RNTupleReader.Open(ROOT.std.move(model), "events", "r.root")
    seen = []
    for entry in reader:
        reader.LoadEntry(entry)
        seen.append((n.value, list(pt)))
    assert seen == [(1, [0.5]), (2, [1.5, 2.5])]
    assert reader.GetNEntries() == 2 and list(reader.GetEntryRange()) == [0, 1]
    assert reader.get() is reader and "2 entries" in repr(reader)
    other = reader.GetModel().CreateEntry()
    reader.LoadEntry(0, other)
    assert other.GetPtr("n").value == 1


def test_a_reader_with_no_model_has_every_field_it_can_hold() -> None:
    _write()
    reader = ROOT.RNTupleReader.Open("events", "r.root")
    assert reader.GetModel().GetFieldNames() == ["n", "pt", "s", "ok", "x"]
    reader.LoadEntry(1)
    assert reader.GetModel().GetDefaultEntry().GetPtr("s").value == "c\\d"
    path = DATA / "test_nested_structs_rntuple_v1-0-0-0.root"
    nested = ROOT.RNTupleReader.Open("ntuple", str(path))
    assert nested.GetModel().GetFieldNames() == []
    assert nested.GetView["int"]("my_struct.i")(2) == 2


def test_a_reader_opens_an_rntuple_a_tfile_holds_or_gave() -> None:
    _write()
    f = ROOT.TFile("r.root")
    assert ROOT.RNTupleReader.Open("events", f).GetNEntries() == 2
    with xrdroot.open_root("r.root") as opened:
        descriptor = ROOT.RNTupleReader.Open(opened["events"]).GetDescriptor()
    assert (descriptor.GetName(), descriptor.GetDescription(), descriptor.GetNEntries()) == (
        "events", "", 2)  # fmt: skip
    assert descriptor.GetNClusters() == 1 and descriptor.GetNFields() == 7
    with pytest.raises(TypeError, match="takes a model or not"):
        ROOT.RNTupleReader.Open("events")


def test_a_view_reads_a_field_by_entry_and_a_vectors_items_by_theirs() -> None:
    _write()
    reader = ROOT.RNTupleReader.Open("events", "r.root")
    view = reader.GetView["int"]("n")
    assert [view(i) for i in view.GetFieldRange()] == [1, 2]
    items = ROOT.RNTupleView["float"](reader.GetView["float"]("pt._0"))
    assert [items(i) for i in items.GetFieldRange()] == [0.5, 1.5, 2.5]


def test_a_reader_prints_its_fields_in_roots_box_and_an_entry_in_json(capsys) -> None:
    _write()
    reader = ROOT.RNTupleReader.Open("events", "r.root")
    reader.PrintInfo()
    reader.Show(0)
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "*" * 36 + " NTUPLE " + "*" * 36
    assert out[1] == "* N-Tuple : events" + " " * 61 + "*"
    assert out[4] == "* Field 1       : n (std::int32_t)" + " " * 45 + "*"
    assert out[6] == "*   Field 2.1   : _0 (float)" + " " * 51 + "*"
    assert out[-7:] == ["{", '  "n": 1,', '  "pt": [0.5],', '  "s": "a\\"b",', '  "ok": true,',
                        '  "x": 0.25', "}"]  # fmt: skip
    assert json_text({"x": 1e-07, "ok": False, "s": "c\\d"}) == (
        '{\n  "x": 1e-07,\n  "ok": false,\n  "s": "c\\\\d"\n}\n')  # fmt: skip
    with pytest.raises(UnsupportedFeatureError, match="storage details"):
        reader.PrintInfo(ROOT.ENTupleInfo.kStorageDetails)
