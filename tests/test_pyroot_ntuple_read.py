"""Reading RNTuples by ROOT's names: entries loaded into a model, views, and what is printed."""

from __future__ import annotations

import pathlib

import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.ntuple.printing import json_text
from xrdroot.pyroot.ntuple.reading import _held_as

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


def test_a_view_of_a_record_field_hands_out_an_object_of_its_class() -> None:
    """``GetView<T>(name)(i)`` of a record is the class the file was written from, its
    members the values the file keeps - what roottest's makeproject reads back as
    ``event.foo``, where reading the file's members alone gives a dict."""
    model = ROOT.RNTupleModel.Create()
    model.AddField(ROOT.RFieldBase.Create("evt", "TLorentzVector").Unwrap())
    model.GetDefaultEntry()["evt"] = {"fP": {"fX": 1.0, "fY": 2.0, "fZ": 3.0}, "fE": 4.0}
    writer = ROOT.RNTupleWriter.Recreate(ROOT.std.move(model), "events", "rec.root")
    writer.Fill()
    del writer
    reader = ROOT.RNTupleReader.Open("events", "rec.root")
    event = reader.GetView["TLorentzVector"]("evt")(0)
    assert isinstance(event, ROOT.TLorentzVector)
    assert (event.fP.fX, event.fP.fY, event.fP.fZ, event.fE) == (1.0, 2.0, 3.0, 4.0)


def test_a_reader_opens_an_rntuple_a_tfile_holds_or_gave() -> None:
    _write()
    f = ROOT.TFile("r.root")
    assert ROOT.RNTupleReader.Open("events", f).GetNEntries() == 2
    # ``f["events"]`` is the RNTuple itself, not a TObject standing for one, so a reader
    # opens it as ROOT's Python does.
    assert ROOT.RNTupleReader.Open(f["events"]).GetNEntries() == 2
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


def test_a_view_of_a_type_xrdroot_cannot_hold_hands_the_column_out_as_it_is() -> None:
    """A view named for a type this layer refuses - a vector of strings - gives the
    column's own value rather than refusing the read of a field it can still give."""
    _write()
    reader = ROOT.RNTupleReader.Open("events", "r.root")
    plain = ROOT.RNTupleView(reader, "n", "std::vector<std::string>")
    assert plain(0) == 1
    #: A view made by its name alone reads the type the file keeps for the field, and a
    #: field the file does not have is a view that hands the column out as it is.
    assert ROOT.RNTupleView(reader, "n")(0) == 1
    assert _held_as(reader, "nothing", None) is None


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


def test_a_reader_is_a_context_manager_that_lets_its_file_go() -> None:
    """ROOT's Python layer wraps a reader: one ``with`` block, then nothing.

    The three refusals are ROOT's own words, because a test of ROOT's matches
    on them.
    """
    _write()
    reader = ROOT.RNTupleReader.Open("events", "r.root")
    with reader as held:
        assert held is reader and held.GetNEntries() == 2
        entry = held.CreateEntry()
        held.LoadEntry(1, entry)
        assert entry["n"] == 2
        with pytest.raises(RuntimeError, match="cannot nest `with` statements using the same"):
            with reader:
                pass
    with pytest.raises(RuntimeError, match="cannot access RNTupleReader after the `with`"):
        reader.GetNEntries()
    with pytest.raises(RuntimeError, match="cannot reuse RNTupleReader in multiple"):
        with reader:
            pass
    assert entry["n"] == 2  # an entry's values outlive the reader that filled them


def test_an_entry_takes_and_gives_its_values_by_field_name() -> None:
    """``entry["f"] = v`` is PyROOT's pythonization of ``GetPtr<T>(name)``, typed as the
    field is: a string field takes a ``str``, a number field a number."""
    model = ROOT.RNTupleModel.Create()
    model.MakeField["int"]("n")
    model.MakeField["std::string"]("s")
    model.MakeField["std::vector<float>"]("v")
    entry = model.CreateEntry()
    entry["n"], entry["s"], entry["v"] = 7, "seven", [1.5, 2.5]
    assert (entry["n"], entry["s"], list(entry["v"])) == (7, "seven", [1.5, 2.5])
    with pytest.raises(TypeError, match="is a std::string, so it holds a str, not a int"):
        entry["s"] = 7
    with pytest.raises(TypeError, match="is a std::int32_t, so it holds a number, not a str"):
        entry["n"] = "seven"
    with pytest.raises(TypeError, match="holds a number, not a object"):
        entry["n"] = object()
    with pytest.raises(KeyError, match="invalid field name: nothing"):
        entry["nothing"] = 1


def test_a_model_of_every_field_leaves_a_record_to_the_views_of_its_members() -> None:
    from xrdroot.pyroot.ntuple.reading import _model_of

    class _Written:
        def cxx_types(self) -> dict[str, str]:
            return {"n": "int", "rec": "TLorentzVector", "vs": "std::vector<std::string>"}

    assert _model_of(_Written()).GetFieldNames() == ["n"]
