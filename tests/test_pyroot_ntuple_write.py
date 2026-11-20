"""Writing RNTuples by ROOT's names: a writer's fills, its options, and filling from threads."""

from __future__ import annotations

import threading

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.pyroot.ntuple import writing


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def _model() -> tuple[object, object, object, object]:
    model = ROOT.RNTupleModel.Create()
    n = model.MakeField["unsigned int"]("n")
    pt = model.MakeField["std::vector<float>"]("pt")
    name = model.MakeField["std::string"]("name")
    return model, n, pt, name


def _read(path: str, name: str = "events") -> dict[str, object]:
    with xrdroot.open_root(path) as f:
        found = f[name].arrays()
        return {key: getattr(value, "tolist", lambda v=value: v)() for key, value in found.items()}


def test_a_record_field_is_written_as_the_members_its_dictionary_lists() -> None:
    """``write.cxx`` of roottest's atlas-datavector makes a field of a class type and
    fills a default entry; the file keeps it as ROOT does, a record of its members."""
    model = ROOT.RNTupleModel.Create()
    model.AddField(ROOT.RFieldBase.Create("evt", "TLorentzVector").Unwrap())
    entry = model.GetDefaultEntry()
    entry["evt"] = {"fP": {"fX": 1.0, "fY": 2.0, "fZ": 3.0}, "fE": 4.0}
    writer = ROOT.RNTupleWriter.Recreate(ROOT.std.move(model), "events", "w.root")
    writer.Fill()
    del writer
    assert _read("w.root")["evt"] == [{"fP": {"fX": 1.0, "fY": 2.0, "fZ": 3.0}, "fE": 4.0}]


def test_an_entry_that_was_never_given_a_record_is_written_as_its_default() -> None:
    """A record's holder starts as its members' defaults, which is the object a class's
    default constructor makes - what ``writer->Fill()`` with no values writes."""
    model = ROOT.RNTupleModel.Create()
    model.AddField(ROOT.RFieldBase.Create("evt", "TLorentzVector").Unwrap())
    writer = ROOT.RNTupleWriter.Recreate(ROOT.std.move(model), "events", "w.root")
    writer.Fill()
    del writer
    assert _read("w.root")["evt"] == [{"fP": {"fX": 0.0, "fY": 0.0, "fZ": 0.0}, "fE": 0.0}]


def test_a_bitset_field_of_a_model_is_written_its_bits_lowest_first() -> None:
    """The holder of a ``std::bitset<16>`` is a bitset, and a fill writes its bits."""
    model = ROOT.RNTupleModel.Create()
    model.AddField(ROOT.RFieldBase.Create("foo", "std::bitset<16>").Unwrap())
    model.GetDefaultEntry()["foo"] = 0xFA2
    writer = ROOT.RNTupleWriter.Recreate(ROOT.std.move(model), "events", "w.root")
    writer.Fill()
    del writer
    assert _read("w.root")["foo"] == [[bool(0xFA2 >> bit & 1) for bit in range(16)]]


def test_a_float_field_told_to_be_kept_in_fewer_bits_is_written_that_way() -> None:
    """``ntpl018_low_precision_floats`` sets a field's column type - ``SetHalfPrecision()``,
    ``SetTruncated(bits)``, ``SetQuantized(low, high, bits)`` - before the model takes it,
    and ROOT reads the float back as a float whatever column holds it."""
    model = ROOT.RNTupleModel.Create()
    for name, store in (
        ("half", lambda field: field.SetHalfPrecision()),
        ("trunc", lambda field: field.SetTruncated(20)),
        ("quant", lambda field: field.SetQuantized(-1.0, 1.0, 24)),
    ):
        field = ROOT.RField["float"](name)
        store(field)
        model.AddField(field)
    for name in ("half", "trunc", "quant"):
        model.GetDefaultEntry()[name] = 0.1
    writer = ROOT.RNTupleWriter.Recreate(ROOT.std.move(model), "events", "w.root")
    writer.Fill()
    del writer
    read = _read("w.root")
    # The values ROOT keeps at each precision, which are what it reads back.
    assert read["half"] == [0.0999755859375]
    assert read["trunc"] == [0.0999755859375]
    assert read["quant"] == [0.09999997168779373]
    with xrdroot.open_root("w.root") as f:
        assert [column.type for column in f["events"]._store.schema.columns] == [0x0B, 0x1C, 0x1D]


def test_a_field_that_is_no_float_has_no_precision_to_give_up() -> None:
    """Only a float or a double can be stored in fewer bits; a vector's items are a field
    of their own, one lower, and a string has no bits of its own to lose."""
    for cxx in ("std::vector<float>", "std::string"):
        field = ROOT.RFieldBase.Create("v", cxx).Unwrap()
        with pytest.raises(xrdroot.UnsupportedFeatureError, match="one float or one double"):
            field.SetTruncated(16)


def test_a_quantized_field_is_set_by_the_range_and_the_bits_in_either_order() -> None:
    """ROOT 6.40 takes ``SetQuantized(min, max, nBits)`` and the ``(nBits, {min, max})``
    that replaces it; a call naming neither is refused with both spellings."""
    field = ROOT.RField["float"]("q")
    field.SetQuantized(24, (-1.0, 1.0))
    assert repr(field.kind().spec) == "<LowPrecisionSpec quantized keeping 24 of a float32>"
    with pytest.raises(TypeError, match="takes the range and the bits"):
        field.SetQuantized(-1.0)


def test_walking_a_fields_subfields_is_refused_by_name() -> None:
    """``for (auto &field : *fieldEvents)`` walks a class's whole hierarchy in ROOT, to
    find the float under each member; nothing here walks one."""
    field = ROOT.RField["float"]("pt")
    with pytest.raises(xrdroot.UnsupportedFeatureError, match="walking the subfields"):
        len(field)


def test_a_set_and_a_map_field_are_written_from_the_items_they_hold() -> None:
    """The three unordered collections of roottest's makeproject header: each is written
    as ROOT writes a collection - the index column over the items, then the items."""
    model = ROOT.RNTupleModel.Create()
    model.AddField(ROOT.RFieldBase.Create("spam", "std::unordered_set<double>").Unwrap())
    model.AddField(ROOT.RFieldBase.Create("eggs", "std::unordered_map<int,std::string>").Unwrap())
    entry = model.GetDefaultEntry()
    entry["spam"] = [1.0, 3.0, 5.0]
    entry["eggs"] = [(2, "two"), (4, "four")]
    writer = ROOT.RNTupleWriter.Recreate(ROOT.std.move(model), "events", "w.root")
    writer.Fill()
    del writer
    assert _read("w.root") == {"spam": [[1.0, 3.0, 5.0]], "eggs": [{2: "two", 4: "four"}]}


def test_a_map_of_vectors_is_written_with_the_items_of_its_vectors() -> None:
    """A multimap's value is a vector, which nests in the file as a collection under the
    pair's record - the last of roottest's makeproject header's members."""
    model = ROOT.RNTupleModel.Create()
    kind = "std::unordered_multimap<std::string,std::vector<float>>"
    model.AddField(ROOT.RFieldBase.Create("strange", kind).Unwrap())
    model.GetDefaultEntry()["strange"].insert(("one", ROOT.std.vector["float"]([1.0, 2.0])))
    writer = ROOT.RNTupleWriter.Recreate(ROOT.std.move(model), "events", "w.root")
    writer.Fill()
    del writer
    assert [(key, list(items)) for key, items in _read("w.root")["strange"][0]] == [
        ("one", [1.0, 2.0])
    ]


def test_a_multimap_field_takes_every_pair_it_is_given_for_a_key() -> None:
    """``entry[name] = [...]`` inserts a pair each time, as ``std::unordered_multimap``'s
    ``insert`` does, where a map keeps one value under a key."""
    model = ROOT.RNTupleModel.Create()
    kind = "std::unordered_multimap<std::string,std::vector<float>>"
    model.AddField(ROOT.RFieldBase.Create("strange", kind).Unwrap())
    entry = model.GetDefaultEntry()
    entry["strange"] = [("one", [1.0]), ("one", [2.0])]
    assert [(pair.first, list(pair.second)) for pair in entry["strange"]] == [
        ("one", [1.0]),
        ("one", [2.0]),
    ]


def test_a_writer_fills_the_default_entry_and_finishes_when_let_go() -> None:
    model, n, pt, name = _model()
    writer = ROOT.RNTupleWriter.Recreate(ROOT.std.move(model), "events", "w.root")
    for i in range(3):
        n[0] = i
        pt.clear()
        for k in range(i):
            pt.push_back(k + 0.5)
        name.value = f"e{i}"
        assert writer.Fill() > 0
    assert writer.GetNEntries() == 3 and writer.get() is writer
    # ROOT's Python writer keeps a clone of the model it was given, so the
    # caller's stays open to another MakeField; the clone keeps its default
    # entry, which is what Fill() with no argument read.
    assert writer.GetModel() is not model and writer.GetModel().IsFrozen()
    assert not model.IsFrozen()
    assert writer.CreateEntry().GetPtr("n").value == 0
    writer.FlushColumns()
    del writer
    assert _read("w.root") == {"n": [0, 1, 2], "pt": [[], [0.5], [0.5, 1.5]],
                               "name": ["e0", "e1", "e2"]}  # fmt: skip


def test_a_writer_fills_an_entry_of_its_own_in_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(writing, "BATCH", 2)
    model, *_ = _model()
    writer = ROOT.RNTupleWriter.Recreate(model, "events", "b.root", ROOT.RNTupleWriteOptions())
    entry = writer.CreateEntry()
    for i in range(5):
        entry.GetPtr("n").value = i
        writer.Fill(entry)
    assert len(writer._sink.ntuple) == 4
    writer.CommitCluster()
    writer.reset()
    writer.reset()
    assert _read("b.root")["n"] == [0, 1, 2, 3, 4]


def test_a_writer_adds_its_rntuple_to_a_file_open_for_writing() -> None:
    f = ROOT.TFile("a.root", "RECREATE")
    model = ROOT.RNTupleModel.Create()
    x = model.MakeField["double"]("x")
    writer = ROOT.RNTupleWriter.Append(model, "events", f)
    x.value = 2.5
    writer.Fill()
    writer.reset()
    f.Close()
    assert _read("a.root") == {"x": [2.5]}
    with xrdroot.create("direct.root") as out:
        other = ROOT.RNTupleModel.Create()
        other.MakeField["int"]("k")
        ROOT.RNTupleWriter.Append(other, "events", out).reset()
    assert _read("direct.root") == {"k": []}


def test_a_file_open_for_reading_takes_no_rntuple() -> None:
    ROOT.TFile("r.root", "RECREATE").Close()
    reading = ROOT.TFile("r.root")
    with pytest.raises(ValueError, match="not open for writing"):
        ROOT.RNTupleWriter.Append(ROOT.RNTupleModel.Create(), "events", reading)


def test_the_options_say_how_the_file_is_compressed() -> None:
    options = ROOT.RNTupleWriteOptions()
    assert options.file_options() == {"compression": "zlib", "level": 5}
    options.SetCompression(0)
    assert options.GetCompression() == 0 and options.file_options()["compression"] is None
    options.SetCompression(1, 3)
    assert options.GetCompression() == 103
    options.SetApproxZippedClusterSize(1000)
    options.SetMaxUnzippedPageSize(64)
    options.SetUseBufferedWrite(False)
    options.SetEnablePageChecksums(True)
    assert (options.GetApproxZippedClusterSize(), options.GetMaxUnzippedPageSize()) == (1000, 64)


def test_writers_still_open_when_the_program_ends_are_finished() -> None:
    model, n, *_ = _model()
    writer = ROOT.RNTupleWriter.Recreate(model, "events", "end.root")
    n[0] = 7
    writer.Fill()
    writing._finish_all()
    assert _read("end.root")["n"] == [7]
    blank = object.__new__(ROOT.RNTupleWriter)
    blank.__del__()
    ROOT.RNTupleParallelWriter.__del__(object.__new__(ROOT.RNTupleParallelWriter))


def _bare() -> object:
    model = ROOT.RNTupleModel.CreateBare()
    model.MakeField["int"]("thread")
    model.MakeField["int"]("i")
    return model


def test_threads_fill_a_parallel_writer_each_through_a_context_of_its_own() -> None:
    writer = ROOT.RNTupleParallelWriter.Recreate(_bare(), "events", "p.root")
    assert writer.get() is writer and writer.GetModel().GetFieldNames() == ["thread", "i"]

    def fill(thread: int) -> None:
        context = writer.CreateFillContext()
        entry = context.CreateEntry()
        for i in range(10):
            entry.GetPtr("thread").value, entry.GetPtr("i").value = thread, i
            context.Fill(entry)

    threads = [threading.Thread(target=fill, args=(k,)) for k in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    writer.reset()
    writer.reset()
    found = _read("p.root")
    assert sorted(zip(found["thread"], found["i"], strict=True)) == [
        (k, i) for k in range(3) for i in range(10)]  # fmt: skip


def test_a_context_that_stages_its_clusters_adds_them_when_it_commits() -> None:
    f = ROOT.TFile("s.root", "RECREATE")
    writer = ROOT.RNTupleParallelWriter.Append(_bare(), "events", f)
    first, second = writer.CreateFillContext(), writer.CreateFillContext()
    first.EnableStagedClusterCommitting()
    assert first.IsStagedClusterCommittingEnabled()
    for context, thread in ((first, 1), (second, 2)):
        entry = context.CreateEntry()
        entry.GetPtr("thread").value = thread
        context.Fill(entry)
    first.FlushCluster()
    first.FlushCluster()  # nothing new to stage
    second.FlushCluster()
    assert len(writer._sink.ntuple) == 1
    first.CommitStagedClusters()
    writer.reset()
    f.Close()
    assert _read("s.root")["thread"] == [2, 1]


def test_a_snapshot_copies_a_vector_so_refilling_it_leaves_the_entry_be() -> None:
    vector = ROOT.std.vector["float"]([1.0])
    copied = writing._snapshot(vector)
    vector.clear()
    assert np.array_equal(copied, [1.0])


def test_a_writer_is_a_context_manager_that_finishes_the_file_it_opened() -> None:
    """ROOT's Python layer wraps a writer too: the block's end writes the rest and closes.

    Entering twice at once, or again after the block, is refused in ROOT's words.
    """
    model = ROOT.RNTupleModel.Create()
    model.MakeField["int"]("n")
    writer = ROOT.RNTupleWriter.Recreate(model, "events", "with.root")
    with writer as held:
        assert held is writer
        entry = held.CreateEntry()
        entry["n"] = 3
        held.Fill(entry)
        with pytest.raises(RuntimeError, match="cannot nest `with` statements using the same"):
            with writer:
                pass
    assert _read("with.root") == {"n": [3]}
    with pytest.raises(RuntimeError, match="cannot access RNTupleWriter after the `with`"):
        writer.Fill(entry)
    with pytest.raises(RuntimeError, match="cannot reuse RNTupleWriter in multiple"):
        with writer:
            pass
    # The model ROOT's Python layer clones from stays open, as ROOT's does.
    assert not model.IsFrozen()


def test_a_writer_whose_model_was_bare_fills_only_entries_of_its_own() -> None:
    """A bare model has no default entry, so the clone a writer holds has none either."""
    model = ROOT.RNTupleModel.CreateBare()
    model.MakeField["int"]("n")
    writer = ROOT.RNTupleWriter.Recreate(model, "events", "bare.root")
    with pytest.raises(RuntimeError, match="default entry of bare model"):
        writer.GetModel().GetDefaultEntry()
    entry = writer.CreateEntry()
    entry["n"] = 1
    writer.Fill(entry)
    writer.reset()
    assert _read("bare.root") == {"n": [1]}


def test_a_writer_recreates_a_model_from_a_list_of_make_pair_fields() -> None:
    """``RNTupleWriter::Recreate({make_pair("int", "f")}, ...)``: ROOT's Python layer takes the
    fields on the fly, each pair a field's type and its name."""
    fields = [ROOT.std.make_pair("int", "f"), ROOT.std.make_pair("std::string", "s")]
    writer = ROOT.RNTupleWriter.Recreate(fields, "events", "pairs.root")
    with writer as held:
        entry = held.CreateEntry()
        entry["f"], entry["s"] = 42, "answer"
        held.Fill(entry)
    assert _read("pairs.root") == {"f": [42], "s": ["answer"]}
