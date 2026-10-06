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
    assert writer.GetModel() is model and writer.CreateEntry().GetPtr("n").value == 0
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
