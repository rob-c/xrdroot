"""What distributed RDataFrame asks of xrdroot's own frame: windows, partials, clusters.

A task's frame looks at a window of the data; its results are left as
partials, added to other tasks' and only then finished; trees are cut where
their clusters end, which ``FlushBaskets``, ``fAutoFlush`` and xrdroot's
writer record.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import xrdroot
from xrdroot import RDataFrame
from xrdroot.chain import _expanded
from xrdroot.file import open_root
from xrdroot.rdf.distributed import (
    Partial,
    Window,
    clusters_and_entries,
    partial_of,
    rebook,
    source_of,
    trigger,
)
from xrdroot.rdf.sources import Empty, Table


def _write(path, count=10, flush_every=None):
    with xrdroot.create(str(path)) as out:
        tree = out.tree("t", {"x": float, "n": int})
        for start in range(0, count, flush_every or count):
            stop = min(count, start + (flush_every or count))
            tree.extend({"x": np.arange(start, stop, dtype=float),
                         "n": np.arange(start, stop)})  # fmt: skip
            tree.flush() if flush_every else None
    return str(path)


def _booked(frame):
    """Every kind of result a task adds up, booked on one frame."""
    kept = frame.Define("x", "rdfentry_ * 1.0").Filter("x > 1", "big")
    return [
        kept.Count(), kept.Sum("x"), kept.Mean("x"), kept.StdDev("x"), kept.Min("x"),
        kept.Max("x"), kept.Histo1D(("h", "h", 10, 0, 10), "x"), kept.Take("x"),
        kept.AsNumpy(["x"]), kept.Stats("x"), kept.Graph("x", "x"), kept.Report(),
    ]  # fmt: skip


def test_a_window_shows_its_entries_numbered_as_the_whole_data_numbers_them():
    window = Window(Empty(10), 3, 7)
    frame = RDataFrame(window)
    assert frame.Define("e", "rdfentry_").Take("e").GetValue().tolist() == [3, 4, 5, 6]
    window.change(Empty(20), 15, 20)
    assert frame.Count().GetValue() == 5
    assert len(window) == 20 and window.names() == []
    assert window.describe() == "entries 15 to 20 of 20 empty entries"
    assert window.root_description() == "Empty dataframe filling 20 rows"


def test_a_window_on_a_tree_reads_and_describes_the_tree(tmp_path):
    with open_root(_write(tmp_path / "a.root")) as file:
        window = Window(Table(file["t"]), 2, 5)
        assert window.readable() == ["x", "n"] and window.cxx_type("x") is None
        assert window.read("x", 2, 5).tolist() == [2.0, 3.0, 4.0]
        assert window.available("x", 2, 5) is None
        assert window.boundaries() == [2, 5]
        assert window.sample(0).AsString().endswith("a.root/t")
        window.close()  # a tree handed over is not the window's to close


def test_a_window_closes_data_it_was_given_to_own(tmp_path):
    path = _write(tmp_path / "a.root")
    inner = xrdroot.rdf.sources.open_named("t", path)
    Window(inner, 0, 1).close()
    assert inner.table._links[0].file is None or inner.table._links[0].file.closed


def test_partials_of_two_ranges_add_up_to_the_whole_frames_results():
    whole = [result.GetValue() for result in _booked(RDataFrame(10))]
    window = Window(Empty(10), 0, 4)
    frame = RDataFrame(window)
    first = _booked(frame)
    trigger(frame)
    window.change(window.inner, 4, 10)
    second = [rebook(result) for result in first]
    trigger(frame)
    assert frame.GetNRuns() == 2
    for one, two, expected in zip(first, second, whole, strict=True):
        merged = partial_of(one)
        merged.Merge(partial_of(two))
        _same(merged.GetValue(), expected)


def _summary(value):
    """What a result is, to compare with another: its numbers, whatever it holds them in."""
    if hasattr(value, "edges"):
        return value.values().tolist()
    if isinstance(value, dict):
        return value["x"].tolist()
    if hasattr(value, "cuts"):
        return [(cut.GetPass(), cut.GetAll()) for cut in value]
    if hasattr(value, "GetMean"):
        return round(value.GetMean(), 9)
    if hasattr(value, "GetN"):
        return value.GetN()
    return np.round(np.asarray(value, dtype=float), 9).tolist()


def _same(found, expected):
    assert _summary(found) == _summary(expected)


def test_a_partial_adds_nothing_from_a_task_that_had_nothing():
    frame = RDataFrame(Window(Empty(4), 0, 4))
    taken = frame.Define("x", "1").Take("x")
    trigger(frame)
    found = partial_of(taken)
    empty = Partial(found.action, None, {})
    empty.Merge(found)
    found.Merge(Partial(found.action, None, {}))
    assert found.GetValue().tolist() == empty.GetValue().tolist() == [1, 1, 1, 1]
    assert repr(found) == "<Partial Take>"


def test_a_snapshot_is_written_by_its_task_and_its_partial_is_not_added_up(tmp_path):
    window = Window(Empty(6), 0, 3)
    frame = RDataFrame(window).Define("x", "rdfentry_")
    written = frame.Snapshot("t", str(tmp_path / "a_0.root"), ["x"], lazy=True)
    trigger(frame)
    again = rebook(written, str(tmp_path / "a_1.root"))
    window.change(window.inner, 3, 6)
    trigger(frame)
    assert partial_of(written).GetValue() == ("t", str(tmp_path / "a_0.root"))
    with pytest.raises(TypeError, match="gathered as it is"):
        partial_of(written).Merge(partial_of(again))
    with RDataFrame("t", str(tmp_path / "a_1.root")) as back:
        assert back.Take("x").GetValue().tolist() == [3, 4, 5]


def test_a_result_whose_loop_ran_no_task_has_no_partial():
    with pytest.raises(RuntimeError, match="has no partial"):
        partial_of(RDataFrame(3).Count())


def test_the_source_of_a_frame_is_what_its_graph_reads():
    window = Window(Empty(3), 0, 3)
    assert source_of(RDataFrame(window).Define("x", "1")) is window


def test_clusters_are_where_flush_ended_them(tmp_path):
    path = _write(tmp_path / "a.root", 25, flush_every=10)
    assert clusters_and_entries("t", path) == ([0, 10, 20, 25], 25)
    with open_root(path) as file:
        assert file["t"].clustering.ranges() == [(0, 10), (10, 20), (20, 25)]


def test_flushing_twice_in_a_row_ends_one_cluster(tmp_path):
    path = str(tmp_path / "a.root")
    with xrdroot.create(path) as out:
        tree = out.tree("t", {"x": float})
        tree.flush()
        tree.extend({"x": np.arange(5.0)})
        tree.flush()
        tree.flush()
    assert clusters_and_entries("t", path) == ([0, 5], 5)


def test_a_snapshot_with_auto_flush_ends_a_cluster_every_so_many_entries(tmp_path):
    out = str(tmp_path / "out.root")
    frame = RDataFrame(25, step=7).Define("x", "rdfentry_")
    frame.Snapshot("t", out, ["x"], auto_flush=10).close()
    assert clusters_and_entries("t", out) == ([0, 10, 20, 25], 25)
    with RDataFrame("t", out) as back:
        assert back.Take("x").GetValue().tolist() == list(range(25))


def test_an_rntuple_snapshot_keeps_its_own_clusters(tmp_path):
    out = str(tmp_path / "out.root")
    frame = RDataFrame(25).Define("x", "rdfentry_")
    frame.Snapshot("n", out, ["x"], rntuple=True, auto_flush=10).close()
    starts, entries = clusters_and_entries("n", out)
    assert entries == 25 and starts[0] == 0 and starts[-1] == 25


def test_a_relative_glob_matches_relative_names_and_a_file_url_its_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "a1.root")
    assert _expanded("a*.root") == ["a1.root"]
    assert _expanded(f"file://{tmp_path}/a*.root") == [os.path.join(str(tmp_path), "a1.root")]
