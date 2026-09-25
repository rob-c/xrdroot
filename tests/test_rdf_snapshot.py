"""``Snapshot``: what reaches a node, written to a file and read back as a frame.

Each test writes, reads the file back with a frame of its own, and compares
what came back with what went in: numbers, collections and strings, as a
``TTree`` and as an RNTuple.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import xrdroot
from frames import DATA, write_xyn
from xrdroot import RDataFrame

FLAT = str(DATA / "small-flat-tree.root")


def test_a_snapshot_writes_what_passed_and_reads_back_as_a_frame(tmp_path):
    out = str(tmp_path / "out.root")
    with RDataFrame("tree", write_xyn(tmp_path / "in.root"), step=30) as df:
        kept = df.Filter("x < 10").Define("z", "x * x")
        with kept.Snapshot("small", out, ["x", "z"]) as written:
            assert written.GetColumnNames() == ["x", "z"]
            assert written.Take("z").GetValue().tolist() == [i * i for i in range(10)]
            assert written.Count().GetValue() == 10


def test_collections_and_strings_survive_a_tree_and_an_rntuple(tmp_path):
    with RDataFrame("tree", FLAT, step=11) as df:
        picked = df.Define("good", "SliceFloat64[SliceFloat64 > 50]").Filter("Int32 % 3 == 0")
        wanted = picked.AsNumpy(["Int32", "Str", "good", "ArrayInt32"]).GetValue()
        for rntuple in (False, True):
            out = str(tmp_path / f"out-{rntuple}.root")
            with picked.Snapshot(
                "events", out, ["Int32", "Str", "good", "ArrayInt32"], rntuple=rntuple
            ) as back:
                got = back.AsNumpy(["Int32", "Str", "good"]).GetValue()
                assert got["Int32"].tolist() == wanted["Int32"].tolist()
                assert list(got["Str"]) == list(wanted["Str"])
                assert got["good"].tolist() == wanted["good"].tolist()
                assert back.Take("ArrayInt32").GetValue().tolist() == wanted["ArrayInt32"].tolist()
        with xrdroot.open_root(str(tmp_path / "out-True.root")) as f:
            assert f.classnames()["events"] == "ROOT::RNTuple"


def test_the_columns_default_to_every_one_or_those_a_pattern_finds(tmp_path):
    with RDataFrame("tree", write_xyn(tmp_path / "in.root")) as df:
        made = df.Define("z", "x + y")
        with made.Snapshot("all", str(tmp_path / "all.root")) as back:
            assert back.GetColumnNames() == ["z", "x", "y", "n"]
        with made.Snapshot("some", str(tmp_path / "some.root"), "^[xz]$") as back:
            assert back.GetColumnNames() == ["z", "x"]


def test_a_lazy_snapshot_waits_for_the_loop(tmp_path):
    out = tmp_path / "lazy.root"
    with RDataFrame("tree", write_xyn(tmp_path / "in.root")) as df:
        pending = df.Snapshot("t", str(out), ["n"], lazy=True)
        assert not pending.IsReady() and not out.exists()
        assert df.Count().GetValue() == 100 and pending.is_ready()
        with pending.GetValue() as back:
            assert back.Sum("n").GetValue() == 4950 and pending.value is back


def test_update_adds_a_tree_beside_what_the_file_has(tmp_path):
    out = str(tmp_path / "both.root")
    with RDataFrame("tree", write_xyn(tmp_path / "in.root")) as df:
        df.Snapshot("first", out, ["x"]).close()
        df.Snapshot("second", out, ["y"], mode="UPDATE", compression=None).close()
    with xrdroot.open_root(out) as f:
        assert f.trees() == ["first", "second"]
    with pytest.raises(ValueError, match="RECREATE, UPDATE"):
        RDataFrame(1).Define("a", "1").Snapshot("t", out, mode="APPEND")
    with pytest.raises(ValueError, match="no columns to write"):
        RDataFrame(1).Snapshot("t", out)


def test_a_snapshot_nothing_passes_still_declares_its_columns(tmp_path):
    out = str(tmp_path / "none.root")
    with RDataFrame("tree", write_xyn(tmp_path / "in.root")) as df:
        with df.Filter("x < 0").Snapshot("empty", out, ["x", "n"]) as back:
            assert back.Count().GetValue() == 0 and back.GetColumnType("n") == "Int_t"


def _breaks(n: np.ndarray) -> np.ndarray:
    if n[0] >= 50:
        raise RuntimeError("the second batch is where this breaks")
    return n


def test_a_loop_that_fails_leaves_no_file_behind(tmp_path):
    out = tmp_path / "broken.root"
    with RDataFrame("tree", write_xyn(tmp_path / "in.root"), step=50) as df:
        with pytest.raises(RuntimeError, match="second batch"):
            df.Define("m", _breaks, ["n"]).Snapshot("t", str(out), ["m"])
    assert not out.exists() or os.path.getsize(out) == 0


def test_a_snapshot_goes_into_directories_its_name_makes(tmp_path):
    out = str(tmp_path / "dirs.root")
    with RDataFrame("tree", write_xyn(tmp_path / "in.root")) as df:
        df.Snapshot("run/1/events", out, ["x"]).close()
    with xrdroot.open_root(out) as f:
        assert len(f["run/1/events"]) == 100


def _broken_at_once(n: np.ndarray) -> np.ndarray:
    raise RuntimeError("the first batch is where this breaks")


def test_a_loop_that_fails_before_writing_anything_opens_no_file(tmp_path):
    out = tmp_path / "never.root"
    with RDataFrame("tree", write_xyn(tmp_path / "in.root")) as df:
        with pytest.raises(RuntimeError, match="first batch"):
            df.Define("m", _broken_at_once, ["n"]).Snapshot("t", str(out), ["m"])
    assert not out.exists()
