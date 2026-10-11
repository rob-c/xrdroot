"""Default columns, ``Fill`` of an object, and ``Snapshot``'s ``std::vector`` branches.

``RDataFrame(tree, files, {"b1"})`` names the columns an action or a
callable given none reads (df001); ``Fill(TH1D(...), {"b1"})`` fills an
object of one's own (df005); a collection ``Snapshot`` writes is the
``std::vector`` branch ROOT writes, with no count branch beside it, and the
frame over what it wrote has those columns as its defaults (df007). What
ROOT 6.40 does is stated in each test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import xrdroot
from xrdroot.hist import Histogram
from xrdroot.rdf import RDataFrame
from xrdroot.tree import Jagged


def _file(tmp_path: Path) -> Path:
    path = tmp_path / "in.root"
    with xrdroot.create(path) as out:
        out["t"] = {"b1": np.arange(6.0), "b2": np.arange(6, dtype=np.int32) % 2}
    return path


def test_an_action_given_no_column_reads_the_first_default_one(tmp_path: Path) -> None:
    """df001: ``Min()``, ``Max()``, ``Mean()``, ``Take()`` and ``Histo1D()`` of ``b1``."""
    with RDataFrame("t", str(_file(tmp_path)), ["b1"]) as frame:
        _check_defaults(frame)


def _check_defaults(frame: RDataFrame) -> None:
    assert (frame.Min().GetValue(), frame.Max().GetValue(), frame.Mean().GetValue()) == (
        0.0,
        5.0,
        2.5,
    )
    assert frame.Take().GetValue().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert frame.Histo1D().GetValue().entries == 6
    model = ("h", "h", 6, 0.0, 6.0)
    assert frame.Histo1D(model).GetValue().values().tolist() == [1.0] * 6
    assert frame.Mean("b2").GetValue() == 0.5


def test_a_callable_given_no_column_reads_as_many_default_ones_as_it_takes(
    tmp_path: Path,
) -> None:
    """As ROOT: the parameter's name does not matter, the default columns' order does."""
    with RDataFrame("t", str(_file(tmp_path)), ["b1", "b2"]) as frame:
        _check_callables(frame)


def _check_callables(frame: RDataFrame) -> None:
    assert frame.Filter(lambda anything: anything < 3).Count().GetValue() == 3
    seen: list[int] = []
    frame.Foreach(lambda a, b: seen.extend(np.asarray(b).tolist()))
    assert seen == [0, 1, 0, 1, 0, 1]
    assert frame.Filter(lambda b1, b2, extra: b1 > 0, ["b1", "b2", "b1"]).Count().GetValue() == 5


def test_an_action_given_no_column_of_a_frame_without_defaults_is_refused(
    tmp_path: Path,
) -> None:
    with RDataFrame("t", str(_file(tmp_path))) as frame:
        with pytest.raises(TypeError, match="no default column"):
            frame.Sum()
        with pytest.raises(KeyError, match="'x'"):
            frame.Filter(lambda x: x > 1).Count()


def test_a_frame_of_a_tree_takes_its_default_columns_after_it(tmp_path: Path) -> None:
    with xrdroot.open_root(_file(tmp_path)) as read:
        assert RDataFrame(read["t"], ["b2"]).Sum().GetValue() == 3


def test_fill_fills_a_histogram_as_its_model_and_any_object_by_its_fill(tmp_path: Path) -> None:
    """df005: a histogram filled as ``Histo1D`` fills one, a weight its extra column; any
    other object has its ``Fill`` called entry by entry, in order."""
    with RDataFrame("t", str(_file(tmp_path))) as frame:
        _check_fill(frame)


def _check_fill(frame: RDataFrame) -> None:
    model = Histogram.book("h", np.linspace(0, 6, 7))
    filled = frame.Fill(model, ["b1"]).GetValue()
    assert filled.values().tolist() == [1.0] * 6
    weighted = frame.Fill(Histogram.book("w", np.linspace(0, 6, 7)), ["b1", "b2"]).GetValue()
    assert weighted.values().tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    class Collector:
        def __init__(self) -> None:
            self.rows: list[tuple[float, int]] = []

        def Fill(self, a: float, b: int) -> None:
            self.rows.append((float(a), int(b)))

    collected = frame.Filter("b1 > 3").Fill(Collector(), ["b1", "b2"]).GetValue()
    assert collected.rows == [(4.0, 0), (5.0, 1)]


def test_a_snapshot_writes_a_collection_as_a_vector_branch_and_defaults_to_its_columns(
    tmp_path: Path,
) -> None:
    """df007: ``b2_vector`` is a ``std::vector<float>`` branch, with no ``nb2_vector``; the frame
    over the file has the columns written as its defaults, so ``Histo1D()`` fills ``b1``."""

    def vectors(b2: np.ndarray) -> Jagged:
        lengths = np.asarray(b2) + 1
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        return Jagged(np.ones(int(lengths.sum()), dtype=np.float32), offsets)

    out = str(tmp_path / "out.root")
    with RDataFrame("t", str(_file(tmp_path))) as frame:
        written = frame.Define("b2_vector", vectors).Snapshot("t", out, ["b1", "b2_vector"])
    with written:
        assert written.Histo1D().GetValue().entries == 6
    with xrdroot.open_root(out) as read:
        tree = read["t"]
        assert list(tree.keys()) == ["b1", "b2_vector"]
        assert [len(row) for row in tree.arrays()["b2_vector"]] == [1, 2, 1, 2, 1, 2]
