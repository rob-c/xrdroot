"""A frame's ``RVec`` of a class: a split collection of four-vectors, read and written.

A tree keeps a ``std::vector<ROOT::Math::XYZTVector>`` in a branch of each
coordinate, so a frame has to put one entry's four-vectors together from
them - and ``Snapshot`` has to take them apart again into the same branches.
Each test writes such a tree as ROOT writes it, reads it with a frame, and
checks the file a snapshot of the column makes.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import xrdroot
from xrdroot import RDataFrame
from xrdroot.pyroot.core.genvector import XYZTVector
from xrdroot.pyroot.rdf.objects import kinds
from xrdroot.pyroot.stl import std

#: Three entries, of three, one and two tracks, each four-vector (px, py, pz, E).
TRACKS = [
    [(1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0), (9.0, 1.0, 2.0, 3.0)],
    [(0.5, -0.5, 1.5, 2.5)],
    [(-1.0, -2.0, -3.0, 4.0), (10.0, 20.0, 30.0, 40.0)],
]


def _vectors(rows):
    """Each entry's four-vectors as the ``std::vector`` a macro's column holds."""
    return [
        std.vector["ROOT::Math::XYZTVector"]([XYZTVector(*one) for one in row]) for row in rows
    ]


def write_tracks(path):
    """A tree of ``TRACKS`` as ``Snapshot`` writes one: a branch of each coordinate."""
    kind = kinds("vector<ROOT::Math::XYZTVector>")
    assert kind is not None
    with xrdroot.create(str(path)) as f:
        tree = f.tree("events", {"tracks": kind.spec("tracks")})
        tree.extend(kind.columns("tracks", _vectors(TRACKS)))
    return str(path)


def test_a_split_collection_of_four_vectors_is_one_column_of_objects(tmp_path):
    source = write_tracks(tmp_path / "tracks.root")
    with RDataFrame("events", source) as df:
        assert df.GetColumnType("tracks") == (
            "ROOT::VecOps::RVec<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > >"
        )
        got = df.Take("tracks").GetValue()
    assert [len(entry) for entry in got] == [3, 1, 2]
    assert [float(one.Pt()) for one in got[0]] == pytest.approx(
        [float(np.hypot(1.0, 2.0)), float(np.hypot(5.0, 6.0)), float(np.hypot(9.0, 1.0))]
    )
    assert float(got[1][0].E()) == 2.5


def test_the_size_of_a_collection_is_a_number_an_expression_cuts_on(tmp_path):
    source = write_tracks(tmp_path / "tracks.root")
    with RDataFrame("events", source) as df:
        assert df.Filter("tracks.size() > 1").Count().GetValue() == 2
        counts = df.Define("n", "(int)tracks.size()").Take("n").GetValue()
        assert counts.tolist() == [3, 1, 2]
        # an expression reading the collection after a filter sees only the kept entries
        kept = df.Filter("tracks.size() > 1").Define("n", "(int)tracks.size()")
        assert kept.Take("n").GetValue().tolist() == [3, 2]
        # the right side of && is evaluated for the rows the left side leaves: the one here
        assert df.Filter("tracks.size() > 1 && tracks.size() < 3").Count().GetValue() == 1


def test_a_snapshot_writes_the_branch_layout_root_gives_the_collection(tmp_path):
    source = write_tracks(tmp_path / "tracks.root")
    out = str(tmp_path / "back.root")
    with RDataFrame("events", source) as df:
        with df.Snapshot("again", out, ["tracks"]) as back:
            assert back.GetColumnType("tracks").startswith("ROOT::VecOps::RVec<")
    with xrdroot.open_root(out) as f:
        branches = f["again"].branches
        assert list(branches) == [
            "tracks",
            "tracks.fCoordinates.fX",
            "tracks.fCoordinates.fY",
            "tracks.fCoordinates.fZ",
            "tracks.fCoordinates.fT",
        ]
        assert branches["tracks"].record.classname == (
            "vector<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > >"
        )
        assert branches["tracks"].record.split == 99
    with RDataFrame("again", out) as df:
        got = df.Take("tracks").GetValue()
    assert [len(entry) for entry in got] == [3, 1, 2]
    assert [float(one.X()) for one in got[2]] == [pytest.approx(-1.0), pytest.approx(10.0)]


def test_a_column_a_macro_filled_is_written_as_the_same_branch(tmp_path):
    """``Define`` of a lambda of no columns, whose answers are ``std::vector<XYZTVector>``:
    its values are collections of objects, so ``Snapshot`` writes ROOT's branch for them."""
    out = str(tmp_path / "filled.root")
    rows = _vectors(TRACKS)
    with RDataFrame(len(TRACKS)) as df:
        made = df.Define("tracks", lambda: rows.pop(0))
        with made.Snapshot("filled", out, ["tracks"]) as back:
            assert back.Count().GetValue() == len(TRACKS)
            got = back.Take("tracks").GetValue()
            assert [len(entry) for entry in got] == [3, 1, 2]
    with xrdroot.open_root(out) as f:
        assert next(iter(f["filled"].branches)) == "tracks"
        assert f["filled"].branches["tracks"].record.split == 99


def test_a_collection_a_filter_kept_is_read_again_entry_by_entry(tmp_path):
    source = write_tracks(tmp_path / "tracks.root")
    out = str(tmp_path / "kept.root")
    with RDataFrame("events", source) as df:
        kept = df.Filter("tracks.size() > 1")
        with kept.Snapshot("kept", out, ["tracks"]) as back:
            assert [len(entry) for entry in back.Take("tracks").GetValue()] == [3, 2]


def test_a_snapshot_of_a_collection_nothing_has_passed_still_declares_it(tmp_path):
    source = write_tracks(tmp_path / "tracks.root")
    out = str(tmp_path / "empty.root")
    with RDataFrame("events", source) as df:
        with df.Filter("tracks.size() > 99").Snapshot("none", out, ["tracks"]) as back:
            assert back.Count().GetValue() == 0
    with xrdroot.open_root(out) as f:
        assert len(f["none"]) == 0
        assert "tracks.fCoordinates.fX" in f["none"].branches


def test_a_callable_that_gives_collections_of_objects_back_is_one_column(tmp_path):
    """A ``Define`` of a callable, handed the column as it is read, may give back one
    collection of objects per entry; ``Snapshot`` then writes ROOT's branch for them."""
    source = write_tracks(tmp_path / "tracks.root")
    out = str(tmp_path / "copied.root")
    with RDataFrame("events", source) as df:
        copied = df.Define("copy", lambda tracks: list(tracks), ["tracks"])
        with copied.Snapshot("copied", out, ["copy"]) as back:
            assert [len(entry) for entry in back.Take("copy").GetValue()] == [3, 1, 2]
    with xrdroot.open_root(out) as f:
        assert f["copied"].branches["copy"].record.classname == (
            "vector<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > >"
        )


def test_a_column_of_objects_and_its_class_go_to_a_worker_process(tmp_path):
    """A frame of workers sends each task to a process of its own, so what a column is -
    the ROOT layer's way with the class, and the objects of each entry - has to pickle."""
    source = write_tracks(tmp_path / "tracks.root")
    with RDataFrame("events", source) as df:
        column = df._graph.source.read("tracks", 0, 2)
        assert repr(column).startswith("<Objects vector<")
        again = pickle.loads(pickle.dumps(column.kind))
        assert again.classname == column.kind.classname
        copied = pickle.loads(pickle.dumps(column))
        assert [len(entry) for entry in copied.rows] == [3, 1]
        assert [float(one.X()) for one in copied.rows[0]] == [pytest.approx(1.0), 5.0, 9.0]
    with RDataFrame("events", source, workers=2, step=1) as df:
        assert df.Filter("tracks.size() > 1").Count().GetValue() == 2
        assert df.Define("n", "(int)tracks.size()").Take("n").GetValue().tolist() == [3, 1, 2]


def test_a_class_no_provider_makes_objects_of_is_left_a_dictionary(tmp_path):
    """A split object per entry is not a collection of them: it stays what the tree reads,
    a dictionary of its members, so nothing else's reading changes."""
    assert kinds("vector<int>") is None
    assert kinds("SomeClass") is None
