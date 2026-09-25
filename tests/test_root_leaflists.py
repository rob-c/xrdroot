"""A branch of several leaves, some of a length that changes from entry to entry.

``pod-advanced.root``'s ``orange`` is ``Evtake_iwant/I:Mc_x/F:Mc_y/F:
Trk_ntracks/I:Trk_px[Trk_ntracks]/F:Trk_py[Trk_ntracks]/F:Mc_q2/F``, which
go-hep filled with ``Trk_ntracks`` of ``10 + entry`` and ``Trk_px`` and
``Trk_py`` counting up from 10 in tenths; ``Mc_q2`` comes after both arrays,
where no fixed offset reaches it. The values are ROOT 6.40's reading of it.
"""

from __future__ import annotations

import pathlib
import struct

import numpy as np
import pytest

from xrdroot import open_root
from xrdroot.interp import build
from xrdroot.leaflist import walkable
from xrdroot.objects import BranchRecord, LeafRecord
from xrdroot.tree import Basket, Branch

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture
def orange():
    with open_root(DATA / "pod-advanced.root") as root:
        yield root["orange"]


def test_each_array_of_a_leaf_list_is_as_long_as_its_counter_says(orange):
    assert orange.unreadable == {}
    tracks = orange["orange.Trk_px"].array(0, 3)
    assert tracks.lengths().tolist() == orange["orange.Trk_ntracks"].array(0, 3).tolist()
    assert tracks.lengths().tolist() == [10, 11, 12]
    assert orange["orange.Trk_py"].array(1, 2)[0][-2:].tolist() == pytest.approx([10.9, 11.0])


def test_a_leaf_after_the_arrays_is_found_by_walking_past_them(orange):
    assert orange["orange.Mc_q2"].array(0, 3).tolist() == [10.0, 10.0, 10.0]
    assert orange["orange.Mc_x"].array(0, 3).tolist() == [5.0, 6.0, 7.0]
    assert orange["orange.Mc_q2"].array(entries=[2, 0]).tolist() == [10.0, 10.0]


def leaf(classname: str, name: str, length: int = 1, count: LeafRecord | None = None) -> LeafRecord:
    made = LeafRecord(classname)
    made.name, made.length, made.count = name, length, count
    return made


def test_a_fixed_array_after_a_counted_one_keeps_its_shape():
    """``n/I:v[n]/F:w[2]/S``, two entries: one of a single ``v``, one of none."""
    n = leaf("TLeafI", "n")
    v, w = leaf("TLeafF", "v", count=n), leaf("TLeafS", "w", length=2)
    record = BranchRecord()
    record.leaves, record.entries, record.basket_entry = [n, v, w], 2, [0, 2]
    data = struct.pack(">ifhh", 1, 1.5, 3, 4) + struct.pack(">ihh", 0, 5, 6)
    record.baskets = [Basket(0, 0, 2, len(data), data, [0, 12])]
    branch = Branch("b.w", record, w, build(record, w, None), None)
    assert np.array_equal(branch.array(), [[3, 4], [5, 6]])
    counted = Branch("b.v", record, v, build(record, v, None), None)
    assert counted.array().tolist() == [[1.5], []]


def test_a_leaf_list_holding_a_string_is_not_walked():
    n = leaf("TLeafI", "n")
    record = BranchRecord()
    record.leaves = [n, leaf("TLeafF", "v", count=n), leaf("TLeafC", "s")]
    assert not walkable(record)
