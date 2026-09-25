"""Split collections, branches in files of their own, and vectors with allocators.

``tclonesarray-split.root`` was written by ROOT 6.40.04 from the macro beside
it, ``tclonesarray_split.C``: five entries of a ``TClonesArray`` of ``Hit``
split into members, the ``n``-th holding ``n`` hits, and a branch ``n``
whose baskets ROOT was told to write to ``tclonesarray-split-baskets.root``.
The values asserted are the ones the macro filled, and they agree with what
ROOT 6.40 reads back from the same file.
"""

from __future__ import annotations

import io
import pathlib
import shutil

import pytest

from xrdroot import UnsupportedFeatureError, open_root
from xrdroot.file import Source
from xrdroot.friends import basket_source
from xrdroot.interp import Refused, build
from xrdroot.objects import BranchRecord, LeafRecord
from xrdroot.streamers import Member

DATA = pathlib.Path(__file__).parent / "data"
SPLIT = DATA / "tclonesarray-split.root"


@pytest.fixture
def hits():
    with open_root(SPLIT) as root:
        yield root["t"]


def test_a_member_of_a_split_clones_array_reads_one_value_per_object(hits):
    assert hits["hits.fId"].array().tolist() == [[], [10], [20, 21], [30, 31, 32], [40, 41, 42, 43]]
    assert hits["hits.fUniqueID"].array().tolist() == [[], [0], [0, 0], [0, 0, 0], [0, 0, 0, 0]]


def test_an_array_member_of_a_split_clones_array_reads_each_objects_run_in_turn(hits):
    assert hits["hits.fPos[3]"].array(2, 3).tolist() == [[2.0, 0.0, -1.0, 2.0, 1.0, -1.0]]


def test_a_packed_member_of_a_split_clones_array_finds_its_range_under_its_own_name(hits):
    """``fE`` is a ``Double32_t`` in ``[0,100]`` at 16 bits: 1 comes back as its nearest step."""
    energies = hits["hits.fE"].array(1, 3).tolist()
    assert energies == [[0.99945068359375], [2.00042724609375, 14.50042724609375]]


def test_a_TString_in_each_object_of_a_split_collection_is_refused_by_name(hits):
    assert (
        "a TString in each object of the split collection hits_" in hits.unreadable["hits.fLabel"]
    )
    with pytest.raises(UnsupportedFeatureError, match="split collection hits_"):
        hits["hits.fLabel"].array()


def test_the_branch_a_split_clones_array_hangs_from_puts_its_objects_back_together(hits):
    rows = hits["hits"].array(1, 3)
    assert [row["fId"].tolist() for row in rows] == [[10], [20, 21]]
    assert set(hits["hits"].unreadable) == {"fLabel"}


def test_a_class_in_each_object_of_a_split_collection_is_refused_by_name():
    source = type("Described", (), {})()
    source.streamers = lambda: {"Hit": {"fArr": Member("fArr", "", 69, "TArrayF*", 0)}}
    branch, leaf = BranchRecord(), LeafRecord("TLeafElement")
    branch.classname, leaf.name, leaf.ltype = "Hit", "hits.fArr", 69
    leaf.count = LeafRecord("TLeafElement")
    leaf.count.name = "hits_"
    column = build(branch, leaf, source)
    assert isinstance(column, Refused)
    assert column.reason.startswith("a TArrayF* in each object of the split collection hits_")


def test_a_branch_written_to_a_file_of_its_own_reads_its_baskets_from_there(hits):
    assert hits["n"].record.file_name == "tclonesarray-split-baskets.root"
    assert hits["n"].array().tolist() == [0, 1, 2, 3, 4]
    source = hits._source
    source.companions.insert(0, Source(io.BytesIO(b""), "a friend's file", owned=True))
    again = basket_source("tclonesarray-split-baskets.root", source)
    assert again is basket_source("tclonesarray-split-baskets.root", source)
    assert basket_source("tclonesarray-split.root", source) is source


def test_a_branch_whose_baskets_file_is_missing_says_where_it_looked(tmp_path):
    shutil.copy(SPLIT, tmp_path / SPLIT.name)
    with open_root(tmp_path / SPLIT.name) as root:
        with pytest.raises(FileNotFoundError, match=r"the baskets of a branch in .* is in"):
            root["t"]["n"].array()


def test_a_vector_with_an_allocator_named_reads_as_the_vector_it_is():
    """``uproot-issue-172.root`` is RVecs of ROOT 6.20, which name ``RAdoptAllocator``."""
    with open_root(DATA / "uproot-issue-172.root") as root:
        events = root["events"]
        assert events.unreadable == {}
        assert events["rec_part_px_VecOps"].array() == events["rec_part_px"].array()
        assert events["rec_part_px_VecOps"].array(0, 1)[0][:2].tolist() == pytest.approx(
            [-0.100937, -0.178590], rel=1e-5
        )
