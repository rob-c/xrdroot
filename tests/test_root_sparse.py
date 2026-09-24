"""Sparse histograms, and the two lists ROOT draws several things at once with.

``tgme.root``'s ``mg`` is ROOT's own ``TMultiGraph``, from go-hep. No file in
the corpus holds a ``THnSparse`` or a ``THStack``, so those are made by
``crafted.py`` from the members their classes declare - the reading is
checked against that layout, not against ROOT, and the coordinates are
packed the way ``THnSparseCoordCompression`` packs them: each axis in as
many bits as its bins and flow need, little end first.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from crafted import craft
from xrdroot import (
    FormatError,
    Graph,
    Histogram,
    MultiGraph,
    SparseHistogram,
    Stack,
    UnsupportedFeatureError,
    open_root,
)
from xrdroot.sparse import SPARSE, _bits

DATA = pathlib.Path(__file__).parent / "data"


def axis(nbins: int) -> dict:
    return Histogram.new("a", np.arange(nbins + 1), np.zeros(nbins)).members["TH1"]["fXaxis"]


def packed(bins: list[tuple[int, ...]], widths: list[int]) -> list[int]:
    """Each bin's coordinates packed the way ROOT packs them, as signed bytes."""
    size = (sum(widths) + 7) // 8
    out = []
    for coordinates in bins:
        number, shift = 0, 0
        for value, width in zip(coordinates, widths):
            number |= value << shift
            shift += width
        out += [byte - 256 if byte > 127 else byte for byte in number.to_bytes(size, "little")]
    return out


def sparse_members(nbins: list[int], bins: list[tuple[int, ...]], values, squares=None) -> dict:
    widths = [_bits(count + 2) for count in nbins]
    size = (sum(widths) + 7) // 8
    chunk = {
        "fSingleCoordinateSize": size,
        "fCoordinatesSize": size * len(bins),
        "fCoordinates": packed(bins, widths),
        "fContent": ("TArrayD", list(values)),
        "fSumw2": None if squares is None else ("TArrayD", list(squares)),
    }
    return {
        "THnSparse": {
            "THnBase": {
                "TNamed": {"fName": "hn", "fTitle": "a sparse one"},
                "fNdimensions": len(nbins),
                "fAxes": [("TAxis", axis(count)) for count in nbins],
                "fEntries": float(len(bins)),
            },
            "fChunkSize": 1024,
            "fFilledBins": len(bins),
            "fBinContent": [("THnSparseArrayChunk", chunk)],
        }
    }


@pytest.fixture
def sparse(tmp_path):
    # Three axes of 4, 300 and 2 bins: 3, 9 and 2 bits, so 14 bits and two bytes a
    # bin, with the middle axis's coordinate running across the byte boundary.
    bins = [(1, 1, 1), (4, 257, 2), (0, 301, 3), (2, 300, 1)]
    members = sparse_members([4, 300, 2], bins, [2.0, 3.0, 7.0, 1.5], [4.0, 5.0, 49.0, 2.25])
    path = craft(tmp_path / "hn.root", [("THnSparseT<TArrayD>", "hn", members)])
    with open_root(str(path)) as handle:
        yield handle["hn"]


def test_a_sparse_histogram_is_its_axes_and_the_bins_something_fell_into(sparse):
    assert isinstance(sparse, SparseHistogram)
    assert sparse.shape == (4, 300, 2) and len(sparse) == 4
    assert (sparse.name, sparse.title, sparse.entries) == ("hn", "a sparse one", 4.0)
    assert repr(sparse) == "<THnSparseT<TArrayD> 'hn' of 3 dimensions, 4 bins filled>"
    assert sparse.axes[1].nbins == 300


def test_the_coordinates_of_a_sparse_bin_count_from_zero_with_the_flow_either_side(sparse):
    assert sparse.coordinates().tolist() == [[0, 0, 0], [3, 256, 1], [-1, 300, 2], [1, 299, 0]]
    assert sparse.values().tolist() == [2.0, 3.0, 7.0, 1.5]
    assert sparse.weighted and sparse.variances().tolist() == [4.0, 5.0, 49.0, 2.25]


def test_a_small_sparse_histogram_can_be_laid_out_as_a_grid(sparse):
    grid = sparse.to_dense()
    assert grid.shape == (4, 300, 2)
    assert (grid[0, 0, 0], grid[3, 256, 1], grid[1, 299, 0]) == (2.0, 3.0, 1.5)
    assert grid.sum() == 6.5  # the bin in the flow is not in the grid
    assert sparse.to_dense(flow=True)[0, 301, 3] == 7.0


def test_a_sparse_histogram_too_big_for_a_grid_is_refused(tmp_path):
    members = sparse_members([1000, 1000, 1000], [(1, 1, 1)], [1.0])
    path = craft(tmp_path / "big.root", [("THnSparseT<TArrayD>", "big", members)])
    with open_root(str(path)) as handle:
        big = handle["big"]
    assert big.coordinates().tolist() == [[0, 0, 0]]
    assert not big.weighted and big.variances().tolist() == [1.0]
    with pytest.raises(UnsupportedFeatureError, match="cells laid out densely"):
        big.to_dense()


def test_a_sparse_histogram_of_no_bins_is_empty(tmp_path):
    members = sparse_members([3], [], [])
    members["THnSparse"]["fBinContent"] = []
    path = craft(tmp_path / "none.root", [("THnSparseT<TArrayD>", "none", members)])
    with open_root(str(path)) as handle:
        empty = handle["none"]
    assert len(empty) == 0 and empty.coordinates().shape == (0, 1)
    assert empty.to_dense().tolist() == [0.0, 0.0, 0.0]


def test_a_sparse_histogram_whose_bins_do_not_fit_its_axes_is_refused():
    members = sparse_members([4], [(1,)], [1.0])
    chunk = members["THnSparse"]["fBinContent"][0][1]
    chunk["fContent"] = [1.0]
    members["THnSparse"]["THnBase"]["fAxes"] = [axis(4)]
    members["THnSparse"]["fBinContent"] = [chunk]
    chunk["fSingleCoordinateSize"] = 2
    with pytest.raises(FormatError, match="2 bytes of coordinates each where its axes need 1"):
        SparseHistogram("THnSparseD", members)
    chunk["fSingleCoordinateSize"], chunk["fContent"] = 1, None
    with pytest.raises(FormatError, match="has 1 bins and contents for 0"):
        SparseHistogram("THnSparseD", members)
    with pytest.raises(FormatError, match="without its axes or its bins"):
        SparseHistogram("THnSparseD", {"THnSparse": {}})


def test_a_sparse_histogram_s_members_are_found_whichever_base_holds_them():
    members = {"TAttLine": {"fLineColor": 1}, **sparse_members([4], [(1,)], [1.0])}
    chunk = members["THnSparse"]["fBinContent"][0][1]
    chunk["fContent"] = [1.0]
    members["THnSparse"]["THnBase"]["fAxes"] = [axis(4)]
    members["THnSparse"]["fBinContent"] = [chunk]
    assert SparseHistogram("THnSparseD", members).name == "hn"


def test_every_sparse_class_is_recognised_by_either_of_its_names():
    assert "THnSparseT<TArrayF>" in SPARSE and "THnSparseF" in SPARSE and "THnSparseC" in SPARSE


def test_a_stack_is_the_histograms_it_was_given_in_order(tmp_path):
    first = Histogram.new("first", [0, 1, 2], [1, 2])
    second = Histogram.new("second", [0, 1, 2], [3, 4])
    members = {
        "TNamed": {"fName": "stack", "fTitle": "stacked"},
        "fHists": [("TH1D", first.members), ("TH1D", second.members)],
        "fMaximum": -1111.0,
        "fMinimum": -1111.0,
    }
    path = craft(tmp_path / "stack.root", [("THStack", "stack", members)])
    with open_root(str(path)) as handle:
        stack = handle["stack"]
    assert isinstance(stack, Stack) and len(stack) == 2
    assert [one.name for one in stack] == ["first", "second"]
    assert sum(one.values() for one in stack).tolist() == [4.0, 6.0]
    assert (stack.name, stack.title) == ("stack", "stacked")
    assert repr(stack) == "<THStack 'stack' of 2>"
    assert stack[-1].name == "second" and len(stack[:1]) == 1


def test_an_empty_stack_is_empty():
    assert len(Stack("THStack", {"TNamed": {"fName": "s", "fTitle": ""}, "fHists": None})) == 0


def test_a_stack_holding_something_other_than_histograms_is_refused():
    with pytest.raises(FormatError, match="holds 1 things that are not a Histogram"):
        Stack("THStack", {"TNamed": {"fName": "s", "fTitle": ""}, "fHists": ["TF1"]})


def test_a_multigraph_is_the_graphs_it_holds():
    with open_root(str(DATA / "tgme.root")) as handle:
        multi = handle["mg"]
    assert isinstance(multi, MultiGraph) and len(multi) == 3
    assert all(isinstance(one, Graph) for one in multi)
    assert multi.name == "mg"
    assert repr(multi).startswith("<TMultiGraph 'mg' of 3>")
