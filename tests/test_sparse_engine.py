"""``THnSparse``: the bins a sparse histogram keeps, read out of the members a file gives. A sparse
histogram is read here without a file at all: the members a ``THnSparse`` is streamed with - its
axes, and its ``THnSparseArrayChunk``s of packed coordinates - are plain dictionaries, so the
engine's own behaviour is what these tests exercise. That is what puts ``xrdroot.sparse`` inside
the ``hist`` component's isolation scope, where nothing above ``hist`` may be imported."""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.errors import FormatError, UnsupportedFeatureError
from xrdroot.sparse import SparseHistogram


#: An axis as a file describes one: ``fNbins`` bins between ``fXmin`` and ``fXmax``.
def _axis(
    nbins: int, name: str = "xaxis", low: float = 0.0, high: float = 1.0
) -> dict[str, object]:  # fmt: skip
    return {
        "TNamed": {"fName": name, "fTitle": name},
        "fNbins": nbins,
        "fXmin": low,
        "fXmax": high,
        "fXbins": np.linspace(low, high, nbins + 1),
    }


def _members(chunks: list[dict[str, object]], **named: object) -> dict[str, object]:
    """What a ``THnSparse`` holds: two axes, the chunks of bins, and the ``TNamed`` members."""
    return {
        "TNamed": {"fName": named.get("name", "hn"), "fTitle": named.get("title", "a sparse one")},
        "fEntries": named.get("entries", 3.0),
        "fAxes": [_axis(3, "xaxis"), _axis(4, "yaxis")],
        "fBinContent": chunks,
    }


def _chunk(
    pairs: list[tuple[int, int]],
    contents: list[float],
    squares: list[float] | None = None,
) -> dict:
    """One ``THnSparseArrayChunk``: a bin's two coordinates packed into one byte, little end first.

    Three bins along x need three bits and four along y three more, so a pair is one byte:
    ``x | y << 3``. ``coordinates()`` takes one back off each, so a stored 1 is the first bin.
    """
    packed = np.array([x | (y << 3) for x, y in pairs], dtype=np.int8)
    chunk: dict[str, object] = {
        "fSingleCoordinateSize": 1,
        "fCoordinates": packed,
        "fContent": np.array(contents),
    }
    if squares is not None:
        chunk["fSumw2"] = np.array(squares)
    return chunk


def test_a_sparse_histograms_bins_are_read_from_the_coordinates_a_file_packs():
    sparse = SparseHistogram("THnSparseD", _members([_chunk([(1, 1), (4, 5)], [2.0, 1.0])]))
    assert (sparse.name, sparse.title, sparse.entries) == ("hn", "a sparse one", 3.0)
    assert sparse.shape == (3, 4)  # the axes' bins, without the two at each end
    assert len(sparse) == 2  # only the bins that hold something are kept
    assert sparse.coordinates().tolist() == [[0, 0], [3, 4]]  # counted from zero, as ROOT does
    assert sparse.values().tolist() == [2.0, 1.0]
    assert "THnSparseD 'hn' of 2 dimensions, 2 bins filled" in repr(sparse)


def test_a_sparse_histogram_laid_out_densely_puts_each_bin_where_it_belongs():
    """Two bins inside the axes and one past them: the plain grid holds the first two, and the grid
    ``flow=True`` asks for holds all three, each axis having gained its two bins at the ends."""
    sparse = SparseHistogram(
        "THnSparseD", _members([_chunk([(1, 1), (3, 4), (4, 5)], [2.0, 1.0, 5.0])])
    )
    dense = sparse.to_dense()
    assert dense.shape == (3, 4)  # x has three bins and y four, without the ends
    assert (dense[0, 0], dense[2, 3]) == (2.0, 1.0)
    assert dense.sum() == 3.0  # the bin past the axes is not in it
    with_flow = sparse.to_dense(flow=True)
    assert with_flow.shape == (5, 6)  # each axis gains its two bins at the ends
    assert (with_flow[1, 1], with_flow[3, 4], with_flow[4, 5]) == (2.0, 1.0, 5.0)
    assert with_flow.sum() == 8.0


def test_a_dense_grid_past_the_limit_is_refused_rather_than_laid_out():
    sparse = SparseHistogram("THnSparseD", _members([_chunk([(1, 1)], [1.0])], name="wide"))
    sparse.axes = tuple(type(axis)(_axis(10_000, axis.name)) for axis in sparse.axes)
    with pytest.raises(UnsupportedFeatureError, match="laid out densely"):
        sparse.to_dense()


def test_a_weighted_sparse_histogram_keeps_the_squares_of_its_weights():
    weighted = SparseHistogram(
        "THnSparseD", _members([_chunk([(1, 1)], [2.0], squares=[1.5])])
    )
    assert weighted.weighted is True
    assert weighted.variances().tolist() == [1.5]
    plain = SparseHistogram("THnSparseD", _members([_chunk([(1, 1)], [2.0])]))
    assert plain.weighted is False
    assert plain.variances().tolist() == [2.0]  # a count without weights is its own variance


def test_a_sparse_histogram_without_axes_or_bins_says_so():
    with pytest.raises(FormatError, match="written without its axes or its bins"):
        SparseHistogram("THnSparseD", {"TNamed": {"fName": "hn"}})


def test_a_chunk_whose_bins_do_not_fit_its_axes_says_so():
    short = {
        "fSingleCoordinateSize": 2, "fCoordinates": np.zeros(2, np.int8), "fContent": np.zeros(1),
    }
    with pytest.raises(FormatError, match="cannot be told apart"):
        SparseHistogram("THnSparseD", _members([short]))
    few = {
        "fSingleCoordinateSize": 1, "fCoordinates": np.zeros(3, np.int8), "fContent": np.zeros(1),
    }
    with pytest.raises(FormatError, match="contents for"):
        SparseHistogram("THnSparseD", _members([few]))


def test_a_sparse_histogram_of_no_bins_is_empty_rather_than_broken():
    empty = SparseHistogram("THnSparseD", _members([]))
    assert (len(empty), empty.values().tolist()) == (0, [])
    assert empty.to_dense().shape == (3, 4)
    assert empty.to_dense().sum() == 0.0
