"""``TKDTreeBinning``: a k-d tree over a data set, whose leaves are a grid's bins.

``TH2Poly`` is what makes it useful - a data-driven binning with no empty bin - and
ROOT's own ``test_THBinIterator`` builds one from a million points, so the numbers here
are small enough to check by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT

#: How many points the tests bin, and how many bins they ask for.
POINTS, BINS = 200, 20


def spread(size: int = POINTS, dim: int = 2) -> np.ndarray:
    """``size`` points of ``dim`` dimensions as ROOT keeps them: one run of each dimension."""
    points = np.random.default_rng(11).normal(0.0, 1.0, (size, dim))
    return points.T.reshape(-1)


def test_every_point_lands_in_exactly_one_bin() -> None:
    found = ROOT.TKDTreeBinning(POINTS, 2, spread(), BINS)
    assert (found.GetNBins(), found.GetDim()) == (BINS, 2)
    assert sum(found.GetBinContent(bin) for bin in range(BINS)) == POINTS


def test_a_bin_is_the_box_of_the_points_it_counts() -> None:
    data = spread()
    found = ROOT.TKDTreeBinning(POINTS, 2, data, BINS)
    low, high = found.GetBinsMinEdges(), found.GetBinsMaxEdges()
    points = np.asarray(data).reshape(2, POINTS).T
    assert np.shape(low) == (BINS * 2,)
    for bin in range(BINS):
        inside = np.ones(POINTS, dtype=bool)
        for axis in range(2):
            low_edge, high_edge = low[bin * 2 + axis], high[bin * 2 + axis]
            inside &= (low_edge <= points[:, axis]) & (points[:, axis] <= high_edge)
        assert found.GetBinContent(bin) <= int(inside.sum())
        assert found.GetBinMinEdges(bin).tolist() == low[bin * 2 : bin * 2 + 2].tolist()
        assert found.GetBinMaxEdges(bin).tolist() == high[bin * 2 : bin * 2 + 2].tolist()


def test_the_data_extremes_are_the_ones_the_data_has() -> None:
    data = spread()
    found = ROOT.TKDTreeBinning(POINTS, 2, data, 4)
    points = np.asarray(data).reshape(2, POINTS).T
    assert (found.GetDataMin(0), found.GetDataMax(0)) == (points[:, 0].min(), points[:, 0].max())
    assert (found.GetDataMin(1), found.GetDataMax(1)) == (points[:, 1].min(), points[:, 1].max())


def test_a_group_with_no_spread_is_still_cut_in_two() -> None:
    found = ROOT.TKDTreeBinning(8, 2, np.ones(16), 4)
    assert sum(found.GetBinContent(bin) for bin in range(4)) == 8
    assert (found.GetBinVolume(0), found.GetBinDensity(0)) == (0.0, 0.0)


def test_a_bins_volume_and_density_are_its_box_s() -> None:
    found = ROOT.TKDTreeBinning(POINTS, 2, spread(), 10)
    assert found.GetBinVolume(0) > 0.0
    assert found.GetBinDensity(0) == found.GetBinContent(0) / found.GetBinVolume(0)


def test_more_bins_than_points_is_refused_in_a_sentence() -> None:
    with pytest.raises(ValueError, match="cannot hold 20 bins"):
        ROOT.TKDTreeBinning(4, 1, [1.0, 2.0, 3.0, 4.0], 20)
    with pytest.raises(ValueError, match="cannot hold 0 bins"):
        ROOT.TKDTreeBinning(4, 1, [1.0, 2.0, 3.0, 4.0], 0)


def test_a_binning_becomes_a_polygons_bins_with_every_count_kept() -> None:
    bins = ROOT.TKDTreeBinning(400, 2, spread(400), 32)
    low, high = bins.GetBinsMinEdges(), bins.GetBinsMaxEdges()
    poly = ROOT.TH2Poly("kd", "kd", bins.GetDataMin(0), bins.GetDataMax(0),
                        bins.GetDataMin(1), bins.GetDataMax(1))  # fmt: skip
    for number in range(32):
        poly.AddBin(low[number * 2], low[number * 2 + 1], high[number * 2], high[number * 2 + 1])
        poly.SetBinContent(number + 1, bins.GetBinContent(number))
    assert (poly.GetNumberOfBins(), poly.Integral()) == (32, 400)
    assert sum(poly.GetBinContent(bin) for bin in ROOT.THistRange(poly)) == 400
