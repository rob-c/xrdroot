"""The statistics of a histogram, as ROOT's ``GetMean``, ``GetStdDev`` and the rest give them.

ROOT makes them from the running sums every fill adds to, falling back on
the bin centres only when those sums have been thrown away; both are checked
here. The expected numbers are worked out by hand from ROOT's formulas, with
fills chosen so that the arithmetic is exact, and from the sums ROOT itself
wrote into ``gauss-h1.root``, ``gauss-h2.root`` and ``tprofile.root``.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from xrdroot import Histogram, Profile, open_root

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def gauss():
    with open_root(str(DATA / "gauss-h1.root")) as one:
        with open_root(str(DATA / "gauss-h2.root")) as two:
            yield one["h1d"], two["h2d"]


def filled(*values, axis=(10, 0.0, 10.0), weight=None):
    h = Histogram.book("h", axis)
    h.fill(list(values), weight=weight)
    return h


def test_the_mean_and_spread_are_made_from_what_root_summed(gauss):
    h1d, h2d = gauss
    core = h1d.members["TH1"]
    mean = core["fTsumwx"] / core["fTsumw"]
    assert h1d.mean() == mean
    assert h1d.std() == math.sqrt(abs(core["fTsumwx2"] / core["fTsumw"] - mean * mean))
    th2 = h2d.members["TH2"]
    assert h2d.mean(1) == th2["fTsumwy"] / th2["TH1"]["fTsumw"]
    assert h2d.effective_entries == th2["TH1"]["fTsumw"] ** 2 / th2["TH1"]["fTsumw2"]


def test_the_moments_of_a_few_fills_are_what_roots_formulas_make_of_them():
    h = filled(1.0, 2.0, 3.0, 4.0)
    assert h.mean() == 2.5
    assert h.std() == math.sqrt(1.25)
    assert h.effective_entries == 4
    assert h.mean_error() == math.sqrt(1.25) / 2
    assert h.std_error() == math.sqrt(1.25 / 8)


def test_the_skewness_and_kurtosis_are_taken_about_the_mean_over_the_bins():
    h = filled(0.5, 1.5, 1.5, 2.5, axis=(3, 0.0, 3.0))
    assert h.skewness() == 0.0
    spread = math.sqrt(0.5)
    assert h.kurtosis() == 2 / (4 * (spread * spread * spread * spread)) - 3  # -1, near enough
    lopsided = filled(0.5, 0.5, 2.5, axis=(3, 0.0, 3.0))
    assert lopsided.skewness() == pytest.approx(0.7071067811865476, abs=1e-15)
    assert h.skewness_error() == math.sqrt(6 / 4)
    assert h.kurtosis_error() == math.sqrt(24 / 4)


def test_an_empty_histogram_has_no_moments_and_a_skewness_of_nan():
    h = Histogram.book("h", (2, 0, 1))
    assert (h.mean(), h.std(), h.mean_error(), h.std_error(), h.effective_entries) == (
        0,
        0,
        0,
        0,
        0,
    )
    assert math.isnan(h.skewness()) and math.isnan(h.kurtosis())
    assert (h.skewness_error(), h.kurtosis_error()) == (0.0, 0.0)


def test_a_histogram_whose_sums_are_gone_takes_its_moments_from_the_bin_centres():
    h = Histogram.book("h", (4, 0.0, 4.0))
    h._cells()[:] = [0, 1, 2, 3, 4, 0]
    h.members["TH1"]["fEntries"] = 10.0  # as SetBinContent leaves it: entries, no total weight
    assert h.mean() == 2.5  # (0.5 + 3 + 7.5 + 14) / 10
    assert h.std() == 1.0
    assert h.effective_entries == 10.0  # sum w = sum w**2 when the errors are roots of counts


def test_a_three_dimensional_histogram_from_its_bins_sums_every_pair_of_axes():
    h = Histogram.book("h", (2, 0, 2), (2, 0, 2), (2, 0, 2))
    h.fill([0.5, 1.5], [0.5, 1.5], [1.5, 0.5], weight=[1.0, 3.0])
    by_fills = [h.mean(axis) for axis in range(3)]
    h.members["TH3"]["TH1"]["fTsumw"] = 0.0
    assert [h.mean(axis) for axis in range(3)] == by_fills == [1.25, 1.25, 0.75]


def test_an_axis_the_histogram_does_not_have_is_refused():
    with pytest.raises(ValueError, match="axis=1 is not an axis of 'h', which has 1"):
        filled(1.0).mean(1)
    with pytest.raises(ValueError, match="axis=2 is not a binned axis of 'h', which has 1"):
        filled(1.0).skewness(2)


def test_a_profile_averages_the_value_along_the_axis_after_its_own(gauss):
    with open_root(str(DATA / "tprofile.root")) as handle:
        p1d = handle["p1d"]
    assert p1d.mean(1) == p1d.members["fTsumwy"] / p1d.members["TH1D"]["TH1"]["fTsumw"]
    p = Profile.book("p", (2, 0, 2))
    p.fill([0.5, 1.5], [1.0, 3.0])
    assert (p.mean(0), p.mean(1)) == (1.0, 2.0)
    p.members["TH1D"]["TH1"]["fTsumw"] = 0.0  # a profile falls back on its bins by weight alone
    assert (p.mean(0), p.mean(1), p.std(1)) == (1.0, 2.0, 1.0)
    heavy = Profile.book("q", (1, 0, 1), (1, 0, 1), (1, 0, 1))
    heavy.fill(0.5, 0.5, 0.5, 4.0, weight=2.0)
    heavy.members["TH3D"]["TH3"]["TH1"]["fTsumw"] = 0.0
    assert (heavy.mean(3), heavy.effective_entries) == (4.0, 1.0)


def test_an_integral_is_the_sum_of_the_bins_it_names_and_by_default_none_of_the_flow():
    h = filled(-1.0, 0.5, 1.5, 1.5, 9.5, 20.0, axis=(10, 0.0, 10.0))
    assert h.integral() == 4
    assert h.integral(0, 11) == 6
    assert h.integral(2, 2) == 2
    assert h.integral(-5, 2) == 4  # a low bin below the underflow is the underflow
    assert h.integral(3, 1) == 2  # a high bin below the low one means up to the overflow
    assert h.integral_error(0, 11) == math.sqrt(6)


def test_an_integral_with_width_multiplies_each_bin_by_its_size():
    h = Histogram.book("h", [0.0, 1.0, 3.0])
    h.fill([0.5, 2.0, 2.0], weight=2.0)
    assert h.integral(width=True) == 2 * 1 + 4 * 2
    assert h.integral_error(width=True) == math.sqrt(4 * 1 + 8 * 4)
    two = Histogram.book("h2", (2, 0, 2), (2, 0, 4))
    two.fill([0.5, 1.5], [1.0, 3.0])
    assert two.integral(width=True) == 4.0
    assert two.integral(1, 1) == 1  # a number is for x alone; y takes all its bins
    assert two.integral((1, 2), (2, 2)) == 1


def test_interpolation_runs_a_line_between_neighbouring_bin_centres():
    h = Histogram.book("h", (3, 0.0, 3.0))
    h.fill([0.5, 1.5, 1.5, 2.5, 2.5, 2.5])
    assert h.interpolate(1.0) == 1.5
    assert h.interpolate([0.0, 0.5, 2.0, 2.9]).tolist() == [1.0, 1.0, 2.5, 3.0]
    assert math.isnan(h.interpolate(np.nan))
    with pytest.raises(ValueError, match="needs one axis, and it has 2"):
        Histogram.book("h2", (1, 0, 1), (1, 0, 1)).interpolate(0.5)


def test_a_bin_is_found_by_roots_global_number_x_fastest():
    h = Histogram.book("h", (2, 0, 2), (3, 0, 3), (4, 0, 4))
    assert h.find_bin(0.5, 0.5, 0.5) == 1 + 4 * (1 + 5 * 1)
    assert h.find_bin([1.5, -1], [2.5, 5], [3.5, 0.5]).tolist() == [
        2 + 4 * (3 + 5 * 4),
        0 + 4 * (4 + 5),
    ]
    with pytest.raises(ValueError, match="one coordinate per axis, not 1"):
        h.find_bin(0.5)


def test_the_largest_and_smallest_bins_are_found_on_the_axes_or_as_set():
    h = filled(1.5, 1.5, 3.5, 3.5, -5.0, -5.0, -5.0, axis=(5, 0.0, 5.0))
    assert (h.maximum(), h.minimum()) == (2.0, 0.0)
    assert (h.argmax(), h.argmin()) == (1, 0)  # the first of equals, as ROOT's loop meets them
    h.members["TH1"]["fMaximum"] = 10.0
    assert h.maximum() == 10.0
    two = Histogram.book("h2", (2, 0, 2), (2, 0, 2))
    two.fill([1.5, 0.5], [0.5, 1.5], weight=[1.0, 3.0])
    assert two.argmax() == (0, 1) and two.argmin() == (0, 0)
