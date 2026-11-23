"""``TH2Poly``'s bin errors and ``TProfile2Poly``'s error option, as ROOT keeps them.

ROOT's own programs for both are ``hist/hist/test/test_TH2Poly_BinError.cxx`` and
``hist/hist/test/test_tprofile2poly.cxx``: a poly keeps a sum of squared weights
per bin - ``fSumw2`` - and an error is its root, or, while no weight other than
one has been filled, the root of the content; a profile's error is the spread of
its values or the error on their mean, which ``SetErrorOption`` chooses.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect
from xrdroot.polybins import PolyBins


def histogram() -> Any:
    """The three bins of ROOT's own test, over a range that covers them all."""
    found = ROOT.TH2Poly()
    found.AddBin(3, [0.0, 5.0, 6.0], [0.0, 0.0, 5.0])
    found.AddBin(4, [0.0, -1.0, -1.0, 0.0], [0.0, 0.0, -1.0, 3.0])
    found.AddBin(5, [4.0, 3.0, 0.0, 1.0, 2.4], [4.0, 3.7, 1.0, 3.7, 2.5])
    return found


def test_an_unweighted_fill_leaves_the_error_the_root_of_the_content() -> None:
    found = histogram()
    for _ in range(9):
        found.Fill(0.1, 0.01)
    expect((found.GetBinError(1), 3.0), (found.GetBinError(2), 0.0))
    found.Fill(-0.5, -0.5, 1.0)
    found.Fill(2.0, -0.5, 1.0)
    expect((found.GetBinError(-5), math.sqrt(2.0)), (found.GetSumw2N(), 0))


def test_a_weight_that_is_not_one_starts_the_sums_of_squared_weights() -> None:
    found = histogram()
    for weight in (4.0, 3.0):
        found.Fill(-0.5, -0.5, weight)
    assert found.GetSumw2N() == 3 + 9
    expect((found.GetBinError(-5), 5.0), (found.GetBinError(-1), 0.0))
    found.Fill(-2.0, 10.0, 3.0)
    expect((found.GetBinError(-1), 3.0))


def test_a_set_error_is_kept_while_a_set_content_does_not_set_one() -> None:
    found = histogram()
    found.SetBinContent(1, 10.0)
    found.SetBinContent(-9, 4.0)
    expect((found.GetBinError(1), math.sqrt(10.0)), (found.GetBinError(-9), 2.0))
    for number, error in ((1, 1.5), (2, 2.5), (-8, 1.0)):
        found.SetBinError(number, error)
    expect((found.GetBinError(1), 1.5), (found.GetBinError(2), 2.5), (found.GetBinError(-8), 1.0))
    found.SetBinContent(-1, 3.0)
    expect((found.GetBinError(-1), 0.0))


def test_a_bin_or_a_region_that_is_not_there_has_no_error() -> None:
    found = histogram()
    for number in (0, 4, -10):
        expect((found.GetBinError(number), 0.0))
        found.SetBinError(number, 2.0)  # nothing to set: no bin of that number
    assert found.GetSumw2N() == 0


def test_the_sums_of_squared_weights_can_be_dropped_and_started_again() -> None:
    found = histogram()
    found.Fill(0.1, 0.01, 2.0)
    found.Sumw2(False)
    expect((found.GetBinError(1), math.sqrt(2.0)))  # the root of the content again
    found.Sumw2()
    expect((found.GetBinError(1), math.sqrt(2.0)))  # seeded from the content, as ROOT seeds it


def test_a_copy_of_a_polygon_histogram_keeps_its_errors_and_its_moments() -> None:
    found = histogram()
    found.SetBinContent(1, 10.0)
    found.SetBinError(1, 2.5)
    copy = ROOT.TH2Poly(found)
    expect((copy.GetName(), found.GetName()), (copy.GetNumberOfBins(), 3))
    expect((copy.GetBinContent(1), 10.0), (copy.GetBinError(1), 2.5))
    copy.Fill(0.1, 0.01, 2.0)
    expect((found.GetBinContent(1), 10.0))  # the copy shares nothing with it
    expect((copy.GetMean(1), 0.1), (copy.GetStdDev(1), 0.0), (copy.GetMean(0), 0.0))


def test_a_histogram_added_keeps_the_sums_of_squared_weights_of_both() -> None:
    def made(weight: float) -> Any:
        poly = ROOT.TH2Poly()
        poly.AddBin(0.0, 0.0, 1.0, 1.0)
        poly.Fill(0.5, 0.5, weight)
        return poly

    total, other = made(2.0), made(3.0)
    assert total.Add(other, 2.0)
    expect((total.GetSumw2N(), 10), (total.GetBinError(1), math.sqrt(4.0 + 4.0 * 9.0)))
    plain = made(1.0)
    plain.Fill(0.5, 0.5)
    assert plain.Add(made(1.0))
    expect((plain.GetSumw2N(), 0))  # neither keeps squared weights, so neither does the sum
    keeper = made(1.0)
    keeper.Fill(0.5, 0.5)
    assert keeper.Add(other)
    expect((keeper.GetSumw2N(), 10))  # the addend's squared weights are kept, times one


def test_a_named_fill_adds_to_the_squares_too_once_the_histogram_keeps_them() -> None:
    found = histogram()
    found.SetBinName(1, "first")
    found.Fill("first", 2.0)
    expect((found.GetBinContent(1), 2.0), (found.GetSumw2N(), 0))
    found.Fill(0.1, 0.01, 3.0)  # a weight other than one: the sums start
    found.Fill("first", 2.0)
    expect((found.GetBinError(1), math.sqrt(2.0 + 9.0 + 4.0)))


def test_a_moment_thrown_away_by_a_region_has_no_mean_and_no_spread() -> None:
    found = histogram()
    found.Fill(-5.0, 5.0)  # the region above the range: no moment, only the region's content
    expect((found.GetEffectiveEntries(), 0.0), (found.GetMean(1), 0.0))
    expect((found.GetMeanError(1), 0.0), (found.GetStdDevError(1), 0.0))


def test_the_search_grid_deals_every_bin_into_a_cell_a_point_can_be_found_in() -> None:
    bins = PolyBins((0.0, 4.0), (0.0, 4.0), floating=False)
    for at in range(4):
        bins.add([([float(at), float(at) + 1.0, float(at) + 1.0, float(at)],
                   [0.0, 0.0, 4.0, 4.0])])  # fmt: skip
    table = bins.grid()
    assert bins.grid() is table
    expect((bins.find(0.5, 3.5), 1), (bins.find(3.5, 3.5), 4), (bins.find(0.5, 9.5), -2))
    expect((bins.find(-1.0, 9.5), -1))
    expect((len(table), 4), (bins.find(3.5, 3.5), 4))  # a second look keeps the table
    bins.sumw2()
    bins.add([([0.0, 0.5, 0.5, 0.0], [3.0, 3.0, 3.5, 3.5])])
    expect((len(bins.squares), 14), (len(bins.grid()), 4))  # a fifth bin, and its square
    expect((bins.find(0.2, 3.2), 1))  # the first strip still holds the point
    copy = bins.copy()
    expect((copy.find(0.5, 0.5), 1), (len(copy.polygons), 5))


#: The point in the middle of one of the profile's bins, and the number of that bin.
MIDDLE = (5.5, 4.5)
MIDDLE_BIN = 78


def profile() -> Any:
    """A profile of ROOT's own test's bins: ten rectangles over -10 to 10 each way."""
    found = ROOT.TProfile2Poly()
    for i in range(-10, 10, 2):
        for j in range(-10, 10, 2):
            found.AddBin(float(i), float(j), float(i + 2), float(j + 2))
    return found


def test_a_profile_with_no_option_gives_the_error_on_each_mean() -> None:
    found = profile()
    for value in (1.0, 3.0, 5.0):
        found.Fill(*MIDDLE, value)
    expect((found.FindBin(*MIDDLE), MIDDLE_BIN), (found.GetBinContent(MIDDLE_BIN), 3.0))
    expect((found.GetBinEntries(MIDDLE_BIN), 3.0))
    expect((found.GetBinEffectiveEntries(MIDDLE_BIN), 3.0))
    spread = math.sqrt(35.0 / 3.0 - 9.0)
    expect((found.GetBinError(MIDDLE_BIN), spread / math.sqrt(3.0)))
    expect((found.GetErrorOption(), ""))


def test_the_spread_option_gives_the_spread_of_the_values_instead() -> None:
    found = profile()
    for value in (1.0, 3.0, 5.0):
        found.Fill(*MIDDLE, value, 2.0)
    found.SetErrorOption(ROOT.kERRORSPREAD)
    spread = math.sqrt((2.0 + 2.0 * 9.0 + 2.0 * 25.0) / 6.0 - 3.0 * 3.0)
    bin = MIDDLE_BIN
    assert found.GetBinError(bin) == pytest.approx(spread)
    assert found.GetErrorOption() == "s"
    found.SetErrorOption("")
    assert found.GetBinError(bin) == pytest.approx(
        spread / math.sqrt(found.GetBinEffectiveEntries(bin)))
    found.SetErrorOption("s")
    assert found.GetBinError(bin) == pytest.approx(spread)
    found.SetErrorOption(ROOT.kERRORMEAN)
    assert found.GetErrorOption() == ""


def test_a_profile_bin_nothing_was_filled_into_has_no_error_and_no_entries() -> None:
    found = profile()
    found.SetErrorOption(ROOT.kERRORSPREAD)
    expect((found.GetBinError(MIDDLE_BIN), 0.0), (found.GetBinEffectiveEntries(MIDDLE_BIN), 0.0))


def test_a_profile_keeps_the_mean_of_what_missed_every_bin_in_its_region() -> None:
    found = profile()
    for value, weight in ((1.0, 1.0), (3.0, 1.0)):
        found.Fill(100.0, 100.0, value, weight)
    expect((found.FindBin(100.0, 100.0), -3), (found.GetBinContent(-3), 2.0))
    expect((found.GetBinEntries(-3), 2.0), (found.GetBinContent(MIDDLE_BIN), 0.0))


def test_a_profile_merged_takes_the_sums_of_every_bin_of_the_other() -> None:
    mine, other = profile(), profile()
    mine.Fill(*MIDDLE, 1.0)
    other.Fill(*MIDDLE, 3.0)
    assert mine.Merge([other]) == 1
    expect((mine.GetBinContent(MIDDLE_BIN), 2.0), (mine.GetBinEntries(MIDDLE_BIN), 2.0))
    assert mine.Merge([profile()]) == 1
    stranger = ROOT.TProfile2Poly()
    stranger.AddBin(0.0, 0.0, 1.0, 1.0)
    assert mine.Merge([stranger]) == 0  # binned otherwise, as ROOT refuses it


def test_a_profile_reset_forgets_its_sums_and_its_moments() -> None:
    found = profile()
    found.Fill(*MIDDLE, 3.0)
    found.Reset()
    expect((found.GetBinContent(MIDDLE_BIN), 0.0), (found.GetBinEntries(MIDDLE_BIN), 0.0))
    expect((found.GetEffectiveEntries(), 0.0))


def test_a_copy_of_a_profile_keeps_the_sums_of_its_bins() -> None:
    found = profile()
    found.Fill(*MIDDLE, 3.0)
    found.SetErrorOption(ROOT.kERRORSPREAD)
    copy = ROOT.TProfile2Poly(found)
    expect((copy.GetBinContent(MIDDLE_BIN), 3.0), (copy.GetBinEntries(MIDDLE_BIN), 1.0))
    expect((copy.GetErrorOption(), "s"))
    expect((copy.GetBinError(MIDDLE_BIN), found.GetBinError(MIDDLE_BIN)))


def test_a_profile_answers_the_moments_of_its_three_axes() -> None:
    found = profile()
    found.Fill(*MIDDLE, 3.0)
    expect((found.GetMean(1), 5.5), (found.GetMean(2), 4.5), (found.GetMean(3), 3.0))
    expect((found.GetMean(4), 0.0), (found.GetStdDev(3), 0.0))
    expect((found.GetMeanError(1), 0.0), (found.GetMean(11), 0.0))


def test_a_spread_is_answered_for_an_axis_the_profile_has_and_no_other() -> None:
    found = profile()
    found.Fill(*MIDDLE, 3.0)
    expect((found.GetStdDev(9), 0.0), (found.GetStdDev(1), 0.0))
    assert found.GetStdDev(13) == 0.0  # one entry has no spread to put an error on


def test_the_error_on_a_spread_is_the_spread_over_twice_the_entries() -> None:
    found = profile()
    found.Fill(*MIDDLE, 3.0)
    found.Fill(*MIDDLE, 1.0)
    spread = found.GetStdDev(3)
    assert spread > 0.0
    on_spread = math.sqrt(spread * spread / (2 * found.GetEffectiveEntries()))
    assert found.GetStdDev(13) == pytest.approx(on_spread)


def test_a_merge_adds_the_fills_that_missed_the_bins_too() -> None:
    other = profile()
    other.Fill(-20.0, 20.0, 7.0)  # a region outside every bin
    region = min(other._sums)  # the region number ROOT gives that point
    assert region < 0
    merged = profile()
    assert merged.Merge([other]) == 1
    expect((merged.GetBinContent(region), 7.0), (merged.GetBinContent(MIDDLE_BIN), 0.0))


def test_a_merge_of_profiles_of_different_binning_says_so_and_adds_nothing() -> None:
    one = profile()
    other = ROOT.TProfile2Poly("fewer", "fewer", 0.0, 10.0, 0.0, 10.0)
    other.AddBin(0.0, 0.0, 10.0, 10.0)
    assert other.Merge([one]) == 0


def test_a_profile_the_constructor_gave_a_range_registers_no_bin_and_reads_zero() -> None:
    """ROOT's ``TProfile2Poly(name, title, nx, xlo, xhi, ny, ylo, yhi)`` builds no bins at
    all - the 6.40 oracle answers ``GetNumberOfBins() == 0`` - so every read is 0.0, regions
    included, however many fills it took. A profile built from ``AddBin`` is the other case:
    the oracle answers ``GetNumberOfBins() == 100`` and a region holding the mean of what
    missed every bin, which the test above pins."""
    ranged = ROOT.TProfile2Poly("t", "t", 10, 1, 4, 10, 1, 4)
    for value in (1.0, 3.0):
        ranged.Fill(0.5, 0.5, value, 1.0)  # outside every cell it might have had
    expect(
        (ranged.GetNumberOfBins(), 0),
        (ranged.GetBinContent(-7), 0.0),
        (ranged.GetBinContent(1), 0.0),
        (ranged[(ROOT.uhi.loc(1), ROOT.uhi.loc(1))], 0.0),
    )
