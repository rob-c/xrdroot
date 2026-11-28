"""A ``TProfile``'s numbers: the mean in each bin, and the four errors ROOT lets it report.

A profile is booked and filled here with plain arrays, so no file is opened and the test
stays inside the ``hist`` component's own isolation scope. What is *not* the same as a
histogram is what the bins mean: ``values`` is the mean of what was filled, ``bin_entries``
the weight that went into it, ``sums`` the sum of ``w*y`` the file actually stores, and
``errors`` whichever of ROOT's four error modes was asked for - the error on the mean, the
spread, the spread of integers, or one over the root of the weights.

Every error in the first half of the file was read off the ROOT 6.40.04 oracle for the same
four fills (three at ``y`` 1, 2, 3 in the first bin and one at ``y`` 4 in the second) and is
quoted beside the assertion. The last two bits of a square root are the C library's, so
those are held to ``REL``.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np
import pytest

from xrdroot import UnsupportedFeatureError
from xrdroot.profile import Profile

#: How close a root of a mean or a weight is to ROOT's: Apple's libm and glibc round
#: ``sqrt`` differently in the last bit, so bit-exactness is only claimed on ROOT's machine.
REL = 1e-12


def _three_and_one() -> Profile:
    """ROOT's ``TProfile p("p", "", 2, 0, 2)`` filled at (0.5, 1), (0.5, 2), (0.5, 3), (1.5, 4).

    The first bin averages three values to 2 and the second holds 4 alone; every number in
    the first half of this file is the oracle's answer for exactly this profile.
    """
    made = Profile.book("p", (2, 0.0, 2.0))
    made.fill([0.5, 0.5, 0.5, 1.5], [1.0, 2.0, 3.0, 4.0])
    return made


def test_a_profile_s_bins_are_the_means_of_what_was_filled():
    """``values`` is the mean and ``bin_entries`` the weight behind it, where a histogram's
    bin would hold the total: ROOT reads 2 and 4 for the means, with three and one entries,
    and the file stores the sums 6 and 4 that dividing them gives."""
    found = _three_and_one()
    assert found.values().tolist() == [2.0, 4.0]
    assert found.bin_entries().tolist() == [3.0, 1.0]
    assert found.sums().tolist() == [6.0, 4.0]


def test_the_error_on_the_mean_is_the_spread_over_the_root_of_the_count():
    """ROOT's default error mode: 0.47140452079103184 in the bin of three, where the spread
    of 1, 2 and 3 is 0.81649658092772626 over sqrt(3) - and zero in a bin of one value."""
    found = _three_and_one()
    assert found.error_mode == ""  # what a profile booked without an option reports
    assert found.errors().tolist() == pytest.approx([0.47140452079103184, 0.0], rel=REL)
    assert found.spread().tolist() == pytest.approx([0.81649658092772626, 0.0], rel=REL)


def test_the_spread_mode_is_the_spread_itself():
    """``"s"`` is the standard deviation of what was filled, not an error on the mean - the
    mode a physics plot wants when it is showing the spread of a distribution."""
    found = _three_and_one()
    spread = found.errors(error_mode="s")
    assert spread.tolist() == pytest.approx([0.81649658092772626, 0.0], rel=REL)
    assert found.errors(error_mode="S").tolist() == found.errors(error_mode="s").tolist()


def test_the_integer_mode_gives_a_bin_of_one_value_the_spread_of_a_flat_distribution():
    """``"i"`` is the error on the mean for values that are whole numbers: a bin whose spread
    is zero is floored at ``sqrt(1/12)``, the spread of a uniform integer step - ROOT's
    0.28867513459481292 where the spread itself is nothing."""
    found = _three_and_one()
    assert found.errors(error_mode="i").tolist() == pytest.approx(
        [0.47140452079103184, 0.28867513459481292], rel=REL
    )
    assert math.sqrt(1 / 12) == pytest.approx(0.28867513459481292, rel=REL)


def test_the_gaussian_mode_is_one_over_the_root_of_the_weights():
    """``"g"`` is for fills weighted by one over their variance, where the error is
    ``1/sqrt(sum of weights)``: 0.57735026918962584 for three entries and 1 for one."""
    found = _three_and_one()
    gaussian = found.errors(error_mode="g")
    assert gaussian.tolist() == pytest.approx([0.57735026918962584, 1.0], rel=REL)


def test_a_weighted_profile_counts_its_weights_over_the_squares_of_them():
    """Two fills of weight 1 and 3 are four units of weight in 1.6 effective entries, so the
    spread 0.8660254037844386 over the root of 1.6 is ROOT's 0.68465319688145754."""
    weighted = Profile.book("w", (2, 0.0, 2.0))
    weighted.fill([0.5, 0.5], [1.0, 3.0], weight=[1.0, 3.0])
    assert weighted.weighted  # the squares of the weights are being kept
    assert weighted.values().tolist() == [2.5, 0.0]
    assert weighted.bin_entries().tolist() == [4.0, 0.0]
    assert weighted.counts().tolist() == pytest.approx([1.6, 0.0], rel=REL)
    assert weighted.spread().tolist() == pytest.approx([0.8660254037844386, 0.0], rel=REL)
    on_the_mean = weighted.errors(error_mode="")
    assert on_the_mean.tolist() == pytest.approx([0.68465319688145754, 0.0], rel=REL)


def test_the_error_option_a_profile_was_booked_with_is_the_one_it_uses():
    """``SetErrorOption`` is fixed when the profile is booked and every later call uses it,
    which is what a file's ``fErrorMode`` is for - until a method is told otherwise."""
    booked = Profile.book("s", (2, 0.0, 2.0), error_option="s")
    booked.fill([0.5, 0.5, 0.5, 1.5], [1.0, 2.0, 3.0, 4.0])
    assert booked.error_mode == "s"
    assert booked.errors().tolist() == booked.spread().tolist()
    assert booked.errors(error_mode="").tolist()[0] == pytest.approx(
        0.47140452079103184, rel=REL
    )


def test_an_error_mode_root_does_not_have_is_refused_with_the_four_it_does():
    """ROOT has four error modes and no others, so a fifth letter is a mistake worth naming
    rather than reading as the default."""
    with pytest.raises(ValueError, match="error_mode='q' is not one of ROOT's"):
        _three_and_one().errors(error_mode="q")


def test_filling_a_profile_takes_a_coordinate_per_axis_and_then_the_value():
    """A one-axis profile takes two arrays - the coordinate and the value - and three is a
    mistake whose sentence says how many it needed.

    The extra axis is what the moments describe: a profile of one binned axis has an x and
    the value, which is why its running sums describe two.
    """
    found = _three_and_one()
    with pytest.raises(ValueError, match="takes a coordinate for each and then the value"):
        found.fill([0.5], [1.0], [2.0])  # type: ignore[call-arg]
    assert found._moment_axes() == len(found.axes) + 1


def test_a_sum_or_a_density_of_means_is_refused():
    """A histogram's bins add up and normalise because they are counts; a profile's are
    means, so both are refused, and each sentence names what to ask for instead."""
    found = _three_and_one()
    with pytest.raises(UnsupportedFeatureError, match="a sum of means is not a thing"):
        found.sum()
    with pytest.raises(UnsupportedFeatureError, match="so they have no density"):
        found.density()


def test_a_profile_that_kept_no_squares_has_no_spread_to_give():
    """A profile sums ``w*y*y`` in its core and the squares of the weights beside it; a file
    written without the first has nothing to work a spread out of, and says so rather than
    reporting zero as if the spreads were all nothing."""
    row = Profile.book("q", (2, 0.0, 2.0)).members
    row["TH1D"]["TH1"]["fSumw2"] = np.zeros(0)  # what a file without Sumw2 holds
    made = Profile("TProfile", row)
    assert made.counts().tolist() == [0.0, 0.0]  # the weights are still there
    for asked in ("spread", "errors"):
        with pytest.raises(UnsupportedFeatureError, match="keeps no sum of the squares"):
            getattr(made, asked)()


def test_a_profile_whose_weights_are_fewer_than_its_bins_cannot_give_a_mean():
    """The mean of a bin is its sum over its weight, so a row holding weights for only some
    of the bins cannot answer for the rest; the refusal says how many it has."""
    row = Profile.book("s", (2, 0.0, 2.0)).members
    row["fBinEntries"] = np.zeros(2)  # four cells, two weights
    with pytest.raises(UnsupportedFeatureError, match="keeps the weights of only 2 of them"):
        Profile("TProfile", row)


def test_turning_a_profile_into_a_hist_hist_needs_the_package(monkeypatch: Any):
    """``to_hist`` imports ``hist`` when it is asked, so a machine without it is told which
    command installs it - and that ``values``/``errors`` need nothing more."""
    monkeypatch.setitem(sys.modules, "hist", None)
    with pytest.raises(UnsupportedFeatureError, match="pip install hist"):
        _three_and_one().to_hist()


def test_turning_it_into_a_hist_hist_carries_the_means_and_the_weights():
    """A profile arrives as a histogram of ``Mean`` storage - or ``WeightedMean`` when the
    squares of the weights were kept - whose view holds the weight and the mean of every
    bin, flow included, so nothing is recomputed on the way across."""
    pytest.importorskip("hist")
    plain = _three_and_one().to_hist()
    assert plain.kind == "MEAN"
    assert plain.values().tolist() == [2.0, 4.0]
    view = plain.view(flow=True)
    assert np.asarray(view["count"]).tolist() == [0.0, 3.0, 1.0, 0.0]
    assert np.asarray(view["value"]).tolist() == [0.0, 2.0, 4.0, 0.0]
    weighted = Profile.book("w", (2, 0.0, 2.0))
    weighted.fill([0.5, 0.5], [1.0, 3.0], weight=[1.0, 3.0])
    carried = weighted.to_hist().view(flow=True)
    assert "sum_of_weights_squared" in carried.dtype.names  # WeightedMean, not Mean
    assert np.asarray(carried["sum_of_weights"]).tolist() == [0.0, 4.0, 0.0, 0.0]
    assert np.asarray(carried["sum_of_weights_squared"]).tolist() == [0.0, 10.0, 0.0, 0.0]
