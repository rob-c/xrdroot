"""``TKDE``, ROOT's kernel density estimate, against the numbers ROOT prints for it.

Every value asserted here was printed by the ROOT 6.40.04 oracle for the same
fixture: ``TRandom3(1111)`` draws 20000 values, one in five from a Gaussian of
width 1 and the rest from one of width 7, both about 10, and the estimate is
asked at ``TAxis(21, -0.5, 20.5)``'s first twenty bin centres - the sample
``gtest/hist/hist/testTKDE`` uses.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError

#: The test's own guess of the density at its twenty points, from ROOT 6.40.04.
ROOT_VALUES = (
    0.0191838,
    0.0238081,
    0.0287209,
    0.0323661,
    0.0358628,
    0.0388449,
    0.0430033,
    0.0501361,
    0.0689577,
    0.103751,
    0.124368,
    0.103501,
    0.0701576,
    0.0521353,
    0.0452547,
    0.0409197,
    0.0357176,
    0.0320111,
    0.0269283,
    0.0223096,
)

FIXED = "KernelType:Gaussian;Iteration:Fixed;Mirror:noMirror;Binning:Unbinned"
ADAPTIVE = "KernelType:Gaussian;Iteration:Adaptive;Mirror:noMirror;Binning:Unbinned"


def mixed_sample(n: int, seed: int = 1111) -> np.ndarray:
    """``n`` values, one in five of width 1 about 10 and the rest of width 7."""
    random = ROOT.TRandom3(seed)
    drawn = np.empty(n)
    for at in range(n):
        drawn[at] = random.Gaus(10, 1) if random.Rndm() < 0.2 else random.Gaus(10, 7)
        # the test draws a second sample as it goes, which moves the stream on the same way
        random.Gaus(10, 1) if random.Rndm() < 0.2 else random.Gaus(10, 7)
    return drawn


def centres(count: int = 20) -> list[float]:
    """The centre of each of the first ``count`` bins of ``TAxis(21, -0.5, 20.5)``."""
    axis = ROOT.TAxis(21, -0.5, 20.5)
    return [axis.GetBinCenter(at + 1) for at in range(count)]


def fixed_estimate(n: int = 20000) -> ROOT.TKDE:
    return ROOT.TKDE(n, mixed_sample(n), 0.0, 20.0, FIXED, 1)


def test_a_fixed_estimate_is_the_density_root_prints_at_every_point():
    kde = fixed_estimate()
    found = [kde.GetValue(x) for x in centres()]
    assert found == pytest.approx(ROOT_VALUES, rel=2e-5)


def test_the_data_statistics_and_the_bandwidth_are_roots_to_the_last_digit():
    kde = fixed_estimate()
    assert kde.GetMean() == pytest.approx(9.950257504173605, rel=1e-12)
    assert kde.GetSigma() == pytest.approx(6.248649466352207, rel=1e-12)
    assert kde.GetFixedWeight() == pytest.approx(0.7417943288613867, rel=1e-12)
    assert kde.GetFixedWeight() == kde.GetFixedWeight()  # a bandwidth once made is kept


def test_a_fixed_estimate_has_no_per_point_bandwidths():
    assert fixed_estimate().GetAdaptiveWeights() is None


def test_an_adaptive_estimate_holds_one_bandwidth_for_every_point_of_the_sample():
    n = 2000
    kde = ROOT.TKDE(n, mixed_sample(n), 0.0, 20.0, ADAPTIVE, 1)
    weights = kde.GetAdaptiveWeights()
    assert weights.size == kde.GetValue.__self__._data.size
    assert np.all(weights > 0.0)


def test_a_point_filled_one_at_a_time_makes_the_estimate_root_prints():
    """``Fill`` then a value, ROOT's ``tkde_fill_empty`` case over 100 drawn points."""
    random = ROOT.TRandom3()
    kde = ROOT.TKDE(0, None, 0, 5, ADAPTIVE, 1)
    for _ in range(100):
        kde.Fill(random.Gaus(2, 1))
    assert kde.GetValue(2) == pytest.approx(0.5026609580519209, rel=1e-12)


def test_filling_a_binned_estimate_is_refused_by_leaving_the_data_alone():
    kde = ROOT.TKDE(100, np.zeros(100), 0.0, 1.0, "KernelType:Gaussian;Iteration:Fixed", 1)
    kde.SetBinning(ROOT.TKDE.kForcedBinning)
    kde.Fill(0.5)
    assert kde._n_events == 100


def test_a_weighted_estimate_carries_each_points_weight():
    data = mixed_sample(200)
    weights = np.full(200, 2.0)
    kde = ROOT.TKDE(200, data, weights, 0.0, 20.0, FIXED, 1)
    inside = (data >= 0.0) & (data < 20.0)
    assert kde._sum_of_counts == pytest.approx(2.0 * inside.sum())
    kde.Fill(10.0, 1.5)  # a weighted sample stays weighted as it is filled
    kde.Fill(10.0, 0.5)
    assert kde._weights.size == kde._data.size
    assert kde.GetValue(10.0) > 0.0


def test_the_estimate_asked_for_many_points_at_once_answers_each_of_them():
    kde = fixed_estimate()
    found = kde(np.asarray(centres()))
    assert list(found) == pytest.approx(list(ROOT_VALUES), rel=2e-5)


def test_an_estimate_with_no_data_has_no_density_to_give():
    kde = ROOT.TKDE(0, None, 0.0, 5.0, FIXED, 1)
    with pytest.raises(ValueError, match="holds no data"):
        kde.GetValue(2)


def test_the_kernel_types_are_roots_four_and_each_smooths_differently():
    data = mixed_sample(300)
    found = [
        ROOT.TKDE(300, data, 0.0, 20.0, f"KernelType:{name};Iteration:Fixed", 1).GetValue(10.0)
        for name in ("Gaussian", "Epanechnikov", "Biweight", "CosineArch")
    ]
    assert len(set(found)) == 4
    assert all(value > 0.0 for value in found)


def test_a_kernel_type_and_the_options_say_the_same_thing():
    data = mixed_sample(300)
    said = ROOT.TKDE(300, data, 0.0, 20.0, "KernelType:Epanechnikov;Iteration:Fixed", 1)
    set_by_hand = ROOT.TKDE(300, data, 0.0, 20.0, "Iteration:Fixed", 1)
    set_by_hand.SetKernelType(ROOT.TKDE.kEpanechnikov)
    assert said.GetValue(10.0) == pytest.approx(set_by_hand.GetValue(10.0), rel=1e-15)


def test_an_unknown_option_word_leaves_root_default_in_place():
    data = mixed_sample(300)
    odd = ROOT.TKDE(
        300, data, 0.0, 20.0, "KernelType:Wobble;Iteration:Wobble;Mirror:Wobble;Binning:Wobble", 1
    )
    plain = ROOT.TKDE(300, data, 0.0, 20.0, "", 1)
    assert odd.GetValue(10.0) == pytest.approx(plain.GetValue(10.0), rel=1e-15)


def test_a_tuning_factor_below_one_makes_a_narrower_kernel():
    data = mixed_sample(300)
    plain = ROOT.TKDE(300, data, 0.0, 20.0, FIXED, 1)
    tighter = ROOT.TKDE(300, data, 0.0, 20.0, FIXED, 0.5)
    assert tighter.GetFixedWeight() < plain.GetFixedWeight()


def test_a_non_positive_tuning_factor_is_root_default_one():
    data = mixed_sample(300)
    assert ROOT.TKDE(300, data, 0.0, 20.0, FIXED, 0.0).GetFixedWeight() == pytest.approx(
        ROOT.TKDE(300, data, 0.0, 20.0, FIXED, 1).GetFixedWeight()
    )
    assert ROOT.TKDE(300, data, 0.0, 20.0, FIXED, 1).SetTuneFactor(-1.0) is None


def test_the_iteration_can_be_turned_to_fixed_after_construction():
    data = mixed_sample(300)
    kde = ROOT.TKDE(300, data, 0.0, 20.0, ADAPTIVE, 1)
    assert kde.GetFixedWeight() == -1.0
    kde.SetIteration(ROOT.TKDE.kFixed)
    assert kde.GetFixedWeight() > 0.0


def test_mirroring_the_data_moves_the_estimate_at_the_edges_and_nowhere_else():
    data = np.abs(mixed_sample(400))
    plain = ROOT.TKDE(400, data, 0.0, 20.0, FIXED, 1)
    both = ROOT.TKDE(400, data, 0.0, 20.0, "Iteration:Fixed;Mirror:MirrorBoth;Binning:Unbinned", 1)
    assert both.GetValue(0.1) > plain.GetValue(0.1)
    assert both.GetValue(10.0) == pytest.approx(plain.GetValue(10.0), rel=5e-2)


def test_set_mirror_mirrors_the_data_it_already_holds():
    data = np.abs(mixed_sample(400))
    kde = ROOT.TKDE(400, data, 0.0, 20.0, FIXED, 1)
    before = kde.GetValue(0.1)
    kde.SetMirror(ROOT.TKDE.kMirrorLeft)
    assert kde.GetValue(0.1) > before


def test_every_mirroring_root_names_is_taken():
    data = np.clip(np.abs(mixed_sample(400)), 0.0, 20.0)
    for mirror in (
        "MirrorLeft",
        "MirrorRight",
        "MirrorBoth",
        "MirrorAsymBoth",
        "MirrorRightAsymLeft",
    ):
        kde = ROOT.TKDE(400, data, 0.0, 20.0, f"Iteration:Fixed;Mirror:{mirror}", 1)
        assert kde.GetValue(10.0) > 0.0


def test_binning_a_large_sample_gives_the_same_density_as_using_it_whole():
    n = 20000
    data = mixed_sample(n)
    whole = ROOT.TKDE(n, data, 0.0, 20.0, FIXED, 1)
    binned = ROOT.TKDE(n, data, 0.0, 20.0, "Iteration:Fixed;Binning:ForcedBinning", 1)
    # ROOT's binned rule of thumb counts the bins, not the events, so the kernel is wider
    assert binned.GetValue(10.0) == pytest.approx(whole.GetValue(10.0), rel=2e-1)
    assert binned.GetValue(10.0) != pytest.approx(whole.GetValue(10.0), rel=1e-6)


def test_the_number_of_bins_and_the_event_threshold_are_set_by_their_setters():
    data = mixed_sample(2000)
    kde = ROOT.TKDE(2000, data, 0.0, 20.0, FIXED, 1)
    kde.SetBinning(ROOT.TKDE.kRelaxedBinning)
    kde.SetUseBinsNEvents(10)
    kde.SetNBins(50)
    assert kde._n_bins == 50 and kde._use_bins
    kde.SetNBins(0)  # ROOT refuses it, keeping the bins it has
    assert kde._n_bins == 50


def test_a_range_once_set_is_not_taken_from_the_data_again():
    data = mixed_sample(300)
    kde = ROOT.TKDE(300, data, 0.0, 20.0, FIXED, 1)
    kde.SetRange(2.0, 18.0)
    assert (kde._x_min, kde._x_max) == (2.0, 18.0)
    kde.SetRange(18.0, 2.0)  # ROOT refuses a range that is inside out
    assert (kde._x_min, kde._x_max) == (2.0, 18.0)


def test_an_estimate_made_over_no_range_takes_it_from_the_data():
    data = mixed_sample(300)
    kde = ROOT.TKDE(300, data, 0.0, 0.0, FIXED, 1)
    assert (kde._x_min, kde._x_max) == (float(data.min()), float(data.max()))


def test_the_estimate_as_a_function_is_sampled_and_drawable():
    kde = fixed_estimate(300)
    function = kde.GetFunction(50)
    assert function.GetNpx() == 50
    sampled = function.GetHistogram().GetBinContent(25)
    assert sampled == pytest.approx(
        kde.GetValue(function.GetHistogram().GetBinCenter(25)), rel=1e-12
    )
    drawn = kde.Draw()
    assert drawn is None
    assert kde.GetDrawnFunction() is not None


def test_the_pointwise_error_and_bias_are_refused_rather_than_guessed():
    kde = fixed_estimate(300)
    with pytest.raises(UnsupportedFeatureError, match="GetError is not implemented"):
        kde.GetError(10.0)
    with pytest.raises(UnsupportedFeatureError, match="GetBias is not implemented"):
        kde.GetBias(10.0)


def test_the_sum_of_the_weights_out_of_range_is_not_counted():
    """ROOT normalises by the weight inside the range only, not by all of it."""
    kde = ROOT.TKDE(2, np.array([5.0, 500.0]), 0.0, 10.0, FIXED, 1)
    assert kde._sum_of_counts == 1.0


def test_chisquare_cannot_be_asked_for_another_weighting_than_roots_default():
    histogram = ROOT.TH1D("h", "h", 10, 0.0, 1.0)
    with pytest.raises(UnsupportedFeatureError, match="only ROOT's default Neyman"):
        histogram.Chisquare(ROOT.TF1("f", "1", 0, 1), "P")


def test_binning_the_mirrored_data_mirrors_its_bins_too():
    data = np.clip(mixed_sample(400), 0.0, 20.0)
    kde = ROOT.TKDE(
        400, data, 0.0, 20.0, "Iteration:Fixed;Mirror:MirrorBoth;Binning:ForcedBinning", 1
    )
    assert kde._data.size == 3 * kde._n_bins
    assert kde.GetValue(1.0) > 0.0


def test_each_edge_is_mirrored_on_its_own():
    data = np.clip(mixed_sample(400), 0.0, 20.0)
    for mirror, expected in ((ROOT.TKDE.kMirrorLeft, 2), (ROOT.TKDE.kMirrorRight, 2)):
        kde = ROOT.TKDE(400, data, 0.0, 20.0, "Iteration:Fixed;Binning:ForcedBinning", 1)
        kde.SetMirror(mirror)
        assert kde._data.size == expected * kde._n_bins


def test_setting_no_mirror_on_mirrored_data_leaves_the_estimate_without_one():
    data = np.clip(mixed_sample(400), 0.0, 20.0)
    kde = ROOT.TKDE(400, data, 0.0, 20.0, "Iteration:Fixed;Mirror:MirrorBoth", 1)
    kde.SetMirror(ROOT.TKDE.kMirrorAsymBoth)
    assert not kde._use_mirroring and kde._asym_left and kde._asym_right


def test_statistics_of_an_estimate_holding_next_to_nothing_are_zero():
    empty = ROOT.TKDE(0, np.empty(0), 0.0, 5.0, FIXED, 1)
    assert empty.GetMean() == 0.0
    single = ROOT.TKDE(1, np.array([2.0]), 0.0, 5.0, FIXED, 1)
    assert single.GetSigma() == 0.0


def test_a_kernel_over_no_bins_is_not_made():
    empty = ROOT.TKDE(0, None, 0.0, 5.0, FIXED, 1)
    empty._set_kernel()
    assert empty._kernel is None


def test_an_adaptive_estimate_over_bins_weighs_them_by_their_count():
    data = np.clip(mixed_sample(400), 0.0, 20.0)
    kde = ROOT.TKDE(400, data, 0.0, 20.0, "Iteration:Adaptive;Binning:ForcedBinning", 1)
    assert kde.GetAdaptiveWeights().size == kde._data.size


def test_a_range_taken_from_the_data_follows_what_is_filled_into_it():
    kde = ROOT.TKDE(0, None, 0.0, 0.0, FIXED, 1)
    for value in (1.0, 4.0, 7.0):
        kde.Fill(value)
    kde.GetValue(1.0)
    assert (kde._x_min, kde._x_max) == (1.0, 7.0)


def test_filling_mirrored_data_keeps_the_reflection():
    kde = ROOT.TKDE(0, None, 0.0, 10.0, "Iteration:Fixed;Mirror:MirrorLeft", 1)
    for value in (1.0, 4.0, 7.0):
        kde.Fill(value)
    assert kde.GetValue(0.0) > 0.0


def test_the_statistics_follow_what_is_filled_after_the_estimate_was_made():
    kde = ROOT.TKDE(200, mixed_sample(200), 0.0, 20.0, ADAPTIVE, 1)
    kde.GetValue(10.0)
    kde.Fill(11.0, 2.0)
    assert kde.GetMean() != 0.0 and kde.GetSigma() != 0.0
    assert kde.GetAdaptiveWeights().size == kde._data.size


def test_a_function_asked_for_no_points_keeps_the_default_sampling():
    assert fixed_estimate(300).GetFunction(0).GetNpx() == 100


def test_a_word_root_does_not_know_is_passed_over_and_an_illegal_setting_is_the_default():
    data = mixed_sample(200)
    kde = ROOT.TKDE(200, data, 0.0, 20.0, "Frobnicate:Yes;" + FIXED, 1)
    kde.SetIteration(7)  # ROOT keeps kAdaptive for an iteration it has no value for
    assert kde.GetFixedWeight() == -1.0
    kde.SetMirror(99)  # and kNoMirror for a mirroring it has none for
    assert kde._mirror == ROOT.TKDE.kNoMirror
    kde.SetBinning(99)  # and kRelaxedBinning for a binning it has none for
    assert kde._binning == ROOT.TKDE.kRelaxedBinning


def test_a_fixed_estimate_filled_after_it_was_made_carries_the_new_point():
    kde = ROOT.TKDE(200, mixed_sample(200), 0.0, 20.0, FIXED, 1)
    weight = kde.GetFixedWeight()
    kde.Fill(10.5)
    assert kde.GetSigma() != 0.0
    assert kde.GetFixedWeight() != weight
    assert kde.GetAdaptiveWeights() is None
    kde.SetIteration(ROOT.TKDE.kAdaptive)
    kde.Fill(10.5)
    assert kde.GetAdaptiveWeights().size == kde._data.size


def test_chisquare_is_neymans_over_the_bins_that_hold_something():
    """``TH1::Chisquare(func)``: ``(bin - func) ** 2 / error ** 2``, ROOT's default weighting."""
    histogram = ROOT.TH1D("h", "h", 4, 0.0, 4.0)
    for at, content in enumerate((2.0, 4.0, 0.0, 3.0), start=1):
        histogram.SetBinContent(at, content)
    histogram.Sumw2()
    for at in range(1, 5):
        histogram.SetBinError(at, 1.0 if histogram.GetBinContent(at) else 0.0)
    found = histogram.Chisquare(ROOT.TF1("f", "2", 0, 4))
    # (2-2)**2 + (4-2)**2 + (3-2)**2 over the three bins whose error is not zero
    assert found == pytest.approx(5.0, rel=1e-12)
