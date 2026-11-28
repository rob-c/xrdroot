"""``TEfficiency``: an efficiency from two histograms, and the intervals ROOT quotes on it.

An efficiency is made with ``Efficiency.from_histograms`` from two histograms booked here, so its
arithmetic is testable without a file. Four interval methods are checked exactly - Clopper-Pearson,
normal, Wilson and Agresti-Coull - against ROOT 6.40.04's own numbers, read off
``TEfficiency.GetEfficiencyErrorLow/Up`` in the oracle for the same counts. The oracle quotes the
distance from the efficiency, so a bound is the efficiency less the low one, or plus the high one.

The three Bayesian methods (Jeffreys, uniform, Bayesian) are computed as ROOT's own
``BetaCentralInterval`` computes them: ``TEfficiency.Bayesian(10, 3, 0.682689492137, 0.5, 0.5,
False, False)`` in the oracle answers 0.17993225994167145, which is what this engine gives. What
``TEfficiency``'s *object* reports instead is the **shortest** interval, which this engine refuses
in a sentence; implementing that interval is `W0-276`, so this test asserts only the invariants a
Bayesian interval must keep.

No file is opened, which is what lets this test sit inside the ``hist`` component's own scope.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.efficiency import USE_WEIGHTS, Efficiency, beta_quantile, regularized_beta
from xrdroot.errors import FormatError
from xrdroot.hist import Histogram
from xrdroot.profile import Profile

#: Three of ten, none of five, five of five and one of two: the bins asked of the oracle.
COUNTS = ((3, 10), (0, 5), (5, 5), (1, 2))


def _efficiency() -> Efficiency:
    """The four counts filled into a four-bin efficiency, one entry at a time."""
    made = Efficiency.from_histograms(
        Histogram.book("passed", (4, 0.0, 4.0)),
        Histogram.book("total", (4, 0.0, 4.0)),
        "eff",
    )
    for number, (passed, total) in enumerate(COUNTS):
        made.fill(np.array([1] * passed + [0] * (total - passed)), np.full(total, number + 0.5))
    return made


def test_an_efficiency_is_what_passed_out_of_what_was_tried():
    """``GetEfficiency`` is the ratio in each bin - 3/10, 0/5, 5/5 and 1/2 - which is what the
    oracle's ``TEfficiency`` answers for the same two histograms."""
    made = _efficiency()
    assert (made.name, made.title) == ("eff", "efficiency")
    assert made.values().tolist() == [0.3, 0.0, 1.0, 0.5]
    assert made.method == "clopper-pearson"  # fStatisticOption 0, ROOT's default
    assert made.level == pytest.approx(0.682689492137086)  # its default level, one sigma


def test_the_clopper_pearson_interval_is_root_s_own_to_twelve_digits():
    """The oracle's errors for 3/10 are 0.158328098893 and 0.20826248199, so the bounds are 0.3
    less and 0.3 plus them - and likewise for the other three bins."""
    low, high = _efficiency().intervals(method="clopper-pearson")
    assert (low[0], high[0]) == pytest.approx((0.141671901107, 0.50826248199), abs=1e-12)
    assert (low[1], high[1]) == pytest.approx((0.0, 0.308024223477), abs=1e-12)  # none of five
    assert (low[2], high[2]) == pytest.approx((0.691975776523, 1.0), abs=1e-12)  # all five
    assert (low[3], high[3]) == pytest.approx((0.08275153526, 0.91724846474), abs=1e-12)


def test_the_normal_wilson_and_agresti_coull_intervals_are_root_s_own_too():
    """Each is the oracle's error quoted as bounds: the normal interval is symmetric and widest at
    a half, Wilson pulls it inward, and Agresti-Coull corrects the counts first."""
    made = _efficiency()
    normal_low, normal_high = made.intervals(method="normal")
    assert (normal_low[0], normal_high[0]) == pytest.approx((0.155086232538, 0.444913767462), 1e-12)
    assert normal_low[3] == pytest.approx(0.5 - 0.353553390593, abs=1e-12)  # 1/2, symmetric
    wilson_low, wilson_high = made.intervals(method="wilson")
    assert (wilson_low[0], wilson_high[0]) == pytest.approx((0.178820820757, 0.457542815607), 1e-12)
    agresti_low, agresti_high = made.intervals(method="agresti-coull")
    assert (agresti_low[0], agresti_high[0]) == pytest.approx(
        (0.17774673167, 0.458616904694), abs=1e-12
    )


def test_a_bayesian_interval_holds_the_unit_interval_and_its_own_efficiency():
    """Jeffreys, uniform and Bayesian quote a Beta posterior's quantiles. Their numbers are not
    ROOT's yet (`W0-276`), so what is pinned is what any interval must keep: both ends inside zero
    and one, the low below the high, and a bin that neither all passed nor all failed held about
    its own efficiency."""
    made = _efficiency()
    values = made.values()
    for method in ("jeffreys", "uniform", "bayesian"):
        low, high = made.intervals(method=method)
        assert np.all(low >= 0.0) and np.all(high <= 1.0)
        assert np.all(low < high)
        middle = (values > 0.0) & (values < 1.0)
        assert np.all(low[middle] <= values[middle])
        assert np.all(values[middle] <= high[middle])


def test_the_incomplete_beta_and_its_inverse_are_mathcore_s_own():
    """``regularized_beta`` is ``TMath::BetaIncomplete`` and ``beta_quantile`` its inverse, as
    ``ROOT.Math.beta_quantile`` gives them: 0.3483 at x=0.3 and 0.38572756813238956 at the
    median."""
    assert regularized_beta(0.3, 2.0, 3.0) == pytest.approx(0.34829999999999978, abs=1e-15)
    assert regularized_beta(0.5, 2.0, 3.0) == pytest.approx(0.6875, abs=1e-15)
    assert beta_quantile(0.5, 2.0, 3.0) == pytest.approx(0.38572756813238956, abs=1e-15)


def test_the_columns_come_back_with_the_edges_and_the_flow_on_asking():
    """``to_numpy`` is the plotting protocol: the values per bin and the axis's edges, and
    ``flow=True`` brings the two ends of each back."""
    made = _efficiency()
    values, edges = made.to_numpy()
    assert values.tolist() == [0.3, 0.0, 1.0, 0.5]
    assert edges.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [column.shape for column in made.to_numpy(flow=True)] == [(6,), (7,)]


def test_a_weighted_fill_marks_the_efficiency_and_keeps_the_squares():
    """ROOT's ``FillWeighted``: the weights are the totals, their squares are kept, and the object
    says by its own bit that it was filled with weights."""
    made = Efficiency.from_histograms(
        Histogram.book("passed", (2, 0.0, 2.0)),
        Histogram.book("total", (2, 0.0, 2.0)),
        "w",
    )
    made.fill(np.array([1, 0]), np.array([0.5, 1.5]), weight=np.array([2.0, 1.0]))
    assert made.total.values().tolist() == [2.0, 1.0]
    assert made.total.variances().tolist() == [4.0, 1.0]  # the squares of the weights
    assert made._bits & USE_WEIGHTS


def test_what_passed_can_be_no_more_than_what_was_tried_bin_by_bin():
    """ROOT's ``CheckConsistency``: an efficiency whose passed histogram holds more than its total
    in any bin is refused, counting the flow bins too."""
    passed = Histogram.book("more", (2, 0.0, 2.0))
    passed.fill(0.5)
    passed.fill(0.5)  # two in the first bin
    total = Histogram.book("less", (2, 0.0, 2.0))
    total.fill(0.5)  # and only one tried there
    with pytest.raises(ValueError, match="must be a part of what was tried"):
        Efficiency.from_histograms(passed, total)


def test_a_fill_takes_whether_each_entry_passed_and_one_coordinate_per_axis():
    """An efficiency of one axis takes two arrays - passed and then x - and a second coordinate is
    refused saying how many axes it has."""
    with pytest.raises(ValueError, match="filling it takes whether each entry passed"):
        _efficiency().fill(np.array([1]), np.array([0.5]), np.array([1.5]))


def test_an_efficiency_needs_two_histograms_of_counts():
    """A file's efficiency without its two histograms cannot be read, and a profile's bins are not
    counts - each refused in a full sentence."""
    with pytest.raises(FormatError, match="was written without its two histograms"):
        Efficiency("TEfficiency", {"TNamed": {"fName": "x", "fTitle": "x"}})
    profile = Profile.book("pr", (2, 0.0, 2.0))
    with pytest.raises(TypeError, match="two histograms of counts"):
        Efficiency.from_histograms(profile, Histogram.book("t", (2, 0.0, 2.0)))
