"""The numbers ``TH1`` answers one at a time, and the members a hand-made row can hold.

Everything here is booked and filled in code, so no file is opened and the test stays
inside the ``hist`` component's own isolation scope. Where a number is ROOT's, it was
read off the ROOT 6.40.04 oracle for the same seven fills and is quoted beside the
assertion. The mean, spread, skewness and kurtosis come from the running sums ROOT keeps
(``fTsumw``, ``fTsumwx``, …), which is why they are the same numbers ROOT prints; the last
two bits of a square root are the C library's and differ between Apple's libm and glibc,
so those are held to ``REL`` with that reason.

The members half of the file covers rows this engine did not make itself: a histogram put
together by hand may hold its bins in an :class:`array.array` and its squared weights as a
plain list, and both are made ordinary NumPy arrays at the first change, so the member and
the change stay one thing.
"""

from __future__ import annotations

import array
import math
import sys
from typing import Any

import numpy as np
import pytest

from xrdroot.errors import FormatError, UnsupportedFeatureError
from xrdroot.hist import Axis, Histogram

#: How close a root of a running sum is to ROOT's own: ``math.sqrt`` rounds to the C
#: library's answer and Apple's arm64 and x86-64 libm, and glibc, differ in the last bit.
REL = 1e-12


def _seven_fills() -> Histogram:
    """ROOT's ``TH1D h("h", "", 4, 0, 4)`` filled at 0.5, 0.5, 1.5, 2.5, 2.5, 2.5, 3.5.

    Its bins hold 2, 1, 3 and 1, which is seven entries over a mean of 1.92857…; every
    number in the first half of this file is the oracle's answer for exactly this
    histogram.
    """
    made = Histogram.book("h", (4, 0.0, 4.0))
    made.fill([0.5, 0.5, 1.5, 2.5, 2.5, 2.5, 3.5])
    return made


def test_the_mean_and_the_error_on_it_are_the_running_sums_quotient():
    """``GetMean`` is ``fTsumwx / fTsumw`` and ``GetMeanError`` the spread over the root of the
    effective entries: 1.9285714285714286 and 0.39678004275966489 in ROOT."""
    found = _seven_fills()
    assert found.mean() == pytest.approx(1.9285714285714286, rel=REL)
    assert found.mean_error() == pytest.approx(0.39678004275966489, rel=REL)


def test_the_spread_and_the_error_on_it_come_from_the_squares():
    """``GetStdDev`` is ``sqrt(fTsumwx2/fTsumw - mean**2)`` - 1.0497813183356477 - and
    ``GetStdDevError`` divides it by the root of twice the entries, 0.28056585887484731."""
    found = _seven_fills()
    assert found.std() == pytest.approx(1.0497813183356477, rel=REL)
    assert found.std_error() == pytest.approx(0.28056585887484731, rel=REL)


def test_the_skewness_and_the_kurtosis_are_the_third_and_fourth_moments():
    """``GetSkewness`` is the third moment about the mean in units of the spread, and
    ``GetKurtosis`` the fourth less the three a Gaussian has: -0.18144368465060587 and
    -1.2613168724279833, the negative the flat sides of four bins give."""
    found = _seven_fills()
    assert found.skewness() == pytest.approx(-0.18144368465060587, rel=REL)
    assert found.kurtosis() == pytest.approx(-1.2613168724279833, rel=REL)


def test_the_moment_errors_are_what_a_gaussian_parent_gives():
    """``GetSkewness(11)`` and ``GetKurtosis(11)`` are ``sqrt(6/n)`` and ``sqrt(24/n)`` for
    seven entries - 0.92582009977255142 and 1.8516401995451028 - and both are zero, not
    infinite, for a histogram with nothing in it, which is ROOT's answer too."""
    found = _seven_fills()
    assert found.skewness_error() == pytest.approx(0.92582009977255142, rel=REL)
    assert found.kurtosis_error() == pytest.approx(1.8516401995451028, rel=REL)
    empty = Histogram.book("e", (2, 0.0, 2.0))
    assert (empty.skewness_error(), empty.kurtosis_error()) == (0.0, 0.0)


def test_the_effective_entries_of_a_weighted_histogram_are_its_weights_over_their_squares():
    """``GetEffectiveEntries`` is ``(sum w)**2 / sum w**2``: five units of weight in three
    fills is 25/9 = 2.7777777777777777, where the entries count 3."""
    weighted = Histogram.book("w", (3, 0.0, 3.0))
    weighted.fill([0.5, 0.5, 1.5], weight=[2.0, 2.0, 1.0])
    assert weighted.effective_entries == pytest.approx(2.7777777777777777, rel=REL)
    assert weighted.entries == 3.0  # ROOT: the fills, not the weight
    assert _seven_fills().effective_entries == 7.0  # unweighted, the two are the same


def test_the_integral_and_the_error_on_it():
    """``Integral`` adds the bins 0 to ``nbins+1`` and no flow by default, so seven; over
    bins 1..2 it is the two bins' three, and ``IntegralAndError`` gives the root of the
    sum of their variances beside it, sqrt(7) and sqrt(3)."""
    found = _seven_fills()
    assert found.integral() == 7.0
    assert found.integral_error() == pytest.approx(math.sqrt(7.0), rel=REL)
    assert found.integral(1, 2) == 3.0  # ROOT's Integral(1, 2)
    assert found.integral_error(1, 2) == pytest.approx(math.sqrt(3.0), rel=REL)
    assert found.integral(width=True) == 7.0  # every bin here is one wide


def test_interpolate_follows_the_line_between_bin_centres():
    """``Interpolate`` reads the centres as points and draws a line between two: 2 at the
    first centre (0.5), 1.25 a quarter of the way to the second, 2 at 3.0 - the last
    centre - because a point past the last bin holds that bin's value."""
    found = _seven_fills()
    assert found.interpolate(0.5) == 2.0
    assert found.interpolate(1.25) == 1.25
    assert found.interpolate(3.0) == 2.0


def test_the_extremes_index_the_values_where_root_counts_its_bins_from_one():
    """``GetMaximum`` and ``GetMinimum`` are the largest and smallest content on the axes and
    ``argmax``/``argmin`` where they are - an index into :meth:`values`, which starts at the
    underflow. ROOT's ``GetMaximumBin`` counts from one, so it reads one more: 3 and 2 where
    these read 2 and 1."""
    found = _seven_fills()
    assert (found.maximum(), found.minimum()) == (3.0, 1.0)
    assert (found.argmax(), found.argmin()) == (2, 1)
    assert found.values().tolist()[found.argmax()] == found.maximum()


def test_stat_overflows_is_root_s_three_valued_flag():
    """``GetStatOverflows`` is ``EStatOverflows``: ``kIgnore`` 0, ``kConsider`` 1 and
    ``kNeutral`` 2, the last being what a histogram booked here starts at, as ROOT's does."""
    found = _seven_fills()
    assert found.stat_overflows() == 2
    found.set_stat_overflows(1)  # kConsider
    assert found.stat_overflows() == 1


def test_the_counts_of_a_weighted_histogram_are_the_fills_it_is_worth():
    """What the plotting protocol means by ``counts``: ``values**2/variances``, the number of
    unweighted fills of the same relative uncertainty. Two weight-2 fills are worth two, one
    weight-1 fill is worth one, and a bin with no variance to divide is zero rather than
    infinite."""
    weighted = Histogram.book("w", (3, 0.0, 3.0))
    weighted.fill([0.5, 0.5, 1.5], weight=[2.0, 2.0, 1.0])
    assert weighted.values().tolist() == [4.0, 1.0, 0.0]
    assert weighted.variances().tolist() == [8.0, 1.0, 0.0]
    assert weighted.counts().tolist() == [2.0, 1.0, 0.0]
    unweighted = _seven_fills()
    assert not unweighted.weighted
    assert unweighted.counts().tolist() == unweighted.values().tolist()


def test_every_bin_s_content_and_error_come_back_one_at_a_time():
    """``bin_content`` and ``bin_error`` are ``GetBinContent``/``GetBinError`` of one cell,
    the flow cells included: ROOT reads 0, 2, 1, 3, 1, 0 and 0, 1.414…, 1, 1.732…, 1, 0
    for the seven fills, and asking for one bin does not build the array of all of them."""
    found = _seven_fills()
    assert [found.bin_content(cell) for cell in range(6)] == [0.0, 2.0, 1.0, 3.0, 1.0, 0.0]
    assert [found.bin_error(cell) for cell in range(6)] == pytest.approx(
        [0.0, math.sqrt(2.0), 1.0, math.sqrt(3.0), 1.0, 0.0], rel=REL
    )


def test_a_row_holding_its_bins_in_a_plain_array_is_made_a_numpy_one_at_the_first_change():
    """A file's bins are a NumPy array and read-only in byte order; a row put together by
    hand may hold an :class:`array.array` instead. The first change makes an ordinary array
    of the storage's own type and puts it in the members, so the change is the member's."""
    filled = Histogram.book("m", (3, 0.0, 3.0))
    filled.fill([0.5, 0.5, 1.5])
    row = filled.members
    row["TArrayD"] = array.array("d", [float(value) for value in row["TArrayD"]])
    made = Histogram("TH1D", row)
    assert made.values().tolist() == [2.0, 1.0, 0.0]
    made.scale(2.0)
    assert isinstance(row["TArrayD"], np.ndarray)
    assert row["TArrayD"].dtype == np.float64
    assert made.values().tolist() == [4.0, 2.0, 0.0]


def test_squares_held_as_a_list_are_made_an_array_when_one_bin_s_error_is_asked_for():
    """``fSumw2`` is read the same way: a list of the right length is turned into an array of
    doubles in place, and the errors are the roots of what it holds."""
    row = Histogram.book("s", (4, 0.0, 4.0)).members
    row["TH1"]["fSumw2"] = [9.0, 1.0, 2.0, 3.0, 4.0, 25.0]
    made = Histogram("TH1D", row)
    assert [made.bin_error(cell) for cell in range(6)] == pytest.approx(
        [3.0, 1.0, math.sqrt(2.0), math.sqrt(3.0), 2.0, 5.0], rel=REL
    )
    assert isinstance(row["TH1"]["fSumw2"], np.ndarray)


def test_a_row_without_bins_or_axes_is_refused_in_a_sentence():
    """A ``TH1D`` row that has neither the member holding the bins nor the one holding the
    axes is not a histogram, and the sentence names the class the file claimed.

    Every way of missing them is the one refusal: no core at all, a core with no array
    under it, a core whose only nested dictionary is not a histogram either, and an array
    with no core to belong to.
    """
    missing = (
        {"TH1": {"fNcells": 3}},
        {"TH1": {"fNcells": 3}, "extra": {"nothing": 1}},
        {"TH1": {"TH1": {"fNcells": 3}, "fScalefactor": 1.0}},
        {"TH1": {"TArrayD": np.zeros(5)}, "fExtra": 1},
        {"TArrayD": np.zeros(5)},
    )
    for row in missing:
        with pytest.raises(FormatError, match="a TH1D was written without its bins or its axes"):
            Histogram("TH1D", row)


def test_a_row_whose_bins_are_too_few_for_its_axes_is_refused():
    """Four bins and their flow are six cells, and a row holding three values is refused
    rather than read past the end of its array."""
    row = Histogram.book("t", (3, 0.0, 3.0)).members
    row["TArrayD"] = np.zeros(3)
    with pytest.raises(FormatError, match="a TH1D of 5 bins counting the ends holds only 3"):
        Histogram("TH1D", row)


def test_values_that_are_not_one_per_bin_are_refused_with_both_counts():
    """Booking a histogram takes one value per bin, or two more counting the flow at each
    end; anything else says what it was given and what it needed, rather than padding or
    truncating the bins to fit."""
    with pytest.raises(ValueError, match="5 values for 2 bins"):
        Histogram.new("h", np.array([0.0, 1.0, 2.0]), np.zeros(5))
    assert Histogram.new("h", np.array([0.0, 1.0, 2.0]), np.zeros(4)).values().tolist() == [
        0.0,
        0.0,
    ]


def test_edges_that_do_not_make_a_bin_are_refused():
    """An axis needs two edges at least, and each one past the first has to be above the
    one before it: ROOT bins by interval and an interval of no width is not one."""
    with pytest.raises(ValueError, match="at least two edges"):
        Histogram.new("h", np.array([1.0]), np.zeros(0))
    with pytest.raises(ValueError, match="edges must increase"):
        Histogram.new("h", np.array([0.0, 1.0, 1.0]), np.zeros(2))


def test_more_than_three_axes_or_fewer_edge_sets_than_axes_are_refused():
    """ROOT has one-, two- and three-dimensional histograms and nothing past that, and
    values of three dimensions need three sets of edges - not two."""
    four = [np.array([0.0, 1.0]), np.arange(3.0), np.arange(4.0), np.arange(3.0)]
    with pytest.raises(ValueError, match="values of 4 dimensions are not a histogram ROOT has"):
        Histogram.new("h", four, np.zeros((1, 2, 3, 2)))
    with pytest.raises(ValueError, match="need 3 sets of edges, one per axis, and 2 were given"):
        Histogram.new("h", four[:2], np.zeros((1, 2, 3)))


def test_errors_and_variances_together_are_refused_because_one_is_the_root_of_the_other():
    """A bin's error and its variance carry the same information; giving both is ambiguous,
    so it is refused rather than one of them being taken silently."""
    with pytest.raises(ValueError, match="give errors or variances, not both"):
        Histogram.new(
            "h",
            np.array([0.0, 1.0, 2.0]),
            np.zeros(2),
            errors=np.ones(2),
            variances=np.ones(2),
        )


def test_an_axis_with_no_title_is_labelled_by_its_name():
    """``label`` is what to write along the axis: the title where there is one, and otherwise
    ROOT's own name for it - ``xaxis``, not an empty string."""
    row = {"TNamed": {"fName": "xaxis", "fTitle": ""}, "fNbins": 1, "fXmin": 0.0}
    row.update({"fXmax": 1.0, "fXbins": np.zeros(0)})
    assert Axis(row).label == "xaxis"
    titled = dict(row)
    titled["TNamed"] = {"fName": "xaxis", "fTitle": "energy [GeV]"}
    assert Axis(titled).label == "energy [GeV]"


def test_turning_a_histogram_into_a_hist_hist_needs_the_package(monkeypatch: Any):
    """``to_hist`` imports ``hist`` when it is asked, so a machine without it is refused with
    the command that installs it, not with a bare ImportError in the middle of a plot."""
    monkeypatch.setitem(sys.modules, "hist", None)  # what an uninstalled package looks like
    with pytest.raises(UnsupportedFeatureError, match="pip install hist"):
        Histogram.book("h", (3, 0.0, 3.0)).to_hist()


def test_turning_it_into_a_hist_hist_carries_the_counts_and_the_squares():
    """With the package there, the histogram arrives as one whose storage keeps the squares:
    a weight-2 fill reads four units of content for eight of variance, and an unweighted
    fill reads one for one."""
    pytest.importorskip("hist")
    weighted = Histogram.book("w", (3, 0.0, 3.0))
    weighted.fill([0.5, 0.5, 1.5], weight=[2.0, 2.0, 1.0])
    made = weighted.to_hist()
    assert made.values().tolist() == [4.0, 1.0, 0.0]
    assert np.asarray(made.variances()).tolist() == [8.0, 1.0, 0.0]
    plain = Histogram.book("p", (3, 0.0, 3.0))
    plain.fill([0.5])
    assert np.asarray(plain.to_hist().variances()).tolist() == [1.0, 0.0, 0.0]
