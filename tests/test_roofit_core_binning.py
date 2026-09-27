"""RooFit's binnings: uniform runs, boundaries of any spacing, and ranges whose ends move.

The numbers for ``RooBinning`` were printed by ROOT 6.40.04 for the same
calls; the rest pins what the engine's own binnings do, since a variable's
ranges, a plot's axis and a histogram's bins are all read from them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot.roofit.binning import (
    EVALUATING,
    RooAbsBinning,
    RooBinning,
    RooParamBinning,
    RooRangeBinning,
    RooUniformBinning,
    evaluating,
)
from xrdroot.roofit.variables import RooRealVar


def test_a_bare_binning_is_one_bin_over_the_whole_line_and_can_be_renamed() -> None:
    """The base binning is what a variable without a range starts with: one bin, no ends."""
    binning = RooAbsBinning(name="first")
    assert binning.GetName() == "first"
    binning.SetName("second")
    assert binning.GetName() == "second"
    assert binning.lowBound() == -math.inf
    assert binning.highBound() == math.inf
    assert binning.numBins() == 1
    assert binning.numBoundaries() == 2
    assert binning.isUniform()
    assert not binning.isParameterized()
    assert binning.servers() == []


def test_a_uniform_binning_cuts_its_range_into_equal_bins() -> None:
    """Every bin of a uniform binning is the same width, and a value finds the bin it is in."""
    binning = RooUniformBinning(0.0, 2.0, 4)
    assert list(binning.array()) == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert binning.binLow(1) == 0.5
    assert binning.binHigh(1) == 1.0
    assert binning.binCenter(2) == 1.25
    assert binning.binWidth(3) == 0.5
    assert binning.averageBinWidth() == 0.5
    assert binning.binNumber(1.2) == 2
    binning.setBins(8)
    assert binning.numBins() == 8
    assert binning.averageBinWidth() == 0.25


def test_a_value_outside_a_binning_is_counted_in_the_nearest_end_bin() -> None:
    """ROOT clamps a bin number to the first and last bin: 0 below, the last above."""
    binning = RooBinning(4, 0.0, 2.0)
    assert binning.binNumber(-5.0) == 0
    assert binning.binNumber(50.0) == 3


def test_moving_the_ends_of_a_uniform_binning_keeps_its_number_of_bins() -> None:
    """``setMin`` and ``setMax`` stretch a uniform binning: the bins widen, not multiply."""
    binning = RooUniformBinning(0.0, 1.0, 10)
    binning.setMin(-1.0)
    binning.setMax(3.0)
    assert (binning.lowBound(), binning.highBound(), binning.numBins()) == (-1.0, 3.0, 10)
    assert binning.averageBinWidth() == 0.4


def test_a_range_binning_is_a_single_bin_between_its_two_ends() -> None:
    """A named range is kept as a binning of one bin, so ``getMin("signal")`` is its low edge."""
    binning = RooRangeBinning(1.0, 3.0, "signal")
    assert binning.GetName() == "signal"
    assert binning.numBins() == 1
    assert list(binning.array()) == [1.0, 3.0]
    assert (RooRangeBinning().lowBound(), RooRangeBinning().highBound()) == (-math.inf, math.inf)


def test_a_cloned_binning_is_independent_and_may_take_a_new_name() -> None:
    """A variable's copy owns its binning: changing one must not move the other."""
    binning = RooUniformBinning(0.0, 1.0, 5, "orig")
    same = binning.clone()
    renamed = binning.clone("copy")
    binning.setBins(2)
    assert same.GetName() == "orig"
    assert renamed.GetName() == "copy"
    assert same.numBins() == 5


def test_boundaries_added_one_at_a_time_make_bins_as_root_makes_them() -> None:
    """``addBoundary`` and ``addBoundaryPair`` build ``[0, 2, 3, 7, 10]``: ROOT's four bins."""
    binning = RooBinning(0, 10)
    assert binning.addBoundary(2)
    binning.addBoundary(7)
    binning.addBoundaryPair(3, 5)
    assert binning.numBins() == 4
    assert [binning.binLow(i) for i in range(4)] == [0.0, 2.0, 3.0, 7.0]
    assert binning.binNumber(6.5) == 2
    assert binning.binWidth(0) == 2.0
    assert binning.binCenter(1) == 2.5
    assert not binning.isUniform()


def test_removing_a_boundary_merges_the_two_bins_beside_it() -> None:
    """Taking out 7 from ``[0, 2, 3, 7, 10]`` leaves ROOT's three bins; a uniform run adds more."""
    binning = RooBinning(0, 10)
    for edge in (2, 7, 3):
        binning.addBoundary(edge)
    binning.removeBoundary(7)
    binning.removeBoundary(7)
    assert binning.numBins() == 3
    binning.addUniform(2, 8, 10)
    assert binning.numBins() == 5
    assert binning.averageBinWidth() == 2.0
    assert list(binning.array()) == [0.0, 2.0, 3.0, 8.0, 9.0, 10.0]


def test_removing_a_boundary_says_false_when_it_was_there_as_root_does() -> None:
    """ROOT's ``removeBoundary`` returns ``false`` for a boundary removed, ``true`` for none."""
    binning = RooBinning(0, 10)
    binning.addBoundary(7)
    assert binning.removeBoundary(7) is False
    assert binning.removeBoundary(7) is True


def test_a_binning_of_n_bins_between_two_ends_is_n_boundaries_apart() -> None:
    """``RooBinning(4, 0, 2)``: four half-wide bins, kept as boundaries, not as a uniform run."""
    binning = RooBinning(4, 0.0, 2.0, "quarters")
    assert binning.GetName() == "quarters"
    assert binning.numBins() == 4
    assert binning.binHigh(0) == 0.5
    assert not binning.isUniform()


def test_narrowing_a_binning_drops_the_boundaries_left_outside() -> None:
    """ROOT's ``setRange(0.5, 2)`` on ``RooBinning(4, 0, 2)`` leaves three bins from 0.5."""
    binning = RooBinning(4, 0.0, 2.0)
    binning.setRange(0.5, 2.0)
    assert binning.numBins() == 3
    assert binning.binLow(0) == 0.5


def test_a_binning_made_with_fewer_than_two_ends_is_open_where_none_was_given() -> None:
    """``RooBinning()`` spans the whole line and ``RooBinning(lo)`` the line above ``lo``."""
    whole = RooBinning()
    assert (whole.lowBound(), whole.highBound()) == (-math.inf, math.inf)
    above = RooBinning(1.0, "above")
    assert (above.lowBound(), above.highBound(), above.GetName()) == (1.0, math.inf, "above")
    assert RooBinning(0.0, 0.0).numBins() == 0


def test_a_range_whose_ends_are_variables_moves_with_them() -> None:
    """``setRange(lo, hi)`` with variables: each end is read whenever the range is asked for."""
    lo = RooRealVar("lo", "lo", -2, -5, 5)
    hi = RooRealVar("hi", "hi", 3, -5, 5)
    binning = RooParamBinning(lo, hi, 5, "moving")
    assert (binning.lowBound(), binning.highBound()) == (-2.0, 3.0)
    lo.setVal(-1)
    assert np.allclose(binning.array(), [-1.0, -0.2, 0.6, 1.4, 2.2, 3.0], rtol=0, atol=1e-15)
    assert binning.isParameterized()
    assert binning.servers() == [lo, hi]


def test_a_range_whose_ends_are_variables_reads_them_from_the_values_being_evaluated() -> None:
    """Inside ``evaluating(ctx)`` an end the context names takes its column: one end per event."""
    lo = RooRealVar("lo", "lo", -2, -5, 5)
    hi = RooRealVar("hi", "hi", 3, -5, 5)
    binning = RooParamBinning(lo, hi)
    with evaluating({"lo": np.array([0.0, 1.0])}):
        assert list(binning.lowBound()) == [0.0, 1.0]
        assert binning.highBound() == 3.0
    assert EVALUATING == []
    assert binning.lowBound() == -2.0


def test_a_range_whose_ends_are_variables_cannot_be_moved_by_number() -> None:
    """Its ends are functions: moving it means setting them, so ``setRange`` refuses."""
    binning = RooParamBinning(RooRealVar("a", "a", 0.0), RooRealVar("b", "b", 1.0))
    with pytest.raises(ValueError, match="cannot be moved"):
        binning.setRange(0.0, 2.0)


def test_a_cloned_moving_range_shares_its_ends_and_may_take_a_new_name() -> None:
    """A copy of a variable follows the same end variables, under its own name if given one."""
    lo, hi = RooRealVar("a", "a", 0.0), RooRealVar("b", "b", 1.0)
    binning = RooParamBinning(lo, hi, 7, "orig")
    same = binning.clone()
    renamed = binning.clone("copy")
    assert (same.GetName(), same.numBins(), same.servers()) == ("orig", 7, [lo, hi])
    assert renamed.GetName() == "copy"
