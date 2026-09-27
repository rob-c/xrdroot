"""``RooRealVar`` and ``RooConstVar``: values, ranges, binnings, errors and how they print.

The printed lines, messages and numbers are ROOT 6.40.04's for the same
calls. Where the engine is known to part from ROOT the test that holds it
to ROOT is marked ``xfail(strict=True)``, so that it turns red - to be
unmarked - as soon as the engine is fixed.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from xrdroot.roofit.binning import RooBinning, RooParamBinning
from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.printing import kClassName, kExtras, kInline, kName, kValue
from xrdroot.roofit.variables import (
    RooAbsRealLValue,
    RooConstVar,
    RooRealVar,
    is_infinite,
)

INF = math.inf


def test_a_number_beyond_1e30_counts_as_infinite() -> None:
    """``RooNumber::isInfinite``: RooFit's infinity is anything at or past 1e30, either sign."""
    assert is_infinite(INF)
    assert is_infinite(-1e30)
    assert not is_infinite(9.9e29)


def test_a_variable_prints_its_value_range_bins_and_unit_as_root_does(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ROOT's one-line ``Print()``: the value, its error, ``L(min - max)``, ``B(n)``, the unit."""
    x = RooRealVar("x", "x title", 1, -10, 10, "cm")
    x.Print()
    x.setError(0.5)
    x.Print()
    x.setAsymError(-0.3, 0.4)
    x.Print()
    x.setBins(20)
    x.Print()
    assert capsys.readouterr().out == (
        "RooRealVar::x = 1  L(-10 - 10) // [cm]\n"
        "RooRealVar::x = 1 +/- 0.5  L(-10 - 10) // [cm]\n"
        "RooRealVar::x = 1 +/- (-0.3,0.4)  L(-10 - 10) // [cm]\n"
        "RooRealVar::x = 1 +/- (-0.3,0.4)  L(-10 - 10) B(20) // [cm]\n"
    )


def test_a_constant_and_an_open_ended_variable_print_as_root_does(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A one-number variable is constant, ``C``; an end at infinity prints ``-INF`` or ``+INF``."""
    RooRealVar("y", "y", 3.0).Print()
    RooRealVar("z", "z", -INF, INF).Print()
    RooRealVar("w2", "w2", 2, INF).Print()
    RooConstVar("c", "c", 2.5).Print()
    assert capsys.readouterr().out == (
        "RooRealVar::y = 3 C  L(-INF - +INF) \n"
        "RooRealVar::z = 0  L(-INF - +INF) \n"
        "RooRealVar::w2 = 2  L(2 - +INF) \n"
        "RooConstVar::c = 2.5\n"
    )


def test_a_variable_made_from_a_range_starts_at_its_middle_or_its_one_finite_end() -> None:
    """ROOT's range constructor puts the value at the middle, or at the finite end, or at 0."""
    assert RooRealVar("a", "a", 0, 4).getVal() == 2.0
    assert RooRealVar("b", "b", -INF, INF).getVal() == 0.0
    assert RooRealVar("c", "c", -INF, 5).getVal() == 5.0
    assert RooRealVar("d", "d", 2, INF).getVal() == 2.0


def test_a_variable_made_with_no_numbers_is_zero_on_the_whole_line() -> None:
    """``RooRealVar(name, title)`` is not a constructor ROOT has, but a copy starts this way."""
    v = RooRealVar("v", "v")
    assert (v.getVal(), v.hasMin(), v.hasMax(), v.isConstant()) == (0.0, False, False, False)


def test_a_value_in_a_range_constructor_is_kept_inside_the_range() -> None:
    """``RooRealVar(x, x, 20, 0, 10)`` starts at the upper end, as ROOT clips it."""
    assert RooRealVar("x", "x", 20, 0, 10).getVal() == 10.0
    assert RooRealVar("x", "x", -3, 0, 10).getVal() == 0.0


def test_the_verbose_print_gives_the_error_with_its_unit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``Print("v")`` ends with ROOT's ``--- RooRealVar ---`` part, the error and its unit."""
    x = RooRealVar("x", "x", 1, -10, 10, "cm")
    x.setError(0.5)
    x.Print("v")
    assert capsys.readouterr().out.endswith("--- RooRealVar ---\n  Error = 0.5 cm\n")
    y = RooRealVar("y", "y", 1, -10, 10)
    y.Print("v")
    assert capsys.readouterr().out.endswith("--- RooRealVar ---\n  Error = 0\n")


@pytest.mark.xfail(strict=True, reason="the verbose print has no RooAbsRealLValue part")
def test_the_verbose_print_gives_the_fit_range_as_root_does(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ROOT's ``Print("v")`` has ``--- RooAbsRealLValue ---`` with the fit range and its unit."""
    RooRealVar("x", "x", 1, -10, 10, "cm").Print("v")
    assert "--- RooAbsRealLValue ---\n  Fit range is [ -10 cm , 10 cm ]\n" in (
        capsys.readouterr().out
    )


def test_the_tree_print_of_a_variable_is_its_address_and_its_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``Print("t")``: ROOT's ``0x... RooRealVar::v = 0.6``, the address left as it may be."""
    v = RooRealVar("v", "v", 0.6, 0, 1)
    v.Print("t")
    assert re.fullmatch(r"0x[0-9a-f]+ RooRealVar::v = 0\.6\n", capsys.readouterr().out)


def test_an_inline_print_leaves_the_range_out() -> None:
    """The ``"I"`` contents are class, name and value: the extras are only for a line of its own."""
    x = RooRealVar("x", "x", 1, -10, 10)
    contents = x.defaultPrintContents("I")
    assert contents == kName | kClassName | kValue
    assert x.printStream(contents, kInline) == "RooRealVar::x = 1"
    assert x.defaultPrintContents("") == kName | kClassName | kValue | kExtras


def test_a_range_given_backwards_warns_and_closes_at_its_low_end(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ROOT's ``setRange(5, 2)`` warns and makes the range ``[5, 5]``."""
    x = RooRealVar("x", "x", 1, -10, 10)
    x.setRange(5, 2)
    assert (x.getMin(), x.getMax()) == (5.0, 5.0)
    assert capsys.readouterr().out == (
        "[#0] WARNING:InputArguments -- RooRealVar::setRange(x): Proposed new fit max. "
        "smaller than min., setting max. to min.\n"
    )


@pytest.mark.xfail(strict=True, reason="setRange clips the value; ROOT 6.40 leaves it alone")
def test_narrowing_the_range_leaves_the_value_where_it_was() -> None:
    """ROOT's ``setRange(2, 5)`` on ``x = 1`` keeps 1: only ``setVal`` checks the range."""
    x = RooRealVar("x", "x", 1, -10, 10)
    x.setRange(2, 5)
    assert x.getVal() == 1.0


def test_raising_the_minimum_moves_the_value_up_with_it() -> None:
    """ROOT's ``setMin(3)`` on ``x = 1`` in ``[-10, 10]`` gives ``[3, 10]`` and ``x = 3``."""
    x = RooRealVar("x", "x", 1, -10, 10)
    x.setMin(3)
    assert (x.getMin(), x.getMax(), x.getVal()) == (3.0, 10.0, 3.0)
    x.setMax(5)
    assert (x.getMin(), x.getMax(), x.getVal()) == (3.0, 5.0, 3.0)


@pytest.mark.xfail(strict=True, reason="setMax below the minimum moves the minimum down")
def test_lowering_the_maximum_below_the_minimum_warns_and_closes_the_range(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ROOT's ``setMax(-20)`` on ``[3, 10]`` warns and makes the range ``[3, 3]``."""
    x = RooRealVar("x", "x", 3, 3, 10)
    x.setMax(-20)
    assert (x.getMin(), x.getMax()) == (3.0, 3.0)
    assert capsys.readouterr().out == (
        "[#0] WARNING:InputArguments -- RooRealVar::setMax(x): Proposed new fit max. "
        "smaller than min., setting max. to min.\n"
    )


def test_a_named_range_is_announced_once_and_then_only_moved(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ROOT says a named range was made the first time only; each end reads back as set."""
    x = RooRealVar("x", "x", 1, -10, 10)
    x.setRange("sig", 1, 3)
    x.setRange("sig", 1, 4)
    assert x.getRange("sig") == (1.0, 4.0)
    assert x.hasRange("sig")
    assert x.hasBinning("sig")
    assert x.getBins("sig") == 1
    assert capsys.readouterr().out == (
        "[#1] INFO:Eval -- RooRealVar::setRange(x) new range named 'sig' "
        "created with bounds [1,3]\n"
    )


def test_one_end_of_a_named_range_can_be_moved_by_name() -> None:
    """``setMin("r", 2)`` and ``setMax("r", 4)``: ROOT's ``[2, 10]`` and then ``[2, 4]``."""
    x = RooRealVar("x", "x", 1, -10, 10)
    x.setMin("r", 2)
    assert x.getRange("r") == (2.0, 10.0)
    x.setMax("r", 4)
    assert x.getRange("r") == (2.0, 4.0)
    x.setMax("q", -20)
    assert x.getRange("q") == (-10.0, -10.0)
    x.setRange("inv", 3, 1)
    assert x.getRange("inv") == (3.0, 3.0)
    assert x.inRange(1.5, None)
    assert x.inRange(3, "r")
    assert not x.inRange(5, "r")


@pytest.mark.xfail(strict=True, reason="a range made by setMin/setMax is announced as setRange's")
def test_a_named_range_made_by_one_end_is_announced_with_default_bounds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ROOT makes the range with ``getBinning`` first, and says so, before moving its end."""
    x = RooRealVar("x", "x", 1, -10, 10)
    x.setMin("r", 2)
    assert capsys.readouterr().out == (
        "[#1] INFO:Eval -- RooRealVar::getBinning(x) new range named 'r' "
        "created with default bounds\n"
    )


def test_an_unknown_range_or_binning_is_the_default_one() -> None:
    """``getMin("nope")`` is the variable's own minimum and ``getBins("nope")`` its 100 bins."""
    x = RooRealVar("x", "x", 1, -10, 10)
    assert x.getMin("nope") == -10.0
    assert x.getBins("nope") == 100
    assert x.numBins() == 100
    assert not x.hasRange("nope")
    assert not x.hasBinning("nope")
    assert x.getBinning() is x.getBinning("nope")
    made = x.getBinning("fly", createOnTheFly=True)
    assert made.GetName() == "fly"
    assert x.getRange("fly") == (-10.0, 10.0)


@pytest.mark.xfail(strict=True, reason="hasRange('') is true here; ROOT has no range called ''")
def test_the_default_range_is_not_a_named_one() -> None:
    """ROOT's ``hasRange("")`` is ``false``: the default range has no name to look up."""
    assert not RooRealVar("x", "x", 1, -10, 10).hasRange("")


def test_a_named_range_can_be_removed_and_the_default_one_opened_up() -> None:
    """``removeMin`` and ``removeMax`` send the default range's ends to infinity, as ROOT does."""
    x = RooRealVar("x", "x", 1, 0, 10)
    x.setRange("r", 1, 2)
    x.removeRange("r")
    assert not x.hasBinning("r")
    x.removeMin()
    assert (x.hasMin(), x.getMax()) == (False, 10.0)
    x.removeMax()
    assert not x.hasMax()
    x.setRange(0, 10)
    x.removeRange()
    assert x.getRange() == (-INF, INF)


@pytest.mark.xfail(strict=True, reason="removeRange(name) drops the range; ROOT opens it up")
def test_removing_a_named_range_keeps_it_with_open_ends() -> None:
    """ROOT's ``removeRange("r")`` is ``removeMin("r"); removeMax("r")``: ``r`` still exists."""
    x = RooRealVar("x", "x", 1, 0, 10)
    x.setRange("r", 1, 2)
    x.removeRange("r")
    assert x.hasRange("r")
    assert x.getRange("r") == (-INF, INF)


def test_setting_a_bin_puts_the_value_at_its_centre_and_refuses_one_past_the_end(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With 20 bins over ``[-10, 10]`` bin 3 is centred on -6.5; bin 30 is ROOT's error."""
    x = RooRealVar("x", "x", 1, -10, 10)
    x.setBins(20)
    x.setBin(30)
    assert x.getVal() == 1.0
    x.setBin(3)
    assert x.getVal() == -6.5
    assert capsys.readouterr().out == (
        "[#0] ERROR:InputArguments -- RooAbsRealLValue::setBin(x) ERROR: bin index 30 "
        "is out of range (0,19)\n"
    )


def test_a_named_binning_sits_beside_the_default_one() -> None:
    """``setBins(5, "coarse")`` leaves the default 100 bins; ``setBin`` may use either."""
    x = RooRealVar("x", "x", 1, -10, 10)
    x.setBins(5, "coarse")
    assert (x.getBins("coarse"), x.getBins()) == (5, 100)
    x.setBins(7, "B")
    x.setBin(2, "B")
    assert x.getVal() == pytest.approx(-2.8571428571428568, rel=1e-15)
    assert sorted(x.getBinningNames()) == ["", "B", "coarse"]


@pytest.mark.xfail(strict=True, reason="bin centres come from edges; ROOT's from lo + (i+.5)*w")
def test_a_bin_centre_is_roots_to_the_last_bit() -> None:
    """ROOT's ``binCenter`` is ``xlo + (i + 0.5) * binw``, here -2.8571428571428568."""
    x = RooRealVar("x", "x", 1, -10, 10)
    x.setBins(7, "B")
    x.setBin(2, "B")
    assert x.getVal() == -2.8571428571428568


@pytest.mark.xfail(strict=True, reason="binning names come in the order made; ROOT's are sorted")
def test_the_binning_names_come_in_roots_order() -> None:
    """ROOT lists ``["", "B", "coarse"]`` for binnings made as ``coarse`` and then ``B``."""
    x = RooRealVar("x", "x", 1, -10, 10)
    x.setBins(5, "coarse")
    x.setBins(7, "B")
    assert x.getBinningNames() == ["", "B", "coarse"]


def test_a_binning_of_any_spacing_can_be_the_default_or_named() -> None:
    """``setBinning(b)`` makes ``b`` the default binning; with a name, one of the others."""
    v = RooRealVar("v", "v", 0.5, 0, 1)
    b = RooBinning(0, 1)
    b.addBoundary(0.2)
    v.setBinning(b)
    assert (v.getBins(), v.getVal()) == (2, 0.5)
    v.setBin(1)
    assert v.getVal() == 0.6
    fine = RooBinning(4, 0.0, 1.0)
    v.setBinning(fine, "fine")
    assert (v.getBins("fine"), v.getBinning("fine").GetName()) == (4, "fine")


def test_a_cloned_variable_has_its_own_binning(capsys: pytest.CaptureFixture[str]) -> None:
    """ROOT's clone keeps ``B(2)`` when the original is given four bins afterwards."""
    v = RooRealVar("v", "v", 0.6, 0, 1)
    b = RooBinning(0, 1)
    b.addBoundary(0.2)
    v.setBinning(b)
    copy = v.clone("v2")
    v.setBins(4)
    copy.Print()
    assert (copy.getBins(), v.getBins()) == (2, 4)
    assert capsys.readouterr().out == "RooRealVar::v2 = 0.6  L(0 - 1) B(2) \n"


def test_a_range_whose_ends_are_variables_prints_where_they_are_now(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ROOT's ``setRange(lo, hi)`` range follows ``lo``: ``L(-1 - 3)`` once ``lo`` is -1."""
    lo = RooRealVar("lo", "lo", -2, -5, 5)
    hi = RooRealVar("hi", "hi", 3, -5, 5)
    t = RooRealVar("t", "t", 0, -10, 10)
    t.setRange(lo, hi)
    assert (t.getMin(), t.getMax(), t.getBins()) == (-2.0, 3.0, 100)
    lo.setVal(-1)
    t.Print()
    assert capsys.readouterr().out == "RooRealVar::t = 0  L(-1 - 3) \n"
    assert isinstance(t.getBinning(), RooParamBinning)


def test_a_moving_range_may_have_a_number_for_one_end() -> None:
    """A plain number as one end becomes a constant of that value, as ROOT's ``RooConstVar``."""
    lo = RooRealVar("lo", "lo", -1, -5, 5)
    t = RooRealVar("t", "t", 0, -10, 10)
    t.setRange(lo, 4.0)
    assert t.getRange() == (-1.0, 4.0)
    assert t.getBinning().servers()[1].GetName() == "4"


def test_an_error_of_zero_counts_as_an_error_unless_zero_is_not_allowed() -> None:
    """``hasError()`` is true for an error of 0, ``hasError(false)`` only for a positive one."""
    x = RooRealVar("x", "x", 1, -10, 10)
    assert (x.hasError(), x.getError()) == (False, 0.0)
    x.setError(0.0)
    assert (x.hasError(), x.hasError(False)) == (True, False)
    x.setError(0.2)
    assert (x.getErrorLo(), x.getErrorHi()) == (-0.2, 0.2)
    x.removeError()
    assert (x.hasError(), x.getError()) == (False, 0.0)


def test_an_asymmetric_error_is_used_for_both_sides_once_set() -> None:
    """ROOT's ``getErrorLo``/``Hi`` read the asymmetric error, and 0 counts unless refused."""
    x = RooRealVar("x", "x", 1, -10, 10)
    x.setAsymError(0.0, 0.0)
    assert (x.hasAsymError(), x.hasAsymError(False)) == (True, False)
    assert (x.getErrorLo(), x.getErrorHi()) == (0.0, 0.0)
    x.setAsymError(-0.1, 0.3)
    assert (x.getAsymErrorLo(), x.getAsymErrorHi()) == (-0.1, 0.3)
    assert (x.getErrorLo(), x.getErrorHi()) == (-0.1, 0.3)
    x.removeAsymError()
    assert not x.hasAsymError()


@pytest.mark.xfail(strict=True, reason="getAsymErrorLo/Hi and getErrorLo/Hi differ from ROOT")
def test_the_error_sides_of_a_fresh_or_cleared_variable_are_roots() -> None:
    """ROOT's fresh ``getErrorLo``/``Hi`` are 1 and -1, and a removed asymmetric error reads 0."""
    x = RooRealVar("x", "x", 1, -10, 10)
    assert (x.getErrorLo(), x.getErrorHi()) == (1.0, -1.0)
    x.setAsymError(-0.1, 0.3)
    x.removeAsymError()
    assert (x.getAsymErrorLo(), x.getAsymErrorHi()) == (0.0, 0.0)


def test_a_title_takes_the_unit_and_a_plot_label_can_be_set() -> None:
    """ROOT's ``getTitle(true)`` is ``the x (cm)``, and the label is the name until set."""
    x = RooRealVar("x", "the x", 1, -10, 10, "cm")
    assert (x.getTitle(), x.getTitle(True), x.getPlotLabel(), x.getUnit()) == (
        "the x",
        "the x (cm)",
        "x",
        "cm",
    )
    x.setPlotLabel("X!")
    x.setUnit("mm")
    assert (x.getPlotLabel(), x.getTitle(True)) == ("X!", "the x (mm)")


def test_a_variable_can_be_made_constant_and_free_again() -> None:
    """``setConstant`` is what a fit reads to leave a parameter out."""
    k = RooRealVar("k", "k", 1, 0, 2)
    k.setConstant()
    assert k.isConstant()
    k.setConstant(False)
    assert not k.isConstant()


def test_a_variable_takes_another_variables_value_and_errors() -> None:
    """Copying values - a snapshot restored - brings the errors along; a constant, only a value."""
    k = RooRealVar("k", "k", 2, 0, 5)
    k.setError(0.1)
    k.setAsymError(-0.2, 0.3)
    j = RooRealVar("j", "j", 1, 0, 5)
    j.copy_value_from(k)
    assert (j.getVal(), j.getError(), j.getAsymErrorLo(), j.getAsymErrorHi()) == (
        2.0,
        0.1,
        -0.2,
        0.3,
    )
    i = RooRealVar("i", "i", 1, 0, 5)
    i.copy_value_from(RooConstVar("c", "c", 4.0))
    assert (i.getVal(), i.hasError()) == (4.0, False)


def test_a_variable_formats_itself_to_the_precision_of_its_error() -> None:
    """ROOT's ``format(2, "NEU")``, ``format(Format("NE", AutoPrecision(1)))`` and ``format()``."""
    m = RooRealVar("mean", "mean", 1.017463, -10, 10, "GeV")
    m.setError(0.0300144)
    assert m.format(2, "NEU") == "mean =  1.017 +/- 0.030 GeV"
    command = RooCmdArg("Format", "NE", RooCmdArg("AutoPrecision", 1))
    assert m.format(command) == "mean =  1.02 +/- 0.03"
    assert m.format() == " 1.0"


def test_a_variable_makes_a_frame_of_its_range() -> None:
    """``x.frame()`` is an empty plot over the variable's range."""
    x = RooRealVar("x", "x", 1, -10, 10)
    frame = x.frame()
    assert frame.numItems() == 0
    assert (frame.GetXmin(), frame.GetXmax(), frame.getPlotVar().GetName()) == (-10.0, 10.0, "x")


def test_a_settable_value_reads_a_column_when_evaluated_over_data() -> None:
    """``compute`` takes the variable's column from the context, else the value it has."""
    x = RooRealVar("x", "x", 1, -10, 10)
    assert x.compute({}) == 1.0
    assert list(x.compute({"x": np.array([2.0, 3.0])})) == [2.0, 3.0]
    assert (x.isFundamental(), x.isDerived()) == (True, False)


def test_a_dataset_keeps_a_variables_value_unclipped_and_checks_it_against_the_range() -> None:
    """A data store loads values as they are; ``isValidReal`` says which fit the range."""
    x = RooRealVar("x", "x", 1, 0, 3)
    x.load_value(7.5)
    assert x.stored_value() == 7.5
    assert x.can_hold(2.5)
    assert not x.can_hold(4.0)
    assert x.value_text(0.1 + 0.2) == "0.3"


def test_the_base_of_a_settable_value_has_a_value_and_prints_it() -> None:
    """``RooAbsRealLValue`` alone: a value on the whole line that can be copied from another."""
    base = RooAbsRealLValue("l", "l")
    assert (base.getVal(), base.printValue()) == (0.0, "0")
    base.copy_value_from(RooConstVar("c", "c", 1.5))
    assert base.getVal() == 1.5
    base.setVal(-2.5)
    assert base.getVal() == -2.5


def test_a_constant_keeps_its_value_whatever_is_copied_into_it() -> None:
    """``RooConstVar`` is fundamental and constant, and ignores the values of others."""
    c = RooConstVar("c", "c", 1.5)
    assert (c.getVal(), c.isConstant(), c.isFundamental(), c.compute({"c": 9.0})) == (
        1.5,
        True,
        True,
        1.5,
    )
    c.copy_value_from(RooConstVar("d", "d", 7.0))
    assert c.getVal() == 1.5
    assert c.printValue() == "1.5"
