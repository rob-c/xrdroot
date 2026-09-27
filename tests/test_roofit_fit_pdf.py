"""``RooAbsPdf``'s own questions - values, norms, ranges, yields - answered as ROOT 6.40.04 does.

Each number below was printed by ROOT through PyROOT with ``repr`` for the
same density: a Gaussian of mean 0 and width 2 in x on [-10, 10], at x = 1.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from xrdroot.roofit.pdf import RooAbsPdf, as_array, check_range, names, normalized
from xrdroot.roofit.pdfs.basic import RooExponential, RooGaussian, RooPolynomial
from xrdroot.roofit.pdfs.extend import RooExtendPdf
from xrdroot.roofit.variables import RooConstVar, RooRealVar


def _gauss() -> tuple[Any, Any, Any]:
    x = RooRealVar("x", "x", 1.0, -10, 10)
    y = RooRealVar("y", "y", 0.5, -5, 5)
    g = RooGaussian("g", "g", x, RooRealVar("m", "m", 0, -5, 5), RooRealVar("s", "s", 2, 0.1, 10))
    return g, x, y


def test_a_density_is_its_formula_over_its_integral_as_roots_is(capsys: Any) -> None:
    """Unnormalised, normalised over x, the norm and the log: ROOT's to the last bit - and
    ``Print`` shows the value over the norm last asked for."""
    g, x, _ = _gauss()
    assert g.getVal() == 0.8824969025845955
    assert g.getVal([x]) == 0.17603276430228695
    assert g.getNorm([x]) == 5.01325367514626
    assert g.getLogVal([x]) == -1.7370851404613097
    assert g.getNorm() == 1.0
    g.Print()
    assert capsys.readouterr().out == "RooGaussian::g[ x=x mean=m sigma=s ] = 0.882497/5.01325\n"


@pytest.mark.xfail(strict=True, reason="a norm over nothing it depends on is its own value")
def test_a_norm_over_variables_the_density_does_not_depend_on_is_one() -> None:
    """Normalising over y, which the Gaussian does not use, divides by one: ROOT's norm is 1
    and the value the formula's own."""
    g, _, y = _gauss()
    assert g.getNorm([y]) == 1.0
    assert g.getVal([y]) == 0.8824969025845955


def test_a_norm_range_normalises_over_part_of_the_range_until_it_is_cleared() -> None:
    """``setNormRange("narrow")`` normalises over [-1, 1]: ROOT's 0.4597; cleared, the whole
    range again."""
    g, x, _ = _gauss()
    x.setRange("narrow", -1, 1)
    g.setNormRange("narrow")
    assert (g.normRange(), g.getVal([x])) == ("narrow", 0.45970542269959436)
    g.setNormRange("")
    assert not g.normRange()
    assert g.getVal([x]) == 0.17603276430228695


def test_a_density_says_whether_it_expects_a_number_of_events() -> None:
    """A Gaussian cannot be extended and expects nothing; an extended one can, but need not,
    be, and expects its yield."""
    g, x, _ = _gauss()
    assert (g.extendMode(), g.canBeExtended(), g.mustBeExtended()) == (0, False, False)
    assert g.expectedEvents([x]) == 0.0
    assert g.expectedEvents() == 0.0
    e = RooExtendPdf("e", "e", g, RooRealVar("n", "n", 7, 0, 100))
    assert (e.extendMode(), e.canBeExtended(), e.mustBeExtended()) == (1, True, False)
    assert e.expectedEvents([x]) == 7.0
    assert RooAbsPdf.MustBeExtended == 2


def test_the_extended_term_is_roots_poisson_term() -> None:
    """``expected - observed log(expected)``, nothing for nothing, scaled by ``sumw2/observed``
    for weighted events, and NaN for a negative expectation: ROOT's values."""
    g, _, _ = _gauss()
    e = RooExtendPdf("e", "e", g, RooRealVar("n", "n", 7, 0, 100))
    assert e.extendedTerm(5, 7.5) == -2.5745151027113238
    assert e.extendedTerm(0.0, 0.0) == 0.0
    assert e.extendedTerm(5, 7.5, 4.0) == -2.059612082169059
    assert math.isnan(e.extendedTerm(5, -1.0))


@pytest.mark.xfail(strict=True, reason="an expectation of zero gives NaN, not infinity")
def test_the_extended_term_of_events_where_none_are_expected_is_infinite() -> None:
    """ROOT computes ``0 - 5 log(0)``: infinitely unlikely, not undefined."""
    g, _, _ = _gauss()
    e = RooExtendPdf("e", "e", g, RooRealVar("n", "n", 7, 0, 100))
    assert e.extendedTerm(5, 0.0) == math.inf


def test_a_negative_density_has_no_logarithm() -> None:
    """``1 - 0.5 x`` at x = 5 is -1.5: normalised it is NaN, and so is its log, as in ROOT."""
    x = RooRealVar("x", "x", 5.0, -10, 10)
    p = RooPolynomial("p", "p", x, [RooConstVar("c", "", -0.5)])
    assert p.getVal() == -1.5
    assert math.isnan(p.getVal([x]))
    assert math.isnan(p.getLogVal([x]))


def test_normalising_packs_how_bad_a_value_is_into_its_nan() -> None:
    """A negative value or norm becomes a NaN carrying how far below zero it was, which the
    likelihood unpacks to tell Minuit how far to back away; nothing over nothing is nothing."""
    from xrdroot.roofit.nanpack import unpack

    assert normalized(2.0, 4.0) == 0.5
    assert normalized(0.0, 0.0) == 0.0
    found = normalized(np.array([-1.5, 1.0, 1.0, math.nan]), np.array([2.0, -3.0, 0.0, 1.0]))
    assert np.isnan(found).all()
    assert unpack(found).tolist() == [1.5, 3.0, 0.0, 0.0]
    assert math.isnan(normalized(-1.0, 1.0))


def test_a_parameter_that_can_leave_its_safe_range_is_warned_of(capsys: Any) -> None:
    """``checkRangeOfParameters``: a width that may reach zero or below is named, with the safe
    range in ROOT's notation."""
    g, _, _ = _gauss()
    capsys.readouterr()
    wide = RooRealVar("w", "w", 1, -1, 5)
    check_range(g, [wide, RooConstVar("k", "", 1)], 0.0)
    check_range(g, [wide], -math.inf, 4.0, closed=True, extra="Mind it.")
    check_range(g, [RooRealVar("z", "z", 1, 0, 3)], 0.0, 3.0, closed=True)
    assert capsys.readouterr().out == (
        "[#0] WARNING:InputArguments -- The parameter 'w' with range [-1, 5] of the RooGaussian "
        "'g' exceeds the safe range of (0, inf). Advise to limit its range.\n"
        "[#0] WARNING:InputArguments -- The parameter 'w' with range [-1, 5] of the RooGaussian "
        "'g' exceeds the safe range of [-inf, 4]. Advise to limit its range.\nMind it.\n"
    )


def test_names_and_arrays_are_had_from_whatever_holds_them() -> None:
    """Small helpers: the names of some variables, and values as a float array."""
    _, x, y = _gauss()
    assert names([x, y]) == frozenset(["x", "y"])
    assert as_array([1, 2]).dtype == np.float64


def test_a_conditional_fit_normalises_each_event_at_its_own_conditional_value() -> None:
    """The Gaussian's mean is each event's y: the norm is one per event, not one for all, and
    the fitted width is ROOT's."""
    from xrdroot.roofit.data.dataset import RooDataSet
    from xrdroot.roofit.rng import generator

    generator().SetSeed(4357)
    x = RooRealVar("x", "x", -10, 10)
    y = RooRealVar("y", "y", -2, 2)
    s = RooRealVar("s", "s", 2, 0.1, 10)
    gc = RooGaussian("gc", "gc", x, y, s)
    proto = RooDataSet("proto", "proto", [y])
    proto.add_columns({"y": np.array([-1.5, -0.5, 0.0, 0.7, 1.9])})
    data = gc.generate([x], 20, ProtoData=proto)
    s.setVal(1.0)
    result = gc.fitTo(data, Save=True, PrintLevel=-1, ConditionalObservables=[y])
    assert (s.getVal(), s.getError()) == pytest.approx((1.619789494366564, 0.2558258583956), 1e-8)
    assert result.minNll() == pytest.approx(38.028886252088455, abs=1e-9)


def test_a_range_that_ends_at_a_per_event_value_normalises_each_event_there() -> None:
    """``t.setRange(0, tmax)`` with tmax an observable: each event's exponential is normalised
    on its own [0, tmax] - ROOT's fit."""
    from xrdroot.roofit.data.dataset import RooDataSet

    t = RooRealVar("t", "t", 0, 10)
    tmax = RooRealVar("tmax", "tmax", 1, 10)
    t.setRange(RooConstVar("zero", "", 0), tmax)
    c = RooRealVar("c", "c", -0.5, -3, 0)
    ex = RooExponential("ex", "ex", t, c)
    data = RooDataSet("dt", "dt", [t, tmax])
    data.add_columns(
        {
            "t": np.array([0.2, 0.5, 1.5, 2.5, 0.1, 4.0, 0.9]),
            "tmax": np.array([1.0, 3.0, 2.0, 6.0, 1.5, 8.0, 1.2]),
        }
    )
    result = ex.fitTo(data, Save=True, PrintLevel=-1, ConditionalObservables=[tmax])
    assert (c.getVal(), c.getError()) == pytest.approx((-0.1725560550447029, 0.31789133100), 1e-8)
    assert result.minNll() == pytest.approx(6.110393063026177, abs=1e-9)


def test_the_integral_of_a_density_over_part_of_its_range_is_its_share_there() -> None:
    """``createIntegral(x, NormSet(x), Range("narrow"))``: the normalised density's share of
    [-1, 1], ROOT's value."""
    x = RooRealVar("x", "x", 1.0, -10, 10)
    x.setRange("narrow", -1, 1)
    g = RooGaussian("g", "g", x, RooRealVar("m", "m", 0, -5, 5), RooConstVar("two", "", 2.0))
    found = g.createIntegral([x], NormSet=[x], Range="narrow")
    assert found.getVal() == pytest.approx(0.3829251420802139, rel=1e-14)
