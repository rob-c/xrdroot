"""``fitTo`` and ``createNLL``: the likelihood RooFit builds from a fit's options, against ROOT.

Every expected number was printed by ROOT 6.40.04 through PyROOT for the
same model and the same 50 events, drawn after
``RooRandom::randomGenerator()->SetSeed(4357)``: likelihood values to the
last bit, fitted values to Minuit's tolerance, messages to the character.
"""

from __future__ import annotations

from typing import Any

import pytest

from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.fitting.kahan import Kahan
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.pdfs.extend import RooExtendPdf
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooRealVar

#: ROOT's fit of the Gaussian to all 50 events: m, s and their errors, and the minimum.
FULL = (0.048219535850741575, 1.8724677853175358, 0.26468352349135704, 0.1871715476438679)
FULL_NLL = 102.3103519117139


def _gauss() -> tuple[Any, Any, Any, Any, Any]:
    """A Gaussian in x, its parameters, and 50 events of it drawn with ROOT's seed."""
    generator().SetSeed(4357)
    x = RooRealVar("x", "x", -10, 10)
    m = RooRealVar("m", "m", 0, -5, 5)
    s = RooRealVar("s", "s", 2, 0.1, 10)
    g = RooGaussian("g", "g", x, m, s)
    return g, g.generate([x], 50), x, m, s


def _fitted(m: Any, s: Any) -> list[float]:
    return [m.getVal(), s.getVal(), m.getError(), s.getError()]


def test_an_extendable_density_is_fitted_extended_unless_told_otherwise(capsys: Any) -> None:
    """A density that says how many events it expects gets the Poisson term by default - RooFit
    says so - and the yield is fitted too; ``Extended(false)`` leaves it out."""
    g, data, _, m, s = _gauss()
    n = RooRealVar("n", "n", 40, 0, 100)
    e = RooExtendPdf("e", "e", g, n)
    capsys.readouterr()
    result = e.fitTo(data, Save=True, PrintLevel=-1)
    out = capsys.readouterr().out
    assert out.startswith(
        "[#1] INFO:Minimization -- p.d.f. provides expected number of events, including "
        "extended term in likelihood.\n"
        "[#1] INFO:Fitting -- RooAbsPdf::fitTo(e) fixing normalization set for coefficient "
        "determination to observables in data\n"
    )
    assert "RooAddition::defaultErrorLevel(nll_e_gData) Summation contains a RooNLLVar" in out
    expected = [0.047947128523434246, 1.8725234971696316, 0.26469131791535994, 0.1871843089097]
    assert _fitted(m, s) == pytest.approx(expected, rel=1e-8)
    assert (n.getVal(), n.getError()) == pytest.approx((50.00014283647089, 7.0475412438586), 1e-8)
    assert result.minNll() == pytest.approx(-43.29079796820673, abs=1e-9)
    for one, value in ((m, 0.0), (s, 2.0), (n, 40.0)):
        one.setVal(value)
        one.setError(0)
    result = e.fitTo(data, Save=True, PrintLevel=-1, Extended=False)
    assert "provides expected number" not in capsys.readouterr().out
    assert [m.getVal(), s.getVal()] == pytest.approx(FULL[:2], rel=1e-9)
    assert (n.getVal(), result.minNll()) == (40.0, pytest.approx(FULL_NLL, abs=1e-9))


def test_extended_on_a_density_that_expects_nothing_fits_as_if_it_were_not() -> None:
    """``Extended(true)`` where the density has no yield adds no term: ROOT's plain fit."""
    g, data, _, m, s = _gauss()
    result = g.fitTo(data, Save=True, PrintLevel=-1, Extended=True)
    assert _fitted(m, s) == pytest.approx(FULL, rel=1e-8)
    assert result.minNll() == pytest.approx(FULL_NLL, abs=1e-9)


#: Fits in a named range, and in two joined: m, s, their errors, and the minimum.
IN_SIG = (-0.7324308607875208, 1.8699027615477501, 0.43356387917279515, 0.4400007944422336)
IN_SIDES = (0.30589711838871025, 2.0767443668082466, 0.3374142495939065, 0.2684803207744433)


def test_a_fit_in_a_range_normalises_there_and_names_its_ranges_for_plotting() -> None:
    """``Range("sig")`` keeps the events there and normalises over it; ``Range("left,right")``
    joins two, and the density remembers them as ``fit_nll_<pdf>_<data>[_<range>]``."""
    g, data, x, m, s = _gauss()
    x.setRange("sig", -3, 3)
    result = g.fitTo(data, Save=True, PrintLevel=-1, Range="sig")
    assert _fitted(m, s) == pytest.approx(IN_SIG, rel=1e-8)
    assert result.minNll() == pytest.approx(74.95574471337984, abs=1e-9)
    assert g.getStringAttribute("fitrange") == "fit_nll_g_gData"
    assert (x.getMin("fit_nll_g_gData"), x.getMax("fit_nll_g_gData")) == (-3.0, 3.0)
    for one, value in ((m, 0.0), (s, 2.0)):
        one.setVal(value)
        one.setError(0)
    x.setRange("left", -10, -1)
    x.setRange("right", 1, 10)
    result = g.fitTo(data, Save=True, PrintLevel=-1, Range="left,right")
    assert _fitted(m, s) == pytest.approx(IN_SIDES, rel=1e-8)
    assert result.minNll() == pytest.approx(48.8809414365939, abs=1e-9)
    assert g.getStringAttribute("fitrange") == "fit_nll_g_gData_left,fit_nll_g_gData_right"
    assert (x.getMin("fit_nll_g_gData_left"), x.getMax("fit_nll_g_gData_right")) == (-10.0, 10.0)
    g.fitTo(data, PrintLevel=-1)
    assert not g.getStringAttribute("fitrange")


@pytest.mark.xfail(strict=True, reason="Range(low, high) is ignored: the fit takes every event")
def test_a_fit_in_a_range_given_by_its_ends_is_the_fit_in_that_range() -> None:
    """ROOT makes ``Range(-3, 3)`` a range called ``fit`` and fits there, as in ``"sig"``."""
    g, data, _, m, s = _gauss()
    result = g.fitTo(data, Save=True, PrintLevel=-1, Range=(-3, 3))
    assert _fitted(m, s) == pytest.approx(IN_SIG, rel=1e-8)
    assert result.minNll() == pytest.approx(74.95574471337984, abs=1e-9)
    assert g.getStringAttribute("fitrange") == "fit_nll_g_gData"


def test_offset_fits_find_roots_minimum_and_report_the_offset_likelihood() -> None:
    """``Offset(true)`` takes the first value off every other: the saved minimum is ROOT's
    offset one, and the parameters ROOT's."""
    g, data, _, m, s = _gauss()
    result = g.fitTo(data, Save=True, PrintLevel=-1, Offset=True)
    expected = [0.048220275870949736, 1.872468056171597, 0.2646836340762951, 0.18717157947085206]
    assert _fitted(m, s) == pytest.approx(expected, rel=1e-6)
    assert result.minNll() == pytest.approx(-0.22226038572752316, abs=1e-9)


def test_a_duplicated_option_is_named_and_minos_can_be_asked_for_some_parameters(
    capsys: Any,
) -> None:
    """RooFit warns of an option given twice; ``Minos({m})`` gives ``m`` asymmetric errors,
    ROOT's, and leaves ``s`` with HESSE's."""
    g, data, _, m, s = _gauss()
    capsys.readouterr()
    level = RooCmdArg("PrintLevel", -1)
    g.fitTo(data, level, level, RooCmdArg("Minos", [m]))
    assert capsys.readouterr().out.startswith(
        "[#0] WARNING:InputArguments -- fitTo(g) WARNING: argument PrintLevel is duplicated\n"
    )
    assert m.hasAsymError() and not s.hasAsymError()
    assert (m.getAsymErrorLo(), m.getAsymErrorHi()) == pytest.approx(
        (-0.26617755034923724, 0.2661034440297161), rel=1e-8
    )


def test_the_likelihood_is_roots_to_the_last_bit_and_knows_its_parameters() -> None:
    """``createNLL``'s value at the starting point is ROOT's exactly; its parameters are the
    density's, less any observables asked to be left out."""
    g, data, x, m, _ = _gauss()
    nll = g.createNLL(data)
    assert nll.getVal() == 102.53261229765363
    assert nll.GetName() == "nll_g_over_g_Int[x]_gData"
    assert nll.defaultErrorLevel() == 0.5
    assert nll.getParameters().names() == ["m", "s"]
    assert nll.getParameters(data).names() == ["m", "s"]
    assert nll.getParameters([m]).names() == ["s"]
    assert nll.getParameters([x, m]).names() == ["s"]


def test_a_compensated_sum_adds_as_kahan_sum_adds() -> None:
    """``KahanSum`` keeps the rounding of each addition and gives it back to the next: a
    thousand tenths are 100, where adding them in turn gives 99.9999999999986, and one at a
    time is the same as all at once."""
    naive = 0.0
    for value in [0.1] * 1000:
        naive += value
    assert naive == 99.9999999999986
    found = Kahan().extend([0.1] * 1000)
    assert (found.total, found.carry) == (100.0, -5.551115123125783e-15)
    one_by_one = Kahan()
    for value in [0.1] * 1000:
        one_by_one.add(value)
    assert (one_by_one.total, one_by_one.carry) == (found.total, found.carry)


def _simultaneous() -> tuple[Any, Any, Any, list[Any]]:
    """Two Gaussians sharing a mean, one per state of ``c``: 30 events of one, 20 of the other."""
    from xrdroot.roofit.categories import RooCategory
    from xrdroot.roofit.data.dataset import RooDataSet
    from xrdroot.roofit.pdfs.simultaneous import RooSimultaneous

    generator().SetSeed(4357)
    x = RooRealVar("x", "x", -10, 10)
    m = RooRealVar("m", "m", 0, -5, 5)
    s1 = RooRealVar("s1", "s1", 2, 0.1, 10)
    s2 = RooRealVar("s2", "s2", 3, 0.1, 10)
    g1 = RooGaussian("g1", "g1", x, m, s1)
    g2 = RooGaussian("g2", "g2", x, m, s2)
    c = RooCategory("c", "c")
    c.defineType("a", 0)
    c.defineType("b", 1)
    first, second = g1.generate([x], 30), g2.generate([x], 20)
    both = RooDataSet("d", "d", [x], Index=c, Import={"a": first, "b": second})
    one = RooDataSet("d3", "d3", [x], Index=c, Import={"a": first})
    sim = RooSimultaneous("sim", "sim", c)
    sim.addPdf(g1, "a")
    sim.addPdf(g2, "b")
    return sim, both, one, [m, s1, s2]


def test_a_simultaneous_likelihood_is_each_channels_with_roots_log_term() -> None:
    """Each channel's events are scored by their own density, plus ``N log(channels)``: ROOT's
    value exactly, and its fit, for both channels or only one with events."""
    sim, both, one, params = _simultaneous()
    assert sim.createNLL(both).getVal() == 145.28211517262633
    assert sim.createNLL(one).getVal() == 83.26171963861647
    result = sim.fitTo(both, Save=True, PrintLevel=-1)
    expected = [0.06356425344098669, 1.9373825440400445, 2.661501463656083]
    assert [p.getVal() for p in params] == pytest.approx(expected, rel=1e-9)
    assert result.minNll() == pytest.approx(144.96771354990528, abs=1e-9)


#: ROOT's fit of the Gaussian with its mean constrained to 0.5 +/- 0.1: m, s, errors, minimum.
CONSTRAINED = (0.4460531001250796, 1.913884680325121, 0.09428786806491227, 0.1922650951844156)


@pytest.mark.xfail(strict=True, reason="a constraint is normalised over its constants too")
def test_an_external_constraint_pulls_the_fit_as_roots_does() -> None:
    """``ExternalConstraints`` multiplies the likelihood by the constraint, normalised over the
    fit's parameters (ROOT: "normalize constraints with respect to the parameters (m,s)")."""
    from xrdroot.roofit.variables import RooConstVar

    g, data, _, m, s = _gauss()
    mc = RooGaussian("mc", "mc", m, RooConstVar("c1", "", 0.5), RooConstVar("c2", "", 0.1))
    result = g.fitTo(data, Save=True, PrintLevel=-1, ExternalConstraints=[mc])
    assert _fitted(m, s) == pytest.approx(CONSTRAINED, rel=1e-6)
    assert result.minNll() == pytest.approx(102.17618960920294, abs=1e-8)


@pytest.mark.xfail(strict=True, reason="a factor of parameters alone is scored once per event")
def test_a_product_with_a_factor_of_parameters_only_takes_it_as_a_constraint() -> None:
    """ROOT finds ``mc`` in ``g * mc`` depends on no observable and takes it once, as a
    constraint - the same fit as ``ExternalConstraints``."""
    from xrdroot.roofit.pdfs.prodpdf import RooProdPdf
    from xrdroot.roofit.variables import RooConstVar

    g, data, _, m, s = _gauss()
    mc = RooGaussian("mc", "mc", m, RooConstVar("c1", "", 0.5), RooConstVar("c2", "", 0.1))
    result = RooProdPdf("prod", "prod", [g, mc]).fitTo(data, Save=True, PrintLevel=-1)
    assert _fitted(m, s) == pytest.approx(CONSTRAINED, rel=1e-6)
    assert result.minNll() == pytest.approx(102.17618960920294, abs=1e-8)


@pytest.mark.xfail(strict=True, reason="Extended() on a density without a yield says nothing")
def test_extended_on_a_density_that_expects_nothing_is_an_error_roofit_names(
    capsys: Any,
) -> None:
    """ROOT reports that the density cannot say how many events it expects."""
    g, data, _, _, _ = _gauss()
    capsys.readouterr()
    g.fitTo(data, PrintLevel=-1, Extended=True)
    assert (
        '[#0] ERROR:InputArguments -- The pdf "g" of type RooGaussian did not overload '
        "RooAbsPdf::createExpectedEventsFunc()!\n"
    ) in capsys.readouterr().out


def _extended_negative_width() -> tuple[Any, Any, Any, Any]:
    g, data, _, m, s = _gauss()
    generator().SetSeed(4357)
    s.setRange(-3, 10)
    s.setVal(0.5)
    data = g.generate([data.get().find("x")], 10)
    s.setVal(0.02)
    s.setError(1)
    n = RooRealVar("n", "n", 10, 0, 100)
    return RooExtendPdf("e", "e", g, n), data, m, s


def test_a_bad_point_of_an_extended_fit_is_logged_for_both_densities(capsys: Any) -> None:
    """A negative width makes the Gaussian's normalisation negative and the extended density's
    value NaN: both are logged, and the fit goes on to ROOT's minimum."""
    e, data, m, s = _extended_negative_width()
    capsys.readouterr()
    result = e.fitTo(data, Save=True, PrintLevel=-1)
    out = capsys.readouterr().out
    assert "Parameter values: \tm=0\tn=10\ts=-0.00253731\n" in out
    assert out.count("getLogVal() top-level p.d.f evaluates to NaN") == 10
    assert (
        "     p.d.f normalization integral is zero or negative @ numerator=g=1, "
        "denominator=g_Int[x]=-0.0063601\n"
    ) * 10 in out
    assert result.numInvalidNLL() == 1
    assert (m.getVal(), s.getVal()) == pytest.approx((0.0339328278528332, 0.2903711897080487))


@pytest.mark.xfail(strict=True, reason="a self-normalised top density is described by name only")
def test_a_bad_point_of_an_extended_fit_describes_the_extended_density_as_root_does(
    capsys: Any,
) -> None:
    """ROOT describes the extended density by its servers, and gives their values."""
    e, data, _, _ = _extended_negative_width()
    capsys.readouterr()
    e.fitTo(data, PrintLevel=-1)
    assert (
        "RooExtendPdf::e[ pdf=g_over_g_Int[x] n=n ]\n"
        "     getLogVal() top-level p.d.f evaluates to NaN @ pdf=g_over_g_Int[x]=nan, n=n=10\n"
    ) in capsys.readouterr().out


def test_minos_after_a_minimum_that_is_not_valid_gives_no_errors(capsys: Any) -> None:
    """MINOS refuses an invalid minimum, as MnMinos does: the parameters keep HESSE's errors."""
    g, data, _, m, s = _gauss()
    result = g.fitTo(data, Save=True, PrintLevel=-1, MaxCalls=5, Minos=True)
    assert not m.hasAsymError() and not s.hasAsymError()
    assert result.statusLabelHistory(0) == "MINIMIZE"
    assert result.statusCodeHistory(0) == -1


@pytest.mark.xfail(strict=True, reason="HESSE and MINOS run on as if the minimum were valid")
def test_hesse_and_minos_after_a_minimum_that_is_not_valid_fail_as_in_root(
    capsys: Any,
) -> None:
    """ROOT's HESSE fails at the invalid minimum (304), MINOS with it (-1), and the matrix is
    not calculated at all."""
    g, data, _, _, _ = _gauss()
    result = g.fitTo(data, Save=True, PrintLevel=-1, MaxCalls=5, Minos=True)
    codes = [result.statusCodeHistory(i) for i in range(result.numStatusHistory())]
    assert (codes, result.covQual()) == ([-1, 304, -1], 0)


def test_a_simultaneous_fit_in_a_range_restricts_only_the_real_observables() -> None:
    """``Range("sig")`` of a simultaneous fit names a fit range for x, not for the category:
    ROOT's fit and its range name."""
    sim, both, _, params = _simultaneous()
    both.get().find("x").setRange("sig", -3, 3)
    result = sim.fitTo(both, Save=True, PrintLevel=-1, Range="sig")
    expected = [-0.43098936677384614, 2.2234689275327986, 0.46333665670326774]
    assert [p.getVal() for p in params] == pytest.approx(expected, rel=1e-9)
    assert result.minNll() == pytest.approx(82.15972244588107, abs=1e-9)
    assert sim.getStringAttribute("fitrange") == "fit_nll_sim_d"


@pytest.mark.xfail(strict=True, reason="a sum of a likelihood has the data's x as a parameter")
def test_a_likelihood_can_be_minimised_as_a_term_of_a_sum() -> None:
    """A sum holding the likelihood has the likelihood's parameters - m and s, not the
    observable - takes its error level, one half, and is minimised to ROOT's fit."""
    from xrdroot.roofit.fitting.minimizer import RooMinimizer
    from xrdroot.roofit.functions import RooAddition

    g, data, _, m, s = _gauss()
    total = RooAddition("total", "total", [g.createNLL(data)])
    assert total.getVal() == 102.53261229765363
    minimizer = RooMinimizer(total)
    minimizer.setPrintLevel(-1)
    assert minimizer.migrad() == 0
    minimizer.hesse()
    assert _fitted(m, s) == pytest.approx(FULL, rel=1e-6)
    assert total.getParameters().names() == ["m", "s"]
