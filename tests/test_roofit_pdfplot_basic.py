"""RooFit's everyday densities - basic, shapes, generic, extended, summed and multivariate.

Every reference number was printed by ROOT 6.40 through PyROOT, ``%.12g``
for values and integrals and ``repr`` for generated events after
``RooRandom::randomGenerator()->SetSeed(4357)``, for the same models built
here with the engine's classes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xrdroot.roofit.functions import RooFormulaVar
from xrdroot.roofit.pdf import CAN_BE_EXTENDED, CAN_NOT_BE_EXTENDED
from xrdroot.roofit.pdfs.basic import (
    RooChebychev,
    RooExponential,
    RooGaussian,
    RooPolynomial,
    RooUniform,
    ref,
)
from xrdroot.roofit.pdfs.extend import RooExtendPdf
from xrdroot.roofit.pdfs.generic import RooGenericPdf
from xrdroot.roofit.pdfs.multivar import RooMultiVarGaussian
from xrdroot.roofit.pdfs.realsum import RooRealSumPdf
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooConstVar, RooRealVar

REL = 1e-11


def column(data: Any, name: str) -> list[float]:
    return [data.get(i).getRealValue(name) for i in range(data.numEntries())]


def integral(pdf: Any, over: list[Any], rng: str | None = None) -> float:
    made = pdf.createIntegral(over, Range=rng) if rng else pdf.createIntegral(over)
    return float(made.getVal())


def observable() -> RooRealVar:
    x = RooRealVar("x", "x", 0.5, -10, 10)
    x.setRange("win", -1.5, 2.5)
    return x


def gaussian() -> tuple[RooRealVar, RooRealVar, RooGaussian]:
    x = observable()
    m = RooRealVar("m", "m", 1, -5, 5)
    s = RooRealVar("s", "s", 2, 0.1, 10)
    return x, m, RooGaussian("g", "g", x, m, s)


def test_a_number_given_for_an_argument_becomes_a_constant_named_after_it() -> None:
    """``RooAbsReal::Ref`` names the constant it makes by the number, as ROOT prints it."""
    for number in (2, 2.5, np.float64(0.25), np.int64(7)):
        made = ref(number)
        assert isinstance(made, RooConstVar)
        assert made.getVal() == float(number)
    assert ref(2.5).GetName() == "2.5"
    x = observable()
    assert ref(x) is x


def test_a_gaussian_is_normalised_and_integrated_over_its_observable_or_its_mean() -> None:
    """The Gaussian integrates in closed form over ``x`` or over ``mean``, as ROOT's does."""
    x, m, g = gaussian()
    assert g.getVal([x]) == pytest.approx(0.193334718961, rel=REL)
    assert g.getVal([m]) == pytest.approx(0.196318937842, rel=REL)
    assert integral(g, [m]) == pytest.approx(4.93703381411, rel=REL)
    assert integral(g, [x], "win") == pytest.approx(3.34746607095, rel=REL)
    assert g.integral_code(frozenset(["x"])) == 1
    assert g.integral_code(frozenset(["m"])) == 2


def test_a_gaussian_of_a_formula_is_integrated_numerically_and_not_drawn_directly() -> None:
    """A Gaussian of ``2x`` has no closed form in ``x``: it is normalised numerically."""
    x = observable()
    xs = RooFormulaVar("xs", "xs", "2*x", [x])
    m, s = RooRealVar("m", "m", 1, -5, 5), RooRealVar("s", "s", 2, 0.1, 10)
    gx = RooGaussian("gx", "gx", xs, m, s)
    assert gx.analytic_names(frozenset(["x"]), None) == frozenset()
    assert gx.getVal([x]) == pytest.approx(0.398942264919, rel=1e-9)
    assert gx.generator_code(frozenset(["x"])) == 0


def test_a_gaussian_draws_its_observable_or_its_mean_as_root_does() -> None:
    """Direct generation takes ROOT's draws: the events are ROOT's to the last bit."""
    x, m, g = gaussian()
    generator().SetSeed(4357)
    assert column(g.generate([x], 4), "x") == [
        2.997865435218796,
        0.13047122117131948,
        2.5635925123910157,
        0.9398944573476911,
    ]
    assert column(g.generate([m], 3), "m") == [
        2.148527370025855,
        0.38656534904193673,
        -1.3017519892651546,
    ]


def test_an_exponential_integrates_over_its_observable_or_its_coefficient_negated_or_not() -> None:
    """``negateCoefficient`` flips the slope - and the range of ``c`` it is integrated over."""
    x = observable()
    c = RooRealVar("c", "c", -0.3, -2, -0.1)
    e = RooExponential("e", "e", x, c)
    assert e.getVal([x]) == pytest.approx(0.012887583185, rel=REL)
    assert integral(e, [c]) == pytest.approx(1.16669996666, rel=REL)
    en = RooExponential("en", "en", x, c, True)
    assert en.getVal([x]) == pytest.approx(0.0173964176706, rel=REL)
    assert integral(en, [c]) == pytest.approx(3.33402146417, rel=REL)
    assert integral(en, [x], "win") == pytest.approx(4.93123954997, rel=REL)
    assert en.analytic_names(frozenset(["s"]), None) == frozenset()


def test_a_polynomial_has_its_constant_term_implied_unless_the_lowest_order_is_zero() -> None:
    """``1 + a1 x + a2 x^2``, or ``a1 + a2 x`` from order zero; no coefficients is flat."""
    x = observable()
    a1, a2 = RooRealVar("a1", "a1", 0.2), RooRealVar("a2", "a2", 0.03)
    p = RooPolynomial("p", "p", x, [a1, a2])
    assert p.getVal([x]) == pytest.approx(0.0276875, rel=REL)
    assert integral(p, [x], "win") == pytest.approx(4.59, rel=REL)
    p0 = RooPolynomial("p0", "p0", x, [a1, a2], 0)
    assert p0.getVal() == pytest.approx(0.215, rel=REL)
    assert integral(p0, [x]) == pytest.approx(4.0, rel=REL)
    assert RooPolynomial("pe", "pe", x).getVal([x]) == pytest.approx(0.05, rel=REL)
    assert RooPolynomial("pz", "pz", x, [], 0).getVal() == 0.0
    assert RooPolynomial("pn", "pn", x, [], -3).getVal() == 0.0


def test_a_chebychev_maps_its_range_or_the_normalisation_range_it_was_given() -> None:
    """Chosen as a fit's range, ``[-1, 1]`` is mapped from that range, not the variable's."""
    x = observable()
    a1, a2 = RooRealVar("a1", "a1", 0.2), RooRealVar("a2", "a2", 0.03)
    ch = RooChebychev("ch", "ch", x, [a1, a2])
    assert ch.getVal([x]) == pytest.approx(0.0495025252525, rel=REL)
    assert integral(ch, [x], "win") == pytest.approx(3.9238, rel=REL)
    ch.selectNormalizationRange("win", True)
    ch.selectNormalizationRange("other")  # a range already chosen is kept unless forced
    assert ch.getVal() == pytest.approx(0.97, rel=REL)
    assert integral(ch, [x]) == pytest.approx(28.475, rel=REL)
    assert ch.getVal([x]) == pytest.approx(0.0489898989899, rel=REL)
    ch.selectNormalizationRange(None)
    assert ch.getVal([x]) == pytest.approx(0.0495025252525, rel=REL)


def test_a_uniform_density_is_one_over_the_volume_of_its_observables() -> None:
    """Flat in every observable, integrated over each as its range's length."""
    x = observable()
    y = RooRealVar("y", "y", 1, 0, 4)
    u = RooUniform("u", "u", [x, y])
    assert u.getVal([x, y]) == pytest.approx(0.0125, rel=REL)
    assert integral(u, [x]) == pytest.approx(20.0, rel=REL)
    assert integral(u, [x, y], "win") == pytest.approx(16.0, rel=REL)
    assert u.generator_code(frozenset()) == 0
    assert u.generator_code(frozenset(["x", "z"])) == 0


def test_a_uniform_density_draws_all_its_observables_as_root_does() -> None:
    """Each observable takes one draw, in the order of the list."""
    x = observable()
    y = RooRealVar("y", "y", 1, 0, 4)
    u = RooUniform("u", "u", [x, y])
    generator().SetSeed(4357)
    drawn = u.generate([x, y], 2)
    assert column(drawn, "x") + column(drawn, "y") == [
        9.99483497813344,
        -4.347643894143403,
        0.6516395015642047,
        3.788804328069091,
    ]
    assert u.generate_event(1, generator(), frozenset(["y"])).keys() == {"y"}


@pytest.mark.xfail(
    strict=True,
    reason="basic.py:190-202: RooUniform draws every observable, not only the ones asked for",
)
def test_a_uniform_density_draws_only_the_observables_asked_for() -> None:
    """ROOT's code is a bit per observable drawn: asked for ``y`` alone, ``x`` takes no draws."""
    x = observable()
    y = RooRealVar("y", "y", 1, 0, 4)
    u = RooUniform("u", "u", [x, y])
    generator().SetSeed(4357)
    u.generate([x, y], 2)
    assert column(u.generate([y], 2), "y") == [0.9266261709854007, 1.939894457347691]


def argus_mass() -> tuple[RooRealVar, RooRealVar]:
    return RooRealVar("mm", "mm", 5.25, 5.0, 5.3), RooRealVar("m0", "m0", 5.291)


def test_an_argus_is_normalised_in_closed_form_for_a_falling_or_flat_slope() -> None:
    """The closed forms for ``c < 0`` and ``c = 0``, and a numeric one for ``p`` not a half."""
    from xrdroot.roofit.pdfs.shapes import RooArgusBG

    mm, m0 = argus_mass()
    k = RooRealVar("k", "k", -20.0, -100, 100)
    argus = RooArgusBG("argus", "argus", mm, m0, k)
    assert argus.getVal() == pytest.approx(0.479029089697, rel=REL)
    assert argus.getVal([mm]) == pytest.approx(4.50247887825, rel=REL)
    flat = RooArgusBG("argus0", "argus0", mm, m0, RooRealVar("k0", "k0", 0.0))
    assert flat.getVal([mm]) == pytest.approx(1.99796346486, rel=REL)
    power = RooArgusBG("argusw", "argusw", mm, m0, k, RooRealVar("pw", "pw", 1.5))
    assert power.analytic_names(frozenset(["mm"]), None) == frozenset()
    assert power.getVal([mm]) == pytest.approx(1.45128859179, rel=1e-9)
    mm.setVal(5.295)
    assert argus.getVal() == 0.0


@pytest.mark.xfail(
    strict=True,
    reason="shapes.py:109: _dawson returns the integral of exp(t^2), not Dawson's function",
)
def test_an_argus_is_normalised_in_closed_form_for_a_rising_slope() -> None:
    """For ``c > 0`` ROOT's closed form takes Dawson's function, ``exp(-x^2) int exp(t^2)``."""
    from xrdroot.roofit.pdfs.shapes import RooArgusBG

    mm, m0 = argus_mass()
    rising = RooArgusBG("argusp", "argusp", mm, m0, RooRealVar("kp", "kp", 3.0))
    assert rising.getVal([mm]) == pytest.approx(1.72012151487, rel=1e-6)


def test_a_shape_refuses_a_member_it_does_not_have() -> None:
    """Inputs are attributes by name; any other name is an error that says so."""
    from xrdroot.roofit.pdfs.shapes import RooBreitWigner

    x = observable()
    bw = RooBreitWigner("bw", "bw", x, 1.0, 1.7)
    with pytest.raises(AttributeError, match="RooBreitWigner has no input or member called 'nope'"):
        bw.nope  # noqa: B018


def crystal_ball_mass() -> RooRealVar:
    cbm = RooRealVar("cbm", "cbm", 0.2, -10, 10)
    cbm.setRange("tail", -8, -1)
    return cbm


def test_a_crystal_ball_has_its_tail_left_of_the_peak_or_right_of_it_for_a_negative_alpha() -> None:
    """The power law takes over ``alpha`` widths from the peak, on the side of ``alpha``'s sign."""
    from xrdroot.roofit.pdfs.shapes import RooCBShape

    cbm = crystal_ball_mass()
    args = (RooRealVar("cbm0", "cbm0", 0.1), RooRealVar("cbs", "cbs", 1.3))
    n = RooRealVar("cbn", "cbn", 3.0)
    cb = RooCBShape("cb", "cb", cbm, *args, RooRealVar("cba", "cba", 1.1), n)
    assert cb.getVal([cbm]) == pytest.approx(0.269251694003, rel=REL)
    assert integral(cb, [cbm], "tail") == pytest.approx(1.05694716705, rel=REL)
    cbm.setVal(-4.0)
    assert cb.getVal() == pytest.approx(0.101355873271, rel=REL)
    right = RooCBShape("cbr", "cbr", cbm, *args, RooRealVar("cban", "cban", -1.1), n)
    cbm.setVal(4.0)
    assert right.getVal() == pytest.approx(0.111805265074, rel=REL)
    assert right.getVal([cbm]) == pytest.approx(0.0302152893917, rel=REL)


def test_a_bifurcated_gaussian_takes_one_width_left_of_its_mean_and_another_right() -> None:
    """``sigmaL`` below the mean, ``sigmaR`` above, integrated in closed form either side."""
    from xrdroot.roofit.pdfs.shapes import RooBifurGauss

    x = observable()
    m = RooRealVar("m", "m", 1, -5, 5)
    bg = RooBifurGauss("bg", "bg", x, m, RooRealVar("sl", "sl", 1.0), RooRealVar("sr", "sr", 2.5))
    assert bg.getVal([x]) == pytest.approx(0.20122592497, rel=REL)
    x.setVal(3.0)
    assert bg.getVal([x]) == pytest.approx(0.165575665165, rel=REL)
    assert integral(bg, [x], "win") == pytest.approx(2.65240760953, rel=REL)


def test_a_breit_wigner_integrates_as_an_arctangent() -> None:
    """The non-relativistic Breit-Wigner, normalised and over a window."""
    from xrdroot.roofit.pdfs.shapes import RooBreitWigner

    x = observable()
    x.setVal(3.0)
    bw = RooBreitWigner("bw", "bw", x, RooRealVar("m", "m", 1, -5, 5), RooRealVar("w", "w", 1.7))
    assert bw.getVal([x]) == pytest.approx(0.0605962107366, rel=REL)
    assert integral(bw, [x], "win") == pytest.approx(2.70388839924, rel=REL)


def test_a_landau_is_normalised_by_its_cumulative_and_zero_for_a_width_not_positive() -> None:
    """ROOT's Landau, its integral the Landau CDF; a width at or below zero makes it vanish."""
    from xrdroot.roofit.pdfs.shapes import RooLandau

    x = observable()
    x.setVal(3.0)
    m = RooRealVar("m", "m", 1, -5, 5)
    lan = RooLandau("lan", "lan", x, m, RooRealVar("ls", "ls", 0.8, 0.1, 5))
    assert lan.getVal([x]) == pytest.approx(0.123037633542, rel=REL)
    assert integral(lan, [x], "win") == pytest.approx(0.449678210745, rel=REL)
    assert RooLandau("bad", "bad", x, m, RooRealVar("b", "b", -0.5)).getVal() == 0.0


def test_a_landau_draws_its_observable_as_root_does() -> None:
    """``TRandom::Landau``, drawn again until inside the range."""
    from xrdroot.roofit.pdfs.shapes import RooLandau

    x = observable()
    lan = RooLandau("lan", "lan", x, RooRealVar("m", "m", 1, -5, 5), RooRealVar("ls", "ls", 0.8))
    generator().SetSeed(4357)
    assert column(lan.generate([x], 3), "x") == [
        0.4437439748363762,
        0.9811650828271925,
        0.7550220236228093,
    ]
    assert lan.generator_code(frozenset()) == 0


def test_a_lognormal_is_normalised_and_integrated_by_the_error_function() -> None:
    """The log-normal of median ``m0`` and shape ``k``."""
    from xrdroot.roofit.pdfs.shapes import RooLognormal

    xp = RooRealVar("xp", "xp", 2.0, 0.5, 20)
    xp.setRange("low", 1.0, 3.0)
    median, shape = RooRealVar("lm0", "lm0", 3.0), RooRealVar("lk", "lk", 1.8)
    logn = RooLognormal("logn", "logn", xp, median, shape)
    assert logn.getVal([xp]) == pytest.approx(0.267980344395, rel=REL)
    assert integral(logn, [xp], "low") == pytest.approx(0.469193209503, rel=REL)


def poisson() -> tuple[RooRealVar, RooRealVar, Any]:
    from xrdroot.roofit.pdfs.shapes import RooPoisson

    n = RooRealVar("n", "n", 3.7, 0, 50)
    for name, low, high in (("r0", 0, 4.5), ("rmid", 2, 6), ("rhigh", 9, 14)):
        n.setRange(name, low, high)
    mu = RooRealVar("mu", "mu", 4.2, 0, 30)
    mu.setRange("mr", 1, 8)
    return n, mu, RooPoisson("po", "po", n, mu)


def test_a_poisson_rounds_its_count_down_and_sums_its_terms_by_incomplete_gammas() -> None:
    """Over counts from zero, from below the mean and from above it, each is ROOT's gamma form."""
    n, _, po = poisson()
    assert po.getVal() == pytest.approx(0.185165382579, rel=REL)
    assert po.getVal([n]) == pytest.approx(0.185165382579, rel=REL)
    assert integral(po, [n], "r0") == pytest.approx(0.589827021311, rel=REL)
    assert integral(po, [n], "rmid") == pytest.approx(0.789486996483, rel=REL)
    assert integral(po, [n], "rhigh") == pytest.approx(0.027897718224, rel=REL)
    n.setVal(0.0)
    assert po.getVal() == pytest.approx(0.0149955768205, rel=REL)


def test_a_poisson_integrates_over_its_mean_rounded_or_not() -> None:
    """Over the mean, the integral is a difference of incomplete gammas of the count plus one."""
    from xrdroot.roofit.pdfs.shapes import RooPoisson

    n, mu, po = poisson()
    assert integral(po, [mu]) == pytest.approx(0.999999999534, rel=REL)
    assert integral(po, [mu], "mr") == pytest.approx(0.938631731132, rel=REL)
    exact = RooPoisson("pr", "pr", n, mu, True)
    assert exact.getVal() == pytest.approx(0.196598362039, rel=REL)
    assert integral(exact, [mu]) == pytest.approx(0.99999999799, rel=REL)


def test_a_poisson_sums_to_nothing_below_zero_and_to_one_far_round_a_large_mean() -> None:
    """Counts all below zero hold nothing; a range a hundred widths round the mean holds all."""
    from xrdroot.roofit.pdfs.shapes import RooPoisson

    n = RooRealVar("n", "n", 3.7, -10, 1000000)
    n.setRange("neg", -5, -2)
    mu = RooRealVar("mu", "mu", 20000, -10, 30000)
    po = RooPoisson("po", "po", n, mu)
    assert integral(po, [n], "neg") == 0.0
    assert integral(po, [n]) == 1.0


def test_a_poisson_draws_whole_counts_as_root_does() -> None:
    """``TRandom::Poisson`` of the mean, drawn again until inside the range."""
    n, _, po = poisson()
    generator().SetSeed(4357)
    assert column(po.generate([n], 4), "n") == [4.0, 9.0, 3.0, 1.0]
    assert po.generator_code(frozenset(["mu"])) == 0


@pytest.mark.xfail(
    strict=True,
    reason="shapes.py:245-254: ROOT's getVal() is 0 for a negative mean, protected or not",
)
def test_a_poisson_of_a_negative_mean_is_zero_protected_or_not() -> None:
    """ROOT's ``getVal`` of a Poisson with a negative mean is 0 either way."""
    from xrdroot.roofit.pdfs.shapes import RooPoisson

    n = RooRealVar("n", "n", 3.7, 0, 100)
    mu = RooRealVar("mu", "mu", -2.0, -10, 30)
    protected = RooPoisson("po", "po", n, mu)
    protected.protectNegativeMean(True)
    unprotected = RooPoisson("po2", "po2", n, mu)
    unprotected.protectNegativeMean(False)
    assert (protected.getVal(), unprotected.getVal()) == (0.0, 0.0)


def test_a_generic_density_is_its_formula_normalised_numerically() -> None:
    """No closed form: the formula is integrated numerically over the observable."""
    x = observable()
    tau = RooRealVar("tau", "tau", 2.0, 0.5, 5)
    gen = RooGenericPdf("gen", "gen", "exp(-abs(x)/tau)", [x, tau])
    assert gen.getVal([x]) == pytest.approx(0.196020974676, rel=1e-9)
    assert integral(gen, [x], "win") == pytest.approx(2.48225712878, rel=1e-9)
    assert gen.expression() == "exp(-abs(x)/tau)"
    titled = RooGenericPdf("t", "x*x", [x])
    assert titled.expression() == "x*x"


@pytest.mark.xfail(
    strict=True,
    reason="generic.py:34: ROOT prints the formula with its variables as x[0], x[1], ...",
)
def test_a_generic_density_prints_its_formula_as_root_does(capsys: Any) -> None:
    """``Print`` shows the formula as ROOT's ``RooFormula`` holds it, its variables numbered."""
    x = observable()
    tau = RooRealVar("tau", "tau", 2.0, 0.5, 5)
    RooGenericPdf("gen", "gen", "exp(-abs(x)/tau)", [x, tau]).Print()
    assert capsys.readouterr().out.endswith(
        'RooGenericPdf::gen[ actualVars=(x,tau) formula="exp(-abs(x[0])/x[1])" ] = 0.778801\n'
    )


def test_an_extended_density_is_its_shape_and_expects_its_yield() -> None:
    """``n`` events in all - or ``n`` in a range, the whole scaled by the fraction there."""
    x, _, g = gaussian()
    nsig = RooRealVar("nsig", "nsig", 200, 0, 1000)
    ext = RooExtendPdf("ext", "ext", g, nsig)
    assert ext.getVal([x]) == pytest.approx(0.193334718961, rel=REL)
    assert ext.expectedEvents([x]) == 200.0
    assert ext.extendMode() == CAN_BE_EXTENDED
    ranged = RooExtendPdf("extr", "extr", g, nsig, "win")
    assert ranged.expectedEvents([x]) == pytest.approx(299.524435163, rel=REL)
    assert integral(ranged, [x], "win") == pytest.approx(3.34746607095, rel=REL)
    assert ranged.expected([x], "win") == pytest.approx(200.0, rel=REL)
    assert ranged.normalized_name([x]) == "extr"


def test_an_extended_density_of_an_extended_density_multiplies_their_yields() -> None:
    """Wrapping an extended density scales what it expects by the outer yield."""
    x, _, g = gaussian()
    inner = RooExtendPdf("inner", "inner", g, RooRealVar("ni", "ni", 50))
    outer = RooExtendPdf("outer", "outer", inner, RooRealVar("fo", "fo", 0.5))
    assert outer.expectedEvents([x]) == 25.0
    assert outer.getVal([x]) == pytest.approx(0.193334718961, rel=REL)


def realsum() -> tuple[RooRealVar, RooRealSumPdf, RooRealSumPdf]:
    x = observable()
    f1 = RooFormulaVar("f1", "f1", "1+0.1*x", [x])
    f2 = RooFormulaVar("f2", "f2", "x*x", [x])
    c1, c2 = RooRealVar("c1", "c1", 0.3), RooRealVar("c2", "c2", 0.2)
    last = RooRealSumPdf("rs", "rs", [f1, f2], [c1])
    return x, last, RooRealSumPdf("rs2", "rs2", [f1, f2], [c1, c2], True)


def test_a_sum_of_functions_takes_the_rest_for_its_last_term_and_is_normalised_whole() -> None:
    """One coefficient fewer than functions: the last takes ``1 - sum``."""
    x, last, both = realsum()
    assert last.getVal() == pytest.approx(0.49, rel=REL)
    assert last.getVal([x]) == pytest.approx(0.00103667136812, rel=REL)
    assert integral(last, [x], "win") == pytest.approx(5.69333333333, rel=REL)
    assert both.getVal([x]) == pytest.approx(0.00261961722488, rel=REL)
    assert both.expectedEvents([x]) == pytest.approx(139.333333333, rel=REL)
    assert (last.extendMode(), both.extendMode()) == (CAN_NOT_BE_EXTENDED, CAN_BE_EXTENDED)


def test_a_sum_of_functions_prints_its_terms_as_root_does(capsys: Any) -> None:
    """``[%]`` stands for the coefficient that is the rest."""
    _, last, both = realsum()
    last.Print()
    both.Print()
    assert capsys.readouterr().out.splitlines()[-2:] == [
        "RooRealSumPdf::rs[ c1 * f1 + [%] * f2 ] = 0.49",
        "RooRealSumPdf::rs2[ c1 * f1 + c2 * f2 ] = 0.365",
    ]


@pytest.mark.xfail(
    strict=True,
    reason="realsum.py:50: a term not depending on the observables is not times their volume",
)
def test_a_sum_of_functions_integrates_a_term_free_of_the_observable_over_its_range() -> None:
    """A term that does not depend on ``x`` still integrates to itself times the range of ``x``."""
    x = observable()
    m = RooRealVar("m", "m", 1, -5, 5)
    k = RooFormulaVar("k", "k", "2.0+0*m", [m])
    f2 = RooFormulaVar("f2", "f2", "x*x", [x])
    summed = RooRealSumPdf("rc", "rc", [k, f2], [RooRealVar("a", "a", 0.25)])
    assert integral(summed, [x]) == pytest.approx(510.0, rel=REL)
    assert summed.getVal([x]) == pytest.approx(0.00134803921569, rel=REL)


def multivariate() -> tuple[list[RooRealVar], RooMultiVarGaussian]:
    xs = [
        RooRealVar("x", "x", 0.5, -10, 10),
        RooRealVar("y", "y", -0.3, -10, 10),
        RooRealVar("z", "z", 1.2, -10, 10),
    ]
    mus = [RooRealVar("mx", "mx", 0.2), RooRealVar("my", "my", -0.1), RooRealVar("mz", "mz", 1.0)]
    cov = [[2.0, 0.6, 0.3], [0.6, 1.5, -0.4], [0.3, -0.4, 1.0]]
    return xs, RooMultiVarGaussian("mvg", "mvg", xs, mus, cov)


def test_a_multivariate_gaussian_is_normalised_over_all_its_variables_in_closed_form() -> None:
    """``sqrt((2 pi)^n det V)`` - over the whole real line, as RooFit integrates it."""
    xs, mvg = multivariate()
    assert mvg.getVal() == pytest.approx(0.945722609592, rel=REL)
    assert mvg.getVal(xs) == pytest.approx(0.0420312650533, rel=1e-7)
    assert integral(mvg, xs) == pytest.approx(22.5004555155, rel=1e-7)
    assert mvg.analytic_names(frozenset(["x", "y"]), None) == frozenset()


def test_a_multivariate_gaussian_draws_its_events_by_the_choleskian_as_root_does() -> None:
    """A standard normal draw per variable, times ``TDecompChol``'s factor, plus the means."""
    xs, mvg = multivariate()
    generator().SetSeed(4357)
    drawn = mvg.generate(xs, 3)
    assert column(drawn, "x") + column(drawn, "y") + column(drawn, "z") == pytest.approx(
        [
            1.6127041971414238,
            0.15749896320365508,
            -1.0740310496257426,
            -0.1756949951524794,
            0.8342565641257067,
            -0.56803821300183,
            2.0847340999348987,
            0.5922148827309964,
            0.8477129935261294,
        ],
        rel=1e-14,
    )
    assert mvg.generator_code(frozenset(["x"])) == 0


def test_a_multivariate_gaussian_draws_again_until_its_event_is_inside_the_ranges() -> None:
    """A width of two on ``[-1, 1]``: events outside are drawn again, as ROOT's are."""
    t = RooRealVar("t", "t", 0.1, -1, 1)
    one = RooMultiVarGaussian("one", "one", [t], [RooRealVar("mt", "mt", 0.0)], [[4.0]])
    generator().SetSeed(4357)
    assert column(one.generate([t], 3), "t") == [
        -0.8695287788286805,
        -0.06010554265230894,
        -0.11343465095806327,
    ]


def test_a_poisson_draws_again_a_count_outside_the_range() -> None:
    """A mean of 4.2 on ``[0, 3]``: counts above three are drawn again, as ROOT draws them."""
    from xrdroot.roofit.pdfs.shapes import RooPoisson

    n = RooRealVar("n", "n", 1, 0, 3)
    po = RooPoisson("po", "po", n, RooRealVar("mu", "mu", 4.2))
    generator().SetSeed(4357)
    assert column(po.generate([n], 6), "n") == [3.0, 1.0, 3.0, 2.0, 2.0, 2.0]
