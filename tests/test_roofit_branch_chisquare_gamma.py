"""``RooChiSquarePdf``, ``RooNonCentralChiSquare`` and ``RooGamma`` against ROOT 6.40.

Each number was printed by ROOT through PyROOT for the same variables and
the same five events: the values plain and normalised, the likelihoods
through RooFit's kernels, the forced Poisson sum and its warning when it
stops early, and the gamma draws after ``SetSeed(4357)``.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from xrdroot.roofit.collections import RooArgList, RooArgSet
from xrdroot.roofit.data.dataset import RooDataSet
from xrdroot.roofit.functions import RooFormulaVar
from xrdroot.roofit.pdfs.chisquare import RooChiSquarePdf, RooNonCentralChiSquare, bessel_i
from xrdroot.roofit.pdfs.gamma import RooGamma, gamma_dist
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooRealVar

EVENTS = (0.5, 1.5, 3.0, 7.0, 12.0)


def _data(x: Any) -> RooDataSet:
    """The five events every likelihood here is of."""
    data = RooDataSet("d", "d", RooArgSet(x))
    for value in EVENTS:
        x.setVal(value)
        data.add(RooArgSet(x))
    return data


def _chi() -> tuple[Any, Any, Any]:
    x = RooRealVar("x", "x", 0.5, 0, 20)
    ndof = RooRealVar("ndof", "ndof", 3, 1, 10)
    return RooChiSquarePdf("chi", "chi", x, ndof), x, ndof


def test_the_chisquare_density_is_roots_plain_and_normalised() -> None:
    """Its value, over ``[0, 20]`` normalised, and nothing at zero."""
    chi, x, _ = _chi()
    xs = RooArgSet(x)
    assert (chi.getVal(), chi.getVal(xs)) == pytest.approx(
        (0.21969564473386122, 0.21973294273873287), rel=1e-14)  # fmt: skip
    x.setVal(0.0)
    assert (chi.getVal(), chi.getVal(xs)) == (0.0, 0.0)


def test_the_chisquare_density_over_a_named_range_is_integrated_numerically() -> None:
    """A range has no closed form here: RooFit's numeric integral gives ROOT's value."""
    chi, x, _ = _chi()
    x.setRange("r", 1, 4)
    integral = chi.createIntegral(RooArgSet(x), Range="r")
    assert integral.getVal() == pytest.approx(0.5397877916206814, rel=1e-13)


def test_the_chisquare_likelihood_takes_the_kernels_digits() -> None:
    """VDT's ``fast_log`` and ``fast_exp`` in the likelihood: ROOT's value to the last bit."""
    chi, x, ndof = _chi()
    nll = chi.createNLL(_data(x))
    assert nll.getVal() == 13.972970374276391
    ndof.setVal(4.5)
    assert nll.getVal() == 13.866060381602283


def _noncentral() -> tuple[Any, Any, Any]:
    x = RooRealVar("x", "x", 2.5, 0, 20)
    k = RooRealVar("k", "k", 3, 1, 10)
    lam = RooRealVar("lam", "lam", 2, 0, 40)
    return RooNonCentralChiSquare("nc", "nc", x, k, lam), x, lam


def test_the_non_central_density_is_the_bessel_expression_its_integral_the_sum() -> None:
    """MathMore's expression by default; at the range's end, a thousandth of it inside."""
    nc, x, _ = _noncentral()
    assert (nc.getVal(), nc.getVal(RooArgSet(x))) == pytest.approx(
        (0.13750706795573614, 0.13803199201175123), rel=1e-13)  # fmt: skip
    x.setVal(0.0)
    assert nc.getVal() == pytest.approx(0.0206861220558898, rel=1e-13)


def test_the_forced_sum_says_so_once_and_matches_roots(capsys: Any) -> None:
    """``SetForceSum(True)``: RooFit's Poisson-weighted sum, announced the first time."""
    nc, x, lam = _noncentral()
    nc.SetForceSum(True)
    capsys.readouterr()
    assert (nc.getVal(), nc.getVal(RooArgSet(x))) == (0.13750689708616642, 0.13803182048989834)
    assert capsys.readouterr().out == (
        "[#1] INFO:InputArguments -- RooNonCentralChiSquare sum being forced\n")
    lam.setVal(0.0)
    assert (nc.getVal(), nc.getVal(RooArgSet(x))) == pytest.approx(
        (0.18072239266818127, 0.18075307413521863), rel=1e-14)  # fmt: skip


def test_a_sum_cut_short_warns_once(capfd: Any) -> None:
    """Too few terms for the tolerance: the warning, once, and the sum so far - ROOT's."""
    nc, x, lam = _noncentral()
    nc.SetForceSum(True)
    lam.setVal(30.0)
    nc.SetMaxIters(1)
    nc.SetErrorTolerance(1e-12)
    capfd.readouterr()
    assert nc.getVal() == pytest.approx(1.8413427633628545e-05, rel=1e-14)
    assert nc.getVal(RooArgSet(x)) == pytest.approx(0.0001630576842307264, rel=1e-14)
    out = capfd.readouterr()
    text = out.out + out.err
    assert text.count("did not converge") == 1
    assert (
        "[#0] WARNING:Eval -- RooNonCentralChiSquare did not converge: for x=2.5 k=3, lambda=30 "
        "fractional error = 0.004162\n either adjust tolerance with SetErrorTolerance(tol) or "
        "max_iter with SetMaxIter(max_it)\n") in text  # fmt: skip


def test_the_bessel_series_at_zero_and_below_the_range() -> None:
    """``I_nu(0)`` is one for order zero, else zero; below zero the density is nothing."""
    assert (bessel_i(0.0, 0.0), bessel_i(0.5, 0.0)) == (1.0, 0.0)
    x = RooRealVar("x", "x", -0.5, -1, 10)
    nc = RooNonCentralChiSquare("nc", "nc", x, RooRealVar("k", "k", 3), RooRealVar("l", "l", 2))
    assert nc.getVal() == 0.0
    assert math.isfinite(nc.getVal(RooArgSet(x)))


def _gamma() -> tuple[Any, Any, Any]:
    x = RooRealVar("x", "x", 2.0, 0, 20)
    shape = RooRealVar("gm", "gm", 2.5, 0.1, 10)
    beta = RooRealVar("beta", "beta", 1.5, 0.1, 10)
    mu = RooRealVar("mu", "mu", 0.5, 0, 5)
    return RooGamma("pdf", "pdf", x, shape, beta, mu), x, shape


def test_the_gamma_density_is_roots_and_nothing_below_its_origin() -> None:
    """``TMath::GammaDist``, normalised by the distribution function; zero below ``mu``."""
    pdf, x, _ = _gamma()
    assert (pdf.getVal(), pdf.getVal(RooArgSet(x))) == pytest.approx(
        (0.18449222107581986, 0.18450868588984093), rel=1e-14)  # fmt: skip
    x.setVal(0.2)
    assert pdf.getVal() == 0.0
    assert (gamma_dist(1.0, 0.0, 0.0, 1.0), gamma_dist(1.0, 1.0, 0.0, -1.0)) == (0.0, 0.0)


def test_the_gamma_likelihood_is_the_kernels_even_at_its_origin() -> None:
    """An event at ``mu`` is a zero for a shape above one - a NaN likelihood, as ROOT's - and
    ``1/beta`` for a shape of one."""
    pdf, x, shape = _gamma()
    nll = pdf.createNLL(_data(x))
    assert math.isnan(nll.getVal())
    shape.setVal(1.0)
    assert nll.getVal() == 16.360647572214347


def test_gamma_draws_are_roots_for_every_shape_and_a_narrowed_range() -> None:
    """Marsaglia and Tsang's draws after ROOT's seed, a shape below one through the one above
    it, and draws outside a narrowed range drawn again."""
    pdf, x, shape = _gamma()
    xs = RooArgSet(x)
    generator().SetSeed(4357)
    shape.setVal(1.0)

    def drawn() -> list[float]:
        data = pdf.generate(xs, 4)
        return [data.get(i).getRealValue("x") for i in range(4)]

    assert drawn() == [3.290194765984295, 3.5572013046407824, 0.7526983789141911,
                       1.5097217714324418]  # fmt: skip
    shape.setVal(0.6)
    assert drawn() == [0.40125685631686464, 0.03596154252327686, 0.06538733677818556,
                       0.26118504245666035]  # fmt: skip
    shape.setVal(3.0)
    x.setRange(1.0, 3.0)
    assert drawn() == [2.7180985515905336, 2.2692447659661905, 2.7240099517778456,
                       2.1105477465598748]  # fmt: skip


def test_gamma_draws_by_the_logarithm_test_and_a_small_shape_redrawn_into_range() -> None:
    """Forty draws of shape one reach Marsaglia and Tsang's second test; a shape below one in
    a narrow range draws again until inside it - ROOT's numbers after its seed."""
    pdf, x, shape = _gamma()
    xs = RooArgSet(x)
    generator().SetSeed(4357)
    shape.setVal(1.0)
    data = pdf.generate(xs, 40)
    # fsum: from Python 3.12 on, sum() compensates its rounding, and before it does not
    assert math.fsum(data.get(i).getRealValue("x") for i in range(40)) == 93.53373544573735
    shape.setVal(0.6)
    x.setRange(1.0, 1.5)
    data = pdf.generate(xs, 3)
    assert [data.get(i).getRealValue("x") for i in range(3)] == [
        1.2205255112588436, 1.3178891001583537, 1.0233723817294365]  # fmt: skip


def test_densities_of_a_formula_are_integrated_numerically() -> None:
    """With a formula for ``x`` there is no closed form: RooFit integrates over its variable."""
    y = RooRealVar("y", "y", 2.0, 0, 20)
    fx = RooFormulaVar("fx", "y*1", RooArgList(y))
    beta = RooRealVar("beta", "beta", 1.5)
    gamma = RooGamma("g2", "g2", fx, RooRealVar("two", "two", 2), beta, RooRealVar("o", "o", 0))
    chi = RooChiSquarePdf("c2", "c2", fx, RooRealVar("three", "three", 3))
    ys = RooArgSet(y)
    assert gamma.getVal(ys) == pytest.approx(0.23431401135232607, rel=1e-13)
    assert chi.getVal(ys) == pytest.approx(0.2075909282121211, rel=1e-13)
    nc, _, _ = _noncentral()
    assert nc.analytic_names(frozenset(["k"]), None) == frozenset()
