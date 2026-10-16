"""``RooProduct``'s integral, taken factor by factor as ``RooProduct::getPartIntList`` takes it.

The values and lines are ROOT 6.40.04's, through PyROOT, for ``k = 3``, a
polynomial ``p(c) = 0.5 + 0.2c + 0.2c^2`` over ``c`` in [-1, 1] and
``f(t) = exp(-t/1.5)`` over ``t`` in [-1, 15]: ``k p f`` over both is the
closed form of ``p`` times ROOT's one-dimensional Romberg integral of ``f``,
the same to the last bit; a product whose factors all share the variables
is integrated whole, by the adaptive cubature.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from refmachine import roots
from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.functions import RooFormulaVar, RooPolyVar, RooProduct
from xrdroot.roofit.messages import service
from xrdroot.roofit.partint import groups
from xrdroot.roofit.pdfs.basic import ref
from xrdroot.roofit.pdfs.realsum import RooRealSumPdf
from xrdroot.roofit.selection import selecting
from xrdroot.roofit.variables import RooRealVar

LINE = "[#1] INFO:NumericIntegration -- RooRealIntegral::init({}) using numeric integrator {}"


@pytest.fixture(autouse=True)
def _messages() -> Iterator[None]:
    """A preceding test's progress dots must not change an integral's announcement."""
    service().reset()
    yield
    service().reset()


def model() -> tuple[RooRealVar, RooRealVar, RooRealVar, RooPolyVar, RooFormulaVar]:
    t = RooRealVar("t", "t", -1.0, 15.0)
    c = RooRealVar("c", "c", -1.0, 1.0)
    k = RooRealVar("k", "k", 3.0)
    p = RooPolyVar("p", "p", c, [ref(0.5), ref(0.2), ref(0.2)], 0)
    f = RooFormulaVar("f", "exp(-@0/@1)", [t, RooRealVar("tau", "tau", 1.5)])
    return t, c, k, p, f


def test_a_product_of_factors_in_separate_variables_is_their_integrals_multiplied(
    capsys: Any,
) -> None:
    """ROOT's 9.9332120726109636: the polynomial's closed form times ``f``'s Romberg integral,
    which says it is numerical, under its own name."""
    t, c, k, p, f = model()
    product = RooProduct("prod", "prod", [k, p, f])
    capsys.readouterr()
    both = product.createIntegral(RooArgSet(t, c))
    assert capsys.readouterr().out == LINE.format("f_Int[t]", "RooIntegrator1D") + (
        " to calculate Int(t)\n"
    )
    assert both.getVal() == roots(9.9332120726109636, rel=1e-14)
    t.setVal(2.0)
    assert product.createIntegral(RooArgSet(c)).getVal() == roots(0.89623026959347107, rel=1e-15)


def test_a_product_whose_factors_all_share_variables_is_integrated_whole(capsys: Any) -> None:
    """``p(c) * t*c * f(t)`` makes one group of all three, which ROOT integrates as the
    product it is: 0.19462358433794164, by RooAdaptiveIntegratorND."""
    t, c, _, p, f = model()
    g = RooFormulaVar("g", "@0*@1", [t, c])
    product = RooProduct("prod2", "prod2", [p, g, f])
    capsys.readouterr()
    whole = product.createIntegral(RooArgSet(t, c))
    assert capsys.readouterr().out.startswith(LINE.format("prod2_Int[c,t]", "RooAdaptive"))
    assert whole.getVal() == roots(0.19462358433794164, rel=1e-12)


def test_factors_that_share_a_variable_are_integrated_together_as_a_subproduct(
    capsys: Any,
) -> None:
    """``k * t*c * f(t)``: ``k`` stays outside, ``g`` and ``f`` are ``SUBPROD_g_X_f``."""
    t, c, k, _, f = model()
    g = RooFormulaVar("g", "@0*@1", [t, c])
    product = RooProduct("prod3", "prod3", [k, g, f])
    capsys.readouterr()
    assert product.createIntegral(RooArgSet(t, c)).getVal() == pytest.approx(0.0, abs=1e-12)
    assert "SUBPROD_g_X_f_Int[c,t]" in capsys.readouterr().out


def test_groups_join_every_group_a_variable_ties_together() -> None:
    """Two groups a third variable's factors both reach become one, where the first was."""
    t, c, k, p, f = model()
    u = RooRealVar("u", "u", 0.0, 2.0)
    tc = RooFormulaVar("tc", "@0*@1", [t, c])
    pu = RooFormulaVar("pu", "@0+@1", [u, c])
    found = groups([k, f, p, pu, tc], frozenset({"t", "u", "c"}), ["t", "u", "c"])
    assert [([one.GetName() for one in group], sorted(over)) for group, over in found] == [
        (["k"], []),
        (["f", "tc", "pu", "p"], ["c", "t", "u"]),
    ]


def test_a_sum_of_products_announces_each_factors_integral_and_only_the_drawn_ones() -> None:
    """rf704: a ``RooRealSumPdf`` of amplitudes says each amplitude's numerical part as its
    integral is made - and, inside a plot of some components, only theirs."""
    t, _, _, p, f = model()
    a1 = RooProduct("a1", "a1", [p, f])
    a2 = RooProduct("a2", "a2", [p, RooFormulaVar("h", "@0*@0", [t])])
    pdf = RooRealSumPdf("pdf", "pdf", [a1, a2], [RooRealVar("f1", "f1", 1.0)])
    seen: list[str] = []
    a1.announce_inner = lambda names, rng: seen.append(f"a1 {sorted(names)}")  # type: ignore[method-assign]
    a2.announce_inner = lambda names, rng: seen.append(f"a2 {sorted(names)}")  # type: ignore[method-assign]
    pdf.announce_inner(frozenset({"t", "c"}), None)
    with selecting({"a1", "p", "f"}):
        pdf.announce_inner(frozenset({"t"}), None)
    assert seen == ["a1 ['c', 't']", "a2 ['c', 't']", "a1 ['t']"]


def test_a_functions_histogram_is_its_values_times_each_bins_volume_unless_told_not() -> None:
    """``createHistogram`` of ``k p f`` on 2 x 2 bins of volume 8: ROOT's TH2F held
    1.4616210460662842 and, with ``Scaling(false)``, 0.18270263075828552 - single precision
    of these, a ``TH2F`` keeping floats."""
    from xrdroot.roofit.cmdargs import RooCmdArg

    t, c, k, p, f = model()
    product = RooProduct("prod", "prod", [k, p, f])
    axes = (RooCmdArg("Binning", 2), RooCmdArg("YVar", c, RooCmdArg("Binning", 2)))
    scaled: Any = product.createHistogram("h", t, *axes)
    plain: Any = product.createHistogram("h2", t, *axes, RooCmdArg("Scaling", False))
    assert float(scaled.values()[0][0]) == pytest.approx(1.4616210460662842, rel=1e-7)
    assert float(plain.values()[0][0]) == pytest.approx(0.18270263075828552, rel=1e-7)
    assert (scaled.values() == plain.values() * 8.0).all()


def test_a_cdf_of_two_variables_is_the_fraction_below_both() -> None:
    """``gxy.createCdf({x, y})`` of two Gaussians: ROOT's 0.19458554441368411 at (1, -1)."""
    from xrdroot.roofit.pdfs.basic import RooGaussian
    from xrdroot.roofit.pdfs.prodpdf import RooProdPdf

    x = RooRealVar("x", "x", -10, 10)
    y = RooRealVar("y", "y", -10, 10)
    gx = RooGaussian("gx", "gx", x, ref(0.0), ref(3.0))
    gy = RooGaussian("gy", "gy", y, ref(0.0), ref(2.0))
    cdf = RooProdPdf("gxy", "gxy", [gx, gy]).createCdf(RooArgSet(x, y))
    x.setVal(1.0)
    y.setVal(-1.0)
    assert cdf.GetName() == "gxy_Int[x_prime,y_prime]_Norm[x,y]"
    assert cdf.getVal() == roots(0.19458554441368411, rel=1e-15)


def test_a_binned_sum_announces_no_integral_of_its_terms(capsys: Any) -> None:
    """HistFactory's sums are integrated bin by bin as a whole, so their terms make none."""
    _, _, _, p, f = model()
    pdf = RooRealSumPdf("pdf", "pdf", [RooProduct("a1", "a1", [p, f])], [RooRealVar("c1", "", 1)])
    pdf.isBinnedDistribution = lambda obs=None: True  # type: ignore[method-assign]
    capsys.readouterr()
    pdf.announce_inner(frozenset({"t"}), None)
    assert capsys.readouterr().out == ""
