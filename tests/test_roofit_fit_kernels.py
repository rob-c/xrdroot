"""A likelihood's events through RooBatchCompute's kernels, and fits in a range, against ROOT.

ROOT 6.40.04 fits through compiled kernels that round as ``evaluate()``
does not: VDT's ``fast_exp`` for the Gaussian and the exponential, the
Gaussian's square multiplied by ``-0.5 / sigma^2``. In a range each
coefficient of a sum, and each yield, is projected by one over its full
integral normalised in the range. The expected numbers were printed by ROOT
through PyROOT (``vdt::fast_exp`` through cling) for the same models and
the same events, drawn after ``RooRandom::randomGenerator()->SetSeed(4357)``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from refmachine import FIT_REL, ROOTS_MACHINE, roots
from xrdroot.roofit import integration, kernels
from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.data.dataset import RooDataSet
from xrdroot.roofit.pdfs.addpdf import RooAddPdf
from xrdroot.roofit.pdfs.basic import RooExponential, RooGaussian
from xrdroot.roofit.pdfs.extend import RooExtendPdf
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooRealVar

#: Arguments where ``vdt::fast_exp`` and the C library's ``exp`` differ, and VDT's answer.
VDT = [
    (-25.967584659528157, 5.277415326675193e-12),
    (-27.3816058906282, 1.2832747706111596e-12),
    (-7.3844526651909845, 0.0006208303727027469),
    (-15.64479150036546, 1.6052895585039868e-07),
    (-16.213483153299027, 9.09020618593571e-08),
    (-19.329885270156833, 4.028445488940469e-09),
    (-17.010000199006015, 4.098743834540406e-08),
    (-9.252089229853429, 9.591106234062467e-05),
]


def test_vdts_fast_exp_is_vdts_to_the_bit_and_its_limits_are_vdts() -> None:
    """Only additions, products and a quotient: VDT's number on every machine, not ``exp``'s."""
    arguments = np.array([one for one, _ in VDT])
    assert kernels.fast_exp(arguments).tolist() == [value for _, value in VDT]
    assert all(kernels.fast_exp(one) == value for one, value in VDT)
    assert kernels.fast_exp(np.array([708.1, -708.1, 0.0, -3.0])).tolist() == [
        math.inf,
        0.0,
        1.0,
        math.exp(-3.0),
    ]
    assert math.isnan(kernels.fast_exp(math.nan))


def test_a_density_is_its_kernel_inside_a_likelihood_and_evaluate_outside_it() -> None:
    """The Gaussian's value at the same point is two numbers, a bit apart, as in ROOT; and a
    numerical integral inside a likelihood evaluates the density as ``evaluate()`` does."""
    x = RooRealVar("x", "x", -0.3, -10, 10)
    g = RooGaussian("g", "g", x, 1.0, 1.5072343349049426)
    plain = g.compute({})
    with kernels.likelihood():
        assert kernels.active()
        fast = g.compute({})
        seen: list[bool] = []

        def inner(ctx: dict[str, Any]) -> Any:
            seen.append(kernels.active())
            return np.ones_like(np.asarray(ctx["x"], dtype=float))

        integration.numeric(g, ["x"], inner, {}, None)
        assert seen and not any(seen) and kernels.active()
    assert not kernels.active()
    assert plain == math.exp(-0.5 * 1.3 * 1.3 / (1.5072343349049426**2))
    assert fast == kernels.fast_exp(1.3 * 1.3 * (-0.5 / (1.5072343349049426**2)))
    one = RooDataSet("d", "d", RooArgSet(x))
    one.add(RooArgSet(x))
    m = g.createNLL(one)
    assert m.getVal() == roots(1.7011740055898574, rel=1e-15)  # ROOT's; libm's log elsewhere


def _exponential() -> tuple[Any, Any, Any]:
    """rf204b's exponential in x, with its three ranges, and its 10000 events."""
    generator().SetSeed(4357)
    x = RooRealVar("x", "x", 10, 100)
    alpha = RooRealVar("alpha", "alpha", -0.04, -0.1, -0.0)
    model = RooExponential("model", "Exponential model", x, alpha)
    for name, low, high in (("LEFT", 10, 20), ("RIGHT", 60, 100), ("FULL", 10, 100)):
        x.setRange(name, low, high)
    return x, alpha, model


def test_an_extension_without_a_range_expects_its_yield_in_the_range_fitted() -> None:
    """``RooExtendPdf`` with no range of its own: ROOT puts ``n`` in the fit's range as it is,
    and one with a range divides by the reciprocal of its fraction there."""
    x, alpha, model = _exponential()
    data = model.generate(x, 10000)
    n = RooRealVar("N", "N", 10000, 0, 20000)
    plain = RooExtendPdf("plain", "e", model, n)
    ranged = RooExtendPdf("ranged", "e", model, n, "FULL")
    alpha.setVal(-0.0388)
    n.setVal(10226.1)
    assert plain.createNLL(data, Range="LEFT").getVal() == roots(-13322.844103078678, rel=1e-13)
    assert ranged.createNLL(data, Range="LEFT").getVal() == roots(-16410.481474593649, rel=1e-13)


def _fit(pdf: Any, data: Any, rng: str) -> Any:
    return pdf.fitTo(data, Range=rng, PrintLevel=-1, Save=True)


def test_rf204bs_four_fits_in_ranges_are_roots(capsys: Any) -> None:
    """rf204b: an extended fit in one range, a plain and an extended fit in two, and a sum of
    signal and background in two - its signal yield at its lower limit, where only where
    MIGRAD stopped says how near zero it got: ROOT's point on its machine, near zero elsewhere."""
    x, alpha, model = _exponential()
    data = model.generate(x, 10000)
    n = RooRealVar("N", "Extended term", 0, 20000)
    extended = RooExtendPdf("extmodel", "Extended model", model, n, "FULL")
    first = _fit(extended, data, "LEFT")
    assert (first.minNll(), n.getVal(), alpha.getVal()) == pytest.approx(
        (-16410.48257115516, 10226.167169357546, -0.038825271226653031), rel=FIT_REL
    )
    second = _fit(model, data, "LEFT,RIGHT")
    assert (second.minNll(), alpha.getVal(), alpha.getError()) == pytest.approx(
        (14265.89151234249, -0.039961641319666813, 0.00055975966623601423), rel=FIT_REL
    )
    third = _fit(extended, data, "LEFT,RIGHT")
    assert (third.minNll(), n.getVal(), n.getError()) == pytest.approx(
        (-19036.888758998659, 9988.1000044228913, 149.88782073999391), rel=FIT_REL
    )
    nsig = RooRealVar("Nsig", "Number of signal events", 1000, 0, 2000)
    nbkg = RooRealVar("Nbkg", "Number of background events", 10000, 0, 20000)
    sig = RooGaussian("sig", "Signal model", x, 40.0, 5.0)
    total = RooAddPdf("modelsum", "NSig*signal + NBkg*background", [sig, model], [nsig, nbkg])
    fourth = _fit(total, data, "LEFT,RIGHT")
    assert (fourth.minNll(), nbkg.getVal(), alpha.getVal()) == pytest.approx(
        (-19036.888797514861, 9987.8578405968965, -0.03997441901008781), rel=FIT_REL
    )
    if ROOTS_MACHINE:
        assert (third.edm(), fourth.edm()) == (3.9013354366067862e-05, 5.0029290349580108e-07)
        assert nsig.getVal() == 1.3503490751791414e-08
    assert 0 <= nsig.getVal() < 0.1
    capsys.readouterr()
