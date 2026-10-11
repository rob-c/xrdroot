"""``RooAbsReal::derivative``: Richardson's derivatives of a function, as ROOT 6.40 takes them.

The expected values are ROOT 6.40's for the same Gaussian. On ROOT's machine
they are equal to the bit. Elsewhere the Gaussian's ``exp`` may round
otherwise in its last place, and the third difference divides by ``h³`` -
1e-9 for the default step - so an ulp of the function (1e-16) moves it by up
to 1e-7 of itself: 1e-6 is a tenfold margin, and a thousandth of what a
step of another size moves it by (a step of 0.02 moves the second
derivative by 2e-3 of itself).
"""

from __future__ import annotations

import pytest

from refmachine import roots
from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.derivative import RooDerivative
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.variables import RooRealVar

#: ROOT 6.40's derivatives of ``exp(-(x-1)^2/2)`` at ``x = 0.3``, by order, for two steps.
ROOTS = {
    0.001: (0.5478931767692504, -0.39917931449195976, -1.3752112411798594),
    0.01: (0.5478931766512818, -0.39917931452970734, -1.3752118713794557),
}


def _gauss() -> tuple[RooGaussian, RooRealVar, RooRealVar]:
    x = RooRealVar("x", "x", 0.3, -10, 10)
    mean = RooRealVar("mean", "mean", 1, -10, 10)
    sigma = RooRealVar("sigma", "sigma", 1, 0.1, 10)
    return RooGaussian("gauss", "gauss", x, mean, sigma), x, sigma


@pytest.mark.parametrize("eps", sorted(ROOTS))
def test_each_order_is_roots_and_leaves_the_variable_half_a_step_down(eps: float) -> None:
    gauss, x, _ = _gauss()
    for order, want in enumerate(ROOTS[eps], start=1):
        x.setVal(0.3)
        assert gauss.derivative(x, order, eps).getVal() == roots(want, rel=1e-6)
        assert x.getVal() == 0.3 - eps / 2


def test_a_derivative_is_named_as_root_names_it_by_its_order_and_variable() -> None:
    gauss, x, sigma = _gauss()
    first, by_width = gauss.derivative(x), gauss.derivative(sigma, 2)
    assert isinstance(first, RooDerivative) and first.GetName() == "gauss_DERIV_1x"
    assert first.GetTitle() == "1Derivative of gauss w.r.t x "
    assert by_width.GetName() == "gauss_DERIV_2sigma"
    assert (first.order(), first.eps(), by_width.order()) == (1, 0.001, 2)
    first.setEps(0.01)
    assert first.eps() == 0.01
    assert by_width.getVal() == roots(-0.9626483106206507, rel=1e-6)


def test_a_derivative_of_the_normalised_density_takes_its_normalisation() -> None:
    gauss, x, _ = _gauss()
    normalised = gauss.derivative(x, RooArgSet(x), 2)
    assert normalised.getVal() == roots(-0.15924950597101528, rel=1e-6)


def test_an_order_past_the_third_is_refused() -> None:
    gauss, x, _ = _gauss()
    with pytest.raises(ValueError, match="order must be 1, 2 or 3"):
        gauss.derivative(x, 4)
