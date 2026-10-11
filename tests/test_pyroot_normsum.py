"""``TF1NormSum``: functions normalised over their range and summed with coefficients.

What ROOT fixes, names and normalises is checked here; fitNormSum checks the
fit through it against ROOT 6.40's numbers, and the normalising integrals
are TF1's own, to the ten digits of the comparisons.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_sum_names_its_coefficients_and_the_shapes_parameters_but_constant() -> None:
    peak = ROOT.TF1("peak", "gaus", -5, 5)
    peak.SetParameters(7, 1, 0.5)
    slope = ROOT.TF1("slope", "[0]+[1]*x", -1, 1)  # no Constant, and a range the sum replaces
    slope.SetParameters(2, 0.1)
    summed = ROOT.TF1NormSum(peak, slope, 10, 20)
    assert summed.GetNpar() == 6
    assert [summed.GetParName(i) for i in range(6)] == ["Coeff0", "Coeff1", "Mean", "Sigma",
                                                       "p0", "p1"]  # fmt: skip
    assert list(summed.GetParameters().data()) == [10.0, 20.0, 1.0, 0.5, 2.0, 0.1]
    # each normalised over [-5, 5]: the Gaussian's whole integral and the line's 2 * 10
    x = np.array([[0.0, 1.0]])
    gauss = np.exp(-0.5 * ((x[0] - 1.0) / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi))
    assert summed(x) == pytest.approx(10 * gauss + 20 * (2 + 0.1 * x[0]) / 20.0, rel=1e-9)
    assert peak.GetParameter(0) == 7  # the originals untouched


def test_new_parameters_renormalise_only_the_shapes_that_changed() -> None:
    a = ROOT.TF1("a", "expo", 0, 2)
    b = ROOT.TF1("b", "expo", 0, 2)
    summed = ROOT.TF1NormSum([a, b], [1.0, 3.0], 0.5)
    summed.SetParameters([1.0, 3.0, -1.0, 0.0])
    point = summed(np.array([1.0]), [2.0, 1.0, -1.0, 0.0])
    flat = 0.5
    expo = np.exp(-1.0) / (1.0 - np.exp(-2.0))
    assert point == pytest.approx(0.5 * (2.0 * expo + 1.0 * flat), rel=1e-9)
    assert [summed.GetParName(i) for i in range(4)] == ["Coeff0", "Coeff1", "Slope1", "Slope2"]
    assert ROOT.TF1NormSum(a, b, ROOT.TF1("c", "gaus", 0, 2)).GetNpar() == 7
