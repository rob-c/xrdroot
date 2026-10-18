"""Dawson's integration stays available with the project's NumPy 1.x floor."""

from __future__ import annotations

import numpy as np
import pytest

from refmachine import roots
from xrdroot.roofit.pdfs.shapes import RooArgusBG, _dawson
from xrdroot.roofit.variables import RooRealVar


@pytest.mark.parametrize(("x", "expected"), [
    (0.0, 0.0),
    (0.01, 0.009999333360040906),
    (0.1, 0.09933599243951952),
    (0.4, 0.35994348460155473),
    (1.0, 0.5380795485794334),
    (2.0, 0.3013407222570031),
    (5.0, 0.10213928270010246),
    (10.0, 0.050295506807390304),
    (-0.4, -0.35994348460155473),
    (-2.0, -0.3013407222570031),
])
def test_dawson_keeps_the_existing_quadrature_without_numpy_two(x, expected, monkeypatch):
    monkeypatch.delattr(np, "trapezoid", raising=False)
    # Apple and glibc exp can differ by an ulp before the trapezoidal sum.
    assert _dawson(x) == roots(expected, rel=1e-14, abs=1e-15)


def test_rising_argus_normalization_does_not_require_numpy_two(monkeypatch):
    monkeypatch.delattr(np, "trapezoid", raising=False)
    mass = RooRealVar("m", "m", 5.25, 5.0, 5.3)
    endpoint = RooRealVar("m0", "m0", 5.291)
    slope = RooRealVar("c", "c", 3.0)
    model = RooArgusBG("argus", "argus", mass, endpoint, slope)
    # The established 2001-point Dawson rule approximates ROOT's closed form at 1e-6.
    assert model.getVal([mass]) == pytest.approx(1.72012151487, rel=1e-6)
