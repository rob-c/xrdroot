"""``createProjection``: a density integrated over some observables, a density of the rest.

The projection of ``f(x, y)`` onto ``x`` is ``Int f dy / Int f dx dy``, as
``RooProjectedPdf`` normalises it: of a product of independent factors it
is the ``x`` factor itself, and of a correlated density the integral the
test works out on its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.pdfs.basic import RooGaussian, ref
from xrdroot.roofit.pdfs.generic import RooGenericPdf
from xrdroot.roofit.pdfs.prodpdf import RooProdPdf
from xrdroot.roofit.pdfs.projected import RooProjectedPdf
from xrdroot.roofit.variables import RooRealVar


def test_the_projection_of_independent_factors_is_the_factor_left() -> None:
    x = RooRealVar("x", "x", 0.5, -5, 5)
    y = RooRealVar("y", "y", 0.0, -5, 5)
    gx = RooGaussian("gx", "gx", x, ref(0.0), ref(1.0))
    gy = RooGaussian("gy", "gy", y, ref(1.0), ref(2.0))
    projected = RooProdPdf("p", "p", [gx, gy]).createProjection(RooArgSet(y))
    assert isinstance(projected, RooProjectedPdf)
    assert projected.GetName() == "p_Proj[y]"
    assert "x" in projected.dependents() and "y" not in projected.dependents()
    xs = np.array([-1.0, 0.0, 2.0])
    want = gx.value({"x": xs}, {"x"})
    assert projected.value({"x": xs}, {"x"}) == pytest.approx(want, rel=1e-9)
    assert projected.value({"x": xs}, {"x", "y"}) == pytest.approx(want, rel=1e-9)
    assert projected.getVal(RooArgSet(x)) == pytest.approx(float(gx.getVal(RooArgSet(x))))
    assert projected.printMetaArgs() == ""


def test_the_projection_of_a_correlated_density_integrates_it_out() -> None:
    """``f = 1 + x*y`` on the unit square: ``Int f dy = 1 + x/2`` over ``Int f = 5/4``."""
    x = RooRealVar("x", "x", 0.25, 0, 1)
    y = RooRealVar("y", "y", 0.5, 0, 1)
    f = RooGenericPdf("f", "f", "1 + x*y", [x, y])
    projected = f.createProjection(RooArgSet(y))
    assert projected.value({"x": 0.25}, {"x"}) == pytest.approx((1 + 0.125) / 1.25, rel=1e-6)
