"""``RooHypatia2``: ROOT's Hypatia line shape, its two cut points and the tails beyond them.

Every value here is the 6.40 oracle's own, read from ``RooHypatia2.cxx:242``'s formula with a macro
that asks ROOT itself (``$R/probe/hyp.C``): seven abscissae across the core and both tails, and the
density's integral over its range. The engine's arithmetic is the same C library calls ROOT makes,
so the points agree to the last of 17 digits on ROOT's machine and within a ulp elsewhere.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from refmachine import ROOTS_MACHINE
from xrdroot.errors import UnsupportedFeatureError

#: The 6.40 oracle's density at each abscissa, for the test's own parameters.
ORACLE = [
    (-10.0, 0.00098518533684157348),
    (-2.0, 0.089442719099991588),
    (-0.5, 0.71554175279993271),
    (0.0, 1.0),
    (0.5, 0.71554175279993271),
    (2.0, 0.26794336563407328),
    (10.0, 0.21632221225408768),
]

#: The oracle's integral over ``[-10, 10]``, which the round measured at 3.8552008214678435.
ORACLE_INTEGRAL = 3.8552008214678435

ARGS = (
    "Hypatia2::hypatia(x[0.0, -10., 10.], lambda[-1.0, -100., -0.01], 0.0, 0.0, "
    "sigma[1.0, 0.01, 10.], mu[0.0], a[50., 0.01, 1000.], n[1.5, 0.01, 10.], "
    "a2[1.0, 0.01, 1000.], n2[0.1, 0.01, 10.])"
)


def _pair(definition: str = ARGS) -> tuple[ROOT.RooWorkspace, ROOT.RooAbsPdf, ROOT.RooRealVar]:
    """The workspace the factory builds from ``definition``, its density and its x."""
    workspace = ROOT.RooWorkspace("ws")
    workspace.factory(definition)
    return workspace, workspace.pdf("hypatia"), workspace.var("x")


def test_the_hypatia_shape_is_the_oracles_at_every_point_it_has_a_branch_of() -> None:
    """The core between the cuts, and a tail either side of them - at ``a*sigma`` = 50 and
    ``a2*sigma`` = 1, so ``x = 2`` is the right tail, ``x = -0.5`` the core and ``x = -10`` the
    left one, and the value must be ROOT's at all three."""
    workspace, density, x = _pair()
    assert workspace and density is not None
    for abscissa, wanted in ORACLE:
        x.setVal(abscissa)
        if ROOTS_MACHINE:
            assert density.getVal() == wanted, f"x = {abscissa}"
        assert density.getVal() == pytest.approx(wanted, rel=1e-15)


def test_the_integral_of_the_hypatia_shape_is_the_oracles() -> None:
    """``createIntegral`` over ``x``'s own range: numeric here, as ROOT's closed form is not written
    (``W0-220``), so the tolerance is the integrator's and not a closed form's - the round measured
    7.3e-8 between ROOT's answer and this one."""
    _workspace, density, x = _pair()
    x.setVal(0.0)
    found = density.createIntegral(ROOT.RooArgSet(x)).getVal()
    assert found == pytest.approx(ORACLE_INTEGRAL, rel=1e-6)


def test_a_zeta_above_zero_is_refused_by_name_rather_than_shaped_another_way() -> None:
    """ROOT's ``zeta > 0`` branch is the Bessel ratio of ``besselK`` and its ``LogEval``; this kit
    does not have it, so the density says so and names what to fit instead."""
    _workspace, density, _x = _pair(ARGS.replace(", 0.0, 0.0,", ", 1.0, 0.0,"))
    with pytest.raises(UnsupportedFeatureError, match="only the zeta = 0 shape"):
        density.getVal()


def test_a_lambda_that_is_not_negative_is_refused_rather_than_answered_zero() -> None:
    """ROOT itself logs and returns 0 there; this kit refuses the shape rather than hand back a
    density ROOT would not, and says which sign it wants."""
    wide = ARGS.replace("lambda[-1.0, -100., -0.01]", "lambda[-1.0, -100., 10.]")
    workspace, density, _x = _pair(wide)
    workspace.var("lambda").setVal(1.0)  # inside the declared range, so no rounding into it
    with pytest.raises(UnsupportedFeatureError, match="only lambda < 0"):
        density.getVal()
