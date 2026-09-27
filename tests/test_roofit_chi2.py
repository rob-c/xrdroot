"""``createChi2`` and ``chi2FitTo``: the chi-square of a density against binned data, ROOT's."""

from __future__ import annotations

import math
from typing import Any

import pytest

from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.pdfs.extend import RooExtendPdf
from xrdroot.roofit.rng import RooRandom
from xrdroot.roofit.variables import RooRealVar

EXPECTED, SUMW2, POISSON, AUTO = 4, 1, 0, 3


def binned() -> tuple[Any, Any, Any, Any]:
    """300 events of ``g(x; 0.3, 1.3)``, seed 21, in 10 bins."""
    x = RooRealVar("x", "x", -5, 5)
    x.setBins(10)
    m, s = RooRealVar("m", "m", 0.3, -2, 2), RooRealVar("s", "s", 1.3, 0.1, 3)
    g = RooGaussian("g", "g", x, m, s)
    RooRandom.randomGenerator().SetSeed(21)
    return g, g.generate([x], 300).binnedClone(), m, s


def test_the_chi_square_takes_each_kind_of_error_as_root_does(capsys: Any) -> None:
    """ROOT 6.40: 13.89109345076403 with the expected errors (and automatically), 5.8045 with
    Poisson's; with squared weights an empty bin predicted full makes it NaN."""
    g, data, _m, _s = binned()
    found = [g.createChi2(data, DataError=kind).getVal() for kind in (EXPECTED, POISSON, AUTO)]
    assert found == pytest.approx([13.89109345076403, 5.804510934833776, 13.89109345076403])
    assert math.isnan(g.createChi2(data, DataError=SUMW2).getVal())
    assert "createChi2(g) fixing normalization set for coefficient determination" in (
        capsys.readouterr().out
    )


def test_an_extended_chi_square_predicts_the_expected_events() -> None:
    """``Extended(true)`` of 280 expected events: ROOT's 16.309761814885363."""
    g, data, _m, _s = binned()
    e = RooExtendPdf("e", "e", g, RooRealVar("n", "n", 280, 0, 1000))
    chi2 = e.createChi2(data, Extended=True)
    assert (chi2.getVal(), chi2.defaultErrorLevel()) == pytest.approx((16.309761814885363, 1.0))


def test_a_chi_square_fit_finds_roots_minimum() -> None:
    """ROOT's 10.103311739227674 at m 0.33163459959767, s 1.38408102529522, m's error 0.08."""
    g, data, m, s = binned()
    result = g.chi2FitTo(data, PrintLevel=-1, Save=True)
    found = (result.minNll(), m.getVal(), s.getVal(), m.getError())
    expected = (10.103311739227674, 0.3316345995976688, 1.384081025295217, 0.08000502681186802)
    assert found == pytest.approx(expected, rel=1e-6)
    assert g.chi2FitTo(data, PrintLevel=-1, Hesse=False) is None


def test_a_chi_square_is_a_function_like_any_other() -> None:
    """Its value in a context is its value: it has no variables of its own to be given."""
    g, data, _m, _s = binned()
    chi2 = g.createChi2(data)
    assert chi2.compute({}) == pytest.approx(13.89109345076403)


def test_bins_empty_and_predicted_empty_are_passed_over() -> None:
    """With no events at all every bin is empty and predicted empty: a chi-square of 0."""
    from xrdroot.roofit.data.datahist import RooDataHist

    g, data, _m, _s = binned()
    nothing = RooDataHist("nothing", "nothing", [g.leaves()[0]])
    assert (g.createChi2(nothing).getVal(), data.numEntries()) == (0.0, 10)
