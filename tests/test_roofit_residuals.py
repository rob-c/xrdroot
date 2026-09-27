"""``chiSquare``, ``residHist`` and ``pullHist``: data against a curve, as ROOT works them out."""

from __future__ import annotations

from typing import Any

import pytest

from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.rng import RooRandom
from xrdroot.roofit.variables import RooRealVar


def framed() -> Any:
    """200 events of ``g(x; 0.5, 1.2)``, seed 11, in 8 bins, and ``g`` drawn over them."""
    x = RooRealVar("x", "x", -5, 5)
    g = RooGaussian(
        "g", "g", x, RooRealVar("m", "m", 0.5, -2, 2), RooRealVar("s", "s", 1.2, 0.1, 3)
    )
    RooRandom.randomGenerator().SetSeed(11)
    data = g.generate([x], 200)
    frame = x.frame(Bins=8)
    data.plotOn(frame)
    g.plotOn(frame)
    return frame


def points(hist: Any) -> list[tuple[float, float, float, float]]:
    return [
        (hist.GetPointX(i), hist.GetPointY(i), hist.GetErrorYlow(i), hist.GetErrorYhigh(i))
        for i in range(hist.GetN())
    ]


def flat(rows: list[tuple[float, ...]]) -> list[float]:
    return [value for row in rows for value in row]


def test_a_frames_chi_square_is_roots_per_degree_of_freedom() -> None:
    """ROOT's 0.46320928040480586, 0.6948139206072088 with 2 parameters, and 0.5559 named."""
    frame = framed()
    found = (frame.chiSquare(), frame.chiSquare(2), frame.chiSquare("g_Norm[x]", "h_gData", 1))
    assert found == pytest.approx((0.46320928040480586, 0.6948139206072088, 0.555851136485767))


def test_residuals_are_the_data_less_the_curves_average_over_each_bin() -> None:
    """``resid_h_gData_g_Norm[x]``: ROOT's points and the data's own error bars."""
    hist = framed().residHist()
    assert (hist.GetName(), hist.GetTitle(), hist.GetErrorXlow(0)) == (
        "resid_h_gData_g_Norm[x]",
        "Residual of Histogram of gData_plot__x and Projection of g",
        0.0,
    )
    assert flat(points(hist)[:4]) == pytest.approx(flat([
        (-4.375, -0.0796482758488136, 0.0, 1.1478744644493182),
        (-3.125, 0.7777631773105322, 1.2918145601810285, 2.6378596234552454),
        (-1.875, 1.731201644463681, 3.8293800985750543, 4.958738432720615),
        (-0.625, -7.228527598766036, 6.757580994790942, 7.8314889141121),
    ]), rel=1e-12)  # fmt: skip


def test_pulls_are_residuals_in_units_of_the_error_on_the_curves_side() -> None:
    """``pull_h_gData_g_Norm[x]``: ROOT's -0.0694 and 0.6021; unaveraged, 1.1327 at -3.125."""
    frame = framed()
    pulls = points(frame.pullHist())
    plain = points(frame.residHist("", "", False, False))
    assert (pulls[0][1], pulls[1][1], pulls[3][2]) == pytest.approx(
        (-0.0693876188691279, 0.6020702980786502, 0.8628730844034), rel=1e-12
    )
    assert plain[1][1] == pytest.approx(1.1327478174633472, rel=1e-12)


def test_a_point_with_no_error_on_the_curves_side_has_no_pull(capsys: Any) -> None:
    """ROOT sets such a pull to zero and says so."""
    frame = framed()
    hist = frame.findObject(None, None)
    data = frame.getObject(0)
    data.members["fEYhigh"][0] = 0.0
    capsys.readouterr()
    pulls = points(frame.pullHist())
    assert (pulls[0][1:], hist is not None) == ((0.0, 0.0, 0.0), True)
    assert capsys.readouterr().out == (
        "[#0] WARNING:Plotting -- RooHist::makeResisHist(h_gData) WARNING: point 0 has zero "
        "error, setting residual to zero\n"
    )


def test_a_residual_histogram_can_be_put_on_a_frame_of_its_own() -> None:
    """rf109's ``frame2.addPlotable(hresid, "P")``: the points drawn, their x errors none."""
    frame = framed()
    hist = frame.residHist()
    other = frame.getPlotVar().frame(Bins=8)
    other.addPlotable(hist, "P")
    found = (other.numItems(), other.getDrawOptions(hist.GetName()), hist.GetErrorXhigh(0))
    assert found == (1, "P", 0.0)
