"""``RooMinimizer::contour``: Minuit's n-sigma contours of two parameters, framed as ROOT does.

The points are MnContours' - the C++ ROOT runs - so what is checked here is
what RooFit makes of them: a frame over both ranges, a marker at the
minimum, one closed blue line per positive n in its own line style, the
error level and the parameters put back, and RooFit's complaints.
"""

from __future__ import annotations

from typing import Any

import pytest

from xrdroot.roofit.fitting import contour as contour_module
from xrdroot.roofit.fitting.minimizer import RooMinimizer
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooRealVar

pytest.importorskip("iminuit")


def _fitted() -> tuple[RooMinimizer, Any, Any, Any]:
    """MIGRAD and MINOS run on 200 events of a Gaussian, and the minimizer, nll, mean, width."""
    generator().SetSeed(4357)
    x = RooRealVar("x", "x", -10, 10)
    m = RooRealVar("m", "the mean", 0, -5, 5)
    s = RooRealVar("s", "the width", 2, 0.1, 10)
    g = RooGaussian("g", "g", x, m, s)
    nll = g.createNLL(g.generate([x], 200))
    minimizer = RooMinimizer(nll)
    minimizer.setPrintLevel(-1)
    minimizer.migrad()
    minimizer.minos([m])
    return minimizer, nll, m, s


def test_each_positive_n_is_a_closed_contour_in_its_own_style_with_the_minimum_marked() -> None:
    minimizer, nll, m, s = _fitted()
    best = (m.getVal(), s.getVal())
    frame = minimizer.contour(m, s, 1, 0, 2, npoints=12)
    assert frame.GetTitle() == "A RooPlot" and frame.GetMaximum() == 10.0
    assert frame.GetMinimum() == 0.1
    marker, one, two = (item[0] for item in frame.items)
    assert (marker.GetX(), marker.GetY(), marker.GetMarkerStyle()) == (*best, 8)
    assert one.GetName() == f"contour_{nll.GetName()}_n1.000000" and one.GetN() == 13
    assert (one.GetLineStyle(), two.GetLineStyle(), two.GetLineWidth()) == (1, 3, 2)
    assert two.GetLineColor() == 600 and [item[1] for item in frame.items] == ["", "L", "L"]
    assert (one.GetX()[0], one.GetY()[0]) == (one.GetX()[12], one.GetY()[12])


def test_a_contour_is_n_sigma_out_and_leaves_the_error_level_and_parameters_as_they_were() -> None:
    minimizer, nll, m, s = _fitted()
    before = (m.getVal(), m.getError(), m.getAsymErrorLo(), s.getVal(), s.getError())
    frame = minimizer.contour(m, s, 1, 2, npoints=12)
    assert minimizer.minuit.errordef == 0.5
    assert (m.getVal(), m.getError(), m.getAsymErrorLo(), s.getVal(), s.getError()) == before
    rises = []
    for graph, _, _ in frame.items[1:]:
        m.setVal(graph.GetX()[3])
        s.setVal(graph.GetY()[3])
        rises.append(nll.getVal() - minimizer.minuit.fval)
    # With nothing else to profile, a point of the n-sigma contour is where the likelihood has
    # risen by n²/2. MnFunctionCross stops within its tolerance of that - here within a
    # ten-thousandth; 1e-3 is a tenfold margin, and a contour at a wrong level would miss by
    # a quarter or more.
    assert rises == pytest.approx([0.5, 2.0], rel=1e-3)


def test_a_parameter_not_fitted_is_named_and_gives_no_frame(capsys: Any) -> None:
    minimizer, nll, m, _ = _fitted()
    other = RooRealVar("y", "y", 0, 1)
    capsys.readouterr()
    assert minimizer.contour(m, other) is None
    assert capsys.readouterr().out == (
        "[#0] ERROR:Minimization -- RooMinimizer::contour(RooMinimizer) ERROR: y is not a "
        f"floating parameter of {nll.GetName()}\n"
    )


def test_a_contour_before_migrad_is_the_marked_frame_and_a_warning(capsys: Any) -> None:
    generator().SetSeed(4357)
    x, m = RooRealVar("x", "x", -10, 10), RooRealVar("m", "m", 0, -5, 5)
    s = RooRealVar("s", "s", 2, 0.1, 10)
    g = RooGaussian("g", "g", x, m, s)
    minimizer = RooMinimizer(g.createNLL(g.generate([x], 20)))
    capsys.readouterr()
    frame = minimizer.contour(m, s)
    assert len(frame.items) == 1 and "run Migrad before contours!" in capsys.readouterr().out


def test_a_contour_minuit_does_not_give_is_said_and_left_out(
    capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    minimizer, _, m, s = _fitted()
    monkeypatch.setattr(contour_module, "minuit_contour", lambda *args: None)
    capsys.readouterr()
    assert len(minimizer.contour(m, s, 1.5, 0).items) == 1
    assert capsys.readouterr().out.splitlines()[-1] == (
        "[#0] ERROR:Minimization -- RooMinimizer::contour(RooMinimizer) ERROR: MINUIT did not "
        "return a contour graph for n=1.5"
    )
