"""``ProjectionRange``: a curve's projected observables integrated over a named range only."""

from __future__ import annotations

from typing import Any

import pytest

from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.pdfs.prodpdf import RooProdPdf
from xrdroot.roofit.rng import RooRandom
from xrdroot.roofit.variables import RooConstVar, RooRealVar


def test_a_projection_range_integrates_the_other_observables_only_there(capsys: Any) -> None:
    """``ProjectionRange("sig")``: ROOT's curve ``model_Int[y|sig]_Norm[x,y]``, 15.4686 at 0."""
    x, y = RooRealVar("x", "x", -5, 5), RooRealVar("y", "y", -5, 5)
    gx = RooGaussian("gx", "gx", x, RooConstVar("0", "0", 0.0), RooConstVar("1", "1", 1.0))
    gy = RooGaussian("gy", "gy", y, RooConstVar("0", "0", 0.0), RooConstVar("2", "2", 2.0))
    model = RooProdPdf("model", "model", [gx, gy])
    y.setRange("sig", -1, 1)
    frame = x.frame(Bins=10)
    RooRandom.randomGenerator().SetSeed(2)
    model.generate([x, y], 100).plotOn(frame)
    capsys.readouterr()
    model.plotOn(frame, ProjectionRange="sig")
    curve = frame.getObject(1)
    assert curve.GetName() == "model_Int[y|sig]_Norm[x,y]"
    assert (curve.Eval(0.0), curve.Eval(1.5)) == pytest.approx(
        (15.468612757002347, 5.021923298171657), rel=1e-9
    )
    assert capsys.readouterr().out == (
        "[#1] INFO:Plotting -- RooAbsReal::plotOn(model) plot on x integrates over variables "
        "(y) in range sig\n"
    )
