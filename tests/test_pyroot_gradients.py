"""``TColor::GetLinearGradient`` and ``GetRadialGradient``: colours that shade, by ROOT's API.

A gradient takes a new index of the table, which ``gROOT->GetColor`` hands
back as the gradient itself - a ``TRadialGradient`` can be cast to and set
up again - and a canvas is given it among the session's colours.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from pyrootgraphics import fresh_session  # noqa: F401
from xrdroot.pyroot.graphics import colors
from xrdroot.pyroot.graphics.gradients import GRADIENTS, Point, TLinearGradient, TRadialGradient
from xrdroot.pyroot.graphics.snapshot import _colour_tables


def test_a_linear_gradient_runs_along_its_angle_between_colours_laid_evenly():
    index = ROOT.TColor.GetLinearGradient(90.0, [ROOT.kYellow, ROOT.kOrange, ROOT.kRed])
    made = ROOT.gROOT.GetColor(index)
    assert isinstance(made, TLinearGradient) and made.GetNumber() == index
    assert made.GetColorPositions() == [0.0, 0.5, 1.0] and made.GetNumberOfSteps() == 3
    assert made.GetColors()[:4] == [1.0, 1.0, 0.0, 1.0]  # kYellow, opaque
    assert (made.GetStart().fX, made.GetStart().fY) == pytest.approx((0.5, 0.0))
    assert (made.GetEnd().fX, made.GetEnd().fY) == pytest.approx((0.5, 1.0))
    assert made.GetCoordinateMode() == made.kObjectBoundingMode
    assert (made.GetRed(), made.GetGreen(), made.GetBlue()) == (1.0, 1.0, 0.0)  # its first stop
    made.SetCoordinateMode(made.kPadMode)
    assert made.members()["fCoordinateMode"] == 0 and made.members()["fEnd"] == {
        "fX": pytest.approx(0.5), "fY": pytest.approx(1.0)}


def test_a_radial_gradient_is_set_up_again_from_a_braced_centre_and_given_opacities():
    index = ROOT.TColor.GetRadialGradient(0.5, [2, 3])
    gradient = ROOT.gROOT.GetColor(index)
    assert isinstance(gradient, TRadialGradient) and gradient.GetRadius() == 0.5
    assert gradient.GetGradientType() == gradient.kSimple
    gradient.SetRadialGradient([0.3, 0.3], 0.7)
    gradient.SetColorAlpha(0, 0.25)
    assert (gradient.GetCenter().fX, gradient.GetRadius(), gradient.GetColorAlpha(0)) == (
        0.3, 0.7, 0.25)
    assert gradient.GetColorAlpha(1) == 1.0 and list(gradient.GetCenter()) == [0.3, 0.3]
    gradient.SetStartEndR1R2(Point(0.1, 0.1), 0.05, (0.6, 0.6), 0.9)
    assert (gradient.GetStart().fX, gradient.GetR1(), gradient.GetEnd().fY, gradient.GetR2()) == (
        0.1, 0.05, 0.6, 0.9)
    assert gradient.GetGradientType() == gradient.kExtended
    assert repr(Point(1, 2)) == "Point(1.0, 2.0)"
    members = gradient.members()
    assert (members["fType"], members["fR2"], members["fStart"]) == (1, 0.9, {"fX": 0.1, "fY": 0.1})


def test_fewer_than_two_colours_is_refused_as_root_refuses_it(capsys):
    assert ROOT.TColor.GetLinearGradient(0.0, [2]) == -1
    assert ROOT.TColor.GetRadialGradient(0.5, []) == -1
    err = capsys.readouterr().err
    assert "Error in <TColor::GetLinearGradient>: number of colors should be at least 2" in err
    assert "Error in <TColor::GetRadialGradient>: number of colors should be at least 2" in err


def test_stops_may_be_given_as_red_green_blue_and_opacity_each_and_set_again():
    made = ROOT.TColorGradient(-1, 2, [0.0, 1.0], [1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 1.0])
    assert made.GetColors() == [1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 1.0]
    assert made.GetAlpha() == 0.5 and GRADIENTS[made.GetNumber()] is made
    made.ResetColor(2, [0.0, 0.3], [ROOT.kRed, ROOT.kBlue])
    assert made.GetColorPositions() == [0.0, 0.3] and made.GetColors()[4:7] == [0.0, 0.0, 1.0]
    assert ROOT.TColorGradient(-1, 0, [], []).GetColors() == []


def test_a_canvas_is_given_each_gradient_as_itself_and_the_rest_as_colours():
    plain = ROOT.TColor.GetColor(0.1, 0.2, 0.3)
    index = ROOT.TColor.GetLinearGradient(0.0, [plain, ROOT.kRed])
    tables = _colour_tables()
    classes = [(prim.classname, prim.get("fNumber")) for prim in tables[0]]
    assert ("TColor", plain) in classes and ("TLinearGradient", index) in classes
    del colors.MADE[index]  # a gradient whose colour is gone is not handed over
    assert all(prim.classname == "TColor" for prim in _colour_tables()[0])
