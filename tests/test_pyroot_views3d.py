"""3-D in a pad - ``TView3D``, ``TPolyLine3D``, ``TPolyMarker3D`` - and the widgets a picture has.

``basic3d.C`` and ``tornado.C`` are ROOT's tutorials these are made for;
what they draw is checked by the tutorial harness against ROOT's pictures,
and here by what it is made of.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from geomsupport import geometry_session  # noqa: F401
from pyrootsupport import expect
from xrdroot.errors import UnsupportedFeatureError


def test_a_view_is_the_current_pad_s_from_its_angles_until_the_pad_is_cleared() -> None:
    canvas = ROOT.TCanvas("c", "c", 400, 300)
    canvas.SetTheta(20)
    view = ROOT.TView.CreateView(1)
    expect((canvas.GetView(), view), (view.GetLongitude(), -120.0), (view.GetLatitude(), 70.0),
           (view.GetPsi(), 0.0), (view.IsPerspective(), False), (view.GetSystem(), 1),
           ((canvas.members["fX1"], canvas.members["fY2"]), (-1.0, 1.0)))  # fmt: skip
    canvas.Clear()
    assert canvas.GetView() is None


def test_a_view_is_ranged_by_hand_or_by_what_it_sees() -> None:
    ROOT.TCanvas("c", "c", 400, 300)
    view = ROOT.TView3D(11, np.zeros(3), np.ones(3))
    low, high = np.zeros(3), np.zeros(3)
    view.GetRange(low, high)
    view.SetRange(5, 5, 5, 25, 25, 25)
    expect((list(high), [1.0] * 3), (list(view.GetRmin()), [5.0] * 3), (view.IsPerspective(), True))
    view.SetRange([0, 0, 0], [2, 2, 2])
    view.SetAutoRange(False)
    assert list(view.GetRmax()) == [2.0] * 3
    view.SetAutoRange()
    assert list(view.GetRmax()) == [1.0] * 3  # nothing 3-D in the pad yet


def test_an_automatic_range_is_the_box_round_the_3d_in_the_pad() -> None:
    ROOT.TCanvas("c", "c", 400, 300)
    view = ROOT.TView.CreateView(1, 0, 0)
    line = ROOT.TPolyLine3D(2)
    line.SetPoint(0, -1, -2, -3)
    line.SetPoint(1, 4, 5, 6)
    line.Draw()
    assert (list(view.GetRmin()), list(view.GetRmax())) == ([-1.0, -2.0, -3.0], [4.0, 5.0, 6.0])


def test_a_view_turns_to_look_from_where_it_is_told_and_takes_the_pad_round() -> None:
    canvas = ROOT.TCanvas("c", "c", 400, 300)
    view = ROOT.TView.CreateView(1)
    irep = type("Cell", (), {"value": 7})()
    view.SetView(10, 20, 30, irep)
    view.Front()
    front = (view.GetLongitude(), view.GetLatitude())
    view.Top(canvas)
    view.Side()
    view.SetPerspective()
    view.SetParallel()
    view.ShowAxis()
    expect((irep.value, 0), (front, (270.0, 90.0)), (view.GetLongitude(), 0.0),
           ((canvas.members["fPhi"], canvas.members["fTheta"]), (-90.0, 0.0)))  # fmt: skip


def test_3d_points_are_set_one_by_one_grown_as_they_go() -> None:
    line = ROOT.TPolyLine3D(1, np.array([1.0, 2.0, 3.0]), "same")
    line.SetPoint(3, 7, 8, 9)
    line.SetNextPoint(1, 1, 1)
    markers = ROOT.TPolyMarker3D(2, np.array([0.0, 1.0]), np.array([2.0, 3.0]),
           np.array([4.0, 5.0]), 20)
    markers.SetPolyMarker(1, np.array([9.0, 9.0, 9.0]))
    line.SetPolyLine(2)
    expect((line.GetN(), 2), (line.GetLastPoint(), 1), (list(markers.GetP()), [9.0, 9.0, 9.0]),
           (markers.GetMarkerStyle(), 20), (ROOT.TPolyLine3D(3).GetN(), 3))  # fmt: skip


def test_3d_is_painted_through_the_pad_s_view(tmp_path: Any) -> None:
    canvas = ROOT.TCanvas("c", "c", 200, 200)
    ROOT.TView.CreateView(1).SetRange(0, 0, 0, 1, 1, 1)
    line = ROOT.TPolyLine3D(2, np.array([0.0, 0, 0, 1, 1, 1]))
    line.SetLineWidth(0)
    line.Draw()
    ROOT.TPolyLine3D(2, np.array([0.0, 0, 0, 1, 1, 1])).Draw()
    ROOT.TPolyMarker3D(1, np.array([0.5, 0.5, 0.5]), 3).Draw()
    canvas.SaveAs(str(tmp_path / "view.png"))
    assert (tmp_path / "view.png").stat().st_size > 0


def test_a_button_is_a_raised_pad_with_its_title_and_its_method() -> None:
    ROOT.TCanvas("c", "c", 200, 200)
    button = ROOT.TButton("Press", "gSystem->Exit(0)", 0.1, 0.1, 0.5, 0.2)
    button.SetTextSize(0.5)
    button.SetMethod("gSystem->Exit(1)")
    button.Draw()
    plain = ROOT.TGroupButton("", "", 0.0, 0.0, 0.1, 0.1)
    expect((button.GetMethod(), "gSystem->Exit(1)"), (button.GetTextSize(), 0.5),
           (button.GetTextFont(), 61), (button.members["fBorderMode"], 1),
           (plain.primitives, []))  # fmt: skip
    with pytest.raises(AttributeError):
        plain.SetTextSize(0.1)


@pytest.mark.parametrize(("widget", "said"), [("TControlBar", "window of buttons"),
                                              ("TSlider", "slider waiting to be dragged")])
def test_a_widget_that_is_only_interaction_is_refused(widget: str, said: str) -> None:
    with pytest.raises(UnsupportedFeatureError, match=said):
        getattr(ROOT, widget)("vertical", "bar")


def test_a_timer_keeps_what_it_is_told_and_never_fires() -> None:
    timer = ROOT.TTimer(20)
    timer.SetCommand("Animate()")
    timer.TurnOn()
    running = timer.IsRunning()
    timer.Stop()
    timer.Start(50)
    timer.Start()
    timer.TurnOff()
    other = ROOT.TTimer("Tick()", 10)
    other.SetObject(ROOT.TNamed("n", "n"))
    other.SetTime(5)
    expect((running, True), (timer.GetTime(), 50), (timer.IsRunning(), False),
           (timer.GetCommand(), "Animate()"), (other.GetCommand(), "Tick()"), (other.GetTime(), 5),
           (ROOT.TTimer(ROOT.TNamed("o", "o"), 3).GetTime(), 3))  # fmt: skip


def test_colours_are_named_as_tcolornumber_names_them() -> None:
    line = ROOT.TLine(0, 0, 1, 1)
    line.SetLineColor("r")
    shape = ROOT.TBRIK("B", "B", "void", 1, 1, 1)
    shape.SetLineColor("black")
    expect((line.GetLineColor(), 2), (shape.GetLineColor(), 1))


def test_a_name_the_namespace_lacks_is_what_groot_finds_by_it() -> None:
    ROOT.TGeometry("g", "g")
    shape = ROOT.TBRIK("YK01", "YK01", "void", 1, 1, 1)
    assert ROOT.YK01 is shape
    with pytest.raises(AttributeError, match="ROOT has _hidden"):
        ROOT._hidden  # noqa: B018
