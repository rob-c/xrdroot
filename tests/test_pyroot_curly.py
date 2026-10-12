"""``TCurlyLine`` and ``TCurlyArc``: gluons as loops, photons as waves, built in pixels."""

from __future__ import annotations

import math

import pytest

import xrdroot.pyroot as ROOT
from pyrootgraphics import fresh_session  # noqa: F401
from xrdroot.canvas.curly import _straight
from xrdroot.pyroot.graphics import snapshot


def test_a_curly_line_is_loops_or_a_wave_of_roots_default_size():
    line = ROOT.TCurlyLine(0, 0, 1, 1)
    assert line.GetCurly() and (line.GetWaveLength(), line.GetAmplitude()) == (0.02, 0.01)
    line.SetWavy()
    assert not line.IsCurly()
    line.SetCurly()
    line.SetStartPoint(0.1, 0.2)
    line.SetEndPoint(0.3, 0.4)
    assert (line.GetX1(), line.GetY2(), line.GetFillColor()) == (0.1, 0.4, 19)
    ROOT.TCurlyLine.SetDefaultWaveLength(0.05)
    ROOT.TCurlyLine.SetDefaultAmplitude(0.02)
    ROOT.TCurlyLine.SetDefaultIsCurly(False)
    try:
        made = ROOT.TCurlyLine(0, 0, 1, 1)
        assert (made.GetWaveLength(), made.GetAmplitude(), made.IsCurly()) == (0.05, 0.02, False)
        assert (ROOT.TCurlyLine.GetDefaultWaveLength(), ROOT.TCurlyLine.GetDefaultAmplitude(),
                ROOT.TCurlyLine.GetDefaultIsCurly()) == (0.05, 0.02, False)  # fmt: skip
    finally:
        ROOT.TCurlyLine.SetDefaultWaveLength(0.02)
        ROOT.TCurlyLine.SetDefaultAmplitude(0.01)
        ROOT.TCurlyLine.SetDefaultIsCurly(True)


def test_an_arc_keeps_its_centre_radius_and_angles():
    arc = ROOT.TCurlyArc(30, 30, 12.5, 135, 225)
    arc.SetRadius(10)
    arc.SetCenter(1, 2)
    assert (arc.GetRadius(), arc.GetX1(), arc.GetY1(), arc.GetPhimin(), arc.GetPhimax()) == (
        10, 1, 2, 135, 225)  # fmt: skip


def test_the_wave_is_forty_steps_a_period_between_two_stretches_as_root_builds_it():
    points = _straight(105.0, 10.0, 2.0, curly=False)
    assert len(points) == 40 * 9 + 20 + 4 and points[0] == (0.0, 0.0) and points[-1] == (105, 0)
    assert points[1] == (2.5, 0.0) and points[2] == pytest.approx((4.5, 0.0), abs=1e-12)
    looped = _straight(105.0, 10.0, 2.0, curly=True)
    assert looped[2][0] == pytest.approx(2.5)  # swung back by the amplitude, at phase 3 pi / 2
    assert len(_straight(5.0, 0.0, 1.0, curly=True)) == 24  # no period: half of one, and its ends


def test_lines_and_arcs_are_drawn_through_their_built_points():
    c = ROOT.TCanvas("fey", "fey", 600, 300)
    c.Range(0, 0, 140, 60)
    gluon = ROOT.TCurlyArc(110, 30, 12.5 * math.sqrt(2), 315, 45)
    shapes = [ROOT.TCurlyLine(30, 30, 55, 30), gluon, ROOT.TCurlyArc(70, 30, 10, 0, 360),
              ROOT.TCurlyArc(70, 30, 0, 0, 90)]  # fmt: skip
    shapes[0].SetWavy()
    for one in shapes:
        one.Draw()
    made = snapshot.model(c)
    assert [prim.classname for prim, _ in made.primitives] == [
        "TCurlyLine", "TCurlyArc", "TCurlyArc", "TCurlyArc"]  # fmt: skip
    assert made.plot().axes
