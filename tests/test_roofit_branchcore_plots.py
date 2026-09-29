"""RooFit's plots as RooStats draws on them: wing modes, a function's curve, a histogram added
to a frame and a dataset's box of statistics.

Every number and label was printed by ROOT 6.40 through PyROOT for the same
calls: ``VLines`` closes a curve a thousandth of a step outside each end, a
function's curve is ``f_Norm[x]``, ``addTH1`` draws ``SAME`` and makes room on
the y axis, and ``statOn`` lists RMS, mean and entries as ``paramOn`` would.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from xrdroot.pyroot import TH1D
from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.collections import RooArgList, RooArgSet
from xrdroot.roofit.data.dataset import RooDataSet
from xrdroot.roofit.functions import RooFormulaVar
from xrdroot.roofit.messages import service
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.plot.curve import EXTENDED, NO_WINGS, STRAIGHT, sample
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooRealVar


@pytest.fixture(autouse=True)
def _messages() -> Iterator[None]:
    """Every test starts from the message streams RooFit starts with."""
    service().reset()
    yield
    service().reset()


def gaussian() -> tuple[RooRealVar, RooGaussian]:
    x = RooRealVar("x", "x", 0, -5, 5)
    return x, RooGaussian("g", "g", x, RooRealVar("m", "m", 0), RooRealVar("s", "s", 1))


def events() -> tuple[RooRealVar, RooDataSet]:
    x, g = gaussian()
    generator().SetSeed(4357)
    return x, g.generate(RooArgSet(x), 50)


def test_the_wing_modes_add_roofits_phantom_points_at_the_ends() -> None:
    """``Extended`` steps out a whole step, ``Straight`` a thousandth, ``NoWings`` not at all."""
    flat = lambda xs: np.ones_like(xs)  # noqa: E731
    xs, ys = sample(flat, 0.0, 1.0, 2, wings=NO_WINGS)
    assert (list(xs), list(ys)) == ([0.0, 0.5, 1.0, 1.0], [1.0] * 4)  # the end twice, as ROOT
    xs, ys = sample(flat, 0.0, 1.0, 2, wings=STRAIGHT)
    assert (xs[0], ys[0], xs[-1], ys[-1]) == pytest.approx((-0.0005, 0.0, 1.0005, 0.0))
    xs, ys = sample(flat, 0.0, 1.0, 2, wings=EXTENDED)
    assert list(xs[:2]) == pytest.approx([-0.5005, -0.5])
    assert list(ys[:2]) == [0.0, 1.0]


def test_vertical_lines_close_a_curve_a_thousandth_of_a_step_outside_as_root_does() -> None:
    """``VLines`` over ten bins of ``[-5, 5]``: 68 points, ``(-5.001, 0)`` to ``(5.001, 0)``."""
    x, g = gaussian()
    frame = x.frame(RooCmdArg("Bins", 10))
    g.plotOn(frame, RooCmdArg("VLines"))
    curve = frame.getObject(0)
    assert (curve.GetName(), curve.GetN()) == ("g_Norm[x]", 68)
    assert (curve.GetPointX(0), curve.GetPointY(0)) == pytest.approx((-5.001, 0.0))
    assert curve.GetPointY(1) == pytest.approx(1.4867203670757582e-06, rel=1e-9)
    assert (curve.GetPointX(67), curve.GetPointY(67)) == pytest.approx((5.001, 0.0))


def test_a_functions_curve_is_named_as_normalised_over_the_frames_variable() -> None:
    """A ``RooFormulaVar`` plotted is ``f_Norm[x]`` - normalised in name only."""
    x, _ = gaussian()
    frame = x.frame()
    RooFormulaVar("f", "f", "x*x", RooArgList(x)).plotOn(frame)
    assert frame.getObject(0).GetName() == "f_Norm[x]"


def test_a_histogram_added_to_a_frame_is_drawn_same_with_room_made_for_it() -> None:
    """``addTH1(h)`` draws ``SAME`` - an option given too - and the y axis reaches 1.05 times
    the highest bin, titled as the histogram's y axis is."""
    x, _ = gaussian()
    h = TH1D("hbc", "h;x;counts", 10, -5, 5)
    h.Fill(0.1, 7)
    frame = x.frame()
    frame.addTH1(h)
    assert frame.getDrawOptions("hbc") == "SAME"
    assert (frame.GetMaximum(), frame.GetMinimum()) == pytest.approx((7.35, 0.0))
    assert frame.GetYaxis().GetTitle() == "counts"
    frame.addTH1(h, "hist same")
    assert frame.items[-1][1] == "HIST SAME"


def test_anything_else_added_as_a_histogram_leaves_the_axis_alone() -> None:
    """An object without a histogram's extremes is only added, with ``SAME``."""
    x, _ = gaussian()
    frame = x.frame()

    class Marker:
        def GetName(self) -> str:
            return "mark"

    frame.addTH1(Marker(), "", True)
    assert frame.items[-1][1:] == ("SAME", True)


class Extremes:
    """A histogram-like object with extremes but no axes."""

    def GetName(self) -> str:
        return "ext"

    def GetMinimum(self) -> float:
        return 1.0

    def GetMaximum(self) -> float:
        return 2.0


def test_a_histogram_without_axes_makes_room_without_a_title() -> None:
    """Its extremes still widen the frame's y axis."""
    x, _ = gaussian()
    frame = x.frame()
    frame.addTH1(Extremes())
    assert frame.GetMaximum() == pytest.approx(2.1)


def lines_of(box: Any) -> list[str]:
    return [line.GetTitle() for line in box.GetListOfLines()]


def test_stat_on_lists_rms_mean_and_entries_in_a_box_as_root_does() -> None:
    """``statOn(frame)``: ``gData_statBox``, R before M before N, each formatted as ``paramOn``."""
    x, data = events()
    frame = x.frame()
    assert data.statOn(frame) is frame
    box = frame.getObject(0)
    assert box.GetName() == "gData_statBox"
    assert lines_of(box) == ["RMS =  0.946 #pm 0.095", "Mean =  0.02 #pm 0.13", "Entries =  50"]
    assert (box.GetX1NDC(), box.GetX2NDC()) == (0.65, 0.99)
    assert (box.GetY1NDC(), box.GetY2NDC()) == pytest.approx((0.77, 0.95))


def test_stat_on_takes_what_label_format_and_layout() -> None:
    """``What("N")``, a label and ``Format("NE", AutoPrecision(1))``: ``Entries =  50.0``."""
    x, data = events()
    frame = x.frame()
    auto = RooCmdArg("AutoPrecision", 1)
    data.statOn(frame, RooCmdArg("What", "N"), RooCmdArg("Label", "lab"),
                RooCmdArg("Format", "NE", auto), RooCmdArg("Layout", 0.1, 0.4, 0.8))  # fmt: skip
    box = frame.getObject(0)
    assert lines_of(box) == ["Entries =  50.0", "lab"]
    assert (box.GetY1NDC(), box.GetY2NDC()) == pytest.approx((0.68, 0.8))


def test_stat_on_counts_only_the_events_a_cut_keeps() -> None:
    """``Cut("x>0")`` restricts every statistic to the events above zero."""
    x, data = events()
    frame = x.frame()
    data.statOn(frame, RooCmdArg("Cut", "x>0"), RooCmdArg("What", "N"))
    kept = int(sum(1 for i in range(50) if data.get(i).getRealValue("x") > 0))
    assert lines_of(frame.getObject(0)) == [f"Entries =  {kept}"]


def test_the_mean_and_rms_variables_carry_roofits_names_labels_and_errors() -> None:
    """``meanVar`` is ``xMean``, ``<x>``, its error the RMS over root N; ``rmsVar`` is ``xRMS``."""
    x, data = events()
    mean, rms = data.meanVar(x), data.rmsVar(x)
    assert (mean.GetName(), mean.GetTitle(), mean.getPlotLabel()) == ("xMean", "Mean of x", "<x>")
    assert (mean.getVal(), mean.getError()) == pytest.approx(
        (0.024091148141621724, 0.13374921915200208), rel=1e-12
    )
    assert (rms.GetName(), rms.GetTitle(), rms.getPlotLabel()) == (
        "xRMS", "RMS         of x", "x_{RMS}"
    )  # fmt: skip
    assert (rms.getVal(), rms.getError()) == pytest.approx(
        (0.9457497984078633, 0.09457497984078633), rel=1e-12
    )


def test_the_mean_of_no_events_has_no_error() -> None:
    """A cut that keeps nothing leaves the mean's error at zero."""
    x, data = events()
    with np.errstate(all="ignore"):
        mean = data.meanVar(x, "x>100")
    assert mean.getError() == 0.0
