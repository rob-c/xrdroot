"""Densities and functions drawn on a frame, held to the curves ROOT 6.40 drew.

Every curve below is ROOT's: its name, its number of points and its
height where ``RooCurve::interpolate`` put it, printed with ten figures
through PyROOT for the same model and the same events - generated after
``SetSeed(4357)``, which the engine reproduces bit for bit. The messages
are ROOT's too. Only the public ``plotOn`` is used, so that how the
engine gets there is its own business.
"""

from __future__ import annotations

from typing import Any

import pytest

from xrdroot.roofit.pdfs.addpdf import RooAddPdf
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.plot.pdfplot import NUM_EVENT, RAW, RELATIVE_EXPECTED
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooRealVar

#: Where the curves are read.
XS = (-7.3, -1.2, 0.4, 2.9, 8.8)
#: ``model`` drawn for the 300 events on the frame.
DEFAULT = [7.657572721, 28.93940264, 44.81222085, 23.06003285, 0.4972392925]
#: ... and drawn on an empty frame: for one event per bin width.
EMPTY = [0.0255252424, 0.09646467545, 0.1493740695, 0.07686677616, 0.001657464308]
#: The curves' heights are ROOT's to the ten figures it printed.
REL = 1e-9
#: What the engine is being changed to do: a failure here is expected until then, a pass welcome.
PENDING = {"strict": False}


def heights(curve: Any, xs: Any = XS) -> list[float]:
    return [float(curve.interpolate(x)) for x in xs]


class Model:
    """``f g1 + (1-f) g2`` in ``x``, and 300 events of it."""

    def __init__(self) -> None:
        self.x = RooRealVar("x", "x", 0, -10, 10)
        self.x.setBins(20)
        for name, low, high in (("win", -1.5, 2.5), ("a", -8, -2), ("b", 1, 4)):
            self.x.setRange(name, low, high)
        self.m1 = RooRealVar("m1", "m1", 1, -5, 5)
        self.s1 = RooRealVar("s1", "s1", 1.5, 0.1, 10)
        self.g1 = RooGaussian("g1", "g1", self.x, self.m1, self.s1)
        self.m2 = RooRealVar("m2", "m2", -2, -5, 5)
        self.s2 = RooRealVar("s2", "s2", 4, 0.1, 10)
        self.g2 = RooGaussian("g2", "g2", self.x, self.m2, self.s2)
        self.f = RooRealVar("f", "f", 0.4, 0, 1)
        self.model = RooAddPdf("model", "model", [self.g1, self.g2], [self.f])

    def data(self) -> Any:
        generator().SetSeed(4357)
        return self.model.generate([self.x], 300)

    def frame(self) -> Any:
        frame = self.x.frame()
        self.data().plotOn(frame)
        return frame


def test_a_density_on_an_empty_frame_is_drawn_for_one_event_per_bin_width() -> None:
    """With no data to scale to, the curve is the density times the bin width."""
    m = Model()
    frame = m.x.frame()
    m.model.plotOn(frame)
    curve = frame.getObject(0)
    assert (curve.GetName(), curve.GetN(), curve.ClassName()) == ("model_Norm[x]", 66, "RooCurve")
    assert heights(curve) == pytest.approx(EMPTY, rel=REL)
    assert frame.GetMaximum() == pytest.approx(0.16104531, rel=1e-7)
    assert curve._core["TAttLine"]["fLineWidth"] == 3
    assert curve._core["TAttLine"]["fLineColor"] == 600


def test_a_density_is_drawn_for_the_events_of_the_data_on_the_frame() -> None:
    """The data's 300 events times the bin width: curve and points share a scale."""
    m = Model()
    frame = m.frame()
    m.model.plotOn(frame)
    curve = frame.getObject(1)
    assert (curve.GetName(), curve.GetN()) == ("model_Norm[x]", 66)
    assert heights(curve) == pytest.approx(DEFAULT, rel=REL)


def test_components_are_drawn_by_name_by_set_and_by_pattern_as_root_says(capsys: Any) -> None:
    """``Components("g2")``, ``Components(g1)``, ``Components("g*")``: each said, each named."""
    m = Model()
    frame = m.frame()
    m.model.plotOn(frame, Components="g2", LineStyle=2, LineColor=2)
    m.model.plotOn(frame, Components=[m.g1], LineWidth=1)
    m.model.plotOn(frame, Components="g*")
    lines = [line for line in capsys.readouterr().out.splitlines() if "selected" in line]
    assert lines == [
        f"[#1] INFO:Plotting -- RooAbsPdf::plotOn(model) {kind} selected PDF components: ({found})"
        for found in ("g2", "g1", "g1,g2")
        for kind, found in (("directly", found), ("indirectly", ""))
    ]
    g2, g1, both = frame.getObject(1), frame.getObject(2), frame.getObject(3)
    assert (g2.GetName(), g2.GetN()) == ("model_Norm[x]_Comp[g2]", 55)
    assert heights(g2) == pytest.approx(
        [7.657562183, 18.02611531, 15.36075757, 8.691727786, 0.4862808391], rel=REL
    )
    assert (g1.GetName(), g1.GetN()) == ("model_Norm[x]_Comp[g1]", 68)
    assert heights(g1) == pytest.approx(
        [1.502111189e-05, 10.89697114, 29.44756312, 14.36430944, 0.0001361370663], rel=REL
    )
    assert (both.GetName(), both.GetN()) == ("model_Norm[x]_Comp[g*]", 66)
    assert heights(both) == pytest.approx(DEFAULT, rel=REL)
    assert g2._core["TAttLine"]["fLineStyle"] == 2
    assert g2._core["TAttLine"]["fLineColor"] == 2
    assert g1._core["TAttLine"]["fLineWidth"] == 1


def test_a_curve_is_scaled_by_a_factor_by_a_number_of_events_or_as_it_is() -> None:
    """``Normalization(0.5)`` halves it, ``NumEvent`` draws for so many events, ``Raw`` as given."""
    m = Model()
    frame = m.frame()
    m.model.plotOn(frame, Normalization=0.5)
    m.model.plotOn(frame, Normalization=(2.0, RAW))
    m.model.plotOn(frame, Normalization=(1000.0, NUM_EVENT))
    assert heights(frame.getObject(1)) == pytest.approx([v / 2 for v in DEFAULT], rel=REL)
    assert heights(frame.getObject(2)) == pytest.approx([2 * v for v in EMPTY], rel=REL)
    assert heights(frame.getObject(3)) == pytest.approx([1000 * v for v in EMPTY], rel=REL)


def extended() -> tuple[Model, RooAddPdf]:
    m = Model()
    n1, n2 = RooRealVar("n1", "n1", 120, 0, 1000), RooRealVar("n2", "n2", 180, 0, 1000)
    return m, RooAddPdf("ext", "ext", [m.g1, m.g2], [n1, n2])


def test_an_extended_density_is_drawn_for_the_events_it_expects() -> None:
    """``RelativeExpected``: the 300 events the yields add to, with or without data drawn."""
    m, ext = extended()
    generator().SetSeed(4357)
    data = ext.generate([m.x], 250)
    frame = m.x.frame()
    data.plotOn(frame)
    ext.plotOn(frame)
    ext.plotOn(frame, Normalization=(1.0, RELATIVE_EXPECTED))
    ext.plotOn(frame, Normalization=(500.0, NUM_EVENT))
    assert heights(frame.getObject(1)) == pytest.approx(DEFAULT, rel=REL)
    assert heights(frame.getObject(2)) == pytest.approx(DEFAULT, rel=REL)
    assert heights(frame.getObject(3)) == pytest.approx([500 * v for v in EMPTY], rel=REL)
    empty = m.x.frame()
    ext.plotOn(empty, Normalization=(1.0, RELATIVE_EXPECTED))
    assert heights(empty.getObject(0)) == pytest.approx(DEFAULT, rel=REL)


@pytest.mark.xfail(
    **PENDING,
    reason="pdfplot.py:121-128: on an empty frame ROOT draws what an extended pdf expects",
)
def test_an_extended_density_on_an_empty_frame_is_drawn_for_the_events_it_expects() -> None:
    """With nothing on the frame, ROOT draws an extended density for its 300 expected events."""
    m, ext = extended()
    empty = m.x.frame()
    ext.plotOn(empty)
    assert heights(empty.getObject(0)) == pytest.approx(DEFAULT, rel=REL)


def test_a_curve_over_a_range_of_numbers_is_normalised_to_the_data_there_or_to_all(
    capsys: Any,
) -> None:
    """``Range(-3, 3)`` draws the curve there, scaled to the data inside - or, asked, to all."""
    m = Model()
    frame = m.frame()
    capsys.readouterr()
    m.model.plotOn(frame, Range=(-3.0, 3.0))
    m.model.plotOn(frame, Range=(-3.0, 3.0, False))
    assert capsys.readouterr().out.splitlines() == [
        "[#1] INFO:Plotting -- RooAbsPdf::plotOn(model) only plotting range [-3,3], curve is "
        "normalized to data in given range",
        "[#1] INFO:Eval -- RooRealVar::setRange(x) new range named 'plotRange' created with "
        "bounds [-3,3]",
        "[#1] INFO:Plotting -- RooAbsPdf::plotOn(model) only plotting range [-3,3], curve is "
        "normalized to data in full range",
    ]
    given, full = frame.getObject(1), frame.getObject(2)
    assert (given.GetN(), full.GetN()) == (49, 49)
    assert heights(given, (-2.0, 0.4, 2.0)) == pytest.approx(
        [20.8429449, 41.10522358, 33.65678831], rel=REL
    )
    assert heights(full, (-2.0, 0.4, 2.0)) == pytest.approx(
        [22.72642838, 44.81971834, 36.69820137], rel=REL
    )
    assert (given.x[0], given.x[-1]) == (-3.0, 3.0)


def test_a_curve_over_several_named_ranges_is_one_curve_each_normalised_in_them_all(
    capsys: Any,
) -> None:
    """``Range("a,b")``: a curve over ``a`` and one over ``b``, both scaled to the data in both."""
    m = Model()
    frame = m.frame()
    capsys.readouterr()
    m.model.plotOn(frame, Range="a,b")
    assert capsys.readouterr().out.splitlines() == [
        "[#1] INFO:Plotting -- RooAbsPdf::plotOn(model) only plotting range 'a,b', curve is "
        "normalized to data in given range"
    ]
    a, b = frame.getObject(1), frame.getObject(2)
    assert (a.GetN(), b.GetN()) == (25, 25)
    assert heights(a, (-6.0, -3.0)) == pytest.approx([11.46280693, 19.26663727], rel=REL)
    assert heights(b, (1.5, 3.5)) == pytest.approx([43.87084413, 15.53527709], rel=REL)


def test_a_range_the_variable_does_not_have_is_said_and_passed_over(capsys: Any) -> None:
    """RooFit's error for an unknown range name, then the curve over the whole frame."""
    m = Model()
    frame = m.frame()
    capsys.readouterr()
    m.model.plotOn(frame, Range="nope")
    assert capsys.readouterr().out.splitlines() == [
        "[#0] ERROR:Plotting -- Range 'nope' not defined for variable 'x'. Ignoring ...",
        "[#1] INFO:Plotting -- RooAbsPdf::plotOn(model) only plotting range 'nope', curve is "
        "normalized to data in given range",
    ]


def test_a_named_range_and_a_normalisation_range_are_said_as_root_says(capsys: Any) -> None:
    """``Range("win")``, ``NormRange("win")`` and both: what ``plotOn`` says of each."""
    m = Model()
    frame = m.frame()
    capsys.readouterr()
    m.model.plotOn(frame, Range="win")
    m.model.plotOn(frame, Range="win", NormRange="win")
    m.model.plotOn(frame, NormRange="win")
    prefix = "[#1] INFO:Plotting -- RooAbsPdf::plotOn(model) "
    explicit = prefix + "p.d.f. curve is normalized using explicit choice of ranges 'win'"
    assert capsys.readouterr().out.splitlines() == [
        prefix + "only plotting range 'win', curve is normalized to data in given range",
        prefix + "only plotting range 'win'",
        explicit,
        explicit,
    ]
    assert [frame.getObject(i).GetN() for i in (1, 2, 3)] == [36, 36, 66]


@pytest.mark.xfail(
    **PENDING,
    reason="hist.py:94-101: ROOT counts the data in a range by the bin centres inside it",
)
def test_a_curve_over_a_named_range_is_normalised_to_the_bins_whose_centres_are_inside() -> None:
    """``RooHist::getFitRangeNEvt(-1.5, 2.5)`` sums the bins centred in the range: 5 bins here."""
    m = Model()
    frame = m.frame()
    m.model.plotOn(frame, Range="win")
    m.model.plotOn(frame, NormRange="win")
    assert heights(frame.getObject(1), (-1.0, 0.4, 2.0)) == pytest.approx(
        [32.72616768, 47.37205839, 38.79780814], rel=REL
    )
    assert heights(frame.getObject(2)) == pytest.approx(
        [8.092327774, 30.58242347, 47.3564134, 24.36925526, 0.5254698171], rel=REL
    )


@pytest.mark.xfail(
    **PENDING, reason="curves.py:221-224: ROOT's curve names say their Range and NormRange"
)
def test_a_curve_drawn_over_a_range_is_named_after_it_as_root_names_it() -> None:
    """``_Range[win]``, ``_Range[-3_3]``, ``_Range[win]_NormRange[win]``, ``_Range[a,b]``."""
    m = Model()
    frame = m.frame()
    m.model.plotOn(frame, Range="win")
    m.model.plotOn(frame, Range=(-3.0, 3.0))
    m.model.plotOn(frame, Range="win", NormRange="win")
    m.model.plotOn(frame, Range="a,b")
    assert [frame.nameOf(i) for i in range(1, 6)] == [
        "model_Norm[x]_Range[win]",
        "model_Norm[x]_Range[-3_3]",
        "model_Norm[x]_Range[win]_NormRange[win]",
        "model_Norm[x]_Range[a,b]",
        "model_Norm[x]_Range[a,b]",
    ]


@pytest.mark.xfail(
    **PENDING, reason="realplot.py:57, curve.py:75-85: VLines drops to zero at the range's ends"
)
def test_vertical_lines_close_a_ranged_curve_at_its_ends_as_root_draws_them() -> None:
    """``VLines``: a point a thousandth of a step outside each end, at zero - no wings."""
    m = Model()
    frame = m.frame()
    m.model.plotOn(frame, Range="win", VLines=True)
    curve = frame.getObject(1)
    assert curve.GetN() == 38
    assert (curve.x[0], curve.y[0]) == pytest.approx((-1.5002, 0.0))
    assert (curve.x[-1], curve.y[-1]) == pytest.approx((2.5002, 0.0))
