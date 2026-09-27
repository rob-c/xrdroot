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

from xrdroot.roofit.cmdargs import RooCmdArg
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


FIT_RANGE = (
    "[#1] INFO:Plotting -- RooAbsPdf::plotOn(model) p.d.f was fitted in a subrange and no "
    "explicit Range() and NormRange() was specified. Plotting / normalising in fit range. To "
    "override, do one of the following\n"
    '\t- Clear the automatic fit range attribute: <pdf>.removeStringAttribute("fitrange");\n'
    '\t- Explicitly specify the plotting range: Range("<rangeName>").\n'
    '\t- Explicitly specify where to compute the normalisation: NormRange("<rangeName>").\n'
    '\tThe default (full) range can be denoted with Range("") / NormRange("").\n'
)


def test_a_density_fitted_in_a_range_is_drawn_in_it_unless_told_otherwise(capsys: Any) -> None:
    """After ``fitTo(Range("win"))`` the curve is drawn and normalised in the fit's range."""
    m = Model()
    data = m.data()
    m.model.fitTo(data, Range="win", PrintLevel=-1)
    frame = m.x.frame()
    data.plotOn(frame)
    capsys.readouterr()
    m.model.plotOn(frame)
    prefix = "[#1] INFO:Plotting -- RooAbsPdf::plotOn(model) "
    assert capsys.readouterr().out == (
        FIT_RANGE
        + prefix
        + "only plotting range 'fit_nll_model_modelData'\n"
        + prefix
        + "p.d.f. curve is normalized using explicit choice of ranges 'fit_nll_model_modelData'\n"
    )
    assert frame.getObject(1).GetN() == 60
    m.model.plotOn(frame, Range="", NormRange="")
    full = frame.getObject(2)
    assert full.GetN() == 69
    assert heights(full) == pytest.approx(
        [10.06249305, 21.06118863, 30.84258269, 18.68275245, 6.602053066], rel=1e-6
    )
    m.model.removeStringAttribute("fitrange")
    capsys.readouterr()
    m.model.plotOn(frame)
    assert "fitted in a subrange" not in capsys.readouterr().out


def test_an_option_given_twice_is_warned_of_by_each_plot_on_that_reads_it(capsys: Any) -> None:
    """The last of a name wins, and both ``RooAbsPdf::plotOn`` and ``RooAbsReal::plotOn`` say so."""
    m = Model()
    frame = m.frame()
    capsys.readouterr()
    m.model.plotOn(frame, RooCmdArg("LineColor", 3), RooCmdArg("LineColor", 4))
    assert capsys.readouterr().out.splitlines() == [
        "[#0] WARNING:InputArguments -- RooAbsPdf::plotOn(model) WARNING: argument LineColor is "
        "duplicated",
        "[#0] WARNING:InputArguments -- RooAbsReal::plotOn(model) WARNING: argument LineColor is "
        "duplicated",
    ]
    assert frame.getObject(1)._core["TAttLine"]["fLineColor"] == 4


def test_a_curve_is_named_hidden_and_put_at_the_back_as_asked() -> None:
    """``Name``, ``Invisible`` and ``MoveToBack``: the frame's list, as ROOT's shows it."""
    m = Model()
    frame = m.frame()
    m.model.plotOn(frame)
    m.model.plotOn(frame, Name="twice", Invisible=True, MoveToBack=True)
    assert [frame.nameOf(i) for i in range(3)] == ["twice", "h_modelData", "model_Norm[x]"]
    assert frame.items[0][2] is True


def test_a_function_is_drawn_as_it_is_scaled_only_if_asked() -> None:
    """``RooAbsReal::plotOn``: ``x*x + 1`` as it is, twice it, and at a coarser precision."""
    from xrdroot.roofit.functions import RooFormulaVar

    m = Model()
    fx = RooFormulaVar("fx", "fx", "x*x+1", [m.x])
    frame = m.x.frame()
    fx.plotOn(frame)
    fx.plotOn(frame, Precision=1e-2, DrawOption="F", FillColor=5)
    plain, coarse = frame.getObject(0), frame.getObject(1)
    assert (plain.GetN(), coarse.GetN()) == (46, 26)
    assert heights(plain) == pytest.approx([54.35, 2.5, 1.2, 9.45, 78.5], rel=REL)
    assert heights(coarse) == pytest.approx([54.5, 2.6, 1.4, 9.5, 78.6], rel=REL)
    assert frame.GetMaximum() == pytest.approx(106.05, rel=REL)
    assert frame.items[1][1] == "F"
    assert coarse._core["TAttFill"]["fFillColor"] == 5
    fx.plotOn(frame, Normalization=2.0, Range=(-2.0, 2.0))
    assert frame.getObject(2).GetN() == 42


@pytest.mark.xfail(**PENDING, reason="curves.py:221: ROOT names a function's curve fx_Norm[x]")
def test_a_functions_curve_is_named_for_its_frames_variable() -> None:
    """ROOT calls the curve of ``fx`` drawn on ``x`` ``fx_Norm[x]``."""
    from xrdroot.roofit.functions import RooFormulaVar

    m = Model()
    fx = RooFormulaVar("fx", "fx", "x*x+1", [m.x])
    frame = m.x.frame()
    fx.plotOn(frame)
    assert frame.nameOf(0) == "fx_Norm[x]"


@pytest.mark.xfail(
    **PENDING, reason="realplot.py:58-59: ROOT divides a ranged function by its integral there"
)
def test_a_function_drawn_over_a_range_is_divided_by_its_integral_there() -> None:
    """``Normalization(2)`` over ``[-2, 2]``: twice ``x*x + 1`` over its integral there, 28/3."""
    from xrdroot.roofit.functions import RooFormulaVar

    m = Model()
    fx = RooFormulaVar("fx", "fx", "x*x+1", [m.x])
    frame = m.x.frame()
    fx.plotOn(frame, Normalization=2.0, Range=(-2.0, 2.0))
    assert heights(frame.getObject(0), (-1.5, 0.4, 1.5)) == pytest.approx(
        [0.6964285714, 0.2485714286, 0.6964285714], rel=REL
    )


class Fitted:
    """``f g1 + (1-f) e`` fitted to 300 of its events: the fit a band is drawn from."""

    def __init__(self) -> None:
        from xrdroot.roofit.pdfs.basic import RooExponential

        self.x = RooRealVar("x", "x", 0, -10, 10)
        self.x.setBins(20)
        self.m1 = RooRealVar("m1", "m1", 1, -5, 5)
        self.s1 = RooRealVar("s1", "s1", 1.5, 0.1, 10)
        g1 = RooGaussian("g1", "g1", self.x, self.m1, self.s1)
        e = RooExponential("e", "e", self.x, RooRealVar("c", "c", -0.2, -2, -0.01))
        self.f = RooRealVar("f", "f", 0.4, 0, 1)
        self.model = RooAddPdf("model", "model", [g1, e], [self.f])
        generator().SetSeed(4357)
        self.data = self.model.generate([self.x], 300)
        self.result = self.model.fitTo(self.data, PrintLevel=-1, Save=True)
        self.frame = self.x.frame()
        self.data.plotOn(self.frame)


def band_points(curve: Any) -> list[float]:
    """The band's heights at the points ROOT's were printed at: out along the top, back below."""
    n = curve.GetN()
    return [float(curve.y[i]) for i in (5, n // 4, (3 * n) // 4)]


def test_an_error_band_is_the_linear_propagation_of_the_fits_errors_as_root_draws_it() -> None:
    """``VisualizeError(fit)``: ``sqrt(F C F)`` round the curve, out and back, filled cyan."""
    fit = Fitted()
    assert fit.f.getVal() == pytest.approx(0.446675644201, rel=1e-9)
    fit.model.plotOn(fit.frame, VisualizeError=fit.result)
    band = fit.frame.getObject(1)
    assert (band.GetName(), band.GetN()) == ("model_Norm[x]_errorband", 160)
    assert (band.x[0], band.x[80], band.x[159]) == pytest.approx((-11.001, 11.001, -11.001))
    assert band_points(band) == pytest.approx([31.465995, 36.9911, 30.740347], rel=1e-7)
    assert fit.frame.items[1][1] == "F"
    assert band._core["TAttFill"]["fFillColor"] == 432
    assert band._core["TAttLine"]["fLineWidth"] == 1


def test_an_error_band_of_some_parameters_at_some_sigmas_is_named_and_coloured_as_asked() -> None:
    """``VisualizeError(fit, f, 2)``: only ``f`` moved, by two of its errors."""
    fit = Fitted()
    fit.model.plotOn(
        fit.frame, VisualizeError=(fit.result, [fit.f], 2.0), FillColor=3, Name="fband"
    )
    band = fit.frame.getObject(1)
    assert band.GetName() == "fband"
    assert band_points(band) == pytest.approx([33.267369, 39.243247, 28.536645], rel=1e-7)
    assert band._core["TAttFill"]["fFillColor"] == 3


def test_an_error_band_beyond_a_parameters_range_is_clipped_and_warned_of(capsys: Any) -> None:
    """Three sigmas of ``m1`` leave ``[1, 1.5]``: RooFit says so and clips the variations."""
    fit = Fitted()
    fit.m1.setRange(1.0, 1.5)
    capsys.readouterr()
    fit.model.plotOn(fit.frame, VisualizeError=(fit.result, [fit.m1], 3.0))
    assert (
        "[#0] WARNING:Plotting -- RooAbsReal::plotOn(model): the 3-sigma error band for the "
        'parameter "m1" is invalid because the variations (0.67137, 1.80919) are outside the '
        "defined range [1, 1.5]!\n                         The variations will be clipped inside "
        "the range. This might or might not be acceptable in your usecase.\n"
    ) in capsys.readouterr().out
    assert band_points(fit.frame.getObject(1)) == pytest.approx(
        [28.805649, 34.976239, 32.398993], rel=1e-7
    )


def test_a_sampled_error_band_takes_the_central_quantiles_of_curves_of_drawn_parameters(
    capsys: Any,
) -> None:
    """``VisualizeError(fit, 0.1, False)``: 108 parameter sets drawn from the fit, as ROOT draws."""
    fit = Fitted()
    capsys.readouterr()
    generator().SetSeed(4357)
    fit.model.plotOn(fit.frame, VisualizeError=(fit.result, 0.1, False), MoveToBack=True)
    assert (
        "[#1] INFO:Plotting -- RooAbsReal::plotOn(model) INFO: visualizing 0.1-sigma uncertainties "
        "in parameters (m1,s1,f,c) from fit result fitresult_model_modelData using 108 samplings."
    ) in capsys.readouterr().out
    band = fit.frame.getObject(0)
    assert band.GetName() == "model_Norm[x]_errorband"
    assert band_points(band) == pytest.approx([28.26745, 33.892244, 34.095749], rel=1e-7)
    assert fit.frame.GetMaximum() == pytest.approx(44.07103, rel=1e-7)


@pytest.mark.xfail(
    **PENDING,
    reason="pdfplot.py:148 strips Components before band.py re-plots: ROOT's band is the part's",
)
def test_an_error_band_of_a_component_is_drawn_round_that_component() -> None:
    """``VisualizeError`` with ``Components("e")``: 72 points round ``e``, named as ROOT does."""
    fit = Fitted()
    fit.model.plotOn(fit.frame, VisualizeError=(fit.result, 1.0, True), Components="e")
    band = fit.frame.getObject(1)
    assert (band.GetName(), band.GetN()) == ("model_Norm[x]_Comp[e]_errorband_Comp[e]", 72)
    assert band.y[5] == pytest.approx(31.470797, rel=1e-7)


class Channels:
    """A Gaussian for ``phys`` and an exponential for ``ctl``, and 120 and 80 events of them."""

    def __init__(self) -> None:
        from xrdroot.roofit.categories import RooCategory
        from xrdroot.roofit.data.dataset import RooDataSet
        from xrdroot.roofit.pdfs.basic import RooExponential
        from xrdroot.roofit.pdfs.simultaneous import RooSimultaneous

        self.x = RooRealVar("x", "x", 0, -10, 10)
        self.x.setBins(20)
        self.g1 = RooGaussian(
            "g1", "g1", self.x, RooRealVar("m1", "m1", 1), RooRealVar("s1", "", 1.5)
        )
        self.e = RooExponential("e", "e", self.x, RooRealVar("c", "c", -0.2, -2, -0.01))
        self.cat = RooCategory("cat", "cat", {"phys": 0, "ctl": 1})
        self.sim = RooSimultaneous("sim", "sim", {"phys": self.g1, "ctl": self.e}, self.cat)
        generator().SetSeed(4357)
        phys, ctl = self.g1.generate([self.x], 120), self.e.generate([self.x], 80)
        self.data = RooDataSet(
            "combined", "combined", [self.x], Index=self.cat, Import={"phys": phys, "ctl": ctl}
        )
        self.frame = self.x.frame()
        self.data.plotOn(self.frame)


AVERAGES = (
    "[#1] INFO:Plotting -- RooSimultaneous::plotOn(sim) plot on x averages with data index "
    "category (cat)"
)
#: ``e``'s part of the channels' average: 80 of the 200 events.
CONTROL = [9.50925692, 2.81342429, 2.045829609, 1.237351128, 0.3807555732]


def test_a_simultaneous_density_is_drawn_as_its_channels_averaged_as_the_data_weigh_them(
    capsys: Any,
) -> None:
    """``ProjWData(cat, data)``: 120/200 of ``g1`` and 80/200 of ``e``, each normalised."""
    c = Channels()
    capsys.readouterr()
    c.sim.plotOn(c.frame, ProjWData=([c.cat], c.data))
    assert capsys.readouterr().out.splitlines() == [AVERAGES, AVERAGES]
    curve = c.frame.getObject(1)
    assert (curve.GetName(), curve.GetN()) == ("sim_Norm[x]", 74)
    assert heights(curve) == pytest.approx(
        [9.509267459, 13.70125737, 31.48384171, 15.60166056, 0.3808917102], rel=REL
    )


def test_a_component_of_a_simultaneous_density_is_drawn_as_its_share_of_the_average() -> None:
    """``Components("e")`` with ``ProjWData``: the control channel's part of the average."""
    c = Channels()
    c.sim.plotOn(c.frame, ProjWData=([c.cat], c.data), Components="e")
    curve = c.frame.getObject(1)
    assert (curve.GetName(), curve.GetN()) == ("sim_Norm[x]_Comp[e]", 36)
    assert heights(curve) == pytest.approx(CONTROL, rel=REL)


@pytest.mark.xfail(
    strict=True,
    reason="simultaneous.py:163-164: ROOT refuses to draw a simultaneous density without "
    "projection data for its index category",
)
def test_a_simultaneous_density_without_projection_data_is_refused_as_root_refuses_it(
    capsys: Any,
) -> None:
    """ROOT says it must have a projection dataset for the index category, and draws nothing."""
    c = Channels()
    capsys.readouterr()
    c.sim.plotOn(c.frame)
    assert capsys.readouterr().out == (
        "[#0] ERROR:InputArguments -- RooSimultaneous::plotOn(sim) ERROR: must have a projection "
        "dataset for index category\n"
    )
    assert c.frame.numItems() == 1


@pytest.mark.xfail(
    strict=True,
    reason="simultaneous.py:182,188-189: ROOT draws a slice as its channel, e_Norm[x], weighted "
    "by that channel's share of the projection data",
)
def test_a_slice_of_a_simultaneous_density_is_its_channel_weighted_by_its_share(
    capsys: Any,
) -> None:
    """``Slice(cat, "ctl")``: ROOT draws ``e`` for its 80 of the 200 events, and says so."""
    c = Channels()
    capsys.readouterr()
    c.sim.plotOn(c.frame, Slice=(c.cat, "ctl"), ProjWData=([c.cat], c.data))
    assert capsys.readouterr().out.splitlines() == [
        "[#1] INFO:Plotting -- RooSimultaneous::plotOn(sim) plot on x represents a slice in the "
        "index category (cat)",
        "[#1] INFO:Plotting -- RooAbsReal::plotOn(e) slice variable cat was not projected anyway",
    ]
    curve = c.frame.getObject(1)
    assert curve.GetName() == "e_Norm[x]"
    assert heights(curve) == pytest.approx(CONTROL, rel=REL)


def test_a_densitys_parameters_are_boxed_on_the_frame_as_root_writes_them() -> None:
    """``paramOn``: one line per free parameter, ``RooRealVar::format(2, "NELU")``."""
    from xrdroot.roofit.plot import params

    x = RooRealVar("x", "x", 0, -10, 10)
    x.setBins(20)
    m1, s1 = RooRealVar("m1", "mean", 1, -5, 5), RooRealVar("s1", "#sigma", 1.5, 0.1, 10)
    g1 = RooGaussian("g1", "g1", x, m1, s1)
    generator().SetSeed(4357)
    data = g1.generate([x], 100)
    g1.fitTo(data, PrintLevel=-1)
    frame = x.frame()
    data.plotOn(frame)
    g1.paramOn(frame)
    g1.paramOn(frame, Layout=(0.55, 0.95, 0.8), Label="fit\nresult", ShowConstants=True)
    g1.paramOn(frame, RooCmdArg("Parameters", [m1]), Format=("NE", RooCmdArg("AutoPrecision", 1)))
    plain, labelled, chosen = (frame.getObject(i) for i in (1, 2, 3))
    assert plain.GetName() == "g1_paramBox"
    assert plain.ClassName() == "TPaveText"
    assert plain.lines == ["m1 =  0.95 #pm 0.14", "s1 =  1.45 #pm 0.10"]
    assert labelled.lines == ["m1 =  0.95 #pm 0.14", "s1 =  1.45 #pm 0.10", "fit", "result"]
    assert chosen.lines == ["m1 =  0.9 +/- 0.1"]
    corners = [sorted(one.corners[i] for i in (1, 3)) for one in (plain, labelled, chosen)]
    assert [v for pair in corners for v in pair] == pytest.approx([0.78, 0.9, 0.56, 0.8, 0.84, 0.9])
    assert (plain.corners[0], plain.corners[2], labelled.corners[0]) == (0.65, 0.9, 0.55)
    assert plain.attributes == {
        "FillColor": 0,
        "BorderSize": 0,
        "TextAlign": 12,
        "TextSize": 0.04,
        "FillStyle": 0,
    }
    assert params.PAVE[0] is params.Pave
    with pytest.raises(AttributeError):
        plain.GetNothing  # noqa: B018
