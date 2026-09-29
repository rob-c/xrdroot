"""``TRatioPlot``: the pads it lays out, what it works out for the lower one, and its axes.

The geometry - pads, margins, where each ``TGaxis`` is and what it spans -
and the numbers are what ROOT 6.40 made of the ``hist029`` to ``hist034``
tutorials, printed there from the objects ``TRatioPlot`` made. ROOT keeps
margins and fractions as ``Float_t``, so those are single precision here too.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from pyrootgraphics import fresh_session  # noqa: F401

#: ``0.1f``, ``0.3f``, ``0.05f`` and ``0.9f + ...``: ROOT's single-precision fractions.
TENTH, SPLIT, TWENTIETH = 0.10000000149011612, 0.30000001192092896, 0.05000000074505806


@pytest.fixture(autouse=True)
def session():
    """ROOT's first draws, and no fit yet, as in a fresh ROOT."""
    from xrdroot.pyroot.core import fitters

    ROOT.gRandom.SetSeed(4357)
    fitters.LATEST["result"] = None
    ROOT.gStyle.SetOptStat(0)


def fitted(entries=2000, option="0"):
    """``hist031``'s histogram: a Gaussian's draws, fitted with ``gaus``."""
    h1 = ROOT.TH1D("h1", "h1", 50, -5, 5)
    h1.FillRandom("gaus", entries)
    h1.Fit("gaus", option + "Q")
    h1.GetXaxis().SetTitle("x")
    h1.GetYaxis().SetTitle("y")
    return h1


def margin_plot():
    """``hist034``: fitted, errors from the function, the pads touching, mirrored axes."""
    c1 = ROOT.TCanvas("c1", "fit residual simple")
    ROOT.gPad.SetFrameFillStyle(0)
    h1 = fitted(5000, "S")
    h1.Sumw2()
    rp = ROOT.TRatioPlot(h1, "errfunc")
    rp.SetGraphDrawOpt("L")
    rp.SetSeparationMargin(0.0)
    rp.Draw()
    rp.GetLowerRefGraph().SetMinimum(-2)
    rp.GetLowerRefGraph().SetMaximum(2)
    c1.Update()
    return c1, rp


def members(pad, *names):
    return tuple(pad.members[f"f{name}"] for name in names)


def ends(gaxis):
    return members(gaxis, "X1", "Y1", "X2", "Y2", "Wmin", "Wmax", "Chopt")


def test_the_pads_split_the_canvas_at_three_tenths_less_rootss_inset():
    _, rp = margin_plot()
    upper, lower = rp.GetUpperPad(), rp.GetLowerPad()
    assert members(upper, "YlowNDC", "BottomMargin", "TopMargin") == (SPLIT, 0.0, TENTH)
    # ROOT's pad heights come back through its pixels, so hold them to a part in 10^9.
    assert upper.members["fHNDC"] == pytest.approx(0.6974999881349504, abs=1e-9)
    assert members(lower, "YlowNDC", "BottomMargin", "TopMargin") == (0.0025, SPLIT, 0.0)
    assert lower.members["fXlowNDC"] == 0.0025 and lower.members["fHNDC"] == pytest.approx(
        0.29750001197680831, abs=1e-9
    )
    assert rp.GetSeparationMargin() == 0.0


def test_the_upper_and_lower_pads_draw_the_fit_its_residuals_and_its_bands():
    _, rp = margin_plot()
    drawn = [(type(obj).__name__, option) for obj, option in rp.GetLowerPad().primitives]
    assert drawn == [("TGraphErrors", "IA3"), ("TGraphErrors", "3"), ("TGraphAsymmErrors", "LSAME"),
                     ("TLine", ""), ("TLine", ""), ("TLine", "")]  # fmt: skip
    above = [(type(obj).__name__, option) for obj, option in rp.GetUpperPad().primitives]
    assert above == [("TH1D", "AE"), ("TF1", "same")]
    lines = [obj for obj, _ in rp.GetLowerPad().primitives if type(obj).__name__ == "TLine"]
    assert [members(line, "X1", "Y1", "X2", "Y2") for line in lines] == [
        (-5, 1, 5, 1), (-5, 0, 5, 0), (-5, -1, 5, -1)]  # fmt: skip
    assert (rp.GetLowerPad().GetUymin(), rp.GetLowerPad().GetUymax()) == (-2, 2)


def test_the_axes_are_drawn_over_both_pads_where_roots_are():
    _, rp = margin_plot()
    axes = rp._axes
    assert ends(axes["upper_x"]) == (TENTH, SPLIT, 0.8999999985098839, SPLIT, -5, 5, "+U")
    assert ends(axes["lower_x"]) == (TENTH, 0.09000000715255752, 0.8999999985098839,
                                     0.09000000715255752, -5, 5, "+S")  # fmt: skip
    assert ends(axes["lower_y"]) == (TENTH, 0.09000000715255752, TENTH, SPLIT, -2, 2, "-S")
    upper_y = ends(axes["upper_y"])
    assert upper_y[:4] == (TENTH, SPLIT, TENTH, 0.9300000001490116)
    assert upper_y[5] == pytest.approx(438.84743421042299, rel=1e-9)
    assert (axes["lower_y"].members["fTickSize"], axes["upper_x"].members["fLabelSize"]) == (
        0.08999999612569809, 0.0)  # fmt: skip
    assert (axes["lower_x"].GetTitle(), axes["upper_y"].GetTitle()) == ("x", "y")
    assert sorted(rp._mirrors) == ["lower_x", "lower_y", "upper_x", "upper_y"]
    assert ends(rp._mirrors["upper_x"])[:4] == (TENTH, 0.9300000001490116, 0.8999999985098839,
                                                0.9300000001490116)  # fmt: skip


def test_the_first_residual_is_the_bin_less_the_fit_over_the_functions_root():
    _, rp = margin_plot()
    graph = rp.GetCalculationOutputGraph()
    h1 = rp.GetUpperRefObject()
    fx = h1.GetFunction("gaus").Eval(-4.9)
    assert graph.GetN() == 50 and graph.GetX()[0] == pytest.approx(-4.9)
    assert graph.GetY()[0] == pytest.approx((h1.GetBinContent(1) - fx) / fx**0.5)
    assert (graph.GetErrorXlow(0), graph.GetErrorYhigh(0)) == pytest.approx((0.1, 0.5))
    band = rp.GetConfidenceInterval1()
    assert band.GetN() == 50 and band.GetY()[0] == 0 and band.GetErrorX(0) == pytest.approx(-4.9)
    assert rp.GetConfidenceInterval2().GetErrorY(0) > band.GetErrorY(0) > 0


def test_where_the_pads_touch_the_lower_axiss_top_label_is_hidden():
    _, rp = margin_plot()
    assert rp._axes["lower_y"].members["_changed_labels"] == [(-1, -1.0, 0.0, -1, -1, -1, "")]


def two(option=None, stack=None):
    """``hist029``'s pair: two draws of a falling exponential, the second scaled to the first."""
    canvas = ROOT.TCanvas("C", "A ratio example")
    h1 = ROOT.TH1D("h1", "TRatioPlot Example; x; y", 50, 0, 10)
    h2 = ROOT.TH1D("h2", "h2", 50, 0, 10)
    f1 = ROOT.TF1("f1", "exp(- x/[0] )")
    f1.SetParameter(0, 3)
    h1.FillRandom("f1", 1900)
    h2.FillRandom("f1", 2000)
    h1.Sumw2()
    h2.Scale(1.9 / 2.0)
    args = (h1, h2) if option is None else (h1, h2, option)
    return canvas, ROOT.TRatioPlot(*args), h1, h2


def test_two_histograms_are_divided_as_poisson_counts_with_rootss_first_points():
    canvas, rp, _, _ = two()
    canvas.SetTicks(0, 1)
    rp.Draw()
    rp.GetLowYaxis().SetNdivisions(505)
    canvas.Update()
    graph = rp.GetCalculationOutputGraph()
    first = (graph.GetY()[0], graph.GetErrorYlow(0), graph.GetErrorYhigh(0))
    assert first == pytest.approx((0.96686159844054564, 0.11961518147810224, 0.13645595340008632))
    assert (graph.GetN(), rp.GetLowerRefGraph()) == (50, graph)
    assert [(type(o).__name__, opt) for o, opt in rp.GetUpperPad().primitives] == [
        ("TH1D", "Ahist"), ("TH1D", "AEsame")]  # fmt: skip


def test_the_ratio_is_drawn_with_its_own_axes_off_and_rootss_reference_lines():
    canvas, rp, _, _ = two()
    canvas.SetTicks(0, 1)
    rp.Draw()
    rp.GetLowYaxis().SetNdivisions(505)
    canvas.Update()
    assert rp.GetLowerPad().primitives[0][1] == "IAAP"
    lines = [o for o, _ in rp.GetLowerPad().primitives if type(o).__name__ == "TLine"]
    assert [line.members["fY1"] for line in lines] == [0.7, 1.0, 1.3]
    assert rp._axes["lower_y"].members["fNdiv"] == 505
    assert sorted(rp._mirrors) == ["lower_y", "upper_y"]  # ticks on the right, as asked
    assert rp._axes["lower_y"].members["_changed_labels"] == []  # the pads are apart


@pytest.mark.parametrize(
    ("option", "gridlines", "first"),
    [
        ("divsym", [0.7, 1.0, 1.3], None),
        ("diff", [0.0], None),
        ("diffsig", [1.0, 0.0, -1.0], None),
        ("diffsig errasym", [1.0, 0.0, -1.0], None),
    ],
)
def test_each_mode_works_out_its_lower_plot_and_its_reference_lines(option, gridlines, first):
    _canvas, rp, h1, h2 = two(option)
    rp.Draw("grid")
    graph = rp.GetCalculationOutputGraph()
    assert rp._gridline_positions == gridlines
    y1, y2, e1 = h1.GetBinContent(1), h2.GetBinContent(1), h1.GetBinError(1)
    expected = {"divsym": y1 / y2, "diff": y1 - y2}.get(option, (y1 - y2) / e1)
    at = 1 if option.startswith("diffsig") else 0  # the underflow is empty: no error, no point
    assert graph.GetY()[0] == pytest.approx(expected) and graph.GetX()[0] == pytest.approx(0.1)
    assert graph.GetN() == 50 + at - 1 + (1 - at)


def test_a_stack_is_divided_as_the_sum_of_its_histograms_either_side():
    canvas = ROOT.TCanvas("C", "stacks")
    a, b = ROOT.TH1D("a", "", 4, 0, 4), ROOT.TH1D("b", "", 4, 0, 4)
    for i in range(1, 5):
        a.SetBinContent(i, i)
        b.SetBinContent(i, 2 * i)
    stack = ROOT.THStack("s", "")
    stack.Add(a)
    stack.Add(b)
    over = ROOT.TRatioPlot(stack, b, "diff")
    over.Draw()
    assert list(over.GetCalculationOutputGraph().GetY()) == [1, 2, 3, 4]
    assert over.GetUpperRefObject() is stack and over.GetUpperRefXaxis() is not None
    assert over.GetUpperRefYaxis() is not None
    under = ROOT.TRatioPlot(b, stack, "diff")
    assert list(under.GetCalculationOutputGraph().GetY()) == [-1, -2, -3, -4]
    canvas.Update()


def test_what_a_ratio_plot_cannot_be_made_of_root_warns_of(capsys):
    ROOT.TCanvas("C", "")
    empty = ROOT.THStack("e", "")
    h = ROOT.TH1D("h", "", 2, 0, 1)
    for args in ((None, h), (empty, h), (h, empty), (None,), (h,), (None, empty)):
        ROOT.TRatioPlot(*args)
    err = capsys.readouterr().err
    for said in (
        "Need two histograms.",
        "Stack does not have histograms",
        "Need a histogram.",
        "Histogram given needs to have a (fit) function",
        "Need a histogram and a stack",
    ):
        assert said in err
    assert ROOT.TRatioPlot().GetUpperPad() is None


def test_the_setters_move_the_pads_and_margins_once_drawn_as_roots_do(capsys):
    _canvas, rp, _, _ = two()
    rp.SetSplitFraction(0.4)
    rp.SetInsetWidth(0.01)
    assert "Can only be used after TRatioPlot has been drawn." in capsys.readouterr().err


def test_the_margins_are_set_on_both_pads_in_single_precision():
    _, rp, _, _ = two()
    for setter, value in (("SetUpTopMargin", 0.2), ("SetUpBottomMargin", 0.1),
                          ("SetLowTopMargin", 0.1), ("SetLowBottomMargin", 0.4),
                          ("SetLeftMargin", 0.15), ("SetRightMargin", 0.05)):  # fmt: skip
        getattr(rp, setter)(value)
    upper, lower = rp.GetUpperPad(), rp.GetLowerPad()
    assert members(upper, "TopMargin", "BottomMargin", "LeftMargin", "RightMargin") == (
        0.20000000298023224, TENTH, 0.15000000596046448, 0.05000000074505806)  # fmt: skip
    assert members(lower, "TopMargin", "BottomMargin") == (TENTH, 0.4000000059604645)
    assert rp.GetSeparationMargin() == pytest.approx(0.1, abs=1e-7)


def test_the_split_and_inset_move_the_pads_once_drawn(capsys):
    canvas, rp, _, _ = two()
    upper, lower = rp.GetUpperPad(), rp.GetLowerPad()
    rp.SetUpTopMargin(0.2)
    rp.Draw("nogrid noconfint hideup")
    rp.SetSplitFraction(0.5)
    rp.SetSplitFraction(2)
    assert "Value 2.000000 is out of allowed range" in capsys.readouterr().err
    rp.SetInsetWidth(0.01)
    assert upper.members["fYlowNDC"] == 0.5 and lower.members["fHNDC"] == pytest.approx(0.49)
    assert rp._top_pad.members["fXlowNDC"] == pytest.approx(0.01)
    assert not any(type(o).__name__ == "TLine" for o, _ in lower.primitives)
    canvas.Update()
    assert rp._axes["upper_y"].members["_changed_labels"] == []


@pytest.mark.parametrize(
    ("option", "axis", "change"),
    [("fhideup", "upper_y", (1, -1.0, 0.0)), ("fhidelow", "lower_y", (-1, -1.0, 0.0)),
     ("hideup", "upper_y", (1, -1.0, 0.0)), ("nohide", "lower_y", None)],
)  # fmt: skip
def test_a_label_where_the_pads_touch_is_hidden_on_the_side_the_option_says(option, axis, change):
    _, rp, _, _ = two()
    rp.SetSeparationMargin(0.01)
    rp.Draw(option + " grid confint")
    changed = rp._axes[axis].members.get("_changed_labels", [])
    assert [c[:3] for c in changed] == ([change] if change else [])


def test_the_drawing_options_are_kept_as_root_keeps_them():
    canvas, rp, _h1, _h2 = two()
    rp.SetH1DrawOpt("E")
    rp.SetH2DrawOpt("hist SAME same")
    rp.SetGraphDrawOpt("L")
    rp.SetFitDrawOpt("C")
    rp.SetC1(2.0)
    rp.SetC2(1.0)
    rp.SetConfidenceIntervalColors("kBlue", 2)
    rp.SetGridlines([0.5, 1.5, 2.5], 2)
    rp.Draw()
    assert [opt for _, opt in rp.GetUpperPad().primitives] == ["AE", "Ahist  same"]
    assert rp._fit_draw_opt == "C" and (rp._ci1_color, rp._ci2_color) == (600, 2)
    assert [line.members["fY1"] for line in rp._gridlines] == [0.5, 1.5]
    rp.SetGridlines([1.0])  # fewer than drawn: the rest are put out of the way
    canvas.Update()
    assert [line.members["fX2"] for line in rp._gridlines] == [10.0, 0.0]
    rp.Draw()  # again: the same axes, placed afresh
    assert len(rp._axes) == 4


def test_a_log_pad_is_carried_into_the_ratio_plot_and_its_axes(capsys):
    canvas, rp, _, _ = two()
    canvas.SetLogy()
    canvas.SetLogx()
    rp.Draw()
    canvas.Update()
    assert rp._axes["upper_y"].members["fChopt"] == "SG"
    assert rp._axes["lower_x"].members["fChopt"] == "+SG"
    assert "Cannot set X axis to log scale" in capsys.readouterr().err


def test_a_fits_residuals_skip_the_empty_bins_and_its_bands_follow_the_latest_fit():
    ROOT.TCanvas("c1", "fit residual simple")
    h1 = fitted()
    rp = ROOT.TRatioPlot(h1)
    rp.SetConfidenceLevels(0.5, 0.9)
    rp.Draw("noconfint")
    assert rp.GetCalculationOutputGraph().GetN() == 36  # as ROOT's, of hist030
    assert rp.GetLowerPad().primitives[0][1] == "IALXSAME"
    assert rp.GetUpperPad().primitives[0][1] == "Ahist"
    assert (rp.GetXaxis().GetNbins(), rp.GetUpYaxis().GetTitle()) == (50, "y")
    ci = rp.GetConfidenceInterval1()
    assert ci.GetErrorY(10) / rp.GetConfidenceInterval2().GetErrorY(10) == pytest.approx(
        0.6744897501960817 / 1.6448536269514722, rel=1e-6)  # fmt: skip


def test_asymmetric_errors_are_the_side_the_function_lies_on():
    ROOT.TCanvas("c1", "")
    h1 = fitted()
    h1.SetBinErrorOption(ROOT.TH1.kPoisson)
    rp = ROOT.TRatioPlot(h1, "errasym")
    graph = rp.GetCalculationOutputGraph()
    i = h1.FindBin(graph.GetX()[5])
    fx = h1.GetFunction("gaus").Eval(graph.GetX()[5])
    side = h1.GetBinErrorLow(i) if h1.GetBinContent(i) > fx else h1.GetBinErrorUp(i)
    assert graph.GetY()[5] == pytest.approx((h1.GetBinContent(i) - fx) / side)


def test_a_residual_needs_a_function_and_says_so_when_it_has_none(capsys):
    canvas = ROOT.TCanvas("c1", "")
    h1 = ROOT.TH1D("h1", "", 10, 0, 1)
    h1.GetListOfFunctions().Add(ROOT.TLine(0, 0, 1, 1))
    assert ROOT.TRatioPlot(h1).GetCalculationOutputGraph() is None
    h1 = ROOT.TH1D("h2", "", 10, 0, 1)
    for i in range(1, 11):
        h1.SetBinContent(i, 5)
    flat = ROOT.TF1("flat", "[0]", 0, 1)
    flat.SetParameter(0, 4)
    h1.GetListOfFunctions().Add(flat)
    rp = ROOT.TRatioPlot(h1)  # no fit made: no bands
    assert rp.GetConfidenceInterval1().GetErrorY(0) == 0
    h1.GetListOfFunctions().Remove(flat)
    rp.Draw()
    canvas.Update()
    err = capsys.readouterr().err
    assert "h1 does not have a fit function" in err
    assert "Error in <TRatioPlot::Draw>: h1 does not have a fit function" in err


def test_a_significance_over_the_function_has_no_function_to_be_over(capsys):
    _, rp, _, _ = two("diffsig errfunc")
    assert rp.GetCalculationOutputGraph().GetN() == 0
    assert (
        "Warning in <TRatioPlot::BuildLowerPlot>: error mode is invalid" in capsys.readouterr().err
    )


def test_without_a_canvas_there_is_nowhere_to_lay_the_pads_out(capsys):
    h1, h2 = ROOT.TH1D("a", "", 2, 0, 1), ROOT.TH1D("b", "", 2, 0, 1)
    h1.SetBinContent(1, 1)
    h2.SetBinContent(1, 2)
    rp = ROOT.TRatioPlot(h1, h2)
    rp.SetLeftMargin(0.2)
    rp.Draw()
    err = capsys.readouterr().err
    assert "Error in <TRatioPlot::SetupPads>: need to create a canvas first" in err
    assert "Error in <TRatioPlot::Draw>: need to create a canvas first" in err
    assert rp.GetLowerRefGraph() is None and rp.GetLowerRefXaxis() is None
    assert "Lower pad has not been defined" in capsys.readouterr().err


def test_what_the_pads_lack_the_reference_getters_say():
    ROOT.TCanvas("C", "")
    _, rp, _, _ = two()
    assert rp.GetLowerRefGraph() is None
    rp.GetLowerPad().add(ROOT.TLine(0, 0, 1, 1), "")
    assert rp.GetLowerRefYaxis() is None and rp.GetUpperRefXaxis() is None
    assert rp.GetUpperRefYaxis() is None


def test_an_explicit_fit_result_gives_the_bands():
    ROOT.TCanvas("c1", "")
    h1 = ROOT.TH1D("h1", "h1", 50, -5, 5)
    h1.FillRandom("gaus", 2000)
    result = h1.Fit("gaus", "SQ0")
    from xrdroot.pyroot.core import fitters

    fitters.LATEST["result"] = None
    rp = ROOT.TRatioPlot(h1, "", result)
    rp.SetFitResult(result)
    assert rp.GetConfidenceInterval1().GetErrorY(20) > 0


def test_with_room_between_the_pads_nohide_leaves_every_label():
    _, rp, _, _ = two()
    rp.Draw("nohide")
    assert rp._axes["upper_y"].members.get("_changed_labels", []) == []
    assert rp._axes["lower_y"].members.get("_changed_labels", []) == []
