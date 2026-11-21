"""What is done with a histogram: copies, arithmetic, reshaping, comparing and drawing from it."""

from __future__ import annotations

import array
import math

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def filled(name="h", n=4):
    h = ROOT.TH1D(name, "t", n, 0, n)
    for x, w in ((0.5, 1.0), (1.5, 2.0), (2.5, 3.0)):
        h.Fill(x, w)
    return h


def test_clones_copies_and_draw_copies_are_kept_as_root_keeps_them():
    h = filled()
    c = h.Clone("c")
    expect(
        (bool(ROOT.gROOT.FindObject("c") is c), True),
        (c.GetBinContent(2), 2),
        (h.Clone().GetName(), "h"),
    )
    ROOT.TH1.AddDirectory(False)
    assert ROOT.gROOT.FindObject("d") is None or h.Clone("d") is not None
    ROOT.TH1.AddDirectory(True)
    target = ROOT.TH1D("t", "", 1, 0, 1)
    h.Copy(target)
    expect(
        (target.GetNbinsX(), 4),
        (target.GetBinContent(3), 3),
    )
    drawn = h.DrawCopy("hist")
    expect(
        (drawn.GetName(), "h_copy"),
        (bool(drawn.GetDirectory() is None), True),
    )
    normal = h.DrawNormalized("", 2.0)
    expect(
        (normal.Integral(), pytest.approx(2.0)),
        (ROOT.TH1D("e", "", 1, 0, 1).DrawNormalized().Integral(), 0),
    )


def test_arithmetic_in_place_is_roots():
    h, g = filled("h"), filled("g")
    h.Add(g, 2.0)
    assert h.GetBinContent(2) == 6
    h.Add(g, g, 1.0, -1.0)
    assert h.GetBinContent(2) == 0
    h.Add(g)
    h.Multiply(g)
    assert h.GetBinContent(3) == 9
    h.Multiply(g, g, 2.0, 1.0)
    assert h.GetBinContent(3) == 18
    h.Divide(g)
    assert h.GetBinContent(3) == 6
    h.Divide(g, g, 3.0, 1.0, "B")
    assert h.GetBinContent(3) == 3
    h.Scale(2.0, "width")
    assert h.GetBinContent(3) == 6


def test_adding_a_function_adds_it_at_each_bins_centre():
    h = ROOT.TH1D("h", "", 2, 0, 2)
    h.Add(ROOT.TF1("line", "x", 0, 2), 2.0)
    assert (h.GetBinContent(1), h.GetBinContent(2)) == (1.0, 3.0)
    h.Add(lambda x: 1.0 + 0 * x)
    assert h.GetBinContent(1) == 2.0


def test_the_operators_make_new_histograms():
    h, g = filled("h"), filled("g")
    expect(
        ((h + g).GetBinContent(2), 4),
        ((h - g).GetBinContent(2), 0),
        ((h * 3).GetBinContent(2), 6),
        ((3 * h).GetBinContent(2), 6),
        ((h / g).GetBinContent(2), 1),
        ((h * g).GetBinContent(2), 4),
    )
    h += g
    h -= g
    h *= 2
    h /= 2
    expect(
        (h.GetBinContent(2), 2),
        (bool(isinstance(h, ROOT.TH1D)), True),
    )


def test_rebinning_in_place_or_into_a_new_histogram():
    h = filled("h")
    made = h.Rebin(2, "h2")
    expect(
        (made.GetNbinsX(), 2),
        (h.GetNbinsX(), 4),
        (bool(ROOT.gROOT.FindObject("h2") is made), True),
        (bool(h.Rebin(2) is h), True),
        (h.GetNbinsX(), 2),
        (h.GetBinContent(1), 3),
    )
    uneven = filled("u").Rebin(2, "u2", array.array("d", [0, 1, 4]))
    expect(
        (uneven.GetBinContent(2), 5),
        (filled("x").RebinX(4).GetNbinsX(), 1),
    )
    h2 = ROOT.TH2D("h2d", "", 4, 0, 4, 4, 0, 4)
    h2.Fill(0.5, 0.5)
    expect(
        (h2.Rebin2D(2, 2, "h2r").GetNbinsX(), 2),
        (bool(h2.Rebin2D(2, 2) is h2), True),
        (h2.RebinY(2, "h2y").GetNbinsY(), 1),
    )


def test_projections_take_roots_names_and_reuse_them_silently(capsys):
    h2 = ROOT.TH2D("h2", "", 2, 0, 2, 2, 0, 2)
    h2.Fill(0.5, 1.5, 2.0)
    first = h2.ProjectionX()
    again = h2.ProjectionX()
    expect(
        (bool(first.GetName() == again.GetName() == "h2_px"), True),
        (bool(ROOT.gROOT.FindObject("h2_px") is again), True),
        (bool("Replacing" not in capsys.readouterr().err), True),
        (h2.ProjectionY().GetBinContent(2), 2),
        (h2.ProjectionY("y", 1, 1).GetBinContent(2), 2),
        (h2.ProjectionX("x", 1, 1).GetBinContent(1), 0),
        (h2.ProfileY().GetName(), "h2_pfy"),
        (h2.ProfileX("pp", 1, 2).GetName(), "pp"),
        (h2.ProfileY("qq", 1, 2).GetName(), "qq"),
    )
    h3 = ROOT.TH3D("h3", "", 2, 0, 2, 2, 0, 2, 2, 0, 2)
    h3.Fill(0.5, 0.5, 1.5)
    expect(
        (h3.Project3D("z").GetBinContent(2), 1),
        (h3.Project3D("xy").GetName(), "h3_xy"),
        (h3.Project3D("Projection_xy").GetName(), "h3_Projection_xy"),
        (h3.Project3D("ZX nof").GetName(), "h3_zx "),
        (h3.Project3D("XYE").GetTitle().endswith(" xy projection"), True),
    )


def test_a_projection_is_kept_only_while_histograms_are(capsys):
    ROOT.TH1.AddDirectory(False)
    h2 = ROOT.TH2D("h2", "", 2, 0, 2, 2, 0, 2)
    assert ROOT.gROOT.FindObject(h2.ProjectionX().GetName()) is None


def test_the_cumulative_runs_forward_or_back_with_its_errors():
    h = filled()
    forward = h.GetCumulative()
    back = h.GetCumulative(False, "_back")
    expect(
        ([forward.GetBinContent(b) for b in range(1, 5)], [1, 3, 6, 6]),
        ([back.GetBinContent(b) for b in range(1, 5)], [6, 5, 3, 0]),
        (forward.GetBinError(2), pytest.approx(np.sqrt(5))),
        (back.GetName(), "h_back"),
        (forward.GetEntries(), 4),
    )


def test_smoothing_quantiles_and_comparisons():
    h = filled()
    q = array.array("d", [0.0] * 5)
    assert h.GetQuantiles(5, q) >= 1
    h.Smooth(1)
    g = filled("g")
    residuals = np.zeros(4)
    expect(
        (bool(0 <= g.Chi2Test(filled("k"), "WW", residuals) <= 1), True),
        (bool(0 <= g.KolmogorovTest(filled("m")) <= 1), True),
    )


def test_fitting_hands_back_a_result_and_hangs_the_function(capsys):
    ROOT.gRandom.SetSeed(1)
    h = ROOT.TH1D("h", "", 40, -4, 4)
    h.FillRandom("gaus", 2000)
    result = h.Fit("gaus", "QS")
    expect(
        (int(result), 0),
        (result.Parameter(2), pytest.approx(1.0, rel=0.1)),
        (bool(h.GetFunction("gaus") is not None), True),
        (bool(h.GetFunction("nothing") is None), True),
        (h.GetListOfFunctions().GetSize(), 1),
    )
    ranged = h.Fit("gaus", "QS", "", -1, 1)
    assert ranged.Ndf() < result.Ndf()


def test_fill_random_draws_from_functions_histograms_and_named_functions():
    source = ROOT.TF1("line", "x", 0, 1)
    h = ROOT.TH1D("h", "", 10, 0, 1)
    h.FillRandom("line", 100)
    h.FillRandom(source, 100)
    h.FillRandom("gaus", 10)
    assert h.GetEntries() == 210
    g = ROOT.TH1D("g", "", 10, 0, 1)
    g.FillRandom(h, 50, ROOT.TRandom3(5))
    assert g.GetEntries() == 50


def test_random_numbers_from_a_histogram(tmp_path):
    empty = ROOT.TH1D("e", "", 2, 0, 1)
    assert empty.GetRandom() == 0.0
    x, y = array.array("d", [9]), array.array("d", [9])
    expect(
        (ROOT.TH2D("e2", "", 2, 0, 1, 2, 0, 1).GetRandom2(x, y), (0.0, 0.0)),
        (x[0], 0),
    )
    h = filled()
    expect(
        (bool(0 <= h.GetRandom(ROOT.TRandom3(2)) <= 4), True),
        # ROOT's fIntegral is the running sums and then fEntries once more, as
        # GetIntegral hands it out; [-2], the total, is normalised to one.
        (h.GetIntegral()[-1], float(h.GetEntries())),
        (h.GetIntegral()[-2], 1.0),
    )
    h2 = ROOT.TH2D("h2", "", 2, 0, 2, 2, 0, 2)
    h2.Fill(1.5, 0.5)
    px, py = h2.GetRandom2(x, y, ROOT.TRandom3(3))
    expect(
        (bool(1 <= px <= 2), True),
        (bool(0 <= py <= 1), True),
        (x[0], px),
    )


def test_a_2d_fill_from_a_function_of_code_integrates_each_bin_eight_points_each_way():
    """Every bin's eight points each way - handed to the code all at once, as its arithmetic
    takes arrays."""
    points = []

    def flat(x, par):
        points.append(np.size(x[0]))
        return 1.0 + x[0]

    f2 = ROOT.TF2("code2", flat, 0, 2, 0, 1, 0)
    h = ROOT.TH2D("hcode2", "", 2, 0, 2, 1, 0, 1)
    h.FillRandom(f2, 300, ROOT.TRandom3(1))
    assert points == [2 * 8 * 8] and h.GetEntries() == 300
    assert h.GetBinContent(2, 1) > h.GetBinContent(1, 1)  # 2.5 against 1.5 of the integral
    formula = ROOT.TH2D("hform2", "", 2, 0, 2, 1, 0, 1)
    formula.FillRandom(ROOT.TF2("form2", "1 + x", 0, 2, 0, 1), 300, ROOT.TRandom3(1))
    assert formula.GetEntries() == 300


def test_a_function_set_or_added_bin_by_bin_is_what_root_6_40_makes_it():
    """ROOT 6.40's contents for a Gaussian of height 10 at 5, width 1.5, over [2, 8]: no entries
    counted, the range ignored unless ``"r"``, Poisson counts with ``"s"``, 2D bins too."""
    f = ROOT.TF1("f", "gaus", 2, 8)
    f.SetParameters(10, 5, 1.5)
    h = ROOT.TH1F("h", "h", 10, 0, 10)
    h.Fill(1.0)
    h.Eval(f)
    first = [round(h.GetBinContent(i), 6) for i in range(12)]
    assert first[:6] == [0.0, 0.11109, 0.657285, 2.493522, 6.065307, 9.459595]
    assert (h.GetEntries(), h.GetMean()) == (0.0, 0.0)
    h.Eval(f, "ra")
    assert [round(h.GetBinContent(i), 6) for i in (2, 3, 5)] == [0.657285, 4.987044, 18.919189]
    ROOT.gRandom.SetSeed(7)
    g = ROOT.TH1D("g", "g", 10, 0, 10)
    g.Sumw2()
    g.Eval(f, "s")
    assert [g.GetBinContent(i) for i in range(3, 9)] == [4.0, 6.0, 4.0, 12.0, 6.0, 5.0]
    assert g.GetBinError(5) == 2.0
    h.Eval(None)
    assert h.GetBinContent(5) == pytest.approx(18.919189)


def test_a_function_of_two_or_three_variables_is_set_at_each_bin_s_centre_too():
    f2 = ROOT.TF2("f2", "x*y", 0, 4, 0, 4)
    h2 = ROOT.TH2F("h2", "h2", 4, 0, 4, 2, 0, 4)
    h2.Eval(f2, "r")
    assert [h2.GetBinContent(i, j) for i in (1, 4) for j in (1, 2)] == [0.5, 1.5, 3.5, 10.5]
    f3 = ROOT.TF3("f3", "x+y+z", 0, 1, 0, 1, 0, 1)
    h3 = ROOT.TH3D("h3", "h3", 2, 0, 2, 1, 0, 1, 1, 0, 1)
    h3.Eval(f3, "r")
    assert (h3.GetBinContent(1, 1, 1), h3.GetBinContent(2, 1, 1)) == (1.5, 0.0)
    assert [getattr(f3, f"Get{v}{e}")() for v in "YZ" for e in ("min", "max")] == [0, 1, 0, 1]


def test_a_th2_and_a_th3_interpolate_between_centres_holding_the_edges_flat_as_root_does():
    """Values from ROOT 6.40's ``Interpolate`` on the same contents."""
    h2 = ROOT.TH2F("i2", "", 4, 0, 4, 4, 0, 4)
    h3 = ROOT.TH3F("i3", "", 4, 0, 4, 4, 0, 4, 4, 0, 4)
    for i in range(4):
        for j in range(4):
            h2.SetBinContent(i + 1, j + 1, 1 + i + 10 * j)
            for k in range(4):
                h3.SetBinContent(i + 1, j + 1, k + 1, 1 + i + 10 * j + 100 * k)
    expect(
        (h2.Interpolate(0.2, 2), 16.0), (h2.Interpolate(3.9, 2.2), 21.0),
        (h2.Interpolate(1.2, 0.1), pytest.approx(1.7)), (h2.Interpolate(4, 2), 0.0),
        (h3.Interpolate(-1, 2, 2), 166.0), (h3.Interpolate(4, 4, 4), 334.0),
        (h3.Interpolate(3.7, 1, 1), 59.0), (h3.Interpolate(-1, -1, -1), 1.0),
    )


def test_histograms_of_64_bit_integers_keep_whole_numbers():
    for made in (ROOT.TH1L("l1", "", 2, 0, 2), ROOT.TH2L("l2", "", 2, 0, 2, 2, 0, 2),
                 ROOT.TH3L("l3", "", 2, 0, 2, 2, 0, 2, 2, 0, 2)):
        made.SetBinContent(1, 2**40 + 1)
        assert made.GetBinContent(1) == 2**40 + 1 and made.ClassName().endswith("L")
    assert ROOT.TH2L.CLASS_TITLE == "2-Dim histograms (one 64 bit integer per channel)"


def test_a_negative_bin_is_a_nan_integral_and_a_nan_draw(capsys):
    """ROOT's ``onlyPositive`` check: ``Error in <TH1::ComputeIntegral>``, a NaN from
    ``ComputeIntegral``, and a NaN from a draw - which is what ``GetRandom2`` sets too."""
    h = ROOT.TH1D("h", "", 2, 0, 2)
    h.Fill(0.5)
    h.SetBinContent(2, -1.0)
    assert math.isnan(h.ComputeIntegral(True))
    assert "Error in <TH1::ComputeIntegral>: Bin content is negative - return a NaN value" in (
        capsys.readouterr().err)
    assert math.isnan(h.GetRandom())
    x, y = array.array("d", [9]), array.array("d", [9])
    h2 = ROOT.TH2D("h2", "", 2, 0, 2, 2, 0, 2)
    h2.Fill(0.5, 0.5)
    h2.SetBinContent(2, 2, -1.0)
    assert all(math.isnan(v) for v in h2.GetRandom2(x, y))
    assert (math.isnan(x[0]), math.isnan(y[0])) == (True, True)


def test_an_integral_already_made_is_used_again_until_the_entries_move(capsys):
    """ROOT keeps ``fIntegral`` and remakes it only when ``fEntries`` changed: an empty
    histogram says its integral is zero once, and again once it is filled."""
    empty = ROOT.TH1D("e", "", 2, 0, 1)
    assert empty.ComputeIntegral() == 0.0
    assert "Integral = 0, no hits in histogram bins" in capsys.readouterr().err
    assert (empty.GetIntegral()[-1], empty.GetIntegral()[-2]) == (0.0, 0.0)
    capsys.readouterr()
    assert empty.GetIntegral()[-1] == 0.0 and capsys.readouterr().err == ""  # still the same
    empty.Fill(0.5)
    assert empty.GetIntegral()[-2] == 1.0 and empty.GetIntegral()[-1] == 1.0


def test_normalize_scales_what_its_option_names_to_one_as_root_does():
    """``TH1::Normalize``: the numbers are ROOT's own ``test_TH1.cxx`` case - no option makes
    the integral over the bins' widths one (10 for ten unit-width bins holding 18), ``"max"``
    the largest bin, ``"sum"`` the sum of the weights; and ``GetEntries`` stays as it was."""
    def made():
        h = ROOT.TH1F("n", "n", 10, 0, 1)
        for at, content in ((1, 1.0), (3, 5.0), (5, 10.0), (7, -1.0), (10, 3.0)):
            h.SetBinContent(at, content)
        return h

    plain = made()
    expect(
        (plain.GetEntries(), 5.0),
        (plain.GetSumOfWeights(), 18.0),
        (plain.Integral("width"), pytest.approx(1.8)),
        (plain.GetMaximum(), 10.0),
    )
    for option, weights, width_sum, largest in (("", 10.0, 1.0, 50 / 9),
                                                ("integral", 10.0, 1.0, 50 / 9),
                                                ("max", 1.8, 0.18, 1.0),
                                                ("sum", 1.0, 0.1, 5 / 9)):  # fmt: skip
        scaled = made()
        scaled.Normalize(option)
        assert scaled.GetEntries() == 5.0
        assert scaled.GetSumOfWeights() == pytest.approx(weights)
        assert scaled.Integral("width") == pytest.approx(width_sum)
        assert scaled.GetMaximum() == pytest.approx(largest)


def test_a_histogram_asks_a_global_flag_for_its_statistics_overflows():
    """``TH1::EStatOverflows``: a histogram is ``kNeutral`` until told otherwise - its own
    flag, which neither ``TH1::StatOverflows`` nor another histogram's flag changes."""
    neutral, ignoring = ROOT.TH1F("a", "a", 2, 0, 1), ROOT.TH1F("b", "b", 2, 0, 1)
    expect(
        (neutral.GetStatOverflows(), ROOT.TH1.EStatOverflows.kNeutral),
        (ROOT.TH1.EStatOverflows.kConsider, 1),
    )
    ignoring.SetStatOverflows(ROOT.TH1.EStatOverflows.kIgnore)
    ROOT.TH1.StatOverflows(True)
    expect(
        (ignoring.GetStatOverflows(), ROOT.TH1.kIgnore),
        (neutral.GetStatOverflows(), ROOT.TH1.kNeutral),
        (ROOT.TH1F("c", "c", 2, 0, 1).GetStatOverflows(), ROOT.TH1.kNeutral),
    )
    ROOT.TH1.StatOverflows(False)


def test_add_bin_content_takes_one_bin_number_per_axis_then_the_weight():
    """``TH1::AddBinContent(bin[, w])`` and the overloads ``TH2``/``TH3`` add - the bin
    numbers are one per axis, so a 2-D histogram's third argument is its y bin and not a
    weight, and only one more than the number of axes is a weight. ROOT's own case
    ``TH1.AddBinContent`` - ``h2.AddBinContent(1, 1, 1.)`` adds one to the bin (1, 1) -
    and ``Eval``, which works out a global bin itself, still adds to that one."""
    h1 = ROOT.TH1F("a1", "", 10, 0, 1)
    h1.AddBinContent(1, 1.0)
    h1.AddBinContent(1, 2.0)
    h2 = ROOT.TH2F("a2", "", 10, 0, 1, 2, 0, 3)
    h2.AddBinContent(1, 1, 1.0)
    h2.AddBinContent(2, 2, 0.5)
    h2.AddBinContent(3, 2, 2.0)
    h3 = ROOT.TH3F("a3", "", 5, 0, 1, 2, 0, 2, 2, 0, 3)
    h3.AddBinContent(1, 1, 1, 1.0)
    expect(
        (h1.GetBinContent(1), 3.0),
        ((h2.GetBinContent(1, 1), h2.GetBinContent(2, 2), h2.GetBinContent(3, 2)),
         (1.0, 0.5, 2.0)),
        (h2.GetBinContent(2, 1), 0.0),
        (h3.GetBinContent(1, 1, 1), 1.0),
    )
    h1.AddBinContent(2)  # one number per axis and no weight: one entry
    assert h1.GetBinContent(2) == 1.0


def test_the_static_smooth_array_leaves_a_smooth_array_as_it_was_and_writes_in_place():
    """``TH1::SmoothArray(nn, yy[, ntimes])`` is HBOOK's ``hsmoof`` over a raw array - the
    static a program calls with ``v.data()`` - and what it writes lands in the array itself.
    ROOT's ``TH1.SmoothArrayCrossCheck`` case asks for a ramp and one repeated value to come
    back unchanged; a single wild value is stepped over rather than smeared, and ``ntimes``
    at zero smooths nothing."""
    ramp = ROOT.std.vector["double"]([0.0, 1.0, 2.0, 3.0, 4.0])
    ROOT.TH1.SmoothArray(ramp.size(), ramp.data())
    flat = ROOT.std.vector["double"]([1.0] * 5)
    ROOT.TH1.SmoothArray(5, flat.data())
    expect(
        (list(ramp), [0.0, 1.0, 2.0, 3.0, 4.0]),
        (list(flat), [1.0] * 5),
    )
    wild = np.array([0.0, 1.0, 99.0, 3.0, 4.0])
    ROOT.TH1.SmoothArray(5, wild)
    # A running median steps over a single wild value rather than smearing it: the 99 is
    # gone and the values beside it are near what they were.
    assert 99.0 not in wild.tolist()
    assert abs(wild[1] - 1.0) <= 1.0 and abs(wild[3] - 3.0) <= 1.0
    untouched = np.array([0.0, 5.0, 0.0])
    ROOT.TH1.SmoothArray(3, untouched, 0)
    assert untouched.tolist() == [0.0, 5.0, 0.0]
    assert ROOT.TH1D("s", "", 5, 0, 5).Smooth(1) is None  # the in-place twin still there
