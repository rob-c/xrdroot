"""``TFractionFitter``: the fractions of the templates a data histogram is made of.

The fit is Barlow and Beeston's, so what is checked here is that its pieces answer
what the paper and ROOT's own source say they answer: a bin the data leaves empty
needs no solve, a bin whose leading template holds no events is solved in closed
form, a fraction stays inside the bounds ``Constrain`` gives it, and the plot is
what the smeared templates add up to. The fixture ROOT's own gtest fills - 1000
events of each of three functions over the same 40 bins - is fitted at the end and
checked against the numbers the ROOT 6.40 oracle gave for it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("iminuit")

from xrdroot.errors import UnsupportedFeatureError
from xrdroot.fit.fraction import FractionFit
from xrdroot.fit.minuit import minuit_messages
from xrdroot.pyroot import TMath
from xrdroot.pyroot.core.fractionfit import TFractionFitter


def likelihood():
    """Two templates and the data they could make, three bins each."""
    data = np.array([10.0, 20.0, 12.0])
    return FractionFit(data, np.array([[8.0, 2.0, 10.0], [5.0, 16.0, 4.0]]))


def two(filled=(0, 1, 1, 2, 3, 4, 4, 5), other=(1, 3, 3, 4, 5, 5), bins=6, tag=""):
    """A data histogram, and the ``TObjArray`` of two templates over the same bins.

    The data is filled at the bin numbers in ``filled`` and the first template at
    those in ``other``; the second template is filled where the data is.
    """
    from xrdroot.pyroot import TH1F, TObjArray

    mcs = TObjArray(2)
    for at, bins_filled in enumerate((other, filled)):
        h = TH1F(f"mc{tag}{at}", f"template {at}", bins, 0.0, 1.0)
        for which in bins_filled:
            h.Fill(which / bins + 0.5 / bins)
        mcs.Add(h)
    data = TH1F(f"data{tag}", "the data", bins, 0.0, 1.0)
    for which in filled:
        data.Fill(which / bins + 0.5 / bins)
    return data, mcs


# -- the likelihood ---------------------------------------------------------------------------


def test_the_fractions_are_each_parameter_over_the_data_it_accounts_for():
    fit = likelihood()
    assert fit.npar == 2
    assert list(fit.fractions([1.0, 1.0])) == pytest.approx([42.0 / 20.0, 42.0 / 25.0])


def test_the_likelihood_is_lowest_where_the_data_is_best_described():
    fit = likelihood()
    near = fit.objective(fit.solve().parameters)
    assert near < fit.objective([1.0, 1.0])
    assert near < fit.objective([0.1, 0.9])


def test_a_bin_the_data_leaves_empty_needs_no_template_variation():
    """ROOT's ``TMath::Nint(d_i) == 0``: ``t_i`` is 1 and nothing is smeared."""
    fit = FractionFit(np.array([0.0, 20.0]), np.array([[0.0, 2.0], [0.0, 16.0]]))
    t, leading, shared = fit.solutions(fit.weights * np.array([0.5, 0.5])[:, None])
    assert t[0] == 1.0 and leading[0] == -1 and shared[0] == 0.0


def test_a_bin_whose_leading_template_holds_no_event_is_solved_in_closed_form():
    """Barlow and Beeston's paragraph 5: ``A_ki`` is what the other templates leave there."""
    fit = FractionFit(np.array([10.0]), np.array([[0.0], [6.0]]))
    t, leading, shared = fit.solutions(np.array([[0.9], [0.1]]))
    assert leading[0] == 0 and t[0] == pytest.approx(-1.0 / 0.9)
    assert shared[0] == pytest.approx(10.0 / 1.9 - 6.0 * 0.1 / 0.8)


def test_a_closed_form_that_would_predict_less_than_nothing_goes_to_newtons_method():
    """``A_ki`` below zero cannot describe the bin, and ROOT solves for ``t_i`` instead."""
    fit = FractionFit(np.array([1.0]), np.array([[0.0], [1000.0]]))
    t, leading, _ = fit.solutions(np.array([[0.99], [0.01]]))
    assert leading[0] == -1 and t[0] != pytest.approx(-1.0 / 0.99)


def test_a_newton_step_that_leaves_the_range_is_taken_again_from_zero_with_a_tenth_of_it():
    """The equation has no solution below ``-1/p_max``, so the iteration restarts inside it."""
    fit = FractionFit(np.array([100.0]), np.array([[1.0], [1000.0]]))
    t, _, _ = fit.solutions(np.array([[0.98], [0.02]]))
    assert -1.0 / 0.98 < t[0] < 1.0


def test_a_solve_that_runs_out_of_turns_answers_what_it_has_so_far():
    """ROOT warns that it did not find ``t_i`` and carries on with the last value it had."""
    fit = FractionFit(np.array([100.0]), np.array([[1.0], [1000.0]]))
    strengths = np.array([[0.98], [0.02]])
    given_up = fit._newton(
        np.array([100.0]), np.array([[1.0], [1000.0]]), strengths, strengths.max(axis=0), 1
    )
    solved = fit._newton(
        np.array([100.0]), np.array([[1.0], [1000.0]]), strengths, strengths.max(axis=0)
    )
    assert -1.0 / 0.98 < given_up[0] < 1.0 and given_up[0] != solved[0]


def test_the_chi_square_is_a_likelihood_ratio_and_is_zero_for_a_perfect_fit():
    fit = likelihood()
    total, fitted = fit.predictions([fit.integral / 20.0, 0.0])
    assert fit.chisquare(np.array([10.0, 20.0, 12.0]), fit.templates) == pytest.approx(0.0)
    assert fit.chisquare(total, fitted) > 0.0


def test_a_template_weighted_bin_by_bin_is_the_weight_that_enters_the_likelihood():
    plain = likelihood()
    weighted = FractionFit(plain.data, plain.templates, np.ones_like(plain.templates))
    assert weighted.measured == pytest.approx(plain.measured)
    assert [weighted.objective([0.5, 0.5])] == pytest.approx([plain.objective([0.5, 0.5])])


# -- the minimiser's own messages --------------------------------------------------------------


def test_minuit2s_warnings_are_written_as_root_words_them(capsys):
    """ROOT's tests read ``Warning in <Minuit2>: ...`` off standard error."""
    from iminuit._core import MnPrint

    with minuit_messages():
        assert MnPrint.global_level >= 1
        print("W Somewhere something is wrong", flush=True)
        print("a line of nobody's", flush=True)
    written = capsys.readouterr()
    assert "Warning in <Minuit2>: Somewhere something is wrong" in written.err
    assert "a line of nobody's" in written.out
    assert MnPrint.global_level == 0


# -- the ROOT interface ------------------------------------------------------------------------


def test_a_data_histogram_is_fitted_as_the_sum_of_its_templates():
    data, mcs = two()
    fit = TFractionFitter(data, mcs)
    assert fit.Fit() == 0
    first, second = fit.GetResult(0)[0], fit.GetResult(1)[0]
    assert first + second == pytest.approx(1.0, abs=0.05)
    assert fit.GetNDF() == 4
    assert 0.0 <= fit.GetProb() <= 1.0
    assert fit.GetChisquare() >= 0.0


def test_the_plot_is_the_prediction_of_the_data_and_the_templates_are_the_smeared_ones():
    data, mcs = two()
    fit = TFractionFitter(data, mcs)
    fit.Fit()
    plot = fit.GetPlot()
    assert plot.GetName() == "Fraction fit to hist: data"
    assert plot.Integral() == pytest.approx(data.Integral(), rel=0.2)
    assert fit.GetPlot() is plot  # worked out once
    prediction = fit.GetMCPrediction(1)
    assert prediction.GetName() == "Prediction for MC sample 1"
    assert prediction.Integral() > 0.0


def test_a_fraction_may_be_kept_between_two_values_or_let_free_again():
    data, mcs = two()
    fit = TFractionFitter(data, mcs)
    fit.Constrain(0, 0.3, 0.6)
    fit.Constrain(1, 0.0, 1.0)
    assert fit.Fit() == 0
    assert fit.GetResult(0)[0] == pytest.approx(0.3, abs=0.2)
    fit.UnConstrain(0)
    assert fit.Fit() == 0


def test_a_range_and_an_excluded_bin_narrow_what_is_fitted():
    data, mcs = two()
    fit = TFractionFitter(data, mcs)
    assert fit.GetNDF() == 4
    fit.SetRangeX(2, 5)
    assert fit.GetNDF() == 2
    fit.ReleaseRangeX()
    assert fit.GetNDF() == 4
    fit.ExcludeBin(data.GetBin(2))
    assert fit.GetNDF() == 3
    fit.IncludeBin(data.GetBin(2))
    assert fit.GetNDF() == 4


def test_an_axis_the_histogram_does_not_have_cannot_be_restricted():
    data, mcs = two()
    fit = TFractionFitter(data, mcs)
    with pytest.raises(ValueError, match="has no 'y' axis"):
        fit.SetRangeY(1, 3)
    with pytest.raises(ValueError, match="has no 'z' axis"):
        fit.ReleaseRangeZ()


def test_the_range_of_a_two_dimensional_histogram_can_be_narrowed_along_both_axes():
    from xrdroot.pyroot import TH2F, TObjArray

    data = TH2F("data", "the data", 4, 0.0, 1.0, 5, 0.0, 1.0)
    data.Fill(0.3, 0.3)
    data.Fill(0.6, 0.7)
    mcs = TObjArray(2)
    for at in range(2):
        h = TH2F(f"mc{at}", f"template {at}", 4, 0.0, 1.0, 5, 0.0, 1.0)
        h.Fill(0.3, 0.3)
        h.Fill(0.6, 0.7)
        mcs.Add(h)
    fit = TFractionFitter(data, mcs)
    assert fit.GetNDF() == 18
    fit.SetRangeX(2, 3)
    fit.SetRangeY(2, 4)
    assert fit.GetNDF() == 4
    fit.ReleaseRangeX()
    fit.ReleaseRangeY()
    assert fit.GetNDF() == 18


def test_a_three_dimensional_histogram_is_fitted_bin_by_bin_as_well():
    from xrdroot.pyroot import TH3F, TObjArray

    data = TH3F("data", "the data", 2, 0.0, 1.0, 2, 0.0, 1.0, 2, 0.0, 1.0)
    data.Fill(0.2, 0.2, 0.2)
    mcs = TObjArray(2)
    for at in range(2):
        h = TH3F(f"mc{at}", f"template {at}", 2, 0.0, 1.0, 2, 0.0, 1.0, 2, 0.0, 1.0)
        h.Fill(0.2, 0.2, 0.2)
        h.Fill(0.7, 0.7, 0.7)
        mcs.Add(h)
    fit = TFractionFitter(data, mcs)
    assert fit.GetNDF() == 6
    fit.SetRangeZ(1, 2)
    fit.ReleaseRangeZ()
    fit.ExcludeBin(data.GetBin(1, 1, 1))
    assert fit.GetNDF() == 5


def test_the_histograms_fitted_may_be_changed_without_beginning_again():
    data, mcs = two()
    other, others = two(filled=(0, 1, 1, 2, 3, 4, 4, 5), other=(2, 2, 3, 4, 5, 5), tag="b")
    fit = TFractionFitter(data, mcs)
    fit.SetData(other)
    fit.SetMC(0, others.At(0))
    fit.SetMC(1, others.At(1))
    assert fit.Fit() == 0


def test_a_template_can_be_weighted_bin_by_bin():
    from xrdroot.pyroot import TH1F

    data, mcs = two()
    weight = TH1F("weight", "one per bin", 6, 0.0, 1.0)
    for bin in range(1, 7):
        weight.SetBinContent(bin, 2.0)
    fit = TFractionFitter(data, mcs)
    fit.SetWeight(0, weight)
    assert fit.Fit() == 0
    fit.SetWeight(0, None)
    assert fit.Fit() == 0


def test_the_templates_may_be_a_plain_list_of_histograms_as_well():
    data, mcs = two()
    fit = TFractionFitter(data, [mcs.At(0), mcs.At(1)])
    assert fit.Fit() == 0


# -- what is refused, and said so -------------------------------------------------------------


def test_asking_for_a_fit_that_has_not_been_made_is_refused():
    data, mcs = two()
    fit = TFractionFitter(data, mcs)
    with pytest.raises(ValueError, match=r"call Fit\(\) first"):
        fit.GetResult(0)
    with pytest.raises(ValueError, match=r"call Fit\(\) first"):
        fit.GetPlot()
    with pytest.raises(ValueError, match=r"call Fit\(\) first"):
        fit.GetMCPrediction(0)
    with pytest.raises(ValueError, match=r"call Fit\(\) first"):
        fit.GetChisquare()


def test_a_fraction_that_names_no_template_is_refused():
    data, mcs = two()
    fit = TFractionFitter(data, mcs)
    fit.UnConstrain(0)  # nothing was constrained, which is not an error
    with pytest.raises(IndexError, match="fraction 2 does not exist"):
        fit.Constrain(2, 0.0, 1.0)
    with pytest.raises(IndexError, match=r"numbered from 0 to 1"):
        fit.UnConstrain(-1)
    fit.Fit()
    with pytest.raises(IndexError, match="fraction 5 does not exist"):
        fit.GetResult(5)


def test_a_bin_cannot_be_excluded_twice_or_included_without_being_excluded():
    data, mcs = two()
    fit = TFractionFitter(data, mcs)
    fit.ExcludeBin(2)
    with pytest.raises(ValueError, match="already excluded"):
        fit.ExcludeBin(2)
    with pytest.raises(ValueError, match="was never excluded"):
        fit.IncludeBin(3)


def test_a_fit_needs_a_data_histogram_and_two_templates():
    from xrdroot.pyroot import TH1F, TObjArray

    empty = TFractionFitter()
    with pytest.raises(ValueError, match="none was given"):
        empty.SetData(None)
    data = TH1F("data", "the data", 4, 0.0, 1.0)
    data.Fill(0.1)
    one = TObjArray(1)
    one.Add(TH1F("mc", "a template", 4, 0.0, 1.0))
    with pytest.raises(ValueError, match=r"two templates' fractions at least"):
        TFractionFitter(data, one)


def test_a_data_histogram_with_nothing_in_the_fitted_bins_is_refused():
    _, mcs = two()
    from xrdroot.pyroot import TH1F

    with pytest.raises(ValueError, match="holds no events in the bins being fitted"):
        TFractionFitter(TH1F("nothing", "empty", 6, 0.0, 1.0), mcs)


def test_a_template_that_is_missing_or_describes_other_bins_is_refused():
    from xrdroot.pyroot import TH1F

    data, mcs = two()
    with pytest.raises(ValueError, match="must have the same bins"):
        TFractionFitter(data, mcs).SetMC(0, TH1F("mcx", "another", 5, 0.0, 1.0))
    with pytest.raises(ValueError, match="template 1 does not exist"):
        TFractionFitter(data, mcs).SetMC(1, None)
    fewer = TH1F("data", "fewer", 4, 0.0, 1.0)
    fewer.Fill(0.1)
    with pytest.raises(ValueError, match="must have the same bins"):
        TFractionFitter(data, mcs).SetData(fewer)


def test_a_weights_histogram_of_other_bins_or_a_template_with_no_events_is_refused():
    from xrdroot.pyroot import TH1F

    data, mcs = two()
    with pytest.raises(ValueError, match="the weights histogram of template 0"):
        TFractionFitter(data, mcs).SetWeight(0, TH1F("weight", "another", 3, 0.0, 1.0))
    empty = TH1F("mcx", "no events", 6, 0.0, 1.0)
    with pytest.raises(ValueError, match="holds no events in the bins being fitted"):
        TFractionFitter(data, mcs).SetMC(0, empty)


def test_the_degrees_of_freedom_may_be_none_and_then_the_probability_is_zero():
    data, mcs = two(filled=(0, 1, 1, 2), other=(1, 1, 2, 3), bins=4)
    fit = TFractionFitter(data, mcs)
    fit.Constrain(0, 0.0, 1.0)
    fit.Constrain(1, 0.0, 1.0)
    assert fit.GetNDF() == 2
    fit.Fit()
    fit.ExcludeBin(data.GetBin(2))
    fit.ExcludeBin(data.GetBin(3))
    assert fit.GetNDF() == 0
    with pytest.raises(ValueError, match="holds no events"):
        fit.ExcludeBin(data.GetBin(4))


def test_a_fit_with_no_degrees_of_freedom_left_has_no_probability():
    data, mcs = two(filled=(0, 1), other=(1, 2), bins=2)
    fit = TFractionFitter(data, mcs)
    fit.Constrain(0, 0.0, 1.0)
    fit.Constrain(1, 0.0, 1.0)
    fit.Fit()
    assert fit.GetNDF() == 0 and fit.GetProb() == 0.0


def test_the_fitter_object_and_the_minos_analysis_are_refused_in_a_sentence():
    data, mcs = two()
    fit = TFractionFitter(data, mcs)
    with pytest.raises(UnsupportedFeatureError, match="xrdroot has no such object"):
        fit.GetFitter()
    with pytest.raises(UnsupportedFeatureError, match="does not do yet"):
        fit.ErrorAnalysis(0.5)


# -- ROOT's own fixture ------------------------------------------------------------------------


def rooted_fixture():
    """ROOT 6.40's ``gtest/hist/hist/testTFractionFitter``: three functions, data and templates."""
    from xrdroot.pyroot import TF1, TH1F, TObjArray, gRandom

    functions = [
        TF1(f"f{at}", text, 0.0, TMath.Pi())
        for at, text in enumerate(
            (
                "[0]*(1-cos(x))/TMath::Pi()",
                "[0]*(1-cos(x)*cos(x))*2./TMath::Pi()",
                "[0]*(1+cos(x))/TMath::Pi()",
            )
        )
    ]
    for function in functions:
        function.SetParameter(0, 1.0)
    data = TH1F("data", "Data angle distribution", 40, 0.0, TMath.Pi())
    for _ in range(1000):
        p = gRandom.Uniform()
        which = 0 if p < 0.01 else (1 if p < 0.31 else 2)
        data.Fill(functions[which].GetRandom())
    mcs = TObjArray(3)
    for at, function in enumerate(functions):
        h = TH1F(f"mc{at}", f"MC sample {at}", 40, 0.0, TMath.Pi())
        for _ in range(1000):
            h.Fill(function.GetRandom())
        mcs.Add(h)
    return data, mcs


def test_the_fixture_of_roots_gtest_fits_to_the_numbers_root_finds():
    """The fit ROOT 6.40 ran on it: 0.28322 and 0.71675, a chi-square of 58.781 on 37 points."""
    data, mcs = rooted_fixture()
    fit = TFractionFitter(data, mcs)
    for par in range(3):
        fit.Constrain(par, 0.0, 1.0)
    assert fit.Fit() == 0
    assert [fit.GetResult(par)[0] for par in range(3)] == pytest.approx(
        [3.70673e-05, 0.2832225764, 0.7167516527], abs=1e-6
    )
    assert fit.GetChisquare() == pytest.approx(58.7810120071, abs=1e-5)
    assert fit.GetNDF() == 37
    assert fit.GetPlot().Integral() == pytest.approx(1000.0064830572, abs=1e-5)
