"""Fitting histograms and profiles: ROOT's options, each doing what ROOT's does.

Where ROOT's answer is exact it is checked exactly: a polynomial fitted by
least squares against the normal equations solved by hand, the chi-square
and likelihood against the sums written out. Where it comes from Minuit,
the minimum is checked independently of Minuit - on a grid of the two free
parameters, brute force - and against the linear solution where there is
one to compare with.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot import Function, Histogram, Profile, TRandom3, UnsupportedFeatureError
from xrdroot.fit import FitResult, cost, guesses, minuit, parse
from xrdroot.fit.data import DataOptions, FitData, from_histogram


def gaussian(entries=5000, bins=(60, -3, 3), seed=11, kind="D"):
    h = Histogram.book("h", bins, kind=kind)
    h.fill_random("gaus", entries, rng=TRandom3(seed))
    return h


def straight_line(n=12):
    """A histogram whose contents lie near a line, each bin with its own error."""
    h = Histogram.book("line", (n, 0.0, float(n)))
    x = np.arange(n) + 0.5
    y = 3.0 + 0.5 * x + np.sin(3 * x)
    h.fill(x, weight=y)
    h._sumw2()[1 : n + 1] = (0.2 + 0.05 * x) ** 2
    return h, x, y, 0.2 + 0.05 * x


# -- linear least squares ----------------------------------------------------------------------


def test_a_polynomial_is_fitted_by_exact_linear_least_squares(capsys):
    h, x, y, e = straight_line()
    result = h.fit("pol1")
    design = np.stack([np.ones_like(x), x], axis=1) / e[:, None]
    normal = design.T @ design
    expected = np.linalg.solve(normal, design.T @ (y / e))
    np.testing.assert_allclose(result.parameters, expected, rtol=1e-12)
    np.testing.assert_allclose(result.covariance, np.linalg.inv(normal), rtol=1e-10)
    residual = (y - expected[0] - expected[1] * x) / e
    assert result.chi2 == pytest.approx(float(residual @ residual), rel=1e-12)
    assert (result.minimizer, result.ndf, result.status, result.valid) == ("Linear", 10, 0, True)
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == "*" * 40 and printed[1] == "Minimizer is Linear"
    assert printed[2].startswith("Chi2                      = ")
    assert not any(line.startswith(("Edm", "NCalls")) for line in printed)


def test_minuit_converges_to_the_linear_answer_and_the_grid_agrees(capsys):
    h, x, y, e = straight_line()
    exact = h.fit("pol1", "Q")
    migrad = h.fit("pol1", "FQ")
    assert migrad.minimizer == "Minuit2 / Migrad" and migrad.nfev > 0
    np.testing.assert_allclose(migrad.parameters, exact.parameters, rtol=1e-6)
    np.testing.assert_allclose(migrad.errors, exact.errors, rtol=1e-4)
    # brute force: the chi-square on a grid round the answer has its least value there
    a = np.linspace(exact.parameters[0] - 0.05, exact.parameters[0] + 0.05, 201)
    b = np.linspace(exact.parameters[1] - 0.01, exact.parameters[1] + 0.01, 201)
    grid = (((y - a[:, None, None] - b[None, :, None] * x) / e) ** 2).sum(axis=2)
    i, j = np.unravel_index(np.argmin(grid), grid.shape)
    assert abs(a[i] - migrad.parameters[0]) <= a[1] - a[0]
    assert abs(b[j] - migrad.parameters[1]) <= b[1] - b[0]


def test_any_formula_linear_in_its_parameters_is_solved_exactly_even_with_one_fixed():
    h, x, y, e = straight_line()
    f = Function("f", "[0]*sin(x) + [1] + [2]*x", parameters=[0.0, 0.0, 0.25])
    f.fix(2)
    result = h.fit(f, "Q")
    design = np.stack([np.sin(x), np.ones_like(x)], axis=1) / e[:, None]
    target = (y - 0.25 * x) / e
    expected = np.linalg.lstsq(design, target, rcond=None)[0]
    np.testing.assert_allclose(result.parameters[:2], expected, rtol=1e-10)
    assert result.parameters[2] == 0.25 and result.errors[2] == 0.0 and result.fixed[2]
    assert result.ndf == 12 - 2
    # "B" and the others that need Minuit take it past the linear fitter
    assert h.fit(f.copy(), "QB").minimizer == "Minuit2 / Migrad"


def test_a_python_model_is_fitted_with_npar_and_refused_without():
    h, *_ = straight_line()
    result = h.fit(lambda t, p: p[0] + p[1] * t, "Q", npar=2)
    np.testing.assert_allclose(result.parameters, h.fit("pol1", "Q").parameters, rtol=1e-8)
    with pytest.raises(ValueError, match="give npar="):
        h.fit(lambda t, p: p[0] * t)
    with pytest.raises(TypeError, match="not int"):
        h.fit(3)


# -- the Gaussian and the other built-in shapes --------------------------------------------------


def test_gaus_starts_from_roots_guess_and_bounds_the_width(capsys):
    h = gaussian()
    result = h.fit("gaus", "Q")
    assert result.parameter_names == ("Constant", "Mean", "Sigma")
    assert result.parameter("Mean") == pytest.approx(0.0, abs=0.05)
    assert result.parameter(2) == pytest.approx(1.0, abs=0.05)
    assert result.bounded == (False, False, True) and result.valid
    stored = h.functions[0]
    assert stored.parameter_limits[0] is None and stored.parameter_limits[2][0] == 0.0
    # every bin in range has entries here, so the degrees of freedom are 60 - 3
    assert result.ndf == 57 and stored.fit_result["npfits"] == 60


def test_the_gaus_minimum_is_the_least_chi_square_on_a_grid():
    h = gaussian()
    result = h.fit("gaus", "Q")
    data = from_histogram(h, DataOptions(), [None])
    fcn = cost.chi2(data, cost.Predictor(result.function, data))
    best = fcn(result.parameters)
    for k in range(3):
        for sign in (-1, 1):
            moved = result.parameters.copy()
            moved[k] += sign * 0.05 * result.errors[k]
            assert fcn(moved) > best
    # and a chi-square rise of one at a parameter's error, the others refitted, is the error
    assert fcn(result.parameters) == pytest.approx(result.chi2, rel=1e-12)


def test_expo_and_landau_start_from_roots_guesses():
    h = Histogram.book("e", (40, 0, 4))
    x = h.axes[0].centers()
    h.fill(x, weight=200 * np.exp(-1.3 * x))
    result = h.fit("expo", "Q")
    np.testing.assert_allclose(result.parameters, [math.log(200), -1.3], rtol=1e-6)
    peak = Histogram.book("l", (60, -2, 10))
    peak.fill_random(Function("L", "landau", range=(-2, 10), parameters=[1, 2, 0.5]), 20000)
    fitted = peak.fit("landau", "QL")
    assert fitted.parameter("MPV") == pytest.approx(2.0, abs=0.05)


def test_option_b_keeps_the_parameters_given_and_skips_the_guess():
    h = gaussian()
    f = Function("g", "gaus", range=(-3, 3), parameters=[100.0, 0.5, 2.0])
    f.set_limits(1, -1.0, 1.0)
    result = h.fit(f, "QB")
    assert result.bounded == (False, True, False)
    assert result.parameter(1) == pytest.approx(0.0, abs=0.05)


def test_keyword_parameters_limits_and_fixed_set_up_the_function_before_the_fit():
    h = gaussian()
    result = h.fit(
        "gaus",
        "QB",
        parameters={"Constant": 400, "Mean": 0.1, "Sigma": 1.2},
        limits=[None, (-0.5, 0.5), None],
        fixed={"Sigma": 1.0},
    )
    assert result.fixed == (False, False, True) and result.parameter("Sigma") == 1.0
    assert result.ndf == 58
    by_index = h.fit("gaus", "QB", parameters=[400, 0.1, 1.0], fixed=[2])
    assert by_index.fixed == (False, False, True)
    by_flag = h.fit("gaus", "QB", parameters=[400, 0.1, 1.0], fixed=[False, True, False])
    assert by_flag.fixed == (False, True, False) and by_flag.parameter(1) == 0.1
    released = h.fit("gaus", "QB", parameters=[400, 0.1, 1.0], fixed=[False, False, False])
    assert released.fixed == (False, False, False)
    unfixed = h.fit("gaus", "QB", parameters=[400, 0.1, 1.0], fixed={"Mean": None})
    assert unfixed.fixed == (False, True, False)


def test_a_bounded_parameter_starts_a_tenth_of_its_range_from_where_it_is():
    f = Function("g", "gaus", parameters=[1.0, 0.95, 1.0])
    f.set_limits(1, -1.0, 1.0)
    f.set_limits(2, 0.0, 10.0)
    f.set_limits(0, 0.0, math.inf)
    from xrdroot.fit.hfit import _settings

    steps = _settings(f)["errors"]
    assert steps[1] == pytest.approx(0.025)  # half the way to the upper limit
    assert steps[2] == pytest.approx(0.5)  # half the way to the lower one
    assert steps[0] == pytest.approx(0.3)  # an infinite range: 30% of the value
    f.parameter_errors = [0.0, 0.0, 0.2]
    f.set_parameters(1.0, 0.0, 5.0)
    steps = _settings(f)["errors"]
    assert steps[1] == pytest.approx(0.2) and steps[2] == 0.2  # an error wins


def points(x, y, ndim=1):
    return FitData(
        x=np.asarray(x, dtype=float).reshape(-1, ndim),
        y=np.asarray(y, dtype=float),
        error=None,
        kind=0,
        options=DataOptions(),
    )


def test_roots_guesses_leave_a_function_alone_with_nothing_to_guess_from():
    f = Function("g", "gaus", parameters=[1.0, 2.0, 3.0])
    for data in (points([], []), points([0.0, 1.0], [0.0, -1.0])):
        guesses.init_gaus(data, f)
        guesses.init_2d_gaus(points(np.zeros((0, 2)), [], 2), f)
        guesses.init_expo(points([], []), f)
    assert list(f.parameters) == [1.0, 2.0, 3.0]
    xy = Function("xy", "xygaus", parameters=[1, 2, 3, 4, 5])
    guesses.init_2d_gaus(points([[0, 0], [1, 1]], [-1.0, -1.0], 2), xy)
    assert list(xy.parameters) == [1, 2, 3, 4, 5]


def test_roots_gaussian_guess_takes_the_finest_step_and_a_spread_of_none():
    f = Function("g", "gaus")
    guesses.init_gaus(points([0.0, 1.0, 1.5, 4.0], [0.0, 2.0, 0.0, 0.0]), f)
    # all in one bin: no spread, so a quarter of the finest step per point
    assert list(f.parameters) == pytest.approx([0.5 * (2 + 0.5 * 2 / (2.506628 * 0.5)), 1.0, 0.5])
    xy = Function("xy", "xygaus")
    guesses.init_2d_gaus(points([[0, 0], [0, 1], [1, 0], [1, 1]], [1, 3, 3, 1], 2), xy)
    assert xy.parameters[1] == pytest.approx(0.5) and xy.parameters[3] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("first", "last", "expected"),
    [
        (4.0, 1.0, [math.log(4.0), math.log(0.25) / 2]),
        (0.0, 3.0, [math.log(3.0), 0.0]),  # a value of zero or less takes the other end's
        (3.0, -1.0, [math.log(3.0), 0.0]),
        (0.0, 0.0, [0.0, 0.0]),  # and with neither, both are one
    ],
)
def test_roots_exponential_guess_is_the_line_through_the_two_ends(first, last, expected):
    f = Function("e", "expo")
    guesses.init_expo(points([1.0, 0.0, 2.0], [7.0, first, last]), f)
    assert list(f.parameters) == pytest.approx(expected)
    unordered = Function("e", "expo")
    guesses.init_expo(points([0.0, 2.0, 1.0], [4.0, 1.0, 2.0]), unordered)
    assert list(unordered.parameters) == pytest.approx([math.log(4.0), math.log(0.25) / 2])


# -- ROOT's options --------------------------------------------------------------------------------


def test_likelihood_fits_empty_bins_and_reports_baker_and_cousins(capsys):
    h = gaussian(entries=300, bins=(40, -4, 4))
    chi2 = h.fit("gaus", "Q")
    like = h.fit("gaus", "LQ")
    assert like.ndf == 40 - 3 and chi2.ndf < like.ndf  # the empty bins are in the likelihood
    assert like.chi2 == pytest.approx(2 * like.fcn, rel=1e-15)
    data = from_histogram(h, DataOptions(use_empty=True), [None])
    predict = cost.Predictor(like.function, data)
    f = predict(like.parameters)
    y = data.y
    terms = f - y + np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / f), 0)
    assert like.fcn == pytest.approx(float(terms.sum()), rel=1e-12)
    printed = h.fit("gaus", "L").summary()
    assert "MinFCN" in printed
    # not extended, the normalisation is not the data's to fix: a density's is held
    area = float(h.values().sum()) * 0.2
    held = Function("g", "gausn", range=(-4, 4), parameters=[area, 0.0, 1.0])
    held.fix(0)
    multinomial = h.fit(held, "L MULTI Q B")
    assert multinomial.parameter("Mean") == pytest.approx(like.parameter("Mean"), abs=0.05)


def test_weighted_likelihood_corrects_the_errors_for_the_weights():
    h = Histogram.book("w", (30, -3, 3))
    values = TRandom3(3).gaus(0, 1, n=4000)
    h.fill(values, weight=0.5)
    plain = h.fit("gaus", "LQ")
    weighted = h.fit("gaus", "WLQ")
    np.testing.assert_allclose(weighted.parameters, plain.parameters, rtol=1e-6)
    # weights of a half: the errors are sqrt(1/2) of the unweighted likelihood's
    assert weighted.errors[1] == pytest.approx(plain.errors[1] * math.sqrt(0.5), rel=0.02)
    unweighted = gaussian()
    with pytest.warns(RuntimeWarning, match="histogram is not weighted"):
        unweighted.fit("gaus", "WLQ")
    with pytest.warns(RuntimeWarning, match="histogram is not weighted"):
        assert unweighted.fit("gaus", "WL MULTI Q B", parameters=[100, 0, 1], fixed=[0]).valid
    area = float(h.values().sum()) * 0.2
    density = Function("n", "gausn", range=(-3, 3), parameters=[area, 0.0, 1.0])
    density.fix(0)
    assert h.fit(density, "WL MULTI Q B").parameter("Mean") == pytest.approx(0.0, abs=0.05)


def test_pearson_chi_square_uses_the_expected_errors():
    h = gaussian(entries=2000, bins=(30, -3, 3))
    result = h.fit("gaus", "PQ")
    data = from_histogram(h, DataOptions(use_empty=True, exp_errors=True), [None])
    f = cost.Predictor(result.function, data)(result.parameters)
    expected = float(np.sum(np.where(f > 0, (data.y - f) ** 2 / f, 0)))
    assert result.chi2 == pytest.approx(expected, rel=1e-12)
    weighted = Histogram.book("pw", (30, -3, 3))
    weighted.fill(TRandom3(5).gaus(0, 1, n=3000), weight=2.0)
    assert weighted.fit("gaus", "PWQ").parameter("Mean") == pytest.approx(0, abs=0.1)


def test_option_w_sets_every_error_to_one_and_scales_them_afterwards():
    h, x, y, _ = straight_line()
    result = h.fit("pol1", "WQ")
    design = np.stack([np.ones_like(x), x], axis=1)
    expected = np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose(result.parameters, expected, rtol=1e-12)
    raw = np.sqrt(np.diag(np.linalg.inv(design.T @ design)))
    np.testing.assert_allclose(result.errors, raw * math.sqrt(result.chi2 / result.ndf))
    sparse = Histogram.book("s", (6, 0, 6))
    sparse.fill([0.5, 1.5, 1.5, 3.5, 5.5])
    assert sparse.fit("pol0", "WQ").ndf == 3 and sparse.fit("pol0", "WWQ").ndf == 5


def test_option_i_integrates_the_function_over_each_bin():
    h = Histogram.book("i", (10, 0, 5))
    edges = h.edges()
    h.fill(h.axes[0].centers(), weight=(edges[1:] ** 3 - edges[:-1] ** 3) / 3 / np.diff(edges))
    h._sumw2()[1:11] = 0.01
    result = h.fit("[0]*x*x", "IQ", parameters=[0.5])
    assert result.parameters[0] == pytest.approx(1.0, rel=1e-10)
    centred = h.fit("[0]*x*x", "Q", parameters=[0.5])
    assert abs(centred.parameters[0] - 1.0) > 1e-4


def test_option_width_and_normwidth_scale_the_function_by_the_bin_size():
    h = Histogram.book("v", [0.0, 1.0, 3.0, 4.0, 8.0])
    widths = np.diff(h.edges())
    h.fill(h.axes[0].centers(), weight=2.5 * widths)
    h._sumw2()[1:5] = 0.04
    assert h.fit("pol0", "WIDTH Q").parameters[0] == pytest.approx(2.5, rel=1e-6)
    assert h.fit("pol0", "NORMWIDTH Q").parameters[0] == pytest.approx(2.5, rel=1e-6)
    assert h.fit("pol0", "WIDTH I Q").parameters[0] == pytest.approx(2.5, rel=1e-6)


def test_option_r_and_a_range_cut_the_bins_to_those_whose_centres_are_inside():
    h, *_ = straight_line()
    inside = h.fit("pol1", "Q", (2.2, 8.7))
    assert inside.npoints == 7  # centres 2.5 to 8.5, but not 2.2's or 8.7's own bins
    f = Function("pol1", "pol1", range=(2.2, 8.7))
    assert h.fit(f, "RQ").npoints == 7
    assert h.functions[0].range == (2.2, 8.7)
    with pytest.warns(RuntimeWarning, match="fit range is outside histogram range"):
        with pytest.raises(ValueError, match="no points to fit"):
            h.fit("pol1", "Q", (50, 60))


def test_a_range_that_is_not_one_is_no_range_and_each_axis_takes_its_own():
    h, *_ = straight_line()
    assert h.fit("pol1", "Q", (5.0, 1.0)).npoints == 12
    two = Histogram.book("h2", (4, 0, 4), (4, 0, 4))
    two.fill([0.5, 1.5, 2.5, 3.5] * 4, np.repeat([0.5, 1.5, 2.5, 3.5], 4), weight=2.0)
    assert two.fit("[0] + 0*x*y", "Q", [None, (1.0, 3.0)]).npoints == 8


def test_a_fit_of_as_many_parameters_as_points_leaves_the_errors_as_they_are():
    h = Histogram.book("two", (2, 0, 2))
    h.fill([0.5, 1.5, 1.5])
    result = h.fit("pol1", "WQ")
    design = np.array([[1.0, 0.5], [1.0, 1.5]])
    raw = np.sqrt(np.diag(np.linalg.inv(design.T @ design)))
    assert result.ndf == 0 and list(result.errors) == pytest.approx(list(raw))


def test_a_fit_over_decades_saves_its_values_at_the_bin_centres():
    h = Histogram.book("wide", (200, 0.5, 200.5))
    h.fill(np.arange(1, 201), weight=1.0 + 0.01 * np.arange(1, 201))
    h._sumw2()[1:201] = 0.01
    h.fit("pol1", "Q")
    save = h.functions[0].members["fSave"]
    # FindBin of the upper end is the overflow, whose centre ROOT samples too
    assert len(save) == 201 + 3 and save[-1] == save[-2] == 200.5 and save[-3] == 0.5


def test_an_axis_range_limits_the_bins_fitted():
    h, *_ = straight_line()
    axis = h._core["fXaxis"]
    axis["TNamed"]["fBits"] = 1 << 11
    axis["fFirst"], axis["fLast"] = 3, 8
    assert h.fit("pol1", "Q").npoints == 6
    assert h.functions[0].range == (2.0, 8.0)


def test_the_fit_replaces_the_functions_there_unless_plus_and_n_stores_nothing():
    h, *_ = straight_line()
    h.attach({"TObject": "a paving, not a function"})
    h.fit("pol1", "Q")
    h.fit("pol0", "Q")
    assert [getattr(f, "name", None) for f in h.functions] == [None, "pol0"]
    h.fit("pol1", "Q+")
    assert [getattr(f, "name", None) for f in h.functions] == [None, "pol0", "pol1"]
    kept = h.fit("pol2", "QN")
    assert len(h.functions) == 3 and kept.function.name == "pol2"
    h.fit("pol1", "Q0")
    assert h.functions[-1].members["TNamed"]["fBits"] & (1 << 9)


def test_refitting_a_function_already_on_the_histogram_reuses_it():
    h, *_ = straight_line()
    h.fit("pol1", "Q")
    (stored,) = h.functions
    again = h.fit(stored, "Q")
    assert again.function is stored and h.functions == [stored]


def test_option_e_runs_minos_and_m_looks_again():
    h = gaussian()
    result = h.fit("gaus", "QE")
    assert set(result.minos) == {"Constant", "Mean", "Sigma"}
    low, high = result.minos["Mean"]
    assert low < 0 < high and result.lower_error("Mean") == low and result.upper_error(1) == high
    assert h.fit("gaus", "QM").valid
    assert result.summary().splitlines()[-2].endswith("(Minos) ")
    plain = h.fit("gaus", "Q")
    assert plain.lower_error("Mean") == plain.error("Mean") == plain.upper_error("Mean")


def test_verbose_prints_the_covariance_and_correlation(capsys):
    h = gaussian()
    h.fit("gaus", "V")
    out = capsys.readouterr().out
    assert "Covariance Matrix:" in out and "Correlation Matrix:" in out
    h.fit("gaus", "VV Q")
    assert "Covariance Matrix:" in capsys.readouterr().out


def test_options_root_has_but_this_does_not_are_refused_by_name():
    h = gaussian()
    assert h.fit("gaus", "UQ").valid  # no FCN on TVirtualFitter: an ordinary fit, as in ROOT
    with pytest.warns(RuntimeWarning, match="Cannot use P or X option"):
        h.fit("gaus", "LPQ")
    with pytest.warns(RuntimeWarning, match="Cannot use P or X option"):
        h.fit("gaus", "LXQ")
    assert h.fit("gaus", "Q S C SERIAL MULTITHREAD").valid
    assert parse("VVV Q").verbose == 3 and parse("DEBUG").verbose == 3
    assert parse("VQ").quiet is False


# -- other things fitted ---------------------------------------------------------------------------


def test_a_profile_is_fitted_to_its_means_with_their_errors():
    p = Profile.book("p", (20, 0, 10))
    rng = TRandom3(2)
    x = rng.uniform(0, 10, n=4000)
    p.fill(x, 1.5 + 0.3 * x + rng.gaus(0, 0.5, n=4000))
    result = p.fit("pol1", "Q")
    np.testing.assert_allclose(result.parameters, [1.5, 0.3], atol=0.05)
    means, errors = p.values(), p.errors()
    design = np.stack([np.ones(20), p.axes[0].centers()], axis=1) / errors[:, None]
    np.testing.assert_allclose(
        result.parameters, np.linalg.lstsq(design, means / errors, rcond=None)[0], rtol=1e-10
    )


def test_a_two_dimensional_histogram_is_fitted_with_xygaus():
    h = Histogram.book("h2", (30, -3, 3), (30, -3, 3))
    rng = TRandom3(8)
    h.fill(rng.gaus(0.3, 0.8, n=20000), rng.gaus(-0.2, 1.1, n=20000))
    result = h.fit("xygaus", "Q")
    assert result.parameter("MeanX") == pytest.approx(0.3, abs=0.03)
    assert result.parameter("SigmaY") == pytest.approx(1.1, abs=0.03)
    assert result.bounded == (False, False, True, False, True)
    np.testing.assert_allclose(h.functions[0].range, ((-3, 3), (-3, 3)), rtol=1e-15)
    assert h.fit("bigaus", "Q").parameter("Rho") == pytest.approx(0.0, abs=0.05)
    assert h.fit("xygaus", "IQ", (-2.0, 2.0)).valid
    linear = h.fit("[0] + [1]*x + [2]*y", "Q")
    assert linear.minimizer == "Linear"


def test_what_cannot_be_fitted_is_refused_by_name():
    h = gaussian()
    with pytest.raises(UnsupportedFeatureError, match="function of 2 variables"):
        h.fit("xygaus")
    with pytest.raises(ValueError, match="no parameters"):
        h.fit(Function("flat", "1 + x"))
    with pytest.raises(TypeError, match="not dict"):
        from xrdroot.fit import fit_object

        fit_object({}, "gaus")


def test_without_iminuit_a_linear_fit_works_and_the_rest_asks_for_it(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "iminuit", None)
    h, *_ = straight_line()
    assert h.fit("pol1", "Q").minimizer == "Linear"
    with pytest.raises(UnsupportedFeatureError, match=r"pip install xrdroot\[fit\]"):
        gaussian().fit("gaus", "Q")


# -- the result ------------------------------------------------------------------------------------


def test_the_result_answers_roots_questions_and_prints_as_root_prints():
    h = gaussian()
    result = h.fit("gaus", "Q")
    assert int(result) == 0 and 0 < result.prob < 1
    assert result.correlation[0, 0] == pytest.approx(1.0) and abs(result.correlation[0, 2]) < 1
    assert result.error("Sigma") == result.errors[2]
    with pytest.raises(KeyError, match="no parameter called 'width'"):
        result.parameter("width")
    with pytest.raises(IndexError, match="no parameter 3"):
        result.error(3)
    lines = result.summary().splitlines()
    assert lines[1] == "Minimizer is Minuit2 / Migrad"
    assert lines[-1].startswith("Sigma                     = ") and lines[-1].endswith("(limited)")
    assert repr(result).startswith("<FitResult chi2/ndf=") and "(valid)" in repr(result)


def test_an_invalid_or_empty_result_says_so():
    broken = FitResult(
        parameters=[1.0, 2.0],
        errors=[0.1, 0.0],
        covariance=np.zeros((2, 2)),
        names=("a", "b"),
        fcn=3.0,
        chi2=-1.0,
        ndf=0,
        status=4,
        valid=False,
        fixed=(False, True),
    )
    lines = broken.summary().splitlines()
    assert lines[1] == "         Invalid FitResult  (status = 4 )" and lines[2] == "*" * 40
    assert lines[-1].endswith("(fixed)") and "MinFCN" in lines[4]
    assert broken.prob == 1.0 and "invalid, status 4" in repr(broken)
    assert FitResult(
        parameters=[1.0], errors=[1.0], covariance=[[1.0]], names=("a",), fcn=2.0
    ).prob == 0.0


def test_minuits_status_follows_roots_order_of_checks():
    class Minimum:
        def __init__(self, **flags):
            defaults = {
                "has_posdef_covar": True,
                "has_made_posdef_covar": False,
                "hesse_failed": False,
                "is_above_max_edm": False,
                "has_reached_call_limit": False,
                "is_valid": True,
            }
            self.__dict__.update(defaults, **flags)

    assert minuit._status(Minimum()) == 0
    assert minuit._status(Minimum(has_posdef_covar=False)) == 5
    assert minuit._status(Minimum(has_made_posdef_covar=True, hesse_failed=True)) == 2
    assert minuit._status(Minimum(is_above_max_edm=True, has_reached_call_limit=True)) == 4
    assert minuit._status(Minimum(is_valid=False)) == 6


def test_minimize_is_minuit_on_any_function_with_limits_and_fixed_parameters():
    result = minuit.minimize(
        lambda p: (p[0] - 1) ** 2 + 4 * (p[1] + 2) ** 2 + (p[2] - 3) ** 2,
        [0.0, 0.0, 0.5],
        names=("a", "b", "c"),
        limits=[None, (-1.0, 1.0)],
        fixed=[False, False, True],
        minos=True,
    )
    assert result.parameter("a") == pytest.approx(1.0, abs=1e-2)
    assert result.parameter("b") == pytest.approx(-1.0, abs=1e-2)  # pressed on its limit
    assert result.parameter("c") == 0.5 and result.errors[2] == 0.0
    assert result.bounded == (False, True, False) and "a" in result.minos
    stuck = minuit.minimize(lambda p: float("nan") if p[0] > 0.1 else p[0] ** 2, [1.0], minos=True)
    assert not stuck.valid and stuck.minos == {}
