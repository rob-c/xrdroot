"""``TF1::GetRandom``, ``TH1::FillRandom`` and the closed-form integrals under them.

The histograms ROOT filled are in ``test_fit_root.py``; here the arrays at a
time are held to ROOT's loops written out one draw at a time, on
``rootrandom``'s transcription of ``TRandom3``: the cumulative table, the
parabola in each interval, ``TMath::BinarySearch`` and the straight line
across a bin, statement for statement. The integrals are ROOT's closed forms,
checked against the numerical integral of the same density, and Cephes's
error function against the C library's.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import rootrandom
from xrdroot import Function, Histogram, TRandom3, UnsupportedFeatureError
from xrdroot.fillrandom import first_last
from xrdroot.function import analytic, numeric, special


def binary_search(table, n, value):
    """``TMath::BinarySearch`` as ROOT writes it, with ``std::lower_bound``."""
    at = int(np.searchsorted(table[:n], value, side="left"))
    return at if at < n and table[at] == value else at - 1


def roots_table(f):
    """``ComputeCdfTable``, statement for statement, as in TF1.cxx."""
    low, high = f.range
    npx = f._npx()
    dx = (high - low) / npx
    xx = [low + i * dx for i in range(npx)] + [high]
    integral = [0.0]
    for i in range(npx):
        integral.append(integral[-1] + abs(f.integral(xx[i], xx[i + 1], 0.0)))
    total = integral[npx]
    integral = [integral[0]] + [value / total for value in integral[1:]]
    alpha, beta, gamma = [], [], []
    for i in range(npx):
        r2 = integral[i + 1] - integral[i]
        r1 = f.integral(xx[i], xx[i] + 0.5 * dx, 0.0) / total
        r3 = 2 * r2 - 4 * r1
        g = r3 / (dx * dx) if abs(r3) > 1e-8 else 0.0
        beta.append(r2 / dx - g * dx)
        alpha.append(xx[i])
        gamma.append(2 * g)
    return npx, integral, alpha, beta, gamma


def roots_get_random(f, draws):
    """``GetRandom``, one ``Rndm()`` at a time, as in TF1.cxx."""
    npx, integral, alpha, beta, gamma = roots_table(f)
    out = []
    for r in draws:
        b = binary_search(np.array(integral), npx, r)
        rr = r - integral[b]
        if gamma[b] != 0:
            yy = (-beta[b] + math.sqrt(beta[b] * beta[b] + 2 * gamma[b] * rr)) / gamma[b]
        else:
            yy = rr / beta[b]
        out.append(alpha[b] + yy)
    return np.array(out)


def test_get_random_is_roots_table_and_parabolas_draw_for_draw():
    f = Function("g", "gaus(0) + expo(3)", range=(-3, 5), parameters=[1, 0.5, 0.7, -1, -0.3])
    mine = f.get_random(2000, rng=TRandom3(99))
    theirs = roots_get_random(f, rootrandom.draws(rootrandom.MT(99), 2000))
    np.testing.assert_array_equal(mine, theirs)


def test_get_random_of_a_shape_integrates_it_in_closed_form_and_one_number_is_a_float():
    f = Function("g", "gaus", range=(-4, 4), parameters=[3, 0.2, 0.9])
    theirs = roots_get_random(f, rootrandom.draws(rootrandom.MT(5), 300))
    np.testing.assert_array_equal(f.get_random(300, rng=TRandom3(5)), theirs)
    rng = TRandom3(5)
    assert f.get_random(rng=rng) == theirs[0] and isinstance(f.get_random(rng=rng), float)
    flat = Function("flat", "pol0", range=(0, 2), parameters=[1.0])  # no curvature at all
    assert np.all(np.abs(flat.get_random(100, rng=TRandom3(1)) - 1) <= 1)
    assert isinstance(f.get_random(), float)  # gRandom's


def test_get_random_in_a_range_draws_again_until_a_number_lands_inside():
    f = Function("g", "gaus", range=(-4, 4), parameters=[1, 0, 1])
    rng = TRandom3(3)
    found = f.get_random(1000, rng=rng, range=(0.5, 1.5))
    assert np.all((found >= 0.5) & (found <= 1.5))
    # each try is one Uniform(pmin, pmax): the first number is the first try inside
    replay = TRandom3(3)
    table_rng = f.get_random(1, rng=replay, range=(0.5, 1.5))
    assert table_rng[0] == found[0]
    assert 0.5 <= f.get_random(rng=TRandom3(3), range=(0.5, 1.5)) <= 1.5
    assert np.all(f.get_random(50, rng=TRandom3(4), range=(-9, 0.1)) <= 0.1)


def test_get_random_over_decades_tabulates_in_log_x_and_refuses_a_range_there():
    f = Function("p", "[0]/x", range=(1, 1e6), parameters=[1.0])
    values = f.get_random(2000, rng=TRandom3(2))
    assert np.all((values >= 1) & (values <= 1e6))
    # 1/x is flat in log x: the decades are equally full
    counts = np.histogram(np.log10(values), bins=6, range=(0, 6))[0]
    assert counts.min() > 250
    with pytest.raises(UnsupportedFeatureError, match="log10"):
        f.get_random(10, range=(10, 100))
    with pytest.raises(ValueError, match="integrates to zero"):
        Function("z", "0*x", range=(0, 1)).get_random(3)
    with pytest.raises(UnsupportedFeatureError, match="TF1's"):
        Function("xy", "x*y").get_random(3)


def roots_fill_random(h, f, draws):
    """``TH1::FillRandom(TF1*)``, one entry at a time, as in TH1.cxx."""
    axis = h.axes[0]
    first, last = first_last(h._core["fXaxis"])
    width = (axis.high - axis.low) / axis.nbins
    integral = [0.0]
    for b in range(first, last + 1):
        integral.append(
            integral[-1] + f.integral(axis.low + (b - 1) * width, axis.low + b * width, 0.0)
        )
    integral = [integral[0]] + [value / integral[-1] for value in integral[1:]]
    xs = []
    for r in draws:
        i = binary_search(np.array(integral), last - first + 1, r)
        low = axis.low + (i + first - 1) * width
        xs.append(low + width * (r - integral[i]) / (integral[i + 1] - integral[i]))
    return np.array(xs)


def test_fill_random_from_a_function_is_roots_loop_entry_for_entry():
    f = Function("f", "landau(0) + pol1(3)", range=(0, 20), parameters=[5, 4, 1, 0.1, 0.01])
    h = Histogram.book("h", (40, 0, 20))
    h.fill_random(f, 3000, rng=TRandom3(17))
    expected = Histogram.book("e", (40, 0, 20))
    expected.fill(roots_fill_random(h, f, rootrandom.draws(rootrandom.MT(17), 3000)))
    np.testing.assert_array_equal(h.values(flow=True), expected.values(flow=True))
    assert h._core["fTsumwx"] == expected._core["fTsumwx"]


def test_fill_random_honours_an_axis_range_and_takes_a_formula_by_name():
    h = Histogram.book("h", (10, 0, 10))
    axis = h._core["fXaxis"]
    axis["TNamed"]["fBits"] = 1 << 11
    axis["fFirst"], axis["fLast"] = 3, 5
    h.fill_random("pol0", 100, rng=TRandom3(1))
    assert h.values()[2:5].sum() == 100
    other = Histogram.book("o", (10, 0, 10))
    other.fill_random("x*x", 50, rng=TRandom3(1))  # a formula of no parameters, over the axis
    assert other.values().sum() == 50
    uneven = Histogram.book("u", [0.0, 1.0, 3.0, 4.0])
    uneven.fill_random(Function("f", "pol0", range=(0, 4), parameters=[1]), 400, rng=TRandom3(2))
    assert uneven.values().sum() == 400 and uneven.values()[1] > uneven.values()[0]


def test_fill_random_refuses_what_it_cannot_draw_from():
    h = Histogram.book("h", (10, -1, 1))
    with pytest.raises(ValueError, match="integrates to zero"):
        h.fill_random(Function("z", "[0]*x", parameters=[0.0]), 5)
    with pytest.raises(TypeError, match="not int"):
        h.fill_random(3, 5)
    with pytest.raises(ValueError, match="not a number of entries"):
        h.fill_random("gaus", -1)
    with pytest.raises(UnsupportedFeatureError, match="2 axes"):
        Histogram.book("h2", (2, 0, 1), (2, 0, 1)).fill_random("gaus", 3)
    with pytest.raises(UnsupportedFeatureError, match="function of 2 variables"):
        h.fill_random("xygaus", 3)


def test_fill_random_from_a_histogram_draws_each_entry_with_get_random():
    source = Histogram.new("s", [0, 1, 2, 3], [1.0, 3.0, 6.0])
    h = Histogram.book("h", (3, 0, 3))
    h.fill_random(source, 20, rng=TRandom3(4))
    xs = TRandom3(4).from_distribution([1.0, 3.0, 6.0], (0, 3), 20)
    expected = Histogram.book("e", (3, 0, 3))
    expected.fill(xs)
    np.testing.assert_array_equal(h.values(flow=True), expected.values(flow=True))
    assert h.entries == 20


def roots_poisson_fill(parent, count, rng):
    """``TH1::FillRandom(TH1*)``'s Poisson branch, one call at a time, as in TH1.cxx."""
    nbins = len(parent)
    sumw = float(np.sum(parent))
    cells, squares = np.zeros(nbins + 2), np.zeros(nbins + 2)
    for b in range(1, nbins + 1):
        drawn = rng.poisson(parent[b - 1] * count / sumw)
        cells[b] += drawn
        squares[b] += drawn  # fSumw2 takes the count itself
    made = int(cells.sum() + 0.5)
    for _ in range(made, count):
        found = 1 + int(nbins * rng.from_distribution(parent, (0, nbins)) / nbins)
        cells[found] += 1
        squares[found] += 1
    while made > count:
        found = 1 + int(rng.from_distribution(parent, (0, nbins)))
        if cells[found] > 0:
            cells[found] -= 1  # SetBinContent, which leaves fSumw2 alone
            made -= 1
    return cells, squares


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_fill_random_past_ten_a_bin_is_roots_poisson_counts_and_corrections(seed):
    parent = np.array([5.0, 20.0, 40.0, 25.0, 10.0])
    source = Histogram.new("s", np.arange(6.0), parent)
    h = Histogram.book("h", (5, 0, 5))
    h.sumw2()
    h.fill_random(source, 300, rng=TRandom3(seed))
    expected, squares = roots_poisson_fill(parent, 300, TRandom3(seed))
    np.testing.assert_array_equal(h.values(flow=True), expected)
    np.testing.assert_array_equal(h.variances(flow=True), squares)
    # ResetStats: the entries are the effective ones, sum(w)^2 / sum(w^2)
    assert h.values().sum() == 300
    assert h.entries == pytest.approx(300.0**2 / squares.sum(), rel=1e-15)


def test_fill_random_from_a_histogram_refuses_what_root_would_not_draw_from():
    h = Histogram.book("h", (3, 0, 3))
    with pytest.raises(ValueError, match="negative or NaN"):
        h.fill_random(Histogram.new("n", [0, 1, 2, 3], [1.0, -1.0, 1.0]), 5)
    with pytest.raises(ValueError, match="is empty"):
        h.fill_random(Histogram.book("e", (3, 0, 3)), 5)
    with pytest.raises(ValueError, match="binned the same"):
        h.fill_random(Histogram.new("o", [0, 1, 2, 4], [1.0, 1.0, 1.0]), 500)


# -- the closed forms -----------------------------------------------------------------------------


def test_cephes_error_function_is_the_c_librarys_to_rounding():
    for x in np.concatenate([np.linspace(-9, 9, 721), [-30.0, 30.0, 0.0]]):
        assert analytic.erf(float(x)) == pytest.approx(math.erf(x), rel=1e-15, abs=1e-300)
        assert analytic.erfc(float(x)) == pytest.approx(math.erfc(x), rel=1e-13, abs=1e-300)
    assert analytic.erfc(-30.0) == 2.0 and analytic.erfc(27.0) == 0.0
    assert analytic.erfc(-26.6) == 2.0  # past the last double below two, it is two


def test_the_landau_distribution_is_the_integral_of_its_density():
    for x in (-8.0, -3.0, 0.0, 2.0, 7.0, 30.0, 120.0, 500.0):
        numerical = numeric.integrate(lambda t: special.landau_pdf(t), -np.inf, x, 1e-10, 1e-10)
        assert analytic.landau_cdf(x) == pytest.approx(numerical, rel=2e-6, abs=1e-12)


@pytest.mark.parametrize(
    ("formula", "params"),
    [
        ("gaus", [2.0, 0.3, 1.1]),
        ("gausn", [2.0, 0.3, 1.1]),
        ("expo", [0.2, -0.7]),
        ("expo", [0.2, 0.0]),
        ("landau", [2.0, 1.0, 0.6]),
        ("landaun", [2.0, 1.0, 0.6]),
        ("pol3", [1.0, -2.0, 0.5, 0.25]),
    ],
)
def test_a_shapes_closed_form_integral_is_its_numerical_one(formula, params):
    f = Function("f", formula, range=(-2, 5), parameters=params)
    closed = f.integral(-1.5, 4.0)
    numerical = numeric.integrate(f.evaluate, -1.5, 4.0, 1e-12, 1e-12)
    assert closed == pytest.approx(numerical, rel=1e-6)


def test_a_width_that_is_not_positive_falls_back_to_the_numerical_integral():
    f = Function("f", "gaus", range=(-2, 2), parameters=[1.0, 0.0, -1.0])
    assert analytic.integral(100, [1.0, 0.0, -1.0], -1, 1, False) != analytic.integral(
        100, [1.0, 0.0, 1.0], -1, 1, False
    )
    assert math.isnan(analytic.integral(100, [1.0, 0.0, -1.0], -1, 1, False))
    assert math.isnan(analytic.integral(400, [1.0, 0.0, 0.0], -1, 1, False))
    assert math.isnan(analytic.integral(500, [1.0, 0.0, 1.0, 1.0, 2.0], -1, 1, False))
    assert f.integral(-1, 1) == pytest.approx(math.erf(1 / math.sqrt(2)) * math.sqrt(2 * math.pi))


def test_an_integral_to_infinity_is_taken_over_the_mapped_range():
    f = Function("f", "exp(-x*x)")
    assert f.integral(-np.inf, np.inf) == pytest.approx(math.sqrt(math.pi), rel=1e-10)
    assert f.integral(0, np.inf) == pytest.approx(math.sqrt(math.pi) / 2, rel=1e-10)
    assert f.integral(-np.inf, 0) == pytest.approx(math.sqrt(math.pi) / 2, rel=1e-10)
    assert f.integral(1, 1) == 0.0 and f.integral(1, 0) == -f.integral(0, 1)


def test_a_function_knows_which_shape_it_is_and_roots_number_for_it():
    assert Function("f", "gaus").number == 100 and Function("f", "gaus").predefined == "gaus"
    assert Function("f", "pol2").number == 302 and Function("f", "xygaus").number == 110
    assert Function("f", "gaus + pol0(3)").number == 0
    renamed = Function("f", "gaus")
    renamed.parameter_names = ("A", "B", "C")
    assert renamed.number == 100
    code = Function.from_callable("c", lambda x, p: p[0] * x, 1)
    assert code.predefined is None and code.number == 0
    normalised = Function("n", "gaus", range=(-5, 5), parameters=[1, 0, 1])
    normalised.normalized = True
    assert normalised.integral(-5, 5) == pytest.approx(1.0, rel=1e-9)
