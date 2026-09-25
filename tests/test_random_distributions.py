"""ROOT's distributions: the same draws, taken in the same order, made into the same numbers.

``TRandom`` turns ``Rndm()`` values into numbers by ROOT's own recipes, and
what has to hold is that ``r.gaus(n=1000)`` is exactly the thousand numbers a
thousand calls of ``gRandom->Gaus()`` give, and leaves the stream exactly
where they would. Each distribution is checked three ways: an array against
the same numbers asked for one at a time, which is what shows a rejection
algorithm took ROOT's draws and no others; against ROOT's arithmetic written
out on ``rootrandom``'s draws, with Python's ``math`` - the C library ROOT
calls; and, where go-hep's ``rrand`` pins numbers from its own transcription
of the same C++, against those.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

import rootrandom
from xrdroot import Histogram
from xrdroot.random import TRandom1, TRandom2, TRandom3, gauss, histogram, landau, poisson

#: go-hep's ``TestDistributionsGolden``, from ``NewRandom3(12345)``.
GOLDEN = {
    "gaus": [0.70912219448387948, 0.54603893859994956, -1.2614145879633725, -1.0795189929194748],
    "exp": [0.18245897029359362, 0.29089999114384701, 2.8770632223226791, 5.0869871321429851],
    "rannor": [-0.24324570567555465, 1.1105366745107808, 0.45495766594352971, -1.5799762331708527],
}

EDGES = [0.0, 0.5, 1.5, 1.75, 3.0, 6.0]
#: Every distribution, as a call for ``n`` numbers (or one, for ``None``).
CALLS = {
    "uniform": lambda r, n: r.uniform(n=n),
    "uniform(5)": lambda r, n: r.uniform(5.0, n=n),
    "uniform(-2, 5)": lambda r, n: r.uniform(-2.0, 5.0, n=n),
    "gaus": lambda r, n: r.gaus(n=n),
    "gaus(-3, 2.5)": lambda r, n: r.gaus(-3.0, 2.5, n=n),
    "rannor": lambda r, n: r.rannor(n=n),
    "exp": lambda r, n: r.exp(2.5, n=n),
    "integer": lambda r, n: r.integer(10, n=n),
    "breit_wigner": lambda r, n: r.breit_wigner(1.0, 2.0, n=n),
    "landau": lambda r, n: r.landau(n=n),
    "landau(2, 0.5)": lambda r, n: r.landau(2.0, 0.5, n=n),
    "circle": lambda r, n: r.circle(2.5, n=n),
    "sphere": lambda r, n: r.sphere(1.5, n=n),
    "binomial": lambda r, n: r.binomial(10, 0.3, n=n),
    "binomial(7, 1)": lambda r, n: r.binomial(7, 1.0, n=n),
    **{
        f"poisson({m})": functools.partial(lambda m, r, n: r.poisson(m, n=n), m)
        for m in (0.5, 3, 24.9, 25, 100, 5000, 2e9)
    },
    "poisson_d(3)": lambda r, n: r.poisson_d(3.0, n=n),
    "poisson_d(2e9)": lambda r, n: r.poisson_d(2e9, n=n),
    "from_distribution": lambda r, n: r.from_distribution([3, 0, 1, 5, 2], EDGES, n=n),
    "from_distribution(even)": lambda r, n: r.from_distribution([3, 0, 1, 5], (-2, 2), n=n),
}


def stacked(values):
    """One call's numbers, or a tuple of them, as one array."""
    return np.asarray(values, dtype=np.float64)


# -- go-hep's numbers -------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_the_distributions_draw_the_numbers_go_hep_pins(name):
    ones = TRandom3(12345)
    draw = {"gaus": ones.gaus, "exp": lambda: ones.exp(2.5), "rannor": lambda: ones.rannor()[0]}[
        name
    ]
    assert [draw() for _ in range(4)] == GOLDEN[name]
    many = TRandom3(12345)
    arrays = {
        "gaus": lambda: many.gaus(n=4),
        "exp": lambda: many.exp(2.5, n=4),
        "rannor": lambda: many.rannor(4)[0],
    }
    assert arrays[name]().tolist() == GOLDEN[name]


def test_poisson_draws_the_counts_go_hep_pins():
    assert TRandom3(12345).poisson(3, n=6).tolist() == [3, 1, 4, 7, 4, 0]
    ones = TRandom3(12345)
    assert [ones.poisson(3) for _ in range(6)] == [3, 1, 4, 7, 4, 0]


# -- an array is so many calls ----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CALLS))
def test_an_array_is_the_numbers_one_call_at_a_time_would_give(name):
    call = CALLS[name]
    many, ones = TRandom3(2024), TRandom3(2024)
    array = stacked(call(many, 1500))
    single = stacked([call(ones, None) for _ in range(1500)])
    assert np.array_equal(array, single.T if single.ndim == 2 else single)
    assert many.rndm() == ones.rndm()  # and the same draws were used up


@pytest.mark.parametrize("name", ["gaus", "poisson(3)", "poisson(100)", "sphere"])
@pytest.mark.parametrize("made", [TRandom1, TRandom2])
def test_the_distributions_draw_from_any_generator_the_same_way(made, name):
    call = CALLS[name]
    many, ones = made(77), made(77)
    array = stacked(call(many, 600))
    single = stacked([call(ones, None) for _ in range(600)])
    assert np.array_equal(array, single.T if single.ndim == 2 else single)


def test_a_few_numbers_and_many_are_made_the_same_way():
    for count in (1, 8, 9, 40):
        few, all_at_once = TRandom3(5), TRandom3(5)
        assert np.array_equal(few.gaus(n=count), all_at_once.gaus(n=count))
    assert np.array_equal(TRandom3(5).gaus(n=8), TRandom3(5).gaus(n=200)[:8])
    assert np.array_equal(TRandom3(5).poisson(40, n=8), TRandom3(5).poisson(40, n=200)[:8])


# -- ROOT's arithmetic on ROOT's draws --------------------------------------------------


def oracle_draws(count, seed=99):
    return rootrandom.draws(rootrandom.MT(seed), count)


def test_the_one_draw_distributions_are_roots_arithmetic_on_its_draws():
    u = oracle_draws(5000).tolist()
    r = TRandom3(99)
    assert r.uniform(-2.0, 5.0, n=1000).tolist() == [-2.0 + 7.0 * x for x in u[:1000]]
    assert r.exp(2.5, n=1000).tolist() == [-2.5 * math.log(x) for x in u[1000:2000]]
    assert r.integer(7, n=1000).tolist() == [int(7 * x) for x in u[2000:3000]]
    half = math.pi / 2
    expected = [1.0 + 0.5 * 2.0 * math.tan((2 * x - 1) * half) for x in u[3000:4000]]
    assert r.breit_wigner(1.0, 2.0, n=1000).tolist() == expected
    x, y = r.circle(2.5, n=1000)
    assert x.tolist() == [2.5 * math.cos(2 * math.pi * v) for v in u[4000:5000]]
    assert y.tolist() == [2.5 * math.sin(2 * math.pi * v) for v in u[4000:5000]]


def test_rannor_is_box_muller_on_two_draws_with_roots_constant():
    u = oracle_draws(2000)
    a, b = TRandom3(99).rannor(1000)
    radius = [math.sqrt(-2 * math.log(v)) for v in u[0::2].tolist()]
    angle = [v * 6.28318530717958623 for v in u[1::2].tolist()]
    assert a.tolist() == [r * math.sin(t) for r, t in zip(radius, angle)]
    assert b.tolist() == [r * math.cos(t) for r, t in zip(radius, angle)]


def ranlan(z, xi):
    """``landau_quantile`` as ROOT writes it, one number at a time."""
    f = landau.TABLE.tolist()
    u = 1000 * z
    i = int(u)
    u -= i
    if 70 <= i < 800:
        return xi * (f[i - 1] + u * (f[i] - f[i - 1]))
    if 7 <= i <= 980:
        return xi * (
            f[i - 1]
            + u * (f[i] - f[i - 1] - 0.25 * (1 - u) * (f[i + 1] - f[i] - f[i - 1] + f[i - 2]))
        )
    return xi * (ranlan_low(z) if i < 7 else ranlan_high(z))


def ranlan_low(z):
    v = math.log(z)
    u = 1 / v
    return (
        (0.99858950 + (3.45213058e1 + 1.70854528e1 * u) * u)
        / (1 + (3.41760202e1 + 4.01244582 * u) * u)
    ) * (-math.log(-0.91893853 - v) - 1)


def ranlan_high(z):
    u = 1 - z
    v = u * u
    if z <= 0.999:
        return (1.00060006 + 2.63991156e2 * u + 4.37320068e3 * v) / (
            (1 + 2.57368075e2 * u + 3.41448018e3 * v) * u
        )
    return (1.00001538 + 6.07514119e3 * u + 7.34266409e5 * v) / (
        (1 + 6.06511919e3 * u + 6.94021044e5 * v) * u
    )


def test_landau_reads_roots_table_the_way_root_reads_it():
    assert len(landau.TABLE) == 982
    assert landau.TABLE[5] == -2.244733 and landau.TABLE[981] == 59.103894
    z = np.concatenate(
        [
            oracle_draws(4000),
            [1e-5, 0.003, 0.0069, 0.007, 0.0699, 0.07, 0.7999, 0.8, 0.98, 0.981, 0.9985, 0.99999],
        ]
    )
    assert landau.quantile(z, 1.5).tolist() == [ranlan(v, 1.5) for v in z.tolist()]
    assert TRandom3(99).landau(2.0, 1.5, n=4000).tolist() == [
        2.0 + ranlan(v, 1.5) for v in z[:4000].tolist()
    ]


def test_sphere_throws_pairs_at_the_square_until_one_lands_in_the_circle():
    oracle = rootrandom.MT(99)
    expected = []
    for _ in range(500):
        r2 = 1.0
        while r2 > 0.25:
            a, b = oracle.rndm() - 0.5, oracle.rndm() - 0.5
            r2 = a * a + b * b
        scale = 8.0 * 1.5 * math.sqrt(0.25 - r2)
        expected.append((a * scale, b * scale, 1.5 * (-1.0 + 8.0 * r2)))
    generator = TRandom3(99)
    assert stacked(generator.sphere(1.5, n=500)).T.tolist() == [list(point) for point in expected]
    assert generator.rndm() == oracle.rndm()


def test_binomial_counts_the_draws_at_or_below_the_chance():
    u = oracle_draws(3000)
    assert (
        TRandom3(99).binomial(6, 0.4, n=500).tolist()
        == (u.reshape(500, 6) <= 0.4).sum(axis=1).tolist()
    )


def test_gaus_is_roots_loop_whatever_region_each_draw_lands_in():
    oracle = rootrandom.MT(4)
    expected = [gauss.one(oracle.rndm) for _ in range(30000)]
    generator = TRandom3(4)
    assert generator.gaus(n=30000).tolist() == expected
    assert generator.rndm() == oracle.rndm()


@pytest.mark.parametrize("mean", [0.2, 3.0, 17.5, 24.99])
def test_poisson_below_25_multiplies_draws_as_roots_loop_does(mean):
    oracle = rootrandom.MT(6)
    expected = [poisson.product_one(oracle.rndm, mean) for _ in range(3000)]
    generator = TRandom3(6)
    assert generator.poisson(mean, n=3000).tolist() == expected
    assert generator.rndm() == oracle.rndm()


@pytest.mark.parametrize("mean", [25.0, 88.5, 1e4, 9.9e8])
def test_poisson_from_25_rejects_as_roots_loop_does(mean):
    oracle = rootrandom.MT(6)
    expected = [poisson.rejection_one(oracle.rndm, mean) for _ in range(3000)]
    generator = TRandom3(6)
    assert generator.poisson(mean, n=3000).tolist() == expected
    assert generator.poisson_d(mean, n=3).tolist() == [
        poisson.rejection_one(oracle.rndm, mean) for _ in range(3)
    ]


def test_poisson_past_1e9_is_roots_rounded_gaussian():
    g = TRandom3(6).gaus(n=100)
    counts = TRandom3(6).poisson(4e9, n=100)
    assert counts.tolist() == [int(v * math.sqrt(4e9) + 4e9 + 0.5) for v in g.tolist()]
    assert TRandom3(6).poisson_d(4e9, n=100).tolist() == (g * math.sqrt(4e9) + 4e9 + 0.5).tolist()


def test_a_count_the_logarithms_mislead_about_is_made_draw_by_draw():
    u = oracle_draws(400)
    exact = rootrandom.MT(99)
    first = poisson.product_one(exact.rndm, 6.0)
    found, used = poisson.product(u, 50, 6.0, guess=11.0)
    assert found[0].tolist() == [first] and used == first + 1
    found, used = poisson.product(np.full(10, 0.5), 50, 20.0, guess=1.0)
    assert len(found[0]) == 0 and used == 0


def test_lngamma_is_the_c_librarys_where_there_is_one():
    assert poisson.c_lgamma(None) is math.lgamma
    assert poisson.c_lgamma("no-such-library-anywhere") is math.lgamma
    assert poisson.LGAMMA(6.0) == pytest.approx(math.log(120), rel=1e-15)


# -- what the numbers look like ---------------------------------------------------------


def test_the_numbers_are_distributed_as_their_names_say():
    r = TRandom3(1)
    g = r.gaus(10.0, 2.0, n=200_000)
    assert abs(g.mean() - 10) < 0.02 and abs(g.std() - 2) < 0.02
    assert abs(np.mean(r.exp(3.0, n=200_000)) - 3) < 0.03
    counts = r.poisson(3.0, n=200_000)
    assert abs(counts.mean() - 3) < 0.02 and abs(counts.var() - 3) < 0.05
    big = r.poisson(400.0, n=100_000)
    assert abs(big.mean() - 400) < 0.3 and abs(big.var() - 400) < 8
    assert abs(np.median(r.breit_wigner(5.0, 2.0, n=100_000)) - 5) < 0.03
    x, y, z = r.sphere(2.0, n=50_000)
    assert np.allclose(x * x + y * y + z * z, 4.0) and abs(z.mean()) < 0.03
    assert abs(np.median(r.landau(n=100_000)) - landau.TABLE[499]) < 0.06  # the median


# -- TH1::GetRandom ---------------------------------------------------------------------


def get_random(contents, edges, draws):
    """``TH1::GetRandom`` over a variably binned axis, one draw at a time, as ROOT writes it."""
    integral = [0.0]
    for y in contents:
        integral.append(integral[-1] + y)
    integral = [integral[0]] + [value / integral[-1] for value in integral[1:]]
    return [get_one(r1, integral, edges, len(contents)) for r1 in draws]


def get_one(r1, integral, edges, nbins):
    """One draw's number: ``TMath::BinarySearch``, then a straight line across the bin."""
    lo = next((k for k, v in enumerate(integral[:nbins]) if not v < r1), nbins)
    ibin = lo if lo < nbins and integral[lo] == r1 else lo - 1
    x = edges[ibin]
    if r1 > integral[ibin]:
        width = edges[ibin + 1] - edges[ibin]
        x += width * (r1 - integral[ibin]) / (integral[ibin + 1] - integral[ibin])
    return x


def test_a_histograms_shape_is_drawn_from_as_th1_getrandom_draws():
    contents = [0.0, 3.0, 0.0, 1.0, 5.0, 2.0]
    edges = [-1.0, 0.0, 0.5, 1.5, 1.75, 3.0, 6.0]
    got = TRandom3(3).from_distribution(contents, edges, n=2000)
    assert got.tolist() == get_random(contents, edges, oracle_draws(2000, seed=3).tolist())


def test_a_draw_landing_on_an_edge_of_the_integral_takes_that_bins_low_edge():
    integral = histogram.cumulative([0.0, 2.0, 0.0, 2.0])
    lows, widths = histogram.bins([0, 1, 2, 3, 4], 4)
    assert integral.tolist() == [0.0, 0.0, 0.5, 0.5, 1.0]
    assert histogram.inverse(np.array([0.5, 0.25, 0.75]), integral, lows, widths).tolist() == [
        2.0,
        1.5,
        3.5,
    ]


def test_an_even_axis_has_its_edges_worked_out_as_taxis_works_them_out():
    lows, widths = histogram.bins((0.1, 0.7), 3)
    step = (0.7 - 0.1) / 3.0
    assert lows.tolist() == [0.1, 0.1 + step, 0.1 + 2 * step] and widths.tolist() == [step] * 3
    h = Histogram.book("h", (3, 0.1, 0.7))
    assert [a.tolist() for a in histogram.bins(h.axes[0], 3)] == [lows.tolist(), widths.tolist()]
    v = Histogram.book("v", [0.0, 1.0, 4.0])
    assert [a.tolist() for a in histogram.bins(v.axes[0], 2)] == [[0.0, 1.0], [1.0, 3.0]]


def test_the_width_option_weights_each_bin_by_its_width():
    edges = [0.0, 1.0, 4.0]
    plain = TRandom3(3).from_distribution([1.0, 1.0], edges, n=500)
    wide = TRandom3(3).from_distribution([1.0, 1.0], edges, n=500, width=True)
    assert wide.tolist() == get_random([1.0, 3.0], edges, oracle_draws(500, seed=3).tolist())
    assert not np.array_equal(plain, wide)


def test_an_empty_histogram_gives_zeros_and_draws_nothing():
    r = TRandom3(3)
    assert r.from_distribution([0, 0, 0], (0, 1), n=4).tolist() == [0.0] * 4
    assert r.from_distribution([0, 0], (0, 1)) == 0.0
    assert r.rndm() == TRandom3(3).rndm()


def test_a_histogram_that_is_not_a_distribution_is_refused():
    with pytest.raises(ValueError, match="negative or NaN"):
        TRandom3().from_distribution([1, -1], (0, 1))
    with pytest.raises(ValueError, match="needs 3 edges"):
        TRandom3().from_distribution([1, 1], [0, 1, 2, 3])


# -- parameters -------------------------------------------------------------------------


def test_arrays_of_parameters_draw_one_number_each():
    means = np.array([[0.0, 10.0], [100.0, 1000.0]])
    drawn = TRandom3(5).gaus(means, 0.5)
    assert drawn.shape == (2, 2)
    assert np.array_equal(drawn, means + 0.5 * TRandom3(5).gaus(n=4).reshape(2, 2))
    assert (
        TRandom3(5).uniform(np.arange(3.0), n=3).tolist()
        == (np.arange(3.0) * TRandom3(5).rndm(3)).tolist()
    )
    assert TRandom3(5).integer([5, 10]).dtype == np.int64


def test_nothing_asked_for_is_an_empty_array_and_draws_nothing():
    r = TRandom3(5)
    for name, call in CALLS.items():
        assert stacked(call(r, 0)).size == 0, name
    assert r.rndm() == TRandom3(5).rndm()


def test_roots_degenerate_parameters_answer_zero_and_draw_nothing():
    r = TRandom3(5)
    assert r.poisson(0.0, n=3).tolist() == [0, 0, 0] and r.poisson(-2.0) == 0
    assert r.landau(1.0, 0.0, n=2).tolist() == [0.0, 0.0]
    assert r.binomial(5, 1.5) == 0 and r.binomial(5, -0.1) == 0 and r.binomial(0, 0.5) == 0
    assert r.rndm() == TRandom3(5).rndm()


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda r: r.gaus(np.zeros(3), n=5), ValueError, "n=5"),
        (lambda r: r.gaus(np.zeros((2, 5)), n=5), ValueError, "more numbers"),
        (lambda r: r.gaus(np.zeros(3), np.ones(2)), ValueError, "do not go together"),
        (lambda r: r.rndm(-1), ValueError, "cannot be -1"),
        (lambda r: r.rndm(2.5), TypeError, "whole number"),
        (lambda r: r.poisson([1.0, 2.0]), ValueError, "one number"),
        (lambda r: r.binomial(3, [0.1, 0.2]), ValueError, "one number"),
        (lambda r: r.landau(sigma=[1.0, 2.0]), ValueError, "one number"),
        (lambda r: r.poisson(float("nan")), ValueError, "NaN"),
        (lambda r: r.poisson(1e19), ValueError, "poisson_d"),
    ],
)
def test_parameters_a_call_cannot_use_are_refused_by_name(call, error, message):
    with pytest.raises(error, match=message):
        call(TRandom3())


def test_a_rejection_algorithm_that_finds_nothing_yet_looks_further():
    def decode(draws, left):
        return ([draws[:1]], 1) if len(draws) >= 500 else ([draws[:0]], 0)

    r = TRandom3(5)
    assert r._sequential(3, decode, 1, 0.0)[0].tolist() == TRandom3(5).rndm(3).tolist()
