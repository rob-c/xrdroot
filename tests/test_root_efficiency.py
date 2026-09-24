"""Efficiencies, and the Beta quantile their intervals are made of.

``tefficiency.root`` is ROOT's, from go-hep: efficiencies in one, two and
three dimensions, saved with ROOT's defaults. ``tconfidence-level.root``
holds one whose first two bins were given priors of their own.

The intervals are checked against go-hep's ``hbook.BinomialInterval``, which
works them out with gonum's inverse incomplete Beta function - code that
shares nothing with this - at both levels anyone quotes. They agree to
better than one part in 10^10; the Beta quantile agrees with gonum's to one
part in 10^13.
"""

from __future__ import annotations

import copy
import math
import pathlib

import numpy as np
import pytest

from xrdroot import Efficiency, FormatError, Histogram, UnsupportedFeatureError, open_root
from xrdroot.efficiency import (
    BIN_PRIOR,
    POSTERIOR_MODE,
    SHORTEST_INTERVAL,
    USE_WEIGHTS,
    beta_quantile,
    regularized_beta,
)

DATA = pathlib.Path(__file__).parent / "data"
ONE_SIGMA = 0.682689492137

#: gonum's ``InvRegIncBeta``: probability, the two shapes, the quantile.
QUANTILES = [
    (0.1586552539314571, 3, 8, 0.14167190110718003),
    (0.8413447460685429, 4, 7, 0.50826248199025259),
    (0.025, 0.5, 10.5, 4.789043315758151e-05),
    (0.975, 10.5, 0.5, 0.99995210956684244),
    (0.5, 2, 2, 0.5),
    (0.05, 50, 150, 0.2012088727675593),
    (0.999, 1.5, 2.5, 0.95220222948600797),
    (0.15865525393145702, 499.5, 501.5, 0.48319754442753848),
    (0.8413447460685429, 1, 100, 0.018241783627291928),
]

#: go-hep's intervals for 47 of 111, 0 of 20, 5 of 5 and 1 of 3, per method and level.
INTERVALS = {
    ("clopper-pearson", ONE_SIGMA): [
        (0.37302190888507208, 0.47532840448617053),
        (0, 0.087941441648593438),
        (0.6919757765229485, 1),
        (0.055957972346085204, 0.74786803961821935),
    ],
    ("normal", ONE_SIGMA): [
        (0.37652540987324756, 0.47032143697359929),
        (0, 0),
        (1, 1),
        (0.061167806357472987, 0.60549886030919364),
    ],
    ("wilson", ONE_SIGMA): [
        (0.37741395881671735, 0.4708003268975679),
        (0, 0.047619047619031504),
        (0.83333333333338278, 1),
        (0.13564322306093723, 0.61435677693904056),
    ],
    ("agresti-coull", ONE_SIGMA): [
        (0.37740899798780581, 0.47080528772647945),
        (0, 0.057078006238350909),
        (0.80383279993566825, 1),
        (0.13293854086205903, 0.61706145913791877),
    ],
    ("jeffreys", ONE_SIGMA): [
        (0.37741850866920368, 0.47080067164504324),
        (0.00098880492186568885, 0.047871653225934313),
        (0.82788265088073087, 0.99619546987931529),
        (0.14173264027994309, 0.61566192798195885),
    ],
    ("uniform", ONE_SIGMA): [
        (0.37828727799191553, 0.47127500931165078),
        (0.0081926264380563635, 0.083934762972247945),
        (0.73577056526047024, 0.97161825201769181),
        (0.18530110612748399, 0.61840242550392011),
    ],
    ("clopper-pearson", 0.95): [
        (0.33019369710683361, 0.52084272150619071),
        (0, 0.16843347098308536),
        (0.47817624989501861, 1),
        (0.0084037586596126448, 0.9057006759497539),
    ],
    ("normal", 0.95): [
        (0.331505005918591, 0.51534184092825586),
        (0, 0),
        (1, 1),
        (0, 0.86676796403947876),
    ],
    ("wilson", 0.95): [
        (0.33558062075345008, 0.51638921510049285),
        (0, 0.16112515805281932),
        (0.56551753521682535, 1),
        (0.061491944720396263, 0.79234039919795229),
    ],
    ("agresti-coull", 0.95): [
        (0.33554555300829358, 0.51642428284564934),
        (0, 0.18980956054248876),
        (0.51094514036713989, 1),
        (0.056274614375314835, 0.79755772954303361),
    ],
    ("jeffreys", 0.95): [
        (0.33445168279896559, 0.51633083360454224),
        (2.4246478459242767e-05, 0.11663898290487534),
        (0.62062285770096082, 0.99990657939996042),
        (0.038747617785165181, 0.82326390286874263),
    ],
    ("uniform", 0.95): [
        (0.33547985502697975, 0.51660063908492337),
        (0.0012048834483635055, 0.16109761521907978),
        (0.54074187356009951, 0.99578925548551056),
        (0.067585986488542985, 0.80587955031675662),
    ],
}


def made(passed, total, **settings) -> Efficiency:
    """An efficiency of one bin per count, saved with the settings given."""
    edges = np.arange(len(total) + 1)
    bits = settings.pop("bits", 0)
    members = {
        "TNamed": {"fName": "eff", "fTitle": "made", "fBits": bits},
        "fPassedHistogram": Histogram.new("passed", edges, passed),
        "fTotalHistogram": Histogram.new("total", edges, total),
        "fStatisticOption": 0,
        "fConfLevel": ONE_SIGMA,
        "fBeta_alpha": 1.0,
        "fBeta_beta": 1.0,
        **settings,
    }
    return Efficiency("TEfficiency", members)


@pytest.fixture
def counts() -> Efficiency:
    return made([47, 0, 5, 1], [111, 20, 5, 3])


def test_the_beta_quantile_is_gonum_s_to_thirteen_places():
    for p, a, b, want in QUANTILES:
        assert beta_quantile(p, a, b) == pytest.approx(want, rel=1e-12, abs=1e-300)


def test_the_beta_quantile_has_the_closed_forms_it_should():
    assert beta_quantile(0.3, 1, 1) == pytest.approx(0.3, rel=1e-14)
    assert beta_quantile(0.3, 3, 1) == pytest.approx(0.3 ** (1 / 3), rel=1e-14)
    assert beta_quantile(0.3, 1, 4) == pytest.approx(1 - 0.7**0.25, rel=1e-14)
    assert (beta_quantile(0.0, 2, 3), beta_quantile(1.0, 2, 3)) == (0.0, 1.0)


def test_the_beta_quantile_inverts_the_incomplete_beta_function():
    for p, a, b in [(1e-12, 3, 3), (0.9, 1e-3, 5), (0.01, 5, 1e-3), (0.3, 1e5, 1e5)]:
        assert regularized_beta(beta_quantile(p, a, b), a, b) == pytest.approx(p, rel=1e-9)
    assert (regularized_beta(0.0, 2, 2), regularized_beta(1.0, 2, 2)) == (0.0, 1.0)


def test_a_quantile_too_close_to_zero_for_a_double_is_zero():
    assert beta_quantile(0.5, 1e-3, 5) == 0.0


def test_a_beta_of_no_shape_is_refused():
    with pytest.raises(ValueError, match="both shapes above zero"):
        beta_quantile(0.5, 0.0, 1.0)


def test_a_continued_fraction_that_never_settles_says_so(monkeypatch):
    import xrdroot.efficiency as efficiency

    monkeypatch.setattr(efficiency, "_TERMS", 2)
    with pytest.raises(ArithmeticError, match="did not converge"):
        regularized_beta(0.3, 200.0, 300.0)


def test_every_method_s_interval_is_go_hep_s(counts):
    for (method, level), want in INTERVALS.items():
        low, high = counts.intervals(level=level, method=method)
        assert low.tolist() == pytest.approx([pair[0] for pair in want], abs=1e-10), method
        assert high.tolist() == pytest.approx([pair[1] for pair in want], abs=1e-10), method


def test_the_bayesian_method_uses_the_prior_it_was_saved_with(counts):
    uniform = counts.intervals(method="uniform")
    bayesian = counts.intervals(method="bayesian")
    assert np.allclose(uniform, bayesian)  # alpha and beta of one are the uniform prior
    jeffreys = made([47, 0, 5, 1], [111, 20, 5, 3], fBeta_alpha=0.5, fBeta_beta=0.5)
    assert np.allclose(jeffreys.intervals(method="bayesian"), counts.intervals(method="jeffreys"))


def test_the_efficiency_is_passed_over_total_or_the_posterior_s_mean(counts):
    assert counts.values().tolist() == pytest.approx([47 / 111, 0, 1, 1 / 3])
    assert counts.values(method="jeffreys").tolist() == pytest.approx(
        [47.5 / 112, 0.5 / 21, 5.5 / 6, 1.5 / 4]
    )
    empty = made([0], [0])
    assert empty.values().tolist() == [0.0]
    assert empty.values(method="uniform").tolist() == [0.5]
    assert np.array_equal(empty.intervals(method="normal"), ([0.0], [1.0]))
    assert np.array_equal(empty.intervals(method="wilson"), ([0.0], [1.0]))


def test_the_posterior_mode_is_quoted_when_the_object_asks_for_it():
    modal = made([47, 0], [111, 20], bits=POSTERIOR_MODE, fStatisticOption=7)
    assert modal.values().tolist() == pytest.approx([47 / 111, 1 / 22])  # a peak, and none


def test_the_error_bars_are_the_distance_to_each_end(counts):
    low, high = counts.errors()
    lower, upper = counts.intervals()
    assert np.allclose(counts.values() - low, lower)
    assert np.allclose(counts.values() + high, upper)


def test_a_saved_efficiency_says_how_it_was_saved():
    with open_root(str(DATA / "tefficiency.root")) as handle:
        eff = handle["eff1"]
        assert isinstance(eff, Efficiency)
        assert (eff.method, eff.level, eff.name, eff.title) == (
            "clopper-pearson",
            0.682689492137,
            "eff1",
            "Eff1D",
        )
        assert repr(eff) == "<TEfficiency 'eff1' of 10 bins, by clopper-pearson>"
        assert eff.values()[0] == pytest.approx(47 / 111)
        low, high = eff.intervals()
        assert (low[0], high[0]) == pytest.approx(INTERVALS[("clopper-pearson", ONE_SIGMA)][0])
        assert eff.axes[0].nbins == 10
        values, edges = eff.to_numpy()
        assert len(edges) == 11 and np.array_equal(values, eff.values())
        assert handle["eff2"].values().shape == (10, 10)
        assert handle["eff3"].intervals(flow=True)[0].shape == (12, 12, 12)


def test_an_efficiency_with_priors_per_bin_uses_them():
    with open_root(str(DATA / "tconfidence-level.root")) as handle:
        eff = handle["eff"]
    assert eff.members["TNamed"]["fBits"] & BIN_PRIOR
    # Nothing was filled, so the posterior is the prior: bins 1 and 2 were given
    # (1, 2) and (2, 3), and the rest keep the object's own (1, 1).
    assert eff.values(method="bayesian")[:3].tolist() == pytest.approx([1 / 3, 2 / 5, 1 / 2])
    low, high = eff.intervals(method="bayesian")
    assert low[0] == pytest.approx(beta_quantile((1 - ONE_SIGMA) / 2, 1, 2))
    assert high[1] == pytest.approx(beta_quantile((1 + ONE_SIGMA) / 2, 2, 3))


def test_a_weighted_efficiency_is_refused_rather_than_approximated():
    heavy = made([1], [2], bits=USE_WEIGHTS)
    with pytest.raises(UnsupportedFeatureError, match="filled with weights"):
        heavy.values()


def test_a_shortest_bayesian_interval_is_refused():
    short = made([1], [2], bits=SHORTEST_INTERVAL)
    with pytest.raises(UnsupportedFeatureError, match="shortest Bayesian interval"):
        short.intervals(method="jeffreys")
    assert short.intervals(method="wilson")[0].shape == (1,)  # a frequentist one is fine


def test_the_methods_this_does_not_work_out_are_refused_by_name(counts):
    for method in ("feldman-cousins", "mid-p"):
        with pytest.raises(UnsupportedFeatureError, match=f"{method} intervals are not"):
            counts.intervals(method=method)
    with pytest.raises(ValueError, match="method='nonsense' is not one of"):
        counts.values(method="nonsense")
    with pytest.raises(ValueError, match="between 0 and 1"):
        counts.intervals(level=1.0)


def test_an_efficiency_saved_to_use_a_method_uses_it():
    wilson = made([47], [111], fStatisticOption=2, fConfLevel=0.95)
    assert wilson.method == "wilson"
    assert wilson.intervals()[0][0] == pytest.approx(INTERVALS[("wilson", 0.95)][0][0])
    assert made([1], [2], fStatisticOption=99).method == "clopper-pearson"


def test_an_efficiency_without_its_histograms_is_refused():
    with pytest.raises(FormatError, match="without its two histograms"):
        Efficiency("TEfficiency", {"TNamed": {"fName": "e", "fTitle": ""}})


def test_a_prior_of_nothing_has_the_whole_range_for_its_interval(counts):
    bare = made([0], [0], fBeta_alpha=0.0, fBeta_beta=1.0, fStatisticOption=7)
    assert np.array_equal(bare.intervals(), ([0.0], [1.0]))
    settings = copy.deepcopy(bare.members)
    assert math.isclose(settings["fConfLevel"], ONE_SIGMA)
