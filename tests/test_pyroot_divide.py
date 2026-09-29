"""``TGraphAsymmErrors::Divide``: a point per bin with its interval, by every method ROOT has.

The numbers are what ROOT 6.40 printed for the same histograms. The exact
intervals come from a Beta quantile that is xrdroot's own - a continued
fraction and a guarded Newton step, not ROOT's code - which agrees with
ROOT's to a part in 10^12, so they are held to that; the mid-P interval is a
bisection to 10^-9, so to that.
"""

from __future__ import annotations

import math

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError

#: How near an interval from xrdroot's Beta quantile is to ROOT's.
BETA = {"rel": 1e-11, "abs": 1e-12}


@pytest.fixture
def counts():
    """Four bins: 1 of 4 passed, 3 of 3, 0 of 2, and nothing tried in the last."""
    passed = ROOT.TH1D("p_divide", "", 4, 0, 4)
    total = ROOT.TH1D("t_divide", "", 4, 0, 4)
    for i, (p, t) in enumerate(zip((1, 3, 0, 0), (4, 3, 2, 0))):
        passed.SetBinContent(i + 1, p)
        total.SetBinContent(i + 1, t)
    return passed, total


def points(graph):
    return [
        (graph.GetX()[i], graph.GetY()[i], graph.GetErrorXlow(i), graph.GetErrorXhigh(i),
         graph.GetErrorYlow(i), graph.GetErrorYhigh(i))
        for i in range(graph.GetN())
    ]  # fmt: skip


def divided(passed, total, option):
    graph = ROOT.TGraphAsymmErrors()
    graph.Divide(passed, total, option)
    return points(graph)


def bars(made):
    """Each point's efficiency and its two bars' lengths."""
    return [(y, low, high) for _, y, _, _, low, high in made]


#: What ROOT 6.40 gives, for each option: each kept bin's efficiency and its bars.
ROOTS = {
    "cp": [(0.25, 0.20773089369631079, 0.36840242550392022), (1, 0.45864167529628797, 0),
           (0, 0, 0.60168447942428849)],
    "n": [(0.25, 0.21650635094607112, 0.21650635094607112), (1, 0, 0), (0, 0, 0)],
    "w": [(0.25, 0.14999999999998395, 0.24999999999995548), (1, 0.24999999999993316, 0),
          (0, 0, 0.33333333333325427)],
    "ac": [(0.25, 0.15493901531917426, 0.25493901531914576), (1, 0.29035945694146248, 0),
           (0, 0, 0.38183240812254882)],
    "b(1,1)": [(0.33333333333333331, 0.1866809585836017, 0.19091729267068719),
               (0.8, 0.16887757085038457, 0.15773089369631077),
               (0.25, 0.19404202765391482, 0.20864167529628791)],
    "b(2,3) mode cen": [(0.2857142857142857, 0.1067898998651966, 0.20385251701844109),
                        (0.66666666666666663, 0.21231882795180085, 0.1272495178783597),
                        (0.2, 0.07860188853736641, 0.25401705599827556)],
    "pois": [(0.25, 0.21484543285417612, 0.851947064271809),
             (1, 0.67883830075717844, 2.1136963167078253), (0, 0, 1.5105725193802002)],
    "cl=0.9 w": [(0.25, 0.1920926788001413, 0.39383199140149361),
                 (1, 0.47419557415751279, 0), (0, 0, 0.57496939099365374)],
}  # fmt: skip


@pytest.mark.parametrize("option", sorted(ROOTS))
def test_each_method_gives_rootss_efficiency_and_interval_for_every_bin_tried(
    counts, option, capsys
):
    made = divided(*counts, option)
    assert [(x, exl, exh) for x, _, exl, exh, _, _ in made] == [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5),
                                                                 (2.5, 0.5, 0.5)]  # fmt: skip
    assert bars(made) == [pytest.approx(row, **BETA) for row in ROOTS[option]]
    assert "1 points have been skipped" in capsys.readouterr().err


def test_the_mid_p_interval_is_bisected_to_a_billionth_as_roots_is(counts):
    expected = [(0.25, 0.17018419411033392, 0.28021316509693861),
                (1, 0.31793125066906214, -9.3132257461547852e-10),
                (0, -9.3132257461547852e-10, 0.43669678922742605)]  # fmt: skip
    assert bars(divided(*counts, "midp")) == [pytest.approx(row, abs=1e-9) for row in expected]


def test_a_count_between_nothing_and_one_is_given_the_mid_p_ends_between_theirs():
    from xrdroot.pyroot.core.divide import mid_p

    low0, low1 = mid_p(4.0, 0.0, 0.68, False), mid_p(4.0, 1.0, 0.68, False)
    assert mid_p(4.0, 0.25, 0.68, False) == pytest.approx(low0 + 0.25 * (low1 - low0))


def test_e0_keeps_a_bin_nothing_was_tried_in_with_nothing_for_its_efficiency(counts, capsys):
    assert bars(divided(*counts, "e0 n"))[3] == (0, 0, 1)
    ratio = bars(divided(*counts, "pois e0"))[3]
    assert ratio[:2] == (0, 0) and math.isinf(ratio[2])
    assert "skipped" not in capsys.readouterr().err


def test_verbose_says_what_it_made_and_prints_the_graph_as_root_does(counts, capsys):
    assert bars(divided(*counts, "v")) == [pytest.approx(row, **BETA) for row in ROOTS["cp"]]
    out, err = capsys.readouterr()
    assert out.splitlines()[0] == (
        "x[0]=0.5, y[0]=0.25, exl[0]=0.5, exh[0]=0.5, eyl[0]=0.207731, eyh[0]=0.368402"
    )
    assert "Info in <TGraphAsymmErrors::Divide>: made a graph with 3 points from 4 bins" in err
    assert "used confidence level: 0.68" in err
    divided(*counts, "b(1,1) v")
    assert "used prior probability ~ beta(1.00,1.00)" in capsys.readouterr().err


@pytest.fixture
def weighted():
    """Three bins filled with weights: 3 of 5, 0.5 of 1, and none of 2."""
    passed = ROOT.TH1D("wp_divide", "", 3, 0, 3)
    total = ROOT.TH1D("wt_divide", "", 3, 0, 3)
    for h in (passed, total):
        h.Sumw2()
    for x, w in ((0.5, 2), (0.5, 1), (1.5, 0.5)):
        passed.Fill(x, w)
    for x, w in ((0.5, 2), (0.5, 3), (1.5, 1), (2.5, 2)):
        total.Fill(x, w)
    return passed, total


#: What ROOT 6.40 gives the weighted histograms: the normal approximation of their
#: variance, a Beta posterior of their effective counts, and a Poisson ratio of those.
WEIGHTED = {
    "n": [(0.6, 0.38366652186494943, 0.38366652186494943),
          (0.5, 0.49999999999991124, 0.49999999999991118), (0, 0, 0)],
    "b(1,1)": [(0.5490196078431373, 0.25003619106654584, 0.2470127506065446),
               (0.5, 0.28424675144492623, 0.28424675144492617),
               (0.33333333333333331, 0.25058179807366743, 0.26835114609095517)],
    "pois": [(0.6, 0.4746257452563889, 2.2112775448036843),
             (0.5, 0.45489142885450773, 5.0421839719474759), (0, 0, 3.977230781300285)],
}  # fmt: skip


@pytest.mark.parametrize("option", sorted(WEIGHTED))
def test_weighted_histograms_are_divided_by_their_effective_counts_as_root_does(weighted, option):
    assert bars(divided(*weighted, option)) == [pytest.approx(r, **BETA) for r in WEIGHTED[option]]


def test_weights_leave_only_the_normal_and_bayesian_intervals_which_root_says(weighted, capsys):
    assert bars(divided(*weighted, "")) == [pytest.approx(r, **BETA) for r in WEIGHTED["n"]]
    err = capsys.readouterr().err
    assert "Histograms have weights: only Normal or Bayesian error calculation" in err
    assert "Using now the Normal approximation for weighted histograms" in err
    divided(*weighted, "n v")
    assert "weight will be considered in the Histogram Ratio" in capsys.readouterr().err


def test_the_intervals_of_nothing_tried_are_the_whole_of_zero_to_one():
    from xrdroot.pyroot.core.divide import beta_mode, normal, wilson

    assert (wilson(0, 0, 0.68, False), wilson(0, 0, 0.68, True)) == (0.0, 1.0)
    assert (normal(0, 0, 0.68, False), normal(0, 0, 0.68, True)) == (0.0, 1.0)
    assert (beta_mode(0.5, 2), beta_mode(2, 0.5), beta_mode(1, 1)) == (0.0, 1.0, 0.5)


def test_what_divide_will_not_do_it_refuses_by_name(counts):
    passed, total = counts
    with pytest.raises(UnsupportedFeatureError, match="Feldman-Cousins"):
        divided(passed, total, "fc")
    with pytest.raises(UnsupportedFeatureError, match="shortest Bayesian"):
        divided(passed, total, "b(1,1) sh")
    with pytest.raises(UnsupportedFeatureError, match="shortest Bayesian"):
        divided(passed, total, "b(1,1) mode")


def test_a_bad_level_or_prior_is_warned_of_and_rootss_default_used(counts, capsys):
    passed, total = counts
    assert bars(divided(passed, total, "cl=2")) == bars(divided(passed, total, "cp"))
    assert bars(divided(passed, total, "cl=x")) == bars(divided(passed, total, ""))
    assert bars(divided(passed, total, "b(-1,0)")) == bars(divided(passed, total, "b(1,1)"))
    err = capsys.readouterr().err
    assert "given confidence level 2.000 is invalid" in err
    assert "given confidence level -1.000 is invalid" in err
    assert "given shape parameter for alpha -1.00 is invalid" in err
    assert "given shape parameter for beta 0.00 is invalid" in err


def test_histograms_that_do_not_match_are_not_divided_and_root_says_why(counts, capsys):
    _passed, total = counts
    graph = ROOT.TGraphAsymmErrors(2, [1.0, 2.0], [3.0, 4.0])
    graph.Divide(ROOT.TH1D("few", "", 3, 0, 3), total, "")
    assert graph.GetN() == 2  # left as it was
    assert divided(ROOT.TH1D("moved", "", 4, 1, 5), total, "pois") == []
    assert divided(ROOT.TH2D("flat", "", 2, 0, 1, 2, 0, 1), total, "") == []
    more = ROOT.TH1D("more", "", 4, 0, 4)
    more.SetBinContent(4, 1)
    assert divided(more, total, "") == []
    err = capsys.readouterr().err
    said = ("they have different number of bins", "they have different bin edges",
            "passed TEfficiency objects have different binning",
            "passed histograms are not one-dimensional", "passed bin content > total bin content",
            "do not have consistent bin contents")  # fmt: skip
    assert [words for words in said if words not in err] == []
    assert err.count("passed histograms are not consistent") == 3


def test_a_poisson_ratio_of_weights_skips_bins_whose_ratio_is_not_a_number_as_root_does(capsys):
    passed, total = ROOT.TH1D("wp2", "", 4, 0, 4), ROOT.TH1D("wt2", "", 4, 0, 4)
    for h in (passed, total):
        h.Sumw2()
    passed.Fill(0.5, 2)  # passed where nothing was tried
    passed.Fill(1.5, -1)  # a negative weight, where nothing was tried either
    total.Fill(2.5, 0.5)  # and a last bin empty on both sides
    kept = pytest.approx((2.5, 0, 0.5, 0.5, 0, 53.029743750670463), **BETA)
    assert divided(passed, total, "pois") == [kept]
    assert "3 points have been skipped" in capsys.readouterr().err
    both = divided(passed, total, "pois e0")  # the empty bin too, its ratio unbounded above
    assert both[0] == kept and both[1] == (3.5, 0, 0.5, 0.5, 0, math.inf)


def test_a_weighted_bin_with_nothing_tried_is_kept_with_e0_as_root_keeps_it(weighted):
    passed, total = weighted
    total.SetBinContent(3, 0)
    total.SetBinError(3, 0)
    assert bars(divided(passed, total, "n e0"))[2] == (0, 0, 0)
    assert bars(divided(passed, total, "b(1,1) e0")) == [
        pytest.approx(row, **BETA) for row in WEIGHTED["b(1,1)"][:2]
    ]


def test_poisson_bin_errors_are_the_gamma_intervals_root_gives():
    h = ROOT.TH1D("pe", "", 3, 0, 3)
    for i, content in enumerate((5, 0, 2.5)):
        h.SetBinContent(i + 1, content)
    h.SetBinErrorOption(ROOT.TH1.kPoisson)
    found = [(h.GetBinErrorLow(i), h.GetBinErrorUp(i)) for i in (1, 2, 3)]
    assert found == [pytest.approx(pair, rel=1e-9) for pair in (
        (2.1596911439740243, 3.3824726512784409), (0, 1.8410216445772389),
        (1.7918145599845221, 2.1378596227967464))]  # fmt: skip
    h.SetBinErrorOption(ROOT.TH1.kPoisson2)
    assert (h.GetBinErrorLow(1), h.GetBinErrorUp(1)) == pytest.approx(
        (3.3765136098815791, 6.6683320793226706), rel=1e-9)  # fmt: skip


def test_a_negative_bin_forces_normal_errors_with_rootss_warning(capsys):
    h = ROOT.TH1D("ne", "", 2, 0, 2)
    h.SetBinContent(1, -4)
    h.SetBinErrorOption(ROOT.TH1.kPoisson)
    assert h.GetBinErrorUp(1) == h.GetBinError(1)
    assert h.GetBinErrorOption() == ROOT.TH1.kNormal
    assert (
        "Histogram has negative bin content-force usage to normal errors" in capsys.readouterr().err
    )
    h.Sumw2()
    h.Fill(1.5, 2.0)
    h.SetBinErrorOption(ROOT.TH1.kPoisson)
    assert h.GetBinErrorLow(2) == h.GetBinError(2)  # weighted: normal errors whatever it is told


def test_dividing_where_nothing_was_tried_anywhere_leaves_no_points():
    empty = ROOT.TH1D("none", "", 2, 0, 2)
    assert divided(empty, empty, "") == []
