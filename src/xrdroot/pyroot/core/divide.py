"""``TGraphAsymmErrors::Divide``: a point per bin, the ratio of two histograms and its interval.

ROOT divides a histogram of what passed by one of everything tried, and
gives each bin with anything tried a point: the efficiency ``p / t`` with an
interval by one of ``TEfficiency``'s methods - Clopper-Pearson by default,
``n`` the normal approximation, ``w`` Wilson's, ``ac`` Agresti-Coull's,
``midp`` the mid-P interval, ``b(a,b)`` a Bayesian one with a Beta prior -
at ``cl=`` a confidence level, one sigma unless told. With ``pois`` it is
instead the ratio of two Poisson counts, ``p / t``, its interval the
binomial one on ``p / (p + t)`` turned into a ratio. Histograms filled with
weights are counted as ROOT counts them: by their effective entries,
``w² / w2``, for a Poisson ratio, and by the normal approximation otherwise.
This is that algorithm, line for line, each skipped bin skipped as ROOT
skips it.
"""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from .distributions import normal_quantile
from .messages import message
from .rmath import beta_cdf_c, beta_pdf, beta_quantile

__all__ = ["divide"]

#: ``TEfficiency``'s default confidence level: one Gaussian sigma, to ROOT's digits.
DEFAULT_LEVEL = 0.682689492137
#: How close ``MidPInterval``'s bisection comes before it stops.
MIDP_TOLERANCE = 1e-9

Bound = Callable[[float, float, float, bool], float]


def clopper_pearson(total: float, passed: float, level: float, upper: bool) -> float:
    """``TEfficiency::ClopperPearson``: the exact interval's end, from the Beta quantile."""
    alpha = (1.0 - level) / 2
    if upper:
        return 1.0 if passed == total else beta_quantile(1 - alpha, passed + 1, total - passed)
    return 0.0 if passed == 0 else beta_quantile(alpha, passed, total - passed + 1.0)


def _clipped(centre: float, delta: float, upper: bool) -> float:
    """``centre`` plus or minus ``delta``, kept in ``[0, 1]``."""
    return min(centre + delta, 1.0) if upper else max(centre - delta, 0.0)


def normal(total: float, passed: float, level: float, upper: bool) -> float:
    """``TEfficiency::Normal``: the Gaussian approximation's end."""
    if total == 0:
        return 1.0 if upper else 0.0
    average = passed / total
    sigma = math.sqrt(average * (1 - average) / total)
    return _clipped(average, normal_quantile(1 - (1.0 - level) / 2, sigma), upper)


def wilson(total: float, passed: float, level: float, upper: bool) -> float:
    """``TEfficiency::Wilson``: the score interval's end."""
    if total == 0:
        return 1.0 if upper else 0.0
    average = passed / total
    kappa = normal_quantile(1 - (1.0 - level) / 2, 1)
    mode = (passed + 0.5 * kappa * kappa) / (total + kappa * kappa)
    delta = (
        kappa
        / (total + kappa * kappa)
        * math.sqrt(total * average * (1 - average) + kappa * kappa / 4)
    )
    return _clipped(mode, delta, upper)


def agresti_coull(total: float, passed: float, level: float, upper: bool) -> float:
    """``TEfficiency::AgrestiCoull``: the adjusted Wald interval's end."""
    kappa = normal_quantile(1 - (1.0 - level) / 2, 1)
    mode = (passed + 0.5 * kappa * kappa) / (total + kappa * kappa)
    delta = kappa * math.sqrt(mode * (1 - mode) / (total + kappa * kappa))
    return _clipped(mode, delta, upper)


def mid_p(total: float, passed: float, level: float, upper: bool) -> float:
    """``TEfficiency::MidPInterval``: the mid-P interval's end, by bisection.

    A count between zero and one is interpolated between theirs, as ROOT does.
    """
    if 0 < passed < 1:
        p0, p1 = mid_p(total, 0.0, level, upper), mid_p(total, 1.0, level, upper)
        return (p1 - p0) * passed + p0
    wanted = (1.0 - level) / 2 if upper else 1.0 - (1.0 - level) / 2
    pmin, pmax, p = 0.0, 1.0, 0.0
    while abs(pmax - pmin) > MIDP_TOLERANCE:
        p = (pmin + pmax) / 2
        v = 0.5 * beta_pdf(p, passed + 1.0, total - passed + 1) / (total + 1)
        if passed - 1 >= 0:
            v += beta_cdf_c(p, passed, total - passed + 1)
        pmin, pmax = (p, pmax) if v > wanted else (pmin, p)
    return p


@dataclass
class Choice:
    """What ``Divide``'s option asks for, as ROOT reads it, piece by piece."""

    bound: Bound = clopper_pearson
    level: float = DEFAULT_LEVEL
    #: A Bayesian interval's Beta prior, and whether its mode is quoted rather than its mean.
    prior: tuple[float, float] | None = None
    mode: bool = False
    pois: bool = False
    zeros: bool = False
    verbose: bool = False


#: The frequentist methods, each by the letters that ask for it, in the order ROOT tries them.
METHODS: tuple[tuple[str, Bound | None], ...] = (
    ("n", normal), ("cp", clopper_pearson), ("w", wilson), ("ac", agresti_coull),
    ("fc", None), ("midp", mid_p),
)  # fmt: skip


def _level(option: str, choice: Choice) -> str:
    """``cl=``: the confidence level, if it is one; ROOT leaves the number in the option."""
    found = re.search(r"cl=([-+0-9.e]*)", option)
    if found is None:
        return option
    try:
        level = float(found.group(1))
    except ValueError:
        level = -1.0
    if 0 < level < 1:
        choice.level = level
    else:
        message("Warning", "TGraphAsymmErrors::Divide", "given confidence level %.3lf is invalid",
                level)  # fmt: skip
    return option.replace("cl=", "")


def _shape(option: str, name: str, at: int) -> float:
    """One of ``b(a,b)``'s shapes: one, with ROOT's warning, where it is not above zero."""
    found = re.search(r"b\(([-+0-9.e]*),([-+0-9.e]*)\)", option)
    text = found.group(at + 1) if found else ""
    value = float(text) if text else 0.0
    if value > 0:
        return value
    message("Warning", "TGraphAsymmErrors::Divide",
            f"given shape parameter for {name} %.2lf is invalid", value)  # fmt: skip
    return 1.0


def _prior(option: str, choice: Choice) -> str:
    """``b(a,b)``: a Bayesian interval, with the Beta prior's shapes where they are valid."""
    alpha, beta = (_shape(option, name, at) for at, name in enumerate(("alpha", "beta")))
    choice.prior = (alpha, beta)
    option = option.replace("b(", "")
    choice.mode = "mode" in option
    option = option.replace("mode", "")
    if "sh" in option or (choice.mode and "cen" not in option):
        raise UnsupportedFeatureError(
            "TGraphAsymmErrors::Divide's shortest Bayesian interval is not worked out here; "
            "ask for the central one, with 'cen'."
        )
    return option


def choose(option: str) -> Choice:
    """``Divide``'s option read as ROOT reads it: lowered, and each piece taken out once read."""
    choice = Choice()
    option = str(option).lower()
    if "v" in option:
        option, choice.verbose = option.replace("v", ""), True
    option = _level(option, choice)
    if "b(" in option:
        option = _prior(option, choice)
    else:
        found = next(((word, bound) for word, bound in METHODS if word in option), None)
        if found is not None:
            if found[1] is None:
                raise UnsupportedFeatureError(
                    "TGraphAsymmErrors::Divide's Feldman-Cousins interval is not worked out "
                    "here; Clopper-Pearson ('cp') and mid-P ('midp') are exact intervals that are."
                )
            option, choice.bound = option.replace(found[0], ""), found[1]
    choice.pois = "pois" in option
    choice.zeros = "e0" in option.replace("pois", "")
    return choice


def sums(h: Any) -> tuple[float, float]:
    """The histogram's sum of weights and of their squares, flow bins and all, as ROOT adds them."""
    if h.GetSumw2N() > 0:
        weights, squares = 0.0, 0.0
        for i, square in enumerate(h.GetSumw2()[: h.GetNcells()].tolist()):
            weights += h.GetBinContent(i)
            squares += square
        return weights, squares
    weights = h.GetSumOfWeights()
    return weights, weights


def effective(passed: Any, total: Any) -> tuple[bool, tuple[float, float], tuple[float, float]]:
    """Whether either histogram has weights other than one, and both's sums."""
    psums, tsums = sums(passed), sums(total)
    weighted = abs(psums[0] - psums[1]) > 1e-6 or abs(tsums[0] - tsums[1]) > 1e-6
    return weighted, psums, tsums


def c_round(value: float) -> float:
    """``std::round``: to the nearest whole number, halves away from zero."""
    return math.copysign(math.floor(abs(value) + 0.5), value)


@dataclass
class Counts:
    """One bin as ``Divide`` counts it: what passed and what was tried, and their weights."""

    p: float = 0.0
    t: float = 0.0
    pw: float = 0.0
    pw2: float = 0.0
    tw: float = 0.0
    tw2: float = 0.0
    wratio: float = 1.0


def _square(h: Any, b: int, content: float) -> float:
    return float(h.GetSumw2()[b]) if h.GetSumw2N() > 0 else content


def _ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator`` as C divides doubles: by zero, an infinity or a NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(numerator) / np.float64(denominator))


def _effective_count(w: float, w2: float) -> float:
    """``w² / w2``: the entries a weighted bin counts as, none for an empty one."""
    return 0.0 if w == 0 and w2 == 0 else _ratio(w * w, w2)


def _weight_ratio(c: Counts, psums: tuple[float, float], tsums: tuple[float, float]) -> Any:
    """What the ratio of effective counts is scaled by to be the ratio of weights, or ``None``."""
    if c.pw > 0 and c.tw > 0:
        return _ratio(c.pw * c.t, c.p * c.tw)
    if c.pw == 0 and c.tw > 0:
        return _ratio(psums[1] * c.t, psums[0] * c.tw)
    if c.tw == 0 and c.pw > 0:
        return _ratio(c.pw * tsums[0], c.p * tsums[1])
    if c.p > 0:  # a negative weight's
        return _ratio(c.pw, c.p)
    return None


def _poisson_weights(c: Counts, psums: tuple[float, float], tsums: tuple[float, float]) -> bool:
    """A weighted bin's effective counts for a Poisson ratio, and their ratio's weight.

    False for a bin with nothing in either, which is skipped unless ``e0`` asks
    for it - and then keeps the weight the bin before it had, as ROOT's does.
    """
    c.p, c.t = _effective_count(c.pw, c.pw2), _effective_count(c.tw, c.tw2)
    wratio = _weight_ratio(c, psums, tsums)
    if wratio is not None:
        c.wratio = wratio
    c.t += c.p
    return wratio is not None


@dataclass
class Setup:
    """What every bin is divided with: the choice, whether weighted, and both histograms' sums."""

    choice: Choice
    weighted: bool
    psums: tuple[float, float]
    tsums: tuple[float, float]


def counted(passed: Any, total: Any, b: int, c: Counts, setup: Setup) -> bool:
    """Bin ``b``'s counts in ``c``; False for a bin ``Divide`` skips."""
    zeros = setup.choice.zeros
    if setup.weighted:
        c.tw, c.pw = total.GetBinContent(b), passed.GetBinContent(b)
        c.tw2, c.pw2 = _square(total, b, c.tw), _square(passed, b, c.pw)
        if setup.choice.pois:
            return _poisson_weights(c, setup.psums, setup.tsums) or zeros
        return c.tw > 0 or zeros
    c.t, c.p = c_round(total.GetBinContent(b)), c_round(passed.GetBinContent(b))
    if setup.choice.pois:
        c.t += c.p
    return c.t != 0.0 or zeros


def _bayesian(c: Counts, setup: Setup) -> tuple[float, float, float]:
    """The posterior's mean or mode, and its central interval."""
    alpha, beta = setup.choice.prior or (1.0, 1.0)
    binomial = setup.weighted and not setup.choice.pois
    if binomial and c.tw2 <= 0:
        eff = _ratio(c.pw, c.tw)
        return eff, eff, eff
    if binomial:
        norm = c.tw / c.tw2
        aa, bb = c.pw * norm + alpha, (c.tw - c.pw) * norm + beta
    else:
        aa, bb = c.p + alpha, c.t - c.p + beta
    eff = beta_mode(aa, bb) if setup.choice.mode else aa / (aa + bb)
    level = setup.choice.level
    return eff, beta_quantile((1 - level) / 2, aa, bb), beta_quantile((1 + level) / 2, aa, bb)


def beta_mode(a: float, b: float) -> float:
    """``TEfficiency::BetaMode``: the Beta distribution's peak, an end or the middle if flat."""
    if a <= 1 or b <= 1:
        return 0.0 if a < b else (1.0 if a > b else 0.5)
    return (a - 1.0) / (a + b - 2.0)


def _weighted_normal(c: Counts, level: float) -> tuple[float, float, float]:
    """A weighted bin's efficiency, with the normal approximation's interval of its variance."""
    if c.tw <= 0:
        return 0.0, 0.0, 0.0
    eff = c.pw / c.tw
    variance = (c.pw2 * (1.0 - 2 * eff) + c.tw2 * eff * eff) / (c.tw * c.tw)
    with np.errstate(invalid="ignore"):  # a negative variance is a NaN, as in C
        sigma = float(np.sqrt(variance))
    delta = normal_quantile(1.0 - 0.5 * (1.0 - level), sigma)
    return eff, max(eff - delta, 0.0), min(eff + delta, 1.0)


def interval(c: Counts, setup: Setup) -> tuple[float, float, float]:
    """The bin's efficiency - or ratio, for ``pois`` - and its interval's two ends."""
    choice = setup.choice
    if choice.prior is not None:
        eff, low, upper = _bayesian(c, setup)
    elif setup.weighted and not choice.pois:
        eff, low, upper = _weighted_normal(c, choice.level)
    else:
        eff = c.p / c.t if c.t != 0.0 else 0.0
        low, upper = (
            choice.bound(c.t, c.p, choice.level, False),
            choice.bound(c.t, c.p, choice.level, True),
        )
    if choice.pois:
        with np.errstate(divide="ignore", invalid="ignore"):
            eff, low, upper = (
                float(np.float64(v) / (1.0 - np.float64(v))) for v in (eff, low, upper)
            )
        if setup.weighted:
            eff, low, upper = eff * c.wratio, low * c.wratio, upper * c.wratio
    return eff, low, upper


#: A point ``Divide`` makes: ``x``, ``y``, and the bar's four lengths ``exl, exh, eyl, eyh``.
Point = tuple[float, float, float, float, float, float]


def _warn(setup: Setup) -> None:
    """ROOT's word that weights leave only the normal and Bayesian intervals."""
    choice = setup.choice
    if setup.weighted and not choice.pois and choice.prior is None and choice.bound is not normal:
        message(
            "Warning",
            "TGraphAsymmErrors::Divide",
            "Histograms have weights: only Normal or Bayesian error calculation is supported",
        )
        message("Info", "TGraphAsymmErrors::Divide",
                "Using now the Normal approximation for weighted histograms")  # fmt: skip


def _points(passed: Any, total: Any, setup: Setup) -> list[Point]:
    """A point for each bin ``Divide`` keeps whose efficiency is a finite number."""
    points: list[Point] = []
    counts = Counts()  # kept from bin to bin, as ROOT's are
    for b in range(1, passed.GetNbinsX() + 1):
        if not counted(passed, total, b, counts, setup):
            continue
        eff, low, upper = interval(counts, setup)
        if math.isfinite(eff):
            centre, edge = passed.GetBinCenter(b), passed.GetBinLowEdge(b)
            width = passed.GetBinWidth(b)
            points.append(
                (centre, eff, centre - edge, edge - centre + width, eff - low, upper - eff)
            )
    return points


def _report(points: list[Point], nbins: int, choice: Choice) -> None:
    """ROOT's word on skipped bins, and with ``v`` on what was made and how."""
    where = "TGraphAsymmErrors::Divide"
    if len(points) < nbins:
        message("Warning", where, "Number of graph points is different than histogram bins - "
                "%d points have been skipped", nbins - len(points))  # fmt: skip
    if choice.verbose:
        message("Info", where, "made a graph with %d points from %d bins", len(points), nbins)
        message("Info", where, "used confidence level: %.2lf\n", choice.level)
        if choice.prior is not None:
            message("Info", where, "used prior probability ~ beta(%.2lf,%.2lf)", *choice.prior)


def _binned_alike(passed: Any, total: Any) -> bool:
    """``TEfficiency::CheckBinning``: as many bins, each edge the same to a part in 10^15."""
    a, b = passed.GetXaxis(), total.GetXaxis()
    if a.GetNbins() != b.GetNbins():
        message("Info", "TROOT::TEfficiency::CheckBinning",
                "Histograms are not consistent: they have different number of bins")  # fmt: skip
        return False
    for i in range(1, a.GetNbins() + 2):
        x, y = a.GetBinLowEdge(i), b.GetBinLowEdge(i)
        if abs(x - y) > 0.5 * 1e-15 * (abs(x) + abs(y)) and abs(x - y) >= sys.float_info.min:
            message("Info", "TROOT::TEfficiency::CheckBinning",
                    "Histograms are not consistent: they have different bin edges")  # fmt: skip
            return False
    return True


def _entries_alike(passed: Any, total: Any) -> bool:
    """``TEfficiency::CheckEntries``: nothing passed that was not tried, flow bins and all."""
    for i in range(passed.GetNbinsX() + 2):
        if passed.GetBinContent(i) > total.GetBinContent(i):
            message(
                "Info", "TROOT::TEfficiency::CheckEntries",
                "Histograms are not consistent: passed bin content > total bin content",
            )  # fmt: skip
            return False
    return True


def consistent(passed: Any, total: Any, pois: bool) -> bool:
    """Whether ``Divide`` will divide these: alike in binning, and - unless a Poisson ratio - no
    bin with more passed than tried, each with ROOT's word if not."""
    if not _binned_alike(passed, total):
        if not pois:
            message("Error", "TROOT::TEfficiency::CheckConsistency",
                    "passed TEfficiency objects have different binning")  # fmt: skip
        return False
    if not pois and not _entries_alike(passed, total):
        message("Error", "TROOT::TEfficiency::CheckConsistency",
                "passed TEfficiency objects do not have consistent bin contents")  # fmt: skip
        return False
    return True


def divide(passed: Any, total: Any, option: str = "cp") -> list[Point] | None:
    """The points ``TGraphAsymmErrors::Divide(passed, total, option)`` makes, or ``None``.

    ``None`` is for histograms it will not divide, with ROOT's error said.
    """
    if passed.GetDimension() > 1 or total.GetDimension() > 1:
        message("Error", "TGraphAsymmErrors::Divide", "passed histograms are not one-dimensional")
        return None
    choice = choose(option)
    weighted, psums, tsums = effective(passed, total)
    setup = Setup(choice, weighted, psums, tsums)
    if choice.verbose and weighted:
        message("Info", "TGraphAsymmErrors::Divide",
                "weight will be considered in the Histogram Ratio")  # fmt: skip
    _warn(setup)
    if not consistent(passed, total, choice.pois):
        message("Error", "TGraphAsymmErrors::Divide", "passed histograms are not consistent")
        return None
    points = _points(passed, total, setup)
    _report(points, passed.GetNbinsX(), choice)
    return points
