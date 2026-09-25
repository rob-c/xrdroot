"""Comparing two histograms: ROOT's ``KolmogorovTest``, ``Chi2Test`` and ``Chi2TestX``.

Two histograms of the same thing never hold the same numbers, and the
question worth asking is whether they differ by more than their fluctuations
explain. ROOT answers it two ways. The Kolmogorov test walks both cumulative
distributions and takes the furthest they stray apart, which notices a shape
drifting where no single bin looks wrong; the chi-square test takes the bins
one at a time, weighing each difference by what each histogram's errors say
it could be - Gagunashvili's tests, for two counts, a count against a
weighted histogram, and two weighted ones.

Each is a port of ROOT's code rather than of its documentation, because the
documentation leaves out what the numbers depend on: which bins are skipped
and what that does to the degrees of freedom, the order the sums are added
in, the count nudged up by one where a formula would otherwise divide zero by
zero. ROOT prints an error and returns zero where a test cannot be made; this
refuses with the reason instead, since a zero is a p-value and would be read
as one.
"""

from __future__ import annotations

import math
import re
import warnings
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from .errors import UnsupportedFeatureError
from .filling import running
from .fillrandom import POISSON_PER_BIN, from_parent
from .moments import statistics
from .reshaping import _low_edge
from .stats import kolmogorov_prob, prob

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from .hist import Axis, Histogram

__all__ = ["Chi2Result", "chi2_test", "chi2_test_full", "kolmogorov_test"]

#: How many pseudo-experiments option ``"X"`` makes unless ``"X=n"`` says.
EXPERIMENTS = 1000

#: ``AreEqualRel``'s tolerance for two bin edges ``KolmogorovTest`` calls one.
EDGE_PRECISION = 1e-15

#: The smallest normal double, below which ``AreEqualRel`` calls any two equal.
TINY = 2.2250738585072014e-308


class Chi2Result(NamedTuple):
    """What ``Chi2TestX`` hands back: the chi-square, its degrees of freedom, and more.

    ``igood`` is ROOT's warning flag - one for a bin of the first histogram
    too thin for the test, two for one of the second, three for both - and
    ``residuals`` the normalised residual of every bin that took part, in
    ROOT's order of visiting them: x outermost.
    """

    chi2: float
    ndf: int
    igood: int
    p: float
    residuals: np.ndarray[Any, Any]


def _counts_only(*histograms: Histogram) -> None:
    for histogram in histograms:
        if histogram.kind == "MEAN":
            raise UnsupportedFeatureError(
                f"{histogram.name!r} is a profile, whose bins are means rather than counts, "
                f"and these tests compare counts; compare the profiles' projections instead"
            )


# -- the Kolmogorov test ----------------------------------------------------


def _same_axis(one: Histogram, other: Histogram) -> None:
    """Refuse what ``KolmogorovTest`` refuses: not one axis each, or not the same one."""
    _counts_only(one, other)
    if len(one.axes) != 1 or len(other.axes) != 1:
        raise ValueError(
            f"the Kolmogorov test compares histograms of one axis, and {one.name!r} has "
            f"{len(one.axes)} and {other.name!r} {len(other.axes)}"
        )
    mine, theirs = one.axes[0], other.axes[0]
    if mine.nbins != theirs.nbins:
        raise ValueError(
            f"{one.name!r} has {mine.nbins} bins and {other.name!r} {theirs.nbins}, and the "
            f"Kolmogorov test compares them bin by bin"
        )
    for bin in range(1, mine.nbins + 2):
        a, b = _low_edge(mine, bin), _low_edge(theirs, bin)
        if not (a == b or abs(a - b) <= 0.5 * EDGE_PRECISION * (abs(a) + abs(b))
                or abs(a - b) < TINY):  # fmt: skip
            raise ValueError(
                f"{one.name!r} and {other.name!r} have different bin edges - {a!r} against "
                f"{b!r} at bin {bin} - and the Kolmogorov test needs the same bins in both"
            )


class _Side(NamedTuple):
    """One histogram as the Kolmogorov test sees it: bins, total, and what it is worth."""

    contents: np.ndarray[Any, Any]
    total: float
    effective: float
    exact: bool


def _side(histogram: Histogram, first: int, last: int) -> _Side:
    """The bins taken, their sum, and the effective entries - or none, for exact bins.

    The errors are ``GetBinError``'s, squared back, and a histogram whose
    errors are all zero is being compared as a function would be: ROOT
    rescales only by the other one's entries.
    """
    contents = histogram._bins.astype(np.float64)[first : last + 1]
    errors = histogram._bin_errors()[first : last + 1]
    total, squares = running(0.0, contents), running(0.0, errors * errors)
    if total == 0:
        raise ValueError(
            f"{histogram.name!r} has nothing in the bins compared, and a Kolmogorov test "
            f"of an empty histogram is not a test"
        )
    return _Side(contents, total, total * total / squares if squares > 0 else 0.0, squares <= 0)


def _distance(one: np.ndarray[Any, Any], other: np.ndarray[Any, Any]) -> float:
    """The largest gap between two cumulative distributions, summed in turn as ROOT sums."""
    first, second = running(0.0, one), running(0.0, other)
    if first == 0 or second == 0:
        return 0.0
    gaps = np.abs(np.add.accumulate((1 / first) * one) - np.add.accumulate((1 / second) * other))
    return float(gaps.max())


def _z(dfmax: float, one: _Side, other: _Side) -> float:
    if one.exact:
        return dfmax * math.sqrt(other.effective)
    if other.exact:
        return dfmax * math.sqrt(one.effective)
    return dfmax * math.sqrt(one.effective * other.effective / (one.effective + other.effective))


def _with_normalisation(probability: float, one: _Side, other: _Side) -> tuple[float, float]:
    """Option ``"N"``: the shape's probability combined with the totals' (Eadie 11.6.2)."""
    apart = one.effective - other.effective
    totals = prob(apart * apart / (one.effective + other.effective), 1)
    if probability > 0 and totals > 0:
        return probability * (totals * (1 - math.log(probability * totals))), totals
    return 0.0, totals


def _experiments(option: str) -> int:
    """How many pseudo-experiments ``"X"`` or ``"X=n"`` asks for, as ROOT reads it."""
    found = re.search(r"X=(\d*)", option)
    if found is None:
        return EXPERIMENTS
    count = int(found.group(1) or 0)
    if count <= 0:
        warnings.warn(
            f"{count} is not a number of pseudo-experiments; ROOT makes 1000 instead, and so "
            f"does this",
            RuntimeWarning,
            stacklevel=4,
        )
        return EXPERIMENTS
    return count


def _toy(parent: np.ndarray[Any, Any], count: int, rng: Any, axis: Axis) -> np.ndarray[Any, Any]:
    """``FillRandom(&hparent, count)`` into an emptied copy: the bins it leaves, flow aside.

    Past ten entries a bin ROOT draws a Poisson count for every bin and
    then adds or takes away single entries until there are exactly
    ``count``; below that it draws every entry with ``GetRandom``. Both are
    ROOT's own, draw for draw (see :mod:`xrdroot.fillrandom`).
    """
    cells = np.zeros(axis.nbins + 2)
    if not parent.any():
        return cells[1:-1]
    if count > POISSON_PER_BIN * axis.nbins:
        from_parent(cells, None, axis, parent, axis, count, rng)
    else:
        np.add.at(cells, axis.find_bin(rng.from_distribution(parent, axis, count)), 1.0)
    return cells[1:-1]


def _pseudo_probability(
    inner: tuple[Any, Any], sides: tuple[_Side, _Side], dfmax: float, count: int, context: Any
) -> float:
    """Option ``"X"``: the fraction of pseudo-experiments that stray further than the data.

    The parent is the histogram of more effective entries - or the exact one,
    if either is - with any negative bin emptied, and only its bins on the
    axis, whatever the flow options; each pseudo-experiment draws as many
    entries as each side is worth, the first then the second, from
    ``gRandom`` unless another generator is given, and a side that is exact
    is the parent itself. When only the second is exact that parent is the
    first, and it is set against pseudo-experiments drawn from itself: that
    is ROOT's code.
    """
    rng, axis = context
    one, other = sides
    parent = inner[0] if one.exact or one.effective > other.effective else inner[1]
    shape = np.maximum(parent, 0.0)
    beyond = 0
    for _ in range(count):
        first = shape if one.exact else _toy(shape, int(one.effective), rng, axis)
        second = shape if other.exact else _toy(shape, int(other.effective), rng, axis)
        beyond += _distance(first, second) > dfmax
    return beyond / count


def _sides(one: Histogram, other: Histogram, opt: str) -> tuple[_Side, _Side]:
    """Both histograms over the bins compared, refusing two that both have no errors."""
    _same_axis(one, other)
    first = 0 if "U" in opt else 1
    last = one.axes[0].nbins + (1 if "O" in opt else 0)
    mine, theirs = _side(one, first, last), _side(other, first, last)
    if mine.exact and theirs.exact:
        raise ValueError(
            f"{one.name!r} and {other.name!r} both have errors of zero, and the Kolmogorov "
            f"test needs one of them to count entries"
        )
    return mine, theirs


def _probabilities(opt: str, mine: _Side, theirs: _Side, dfmax: float) -> tuple[float, ...]:
    """The probability, and with ``"N"`` the shape's and the totals' that went into it."""
    probability = kolmogorov_prob(_z(dfmax, mine, theirs))
    if "N" in opt and not (mine.exact or theirs.exact):
        combined, totals = _with_normalisation(probability, mine, theirs)
        return combined, probability, totals
    return probability, 0.0, 0.0


def kolmogorov_test(one: Histogram, other: Histogram, option: str, rng: Any = None) -> float:
    """``TH1::KolmogorovTest``: the probability that the two have the same shape.

    The options are ROOT's letters, in any order and either case: ``"U"``
    and ``"O"`` take the underflow and overflow in, ``"N"`` combines the
    shape's probability with the totals', ``"M"`` gives back the largest
    distance itself, ``"X"`` - or ``"X=n"`` - the fraction of ``n``
    pseudo-experiments straying further, and ``"D"`` prints ROOT's debug lines.
    """
    opt = option.upper()
    mine, theirs = _sides(one, other, opt)
    dfmax = _distance(mine.contents, theirs.contents)
    probability, shape, totals = _probabilities(opt, mine, theirs, dfmax)
    pseudo, count = 0.0, 0
    if "X" in opt:
        count = _experiments(opt)
        inner = [h._bins.astype(np.float64)[1 : h.axes[0].nbins + 1] for h in (one, other)]
        if rng is None:
            from .random import gRandom as rng
        pseudo = _pseudo_probability(
            (inner[0], inner[1]), (mine, theirs), dfmax, count, (rng, one.axes[0])
        )
    if "D" in opt:
        found = (probability, dfmax, shape, totals, pseudo, count)
        _debug(opt, [(one, mine), (other, theirs)], found)
    if "M" in opt:
        return dfmax
    return pseudo if "X" in opt else probability


def _debug(opt: str, sides: list[tuple[Any, _Side]], found: Any) -> None:
    """Option ``"D"``: the lines ROOT prints, in its words and its formats."""
    probability, dfmax, shape, totals, pseudo, count = found
    for number, (histogram, side) in enumerate(sides, 1):
        print(
            f" Kolmo Prob  h{number} = {histogram.name}, sum bin content ={side.total:g}  "
            f"effective entries ={side.effective:g}"
        )
    print(f" Kolmo Prob     = {probability:g}, Max Dist = {dfmax:g}")
    if "N" in opt:
        print(f" Kolmo Prob     = {shape:f} for shape alone, ={totals:f} for normalisation alone")
    if "X" in opt:
        print(f" Kolmo Prob     = {pseudo:f} with {count} pseudo-experiments")


# -- the chi-square test ----------------------------------------------------


def _chi2_ranges(one: Histogram, other: Histogram, opt: str) -> list[tuple[int, int]]:
    """The first and last bin of each axis compared: the axis, and its flow if asked."""
    _counts_only(one, other)
    if len(one.axes) != len(other.axes):
        raise ValueError(
            f"{one.name!r} has {len(one.axes)} axes and {other.name!r} {len(other.axes)}, and "
            f"the chi-square test compares them bin by bin"
        )
    for letter, mine, theirs in zip("xyz", one.axes, other.axes):
        if mine.nbins != theirs.nbins:
            raise ValueError(
                f"{one.name!r} has {mine.nbins} {letter} bins and {other.name!r} "
                f"{theirs.nbins}, and the chi-square test compares them bin by bin"
            )
    first = 0 if "UF" in opt else 1
    extra = 1 if "OF" in opt else 0
    return [(first, axis.nbins + extra) for axis in one.axes]


def _in_loop_order(histogram: Histogram, ranges: list[tuple[int, int]]) -> tuple[Any, Any]:
    """Contents and ``GetBinErrorSqUnchecked`` of the bins compared, x outermost as ROOT loops."""
    cut = tuple(slice(first, last + 1) for first, last in ranges)
    contents = histogram.values(flow=True).astype(np.float64)[cut].ravel()
    squares = histogram._shaped(histogram._variance_cells(), True)[cut].ravel()
    return contents, squares


def _unweighted(histogram: Histogram) -> bool:
    """Whether ``GetStats`` says every weight was one: total and effective entries agree."""
    found = statistics(histogram)
    effective = found[0] * found[0] / found[1] if found[1] else 0.0
    return abs(found[0] - effective) < 1


def _mode(opt: str, one: Histogram, other: Histogram) -> str:
    """``"UU"``, ``"UW"`` or ``"WW"``: as asked, or - asked none - as the histograms are."""
    asked = [mode for mode in ("UU", "UW", "WW") if mode in opt]
    first, second = _unweighted(one), _unweighted(other)
    if not asked:
        return ("UU" if second else "UW") if first else "WW"
    _warned(asked[0], "NORM" in opt, first, second)
    return asked[0]


def _warned(mode: str, norm: bool, first: bool, second: bool) -> None:
    """ROOT's two warnings: a test for counts asked of histograms that are not counts."""
    if mode == "UW" and not first:
        _warn("the first histogram is weighted and option UW has been requested")
    if mode == "UU" and not norm and not (first and second):
        _warn("both histograms are not unweighted and option UU has been requested")


def _warn(what: str) -> None:
    warnings.warn(f"Chi2TestX: {what}, which is ROOT's warning too", RuntimeWarning, stacklevel=6)


class _Bins(NamedTuple):
    """Both histograms' bins, in ROOT's order, and ROOT's totals of them."""

    c1: np.ndarray[Any, Any]
    c2: np.ndarray[Any, Any]
    e1: np.ndarray[Any, Any]
    e2: np.ndarray[Any, Any]
    sum1: float
    sum2: float
    sumw1: float
    sumw2: float


def _scaled(contents: Any, squares: Any) -> Any:
    """Option ``"NORM"``: a scaled bin taken back to its effective entries, rounded."""
    with np.errstate(all="ignore"):
        return np.where(squares > 0, np.floor(contents * contents / squares + 0.5), 0.0)


def _normed(c1: Any, e1: Any, c2: Any, e2: Any) -> tuple[Any, Any]:
    """Option ``"NORM"``: both histograms' bins as effective entries, refusing no errors."""
    if running(0.0, e1) <= 0 or running(0.0, e2) <= 0:
        raise ValueError(
            "option NORM takes each bin back to its effective entries by its error, "
            "and one histogram has errors of zero throughout"
        )
    return _scaled(c1, e1), _scaled(c2, e2)


def _gathered(one: Histogram, other: Histogram, ranges: Any, mode: str, norm: bool) -> _Bins:
    c1, e1 = _in_loop_order(one, ranges)
    c2, e2 = _in_loop_order(other, ranges)
    if norm:
        c1, c2 = _normed(c1, e1, c2, e2)
    weighed = norm or mode == "WW"
    sumw1 = running(0.0, e1) if weighed else 0.0
    sumw2 = running(0.0, e2) if weighed or mode == "UW" else 0.0
    found = _Bins(c1, c2, e1, e2, running(0.0, c1), running(0.0, c2), sumw1, sumw2)
    _refuse_nothing(one, other, found, mode)
    return found


def _refuse_nothing(one: Histogram, other: Histogram, found: _Bins, mode: str) -> None:
    """What ``Chi2TestX`` cannot test: an empty histogram, or no errors on either side."""
    if found.sum1 == 0 or found.sum2 == 0:
        raise ValueError(
            f"{(one if found.sum1 == 0 else other).name!r} has nothing in the bins compared, "
            f"and a chi-square test of an empty histogram is not a test"
        )
    if mode == "WW" and found.sumw1 <= 0 and found.sumw2 <= 0:
        raise ValueError("both histograms have errors of zero throughout, and so no weights")


def _uu(bins: _Bins, ndf: int, where: Any) -> tuple[float, int, int, np.ndarray[Any, Any]]:
    """Two counts: Pearson's chi-square with both totals estimated, and Haberman's residuals."""
    empty = (np.trunc(bins.c1) == 0) & (np.trunc(bins.c2) == 0)
    c1, c2 = bins.c1[~empty], bins.c2[~empty]
    sum1, sum2 = bins.sum1, bins.sum2
    total, both = sum1 + sum2, c1 + c2
    expected = both * sum1 / total
    with np.errstate(all="ignore"):
        residuals = (c1 - expected) / np.sqrt(expected)
        residuals /= np.sqrt((1.0 - sum1 / total) * (1.0 - both / total))
    delta = sum2 * c1 - sum1 * c2
    chi2 = running(0.0, delta * delta / both) / (sum1 * sum2)
    igood = (1 if np.any(c1 < 1) else 0) + (2 if np.any(c2 < 1) else 0)
    return chi2, ndf - int(empty.sum()), igood, residuals


def _uw(bins: _Bins, ndf: int, where: Any) -> tuple[float, int, int, np.ndarray[Any, Any]]:
    """A count against a weighted histogram, bin by bin as ROOT loops - its total nudged too."""
    state = [bins.sum1, 0.0, 0.0, 0.0]
    chi2, thin1, thin2, residuals = 0.0, 0, 0, []
    rows = zip(bins.c1.tolist(), bins.c2.tolist(), bins.e2.tolist())
    for at, (cnt1, cnt2, e2sq) in enumerate(rows):
        if cnt1 * cnt1 == 0 and cnt2 * cnt2 == 0:
            ndf -= 1
            continue
        e2sq = _stand_in_error(cnt2, e2sq, bins, where, at)
        thin1 += cnt1 < 1
        thin2 += e2sq > 0 and cnt2 * cnt2 / e2sq < 10
        state[1] = cnt1
        _settle(state, cnt2, e2sq, bins.sum2)
        term, residual = _uw_terms(state, cnt2, e2sq, bins.sum2)
        chi2 += term[0]
        chi2 += term[1]
        residuals.append(residual)
    return chi2, ndf, (1 if thin1 else 0) + (2 if thin2 else 0), np.array(residuals)


def _stand_in_error(cnt2: float, e2sq: float, bins: _Bins, where: Any, at: int) -> float:
    """A weighted bin with nothing in it takes the average error, if there is one to take."""
    if cnt2 * cnt2 != 0 or e2sq != 0:
        return e2sq
    if bins.sumw2 > 0:
        return bins.sumw2 / bins.sum2
    raise ValueError(
        f"the weighted histogram has nothing and no error in bin {where(at)}, where the other "
        f"has entries, and no error anywhere to stand in: the discrepancy is infinite"
    )


def _variances(state: list[float], cnt2: float, e2sq: float, sum2: float) -> None:
    sum1, cnt1 = state[0], state[1]
    state[2] = sum2 * cnt2 - sum1 * e2sq
    state[3] = state[2] * state[2] + 4.0 * sum2 * sum2 * cnt1 * e2sq


def _nudged(state: list[float], cnt2: float, e2sq: float, sum2: float) -> None:
    """ROOT's first ``while``: one more entry each time the formula would be zero over zero."""
    _variances(state, cnt2, e2sq, sum2)
    while state[2] * state[2] + state[1] == 0 or state[2] + state[3] == 0:
        state[0] += 1
        state[1] += 1
        _variances(state, cnt2, e2sq, sum2)


def _settle(state: list[float], cnt2: float, e2sq: float, sum2: float) -> None:
    """ROOT's nudging of the count and total, both of its loops, until the estimate is not zero.

    ``state`` is the total of the counts, this bin's count, and ROOT's
    ``var1`` and ``var2``, which is left as its root. The total stays nudged
    for the bins after this one, as ROOT's does.
    """
    _nudged(state, cnt2, e2sq, sum2)
    state[3] = math.sqrt(state[3])
    while state[2] + state[3] == 0:
        state[0] += 1
        state[1] += 1
        _nudged(state, cnt2, e2sq, sum2)
        state[3] = math.sqrt(state[3])


def _uw_terms(
    state: list[float], cnt2: float, e2sq: float, sum2: float
) -> tuple[tuple[float, float], float]:
    """One bin's two terms of the chi-square, and its residual, from the estimated probability."""
    sum1, cnt1, var1, var2 = state
    probb = (var1 + var2) / (2.0 * sum2 * sum2)
    nexp1, nexp2 = probb * sum1, probb * sum2
    delta1, delta2 = cnt1 - nexp1, cnt2 - nexp2
    if e2sq <= 0:
        return (delta1 * delta1 / nexp1, 0.0), delta1 / math.sqrt(nexp1)
    residual = _uw_residual(state, cnt2, e2sq, sum2, probb)
    return (delta1 * delta1 / nexp1, delta2 * delta2 / e2sq), residual


def _uw_residual(state: list[float], cnt2: float, e2sq: float, sum2: float, probb: float) -> float:
    """The weighted bin's residual, over the variance of its difference from the estimate."""
    sum1, var2 = state[0], state[3]
    temp1 = sum2 * e2sq / var2
    temp2 = 1.0 + (sum1 * e2sq - sum2 * cnt2) / var2
    temp2 = temp1 * temp1 * sum1 * probb * (1.0 - probb) + temp2 * temp2 * e2sq / 4.0
    return -(cnt2 - probb * sum2) / math.sqrt(temp2)


def _ww(bins: _Bins, ndf: int, where: Any) -> tuple[float, int, int, np.ndarray[Any, Any]]:
    """Two weighted histograms: each difference over what both errors allow, and its residual."""
    empty = (bins.c1 * bins.c1 == 0) & (bins.c2 * bins.c2 == 0)
    blind = np.flatnonzero(~empty & (bins.e1 == 0) & (bins.e2 == 0))
    if len(blind):
        raise ValueError(
            f"both histograms have an error of zero in bin {where(int(blind[0]))}, and a "
            f"difference nothing allows for cannot be weighed"
        )
    c1, c2, e1, e2 = bins.c1[~empty], bins.c2[~empty], bins.e1[~empty], bins.e2[~empty]
    sum1, sum2 = bins.sum1, bins.sum2
    sigma = sum1 * sum1 * e2 + sum2 * sum2 * e1
    delta = sum2 * c1 - sum1 * c2
    chi2 = running(0.0, delta * delta / sigma)
    igood = (1 if _thin(c1, e1) else 0) + (2 if _thin(c2, e2) else 0)
    return chi2, ndf - int(empty.sum()), igood, _ww_residuals((c1, c2, e1, e2), sum1, sum2, sigma)


def _thin(contents: Any, squares: Any) -> bool:
    """Whether any bin compared holds fewer than ten effective entries."""
    with np.errstate(all="ignore"):
        return bool(np.any((squares > 0) & (contents * contents / squares < 10)))


def _ww_residuals(bins: tuple[Any, ...], sum1: float, sum2: float, sigma: Any) -> Any:
    """Each bin's studentised residual, taken on the side whose error is the larger."""
    c1, c2, e1, e2 = bins
    probb = (c1 * sum1 * e2 + c2 * sum2 * e1) / sigma
    with np.errstate(all="ignore"):
        first = (c1 - sum1 * probb) / np.sqrt(e1 * (1.0 - e2 * sum1 * sum1 / sigma))
        second = -(c2 - sum2 * probb) / np.sqrt(e2 * (1.0 - e1 * sum2 * sum2 / sigma))
    return np.where(e1 > e2, first, second)


#: The three tests, by the option that asks for each.
TESTS = {"UU": _uu, "UW": _uw, "WW": _ww}


def chi2_test_full(one: Histogram, other: Histogram, option: str) -> Chi2Result:
    """``TH1::Chi2TestX``: the chi-square, degrees of freedom, flag, p-value and residuals.

    ``option`` holds ROOT's words: ``"UU"`` for two counts, ``"UW"`` for a
    count against a weighted histogram - the first the count - and ``"WW"``
    for two weighted ones; none of the three lets the histograms' own sums say
    which. ``"NORM"`` goes with ``"UU"`` for counts that were scaled, and
    ``"UF"`` and ``"OF"`` take the underflow and overflow in.
    """
    opt = option.upper()
    ranges = _chi2_ranges(one, other, opt)
    ndf = math.prod(last - first + 1 for first, last in ranges) - 1
    mode = _mode(opt, one, other)
    norm = "NORM" in opt and mode == "UU"
    bins = _gathered(one, other, ranges, mode, norm)
    shape = tuple(last - first + 1 for first, last in ranges)

    def where(at: int) -> str:
        found = np.unravel_index(at, shape)
        return "(" + ", ".join(str(int(i) + r[0]) for i, r in zip(found, ranges)) + ")"

    chi2, ndf, igood, residuals = TESTS[mode](bins, ndf, where)
    return Chi2Result(chi2, ndf, igood, prob(chi2, ndf), residuals)


def chi2_test(one: Histogram, other: Histogram, option: str) -> float:
    """``TH1::Chi2Test``: the p-value, or with ``"CHI2"`` the chi-square, ``"CHI2/NDF"`` over ndf.

    ``"P"`` prints ROOT's line of the chi-square, p-value, degrees of freedom
    and flag, in ROOT's formats.
    """
    opt = option.upper()
    found = chi2_test_full(one, other, option)
    if "P" in opt:
        print(
            f"Chi2 = {found.chi2:f}, Prob = {found.p:g}, NDF = {found.ndf}, igood = {found.igood}"
        )
    if "CHI2/NDF" in opt:
        return found.chi2 / found.ndf if found.ndf else 0.0
    if "CHI2" in opt:
        return found.chi2
    return found.p
