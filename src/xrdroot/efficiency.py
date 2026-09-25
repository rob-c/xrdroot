"""Efficiencies: how often something passed, with an interval that stays in [0, 1].

A ``TEfficiency`` in a file is two histograms - what passed, and everything
that was tried - and the few settings that say how the fraction's
uncertainty is to be worked out. The efficiency itself and its interval are
not stored anywhere; ROOT works them out when asked, and so does this, the
same way and with the same defaults read out of the object.

Dividing the two histograms and propagating their errors is the one way not
to do it: an event in the numerator is also in the denominator, and the
interval that pretends otherwise runs past one and has no width at zero.
Every method here is one of ROOT's, and none of them can leave [0, 1].

The Beta quantile the exact and Bayesian intervals need is worked out here,
by a continued fraction and a guarded Newton step, rather than by SciPy,
which is not a dependency: it agrees with SciPy's ``betaincinv`` to better
than one part in 10^12 over the ranges an efficiency is ever computed at.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

import numpy as np

from .arithmetic import compatible
from .booking import FILL, LINE, MARKER, split_title
from .errors import FormatError, UnsupportedFeatureError
from .filling import arrays, fill_histogram
from .hist import Histogram
from .moments import statistics as get_stats

__all__ = ["EFFICIENCIES", "METHODS", "Efficiency", "beta_quantile", "regularized_beta"]

#: The class this reads.
EFFICIENCIES = ("TEfficiency",)

#: ROOT's ``EStatOption``, by the number ``fStatisticOption`` holds.
METHODS = {
    0: "clopper-pearson",
    1: "normal",
    2: "wilson",
    3: "agresti-coull",
    4: "feldman-cousins",
    5: "jeffreys",
    6: "uniform",
    7: "bayesian",
    8: "mid-p",
}

#: The methods whose interval is a Beta posterior's, and so whose prior counts.
BAYESIAN = ("jeffreys", "uniform", "bayesian")

#: The bits of ``fBits`` a ``TEfficiency`` keeps its own settings in: quote
#: the posterior's mode rather than its mean, take the shortest interval
#: rather than the central one, give each bin a prior of its own, and weight
#: the fills.
POSTERIOR_MODE = 1 << 15
SHORTEST_INTERVAL = 1 << 16
BIN_PRIOR = 1 << 17
USE_WEIGHTS = 1 << 18

#: How close the continued fraction has to come before it is taken as done,
#: and how many terms it may take getting there.
_EPSILON = 1e-16
_TERMS = 500
#: The smallest number the continued fraction divides by, for a zero term.
_TINY = 1e-300


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _fraction_step(a: float, b: float, x: float, m: int) -> tuple[float, float]:
    """The two coefficients of the ``m``-th pair of terms of the fraction."""
    even = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
    odd = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
    return even, odd


def _guarded(value: float) -> float:
    return value if abs(value) > _TINY else _TINY


def _continued_fraction(a: float, b: float, x: float) -> float:
    """The continued fraction for the incomplete Beta function, by Lentz's method.

    It converges quickly for ``x`` below ``(a + 1) / (a + b + 2)``, which is
    where :func:`regularized_beta` uses it; above that it is used for the
    other tail, which is the same sum the other way round.
    """
    c, d = 1.0, 1.0 / _guarded(1.0 - (a + b) * x / (a + 1.0))
    result = d
    for m in range(1, _TERMS):
        for coefficient in _fraction_step(a, b, x, m):
            d = 1.0 / _guarded(1.0 + coefficient * d)
            c = _guarded(1.0 + coefficient / c)
            result *= d * c
        if abs(d * c - 1.0) < _EPSILON:
            return result
    raise ArithmeticError(f"the incomplete Beta function of ({a}, {b}) at {x} did not converge")


def regularized_beta(x: float, a: float, b: float) -> float:
    """``I_x(a, b)``: the probability a Beta(a, b) variable falls below ``x``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - _log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _continued_fraction(a, b, x) / a
    return 1.0 - front * _continued_fraction(b, a, 1.0 - x) / b


def _density(x: float, a: float, b: float) -> float:
    """The Beta(a, b) density at ``x``, which is the slope Newton steps along.

    Near an end a shape below one sends it to infinity; capping the exponent
    keeps the step finite, and a step that is too short is one the bracket
    round it catches up with anyway.
    """
    power = (a - 1.0) * math.log(x) + (b - 1.0) * math.log1p(-x) - _log_beta(a, b)
    return math.exp(min(power, 700.0))


def _between(low: float, high: float) -> float:
    """Halfway across the bracket - by ratio, not difference, when it spans decades.

    A quantile of a lopsided Beta can be 10^-200 and more; halving the
    difference would take a thousand steps to get there, and halving the
    ratio takes a few dozen.
    """
    if high > 4.0 * low:
        return math.sqrt(max(low, 1e-300) * high)
    return (low + high) / 2.0


def _newton(x: float, miss: float, a: float, b: float, low: float, high: float) -> float:
    """One Newton step towards the quantile, or a bisection when it overshoots."""
    slope = _density(x, a, b)
    guess = x - miss / slope if slope > 0.0 else low
    return guess if low < guess < high else _between(low, high)


def _lower_quantile(p: float, a: float, b: float) -> float:
    """:func:`beta_quantile` for an answer of at most a half, which is nearer zero."""
    low, high, x = 0.0, 1.0, a / (a + b)
    for _ in range(400):
        miss = regularized_beta(x, a, b) - p
        if miss == 0.0:
            return x
        low, high = (low, x) if miss > 0.0 else (x, high)
        step = _newton(x, miss, a, b, low, high)
        if abs(step - x) <= 1e-15 * x or step == 0.0:
            return step  # converged, or so close to zero that a double cannot say
        x = step
    return x  # pragma: no cover - the bracket shrinks every step, far past any double


def beta_quantile(p: float, a: float, b: float) -> float:
    """The ``x`` below which a Beta(a, b) variable falls with probability ``p``.

        >>> round(beta_quantile(0.5, 2.0, 2.0), 12)
        0.5

    This is ``ROOT::Math::beta_quantile`` and SciPy's ``betaincinv``: Newton
    on :func:`regularized_beta`, kept inside a bracket that is halved whenever
    a step would leave it, so it converges however lopsided the distribution.
    An answer above a half is worked out as one minus the other tail's, so
    one close to one keeps the digits that say how close.
    """
    if not (a > 0.0 and b > 0.0):
        raise ValueError(f"a Beta distribution needs both shapes above zero, not ({a}, {b})")
    if p <= 0.0 or p >= 1.0:
        return 0.0 if p <= 0.0 else 1.0
    if p > regularized_beta(0.5, a, b):
        return 1.0 - _lower_quantile(1.0 - p, b, a)
    return _lower_quantile(p, a, b)


def _normal_quantile(p: float) -> float:
    return statistics.NormalDist().inv_cdf(p)


def _clopper_pearson(passed: float, total: float, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    low = 0.0 if passed == 0 else beta_quantile(tail, passed, total - passed + 1.0)
    high = 1.0 if passed == total else beta_quantile(1.0 - tail, passed + 1.0, total - passed)
    return low, high


def _clipped(centre: float, half: float) -> tuple[float, float]:
    return max(centre - half, 0.0), min(centre + half, 1.0)


def _normal(passed: float, total: float, level: float) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    average = passed / total
    kappa = _normal_quantile((1.0 + level) / 2.0)
    return _clipped(average, kappa * math.sqrt(average * (1.0 - average) / total))


def _wilson(passed: float, total: float, level: float) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    average = passed / total
    kappa = _normal_quantile((1.0 + level) / 2.0)
    squared = kappa * kappa
    centre = (passed + squared / 2.0) / (total + squared)
    root = math.sqrt(total * average * (1.0 - average) + squared / 4.0)
    return _clipped(centre, kappa / (total + squared) * root)


def _agresti_coull(passed: float, total: float, level: float) -> tuple[float, float]:
    kappa = _normal_quantile((1.0 + level) / 2.0)
    squared = kappa * kappa
    centre = (passed + squared / 2.0) / (total + squared)
    return _clipped(centre, kappa * math.sqrt(centre * (1.0 - centre) / (total + squared)))


#: The frequentist intervals, each of one bin's passed and total at a level.
FREQUENTIST = {
    "clopper-pearson": _clopper_pearson,
    "normal": _normal,
    "wilson": _wilson,
    "agresti-coull": _agresti_coull,
}


def _central(a: float, b: float, level: float) -> tuple[float, float]:
    """The central interval of a Beta(a, b) posterior, as ROOT's ``BetaCentralInterval``."""
    if a <= 0.0 or b <= 0.0:
        return 0.0, 1.0
    return beta_quantile((1.0 - level) / 2.0, a, b), beta_quantile((1.0 + level) / 2.0, a, b)


#: What a freshly made ``TEfficiency`` holds beside its histograms, as ROOT
#: makes one: a flat Beta prior, ``kDefConfLevel`` - one Gaussian sigma, to
#: the digits ROOT writes it with - Clopper-Pearson, a weight of one, and a
#: fill colour of its own.
DEFAULTS = {
    "fBeta_alpha": 1.0,
    "fBeta_beta": 1.0,
    "fConfLevel": 0.682689492137,
    "fStatisticOption": 0,
    "fWeight": 1.0,
}
EFFICIENCY_FILL = {**FILL, "fFillColor": 19}

#: The bits every freshly made object carries: on the heap, and not deleted.
FRESH = 0x03000000


def _suffixed(title: str, suffix: str) -> str:
    """A title with ``suffix`` after its first part, before any axis titles."""
    head, semicolon, rest = str(title).partition(";")
    return head + suffix + semicolon + rest


def _members(
    name: str, title: str, passed: Histogram, total: Histogram, bits: int
) -> dict[str, Any]:
    """Every member of a ``TEfficiency`` holding ``passed`` and ``total``."""
    return {
        "TNamed": {"fName": str(name), "fTitle": str(title), "fBits": FRESH | bits},
        "TAttLine": dict(LINE),
        "TAttFill": dict(EFFICIENCY_FILL),
        "TAttMarker": dict(MARKER),
        **DEFAULTS,
        "fBeta_bin_params": [],
        "fFunctions": [],
        "fPassedHistogram": passed,
        "fTotalHistogram": total,
    }


def _same(one: float, other: float, tolerance: float) -> bool:
    """``TMath::AreEqualRel``: equal to within a fraction of their size."""
    return abs(one - other) <= 0.5 * tolerance * (abs(one) + abs(other))


def _filled_with_weights(passed: Histogram, total: Histogram) -> bool:
    """``TEfficiency::CheckWeights``: whether the sums of weights are not simply counts."""
    if passed._sumw2() is None and total._sumw2() is None:
        return False
    tolerance = 1e-5 if total.classname == "TH1F" else 1e-12
    return not all(
        _same(found[0], found[1], tolerance) for found in (get_stats(passed), get_stats(total))
    )


class Efficiency:
    """A ``TEfficiency``: what passed, out of what was tried, bin by bin.

        >>> eff = f["trigger"]                               # doctest: +SKIP
        >>> eff.values(), eff.intervals()
        >>> eff.intervals(level=0.95, method="wilson")

    :attr:`passed` and :attr:`total` are the two histograms it was filled
    into; :meth:`values` is the efficiency in each bin and :meth:`intervals`
    the low and high end of the confidence interval on it, by the method and
    at the level the object was saved with unless asked for another. Bins
    are indexed as a :class:`~.hist.Histogram`'s are, and ``flow=True``
    brings back the two at each end.
    """

    __slots__ = ("classname", "members", "passed", "total")

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        passed, total = members.get("fPassedHistogram"), members.get("fTotalHistogram")
        if not isinstance(passed, Histogram) or not isinstance(total, Histogram):
            raise FormatError(f"a {classname} was written without its two histograms")
        #: The class the file says this is.
        self.classname = classname
        #: Every member, as it was written.
        self.members = members
        #: What passed, and what was tried: the two histograms it was filled into.
        self.passed: Histogram = passed
        self.total: Histogram = total

    def __repr__(self) -> str:
        shape = " x ".join(str(count) for count in self.total.shape)
        return f"<{self.classname} {self.name!r} of {shape} bins, by {self.method}>"

    @property
    def name(self) -> str:
        return str(self.members["TNamed"]["fName"])

    @property
    def title(self) -> str:
        return str(self.members["TNamed"]["fTitle"])

    @property
    def axes(self) -> tuple[Any, ...]:
        """The axes of the histograms, which are the same two sets of axes."""
        return self.total.axes

    @property
    def method(self) -> str:
        """How the interval is worked out unless asked otherwise: ``fStatisticOption``."""
        return METHODS.get(int(self.members.get("fStatisticOption", 0)), "clopper-pearson")

    @property
    def level(self) -> float:
        """The confidence level the interval is quoted at: ``fConfLevel``."""
        return float(self.members.get("fConfLevel", 0.682689492137086))

    @property
    def _bits(self) -> int:
        return int(self.members["TNamed"].get("fBits", 0))

    def _counts(self, flow: bool) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        if self._bits & USE_WEIGHTS:
            raise UnsupportedFeatureError(
                f"{self.name!r} was filled with weights, whose intervals ROOT works out by "
                f"a normal approximation over the weighted sums that this reader does not "
                f"reproduce; passed and total are here to work from"
            )
        return (
            np.asarray(self.passed.values(flow), dtype=np.float64),
            np.asarray(self.total.values(flow), dtype=np.float64),
        )

    def _chosen(self, method: str | None) -> str:
        chosen = self.method if method is None else method.lower()
        if chosen in ("feldman-cousins", "mid-p"):
            raise UnsupportedFeatureError(
                f"{chosen} intervals are not worked out by this reader; clopper-pearson, "
                f"normal, wilson, agresti-coull, jeffreys, uniform and bayesian are"
            )
        if chosen not in METHODS.values():
            raise ValueError(f"method={method!r} is not one of {', '.join(METHODS.values())}")
        return chosen

    def _priors(self, method: str, shape: tuple[int, ...], flow: bool) -> tuple[Any, Any]:
        """The Beta prior of every bin, which a Bayesian interval is built on."""
        if method == "jeffreys":
            return np.full(shape, 0.5), np.full(shape, 0.5)
        if method == "uniform":
            return np.ones(shape), np.ones(shape)
        alpha = float(self.members.get("fBeta_alpha", 1.0))
        beta = float(self.members.get("fBeta_beta", 1.0))
        cells = [len(axis) + 2 for axis in self.axes]
        flat_alpha = np.full(int(np.prod(cells)), alpha)
        flat_beta = np.full(int(np.prod(cells)), beta)
        if self._bits & BIN_PRIOR:
            for index, pair in enumerate(self.members.get("fBeta_bin_params", ())):
                flat_alpha[index], flat_beta[index] = pair
        return self.total._shaped(flat_alpha, flow), self.total._shaped(flat_beta, flow)

    def values(self, flow: bool = False, method: str | None = None) -> np.ndarray[Any, Any]:
        """The efficiency in each bin, as ``TEfficiency::GetEfficiency`` gives it.

        ``passed / total``, and zero where nothing was tried; for a Bayesian
        method the mean of the posterior instead - or its mode, if the object
        was told to quote that - which is not zero even for an empty bin.
        """
        chosen = self._chosen(method)
        passed, total = self._counts(flow)
        if chosen not in BAYESIAN:
            return np.divide(passed, total, out=np.zeros_like(passed), where=total != 0)
        alpha, beta = self._priors(chosen, passed.shape, flow)
        a, b = passed + alpha, total - passed + beta
        if self._bits & POSTERIOR_MODE:
            peaked = (a > 1) & (b > 1)
            mode = np.divide(a - 1, a + b - 2, out=np.zeros_like(a), where=peaked)
            return np.where(peaked, mode, a / (a + b))
        return a / (a + b)

    def intervals(
        self, level: float | None = None, method: str | None = None, flow: bool = False
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """The low and high end of the confidence interval in every bin.

            >>> low, high = eff.intervals(level=0.95, method="jeffreys")  # doctest: +SKIP

        ``level`` is ``fConfLevel`` and ``method`` ``fStatisticOption`` unless
        given: ``clopper-pearson`` (ROOT's default, and exact), ``normal``,
        ``wilson``, ``agresti-coull``, and the Bayesian ``jeffreys``,
        ``uniform`` and ``bayesian`` - the last with the prior the object was
        saved with. Each is ROOT's own formula, so the numbers are the ones
        ``GetEfficiencyErrorLow`` and ``GetEfficiencyErrorUp`` are made from.
        """
        chosen = self._chosen(method)
        level = self.level if level is None else float(level)
        if not 0.0 < level < 1.0:
            raise ValueError(f"a confidence level is between 0 and 1, not {level}")
        passed, total = self._counts(flow)
        if chosen in FREQUENTIST:
            one = FREQUENTIST[chosen]
            pairs = [one(p, t, level) for p, t in zip(passed.ravel(), total.ravel())]
        else:
            pairs = self._bayesian(chosen, passed, total, level, flow)
        low = np.array([pair[0] for pair in pairs], dtype=np.float64).reshape(passed.shape)
        high = np.array([pair[1] for pair in pairs], dtype=np.float64).reshape(passed.shape)
        return low, high

    def _bayesian(
        self,
        method: str,
        passed: np.ndarray[Any, Any],
        total: np.ndarray[Any, Any],
        level: float,
        flow: bool,
    ) -> list[tuple[float, float]]:
        if self._bits & SHORTEST_INTERVAL:
            raise UnsupportedFeatureError(
                f"{self.name!r} asks for the shortest Bayesian interval rather than the "
                f"central one, which this reader does not work out"
            )
        alpha, beta = self._priors(method, passed.shape, flow)
        a, b = (passed + alpha).ravel(), (total - passed + beta).ravel()
        return [_central(float(x), float(y), level) for x, y in zip(a, b)]

    def errors(
        self, level: float | None = None, method: str | None = None, flow: bool = False
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """How far each end of the interval is from the efficiency: the error bars."""
        values = self.values(flow, method)
        low, high = self.intervals(level, method, flow)
        return values - low, high - values

    # -- booking and filling ----------------------------------------------------

    @classmethod
    def book(cls, name: str, *axes: Any, title: str = "") -> Efficiency:
        """An empty efficiency of one, two or three axes, as ROOT's constructors make one.

            >>> eff = Efficiency.book("trigger", (20, 0, 100))    # TEfficiency(..., 20, 0, 100)

        The axes are given as :meth:`Histogram.book` takes them. The two
        histograms are named and titled as ROOT names them - ``trigger_total``
        and ``trigger_passed``, the title with `` (total)`` and `` (passed)``
        after it - and the settings are ROOT's defaults: Clopper-Pearson at
        one sigma, and a flat prior.
        """
        main, _titles = split_title(title)
        total = Histogram.book(f"{name}_total", *axes, title=_suffixed(title, " (total)"))
        passed = Histogram.book(f"{name}_passed", *axes, title=_suffixed(title, " (passed)"))
        return cls("TEfficiency", _members(name, main, passed, total, 0))

    @classmethod
    def from_histograms(
        cls, passed: Histogram, total: Histogram, name: str | None = None
    ) -> Efficiency:
        """``TEfficiency(passed, total)``: an efficiency from two histograms already filled.

        They must be binned alike, and what passed can be no more than what
        was tried in any bin, flow bins included - ROOT's ``CheckConsistency``.
        Both are copied, and renamed after the efficiency, which is called
        ``name`` or, as ROOT calls it, ``total``'s name with ``_clone`` after.
        Histograms filled with weights make an efficiency that says so.
        """
        for histogram in (passed, total):
            if not isinstance(histogram, Histogram) or histogram.kind != "COUNT":
                raise TypeError(
                    "an efficiency is made from two histograms of counts, what passed and what "
                    f"was tried; a {type(histogram).__name__} of kind "
                    f"{getattr(histogram, 'kind', None)!r} is not one"
                )
        compatible(passed, total)
        over = int(np.count_nonzero(passed.values(flow=True) > total.values(flow=True)))
        if over:
            raise ValueError(
                f"{passed.name!r} holds more than {total.name!r} in {over} bins, and what "
                f"passed must be a part of what was tried, bin by bin, flow bins included"
            )
        called = name or f"{total.name}_clone"
        bits = USE_WEIGHTS if _filled_with_weights(passed, total) else 0
        members = _members(
            called,
            "efficiency",
            passed.copy(f"{called}_passed"),
            total.copy(f"{called}_total"),
            bits,
        )
        return cls("TEfficiency", members)

    def fill(
        self, passed: Any, x: Any, y: Any = None, z: Any = None, *, weight: Any = None
    ) -> None:
        """``Fill`` - or ``FillWeighted``, given a weight - for every entry.

            >>> eff.fill(fired, pt)                                   # doctest: +SKIP

        ``passed`` says, entry by entry, whether it passed; every entry goes
        into the total and the ones that passed into the passed histogram,
        with ROOT's bookkeeping in both. A weight, even of one, is ROOT's
        ``FillWeighted``: the efficiency is marked as weighted, and both
        histograms keep the squares of their weights from then on.
        """
        given = [value for value in (x, y, z) if value is not None]
        if len(given) != len(self.axes):
            raise ValueError(
                f"{self.name!r} has {len(self.axes)} axes, and filling it takes whether each "
                f"entry passed and then one coordinate per axis, not {len(given)}"
            )
        flat, weights = arrays([passed, *given], weight)
        if weight is not None:
            self._use_weights()
        chosen = flat[0] != 0
        coordinates = flat[1:]
        fill_histogram(self.total, coordinates, weights)
        fill_histogram(self.passed, [value[chosen] for value in coordinates], weights[chosen])

    def _use_weights(self) -> None:
        """``SetUseWeightedEvents``: say so, and keep the squares of the weights in both."""
        named = self.members["TNamed"]
        named["fBits"] = int(named.get("fBits", FRESH)) | USE_WEIGHTS
        self.total._ensure_sumw2()
        self.passed._ensure_sumw2()

    def to_numpy(self, flow: bool = False) -> tuple[np.ndarray[Any, Any], ...]:
        """The efficiency and each axis's edges, the way :func:`numpy.histogram` gives them."""
        return (self.values(flow), *self.total.to_numpy(flow)[1:])
