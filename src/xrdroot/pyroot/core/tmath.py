"""``TMath``: ROOT's mathematics namespace, function by function, by ROOT's names.

``ROOT.TMath.Gaus(x, 0, 1)`` and ``TMath::Prob(chi2, ndf)`` are this module's
functions, so ``TMath`` is the module itself in the namespace. The constants
are functions, as in ROOT (``TMath::Pi()``); the arrays ROOT takes as a count
and a pointer are taken as a count and anything indexable, only the first
``n`` of it used. Where ROOT's answer differs from the textbook's - the
``n - 1`` of ``RMS``, ``Poisson`` of a non-integer, ``Prob`` of no degrees of
freedom - it is ROOT's; the functions ROOT's fits are written in are those
:mod:`xrdroot.function.special` and :mod:`xrdroot.stats` compute as ROOT does.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np

from ...efficiency import regularized_beta
from ...function import densities, special
from ...function import distributions as dist
from ...stats import chisquare_quantile, incomplete_gamma, kolmogorov_prob, prob
from .messages import message
from .refs import store_many

# -- constants, as functions ------------------------------------------------------------------

#: Each constant ROOT's ``TMath`` has, as the function that answers it.
CONSTANTS = {
    "Pi": math.pi,
    "TwoPi": 2 * math.pi,
    "PiOver2": math.pi / 2,
    "PiOver4": math.pi / 4,
    "InvPi": 1 / math.pi,
    "RadToDeg": 180.0 / math.pi,
    "DegToRad": math.pi / 180.0,
    "Sqrt2": math.sqrt(2.0),
    "E": math.e,
    "Ln10": math.log(10.0),
    "LogE": math.log10(math.e),
    "C": 2.99792458e8,
    "Ccgs": 2.99792458e10,
    "CUncertainty": 0.0,
    "G": 6.67430e-11,
    "Gcgs": 6.67430e-8,
    "GUncertainty": 0.00015e-11,
    "GhbarC": 6.70883e-39,
    "GhbarCUncertainty": 0.00015e-39,
    "Gn": 9.80665,
    "GnUncertainty": 0.0,
    "H": 6.62607015e-34,
    "Hcgs": 6.62607015e-27,
    "HUncertainty": 0.0,
    "Hbar": 1.054571817e-34,
    "Hbarcgs": 1.054571817e-27,
    "HbarUncertainty": 0.0,
    "HC": 6.62607015e-34 * 2.99792458e8,
    "HCcgs": 6.62607015e-27 * 2.99792458e10,
    "K": 1.380649e-23,
    "Kcgs": 1.380649e-16,
    "KUncertainty": 0.0,
    "Sigma": 5.670374419e-8,
    "SigmaUncertainty": 0.0,
    "Na": 6.02214076e23,
    "NaUncertainty": 0.0,
    "R": 1.380649e-23 * 6.02214076e23,
    "RUncertainty": 0.0,
    "MWair": 28.9644,
    "Rgair": 1.380649e-23 * 6.02214076e23 / 28.9644 * 1000,
    "EulerGamma": 0.577215664901532860606512090082402431042,
    "Qe": 1.602176634e-19,
    "QeUncertainty": 0.0,
}


def _constant(value: float) -> Any:
    def constant() -> float:
        return value

    return constant


globals().update({name: _constant(value) for name, value in CONSTANTS.items()})


# -- one number at a time ---------------------------------------------------------------------


def _elements(n: Any, a: Any = None) -> np.ndarray[Any, Any]:
    """``(n, a)`` as ROOT takes an array, or one array alone: the first ``n`` of it."""
    if a is None:
        return np.asarray(n, dtype=np.float64).reshape(-1)
    return np.asarray(a, dtype=np.float64).reshape(-1)[: int(n)]


def Abs(x: Any) -> Any:
    return abs(x)


def Sign(a: float, b: float) -> float:
    """``Sign(a, b)``: ``a``'s size with ``b``'s sign."""
    return math.copysign(abs(a), b) if not isinstance(a, int) else (abs(a) if b >= 0 else -abs(a))


def Min(*args: Any) -> Any:
    """``Min(a, b)``, or ``Min(n, array)`` - ROOT's ``MinElement`` spelt the old way."""
    if len(args) == 2 and np.ndim(args[1]) > 0:
        return MinElement(*args)
    return min(args)


def Max(*args: Any) -> Any:
    if len(args) == 2 and np.ndim(args[1]) > 0:
        return MaxElement(*args)
    return max(args)


def MinElement(n: Any, a: Any = None) -> float:
    return float(np.min(_elements(n, a)))


def MaxElement(n: Any, a: Any = None) -> float:
    return float(np.max(_elements(n, a)))


def LocMin(n: Any, a: Any = None) -> int:
    """``LocMin``: the index of the first smallest element, ``-1`` for none."""
    values = _elements(n, a)
    return int(np.argmin(values)) if len(values) else -1


def LocMax(n: Any, a: Any = None) -> int:
    values = _elements(n, a)
    return int(np.argmax(values)) if len(values) else -1


def Sqrt(x: float) -> float:
    return math.sqrt(x) if x >= 0 else math.nan


def Sq(x: float) -> float:
    return x * x


def Power(x: float, y: float) -> float:
    try:
        return float(math.pow(x, y))
    except (ValueError, OverflowError):
        return math.nan if x < 0 else math.inf


def Exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def Log(x: float) -> float:
    return math.log(x) if x > 0 else (-math.inf if x == 0 else math.nan)


def Log10(x: float) -> float:
    return math.log10(x) if x > 0 else (-math.inf if x == 0 else math.nan)


def Log2(x: float) -> float:
    return math.log2(x) if x > 0 else (-math.inf if x == 0 else math.nan)


def _unary(fn: Any) -> Any:
    def call(x: float) -> float:
        try:
            return float(fn(x))
        except ValueError:
            return math.nan

    call.__doc__ = f"``{fn.__name__}``, as C's."
    return call


#: The C library functions ROOT's names are, one argument each.
UNARY = {
    "Sin": math.sin,
    "Cos": math.cos,
    "Tan": math.tan,
    "ASin": math.asin,
    "ACos": math.acos,
    "ATan": math.atan,
    "SinH": math.sinh,
    "CosH": math.cosh,
    "TanH": math.tanh,
    "ASinH": math.asinh,
    "ACosH": math.acosh,
    "ATanH": math.atanh,
    "Erf": math.erf,
    "Erfc": math.erfc,
}
globals().update({name: _unary(fn) for name, fn in UNARY.items()})


def ATan2(y: float, x: float) -> float:
    return math.atan2(y, x)


def Hypot(x: float, y: float) -> float:
    return math.hypot(x, y)


def Floor(x: float) -> float:
    return float(math.floor(x))


def Ceil(x: float) -> float:
    return float(math.ceil(x))


def FloorNint(x: float) -> int:
    return math.floor(x)


def CeilNint(x: float) -> int:
    return math.ceil(x)


def Nint(x: float) -> int:
    """``Nint``: the nearest whole number, a half going to the even one, as ROOT rounds."""
    below = math.floor(x)
    fraction = x - below
    if fraction > 0.5 or (fraction == 0.5 and int(below) % 2):
        return int(below) + 1
    return int(below)


def Even(x: int) -> bool:
    return int(x) % 2 == 0


def Odd(x: int) -> bool:
    return int(x) % 2 == 1


def IsNaN(x: float) -> bool:
    return math.isnan(x)


def Finite(x: float) -> bool:
    return math.isfinite(x)


def QuietNaN() -> float:
    return math.nan


def SignalingNaN() -> float:
    return math.nan


def Infinity() -> float:
    return math.inf


def AreEqualAbs(af: float, bf: float, epsilon: float) -> bool:
    return bool(abs(af - bf) < epsilon or abs(af - bf) < sys.float_info.min)


def AreEqualRel(af: float, bf: float, relPrec: float) -> bool:
    """``AreEqualRel``: within ``relPrec`` of their mean size, or exactly equal."""
    gap = abs(af - bf)
    return bool(af == bf or gap <= 0.5 * relPrec * (abs(af) + abs(bf)) or gap < sys.float_info.min)


def Factorial(n: int) -> float:
    return float(math.factorial(int(n))) if n >= 0 else 0.0


def Binomial(n: int, k: int) -> float:
    """``Binomial(n, k)``: ``n`` choose ``k``, zero where it is not defined."""
    if n < 0 or k < 0 or n < k:
        return 0.0
    return float(math.comb(int(n), int(k)))


def BinomialI(p: float, n: int, k: int) -> float:
    """``BinomialI``: the chance of ``k`` or more successes in ``n`` tries of chance ``p``."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return regularized_beta(p, float(k), float(n - k + 1))


# -- special functions -----------------------------------------------------------------------


def Gamma(z: float, x: Any = None) -> float:
    """``Gamma(z)``; ``Gamma(a, x)`` is the regularised lower incomplete gamma function."""
    if x is not None:
        return incomplete_gamma(z, float(x))
    if z <= 0 and float(z).is_integer():
        return 0.0
    return math.gamma(z)


def LnGamma(z: float) -> float:
    return 0.0 if z <= 0 else math.lgamma(z)


def Beta(p: float, q: float) -> float:
    return math.exp(math.lgamma(p) + math.lgamma(q) - math.lgamma(p + q))


def BetaIncomplete(x: float, a: float, b: float) -> float:
    return regularized_beta(x, a, b)


def ErfInverse(x: float) -> float:
    """``ErfInverse``: the ``y`` whose ``Erf`` is ``x``, for ``x`` in (-1, 1), else 0."""
    if not -1 < x < 1:
        return 0.0
    return dist.normal_quantile((x + 1) / 2) / math.sqrt(2.0)


def ErfcInverse(x: float) -> float:
    return ErfInverse(1.0 - x)


def Freq(x: float) -> float:
    """``Freq``: the normal distribution function, ``(1 + erf(x/sqrt 2)) / 2``."""
    return dist.normal_cdf(x)


def NormQuantile(p: float) -> float:
    """``NormQuantile``: 0 outside (0, 1), as ROOT's is."""
    return 0.0 if p <= 0 or p >= 1 else dist.normal_quantile(p)


def Gaus(x: float, mean: float = 0.0, sigma: float = 1.0, norm: bool = False) -> float:
    """``TMath::Gaus``, of numbers in plain Python - the C library's ``exp``, as
    :func:`xrdroot.function.special.gaus` uses, without NumPy's cost for one number - which a
    macro's function, called a point at a time, calls many times over."""
    if sigma == 0:
        return special.GAUS_ZERO_WIDTH
    arg = (float(x) - mean) / sigma
    if arg < -special.GAUS_CUT or arg > special.GAUS_CUT:
        return 0.0
    found = math.exp(-0.5 * arg * arg)
    return found / (special.SQRT_TWO_PI * sigma) if norm else found


def Landau(x: float, mpv: float = 0.0, sigma: float = 1.0, norm: bool = False) -> float:
    return float(special.landau(x, mpv, sigma, norm))


def BreitWigner(x: float, mean: float = 0.0, gamma: float = 1.0) -> float:
    return float(special.breit_wigner(x, mean, gamma))


def BreitWignerRelativistic(x: float, median: float = 0.0, gamma: float = 1.0) -> float:
    return float(densities.breit_wigner_relativistic(x, median, gamma))


def LaplaceDist(x: float, alpha: float = 0.0, beta: float = 1.0) -> float:
    return float(densities.laplace_dist(x, alpha, beta))


def LaplaceDistI(x: float, alpha: float = 0.0, beta: float = 1.0) -> float:
    return float(densities.laplace_dist_i(x, alpha, beta))


def CauchyDist(x: float, t: float = 0.0, s: float = 1.0) -> float:
    return dist.cauchy_pdf(x, s, t)


def Poisson(x: float, par: float) -> float:
    """``Poisson``: ``par**x exp(-par) / Gamma(x + 1)``, for a non-integer ``x`` too."""
    if x < 0:
        return 0.0
    if x == 0.0:
        return 1.0 / math.exp(par)
    return math.exp(x * math.log(par) - par - math.lgamma(x + 1.0))


def PoissonI(x: float, par: float) -> float:
    return Poisson(float(int(x)), par)


def Prob(chi2: float, ndf: int) -> float:
    return prob(chi2, ndf)


def KolmogorovProb(z: float) -> float:
    return kolmogorov_prob(z)


def ChisquareQuantile(p: float, ndf: float) -> float:
    return chisquare_quantile(p, ndf)


def Student(T: float, ndf: float) -> float:
    return dist.student_pdf(T, ndf)


def StudentI(T: float, ndf: float) -> float:
    return dist.student_cdf(T, ndf)


def StudentQuantile(p: float, ndf: float, lower_tail: bool = True) -> float:
    """``StudentQuantile``: Hill's algorithm 396, as ROOT computes it, to about nine digits."""
    if ndf < 1 or p >= 1 or p <= 0:
        message("Error", "TMath::StudentQuantile", "illegal parameter values")
        return 0.0
    upper = p > 0.5 if lower_tail else p < 0.5
    q = 2 * ((1 - p) if lower_tail == upper else p)
    found = _hill(q, ndf)
    return found if upper else -found


def _hill(q: float, ndf: float) -> float:
    """The two-sided quantile of Student's t for a tail of ``q``: ``TMath::StudentQuantile``'s."""
    if ndf - 1 < 1e-8:
        angle = math.pi / 2 * q
        return math.cos(angle) / math.sin(angle)
    if ndf - 2 < 1e-8:
        return math.sqrt(2.0 / (q * (2 - q)) - 2)
    a, b, c, d = _hill_terms(ndf)
    y = math.pow(q * d, 2.0 / ndf)
    y = _hill_normal(q, ndf, (a, b, c, d)) if y > 0.05 + a else _hill_near(ndf, y, d)
    return math.sqrt(ndf * y)


def _hill_terms(ndf: float) -> tuple[float, float, float, float]:
    """Hill's ``a``, ``b``, ``c`` and ``d`` for ``ndf`` degrees of freedom."""
    a = 1.0 / (ndf - 0.5)
    b = 48.0 / (a * a)
    c = ((20700 * a / b - 98) * a - 16) * a + 96.36
    d = ((94.5 / (b + c) - 3.0) / b + 1) * math.sqrt(a * math.pi / 2) * ndf
    return a, b, c, d


def _hill_near(ndf: float, y: float, d: float) -> float:
    inner = 1.0 / (((ndf + 6.0) / (ndf * y) - 0.089 * d - 0.822) * (ndf + 2.0) * 3)
    return ((inner + 0.5 / (ndf + 4.0)) * y - 1.0) * (ndf + 1.0) / (ndf + 2.0) + 1 / y


def _hill_normal(q: float, ndf: float, terms: tuple[float, float, float, float]) -> float:
    """Hill's asymptotic inverse expansion about the normal distribution."""
    a, b, c, d = terms
    x = dist.normal_quantile(q * 0.5)
    if ndf < 5:
        c += 0.3 * (ndf - 4.5) * (x + 0.6)
    c += (((0.05 * d * x - 5.0) * x - 7.0) * x - 2.0) * x + b
    y = _hill_series(x, b, c)
    y = a * y * y
    return math.exp(y) - 1 if y > 0.002 else y + 0.5 * y * y


def _hill_series(x: float, b: float, c: float) -> float:
    y = x * x
    return (((((0.4 * y + 6.3) * y + 36.0) * y + 94.5) / c - y - 3.0) / b + 1) * x


def FDist(F: float, N: float, M: float) -> float:
    return dist.f_pdf(F, N, M)


def FDistI(F: float, N: float, M: float) -> float:
    return dist.f_cdf(F, N, M)


def GammaDist(x: float, gamma: float, mu: float = 0.0, beta: float = 1.0) -> float:
    return dist.gamma_pdf(x, gamma, beta, mu)


def LogNormal(x: float, sigma: float, theta: float = 0.0, m: float = 1.0) -> float:
    return dist.lognormal_pdf(x, math.log(m), sigma, theta)


def BetaDist(x: float, p: float, q: float) -> float:
    return dist.beta_pdf(x, p, q)


def BetaDistI(x: float, p: float, q: float) -> float:
    return dist.beta_cdf(x, p, q)


# -- arrays -------------------------------------------------------------------------------------


def Mean(n: Any, a: Any = None, w: Any = None) -> float:
    """``Mean(n, a)`` or ``Mean(n, a, w)``: the mean, weighted if weights are given."""
    values = _elements(n, a)
    if w is None:
        return float(np.mean(values)) if len(values) else 0.0
    weights = _elements(len(values), w)
    return float(np.sum(values * weights) / np.sum(weights))


def GeomMean(n: Any, a: Any = None) -> float:
    values = _elements(n, a)
    return float(np.exp(np.mean(np.log(np.abs(values)))))


def RMS(n: Any, a: Any = None, w: Any = None) -> float:
    """``RMS``, which is ROOT's name for the standard deviation, with ``n - 1``."""
    values = _elements(n, a)
    if w is not None:
        weights = _elements(len(values), w)
        mean = np.sum(values * weights) / np.sum(weights)
        total, sumw = np.sum(weights * (values - mean) ** 2), np.sum(weights)
        return float(np.sqrt(total * sumw / (sumw * sumw - np.sum(weights * weights))))
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def StdDev(n: Any, a: Any = None, w: Any = None) -> float:
    return RMS(n, a, w)


def Median(n: Any, a: Any = None, w: Any = None, work: Any = None) -> float:
    """``Median``: the middle value - or with weights, where half the weight is either side."""
    values = _elements(n, a)
    if not len(values):
        return 0.0
    if w is None:
        return float(np.median(values))
    weights = _elements(len(values), w)
    order = np.argsort(values, kind="stable")
    half = np.sum(weights) / 2.0
    low = int(np.argmax(np.cumsum(weights[order]) >= half))
    high = len(values) - 1 - int(np.argmax(np.cumsum(weights[order][::-1]) >= half))
    return float(0.5 * (values[order][low] + values[order][high]))


def KOrdStat(n: Any, a: Any, k: int, work: Any = None) -> float:
    """``KOrdStat``: the ``k``th smallest element, counting from zero."""
    return float(np.sort(_elements(n, a))[int(k)])


def Sort(n: Any, a: Any, index: Any, down: bool = True) -> None:
    """``Sort(n, a, index, down)``: the indices that put ``a`` in order, into ``index``."""
    values = _elements(n, a)
    order = np.argsort(-values if down else values, kind="stable")
    store_many(index, order.tolist())


def BinarySearch(n: Any, array: Any, value: float) -> int:
    """``BinarySearch``: the index of the last element at most ``value``, ``-1`` if none."""
    values = _elements(n, array)
    return int(np.searchsorted(values, value, side="right")) - 1


def Normalize(v: Any) -> float:
    """``Normalize(v)``: make a three-vector a unit one, in place, and hand back its length."""
    length = math.sqrt(sum(float(x) * float(x) for x in list(v)[:3]))
    if length > 0:
        store_many(v, [float(x) / length for x in list(v)[:3]])
    return length


def Cross(v1: Any, v2: Any, out: Any) -> Any:
    """``Cross(v1, v2, out)``: the cross product, into ``out``."""
    store_many(out, np.cross(np.asarray(v1)[:3], np.asarray(v2)[:3]).tolist())
    return out


def Quantiles(
    n: int, nprob: int, x: Any, quantiles: Any, prob: Any, isSorted: bool = True,
    index: Any = None, type: int = 7,
) -> None:  # fmt: skip
    """``Quantiles``: Hyndman and Fan's quantiles - type 7 by default - into ``quantiles``."""
    values = np.sort(_elements(n, x))
    probabilities = _elements(nprob, prob)
    methods = {4: "interpolated_inverted_cdf", 5: "hazen", 6: "weibull", 7: "linear",
               8: "median_unbiased", 9: "normal_unbiased", 1: "inverted_cdf",
               2: "averaged_inverted_cdf", 3: "closest_observation"}  # fmt: skip
    found = np.quantile(values, probabilities, method=methods[int(type)])
    store_many(quantiles, found.tolist())


def KolmogorovTest(na: int, a: Any, nb: int, b: Any, option: str = "") -> float:
    """``KolmogorovTest`` of two sorted samples: the probability, or with ``M`` the distance."""
    if na <= 2 or nb <= 2:
        message("Error", "KolmogorovTest", "Sets must have more than 2 points")
        return -1.0
    one, other = _elements(na, a), _elements(nb, b)
    points = np.union1d(one, other)
    below_one = np.searchsorted(one, points, side="right") / na
    below_other = np.searchsorted(other, points, side="right") / nb
    distance = float(np.max(np.abs(below_one - below_other)))
    chance = kolmogorov_prob(distance * math.sqrt(na * nb / (na + nb)))
    upper = str(option).upper()
    if "D" in upper:
        print(f" Kolmogorov Probability = {chance:g}, Max Dist = {distance:g}")
    return distance if "M" in upper else chance


def Permute(n: int, a: Any) -> bool:
    """``Permute``: ``a`` made its next permutation in lexicographic order, false after the last."""
    values = [int(v) for v in list(a)[:n]]
    pivot = next((i for i in range(n - 2, -1, -1) if values[i] < values[i + 1]), -1)
    if pivot < 0:
        return False
    swap = next(j for j in range(n - 1, pivot, -1) if values[j] > values[pivot])
    values[pivot], values[swap] = values[swap], values[pivot]
    values[pivot + 1 :] = values[pivot + 1 :][::-1]
    store_many(a, values)
    return True


__all__ = sorted(
    name
    for name, value in list(globals().items())
    if callable(value) and name[:1].isupper() and getattr(value, "__module__", "") == __name__
)
