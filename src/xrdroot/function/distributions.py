"""The standard distributions ``TMath`` and ``ROOT::Math`` share: densities, tails and quantiles.

Each is the textbook closed form over the special functions this library
already computes as ROOT does - Cephes's incomplete gamma from
:mod:`xrdroot.stats`, the regularised incomplete beta and its inverse from
:mod:`xrdroot.efficiency` - and Python's own ``erf``, ``lgamma`` and
normal quantile. A quantile with no closed form is found by bisection on the
distribution function, to the last bit a double holds.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable

from ..efficiency import beta_quantile, regularized_beta
from ..stats import incomplete_gamma, incomplete_gamma_c

__all__ = [
    "normal_cdf",
    "normal_quantile",
    "chi2_cdf",
    "chi2_quantile",
    "gamma_pdf",
    "student_pdf",
    "student_cdf",
    "student_quantile",
    "f_pdf",
    "f_cdf",
    "f_quantile",
    "beta_pdf",
    "beta_cdf",
    "beta_quantile",
    "poisson_pdf",
    "poisson_cdf",
    "binomial_pdf",
    "binomial_cdf",
    "lognormal_pdf",
    "lognormal_cdf",
    "cauchy_pdf",
    "cauchy_cdf",
    "exponential_cdf",
    "invert",
]


def normal_cdf(x: float, sigma: float = 1.0, x0: float = 0.0) -> float:
    return 0.5 * (1.0 + math.erf((x - x0) / (sigma * math.sqrt(2.0))))


def normal_quantile(p: float, sigma: float = 1.0) -> float:
    """The ``x`` a Gaussian of width ``sigma`` falls below with chance ``p``."""
    if p <= 0.0 or p >= 1.0:
        return -math.inf if p <= 0.0 else math.inf
    return sigma * statistics.NormalDist().inv_cdf(p)


def invert(cdf: Callable[[float], float], p: float, low: float, high: float) -> float:
    """The ``x`` where a rising ``cdf`` reaches ``p``, by bisection from ``[low, high]``.

    The bracket is widened upwards until it holds the answer, then halved
    until its ends are neighbouring doubles.
    """
    while cdf(high) < p:
        low, high = high, high * 2.0 + 1.0
    middle = 0.5 * (low + high)
    while middle not in (low, high):
        low, high = (middle, high) if cdf(middle) < p else (low, middle)
        middle = 0.5 * (low + high)
    return middle


def gamma_pdf(x: float, alpha: float, theta: float, x0: float = 0.0) -> float:
    """The gamma density of shape ``alpha`` and scale ``theta``, from ``x0``."""
    u = x - x0
    if u < 0:
        return 0.0
    if u == 0:
        return 1.0 / theta if alpha == 1 else 0.0
    return math.exp((alpha - 1) * math.log(u / theta) - u / theta - math.lgamma(alpha)) / theta


def chi2_cdf(x: float, r: float, x0: float = 0.0) -> float:
    return incomplete_gamma(0.5 * r, 0.5 * (x - x0))


def chi2_quantile(p: float, r: float) -> float:
    """The chi-square on ``r`` degrees of freedom that is exceeded with chance ``1 - p``."""
    if p <= 0.0:
        return 0.0
    return invert(lambda x: chi2_cdf(x, r), p, 0.0, max(r, 1.0))


def student_pdf(x: float, r: float) -> float:
    head = math.lgamma((r + 1) / 2) - math.lgamma(r / 2) - 0.5 * math.log(r * math.pi)
    return math.exp(head - (r + 1) / 2 * math.log1p(x * x / r))


def student_cdf(x: float, r: float) -> float:
    """The chance a Student's t on ``r`` degrees falls below ``x``."""
    tail = 0.5 * regularized_beta(r / (r + x * x), r / 2, 0.5)
    return 1.0 - tail if x > 0 else tail


def student_quantile(p: float, r: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return -math.inf if p <= 0.0 else math.inf
    if p == 0.5:
        return 0.0
    upper = abs(normal_quantile(p)) * 2.0 + 1.0
    found = invert(lambda x: student_cdf(x, r), max(p, 1.0 - p), 0.0, upper)
    return found if p > 0.5 else -found


def f_pdf(x: float, n: float, m: float) -> float:
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0 if n > 2 else (1.0 if n == 2 else math.inf)
    head = math.lgamma((n + m) / 2) - math.lgamma(n / 2) - math.lgamma(m / 2)
    body = 0.5 * n * math.log(n / m) + (0.5 * n - 1) * math.log(x)
    return math.exp(head + body - 0.5 * (n + m) * math.log1p(n * x / m))


def f_cdf(x: float, n: float, m: float) -> float:
    return 0.0 if x <= 0 else regularized_beta(n * x / (n * x + m), n / 2, m / 2)


def f_quantile(p: float, n: float, m: float) -> float:
    if p <= 0.0:
        return 0.0
    found = beta_quantile(p, n / 2, m / 2)
    return m * found / (n * (1.0 - found)) if found < 1.0 else math.inf


def beta_pdf(x: float, a: float, b: float) -> float:
    if x < 0 or x > 1:
        return 0.0
    if x in (0.0, 1.0):
        edge = a if x == 0 else b
        return 0.0 if edge > 1 else (math.inf if edge < 1 else (a if x == 1 else b))
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return math.exp((a - 1) * math.log(x) + (b - 1) * math.log1p(-x) - log_beta)


def beta_cdf(x: float, a: float, b: float) -> float:
    return regularized_beta(x, a, b)


def poisson_pdf(n: float, mu: float) -> float:
    if n < 0:
        return 0.0
    if n == 0:
        return math.exp(-mu)
    return math.exp(n * math.log(mu) - mu - math.lgamma(n + 1))


def poisson_cdf(n: float, mu: float) -> float:
    return incomplete_gamma_c(math.floor(n) + 1.0, mu) if n >= 0 else 0.0


def binomial_pdf(k: float, p: float, n: float) -> float:
    if k < 0 or k > n:
        return 0.0
    log_choose = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    if p in (0.0, 1.0):
        return float((k == 0) if p == 0 else (k == n))
    return math.exp(log_choose + k * math.log(p) + (n - k) * math.log1p(-p))


def binomial_cdf(k: float, p: float, n: float) -> float:
    if k >= n:
        return 1.0
    whole = math.floor(k)
    return 0.0 if whole < 0 else 1.0 - regularized_beta(p, whole + 1.0, n - whole)


def lognormal_pdf(x: float, m: float, s: float, x0: float = 0.0) -> float:
    u = x - x0
    if u <= 0:
        return 0.0
    z = (math.log(u) - m) / s
    return math.exp(-0.5 * z * z) / (u * s * math.sqrt(2 * math.pi))


def lognormal_cdf(x: float, m: float, s: float, x0: float = 0.0) -> float:
    u = x - x0
    return 0.0 if u <= 0 else normal_cdf(math.log(u), s, m)


def cauchy_pdf(x: float, b: float = 1.0, x0: float = 0.0) -> float:
    return b / (math.pi * ((x - x0) ** 2 + b * b))


def cauchy_cdf(x: float, b: float = 1.0, x0: float = 0.0) -> float:
    return 0.5 + math.atan((x - x0) / b) / math.pi


def exponential_cdf(x: float, lam: float, x0: float = 0.0) -> float:
    return 0.0 if x < x0 else -math.expm1(-lam * (x - x0))
