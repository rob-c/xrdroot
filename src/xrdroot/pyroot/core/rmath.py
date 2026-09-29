"""``ROOT.Math``: the MathCore namespace - distributions, special functions and vectors.

Densities are ``*_pdf``, lower tails ``*_cdf``, upper tails ``*_cdf_c`` and
their inverses ``*_quantile`` and ``*_quantile_c``, with ROOT's argument
orders: the variable first, then the shape, then the location ``x0``. The
vectors are those of :mod:`.genvector`, and ``VectorUtil`` beside them.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from ...efficiency import regularized_beta
from ...fit.defaults import DEFAULTS, minimizer_algo, minimizer_type, set_minimizer
from ...function import special
from ...stats import incomplete_gamma, incomplete_gamma_c
from . import distributions as dist
from .genvector import *  # noqa: F403
from .genvector import __all__ as _vectors
from .legendres import (  # noqa: F401 - ROOT::Math's Legendre polynomials, by name
    assoc_legendre,
    legendre,
    sph_legendre,
)
from .mathtools import (  # noqa: F401 - ROOT::Math's function objects and tools, by name
    Factory,
    Functor,
    Functor1D,
    GradFunctor,
    GradFunctor1D,
    Integrator,
    IntegratorOneDim,
    Minimizer,
    RootFinder,
)

# -- the Gaussian -------------------------------------------------------------------------------


def gaussian_pdf(x: float, sigma: float = 1.0, x0: float = 0.0) -> float:
    return float(special.gaussian_pdf(x, sigma, x0))


def gaussian_cdf(x: float, sigma: float = 1.0, x0: float = 0.0) -> float:
    return dist.normal_cdf(x, sigma, x0)


def gaussian_cdf_c(x: float, sigma: float = 1.0, x0: float = 0.0) -> float:
    return 0.5 * math.erfc((x - x0) / (sigma * math.sqrt(2.0)))


def gaussian_quantile(z: float, sigma: float) -> float:
    return dist.normal_quantile(z, sigma)


def gaussian_quantile_c(z: float, sigma: float) -> float:
    return -dist.normal_quantile(z, sigma)


normal_pdf, normal_cdf, normal_cdf_c = gaussian_pdf, gaussian_cdf, gaussian_cdf_c
normal_quantile, normal_quantile_c = gaussian_quantile, gaussian_quantile_c

# -- chi-square, gamma, exponential ---------------------------------------------------------------


def chisquared_pdf(x: float, r: float, x0: float = 0.0) -> float:
    return dist.gamma_pdf(x, r / 2.0, 2.0, x0)


def chisquared_cdf(x: float, r: float, x0: float = 0.0) -> float:
    return incomplete_gamma(0.5 * r, 0.5 * (x - x0))


def chisquared_cdf_c(x: float, r: float, x0: float = 0.0) -> float:
    return incomplete_gamma_c(0.5 * r, 0.5 * (x - x0))


def chisquared_quantile(z: float, r: float) -> float:
    return dist.chi2_quantile(z, r)


def chisquared_quantile_c(z: float, r: float) -> float:
    return dist.chi2_quantile(1.0 - z, r)


def gamma_pdf(x: float, alpha: float, theta: float, x0: float = 0.0) -> float:
    return dist.gamma_pdf(x, alpha, theta, x0)


def gamma_cdf(x: float, alpha: float, theta: float, x0: float = 0.0) -> float:
    return incomplete_gamma(alpha, (x - x0) / theta)


def gamma_cdf_c(x: float, alpha: float, theta: float, x0: float = 0.0) -> float:
    return incomplete_gamma_c(alpha, (x - x0) / theta)


def gamma_quantile(z: float, alpha: float, theta: float) -> float:
    return theta * dist.invert(lambda u: incomplete_gamma(alpha, u), z, 0.0, max(alpha, 1.0))


def gamma_quantile_c(z: float, alpha: float, theta: float) -> float:
    return gamma_quantile(1.0 - z, alpha, theta)


def exponential_pdf(x: float, lamda: float, x0: float = 0.0) -> float:
    return 0.0 if x < x0 else lamda * math.exp(-lamda * (x - x0))


def exponential_cdf(x: float, lamda: float, x0: float = 0.0) -> float:
    return dist.exponential_cdf(x, lamda, x0)


def exponential_cdf_c(x: float, lamda: float, x0: float = 0.0) -> float:
    return 1.0 if x < x0 else math.exp(-lamda * (x - x0))


def exponential_quantile(z: float, lamda: float) -> float:
    return -math.log1p(-z) / lamda


def exponential_quantile_c(z: float, lamda: float) -> float:
    return -math.log(z) / lamda


# -- counts -------------------------------------------------------------------------------------


def poisson_pdf(n: float, mu: float) -> float:
    return dist.poisson_pdf(n, mu)


def poisson_cdf(n: float, mu: float) -> float:
    return dist.poisson_cdf(n, mu)


def poisson_cdf_c(n: float, mu: float) -> float:
    return incomplete_gamma(math.floor(n) + 1.0, mu) if n >= 0 else 1.0


def binomial_pdf(k: float, p: float, n: float) -> float:
    return dist.binomial_pdf(k, p, n)


def binomial_cdf(k: float, p: float, n: float) -> float:
    return dist.binomial_cdf(k, p, n)


def binomial_cdf_c(k: float, p: float, n: float) -> float:
    return 1.0 - dist.binomial_cdf(k, p, n)


# -- the rest -------------------------------------------------------------------------------------


def landau_pdf(x: float, xi: float = 1.0, x0: float = 0.0) -> float:
    return float(special.landau_pdf(x, xi, x0))


def breitwigner_pdf(x: float, gamma: float, x0: float = 0.0) -> float:
    return float(special.breitwigner_pdf(x, gamma, x0))


def breitwigner_cdf(x: float, gamma: float, x0: float = 0.0) -> float:
    return dist.cauchy_cdf(x, gamma / 2.0, x0)


def breitwigner_cdf_c(x: float, gamma: float, x0: float = 0.0) -> float:
    return 1.0 - dist.cauchy_cdf(x, gamma / 2.0, x0)


def cauchy_pdf(x: float, b: float = 1.0, x0: float = 0.0) -> float:
    return dist.cauchy_pdf(x, b, x0)


def cauchy_cdf(x: float, b: float, x0: float = 0.0) -> float:
    return dist.cauchy_cdf(x, b, x0)


def cauchy_cdf_c(x: float, b: float, x0: float = 0.0) -> float:
    return 1.0 - dist.cauchy_cdf(x, b, x0)


def lognormal_pdf(x: float, m: float, s: float, x0: float = 0.0) -> float:
    return dist.lognormal_pdf(x, m, s, x0)


def lognormal_cdf(x: float, m: float, s: float, x0: float = 0.0) -> float:
    return dist.lognormal_cdf(x, m, s, x0)


def lognormal_cdf_c(x: float, m: float, s: float, x0: float = 0.0) -> float:
    return 1.0 - dist.lognormal_cdf(x, m, s, x0)


def tdistribution_pdf(x: float, r: float, x0: float = 0.0) -> float:
    return dist.student_pdf(x - x0, r)


def tdistribution_cdf(x: float, r: float, x0: float = 0.0) -> float:
    return dist.student_cdf(x - x0, r)


def tdistribution_cdf_c(x: float, r: float, x0: float = 0.0) -> float:
    return dist.student_cdf(x0 - x, r)


def tdistribution_quantile(z: float, r: float) -> float:
    return dist.student_quantile(z, r)


def tdistribution_quantile_c(z: float, r: float) -> float:
    return -dist.student_quantile(z, r)


def fdistribution_pdf(x: float, n: float, m: float, x0: float = 0.0) -> float:
    return dist.f_pdf(x - x0, n, m)


def fdistribution_cdf(x: float, n: float, m: float, x0: float = 0.0) -> float:
    return dist.f_cdf(x - x0, n, m)


def fdistribution_cdf_c(x: float, n: float, m: float, x0: float = 0.0) -> float:
    return 1.0 - dist.f_cdf(x - x0, n, m)


def fdistribution_quantile(z: float, n: float, m: float) -> float:
    return dist.f_quantile(z, n, m)


def fdistribution_quantile_c(z: float, n: float, m: float) -> float:
    return dist.f_quantile(1.0 - z, n, m)


def beta_pdf(x: float, a: float, b: float) -> float:
    return dist.beta_pdf(x, a, b)


def beta_cdf(x: float, a: float, b: float) -> float:
    return dist.beta_cdf(x, a, b)


def beta_cdf_c(x: float, a: float, b: float) -> float:
    return 1.0 - dist.beta_cdf(x, a, b)


def beta_quantile(z: float, a: float, b: float) -> float:
    return dist.beta_quantile(z, a, b)


def beta_quantile_c(z: float, a: float, b: float) -> float:
    return dist.beta_quantile(1.0 - z, a, b)


def uniform_pdf(x: float, a: float, b: float, x0: float = 0.0) -> float:
    return 1.0 / (b - a) if a <= x - x0 < b else 0.0


def uniform_cdf(x: float, a: float, b: float, x0: float = 0.0) -> float:
    return min(max((x - x0 - a) / (b - a), 0.0), 1.0)


def uniform_cdf_c(x: float, a: float, b: float, x0: float = 0.0) -> float:
    return 1.0 - uniform_cdf(x, a, b, x0)


def crystalball_function(
    x: float, alpha: float, n: float, sigma: float, mean: float = 0.0
) -> float:
    return float(special.crystalball_function(x, alpha, n, sigma, mean))


def crystalball_pdf(x: float, alpha: float, n: float, sigma: float, mean: float = 0.0) -> float:
    return float(special.crystalball_pdf(x, alpha, n, sigma, mean))


# -- special functions ------------------------------------------------------------------------


def erf(x: float) -> float:
    return math.erf(x)


def erfc(x: float) -> float:
    return math.erfc(x)


def tgamma(x: float) -> float:
    return math.gamma(x)


def lgamma(x: float) -> float:
    return math.lgamma(x)


def beta(x: float, y: float) -> float:
    return math.exp(math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y))


def inc_gamma(a: float, x: float) -> float:
    return incomplete_gamma(a, x)


def inc_gamma_c(a: float, x: float) -> float:
    return incomplete_gamma_c(a, x)


def inc_beta(x: float, a: float, b: float) -> float:
    return regularized_beta(x, a, b)


def Pi() -> float:
    return math.pi


class MinimizerOptions:
    """``ROOT::Math::MinimizerOptions``: the minimiser a fit uses, which is Minuit2's MIGRAD here.

    The defaults can be set and read back; fits run through iminuit whatever
    is set, so a script choosing ``"Minuit"`` or ``"Minuit2"`` gets the same.
    """

    #: The table the engine reads too: RooStats' calculators minimise with these.
    _defaults: ClassVar[dict[str, Any]] = DEFAULTS

    @classmethod
    def SetDefaultMinimizer(cls, name: Any, algorithm: Any = None) -> None:
        set_minimizer(name, algorithm)

    @classmethod
    def DefaultMinimizerType(cls) -> str:
        return minimizer_type()

    @classmethod
    def DefaultMinimizerAlgo(cls) -> str:
        return minimizer_algo()


def _default_accessors(key: str) -> None:
    setattr(MinimizerOptions, f"SetDefault{key}",
            classmethod(lambda cls, value: cls._defaults.__setitem__(key, value)))  # fmt: skip
    setattr(MinimizerOptions, f"Default{key}", classmethod(lambda cls: cls._defaults[key]))


for _key in ("Tolerance", "Precision", "PrintLevel", "MaxFunctionCalls", "MaxIterations",
             "Strategy", "ErrorDef"):  # fmt: skip
    _default_accessors(_key)


def _exported(name: str, value: object) -> bool:
    return (
        callable(value)
        and not name.startswith("_")
        and getattr(value, "__module__", "") == __name__
    )


__all__ = sorted(
    [name for name, value in list(globals().items()) if _exported(name, value)]
    + list(_vectors)
    + [
        "Factory",
        "Minimizer",
        "Functor",
        "Functor1D",
        "GradFunctor",
        "GradFunctor1D",
        "Integrator",
        "IntegratorOneDim",
        "RootFinder",
        "assoc_legendre",
        "legendre",
        "sph_legendre",
    ]
)
