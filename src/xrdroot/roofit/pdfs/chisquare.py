"""``RooChiSquarePdf`` and ``RooNonCentralChiSquare``: the chi-square density, central and not.

The central one is ``x^(n/2-1) e^(-x/2) / (Gamma(n/2) 2^(n/2))``, integrated
over ``x`` by the regularised incomplete gamma function; a likelihood's
events take RooFit's kernel, VDT's ``fast_exp`` of the logarithm.

The non-central one of non-centrality ``lambda`` is, by default, MathMore's
Bessel-function expression - here the modified Bessel function's series -
or with ``SetForceSum(True)`` RooFit's Poisson-weighted sum of central
densities, from the dominant term out, warning once when it stops before
the terms fall below the tolerance. Its integral over ``x`` is always that
sum, of central distribution functions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .. import kernels
from ..messages import INFO, WARNING, log
from ..pdf import RooAbsPdf, check_range
from ..printing import g
from ..real import Context
from .basic import ref

__all__ = ["RooChiSquarePdf", "RooNonCentralChiSquare"]

#: ``ln 2``, as the kernel spells it.
LN2 = 0.693147180559945309417232121458


def _each(function: Any, *columns: Any) -> Any:
    """``function`` of each event's values - an array if any column is one."""
    arrays = np.broadcast_arrays(*(np.asarray(c, dtype=np.float64) for c in columns))
    if not arrays[0].ndim:
        return function(*(float(a) for a in arrays))
    flat = [a.reshape(-1) for a in arrays]
    found = [function(*(float(a[i]) for a in flat)) for i in range(flat[0].size)]
    return np.asarray(found, dtype=np.float64).reshape(arrays[0].shape)


def _central(x: float, ndof: float) -> float:
    """``RooChiSquarePdf::evaluate``."""
    if x <= 0:
        return 0.0
    return x ** (ndof / 2.0 - 1.0) * math.exp(-x / 2.0) / math.gamma(ndof / 2.0) / 2.0 ** (ndof / 2.0)


class RooChiSquarePdf(RooAbsPdf):
    """The chi-square density of ``ndof`` degrees of freedom."""

    def __init__(self, name: Any, title: Any, x: Any, ndof: Any) -> None:
        super().__init__(name, title)
        self.x = self._proxy("x", ref(x))
        self.ndof = self._proxy("ndof", ref(ndof))
        check_range(self, [x, ndof], 0.0)

    def compute(self, ctx: Context) -> Any:
        x, ndof = self.x.compute(ctx), self.ndof.compute(ctx)
        if not kernels.active():
            return _each(_central, x, ndof)
        ndof = float(np.asarray(ndof).reshape(-1)[0])
        arg = (ndof - 2) * kernels.fast_log(x) - np.asarray(x, dtype=np.float64) - ndof * LN2
        return (1.0 / math.gamma(ndof / 2.0)) * kernels.fast_exp(0.5 * arg)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        if rng or self.x.GetName() not in names or not self.x.isFundamental():
            return frozenset()
        return frozenset([self.x.GetName()])

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        from ...stats import incomplete_gamma

        low, high = self.x.getMin(rng), self.x.getMax(rng)
        half = lambda n: float(n) / 2.0  # noqa: E731
        return _each(
            lambda n: incomplete_gamma(half(n), high / 2) - incomplete_gamma(half(n), low / 2),
            self.ndof.compute(ctx),
        )


def bessel_i(nu: float, z: float) -> float:
    """The modified Bessel function of the first kind, ``I_nu(z)``, by its power series - each
    term the last times ``(z/2)^2 / (m (m + nu))`` - summed until the terms stop counting."""
    if z == 0.0:
        return 1.0 if nu == 0 else 0.0
    quarter = 0.25 * z * z
    term = math.exp(nu * math.log(0.5 * z) - math.lgamma(nu + 1.0))
    total, m = term, 0
    while term > total * 1e-17 or m < quarter:
        m += 1
        term *= quarter / (m * (m + nu))
        total += term
    return total


def noncentral_pdf(x: float, r: float, lam: float) -> float:
    """``ROOT::Math::noncentral_chisquared_pdf``: MathMore's Bessel-function expression."""
    if x < 0:
        return 0.0
    z = math.sqrt(lam * x)
    return 0.5 * math.exp(-0.5 * (x + lam)) * (x / lam) ** (0.25 * r - 0.5) * bessel_i(0.5 * r - 1.0, z)


def _term(x: float, k: float, lam: float, i: int, cdf: Any = None) -> float:
    """The sum's ``i``-th term: the Poisson weight of ``i`` times the central density of ``k +
    2 i`` - or, given ``cdf``, the difference of its distribution function over a range."""
    from ...pyroot.core.rmath import chisquared_pdf

    weight = math.exp(-lam / 2.0) * (lam / 2.0) ** i
    if cdf is None:
        return weight * chisquared_pdf(x, k + 2 * i) / math.gamma(i + 1)
    return weight * (cdf(k + 2 * i)[1] / math.gamma(i + 1) - cdf(k + 2 * i)[0] / math.gamma(i + 1))


class RooNonCentralChiSquare(RooAbsPdf):
    """The non-central chi-square density of ``k`` degrees of freedom and non-centrality
    ``lambda``."""

    def __init__(self, name: Any, title: Any, x: Any, k: Any, lambda_: Any) -> None:
        super().__init__(name, title)
        self.x = self._proxy("x", ref(x))
        self.k = self._proxy("k", ref(k))
        self.lam = self._proxy("lambda", ref(lambda_))
        self._tolerance, self._iterations, self._force = 1e-3, 10, False
        self._warned_convergence = self._warned_sum = False

    def SetForceSum(self, flag: bool) -> None:
        self._force = bool(flag)

    def SetErrorTolerance(self, tolerance: float) -> None:
        self._tolerance = float(tolerance)

    def SetMaxIters(self, iterations: int) -> None:
        self._iterations = int(iterations)

    def compute(self, ctx: Context) -> Any:
        return _each(self._one, self.x.compute(ctx), self.k.compute(ctx), self.lam.compute(ctx))

    def _one(self, x: float, k: float, lam: float) -> float:
        """``evaluate``: a value at the range's lower end nudged a thousandth of it in."""
        from ...pyroot.core.rmath import chisquared_pdf

        at = x if x > 0 else self.x.getMin() + 1e-3 * (self.x.getMax() - self.x.getMin())
        if lam == 0:
            return float(chisquared_pdf(at, k))
        if not self._force:
            return noncentral_pdf(at, k, lam)
        if not self._warned_sum:
            self._warned_sum = True
            log(None, INFO, "InputArguments", "RooNonCentralChiSquare sum being forced")
        where = f"for x={g(x)} k={g(k)}, lambda={g(lam)}"
        return self._sum(lambda i: _term(at, k, lam, i), lam, "", where)

    def _sum(self, term: Any, lam: float, what: str, where: str) -> float:
        """From the dominant term up until one falls below the tolerance - or the terms run out,
        said once - then down to the first."""
        dominant = int(math.floor(lam / 2))
        total, i = 0.0, dominant
        while True:
            value = term(i)
            total += value
            if value / total < self._tolerance:
                break
            if i > dominant + self._iterations:
                if not self._warned_convergence:
                    self._warned_convergence = True
                    log(None, WARNING, "Eval", f"RooNonCentralChiSquare {what}did not converge: "
                        f"{where} fractional error = {g(value / total)}\n either adjust tolerance "
                        "with SetErrorTolerance(tol) or max_iter with SetMaxIter(max_it)")  # fmt: skip
                break
            i += 1
        for i in range(dominant - 1, -1, -1):
            total += term(i)
        return total

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        if self.x.GetName() not in names or not self.x.isFundamental():
            return frozenset()
        return frozenset([self.x.GetName()])

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        return _each(lambda k, lam: self._integral(k, lam, rng), self.k.compute(ctx),
                     self.lam.compute(ctx))  # fmt: skip

    def _integral(self, k: float, lam: float, rng: Any) -> float:
        from ...pyroot.core.rmath import chisquared_cdf

        low, high = self.x.getMin(rng), self.x.getMax(rng)
        if lam == 0:
            return float(chisquared_cdf(high, k) - chisquared_cdf(low, k))
        ends = lambda n: (chisquared_cdf(low, n), chisquared_cdf(high, n))  # noqa: E731
        return self._sum(lambda i: _term(0.0, k, lam, i, ends), lam, "Normalization ",
                         f"for k={g(k)}, lambda={g(lam)}")  # fmt: skip
