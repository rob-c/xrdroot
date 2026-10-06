"""The functions RooStats' Bayesian calculator integrates: the likelihood times the prior, and
its integrals.

``RooFunctor`` binds a function to some of its variables - each call sets
them and asks for the value, normalised over them, as ``RooRealBinding``
asks, warning of any it does not depend on. The likelihood function is
``exp(-(nll - offset)) * prior``; the posterior is it integrated over the
nuisance parameters, at each value of the parameter of interest - by
GSL's QAGS in one dimension, ``AdaptiveIntegratorMultiDim`` in more - and
the cumulative posterior integrates the parameter of interest too.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable
from typing import Any

from ..numerics.qags import qags
from ..roofit.collections import as_list
from ..roofit.messages import ERROR, INFO, WARNING, log, log_plain
from ..roofit.printing import g
from ..roofit.real import RooAbsReal

__all__ = ["CdfFunction", "Functor", "PosteriorFunction", "Posterior"]

#: ``IntegratorOneDimOptions``' and ``IntegratorMultiDimOptions``' defaults.
ONE_DIM: dict[str, Any] = {"abs": 1e-9, "rel": 1e-9, "limit": 1000}
MULTI_DIM: dict[str, Any] = {"abs": 0.0, "rel": 1e-9, "calls": 100000, "size": 100000}


class Functor:
    """``RooFunctor``: ``func`` as a function of ``params``, normalised over them."""

    def __init__(self, func: Any, params: Any) -> None:
        self.func = func
        self.params = as_list(params)
        self.names = frozenset(p.GetName() for p in self.params)
        for par in self.params:
            if par.GetName() not in func.dependents():
                log(None, WARNING, "InputArguments", f"RooRealBinding: The function "
                    f"{func.GetName()} does not depend on the parameter {par.GetName()}. Note "
                    "that passing copies of the parameters is not supported.")  # fmt: skip
        self.calls = 0

    def __call__(self, x: Any) -> float:
        for par, value in zip(self.params, x, strict=False):
            par.setVal(float(value))
        self.calls += 1
        return float(self.func.getVal(self.params))


class Likelihood:
    """``LikelihoodFunction``: ``exp(-(nll - offset))``, times the prior if there is one."""

    def __init__(self, nll: Functor, prior: Functor | None, offset: float) -> None:
        self.nll, self.prior, self.offset = nll, prior, offset
        self.max = 0.0

    def __call__(self, x: Any) -> float:
        nll = self.nll(x) - self.offset
        try:
            found = math.exp(-nll)
        except OverflowError:  # C's exp is infinite there
            found = math.inf
        if self.prior is not None:
            found *= self.prior(x)
        if found > self.max:
            self.max = found
            if found > 1e10:
                where = "".join(f" x[{i} ] = {g(v)}" for i, v in enumerate(x))
                log(None, WARNING, "Eval", "LikelihoodFunction::()  WARNING - Huge likelihood "
                    f"value found for  parameters {where}  nll = {g(nll)} L = {g(found)}")
        return found


def integrate(like: Likelihood, lows: list[float], highs: list[float],
              calls: int = 0) -> tuple[float, float]:  # fmt: skip
    """The likelihood's integral over a box, and its error: QAGS in 1D, cubature beyond."""
    if len(lows) == 1:
        found, error, _status = qags(lambda x: like([x]), lows[0], highs[0], ONE_DIM["abs"],
                                     ONE_DIM["rel"], ONE_DIM["limit"])  # fmt: skip
        return found, error
    from ..roofit.cubature import adaptive_integral

    most = calls or MULTI_DIM["calls"]
    result = adaptive_integral(like, lows, highs, MULTI_DIM["abs"], MULTI_DIM["rel"],
                               max_pts=most, size=MULTI_DIM["size"])  # fmt: skip
    return result.value, result.relerr * abs(result.value)


class PosteriorFunction:
    """``PosteriorFunction``: the likelihood integrated over the nuisance parameters, at a value
    of the parameter of interest."""

    def __init__(self, nll: Any, poi: Any, nuisance: list[Any], prior: Any, norm: float,
                 offset: float, calls: int = 0) -> None:  # fmt: skip
        self.poi = poi
        self.like = Likelihood(Functor(nll, nuisance), Functor(prior, nuisance) if prior else None,
                               offset)  # fmt: skip
        self.lows = [float(p.getMin()) for p in nuisance]
        self.highs = [float(p.getMax()) for p in nuisance]
        self.norm, self.calls, self.error = norm, calls, 0.0

    def __call__(self, x: float) -> float:
        self.poi.setVal(float(x))
        if not self.lows:
            found, error = self.like([x]), 0.0
        else:
            found, error = integrate(self.like, self.lows, self.highs, self.calls)
        if found != 0 and error / found > 0.2:
            log_plain(None, WARNING, "NumericIntegration", "PosteriorFunction::DoEval - Error "
                      f"from integration in {len(self.lows)} Dim is larger than 20 % x = {g(x)} "
                      f"p(x) = {g(found)} +/- {g(error)}\n")  # fmt: skip
        self.error = error / self.norm
        return found / self.norm


class CdfFunction:
    """``PosteriorCdfFunction``: the posterior integrated from the lower end up to ``x``,
    normalised by its whole integral, less an offset for root finding."""

    def __init__(self, nll: Any, params: list[Any], prior: Any, offset: float) -> None:
        self.like = Likelihood(Functor(nll, params), Functor(prior, params) if prior else None,
                               offset)  # fmt: skip
        self.lows = [float(p.getMin()) for p in params]
        self.highs = [float(p.getMax()) for p in params]
        self.max_poi = self.highs[0]
        self.norm, self.norm_error, self.offset = 1.0, 0.0, 0.0
        self.has_norm, self.error = False, False
        self.cached: dict[float, float] = {}
        self.norm = self(self.max_poi)
        if self.error:
            log_plain(None, ERROR, "NumericIntegration", "PosteriorFunction::Error computing "
                      f"normalization - norm = {g(self.norm)}\n")  # fmt: skip
        self.has_norm = True
        self.cached = {self.lows[0]: 0.0, self.highs[0]: 1.0}

    def SetOffset(self, offset: float) -> None:
        self.offset = float(offset)

    def __call__(self, x: float) -> float:
        self.highs[0] = x
        if x <= self.lows[0]:
            return -self.offset
        if x >= self.max_poi and self.has_norm:
            return 1.0 - self.offset
        start = self._from_cached(x) if self.has_norm else 0.0
        cdf, error = integrate(self.like, list(self.lows), list(self.highs))
        self._checked(x, cdf, error)
        if not self.has_norm:
            log(None, INFO, "NumericIntegration", f"PosteriorCdfFunction - integral of posterior "
                f"= {g(cdf)} +/- {g(error)}")  # fmt: skip
            self.norm_error = error
            return cdf
        normcdf = cdf / self.norm + start
        self.cached[x] = normcdf
        errnorm = math.sqrt(error * error + normcdf * normcdf * self.norm_error**2) / self.norm
        if normcdf > 1.0 + 3 * errnorm:
            log(None, WARNING, "NumericIntegration", "PosteriorCdfFunction: normalized cdf values "
                f"is larger than 1 x = {g(x)} normcdf(x) = {g(normcdf)} +/- "
                f"{g(error / self.norm)}")  # fmt: skip
        return normcdf - self.offset

    def _from_cached(self, x: float) -> float:
        """The integral starts at the nearest cached point below ``x`` - the lower end is one."""
        keys = sorted(self.cached)
        where = bisect.bisect_right(keys, x) - 1
        self.lows[0] = keys[where]
        return self.cached[keys[where]]

    def _checked(self, x: float, cdf: float, error: float) -> None:
        """A failed integral, or one too imprecise, said."""
        if math.isnan(cdf) or cdf > 1.7976931348623157e308:
            log_plain(None, ERROR, "NumericIntegration", "PosteriorFunction::Error computing "
                      f"integral - cdf = {g(cdf)}\n")  # fmt: skip
            self.error = True
        if cdf != 0 and error / cdf > 0.2:
            log(None, WARNING, "NumericIntegration", "PosteriorCdfFunction: integration error  is "
                f"larger than 20 %   x0 = {g(self.lows[0])} x = {g(x)} cdf(x) = {g(cdf)} +/- "
                f"{g(error)}")  # fmt: skip


class Posterior(RooAbsReal):
    """``RooFunctor1DBinding``: a Python function of the parameter of interest as a RooFit
    function of it."""

    def __init__(self, name: str, function: Callable[[float], float], poi: Any) -> None:
        super().__init__(name, name)
        self.function = function
        self.poi = self._proxy("x", poi)

    def compute(self, ctx: Any) -> Any:
        import numpy as np

        values = np.asarray(self.poi.compute(ctx), dtype=np.float64)
        if not values.ndim:
            return float(self.function(float(values)))
        return np.array([self.function(float(v)) for v in values.reshape(-1)]).reshape(values.shape)
