"""``LikelihoodInterval``: the region where the profile likelihood ratio is below Wilks' cut.

A point is in the interval when the chi-square probability of twice its
profile value exceeds the test size. Its ends in one parameter are found as
RooStats finds them: MIGRAD on the likelihood itself - Minuit2 through
``ROOT::Math::Minimizer``, at its default settings - from the best fit, and
MINOS at an error level of half the chi-square quantile of the confidence
level, so the ends are where ``-log L`` has risen by that much with every
other parameter profiled; the contour of two is MINOS' contour at the
two-dimensional level.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..fit.defaults import default
from ..fit.minuit import iminuit
from ..roofit.collections import RooArgList, RooArgSet, as_list
from ..roofit.messages import ERROR, FATAL, INFO, WARNING, log, log_plain
from ..roofit.printing import g
from ..stats import chisquare_quantile, prob
from .intervals import ConfInterval
from .modelconfig import quieted

__all__ = ["LikelihoodInterval", "floating"]


class LikelihoodInterval(ConfInterval):
    """The profile likelihood ratio's interval: a profile, its parameters, the best fit."""

    def __init__(self, name: Any = "", lr: Any = None, params: Any = (), bestParams: Any = None):
        super().__init__(name)
        self._parameters = RooArgSet(as_list(params))
        self._ratio = lr
        self._best = bestParams
        self._lower: dict[str, float] = {}
        self._upper: dict[str, float] = {}
        self._minuit: Any = None
        self._params: list[Any] = []

    def GetLikelihoodRatio(self) -> Any:
        return self._ratio

    def GetBestFitParameters(self) -> Any:
        return self._best

    def SetConfidenceLevel(self, cl: float) -> None:
        self._cl = float(cl)
        self.ResetLimits()

    def ResetLimits(self) -> None:
        self._lower.clear()
        self._upper.clear()

    def IsInInterval(self, point: Any) -> bool:
        """Whether ``point`` passes Wilks' cut, with RooFit's messages held back meanwhile."""
        with quieted(FATAL):
            if not self.CheckParameters(point):
                log(self, ERROR, "InputArguments", "parameters don't match")
                return False
            variables = RooArgSet(list(self._ratio.getVariables()))
            variables.assign(as_list(point))
            value = self._ratio.getVal()
            if value < 0:
                log(self, WARNING, "Eval", "The likelihood ratio is < 0, indicates a bad minimum "
                    "or numerical precision problems.  Will return true")  # fmt: skip
                return True
            return prob(2 * value, len(as_list(point))) >= 1.0 - self._cl

    # -- the ends ---------------------------------------------------------------

    def LowerLimit(self, param: Any, *status: Any) -> float:
        return self._limits(param)[0]

    def UpperLimit(self, param: Any, *status: Any) -> float:
        return self._limits(param)[1]

    def FindLimits(self, param: Any, lower: Any = None, upper: Any = None) -> bool:
        """Both ends, into ``lower`` and ``upper`` if they are cells; whether they were found."""
        low, high, found = self._limits(param)
        for cell, value in ((lower, low), (upper, high)):
            if cell is not None and hasattr(cell, "value"):
                cell.value = value
        return found

    def _limits(self, param: Any) -> tuple[float, float, bool]:
        name = param.GetName()
        if name in self._lower and name in self._upper:
            return self._lower[name], self._upper[name], True
        names = [p.GetName() for p in floating(self._ratio)]
        if name not in names:
            log_plain(self, ERROR, "InputArguments", f"Error - invalid parameter {name} specified "
                      "for finding the interval limits \n")  # fmt: skip
            return 0.0, 0.0, False
        if self._minuit is None and not self._create_minimizer():
            log_plain(self, ERROR, "Eval", "Error returned from minimization of likelihood "
                      "function - cannot find interval limits \n")  # fmt: skip
            return 0.0, 0.0, False
        level = chisquare_quantile(self.ConfidenceLevel(), 1) / 2
        low, high = self._minos(name, level)
        at = float(self._minuit.values[name])
        if low == 0:
            lower = float(param.getMin())
            log_plain(self, WARNING, "Minimization", f"Warning: lower value for {name} is at "
                      f"limit {g(lower)}\n")  # fmt: skip
        else:
            lower = at + low
        if high == 0:
            log_plain(self, WARNING, "Minimization", f"Warning: upper value for {name} is at "
                      "limit 0\n")  # fmt: skip
            upper = float(param.getMax())
        else:
            upper = at + high
        self._lower[name], self._upper[name] = lower, upper
        return lower, upper, True

    def _create_minimizer(self) -> bool:
        """``CreateMinimizer``: Minuit2's MIGRAD on the likelihood, from the best fit, over every
        free parameter at its best-fit value and error - ``ROOT::Math::Minimizer``'s defaults."""
        nll = getattr(self._ratio, "nll", None)
        if nll is None:
            return False
        nll = nll()
        self._params = floating(self._ratio)
        for par in self._params:
            best = self._best.find(par.GetName()) if self._best is not None else None
            if best is not None:
                par.setVal(best.getVal())
                par.setError(best.getError())

        def fcn(x: Any) -> float:
            for par, value in zip(self._params, x):
                par.setVal(float(value))
            return float(nll.getVal())

        minuit = iminuit().Minuit(fcn, [p.getVal() for p in self._params],
                                  name=[p.GetName() for p in self._params])  # fmt: skip
        minuit.errordef = 1.0
        minuit.tol = float(default("Tolerance"))
        minuit.strategy = int(default("Strategy"))
        minuit.print_level = 0
        minuit.errors = [p.getError() for p in self._params]
        minuit.limits = [(p.getMin(), p.getMax()) for p in self._params]
        minuit.migrad(iterate=1, use_simplex=False)
        if not minuit.fmin.is_valid and not np.all(np.isfinite(minuit.values)):
            log_plain(self, ERROR, "Minimization", "Error: Minimization failed  \n")
            return False
        self._minuit = minuit
        return True

    def _minos(self, name: str, level: float) -> tuple[float, float]:
        """``GetMinosError`` at ``level``: nothing for an invalid minimum, as Minuit2 gives."""
        self._minuit.errordef = level
        try:
            self._minuit.minos(name)
        except RuntimeError:  # an invalid minimum: Minuit2Minimizer returns no errors at all
            return 0.0, 0.0
        found = self._minuit.merrors[name]
        return float(found.lower), float(found.upper)

    def GetContourPoints(self, paramX: Any, paramY: Any, x: Any, y: Any, npoints: int = 30) -> int:
        """``npoints`` points of the two-dimensional contour at the confidence level, into
        ``x`` and ``y``; how many were found."""
        names = [p.GetName() for p in floating(self._ratio)]
        if paramX.GetName() not in names or paramY.GetName() not in names:
            log(self, ERROR, "InputArguments", "LikelihoodInterval - Error - invalid parameters "
                f"specified for finding the contours; parX = {paramX.GetName()} parY = "
                f"{paramY.GetName()}")  # fmt: skip
            return 0
        if self._minuit is None and not self._create_minimizer():
            log(self, ERROR, "Eval", "LikelihoodInterval - Error returned creating minimizer for "
                "likelihood function - cannot find contour points ")  # fmt: skip
            return 0
        self._minuit.errordef = chisquare_quantile(self.ConfidenceLevel(), 2) / 2
        ix, iy = names.index(paramX.GetName()), names.index(paramY.GetName())
        log(self, INFO, "Minimization", f"LikelihoodInterval - Finding the contour of "
            f"{paramX.GetName()} ( {ix} ) and {paramY.GetName()} ( {iy} ) ")  # fmt: skip
        points = _contour(self._minuit, ix, iy, int(npoints))
        for index, (px, py) in enumerate(points[: int(npoints)]):
            x[index], y[index] = float(px), float(py)
        if len(points) < int(npoints):
            log(self, WARNING, "Minimization", "LikelihoodInterval -Warning - Less points "
                f"calculated in contours np = {len(points)} / {npoints}")  # fmt: skip
        return min(len(points), int(npoints))


def floating(ratio: Any) -> list[Any]:
    """The likelihood's free parameters, by name, as RooStats lists them for Minuit."""
    return [p for p in RooArgList(list(ratio.getVariables())) if not p.isConstant()]


def _contour(minuit: Any, ix: int, iy: int, npoints: int) -> list[Any]:
    """``Minuit2Minimizer::Contour``: MnContours at the minimum's own error level.

    iminuit's ``mncontour`` scales the level by a confidence it is given,
    which ``Contour`` does not, so MnContours is run here as ROOT runs it.
    """
    from iminuit._core import MnContours

    found = MnContours(minuit._fcn, minuit._fmin._src, minuit.strategy)(ix, iy, npoints)
    return list(found[2])
