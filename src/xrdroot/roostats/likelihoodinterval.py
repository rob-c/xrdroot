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

import sys
from typing import Any

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
        #: The points MINOS asks for, while it runs.
        self._seen: list[Any] | None = None

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
        """Both ends, into ``lower`` and ``upper`` if they are cells - left as they were if the
        ends are not found; whether they were."""
        before = float(upper.value) if hasattr(upper, "value") else 0.0
        low, high, found = self._limits(param, before)
        for cell, value in ((lower, low), (upper, high)):
            if found and hasattr(cell, "value"):
                cell.value = value
        return found

    def _limits(self, param: Any, before: float = 0.0) -> tuple[float, float, bool]:
        """``FindLimits``: MINOS' ends, kept - or the parameter's own where MINOS gives none,
        said as ROOT says it (the upper end's message naming ``upper`` as it was, ``before``)."""
        name = param.GetName()
        if name in self._lower and name in self._upper:
            return self._lower[name], self._upper[name], True
        if not self._ready(name):
            return 0.0, 0.0, False
        level = chisquare_quantile(self.ConfidenceLevel(), 1) / 2
        low, high = self._minos(name, level)
        at = float(self._minuit.values[name])
        lower = at + low if low != 0 else self._at_limit(name, "lower", float(param.getMin()))
        upper = at + high if high != 0 else self._at_limit(name, "upper", before,
                                                           float(param.getMax()))  # fmt: skip
        self._lower[name], self._upper[name] = lower, upper
        return lower, upper, True

    def _ready(self, name: str) -> bool:
        """Whether the limits of ``name`` can be found: a free parameter, and a minimizer."""
        if name not in [p.GetName() for p in floating(self._ratio)]:
            log_plain(self, ERROR, "InputArguments", f"Error - invalid parameter {name} specified "
                      "for finding the interval limits \n")  # fmt: skip
            return False
        if self._minuit is None and not self._create_minimizer():
            log_plain(self, ERROR, "Eval", "Error returned from minimization of likelihood "
                      "function - cannot find interval limits \n")  # fmt: skip
            return False
        return True

    def _at_limit(self, name: str, side: str, said: float, value: Any = None) -> float:
        """MINOS found no end: the parameter's own, said - the upper as it was before."""
        log_plain(self, WARNING, "Minimization", f"Warning: {side} value for {name} is at limit "
                  f"{g(said)}\n")  # fmt: skip
        return said if value is None else float(value)

    def _from_best(self) -> None:
        """Each free parameter at its best-fit value and error."""
        for par in self._params:
            best = self._best.find(par.GetName()) if self._best is not None else None
            if best is not None:
                par.setVal(best.getVal())
                par.setError(best.getError())

    def _create_minimizer(self) -> bool:
        """``CreateMinimizer``: Minuit2's MIGRAD on the likelihood, from the best fit, over every
        free parameter at its best-fit value and error - ``ROOT::Math::Minimizer``'s defaults."""
        nll = getattr(self._ratio, "nll", None)
        if nll is None:
            return False
        nll = nll()
        self._params = floating(self._ratio)
        self._from_best()

        def fcn(x: Any) -> float:
            for par, value in zip(self._params, x, strict=False):
                par.setVal(float(value))
            if self._seen is not None:
                self._seen.append(tuple(float(v) for v in x))
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
        self._minuit = minuit  # kept, as ROOT keeps its minimizer, even if its minimum is not
        if not minuit.fmin.is_valid:
            log_plain(self, ERROR, "Minimization", "Error: Minimization failed  \n")
            return False
        return True

    def _minos(self, name: str, level: float) -> tuple[float, float]:
        """``GetMinosError`` at ``level``: nothing for an invalid minimum, as Minuit2 gives."""
        if not self._minuit.fmin.is_valid:  # Minuit2Minimizer returns no errors at all
            sys.stderr.write("Error in <Minuit2>: Minuit2Minimizer::GetMinosError Failed - "
                             "invalid function minimum\n")  # fmt: skip
            return 0.0, 0.0
        self._minuit.errordef = level
        self._seen = []
        self._minuit.minos(name)
        seen, self._seen = self._seen, None
        self._left_as_root_leaves(name, seen)
        found = self._minuit.merrors[name]
        return float(found.lower), float(found.upper)

    def _left_as_root_leaves(self, name: str, seen: list[Any]) -> None:
        """The parameters where ROOT's MINOS leaves them: at the last point of the upper search.

        ``GetMinosError`` runs the lower search and then the upper one, where
        iminuit's MINOS runs the upper first; each is a search of its own from
        the minimum, the same either way, but the likelihood's parameters are
        left where the last call put them - so they are put back there, the
        last point beyond the minimum.
        """
        index = [p.GetName() for p in self._params].index(name)
        at = float(self._minuit.values[index])
        above = [point for point in seen if point[index] > at]
        for par, value in zip(self._params, above[-1] if above else (), strict=False):
            par.setVal(value)

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
        if points is None:
            log(self, ERROR, "Minimization", "LikelihoodInterval - Error finding contour for "
                f"parameters {paramX.GetName()} and {paramY.GetName()}")  # fmt: skip
            return 0
        for index, (px, py) in enumerate(points):
            x[index], y[index] = float(px), float(py)
        return len(points)


def floating(ratio: Any) -> list[Any]:
    """The likelihood's free parameters, by name, as RooStats lists them for Minuit."""
    return [p for p in RooArgList(list(ratio.getVariables())) if not p.isConstant()]


def _contour(minuit: Any, ix: int, iy: int, npoints: int) -> list[Any] | None:
    """``Minuit2Minimizer::Contour``: MnContours at the minimum's own error level - ``None``,
    with Minuit2's error, for an invalid minimum or any number of points but ``npoints``.

    iminuit's ``mncontour`` scales the level by a confidence it is given,
    which ``Contour`` does not, so MnContours is run here as ROOT runs it.
    """
    from iminuit._core import MnContours

    if not minuit.fmin.is_valid:
        sys.stderr.write("Error in <Minuit2>: Minuit2Minimizer::Contour Invalid function "
                         "minimum\n")  # fmt: skip
        return None
    found = MnContours(minuit._fcn, minuit._fmin._src, minuit.strategy)(ix, iy, npoints)
    if len(found[2]) != npoints:
        sys.stderr.write("Error in <Minuit2>: Minuit2Minimizer::Contour Invalid result from "
                         "MnContours\n")  # fmt: skip
        return None
    return list(found[2])
