"""``ROOT.Math``'s function objects, integrator and root finder, and ``ROOT.Fit.Fitter``.

A Python function becomes a ``Functor`` to hand to ROOT's numerical tools:
``Integrator`` integrates one - adaptively, as ROOT's default does - a
``RootFinder`` finds where one is zero, by Brent's method or, given its
derivative, Newton's, and ``Fit.Fitter.FitFCN`` minimises one with Minuit,
printing what ROOT prints for such a fit.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

__all__ = [
    "Functor1D",
    "Functor",
    "GradFunctor1D",
    "GradFunctor",
    "Integrator",
    "IntegratorOneDim",
    "RootFinder",
    "Fitter",
    "Fit",
]


class Functor1D:
    """``ROOT::Math::Functor1D``: a function of one number."""

    def __init__(self, fn: Callable[[float], float]) -> None:
        self._fn = fn

    def __call__(self, x: float) -> float:
        return float(self._fn(x))

    def NDim(self) -> int:
        return 1


class Functor:
    """``ROOT::Math::Functor``: a function of an array of ``ndim`` numbers."""

    def __init__(self, fn: Callable[[Any], float], ndim: int) -> None:
        self._fn, self._ndim = fn, int(ndim)

    def __call__(self, x: Any) -> float:
        return float(self._fn(np.asarray(x, dtype=np.float64)))

    def NDim(self) -> int:
        return self._ndim


class GradFunctor1D(Functor1D):
    """``ROOT::Math::GradFunctor1D``: a function of one number, and its derivative."""

    def __init__(self, fn: Callable[[float], float], derivative: Callable[[float], float]) -> None:
        super().__init__(fn)
        self._derivative = derivative

    def Derivative(self, x: float) -> float:
        return float(self._derivative(x))


class GradFunctor(Functor):
    """``ROOT::Math::GradFunctor``: a function of ``ndim`` numbers, and each partial derivative."""

    def __init__(
        self, fn: Callable[[Any], float], gradient: Callable[[Any, int], float], ndim: int
    ):
        super().__init__(fn, ndim)
        self._gradient = gradient

    def Derivative(self, x: Any, coordinate: int) -> float:
        return float(self._gradient(np.asarray(x, dtype=np.float64), int(coordinate)))


#: GSL's ``qk21``: the Kronrod abscissae, then the Gauss and the Kronrod weights.
XGK = (
    0.995657163025808080735527280689003,
    0.973906528517171720077964012084452,
    0.930157491355708226001207180059508,
    0.865063366688984510732096688423493,
    0.780817726586416897063717578345042,
    0.679409568299024406234327365114874,
    0.562757134668604683339000099272694,
    0.433395394129247190799265943165784,
    0.294392862701460198131126603103866,
    0.148874338981631210884826001129720,
    0.0,
)
WG = (
    0.066671344308688137593568809893332,
    0.149451349150580593145776339657697,
    0.219086362515982043995534934228163,
    0.269266719309996355091226921569469,
    0.295524224714752870173892994651338,
)
WGK = (0.011694638867371874278064396062192, 0.032558162307964727478818972459390,
       0.054755896574351996031381300244580, 0.075039674810919952767043140916190,
       0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
       0.123491976262065851077589750180300, 0.134709217311473325928054001771707,
       0.142775938577060080797094273138717, 0.147739104901338491374841515972068,
       0.149445554002916905664936468389821)  # fmt: skip


def kronrod(fn: Callable[[float], float], a: float, b: float) -> tuple[float, float]:
    """``gsl_integration_qk21``: the rule's integral over ``[a, b]``, and its error - the
    difference from the Gauss rule inside it - summed in GSL's order, so to GSL's bits."""
    center, half = 0.5 * (a + b), 0.5 * (b - a)
    kronrod_sum = float(fn(center)) * WGK[10]
    gauss_sum = 0.0
    for j in range(5):
        at = half * XGK[2 * j + 1]
        pair = float(fn(center - at)) + float(fn(center + at))
        gauss_sum += WG[j] * pair
        kronrod_sum += WGK[2 * j + 1] * pair
    for j in range(5):
        at = half * XGK[2 * j]
        kronrod_sum += WGK[2 * j] * (float(fn(center - at)) + float(fn(center + at)))
    return kronrod_sum * half, abs((kronrod_sum - gauss_sum) * half)


class Integrator:
    """``ROOT::Math::Integrator``: the integral of a function of one number over a range."""

    def __init__(self, *args: Any) -> None:
        self._fn: Any = next((arg for arg in args if callable(arg)), None)
        self._error = 0.0

    def SetFunction(self, fn: Any, *rest: Any) -> None:
        self._fn = fn

    def Integral(self, a: float, b: float) -> float:
        """``Integral(a, b)``: GSL's 21-point Gauss-Kronrod rule, as ROOT's default begins,
        and past it - when that rule alone is not good enough - an adaptive one."""
        found, error = kronrod(self._fn, float(a), float(b))
        if error > max(1e-9, 1e-9 * abs(found)):
            from ...function.function import _numerically

            found = float(_numerically(np.vectorize(self._fn), float(a), float(b), 1e-12))
        self._error = error
        return found

    def Error(self) -> float:
        return self._error

    def Status(self) -> int:
        return 0


IntegratorOneDim = Integrator


class RootFinder:
    """``ROOT::Math::RootFinder``: where a function of one number is zero."""

    (
        kBRENT,
        kGSL_BISECTION,
        kGSL_FALSE_POS,
        kGSL_BRENT,
        kGSL_NEWTON,
        kGSL_SECANT,
        kGSL_STEFFENSON,
    ) = range(7)

    def __init__(self, kind: int = 0) -> None:
        self._kind = int(kind)
        self._fn: Any = None
        self._bounds: tuple[float, ...] = ()
        self._root = math.nan

    def SetFunction(self, fn: Any, *bounds: float) -> bool:
        """``SetFunction(f, low, high)`` to bracket, or ``(f, start)`` for Newton's method."""
        self._fn, self._bounds = fn, tuple(float(value) for value in bounds)
        return True

    def Solve(self, maxIter: int = 100, absTol: float = 1e-8, relTol: float = 1e-10) -> bool:
        """``Solve``: Newton's from the start given, or bisection inside the bracket given."""
        if len(self._bounds) == 1:
            self._root = self._newton(self._bounds[0], maxIter)
        else:
            self._root = _bisected(self._fn, self._bounds[0], self._bounds[1])
        return not math.isnan(self._root)

    def _newton(self, x: float, iterations: int) -> float:
        for _ in range(int(iterations)):
            step = self._fn(x) / self._fn.Derivative(x)
            x -= step
            if abs(step) <= 1e-15 * max(abs(x), 1.0):
                break
        return float(x)

    def Root(self) -> float:
        return self._root


def _bisected(fn: Callable[[float], float], low: float, high: float) -> float:
    """Where ``fn`` changes sign between ``low`` and ``high``, to the last bit."""
    below = fn(low) < 0
    middle = 0.5 * (low + high)
    while middle not in (low, high):
        low, high = (middle, high) if (fn(middle) < 0) == below else (low, middle)
        middle = 0.5 * (low + high)
    return middle


class _FCNResult:
    """What ``Fitter.Result()`` hands back after ``FitFCN``: ``TFitResult``'s accessors."""

    def __init__(self, found: Any) -> None:
        from .fits import TFitResult

        self._result = TFitResult(found, "FitResult")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._result, name)

    def Print(self, stream: Any = None, covariance: bool = False) -> None:
        """``Print(std::cout)``: ROOT's summary of a minimisation - its ``MinFCN``, not a chi-square."""
        lines = self._result._xrd.summary(covariance=covariance).splitlines()
        print("\n".join(line for line in lines if not line.startswith("Chi2 ")))


class Fitter:
    """``ROOT::Fit::Fitter``: a function minimised - ``FitFCN`` - with Minuit's MIGRAD."""

    def __init__(self) -> None:
        self._result: Any = None

    def FitFCN(self, fcn: Any, params: Any = None, *rest: Any) -> bool:
        """``FitFCN(fcn, start)``: ``fcn`` minimised from ``start``; the result in ``Result()``."""
        start = np.asarray(params if params is not None else np.zeros(fcn.NDim()), dtype=np.float64)
        names = [f"Par_{index}" for index in range(len(start))]
        found = _minimised(fcn, start, names)
        found.chi2 = -1.0
        self._result = _FCNResult(found)
        return bool(found.valid)

    def Result(self) -> Any:
        return self._result


def _minimised(fcn: Any, start: Any, names: list[str]) -> Any:
    """MIGRAD on ``fcn`` from ``start`` - with its gradient, when it is a ``GradFunctor``."""
    from ...fit import minuit as core

    value = lambda p: float(fcn(p))  # noqa: E731 - Minuit is handed a plain function
    if not isinstance(fcn, GradFunctor):
        return core.minimize(value, start, names=names)
    grad = lambda p: [fcn.Derivative(p, at) for at in range(len(start))]  # noqa: E731
    made = core.iminuit().Minuit(value, start, grad=grad, name=tuple(names))
    made.errordef, made.tol, made.strategy, made.print_level = 1.0, core.TOLERANCE, core.STRATEGY, 0
    made.errors = core.default_steps(start)
    made.migrad(iterate=1, use_simplex=False)
    return core._result(made, names, False)


class _FitNamespace:
    """``ROOT.Fit``: the ``ROOT::Fit`` namespace."""

    Fitter = Fitter


#: ``ROOT.Fit``.
Fit = _FitNamespace()
