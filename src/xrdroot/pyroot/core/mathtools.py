"""``ROOT.Math``'s function objects, integrator and root finder, and ``ROOT.Fit.Fitter``.

A Python function becomes a ``Functor`` to hand to ROOT's numerical tools:
``Integrator`` integrates one - adaptively, as ROOT's default does - a
``RootFinder`` finds where one is zero, by Brent's method or, given its
derivative, Newton's, and ``Fit.Fitter.FitFCN`` minimises one with Minuit,
printing what ROOT prints for such a fit.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from ...fit.defaults import DEFAULTS
from ...tmva import genetic as tmva_genetic
from . import fitdata, messages
from .funcbase import (
    IBaseFunctionMultiDim,
    IBaseFunctionOneDim,
    IGradientFunctionMultiDim,
    IGradientFunctionOneDim,
)

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
    "Minimizer",
    "GeneticMinimizer",
    "GeneticMinimizerParameters",
    "Factory",
]


class Functor1D(IBaseFunctionOneDim):
    """``ROOT::Math::Functor1D``: a function of one number.

    ``Derivative`` answers zero, as ROOT's own ``Functor1D`` does - a functor wraps
    only what its caller could write, a value or that and a derivative.
    """

    def __init__(self, fn: Callable[[float], float]) -> None:
        self._fn = fn

    def __call__(self, x: float) -> float:
        return float(self._fn(x))

    def Clone(self) -> Any:
        """``Clone()``: another functor over the same Python function."""
        return type(self)(self._fn)


class Functor(IBaseFunctionMultiDim):
    """``ROOT::Math::Functor``: a function of an array of ``ndim`` numbers."""

    def __init__(self, fn: Callable[[Any], float], ndim: int) -> None:
        self._fn, self._ndim = fn, int(ndim)

    def __call__(self, x: Any) -> float:
        return float(self._fn(np.asarray(x, dtype=np.float64)))

    def DoEval(self, x: Any) -> float:
        """``DoEval(x)``: the Python function at the point, with ``x`` as an array."""
        return float(self._fn(np.asarray(x, dtype=np.float64)))

    def NDim(self) -> int:
        return self._ndim

    def Clone(self) -> Any:
        """``Clone()``: another functor over the same Python function."""
        return type(self)(self._fn, self._ndim)


class GradFunctor1D(Functor1D, IGradientFunctionOneDim):
    """``ROOT::Math::GradFunctor1D``: a function of one number, and its derivative."""

    def __init__(self, fn: Callable[[float], float], derivative: Callable[[float], float]) -> None:
        super().__init__(fn)
        self._derivative = derivative

    def Derivative(self, x: float) -> float:
        return float(self._derivative(x))

    def DoDerivative(self, x: Any) -> float:
        """``DoDerivative(x)``: the derivative the caller gave, as ``Derivative`` is."""
        return float(self._derivative(float(x)))

    def Clone(self) -> Any:
        """``Clone()``: another functor over the same function and its derivative."""
        return type(self)(self._fn, self._derivative)

    def HasGradient(self) -> bool:
        """``HasGradient()``: yes, the derivative was given."""
        return True


class GradFunctor(Functor, IGradientFunctionMultiDim):
    """``ROOT::Math::GradFunctor``: a function of ``ndim`` numbers, and each partial derivative."""

    def __init__(
        self, fn: Callable[[Any], float], gradient: Callable[[Any, int], float], ndim: int
    ):
        super().__init__(fn, ndim)
        self._gradient = gradient

    def Derivative(self, x: Any, icoord: int = 0) -> float:
        return float(self._gradient(np.asarray(x, dtype=np.float64), int(icoord)))

    def DoDerivative(self, x: Any, icoord: int) -> float:
        """``DoDerivative(x, icoord)``: the partial derivative the caller gave."""
        return float(self._gradient(np.asarray(x, dtype=np.float64), int(icoord)))

    def Clone(self) -> Any:
        """``Clone()``: another functor over the same function and gradient."""
        return type(self)(self._fn, self._gradient, self._ndim)

    def HasGradient(self) -> bool:
        """``HasGradient()``: yes, the gradient was given."""
        return True


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
        """``Print(std::cout)``: ROOT's summary of a minimisation - ``MinFCN``, and no
        chi-square, unless the function was said to be one."""
        lines = self._result._xrd.summary(covariance=covariance).splitlines()
        chi2 = self._result._xrd.chi2 >= 0
        print("\n".join(line for line in lines if chi2 or not line.startswith("Chi2 ")))


class ParameterSettings:
    """``ROOT::Fit::ParameterSettings``: one parameter's start, step, limits and whether fixed."""

    def __init__(self, name: str, value: float, step: float) -> None:
        self._name, self._value, self._step = name, float(value), float(step)
        self._limits: tuple[float, float] | None = None
        self._fixed = False

    def SetValue(self, value: float) -> None:
        self._value = float(value)

    def Value(self) -> float:
        return self._value

    def SetStepSize(self, step: float) -> None:
        self._step = float(step)

    def StepSize(self) -> float:
        return self._step

    def SetName(self, name: Any) -> None:
        self._name = str(name)

    def Name(self) -> str:
        return self._name

    def SetLimits(self, low: float, high: float) -> None:
        self._limits = (float(low), float(high))

    def Fix(self) -> None:
        self._fixed = True

    def Release(self) -> None:
        self._fixed = False

    def IsFixed(self) -> bool:
        return self._fixed


class FitConfig:
    """``ROOT::Fit::FitConfig``: the settings of every parameter ``FitFCN`` starts from."""

    def __init__(self) -> None:
        self._settings: list[ParameterSettings] = []

    def SetParamsSettings(self, npar: int, params: Any = None, errors: Any = None) -> None:
        """``SetParamsSettings``: ``Par_i`` from each value, stepped 30% of it or by ``errors``."""
        values = np.zeros(int(npar)) if params is None else np.asarray(params, np.float64)
        from ...fit.minuit import default_steps

        steps = default_steps(values[: int(npar)]) if errors is None else np.asarray(errors)
        self._settings = [ParameterSettings(f"Par_{i}", values[i], steps[i])
                          for i in range(int(npar))]  # fmt: skip

    def ParSettings(self, i: int) -> ParameterSettings:
        return self._settings[int(i)]

    def ParamsSettings(self) -> list[ParameterSettings]:
        return self._settings

    def NPar(self) -> int:
        return len(self._settings)

    def MinimizerOptions(self) -> Any:
        """``MinimizerOptions()``: the minimiser's settings - MIGRAD's, at ROOT's defaults."""
        found = self.__dict__.setdefault("_options", _MinimizerSettings())
        return found

    def SetMinimizer(self, name: Any, algorithm: Any = None) -> None:
        """``SetMinimizer("Minuit2", "Migrad")``: what is minimised with - always that, here."""
        self.__dict__["_minimizer"] = (str(name), str(algorithm or "Migrad"))


class _MinimizerSettings:
    """``ROOT::Math::MinimizerOptions`` of one fit: what is set, kept and read back."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {"PrintLevel": 0}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("Set"):
            return lambda value: self._values.__setitem__(name[3:], value)
        if name in self._values:
            return lambda: self._values[name]
        raise AttributeError(f"ROOT's MinimizerOptions has {name}; this one does not yet")


class Fitter:
    """``ROOT::Fit::Fitter``: a function minimised - ``FitFCN`` - with Minuit's MIGRAD."""

    def __init__(self) -> None:
        self._result: Any = None
        self._fcn: Any = None
        self._config = FitConfig()

    def Config(self) -> FitConfig:
        return self._config

    def SetFCN(self, fcn: Any, params: Any = None, *rest: Any) -> bool:
        """``SetFCN(fcn, start)``: what ``FitFCN()`` minimises, and from where."""
        self._fcn = fcn
        self._config.SetParamsSettings(fcn.NDim(), params)
        return True

    def FitFCN(self, fcn: Any = None, params: Any = None, *rest: Any) -> bool:
        """``FitFCN(fcn, start, dataSize, chi2)``: ``fcn`` minimised from ``start``; the result
        in ``Result()``. ``FitFCN(npar, fcn, ...)`` names how many parameters it takes, set up
        in ``Config()`` unless ``start`` is given. Said to be a chi-square, its minimum is the
        fit's, of ``dataSize`` less the free parameters' degrees of freedom."""
        npar, fcn, start, rest = _fcn_arguments(fcn, params, rest)
        if fcn is not None:
            self._take(fcn, npar, start)
        if self._fcn is None:
            raise ValueError("FitFCN() minimises the function SetFCN gave it, and none was given")
        found = _minimised(self._fcn, self._config.ParamsSettings())
        _counted(found, self._config.ParamsSettings(), rest)
        self._result = _FCNResult(found)
        return bool(found.valid)

    def SetFunction(self, function: Any, useGradient: bool = False) -> bool:
        """``SetFunction(wrapped)``: the model ``Fit`` fits, its parameters set up from the
        function's - named, valued and stepped as they stand."""
        self.__dict__["_model"] = function
        tf = function._function
        values = [tf.GetParameter(i) for i in range(tf.GetNpar())]
        self._config.SetParamsSettings(len(values), np.array(values, dtype=np.float64))
        for i, one in enumerate(self._config.ParamsSettings()):
            one.SetName(tf.GetParName(i))
        return True

    def Fit(self, data: Any, *rest: Any) -> bool:
        """``Fit(data)``: the model's chi-square over the data minimised."""
        chi2 = fitdata.Chi2Function(data, self.__dict__["_model"])
        return self.FitFCN(chi2.NDim(), chi2, None, data.Size(), True)

    def _take(self, fcn: Any, npar: int | None, start: Any) -> None:
        """The function to minimise, its parameters set up from ``start`` - or kept as
        ``Config()`` has them, if they are as many."""
        self._fcn = fcn
        count = npar if npar is not None else int(fcn.NDim())
        if start is not None or self._config.NPar() != count:
            self._config.SetParamsSettings(count, start)

    def Result(self) -> Any:
        return self._result


def _counted(found: Any, settings: list[ParameterSettings], rest: tuple[Any, ...]) -> None:
    """The chi-square - the minimum, if the function was said to be one - and the degrees of
    freedom of ``dataSize`` less the free parameters, none for no data, as ``FitResult`` has."""
    size, is_chi2 = (int(rest[0]) if rest else 0), bool(rest[1:2] and rest[1])
    found.chi2 = found.fcn if is_chi2 else -1.0
    free = sum(1 for one in settings if not one.IsFixed())
    found.ndf = size - free if size > free else 0


def _fcn_arguments(fcn: Any, params: Any, rest: tuple[Any, ...]) -> tuple[Any, ...]:
    """``FitFCN``'s overloads: how many parameters if said, the function, where it starts
    - ``nullptr``, ``0``, none - and what follows."""
    npar = None
    if isinstance(fcn, (int, np.integer)):
        npar, fcn, params, rest = int(fcn), params, (rest[0] if rest else None), rest[1:]
    start = None if params is None or np.ndim(params) == 0 else params
    return npar, fcn, start, rest


def _minimised(fcn: Any, settings: list[ParameterSettings]) -> Any:
    """MIGRAD on ``fcn`` from the settings - with its gradient, when it is a ``GradFunctor``."""
    from ...fit import minuit as core

    start = np.array([one.Value() for one in settings])
    names = [one.Name() for one in settings]
    steps = np.array([one.StepSize() for one in settings])
    limits = [one._limits for one in settings]
    fixed = [one.IsFixed() for one in settings]
    if gradient(fcn, len(start)) is None:
        return core.minimize(lambda p: float(fcn(p)), start, names=names, errors=steps,
                             limits=limits, fixed=fixed)  # fmt: skip
    made = _minuit(fcn, start, names)
    made.tol, made.strategy = core.TOLERANCE, core.STRATEGY
    made.errors = steps
    made.migrad(iterate=1, use_simplex=False)
    return core._result(made, names, False, int(made.nfcn))


class _FitNamespace:
    """``ROOT.Fit``: the ``ROOT::Fit`` namespace."""

    Fitter = Fitter
    FitConfig = FitConfig
    ParameterSettings = ParameterSettings
    BinData = fitdata.BinData
    Chi2Function = fitdata.Chi2Function
    DataOptions = fitdata.DataOptions
    DataRange = fitdata.DataRange
    FillData = staticmethod(fitdata.FillData)


#: ``ROOT.Fit``.
Fit = _FitNamespace()


class Minimizer:
    """``ROOT::Math::Minimizer``, as ``Factory::CreateMinimizer("Minuit2")`` makes one."""

    def __init__(self, name: str = "Minuit2", algorithm: str = "") -> None:
        self._name, self._algorithm = str(name or "Minuit2"), str(algorithm or "Migrad")
        self._settings = {"calls": 0, "iterations": 0, "tolerance": 0.01, "print": 0,
                          "strategy": 1}  # fmt: skip
        self._fcn: Any = None
        self._variables: dict[int, list[Any]] = {}
        self._found: Any = None

    def SetMaxFunctionCalls(self, calls: int) -> None:
        self._settings["calls"] = int(calls)

    def SetMaxIterations(self, iterations: int) -> None:
        """``SetMaxIterations``: for GSL's and the genetic minimisers - Minuit counts calls.

        A count that is not positive is no limit, so it is not set, as ROOT's own
        ``Minimizer::SetMaxIterations`` ignores it.
        """
        if int(iterations) > 0:
            self._settings["iterations"] = int(iterations)

    def SetTolerance(self, tolerance: float) -> None:
        self._settings["tolerance"] = float(tolerance)

    def SetPrintLevel(self, level: int) -> None:
        self._settings["print"] = int(level)

    def SetStrategy(self, strategy: int) -> None:
        self._settings["strategy"] = int(strategy)

    def MaxFunctionCalls(self) -> int:
        """``MaxFunctionCalls()``: the limit ``SetMaxFunctionCalls`` was given."""
        return int(self._settings["calls"])

    def MaxIterations(self) -> int:
        """``MaxIterations()``: the limit ``SetMaxIterations`` was given."""
        return int(self._settings["iterations"])

    def Tolerance(self) -> float:
        """``Tolerance()``: the convergence tolerance last set."""
        return float(self._settings["tolerance"])

    def PrintLevel(self) -> int:
        """``PrintLevel()``: how much the minimisation is to say, zero by default."""
        return int(self._settings["print"])

    def Strategy(self) -> int:
        """``Strategy()``: Minuit's strategy, which the genetic search has no use for."""
        return int(self._settings["strategy"])

    def SetFunction(self, fcn: Any) -> None:
        self._fcn = fcn

    def Clear(self) -> None:
        """``Clear()``: forget the variables and what was found, so that the same minimiser
        can be set up and run again - what a test program does between two functions."""
        self._variables, self._found = {}, None

    def GetMinosError(self, ivar: int, el: Any, eu: Any) -> bool:
        """``GetMinosError(ivar, el, eu)``: MINOS's two errors on one parameter, written into
        the two references given, and whether Minuit had them to give.

        MINOS needs a minimum it trusts, so a minimisation that ended anywhere else - or
        one that has not run - answers false rather than an error read off nothing.
        """
        if self._found is None:
            return False
        name = self._variables[sorted(self._variables)[int(ivar)]][0]
        try:
            self._found.minos(name)
            error = self._found.merrors[name]
        except (RuntimeError, KeyError):  # no minimum MINOS trusts, or no error from it
            return False
        el.value, eu.value = float(error.lower), float(error.upper)
        return True

    def SetVariable(self, index: int, name: Any, value: float, step: float) -> bool:
        self._variables[int(index)] = [str(name), float(value), float(step), None]
        return True

    def SetLimitedVariable(self, index: int, name: Any, value: float, step: float,
                           low: float, high: float) -> bool:  # fmt: skip
        self._variables[int(index)] = [str(name), float(value), float(step), (low, high)]
        return True

    def Minimize(self) -> bool:
        """``Minimize``: MIGRAD from the variables set, and - at print level 1 - ROOT's lines."""
        order = sorted(self._variables)
        names = [self._variables[at][0] for at in order]
        made = _minuit(self._fcn, [self._variables[at][1] for at in order], names)
        made.errors = [self._variables[at][2] for at in order]
        for place, at in enumerate(order):
            bounds = self._variables[at][3]
            made.limits[place] = bounds if bounds is not None else (-np.inf, np.inf)
        made.tol, made.strategy = self._settings["tolerance"], self._settings["strategy"]
        made.migrad(ncall=self._settings["calls"] or None, iterate=1, use_simplex=False)
        self._found = made
        if self._settings["print"] > 0:
            self._report(names)
        return bool(made.valid)

    def _report(self, names: list[str]) -> None:
        found, settings = self._found, self._settings
        print(f"{self._name}Minimizer: Minimize with max-calls {settings['calls']} convergence for "
              f"edm < {settings['tolerance']:g} strategy {settings['strategy']}")  # fmt: skip
        verdict = "Valid minimum" if found.valid else "Invalid minimum"
        print(f"{self._name}Minimizer : {verdict} - status = {self.Status()}")
        print(f"FVAL  = {found.fval:.18g}\nEdm   = {found.fmin.edm:.18g}\nNfcn  = {found.nfcn}")
        for name, value, error in zip(names, found.values, found.errors, strict=False):
            print(f"{name}\t  = {value:g}\t +/-  {error:g}")

    def X(self) -> np.ndarray[Any, Any]:
        return np.array(self._found.values)

    def Errors(self) -> np.ndarray[Any, Any]:
        return np.array(self._found.errors)

    def MinValue(self) -> float:
        return float(self._found.fval)

    def Edm(self) -> float:
        return float(self._found.fmin.edm)

    def NCalls(self) -> int:
        return int(self._found.nfcn)

    def NDim(self) -> int:
        return len(self._variables)

    def Status(self) -> int:
        return 0 if self._found.valid else 3


#: ``GeneticMinimizerParameters``' defaults, as ROOT's own struct has them: the search
#: runs at most this many generations of this many points, and narrows its step by
#: ``fSC_factor`` after ``fSC_rate`` good generations of the last ``fSC_steps``.
_GENETIC_STEPS = 100
_GENETIC_POPULATION = 300
_GENETIC_CYCLES = 3
_GENETIC_SC_STEPS = 10
_GENETIC_SC_RATE = 5
_GENETIC_SC_FACTOR = 0.95


class _GeneticTarget(tmva_genetic.IFitterTarget):
    """``TMVA::IFitterTarget`` over a ``ROOT::Math::IMultiGenFunction``: the function at the
    coordinates TMVA's search draws, asked of the minimiser that holds it."""

    def __init__(self, minimizer: Any) -> None:
        self.minimizer = minimizer

    def EstimatorFunction(self, parameters: Any) -> float:
        """``EstimatorFunction``: the function at one drawn point of the population."""
        return float(self.minimizer._evaluate(parameters))


class GeneticMinimizerParameters:
    """``ROOT::Math::GeneticMinimizerParameters``: how the genetic search is run.

    The fields are ROOT's, ``f``-prefixed as its struct spells them, and so are the
    defaults: a hundred generations of three hundred points, a step narrowed by
    ``0.95`` after five good generations of the last ten, and a seed of zero - which is
    ROOT's way of saying "draw one", so a search is repeatable only once a seed is set.
    """

    def __init__(self) -> None:
        steps = int(DEFAULTS["MaxIterations"])
        self.fNsteps = steps if steps > 0 else _GENETIC_STEPS
        self.fPopSize = _GENETIC_POPULATION
        self.fCycles = _GENETIC_CYCLES
        self.fSC_steps = _GENETIC_SC_STEPS
        self.fSC_rate = _GENETIC_SC_RATE
        self.fSC_factor = _GENETIC_SC_FACTOR
        self.fConvCrit = 10.0 * float(DEFAULTS["Tolerance"])
        if self.fConvCrit <= 0:
            self.fConvCrit = 0.001
        self.fSeed = 0


class GeneticMinimizer(Minimizer):
    """``ROOT::Math::GeneticMinimizer``: TMVA's genetic algorithm, as a minimiser.

    It is for a function whose derivative is noise, or one with several minima that a
    gradient search would miss: a population spread over the ranges is bred - children
    combine two parents' coordinates, and a share of them is mutated - so nothing but
    the function's value is asked of it. A coordinate needs a range, which is why
    ``SetVariable`` invents one fifty steps either side of the value.

    What a genetic search cannot give is an error: ``MinValue`` and ``X`` are the least
    point found, ``Edm`` and ``CovMatrix`` are zero, and ``Errors`` is empty. Give it a
    print level above zero to have the generations reported, as ROOT reports them.
    """

    def __init__(self, i: int = 0) -> None:
        super().__init__("Genetic", "")
        self._parameters = GeneticMinimizerParameters()
        self._ranges: dict[int, tuple[float, float]] = {}
        self._fixed: dict[int, float] = {}
        self._result: list[float] = []
        self._ndim = 0
        self._best = math.inf
        self._calls = 0
        self._status = 0
        self.SetTolerance(0.1 * self._parameters.fConvCrit)
        self.SetMaxIterations(self._parameters.fNsteps)

    # -- what is minimised -------------------------------------------------------------

    def Clear(self) -> None:
        """``Clear()``: forget the function, its coordinates and what was found, as ROOT's
        ``Clear`` does - ``SetFunction`` starts from here, so set the variables after it."""
        self._fcn = None
        self._ranges, self._fixed, self._result = {}, {}, []
        self._ndim, self._best, self._calls, self._status = 0, math.inf, 0, 0

    def SetFunction(self, fcn: Any) -> None:
        """``SetFunction(func)``: the function of the parameters, whose ``NDim`` is the
        number of coordinates the search works in."""
        self.Clear()
        self._fcn, self._ndim = fcn, int(fcn.NDim())
        self._result = [0.0] * self._ndim

    def SetLimitedVariable(self, index: int, name: Any, value: float, step: float,
                           low: float, high: float) -> bool:  # fmt: skip
        """``SetLimitedVariable``: a coordinate and where it may go, which is what the
        search needs and all it keeps of the name and the step."""
        self._ranges[int(index)] = (float(low), float(high))
        return True

    def SetVariable(self, index: int, name: Any, value: float, step: float) -> bool:
        """``SetVariable``: a coordinate with no limits, given the range fifty steps either
        side of the value - a genetic search has no other way to bound one - and saying
        so, as ROOT's ``Info`` does."""
        low, high = float(value) - 50.0 * float(step), float(value) + 50.0 * float(step)
        messages.Info("GeneticMinimizer::SetVariable",
             "Variables should be limited - set automatic range to 50 times step size "
             "for %s : [%f, %f]", name, low, high)  # fmt: skip
        self._ranges[int(index)] = (low, high)
        return True

    def SetFixedVariable(self, index: int, name: Any, value: float) -> bool:
        """``SetFixedVariable``: a coordinate held at ``value`` while the others are searched.

        The function is given the value at every point the search tries rather than the
        coordinate being drawn, so ``NFree`` counts what is left to search over.
        """
        self._fixed[int(index)] = float(value)
        return True

    def SetParameters(self, parameters: GeneticMinimizerParameters) -> None:
        """``SetParameters(params)``: the search's own settings, with the tolerance and the
        iteration count of the base class kept in step with them, as ROOT keeps them."""
        self._parameters = parameters
        self.SetTolerance(0.1 * parameters.fConvCrit)
        self.SetMaxIterations(parameters.fNsteps)

    def SetRandomSeed(self, seed: int) -> None:
        """``SetRandomSeed(seed)``: where the population's draws come from, so that a
        search can be repeated."""
        self._parameters.fSeed = int(seed)

    def MinimizerParameters(self) -> GeneticMinimizerParameters:
        """``MinimizerParameters()``: the settings the search runs with."""
        return self._parameters

    # -- the search ---------------------------------------------------------------------

    def Minimize(self) -> bool:
        """``Minimize()``: the genetic search, and ``True`` once it has run.

        The generations stop when the least value has moved by no more than
        ``fConvCrit`` in each of ``fNsteps`` generations running, or when
        ``MaxIterations`` of them have been bred - which is ``Status()`` 1, as ROOT's
        is. The value is the best point's own, so nothing is evaluated twice.
        """
        if self._fcn is None:
            messages.Error("GeneticMinimizer::Minimize", "Fitness function has not been set")
            return False
        settings = self._synced()
        ranges = [tmva_genetic.Interval(low, high) for low, high in self._ranges_in_order()]
        found = tmva_genetic.GeneticAlgorithm(_GeneticTarget(self), settings.fPopSize, ranges,
                                              settings.fSeed)  # fmt: skip
        if self.PrintLevel() > 0:
            print("GeneticMinimizer::Minimize  - Start iterating - max iterations = "
                  f"{settings.fNsteps} conv criteria (tolerance) =  {settings.fConvCrit}")
        self._status, iterations = self._generations(found, settings)
        self._reported(found, iterations)
        return True

    def _synced(self) -> GeneticMinimizerParameters:
        """The settings the search runs with: ``Minimize`` takes the generation count and
        the tolerance from the base class, as ROOT's does, when they were set."""
        settings = self._parameters
        if self.MaxIterations() > 0:
            settings.fNsteps = self.MaxIterations()
        if self.Tolerance() > 0:
            settings.fConvCrit = 10.0 * self.Tolerance()
        return settings

    def _generations(self, found: Any, settings: GeneticMinimizerParameters) -> tuple[int, int]:
        """How the search ended - ``Status()`` and the generations bred - once it settled
        or ran out of the generations ROOT allows itself."""
        iterations = 0
        while True:
            found.Init()
            found.CalculateFitness()
            found.GetGeneticPopulation().TrimPopulation()
            found.SpreadControl(settings.fSC_steps, settings.fSC_rate, settings.fSC_factor)
            iterations += 1
            if 0 < settings.fNsteps < iterations:
                if self.PrintLevel() > 0:
                    messages.Info("GeneticMinimizer::Minimize",
                                  "Max number of iterations %d reached - stop iterating",
                                  settings.fNsteps)  # fmt: skip
                return 1, iterations
            if found.HasConverged(settings.fNsteps, settings.fConvCrit):
                return 0, iterations

    def _reported(self, found: Any, iterations: int) -> None:
        """Keep - and at a print level above zero, say - what the search found."""
        genes = found.GetGeneticPopulation().GetGenes(0)
        self._best = genes.fitness
        self._result = self._point(genes.factors)
        if self.PrintLevel() > 0:
            print("Finished Iteration (niter = " + str(iterations) +
                  "  with fitness function value = " + str(self._best) + ")")
            for at, value in enumerate(self._result):
                print(f" Parameter-{at}\t=\t{value}")

    def _ranges_in_order(self) -> list[tuple[float, float]]:
        """The ranges of the coordinates the search may move, in the order of their
        indices, which is the order the function's own coordinates are in."""
        return [self._ranges[at] for at in sorted(self._ranges) if at not in self._fixed]

    def _point(self, factors: Any) -> list[float]:
        """The function's coordinates: the ones drawn, with the fixed values filled in."""
        if not self._fixed:
            return list(factors)
        free = iter(factors)
        return [self._fixed[at] if at in self._fixed else next(free) for at in range(self._ndim)]

    def _evaluate(self, factors: Any) -> float:
        """The function at one point of the population, as the search asks for it."""
        self._calls += 1
        return float(self._fcn(self._point(factors)))

    # -- what was found -----------------------------------------------------------------

    def MinValue(self) -> float:
        """``MinValue()``: the least value the search met, zero for a search not run."""
        return 0.0 if self._fcn is None else float(self._best)

    def X(self) -> np.ndarray[Any, Any]:
        """``X()``: the coordinates of the least point found."""
        return np.array(self._result, dtype=np.float64)

    def NCalls(self) -> int:
        """``NCalls()``: how many times the function was evaluated."""
        return self._calls

    def NDim(self) -> int:
        """``NDim()``: the coordinates the function takes, the fixed ones included."""
        return self._ndim

    def NFree(self) -> int:
        """``NFree()``: how many of them the search may move."""
        return len(self._ranges_in_order())

    def MinGradient(self) -> list[float]:
        """``MinGradient()``: nothing, the search never looking at a gradient."""
        return []

    def ProvidesError(self) -> bool:
        """``ProvidesError()``: no - a genetic search has no covariance to give."""
        return False

    def Errors(self) -> list[float]:
        """``Errors()``: none, as ROOT's ``Errors`` is a null pointer here."""
        return []

    def Edm(self) -> float:
        """``Edm()``: zero, the estimated distance to the minimum being Minuit's own."""
        return 0.0

    def CovMatrix(self, i: int, j: int) -> float:
        """``CovMatrix(i, j)``: zero, there being no covariance to give."""
        return 0.0

    def Status(self) -> int:
        """``Status()``: one, a search that ran to the generations ROOT allows itself.

        ROOT asks the same ``MaxIterations`` of the base class for both the count it
        stops at and the count of quiet generations it wants, and stops on the first
        before asking about the second, so ROOT's own search reports 1 as well.
        """
        return self._status


def _minuit(fcn: Any, start: Any, names: list[str]) -> Any:
    """iminuit's Minuit over ``fcn``, with its gradient when it can say one."""
    from ...fit import minuit as core

    value = lambda p: float(fcn(p))  # noqa: E731 - Minuit is handed a plain function
    made = core.iminuit().Minuit(value, np.asarray(start, dtype=np.float64), grad=gradient(fcn),
                                 name=tuple(names))  # fmt: skip
    made.errordef, made.print_level = 1.0, 0
    return made


def gradient(fcn: Any, ndim: int = 0) -> Any:
    """The gradient of ``fcn``, or ``None`` when it has none: ROOT's tools ask a
    function whether it can say its derivatives - ``HasGradient`` - and difference
    it themselves when it cannot."""
    if not getattr(fcn, "HasGradient", _no_gradient)():
        return None
    return lambda p: [float(fcn.Derivative(p, at)) for at in range(ndim or fcn.NDim())]


def _no_gradient() -> bool:
    """A plain Python callable's answer to ``HasGradient()``: it has no gradient."""
    return False


class Factory:
    """``ROOT::Math::Factory``: minimisers by name - Minuit2's, or the genetic search."""

    @staticmethod
    def CreateMinimizer(name: Any = "Minuit2", algorithm: Any = "") -> Minimizer:
        """``CreateMinimizer(type, algorithm)``: a ``Minimizer``, the genetic search being
        the one kind whose algorithm is its own."""
        if str(name) == "Genetic":
            return GeneticMinimizer()
        return Minimizer(str(name), str(algorithm))
