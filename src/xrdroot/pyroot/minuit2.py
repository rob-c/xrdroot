"""``ROOT::Minuit2``: Minuit2's own interface - ``FCNBase``, ``MnMigrad``, ``MnMinos`` and the rest.

A program written against Minuit2 directly subclasses ``FCNBase`` (or
``FCNGradientBase``, which adds ``Gradient``), describes its parameters in
an ``MnUserParameters``, and runs an application - ``MnMigrad``,
``MnSimplex``, ``MnMinimize``, ``MnScan`` - whose call gives a
``FunctionMinimum``; ``MnHesse``, ``MnMinos`` and ``MnContours`` work from
that minimum. iminuit binds the very same C++ classes, so each of these
drives Minuit2's own algorithm and gives ROOT's numbers; what is here is
ROOT's names and calling conventions over them.

Minuit2 calls the function with a ``std::vector<double>``, so the
function given here is called with one too; an application's state after
a call is the minimum's, as ``MnApplication`` keeps it, so calling it
again starts from there.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..cppstd.containers import std
from ..cppstd.objects import Pair
from ..errors import UnsupportedFeatureError
from ..fit.minuit import core

__all__ = [
    "FCNBase", "FCNGradientBase", "ParametricFunction", "FumiliFCNBase", "FumiliChi2FCN",
    "FumiliStandardChi2FCN", "MinuitParameter", "MnUserParameters", "MnUserParameterState",
    "MnUserCovariance", "MnStrategy", "FunctionMinimum", "MnApplication", "MnMigrad",
    "MnSimplex", "MnMinimize", "MnScan", "MnFumiliMinimize", "MnHesse", "MnMinos",
    "MinosError", "MnContours", "ContoursError", "MnPlot", "MnPrint", "ModularFunctionMinimizer",
    "VariableMetricMinimizer", "SimplexMinimizer", "CombinedMinimizer", "ScanMinimizer",
    "FumiliMinimizer", "MnMachinePrecision", "Minuit2",
]  # fmt: skip

#: ``MnApplication``'s default tolerance: MIGRAD stops at an EDM of ``0.002 * tol * up``.
TOLERANCE = 0.1


def _pairs(points: Any) -> Any:
    """A ``std::vector<std::pair<double, double>>`` of ``(x, y)`` points."""
    return std.vector["std::pair<double,double>"]([Pair(float(x), float(y)) for x, y in points])


def _vector(values: Any) -> Any:
    """A ``std::vector<double>`` of ``values``, as Minuit2 hands the function its parameters."""
    return std.vector["double"]([float(v) for v in values])


class FCNBase:
    """``FCNBase``: the function to minimise - ``operator()`` of the parameters - and ``Up``,
    the change in it that is one standard deviation (1 for a chi-square, 0.5 for -log L)."""

    def __call__(self, par: Any) -> float:
        raise NotImplementedError("an FCNBase's operator() is the function to minimise")

    def Up(self) -> float:
        return 1.0

    def ErrorDef(self) -> float:
        return float(self.Up())

    def SetErrorDef(self, up: Any) -> None:
        """``SetErrorDef``: a function whose ``Up`` is fixed has none to set."""

    def HasGradient(self) -> bool:
        return False


class FCNGradientBase(FCNBase):
    """``FCNGradientBase``: a function that gives its own ``Gradient`` too."""

    def Gradient(self, par: Any) -> Any:
        raise NotImplementedError("an FCNGradientBase's Gradient is the function's derivatives")

    def HasGradient(self) -> bool:
        return True


class ParametricFunction(FCNBase):
    """``ParametricFunction``: a function of variables with parameters of its own, ``par``,
    which Fumili's chi-square sets to each measurement's position in turn."""

    def __init__(self, params: Any = 0) -> None:
        self.par = [0.0] * int(params) if isinstance(params, (int, np.integer)) else \
            [float(v) for v in params]

    def SetParameters(self, params: Any) -> None:
        self.par = [float(v) for v in params]

    def GetParameters(self) -> list[float]:
        return self.par

    def NumberOfParameters(self) -> int:
        return len(self.par)


class FumiliFCNBase(FCNBase):
    """``FumiliFCNBase``: a figure of merit made of terms, which Fumili can minimise."""


class FumiliChi2FCN(FumiliFCNBase):
    """``FumiliChi2FCN``: the chi-square of a model function against measurements."""


class FumiliStandardChi2FCN(FumiliChi2FCN):
    """``FumiliStandardChi2FCN(model, measurements, positions, variances)``: the chi-square of
    a ``ParametricFunction`` whose parameters are each position, at the fit's parameters; a
    zero variance counts as one, as ROOT and PAW take it."""

    def __init__(self, model: ParametricFunction, measurements: Any, positions: Any,
                 variances: Any) -> None:  # fmt: skip
        self._model, self._measured = model, [float(v) for v in measurements]
        self._positions = [list(p) if isinstance(p, Sequence) or hasattr(p, "__len__") else
                           [float(p)] for p in positions]  # fmt: skip
        self._weights = [1.0 if float(v) == 0 else 1.0 / math.sqrt(float(v)) for v in variances]

    def __call__(self, par: Any) -> float:
        total = 0.0
        for position, measured, weight in zip(self._positions, self._measured, self._weights,
                                              strict=True):  # fmt: skip
            self._model.SetParameters(position)
            total += ((float(self._model(par)) - measured) * weight) ** 2
        return total


def _function(fcn: FCNBase) -> Any:
    """Minuit2's ``FCN`` of a user's function: called with a vector, its gradient if it has one."""
    gradient = None
    if fcn.HasGradient():
        def gradient(x: Any) -> Any:
            derivatives = fcn.Gradient(_vector(x))  # type: ignore[attr-defined]
            return np.asarray(list(derivatives), dtype=np.float64)

    return core().FCN(lambda x: float(fcn(_vector(x))), gradient, None, None, True,
                      float(fcn.Up()))


class MinuitParameter:
    """``MinuitParameter``: one parameter's number, name, value, error, and limits."""

    def __init__(self, held: Any) -> None:
        self._held = held

    def Number(self) -> int:
        return int(self._held.number)

    def GetName(self) -> str:
        return str(self._held.name)

    Name = GetName

    def Value(self) -> float:
        return float(self._held.value)

    def Error(self) -> float:
        return float(self._held.error)

    def IsFixed(self) -> bool:
        return bool(self._held.is_fixed)

    def IsConst(self) -> bool:
        return bool(self._held.is_const)

    def HasLimits(self) -> bool:
        return bool(self._held.has_limits)

    def HasLowerLimit(self) -> bool:
        return bool(self._held.has_lower_limit)

    def HasUpperLimit(self) -> bool:
        return bool(self._held.has_upper_limit)

    def LowerLimit(self) -> float:
        return float(self._held.lower_limit)

    def UpperLimit(self) -> float:
        return float(self._held.upper_limit)


class MnUserParameters:
    """``MnUserParameters``: the parameters by name - values, errors, limits, which are fixed.

    ``MnUserParameters(values, errors)`` names them ``p0``, ``p1``...; a
    parameter is found by its name or its number from 0.
    """

    def __init__(self, values: Any = None, errors: Any = None) -> None:
        given = values._state if isinstance(values, MnUserParameters) else None
        self._state: Any = core().MnUserParameterState(given) if given is not None else \
            core().MnUserParameterState()
        for at, value in enumerate(values if given is None and values is not None else ()):
            self._state.add(f"p{at}", float(value), float(errors[at]))

    def _index(self, which: Any) -> int:
        return int(self._state.trafo.index(which)) if isinstance(which, str) else int(which)

    def Add(self, name: Any, value: Any, error: Any = None, low: Any = None,
            high: Any = None) -> bool:  # fmt: skip
        """``Add(name, value[, error[, low, high]])``: a parameter; with no error, a constant."""
        if error is None:
            self._state.add(str(name), float(value))
        elif low is None:
            self._state.add(str(name), float(value), float(error))
        else:
            self._state.add(str(name), float(value), float(error), float(low), float(high))
        return True

    def Fix(self, which: Any) -> None:
        self._state.fix(self._index(which))

    def Release(self, which: Any) -> None:
        self._state.release(self._index(which))

    def SetValue(self, which: Any, value: Any) -> None:
        self._state.set_value(self._index(which), float(value))

    def SetError(self, which: Any, error: Any) -> None:
        self._state.set_error(self._index(which), float(error))

    def SetLimits(self, which: Any, low: Any, high: Any) -> None:
        self._state.set_limits(self._index(which), float(low), float(high))

    def SetLowerLimit(self, which: Any, low: Any) -> None:
        self._state.set_lower_limit(self._index(which), float(low))

    def SetUpperLimit(self, which: Any, high: Any) -> None:
        self._state.set_upper_limit(self._index(which), float(high))

    def RemoveLimits(self, which: Any) -> None:
        self._state.remove_limits(self._index(which))

    def Value(self, which: Any) -> float:
        return float(self._state[self._index(which)].value)

    def Error(self, which: Any) -> float:
        return float(self._state[self._index(which)].error)

    def Index(self, name: Any) -> int:
        return self._index(str(name))

    def GetName(self, index: Any) -> str:
        return str(self._state[int(index)].name)

    Name = GetName

    def Parameter(self, which: Any) -> MinuitParameter:
        return MinuitParameter(self._state[self._index(which)])

    def Parameters(self) -> list[MinuitParameter]:
        return [MinuitParameter(held) for held in self._state]

    def Params(self) -> Any:
        return _vector(held.value for held in self._state)

    def Errors(self) -> Any:
        return _vector(held.error for held in self._state)

    def VariableParameters(self) -> int:
        return int(self._state.trafo.variable_parameters)

    def __len__(self) -> int:
        return len(self._state)

    def __str__(self) -> str:
        """The parameters as Minuit2 prints them: number, name, kind, value and error."""
        lines = ["  Pos |    Name    |  type   |      Value       |    Error +/-"]
        for held in self._state:
            kind = "const" if held.is_const else "fixed" if held.is_fixed else \
                "limited" if held.has_limits else "free"
            lines.append(f"{held.number:5d} |{held.name!s:>11} |  {kind:<7}|"
                         f"{held.value:17.10g} |{held.error:13.10g}")  # fmt: skip
        return "\n".join(lines)


class MnUserCovariance:
    """``MnUserCovariance``: the covariance of the free parameters, ``cov(i, j)``."""

    def __init__(self, matrix: Any = None) -> None:
        self._matrix = np.zeros((0, 0)) if matrix is None else np.asarray(matrix, dtype=float)

    def __call__(self, row: Any, column: Any) -> float:
        return float(self._matrix[int(row), int(column)])

    def Nrow(self) -> int:
        return int(self._matrix.shape[0])

    def __str__(self) -> str:
        return "\n".join(" ".join(f"{v:12.6g}" for v in row) for row in self._matrix) + "\n"


class MnUserParameterState(MnUserParameters):
    """``MnUserParameterState``: parameters, and what a minimisation found of them - the
    function's value, the estimated distance to the minimum, the covariance, the calls."""

    def __init__(self, values: Any = None, errors: Any = None, minimum: Any = None,
                 calls: int = 0) -> None:  # fmt: skip
        super().__init__(values, errors)
        if minimum is not None:
            self._state = core().MnUserParameterState(minimum.state)
        self._found, self._calls = minimum, int(calls)

    def Fval(self) -> float:
        return float(self._found.fval) if self._found is not None else 0.0

    def Edm(self) -> float:
        return float(self._found.edm) if self._found is not None else 0.0

    def NFcn(self) -> int:
        return self._calls

    def IsValid(self) -> bool:
        return bool(self._state.is_valid)

    def HasCovariance(self) -> bool:
        return bool(self._state.has_covariance)

    def Covariance(self) -> MnUserCovariance:
        held = self._state.covariance
        size = int(held.nrow)
        return MnUserCovariance([[held[i, j] for j in range(size)] for i in range(size)])

    def Parameters(self) -> MnUserParameters:  # type: ignore[override]
        """``Parameters``: the parameters alone, as an ``MnUserParameters``."""
        return MnUserParameters(self)


class MnStrategy:
    """``MnStrategy(level)``: how hard Minuit2 works - 0 low, 1 medium, 2 high."""

    def __init__(self, level: Any = 1) -> None:
        self._level: int = int(level.Strategy() if isinstance(level, MnStrategy) else level)

    def Strategy(self) -> int:
        return self._level

    def SetLowStrategy(self) -> None:
        self._level = 0

    def SetMediumStrategy(self) -> None:
        self._level = 1

    def SetHighStrategy(self) -> None:
        self._level = 2

    def __int__(self) -> int:
        return self._level


class FunctionMinimum:
    """``FunctionMinimum``: what a minimisation found, and whether it is to be trusted."""

    def __init__(self, found: Any, fcn: Any, calls: int, goal: float = 0.0) -> None:
        self._found, self._fcn, self._calls = found, fcn, int(calls)
        #: The EDM the minimisation aimed below, which HESSE afterwards checks it against.
        self._goal = float(goal)

    def IsValid(self) -> bool:
        return bool(self._found.is_valid)

    def Fval(self) -> float:
        return float(self._found.fval)

    def Edm(self) -> float:
        return float(self._found.edm)

    def Up(self) -> float:
        return float(self._found.errordef)

    ErrorDef = Up

    def NFcn(self) -> int:
        return self._calls

    def UserState(self) -> MnUserParameterState:
        return MnUserParameterState(minimum=self._found, calls=self._calls)

    def UserParameters(self) -> MnUserParameters:
        return self.UserState().Parameters()

    def UserCovariance(self) -> MnUserCovariance:
        return self.UserState().Covariance()

    def HasValidParameters(self) -> bool:
        return bool(self._found.is_valid or not self._found.is_above_max_edm)

    def HasValidCovariance(self) -> bool:
        return bool(self._found.has_covariance and self._found.has_posdef_covar)

    def HasAccurateCovar(self) -> bool:
        return bool(self._found.has_accurate_covar)

    def HasPosDefCovar(self) -> bool:
        return bool(self._found.has_posdef_covar)

    def HasMadePosDefCovar(self) -> bool:
        return bool(self._found.has_made_posdef_covar)

    def HesseFailed(self) -> bool:
        return bool(self._found.hesse_failed)

    def HasCovariance(self) -> bool:
        return bool(self._found.has_covariance)

    def IsAboveMaxEdm(self) -> bool:
        return bool(self._found.is_above_max_edm)

    def HasReachedCallLimit(self) -> bool:
        return bool(self._found.has_reached_call_limit)

    def __str__(self) -> str:
        """As ``operator<<`` prints a minimum: whether it converged, its calls, value and EDM,
        and the parameters."""
        lines = ["", f"  Valid         : {'yes' if self.IsValid() else 'no'}",
                 f"  Function calls: {self.NFcn()}", f"  Minimum value : {self.Fval():.10g}",
                 f"  Edm           : {self.Edm():.10g}", "  External parameters: "]
        return "\n".join(lines) + "\n" + str(self.UserParameters())


def _counted(fcn: Any) -> int:
    return int(fcn._nfcn)


def _from(given: Sequence[Any]) -> tuple[MnUserParameters, int]:
    """An application's parameters and strategy, from any of its constructors' arguments:
    ``(parameters[, covariance][, strategy])`` or ``(values, errors[, strategy])``."""
    first, rest = given[0], list(given[1:])
    if isinstance(first, MnUserParameters):
        parameters = MnUserParameters(first)
    else:
        values, errors = (first[0], first[1]) if len(first) == 2 and hasattr(first[0], "__len__") \
            else (first, rest.pop(0))
        parameters = MnUserParameters(values, errors)
    rest = [item for item in rest if not isinstance(item, MnUserCovariance)]
    return parameters, MnStrategy(rest[0]).Strategy() if rest else 1


class MnApplication:
    """``MnApplication``: a minimisation of ``fcn`` from parameters, run by calling it.

    ``app(maxfcn, tolerance)`` minimises, from where the last call ended.
    """

    #: The core application this is: Minuit2's ``MnMigrad`` unless a subclass says otherwise.
    CORE = "MnMigrad"

    def __init__(self, fcn: FCNBase, *given: Any) -> None:
        self._fcn = fcn
        state, self._strategy = _from(given)
        self._state = MnUserParameterState(state)

    def State(self) -> MnUserParameterState:
        return self._state

    def Parameters(self) -> MnUserParameters:
        return self._state.Parameters()

    def Strategy(self) -> MnStrategy:
        return MnStrategy(self._strategy)

    def _run(self, function: Any, maxfcn: int, tolerance: float) -> Any:
        made = getattr(core(), self.CORE)(function, self._state._state,
                                          core().MnStrategy(self._strategy))
        return made(int(maxfcn), float(tolerance))

    def __call__(self, maxfcn: Any = 0, tolerance: Any = TOLERANCE) -> FunctionMinimum:
        function = _function(self._fcn)
        found = self._run(function, int(maxfcn), float(tolerance))
        goal = 0.002 * float(tolerance) * float(self._fcn.Up())
        minimum = FunctionMinimum(found, function, _counted(function), goal)
        self._state = minimum.UserState()
        return minimum

    def __getattr__(self, name: str) -> Any:
        """The state's ``Fix``, ``Release``, ``SetLimits``, ``Value``... on the application."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._state, name)


class MnMigrad(MnApplication):
    """``MnMigrad``: MIGRAD, the variable-metric method."""


class MnSimplex(MnApplication):
    """``MnSimplex``: the simplex method, which needs no derivatives."""

    CORE = "MnSimplex"


class MnMinimize(MnApplication):
    """``MnMinimize``: MIGRAD, and the simplex method if MIGRAD does not converge."""

    def _run(self, function: Any, maxfcn: int, tolerance: float) -> Any:
        found = super()._run(function, maxfcn, tolerance)
        if found.is_valid:
            return found
        made = core().MnSimplex(function, found.state, core().MnStrategy(self._strategy))
        return made(int(maxfcn), float(tolerance))


class MnFumiliMinimize(MnApplication):
    """``MnFumiliMinimize``: Fumili's method, which Minuit2 here runs as MIGRAD."""


class MnScan(MnApplication):
    """``MnScan``: the parameters stepped one at a time, keeping the lowest point."""

    CORE = "MnScan"

    def Scan(self, par: Any, maxsteps: Any = 41, low: Any = 0.0,
             high: Any = 0.0) -> Any:  # fmt: skip
        """``Scan(par[, steps, low, high])``: the function along one parameter, from its value
        and then ``steps`` points; over two errors each side unless a range is given.
        The parameter is left at the lowest point found."""
        values = list(self._state.Params())
        here = MinuitParameter(self._state._state[int(par)])
        low, high = float(low), float(high)
        if low >= high:
            low, high = here.Value() - 2 * here.Error(), here.Value() + 2 * here.Error()
        low = max(low, here.LowerLimit()) if here.HasLowerLimit() else low
        high = min(high, here.UpperLimit()) if here.HasUpperLimit() else high
        best = float(self._fcn(_vector(values)))
        found = [Pair(values[int(par)], best)]
        step = (high - low) / (int(maxsteps) - 1)
        for i in range(int(maxsteps)):
            values[int(par)] = low + i * step
            value = float(self._fcn(_vector(values)))
            found.append(Pair(values[int(par)], value))
            if value < best:
                best = value
                self._state.SetValue(int(par), values[int(par)])
        return _pairs((point.first, point.second) for point in found)


class MnHesse:
    """``MnHesse([strategy])``: the covariance at a minimum, from the second derivatives."""

    def __init__(self, strategy: Any = 1) -> None:
        self._strategy = MnStrategy(strategy).Strategy()

    def __call__(self, fcn: FCNBase, minimum: Any, maxcalls: Any = 0) -> None:
        """``hesse(fcn, minimum)``: the minimum's covariance calculated, in place."""
        if not isinstance(minimum, FunctionMinimum):
            raise UnsupportedFeatureError(
                "MnHesse here works on a FunctionMinimum, as hesse(fcn, minimum); run MnMigrad "
                "first to have one, rather than giving it parameters")
        function = _function(fcn)
        core().MnHesse(core().MnStrategy(self._strategy))(function, minimum._found,
                                                         int(maxcalls), minimum._goal)
        minimum._calls += _counted(function)


class MinosError:
    """``MinosError``: MINOS's interval for one parameter, round the minimum's value."""

    def __init__(self, held: Any) -> None:
        self._held = held

    def Parameter(self) -> int:
        return int(self._held.number)

    def Lower(self) -> float:
        return float(self._held.lower)

    def Upper(self) -> float:
        return float(self._held.upper)

    def Value(self) -> float:
        return float(self._held.min)

    Min = Value

    def IsValid(self) -> bool:
        return bool(self._held.is_valid)

    def LowerValid(self) -> bool:
        return bool(self._held.lower_valid)

    def UpperValid(self) -> bool:
        return bool(self._held.upper_valid)

    def NFcn(self) -> int:
        return int(self._held.nfcn)

    def AtLowerLimit(self) -> bool:
        return bool(self._held.at_lower_limit)

    def AtUpperLimit(self) -> bool:
        return bool(self._held.at_upper_limit)

    def __call__(self) -> Pair:
        return Pair(self.Lower(), self.Upper())

    def __str__(self) -> str:
        return (f"Minos # of function calls: {self.NFcn()}\n"
                f"Minos error is {'valid' if self.IsValid() else 'invalid'}\n"
                f"lower error: {self.Lower():.10g}\nupper error: {self.Upper():.10g}\n")


class MnMinos:
    """``MnMinos(fcn, minimum[, strategy])``: MINOS intervals at the function's ``Up`` now."""

    def __init__(self, fcn: FCNBase, minimum: FunctionMinimum, strategy: Any = 1) -> None:
        self._fcn, self._minimum = fcn, minimum
        self._strategy = MnStrategy(strategy).Strategy()

    def Minos(self, par: Any, maxcalls: Any = 0, tolerance: Any = TOLERANCE) -> MinosError:
        made = core().MnMinos(_function(self._fcn), self._minimum._found,
                              core().MnStrategy(self._strategy))
        return MinosError(made(int(par), int(maxcalls), float(tolerance)))

    def __call__(self, par: Any, maxcalls: Any = 0, tolerance: Any = TOLERANCE) -> Pair:
        return self.Minos(par, maxcalls, tolerance)()

    def Lower(self, par: Any, maxcalls: Any = 0, tolerance: Any = TOLERANCE) -> float:
        return self.Minos(par, maxcalls, tolerance).Lower()

    def Upper(self, par: Any, maxcalls: Any = 0, tolerance: Any = TOLERANCE) -> float:
        return self.Minos(par, maxcalls, tolerance).Upper()


class ContoursError:
    """``ContoursError``: a contour's points, and the MINOS errors of its two parameters."""

    def __init__(self, xpar: int, ypar: int, found: Any) -> None:
        self._xpar, self._ypar = xpar, ypar
        self._xminos, self._yminos = MinosError(found[0]), MinosError(found[1])
        self._points = _pairs(found[2])

    def __call__(self) -> Any:
        return self._points

    Points = __call__

    def Xpar(self) -> int:
        return self._xpar

    def Ypar(self) -> int:
        return self._ypar

    def XMinosError(self) -> MinosError:
        return self._xminos

    def YMinosError(self) -> MinosError:
        return self._yminos

    def __str__(self) -> str:
        head = f"Contours # of function calls: {self._xminos.NFcn() + self._yminos.NFcn()}\n"
        return head + "".join(f"{p.first:12.6g} {p.second:12.6g}\n" for p in self._points)


class MnContours:
    """``MnContours(fcn, minimum[, strategy])``: points round the minimum at the function's
    ``Up`` above it, in two of its parameters."""

    def __init__(self, fcn: FCNBase, minimum: FunctionMinimum, strategy: Any = 1) -> None:
        self._fcn, self._minimum = fcn, minimum
        self._strategy = MnStrategy(strategy).Strategy()

    def Contour(self, xpar: Any, ypar: Any, npoints: Any = 20) -> ContoursError:
        made = core().MnContours(_function(self._fcn), self._minimum._found,
                                 core().MnStrategy(self._strategy))
        return ContoursError(int(xpar), int(ypar), made(int(xpar), int(ypar), int(npoints)))

    def __call__(self, xpar: Any, ypar: Any, npoints: Any = 20) -> Any:
        return self.Contour(xpar, ypar, npoints)()


class MnPlot:
    """``MnPlot``: points listed, where Minuit2 draws them in characters."""

    def __call__(self, *given: Any) -> None:
        points = given[-1]
        if len(given) == 3:
            print(f"minimum at ({float(given[0]):.6g}, {float(given[1]):.6g})")
        for point in points:
            print(f"{float(point.first):14.6g} {float(point.second):14.6g}")


class MnPrint:
    """``MnPrint``: Minuit2's messages, at a level for all of them or per prefix."""

    #: ``MnPrint::GlobalLevel``: Minuit2 prints nothing at 0.
    _level = 0

    def __init__(self, prefix: Any = "", level: Any = None) -> None:
        self._prefix = str(prefix)
        self._own = MnPrint._level if level is None else int(level)

    @staticmethod
    def SetGlobalLevel(level: Any) -> int:
        previous, MnPrint._level = MnPrint._level, int(level)
        return previous

    @staticmethod
    def GlobalLevel() -> int:
        return MnPrint._level

    def SetLevel(self, level: Any) -> int:
        previous, self._own = self._own, int(level)
        return previous

    def Level(self) -> int:
        return self._own

    def _say(self, level: int, words: Sequence[Any]) -> None:
        if self._own >= level:
            print(f"{self._prefix}:", *words)

    def Error(self, *words: Any) -> None:
        self._say(0, words)

    def Warn(self, *words: Any) -> None:
        self._say(1, words)

    def Info(self, *words: Any) -> None:
        self._say(2, words)

    def Debug(self, *words: Any) -> None:
        self._say(3, words)


class MnMachinePrecision:
    """``MnMachinePrecision``: the precision Minuit2 assumes of a double."""

    def Eps(self) -> float:
        return float(np.finfo(np.float64).eps) * 4

    def Eps2(self) -> float:
        return 2 * math.sqrt(self.Eps())


class ModularFunctionMinimizer:
    """``ModularFunctionMinimizer::Minimize``: one of Minuit2's minimisers, run once."""

    APPLICATION: type[MnApplication] = MnMigrad

    def Minimize(self, fcn: FCNBase, *given: Any) -> FunctionMinimum:
        """``Minimize(fcn, parameters[, strategy, maxfcn, tolerance])``; ``parameters`` an
        ``MnUserParameters``, a state, or ``{values, errors}``."""
        first, rest = given[0], given[1:]
        strategy = rest[0] if rest else 1
        application = self.APPLICATION(fcn, first, strategy)
        return application(*(rest[1:3]))


class VariableMetricMinimizer(ModularFunctionMinimizer):
    """``VariableMetricMinimizer``: MIGRAD."""


class SimplexMinimizer(ModularFunctionMinimizer):
    """``SimplexMinimizer``: the simplex method."""

    APPLICATION = MnSimplex


class CombinedMinimizer(ModularFunctionMinimizer):
    """``CombinedMinimizer``: MIGRAD, then the simplex if MIGRAD fails."""

    APPLICATION = MnMinimize


class ScanMinimizer(ModularFunctionMinimizer):
    """``ScanMinimizer``: the parameters stepped one at a time."""

    APPLICATION = MnScan


class FumiliMinimizer(ModularFunctionMinimizer):
    """``FumiliMinimizer``: Fumili's method, run as MIGRAD."""


class _Namespace:
    """``ROOT::Minuit2``: the names above, by their C++ path."""

    def __init__(self, names: dict[str, Any]) -> None:
        self.__dict__.update(names)

    def __repr__(self) -> str:
        return "<namespace ROOT::Minuit2>"


#: ``ROOT.Minuit2``.
Minuit2 = _Namespace({name: globals()[name] for name in __all__[:-1]})
