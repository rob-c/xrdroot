"""``ROOT.Math``: the MathCore namespace - distributions, special functions and vectors.

Densities are ``*_pdf``, lower tails ``*_cdf``, upper tails ``*_cdf_c`` and
their inverses ``*_quantile`` and ``*_quantile_c``, with ROOT's argument
orders: the variable first, then the shape, then the location ``x0``. The
vectors are those of :mod:`.genvector`, and ``VectorUtil`` beside them.
"""

from __future__ import annotations

import cmath
import math
from typing import Any, ClassVar

import numpy as np

from ...fit.defaults import DEFAULTS, minimizer_algo, minimizer_type, set_minimizer
from ...function.rootmath import *  # noqa: F403 - scalar ROOT::Math compatibility exports
from ...function.rootmath import __all__ as _scalars
from .fitdata import WrappedMultiTF1  # noqa: F401 - ROOT::Math's, by name
from .genvector import *  # noqa: F403
from .genvector import __all__ as _vectors
from .legendres import (  # noqa: F401 - ROOT::Math's Legendre polynomials, by name
    assoc_legendre,
    legendre,
    sph_legendre,
)
from .lfsr import LFSR  # noqa: F401 - ROOT::Math::LFSR, by name
from .mathtools import (  # noqa: F401 - ROOT::Math's function objects and tools, by name
    Factory,
    Functor,
    Functor1D,
    GeneticMinimizer,
    GeneticMinimizerParameters,
    GradFunctor,
    GradFunctor1D,
    IntegratorOneDim,
    Minimizer,
    RootFinder,
)
from .mathtools import Integrator as _Integrator

#: The fraction of an interval a golden-section search keeps: ``(3 - sqrt 5) / 2``, as
#: GSL's ``GSL_MIN_GOLDEN_RATIO`` and ROOT's ``GSLMinimizer1D`` use it.
_GOLDEN_RATIO = 0.381966011250105151795413165634

#: GSL's status codes, which ``GSLMinimizer1D.Minimize`` returns.
_GSL_SUCCESS = 0
_GSL_EMAXITER = 1

#: Below this the sine and cosine integrals' power series is a few dozen terms whose partial
#: sums stay small, so summing it exactly loses only the last bit or two; above it their
#: terms grow to the size of ``x**x / x!`` and cancel down to the answer, which costs the
#: places a caller reads. ``Si(10)`` says it: 228 cancelling to 1.66 keeps 14 digits, which
#: is not the last place of a double, while the continued fraction below converges in a few
#: dozen terms and loses none.
_SICI_SERIES_LIMIT = 5.0

#: Enough terms of each sum for ``|x| < 5`` at the last place of a double: the series needs
#: about thirty of them and the fraction about ninety.
_SICI_SERIES_TERMS = 60
_SICI_FRACTION_TERMS = 200

#: ``pi/2``, which the sine integral tends to.
_HALF_PI = 1.5707963267948966
#: Euler's constant, which the cosine integral's series starts from.
_EULER = 0.57721566490153286060651209008240243104215933593992
#: What Lentz's algorithm divides by where a denominator is zero, as ROOT's own does.
_TINY = 1e-300


class Minim1D:
    """``ROOT::Math::Minim1D``: which one-dimensional search ``GSLMinimizer1D`` runs."""

    kBRENT, kGOLDENSECTION, kBISECTION = range(3)


class IGenFunction:
    """``ROOT::Math::IGenFunction``: a function of one number, as ROOT's numerical tools take it.

    A macro's own class derives from this and overrides ``DoEval``; ``__call__``
    is what the tools call, as they call ``operator()``, which ROOT defines in
    terms of ``DoEval``.
    """

    def __call__(self, x: Any) -> float:
        return float(self.DoEval(float(x)))

    def DoEval(self, x: float) -> float:
        """``DoEval(x)``: what the function is at ``x``, which a subclass says."""
        raise NotImplementedError(
            "IGenFunction is the interface ROOT's tools call a function of one number through; "
            "derive from it and override DoEval, or pass a plain Python function to Functor1D."
        )

    def NDim(self) -> int:
        """``NDim()``: one, this being the one-dimensional interface."""
        return 1


class Util:
    """``ROOT::Math::Util``: ``ToString``, which ROOT's minimisers name their variables with.

    ``ToString`` is a template over anything an output stream takes, so what it answers is
    what ``std::ostream`` would print: a string as itself, a float in C++'s default six
    significant digits, anything else in its own ``str``.
    """

    @staticmethod
    def ToString(value: Any) -> str:
        """``ToString(value)``: ``value`` as a C++ output stream writes it."""
        if isinstance(value, str):
            return value
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)


class IMultiGenFunction:
    """``ROOT::Math::IMultiGenFunction``: a function of ``n`` numbers, as a minimiser takes it.

    A macro's own class overrides ``DoEval(x)`` - the point as an array of ``n`` numbers -
    and ``NDim()``; calling the object evaluates it, as ROOT's ``operator()`` does, which
    is ``DoEval``.
    """

    def __call__(self, x: Any) -> float:
        return float(self.DoEval(np.asarray(x, dtype=float)))

    def DoEval(self, x: Any) -> float:
        """``DoEval(x)``: the function at the point ``x``, which a subclass says."""
        raise NotImplementedError(
            "IMultiGenFunction is the interface ROOT's minimisers call a function of several "
            "numbers through; derive from it and override DoEval and NDim."
        )

    def NDim(self) -> int:
        """``NDim()``: how many numbers the function takes, which a subclass says."""
        raise NotImplementedError(
            "IMultiGenFunction is the interface ROOT's minimisers call a function of several "
            "numbers through; derive from it and return the dimension from NDim."
        )


class IMultiGradFunction(IMultiGenFunction):
    """``ROOT::Math::IMultiGradFunction``: a function of ``n`` numbers, and each partial
    derivative, which a minimiser using the gradient takes.

    A macro's own class overrides ``DoEval``, ``Derivative(x, icoord)`` and ``NDim``.
    """

    def Derivative(self, x: Any, icoord: int = 0) -> float:
        """``Derivative(x, icoord)``: the partial derivative along ``icoord`` there."""
        raise NotImplementedError(
            "IMultiGradFunction is the interface ROOT's gradient minimisers call a function "
            "through; derive from it and override Derivative, DoEval and NDim."
        )

    def Gradient(self, x: Any, grad: Any) -> None:
        """``Gradient(x, grad)``: every partial derivative at ``x``, into ``grad``."""
        for icoord in range(int(self.NDim())):
            grad[icoord] = self.Derivative(x, icoord)


class IParamMultiFunction(IMultiGenFunction):
    """``ROOT::Math::IParamMultiFunction``: a function of ``n`` numbers, with parameters.

    A macro's own class overrides ``DoEvalPar(x, p)``, ``Parameters``, ``SetParameters``
    and ``NPar``; calling the object evaluates it at the parameters it holds.
    """

    def __call__(self, x: Any) -> float:
        return float(self.DoEvalPar(np.asarray(x, dtype=float), self.Parameters()))

    def DoEvalPar(self, x: Any, params: Any) -> float:
        """``DoEvalPar(x, p)``: the function at the point ``x`` with parameters ``p``."""
        raise NotImplementedError(
            "IParamMultiFunction is the interface ROOT's fitters call a function of several "
            "numbers with parameters through; derive from it and override DoEvalPar, "
            "Parameters, SetParameters and NPar."
        )

    def Parameters(self) -> Any:
        """``Parameters()``: the parameters it holds, which a subclass says."""
        raise NotImplementedError(
            "IParamMultiFunction is the interface ROOT's fitters call a function of several "
            "numbers with parameters through; derive from it and return the parameters "
            "from Parameters."
        )

    def SetParameters(self, params: Any) -> None:
        """``SetParameters(p)``: replace the parameters it holds, which a subclass says."""
        raise NotImplementedError(
            "IParamMultiFunction is the interface ROOT's fitters call a function of several "
            "numbers with parameters through; derive from it and store the parameters in "
            "SetParameters."
        )

    def NPar(self) -> int:
        """``NPar()``: how many parameters it takes, which a subclass says."""
        raise NotImplementedError(
            "IParamMultiFunction is the interface ROOT's fitters call a function of several "
            "numbers with parameters through; derive from it and return their count from NPar."
        )


class IParamMultiGradFunction(IMultiGradFunction):
    """``ROOT::Math::IParamMultiGradFunction``: a function of ``n`` numbers, its parameters,
    and each partial derivative with respect to a parameter.

    A macro's own class overrides ``ParameterGradient(x, p, grad)``, which is what a
    fitter's gradient step reads.
    """

    def ParameterGradient(self, x: Any, params: Any, grad: Any) -> None:
        """``ParameterGradient(x, p, grad)``: each derivative w.r.t. a parameter there."""
        raise NotImplementedError(
            "IParamMultiGradFunction is the interface ROOT's fitters call a function with "
            "parameters and their gradient through; derive from it and override "
            "ParameterGradient, DoEvalPar, Parameters, SetParameters and NPar."
        )


class IParamFunction(IGenFunction):
    """``ROOT::Math::IParamFunction``: a function of one number, given a set of parameters.

    A macro's own class overrides ``DoEvalPar(x, p)``, ``Parameters``, ``SetParameters``
    and ``NPar``; calling the object evaluates it at the parameters it holds, as ROOT's
    ``operator()`` - which is ``DoEvalPar(x, Parameters())`` - does.
    """

    def __call__(self, x: Any) -> float:
        return float(self.DoEvalPar(float(x), self.Parameters()))

    def DoEval(self, x: float) -> float:
        """``DoEval(x)``: the function at ``x``, at the parameters it holds."""
        return float(self.DoEvalPar(float(x), self.Parameters()))

    def DoEvalPar(self, x: float, params: Any) -> float:
        """``DoEvalPar(x, p)``: the function at ``x`` with parameters ``p``."""
        raise NotImplementedError(
            "IParamFunction is the interface ROOT's fitters call a function with parameters "
            "through; derive from it and override DoEvalPar, Parameters, SetParameters and NPar."
        )

    def Parameters(self) -> Any:
        """``Parameters()``: the parameters it holds, which a subclass says."""
        raise NotImplementedError(
            "IParamFunction is the interface ROOT's fitters call a function with parameters "
            "through; derive from it and return the parameters from Parameters()."
        )

    def SetParameters(self, params: Any) -> None:
        """``SetParameters(p)``: replace the parameters it holds, which a subclass says."""
        raise NotImplementedError(
            "IParamFunction is the interface ROOT's fitters call a function with parameters "
            "through; derive from it and store the parameters in SetParameters()."
        )

    def NPar(self) -> int:
        """``NPar()``: how many parameters it takes, which a subclass says."""
        raise NotImplementedError(
            "IParamFunction is the interface ROOT's fitters call a function with parameters "
            "through; derive from it and return their count from NPar()."
        )


class Polynomial(IGenFunction):
    """``ROOT::Math::Polynomial``: ``p[0] + p[1] x + … + p[order] x**order``.

    The coefficients are ROOT's to set in place, so a minimiser holding a reference
    to one sees whatever the caller last set, as ROOT's ``IGenFunction &`` does.
    """

    def __init__(self, order: int, *coefficients: Any) -> None:
        self._p = [0.0] * (int(order) + 1)
        if coefficients:
            self.SetParameters(*coefficients)

    def Order(self) -> int:
        """``Order()``: one less than the number of coefficients."""
        return len(self._p) - 1

    def SetParameters(self, *params: Any) -> None:
        """``SetParameters``: the ``order+1`` coefficients, constant first.

        They come as one array - the pointer ROOT's own signature takes - or one
        at a time.
        """
        values = [float(value) for value in (params[0] if len(params) == 1 else params)]
        if len(values) != len(self._p):
            raise ValueError(
                f"a polynomial of order {self.Order()} takes {len(self._p)} coefficients and was "
                f"given {len(values)}; give one per coefficient, constant first."
            )
        self._p = values

    def Parameters(self) -> list[float]:
        """``Parameters()``: the coefficients, constant first."""
        return list(self._p)

    def DoEval(self, x: float) -> float:
        """``DoEval(x)``: Horner's rule over the coefficients."""
        value = 0.0
        for coefficient in reversed(self._p):
            value = value * x + coefficient
        return value

    def Derivative(self, x: float) -> float:
        """``Derivative(x)``: the polynomial's own derivative at ``x``, exactly."""
        value = 0.0
        for power in range(len(self._p) - 1, 0, -1):
            value = value * x + power * self._p[power]
        return value


def _si_series(x: float) -> float:
    """``Si(x)`` for a small ``x``: ``sum (-1)**k x**(2k+1) / ((2k+1) (2k+1)!)``."""
    x2 = x * x
    terms = [x]
    term = x
    for k in range(1, _SICI_SERIES_TERMS):
        term *= -x2 / (2 * k * (2 * k + 1))
        terms.append(term / (2 * k + 1))
        if abs(terms[-1]) <= 1e-18 * abs(x):
            break
    return math.fsum(terms)


def _ci_series(x: float) -> float:
    """``Ci(x)`` for a small ``x``: ``gamma + ln x + sum (-1)**k x**(2k) / (2k (2k)!)``."""
    x2 = x * x
    head = _EULER + math.log(x)
    terms = [head]
    term = 1.0
    for k in range(1, _SICI_SERIES_TERMS):
        term *= -x2 / ((2 * k - 1) * (2 * k))
        terms.append(term / (2 * k))
        if abs(terms[-1]) <= 1e-18 * abs(head):
            break
    return math.fsum(terms)


def _e1_of_ix(x: float) -> complex:
    """``E1(i x)`` by the incomplete gamma's continued fraction, which is ``-Ci(x)+i (Si(x)-pi/2)``.

    The fraction is the exponential integral's, ``e**-z / (z + 1/(1 + 1/(z + 2/(1 + ...))))``,
    taken by Lentz's algorithm at ``z = i x``, whose denominators alternate ``z`` and one
    while its numerators grow by one every other term. Working in the complex plane is the
    point: the tail of both integrals is one number here, where ``pi/2 - f cos x - g sin x``
    would have a cosine and a sine cancel down to it.
    """
    z = complex(0.0, x)
    fraction = z
    root = fraction
    tail = 0j
    for k in range(1, _SICI_FRACTION_TERMS):
        b = 1.0 if k % 2 else z
        tail = b + (k + 1) // 2 * tail
        tail = 1.0 / tail if tail else 1.0 / _TINY
        root = b + (k + 1) // 2 / root if root else b + _TINY
        step = root * tail
        fraction *= step
        if abs(step - 1.0) <= 1e-17:
            break
    return cmath.exp(-z) / fraction


def sinint(x: Any) -> float:
    """``sinint(x)``: the sine integral ``Si(x)``, as ``ROOT::Math::sinint`` answers it.

    Odd, zero at zero and tending to ``pi/2``. ROOT's own answer is Cephes's, which meets
    the tables of the integral to a unit or two in the last place; these two pieces meet
    them too, split where each is cheapest, so a caller comparing against such a table sees
    ROOT's numbers. A non-finite argument answers a NaN, as ROOT's does.
    """
    value = float(x)
    if not math.isfinite(value):
        return math.nan
    if value == 0.0:
        return value
    if abs(value) < _SICI_SERIES_LIMIT:
        return _si_series(value)
    return math.copysign(_HALF_PI + _e1_of_ix(abs(value)).imag, value)


def cosint(x: Any) -> float:
    """``cosint(x)``: the cosine integral ``Ci(x)``, which is even, so a negative argument
    gives the real part ROOT gives.

    ``-inf`` at zero, where the logarithm its series starts from is; the sine integral's
    series and continued fraction serve it, so the two agree where ROOT's do.
    """
    value = float(x)
    if not math.isfinite(value):
        return math.nan
    magnitude = abs(value)
    if magnitude == 0.0:
        return -math.inf
    if magnitude < _SICI_SERIES_LIMIT:
        return _ci_series(magnitude)
    return -_e1_of_ix(magnitude).real


class Integrator(_Integrator):
    """``ROOT::Math::Integrator``: ROOT's ``Integral`` overloads as well as the engine's rule.

    ``Integral(a, b)`` integrates the function ``SetFunction`` was given; ROOT's second
    overload, ``Integral(f, a, b)``, integrates a function handed to the call itself, and
    a macro that wraps a ``TF1`` per integral uses that one, so it sets the function and
    then integrates as the first overload does.
    """

    def Integral(self, *args: Any) -> float:
        """``Integral(a, b)`` or ``Integral(f, a, b)``: the integral over ``[a, b]``."""
        if len(args) == 3:
            self.SetFunction(args[0])
            args = args[1:]
        return super().Integral(*args)


class IntegrationOneDim:
    """``ROOT::Math::IntegrationOneDim``: which rule ``Integrator`` integrates with."""

    kGAUSS, kLEGENDRE, kADAPTIVE, kADAPTIVESINGULAR, kNONADAPTIVE = range(5)


class IntegrationMultiDim:
    """``ROOT::Math::IntegrationMultiDim``: which rule ``IntegratorMultiDim`` integrates with."""

    kADAPTIVE, kVEGAS, kMISER, kPLAIN = range(4)


#: The points a multi-dimensional integral is sampled at over all its dimensions together,
#: so that the lattice grows with the dimension as ``dim``-th root: two dimensions get 256 a
#: side, seven get five, and none spends more time than a two-dimensional one of 65536 calls.
_MULTIDIM_POINTS = 65536


def _lattice(low: Any, high: Any, per_axis: int) -> np.ndarray:
    """The centres of a ``per_axis``-a-side lattice over the box from ``low`` to ``high``,
    one row per point, which is the shape a function of several numbers is called with.

    The centres rather than the corners: a midpoint rule is exact for a polynomial of the
    first degree in each dimension, which is the whole of what a smooth integrand looks
    like over one cell of a lattice this dense.
    """
    steps = (np.arange(per_axis) + 0.5) / per_axis
    axes = [lo + (hi - lo) * steps for lo, hi in zip(low, high, strict=True)]
    return np.stack([axis.ravel() for axis in np.meshgrid(*axes, indexing="ij")], axis=-1)


class IntegratorMultiDim:
    """``ROOT::Math::IntegratorMultiDim``: the integral of a function of ``n`` numbers over a box.

    ROOT sends ``kADAPTIVE`` to Cernlib's rule and ``kVEGAS``, ``kMISER`` and ``kPLAIN``
    to GSL's three Monte Carlos - four samplings of two libraries, which agree to the
    tolerance they are asked for and not to the last digit. One deterministic lattice
    answers all four here, at the midpoints :func:`_lattice` gives, so the same integral
    twice is the same number; the error is the standard error of the lattice's own
    spread, which is what a caller's tolerance is compared against.
    """

    def __init__(self, *args: Any) -> None:
        self._fn: Any = next((arg for arg in args if callable(arg)), None)
        kind = next((arg for arg in args if not callable(arg)), IntegrationMultiDim.kADAPTIVE)
        self._kind = int(kind)
        self._result = math.nan
        self._error = math.nan
        self._neval = 0
        self._status = 0

    def SetFunction(self, fn: Any, *rest: Any) -> None:
        """``SetFunction(f)``: the function to integrate, a function of an array of numbers."""
        self._fn = fn

    def Integral(self, a: Any, b: Any) -> float:
        """``Integral(a, b)``: the integral over the box whose lower corners are ``a`` and
        whose upper corners are ``b``, one number each."""
        low = np.asarray(a, dtype=float).ravel()
        high = np.asarray(b, dtype=float).ravel()
        if low.shape != high.shape or low.size == 0:
            raise ValueError(
                "the box to integrate over takes as many lower corners as upper ones, at "
                f"least one of each; got {low.size} and {high.size}."
            )
        per_axis = max(2, round(_MULTIDIM_POINTS ** (1.0 / low.size)))
        volume = float(np.prod(high - low))
        values = np.array([float(self._fn(point)) for point in _lattice(low, high, per_axis)])
        self._neval = int(values.size)
        self._result = volume * float(values.mean())
        self._error = volume * float(values.std()) / math.sqrt(values.size)
        return self._result

    def Result(self) -> float:
        """``Result()``: the integral ``Integral`` last found."""
        return self._result

    def Error(self) -> float:
        """``Error()``: the standard error of the points that integral was summed over."""
        return self._error

    def NEval(self) -> int:
        """``NEval()``: how many points the function was called at."""
        return self._neval

    def Status(self) -> int:
        """``Status()``: 0, a fixed lattice having nothing to fail to converge."""
        return self._status

    def SetRelTolerance(self, tolerance: float) -> None:
        """``SetRelTolerance``: accepted, and nothing to do - see :meth:`SetAbsTolerance`."""

    def SetAbsTolerance(self, tolerance: float) -> None:
        """``SetAbsTolerance``: accepted, and nothing to do.

        GSL's Monte Carlos stop when their running estimate moves less than the tolerance
        asked for; a lattice of fixed size is summed once, so the tolerance is a number a
        caller may set and read back through ``Error()`` instead.
        """

    def SetMaxPts(self, points: int) -> None:
        """``SetMaxPts``: accepted, and nothing to do - the lattice's size is the dimension's."""


class AdaptiveIntegratorMultiDim(IntegratorMultiDim):
    """``ROOT::Math::AdaptiveIntegratorMultiDim``: Cernlib's adaptive rule, by name.

    The arguments are ``(function, absTol, relTol, maxPts)``; they are ROOT's to accept,
    and the lattice above is what answers, as it answers ``kADAPTIVE``.
    """


class WrappedTF1(IGenFunction):
    """``ROOT::Math::WrappedTF1``: a ``TF1`` as the function of one number ROOT's tools take.

    ROOT's own wrapper holds the ``TF1`` by reference and hands on its parameters; here
    ``Eval`` and ``Derivative`` are reached through the object itself, so a ``TF1``
    whose parameters are set after the wrapping is integrated with those, as in ROOT.
    """

    def __init__(self, function: Any) -> None:
        self._function = function

    def DoEval(self, x: float) -> float:
        """``DoEval(x)``: what the ``TF1`` is at ``x``."""
        return float(self._function.Eval(float(x)))

    def Derivative(self, x: float) -> float:
        """``Derivative(x)``: the ``TF1``'s own derivative there."""
        return float(self._function.Derivative(float(x)))


class GSLMinimizer1D:
    """``ROOT::Math::GSLMinimizer1D``: where a function of one number is least inside a bracket.

    Both of GSL's kinds run the search below - a golden-section one, which needs
    no derivative and no constant tuned to the function - and stop at the
    tolerances asked for; the two kinds differ in GSL in how many evaluations
    they spend reaching them, not in the minimum they reach.
    """

    def __init__(self, kind: int = Minim1D.kBRENT) -> None:
        self._kind = int(kind)
        self._fn: Any = None
        self._bounds: tuple[float, float] = (0.0, 1.0)
        self._x = math.nan
        self._fval = math.nan
        self._iterations = 0
        self._status = _GSL_SUCCESS

    def SetFunction(self, fn: Any, *bounds: float) -> bool:
        """``SetFunction(f, xlow, xup)``: the function to minimise and where to look.

        Numbers in front of the two bounds are parameters a minimiser of one
        variable has no use for; ROOT's own signature takes none, and a macro
        written against GSL's may pass them, so they are accepted and ignored.
        """
        if len(bounds) < 2:
            raise ValueError(
                "SetFunction takes the function and the two ends of the bracket to search in; "
                "give SetFunction(f, low, high)."
            )
        self._fn, self._bounds = fn, (float(bounds[-2]), float(bounds[-1]))
        return True

    def Minimize(self, maxIter: int = 100, absTol: float = 1e-8, relTol: float = 1e-8) -> int:
        """``Minimize``: search, and return GSL's status - 0 found, 1 out of iterations."""
        low, high = self._bounds
        golden = _GOLDEN_RATIO
        x = low + golden * (high - low)
        self._fval = float(self._fn(x))
        self._x, self._iterations, self._status = x, 0, _GSL_EMAXITER
        while self._iterations < int(maxIter):
            self._iterations += 1
            left = low + golden * (x - low)
            right = x + golden * (high - x)
            fleft, fright = float(self._fn(left)), float(self._fn(right))
            if fleft < self._fval <= fright:
                high, x, self._fval = x, left, fleft
            elif fright < self._fval:
                low, x, self._fval = x, right, fright
            else:  # the middle point is the lowest of the three: both sides move in
                low, high = left, right
            if high - low <= absTol + relTol * abs(x):
                self._status = _GSL_SUCCESS
                break
        self._x = x
        return self._status

    def XMinimum(self) -> float:
        """``XMinimum()``: the abscissa of the least point found."""
        return self._x

    def FValMinimum(self) -> float:
        """``FValMinimum()``: the function there."""
        return self._fval

    def Iterations(self) -> int:
        """``Iterations()``: how many steps of the search were taken."""
        return self._iterations

    def Status(self) -> int:
        """``Status()``: GSL's code for how the search ended, 0 when it converged."""
        return self._status

    def Type(self) -> int:
        """``Type()``: the kind of search asked for."""
        return self._kind

    def Name(self) -> str:
        """``Name()``: what GSL calls the search asked for."""
        if self._kind == Minim1D.kGOLDENSECTION:
            return "goldensection"
        return "brent"


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


class MathMoreLibrary:
    """``ROOT::Math::MathMoreLibrary``: what loads MathMore, whose functions - the quantiles,
    the special functions - are here already."""

    @staticmethod
    def Load() -> None:
        """``Load()``: nothing to load."""


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
    + list(_scalars)
    + list(_vectors)
    + [
        "Factory",
        "Minimizer",
        "Functor",
        "Functor1D",
        "GeneticMinimizer",
        "GeneticMinimizerParameters",
        "GradFunctor",
        "GradFunctor1D",
        "IntegratorOneDim",
        "LFSR",
        "WrappedMultiTF1",
        "RootFinder",
    ]
)
