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
    GradFunctor,
    GradFunctor1D,
    Integrator,
    IntegratorOneDim,
    Minimizer,
    RootFinder,
)

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
        "GradFunctor",
        "GradFunctor1D",
        "Integrator",
        "IntegratorOneDim",
        "LFSR",
        "WrappedMultiTF1",
        "RootFinder",
    ]
)
