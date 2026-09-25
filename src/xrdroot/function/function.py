"""ROOT's ``TF1``, ``TF2`` and ``TF3``, and the ``TFormula`` inside them, as one class.

A :class:`Function` is a formula - ``"[0]*exp(-x/[tau])"``, ``"gaus"``,
``"pol3"`` - with a value for each of its parameters, a range for each of
its variables and, once it has been fitted, the uncertainties and the chi
squared the fit came back with. It is built by hand, the way ``TF1``'s
constructor builds one; read from a file, the way ROOT wrote it, on its own
or among the functions a histogram or graph was fitted with; or made from a
Python function of the variables and the parameters, the way ``TF1`` takes
a C++ one.

It is evaluated over whole arrays at once. Its gradient with respect to its
parameters - what a fit needs - is exact wherever the formula is built from
arithmetic and the elementary functions, and ROOT's own numerical one where
it is not. Integrals, extrema, roots and derivatives are ``TF1``'s methods,
by the same numerical recipes.

As with a histogram, the members are the whole of the state: setting a
parameter sets it in the ``TFormula``'s ``fClingParameters`` where ROOT
keeps it, so what is written is what was computed.
"""

from __future__ import annotations

import copy as copying
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from ..formula.errors import FormulaError
from . import numeric, saved
from .language import parse
from .members import (
    CLASSES,
    first_of,
    formula_members,
    function_members,
    parameters_members,
    ranges_of,
)
from .nodes import Env, Node, Var, is_zero
from .shapes import LABEL, expand, param_order

__all__ = ["Function", "FUNCTIONS"]

Array = Any
#: A Python model: the coordinates, ``(n,)`` or ``(n, dimensions)``, and the
#: parameters, to one value per point.
Model = Callable[[Array, Array], Array]

#: The classes that come back as a :class:`Function`.
FUNCTIONS = ("TF1", "TF2", "TF3", "TFormula")
#: ``TF1::GradientPar``'s step, in units of the parameter's error when it has one.
GRADIENT_STEP = 0.01
#: ``TF1::Derivative``'s step, as a fraction of the range.
DERIVATIVE_STEP = 0.001
#: The range ``TF1``'s constructor gives a function it is not given one for.
DEFAULT_RANGE = (0.0, 1.0)


def _limits(given: Any, ndim: int) -> list[tuple[float, float]]:
    """One ``(low, high)`` per axis: from ``(lo, hi)``, a pair of pairs, or nothing."""
    if given is None:
        pairs: list[Any] = []
    elif np.ndim(given) == 1:
        pairs = [given]
    else:
        pairs = list(given)
    pairs = pairs + [DEFAULT_RANGE] * (ndim - len(pairs))
    if len(pairs) > 3:
        raise UnsupportedFeatureError(
            f"a function of {len(pairs)} variables has no ROOT class: there are TF1, TF2 and TF3"
        )
    return [(float(low), float(high)) for low, high in pairs]


def _dimensions(tree: Node) -> int:
    """How many variables a formula uses: the highest it names, counting from one."""
    return max((node.index + 1 for node in tree.walk() if isinstance(node, Var)), default=1)


def _renamed(text: str, old: Sequence[str], new: Sequence[str]) -> str:
    """The formula with each parameter's old name in brackets replaced by its new one."""
    swap = dict(zip(old, new))
    return LABEL.sub(lambda found: f"[{swap.get(found.group(1), found.group(1))}]", text)


def _ordered(names: Sequence[str]) -> dict[str, int]:
    """A ``TFormula``'s ``fParams``: each name's index, in ``TFormulaParamOrder``."""
    return {names[i]: i for i in sorted(range(len(names)), key=lambda i: param_order(names[i]))}


class Function:
    """A function of one, two or three variables and some parameters, as ROOT's ``TF1`` is.

        >>> f = Function("decay", "[N]*exp(-x/[tau])", range=(0, 10), parameters=[100, 2])
        >>> round(float(f(2.0)), 6)
        36.787944
        >>> f.parameter_names
        ('N', 'tau')

    ``formula`` is ROOT's language: ``x``, ``y`` and ``z``, parameters by
    number ``[0]`` or by name ``[mean]``, ``^`` for a power, ``TMath`` and
    ``<cmath>``, and the predefined shapes ``gaus``, ``gausn``, ``expo``,
    ``landau``, ``landaun``, ``crystalball``, ``breitwigner``, ``bigaus``,
    ``pol0`` to ``polN`` and ``cheb0`` to ``cheb10``, with ``gaus(3)`` to
    start one's parameters at ``[3]``. ``range`` is ``(low, high)``, or one
    pair per variable; a function of ``y`` is a ``TF2``, and of ``z`` a
    ``TF3``.
    """

    __slots__ = ("classname", "members", "_layers", "_formula", "_tree", "_model", "_problem")

    def __init__(
        self,
        name: str,
        formula: str,
        *,
        range: Any = None,
        parameters: Any = None,
        parameter_names: Sequence[str] | None = None,
        title: str = "",
    ) -> None:
        expanded = expand(formula)
        names = tuple(parameter_names) if parameter_names is not None else expanded.names
        if len(names) != len(expanded.names):
            raise ValueError(
                f"{formula!r} has {len(expanded.names)} parameters, and "
                f"{len(names)} names were given for them"
            )
        text = _renamed(expanded.text, expanded.names, names)
        tree = parse(text, {label: index for index, label in enumerate(names)})
        limits = _limits(range, _dimensions(tree))
        values = _values(parameters, len(names), formula)
        heading = title or formula
        held = formula_members(name, heading, text, names, values, len(limits))
        members = function_members(name, heading, limits, len(names), formula=held)
        self._adopt(CLASSES[len(limits)], members, None)

    # -- the ways to make one ------------------------------------------------

    @classmethod
    def from_callable(
        cls,
        name: str,
        fn: Model,
        npar: int,
        *,
        dimensions: int = 1,
        range: Any = None,
        parameters: Any = None,
        parameter_names: Sequence[str] | None = None,
        title: str = "",
    ) -> Function:
        """A function whose values come from Python code, as a ``TF1`` takes C++.

        ``fn(x, params)`` is given the coordinates - an array of ``n``
        points, or of ``n`` rows of ``dimensions`` - and the parameters,
        and gives back one value per point. Written to a file, it goes the
        way ROOT writes a function of code: sampled over its range into
        ``fSave``, which is what ROOT and this library read it back from.
        """
        names = tuple(parameter_names) if parameter_names is not None else _numbered(npar)
        if len(names) != npar:
            raise ValueError(f"{name!r} has {npar} parameters, and {len(names)} names were given")
        values = _values(parameters, npar, name)
        limits = _limits(range, dimensions)
        held = parameters_members(names, values)
        members = function_members(name, title or name, limits, npar, parameters=held)
        made = cls.__new__(cls)
        made._adopt(CLASSES[len(limits)], members, fn)
        return made

    @classmethod
    def from_members(cls, classname: str, members: dict[str, Any]) -> Function | dict[str, Any]:
        """A function read from a file: a ``TF1``, ``TF2``, ``TF3`` or ``TFormula``.

        A layout that is none of ROOT 6's - a ``TF1`` from ROOT 5, which was
        a ``TFormula`` rather than holding one - stays the dictionary it was
        read as, so the histogram it hangs off still reads.
        """
        layers = first_of(members) if classname != "TFormula" else [members]
        if classname != "TFormula" and "fFormula" not in layers[0]:
            return members
        if classname == "TFormula" and "fClingParameters" not in members:
            return members
        made = cls.__new__(cls)
        made._adopt(classname, members, None)
        return made

    def _adopt(self, classname: str, members: dict[str, Any], model: Model | None) -> None:
        #: The class ROOT calls it: ``TF1``, ``TF2``, ``TF3`` or ``TFormula``.
        self.classname = classname
        #: Every member, as it is written, for whatever is not here by name.
        self.members = members
        self._model = model
        formula: dict[str, Any] | None = members
        if classname == "TFormula":
            limits = [DEFAULT_RANGE] * int(members.get("fNdim", 1) or 1)
            npar = len(members["fClingParameters"])
            self._layers = first_of(function_members("", "", limits, npar))
        else:
            self._layers = first_of(members)
            formula = _unwrapped(self._layers[0])
        self._formula = formula
        self._tree, self._problem = self._compiled()

    def _compiled(self) -> tuple[Node | None, str]:
        """The formula parsed, or why it could not be - kept, not raised, so the
        function still reads and can be written back as it came."""
        if self._formula is None:
            return None, ""
        text, names = str(self._formula["fFormula"]), self.parameter_names
        try:
            written = expand(text, names).text
            return parse(written, {label: index for index, label in enumerate(names)}), ""
        except (FormulaError, UnsupportedFeatureError) as why:
            return None, str(why)

    # -- what it is ------------------------------------------------------------

    @property
    def _f1(self) -> dict[str, Any]:
        """The ``TF1`` members, which hold the range, the errors and the fit."""
        return self._layers[0]

    def _named(self) -> dict[str, Any]:
        holder = self.members if self.classname == "TFormula" else self._f1
        named: dict[str, Any] = holder["TNamed"]
        return named

    @property
    def name(self) -> str:
        """What the function is called, which is the key it was written under."""
        return str(self._named()["fName"])

    @property
    def title(self) -> str:
        """Its title, which ROOT makes the formula it was given unless told otherwise."""
        return str(self._named()["fTitle"])

    @property
    def formula(self) -> str | None:
        """The formula as ROOT keeps it, shapes expanded; ``None`` for a function of code."""
        return None if self._formula is None else str(self._formula["fFormula"])

    @property
    def dimensions(self) -> int:
        """How many variables it takes: one for a ``TF1``, two for a ``TF2``, three a ``TF3``."""
        return len(self._layers)

    @property
    def npar(self) -> int:
        """How many parameters it has."""
        return len(self.parameters)

    # -- the parameters --------------------------------------------------------

    def _home(self) -> tuple[dict[str, Any], str]:
        """Where the parameter values are kept: the formula, or the ``TF1Parameters``."""
        if self._formula is not None:
            return self._formula, "fClingParameters"
        held = self._f1.get("fParams")
        if held is None:
            held = self._f1["fParams"] = parameters_members((), ())
        return held, "fParameters"

    @property
    def parameters(self) -> np.ndarray[Any, Any]:
        """Each parameter's value, as an array that changes the function when changed."""
        home, key = self._home()
        held = home[key]
        if not (isinstance(held, np.ndarray) and held.dtype == np.float64 and held.flags.writeable):
            held = home[key] = np.array(held if held is not None else (), dtype=np.float64)
        return held

    @parameters.setter
    def parameters(self, values: Any) -> None:
        self.set_parameters(*np.asarray(values, dtype=np.float64).ravel())

    def set_parameters(self, *values: float, **by_name: float) -> None:
        """``SetParameters``: the first values in order, then any by name.

        >>> f.set_parameters(1.0, 0.0, 2.0)                     # doctest: +SKIP
        >>> f.set_parameters(Sigma=0.5)                          # doctest: +SKIP
        """
        if len(values) > self.npar:
            raise ValueError(
                f"{self.name!r} has {self.npar} parameters, and {len(values)} were given"
            )
        held = self.parameters
        held[: len(values)] = values
        for label, value in by_name.items():
            held[self._index(label)] = value
        self._renormalise()

    def _index(self, parameter: int | str) -> int:
        if isinstance(parameter, str):
            if parameter not in self.parameter_names:
                raise KeyError(
                    f"{self.name!r} has no parameter called {parameter!r}; it has "
                    f"{', '.join(self.parameter_names) or 'none'}"
                )
            return self.parameter_names.index(parameter)
        if not 0 <= parameter < self.npar:
            raise IndexError(f"{self.name!r} has no parameter {parameter}: it has {self.npar}")
        return parameter

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Each parameter's name, in the order of its index: ``p0`` and so on unless named."""
        if self._formula is not None:
            found = [f"p{index}" for index in range(len(self._formula["fClingParameters"]))]
            for label, index in (self._formula.get("fParams") or {}).items():
                if 0 <= int(index) < len(found):
                    found[int(index)] = str(label)
            return tuple(found)
        held = self._f1.get("fParams") or {}
        return tuple(str(label) for label in held.get("fParNames") or ())

    @parameter_names.setter
    def parameter_names(self, names: Sequence[str]) -> None:
        names = tuple(names)
        if len(names) != self.npar:
            raise ValueError(
                f"{self.name!r} has {self.npar} parameters, and {len(names)} names were given"
            )
        if self._formula is not None:
            self._formula["fFormula"] = _renamed(self.formula or "", self.parameter_names, names)
            self._formula["fParams"] = _ordered(names)
        else:
            self._home()[0]["fParNames"] = list(names)

    def _per_parameter(self, key: str) -> np.ndarray[Any, Any]:
        """One of the ``TF1``'s arrays of a value per parameter, as one to change in place."""
        held = self._f1.get(key)
        if not (isinstance(held, np.ndarray) and held.dtype == np.float64 and held.flags.writeable):
            held = np.zeros(self.npar)
            given = np.asarray(self._f1.get(key) if self._f1.get(key) is not None else ())
            held[: min(len(given), self.npar)] = given[: self.npar]
            self._f1[key] = held
        return held

    @property
    def parameter_errors(self) -> np.ndarray[Any, Any]:
        """Each parameter's uncertainty, as the last fit left it: ``fParErrors``."""
        return self._per_parameter("fParErrors")

    @parameter_errors.setter
    def parameter_errors(self, errors: Any) -> None:
        self._per_parameter("fParErrors")[:] = np.asarray(errors, dtype=np.float64)

    @property
    def parameter_limits(self) -> tuple[tuple[float, float] | None, ...]:
        """Each parameter's ``(low, high)`` limits for a fit, or ``None`` where it has none.

        ROOT keeps them in ``fParMin`` and ``fParMax``: two zeros are no
        limits, and a low end at or above the high one - which is how
        ``FixParameter`` marks a parameter - is not a range either.
        """
        low, high = self._per_parameter("fParMin"), self._per_parameter("fParMax")
        return tuple((float(a), float(b)) if a < b else None for a, b in zip(low, high))

    @parameter_limits.setter
    def parameter_limits(self, limits: Sequence[tuple[float, float] | None]) -> None:
        for index, pair in enumerate(limits):
            self.set_limits(index, *(pair if pair is not None else (0.0, 0.0)))

    def set_limits(self, parameter: int | str, low: float, high: float) -> None:
        """``SetParLimits``: the range a fit keeps the parameter in; ``(0, 0)`` for none."""
        index = self._index(parameter)
        self._per_parameter("fParMin")[index] = low
        self._per_parameter("fParMax")[index] = high

    @property
    def fixed(self) -> tuple[bool, ...]:
        """Whether each parameter is fixed, which ROOT says with equal, non-zero limits."""
        low, high = self._per_parameter("fParMin"), self._per_parameter("fParMax")
        return tuple(bool(a * b != 0 and a >= b) for a, b in zip(low, high))

    @fixed.setter
    def fixed(self, flags: Sequence[bool]) -> None:
        for index, flag in enumerate(flags):
            if flag:
                self.fix(index)
            elif self.fixed[index]:
                self.release(index)

    def fix(self, parameter: int | str, value: float | None = None) -> None:
        """``FixParameter``: hold a parameter at ``value``, or where it is.

        ROOT marks it by setting both limits to the value - to one for a
        value of zero, which the limits could not otherwise tell from none.
        """
        index = self._index(parameter)
        if value is not None:
            self.set_parameters(**{self.parameter_names[index]: value})
        held = float(self.parameters[index])
        mark = held if held != 0 else 1.0
        self.set_limits(index, mark, mark)

    def release(self, parameter: int | str) -> None:
        """``ReleaseParameter``: let a fixed parameter move again, with no limits."""
        self.set_limits(parameter, 0.0, 0.0)

    # -- the range and the fit -------------------------------------------------

    @property
    def range(self) -> Any:
        """``(low, high)`` for a function of one variable, and a pair per variable otherwise."""
        pairs = ranges_of(self._layers, self.dimensions)
        return pairs[0] if len(pairs) == 1 else tuple(pairs)

    @range.setter
    def range(self, given: Any) -> None:
        limits = _limits(given, self.dimensions)
        for layer, (low, high), names in zip(self._layers, limits, _LIMIT_NAMES):
            layer[names[0]], layer[names[1]] = low, high
        self._renormalise()

    @property
    def fit_result(self) -> dict[str, Any] | None:
        """What the last fit left behind: ``chi2``, ``ndf``, ``npfits``, and the
        parameters and their errors; ``None`` for a function never fitted.

        These are ``fChisquare``, ``fNDF`` and ``fNpfits``, which is all a
        ``TF1`` keeps of its fit - not the covariance, nor whether the fit
        converged, which ROOT keeps in the ``TFitResult`` if anywhere.
        """
        f1 = self._f1
        if not (f1.get("fNpfits") or f1.get("fNDF") or f1.get("fChisquare")):
            return None
        return {
            "chi2": float(f1["fChisquare"]),
            "ndf": int(f1["fNDF"]),
            "npfits": int(f1["fNpfits"]),
            "parameters": self.parameters.copy(),
            "errors": self.parameter_errors.copy(),
        }

    @fit_result.setter
    def fit_result(self, result: dict[str, Any] | None) -> None:
        given = result or {}
        self._f1["fChisquare"] = float(given.get("chi2", 0.0))
        self._f1["fNDF"] = int(given.get("ndf", 0))
        self._f1["fNpfits"] = int(given.get("npfits", 0))
        if "errors" in given:
            self.parameter_errors = given["errors"]

    @property
    def normalized(self) -> bool:
        """``SetNormalized``: whether it is divided by its integral over its range."""
        return bool(self._f1.get("fNormalized"))

    @normalized.setter
    def normalized(self, flag: bool) -> None:
        self._f1["fNormalized"] = bool(flag)
        self._f1["fNormIntegral"] = 0.0
        self._renormalise()

    def _renormalise(self) -> None:
        """``TF1::Update``: the integral a normalised function is divided by, redone."""
        if not self.normalized or self.dimensions != 1:
            return
        self._f1["fNormIntegral"] = 0.0
        low, high = self.range
        self._f1["fNormIntegral"] = numeric.integrate(self.evaluate, low, high)

    # -- evaluation ------------------------------------------------------------

    def __call__(self, x: Any, y: Any = None, z: Any = None) -> Any:
        """``Eval``: the value at ``x``, or ``(x, y)``, or ``(x, y, z)``, arrays or numbers."""
        given = [coordinate for coordinate in (x, y, z) if coordinate is not None]
        if len(given) != self.dimensions:
            raise ValueError(
                f"{self.name!r} is a function of {self.dimensions} variables, and was given "
                f"{len(given)}"
            )
        columns = [np.asarray(column, dtype=np.float64) for column in np.broadcast_arrays(*given)]
        shape = columns[0].shape
        values = self._values([column.ravel() for column in columns], self.parameters)
        return float(values[0]) if shape == () else values.reshape(shape)

    def evaluate(self, x: Any, params: Any = None) -> np.ndarray[Any, Any]:
        """``EvalPar``: the value at each point, with ``params`` in place of the parameters.

        ``x`` is ``n`` points, ``(n,)`` for a function of one variable or
        ``(n, dimensions)`` for more.
        """
        return self._values(self._columns(x), self._params(params))

    def _params(self, params: Any) -> np.ndarray[Any, Any]:
        if params is None:
            return self.parameters
        given = np.asarray(params, dtype=np.float64)
        if given.shape != (self.npar,):
            raise ValueError(
                f"{self.name!r} takes {self.npar} parameters, and was given {given.size}"
            )
        return given

    def _columns(self, x: Any) -> list[np.ndarray[Any, Any]]:
        points = np.asarray(x, dtype=np.float64)
        if self.dimensions == 1 and points.ndim <= 1:
            return [np.atleast_1d(points)]
        if points.ndim != 2 or points.shape[1] != self.dimensions:
            raise ValueError(
                f"{self.name!r} is a function of {self.dimensions} variables, so its points are "
                f"an array of shape (n, {self.dimensions}), not {points.shape}"
            )
        return [points[:, axis] for axis in range(self.dimensions)]

    def _raw(self, columns: list[np.ndarray[Any, Any]], params: np.ndarray[Any, Any]) -> Array:
        """The value before any normalisation, from the formula, the code, or the samples."""
        if self._tree is not None:
            with np.errstate(all="ignore"):
                found = self._tree.evaluate(Env(columns, params))
            return np.broadcast_to(np.asarray(found, dtype=np.float64), columns[0].shape)
        if self._model is not None:
            points = columns[0] if len(columns) == 1 else np.stack(columns, axis=1)
            found = np.asarray(self._model(points, params), dtype=np.float64)
            return np.broadcast_to(found, columns[0].shape)
        return self._from_save(columns)

    def _from_save(self, columns: list[np.ndarray[Any, Any]]) -> Array:
        """What ROOT saved, for a function whose code is not here - or whose formula
        calls what this cannot evaluate, which is refused when nothing was saved."""
        if self._problem and not saved.kept(self._f1.get("fSave")):
            raise UnsupportedFeatureError(
                f"{self.name!r} is the formula {self.formula!r}, which cannot be evaluated "
                f"here, and holds no saved values to fall back on: {self._problem}"
            )
        if len(columns) != 1:
            raise UnsupportedFeatureError(
                f"{self.name!r} is a {self.classname} of compiled code, whose saved values "
                f"are a grid this reader does not interpolate"
            )
        return saved.interpolate(self._f1.get("fSave"), self.name, columns[0])

    def _scale(self) -> float:
        """What ``EvalPar`` divides by: the integral of a normalised function, else one.

        Values ROOT saved are its own ``EvalPar``, normalisation and all,
        and are not divided again.
        """
        integral = float(self._f1.get("fNormIntegral") or 0.0)
        live = self._tree is not None or self._model is not None
        return integral if self.normalized and integral != 0 and live else 1.0

    def _values(self, columns: list[np.ndarray[Any, Any]], params: np.ndarray[Any, Any]) -> Array:
        return np.array(self._raw(columns, params), dtype=np.float64) / self._scale()

    # -- the gradient ------------------------------------------------------------

    def gradient(self, x: Any, params: Any = None) -> np.ndarray[Any, Any]:
        """``GradientPar`` for every parameter at once: an array of shape ``(n, npar)``.

        Exact where the formula allows - sums, products, powers, ``exp``,
        ``log``, ``sqrt`` and the trigonometry, which covers polynomials,
        ``gaus``, ``expo`` and their combinations - and ROOT's numerical
        derivative otherwise: two central differences of steps ``h`` and
        ``h/2``, Richardson-combined, with ``h`` a hundredth of the
        parameter's error or a hundredth when it has none. A fixed
        parameter's column is zero, as ROOT gives it.
        """
        columns = self._columns(x)
        values = self._params(params)
        fixed = self.fixed
        out = np.zeros((len(columns[0]), self.npar))
        for k in range(self.npar):
            if not fixed[k]:
                out[:, k] = self._partial(columns, values, k)
        return out / self._scale()

    def _partial(self, columns: list[np.ndarray[Any, Any]], values: Array, k: int) -> Array:
        if self._tree is not None:
            with np.errstate(all="ignore"):
                _value, tangent = self._tree.dual(Env(columns, values), k)
            if tangent is not None:
                return 0.0 if is_zero(tangent) else tangent
        return self._numerical(columns, values, k)

    def _numerical(self, columns: list[np.ndarray[Any, Any]], values: Array, k: int) -> Array:
        error = float(self.parameter_errors[k]) if k < len(self.parameter_errors) else 0.0
        h = GRADIENT_STEP * error if error != 0 else GRADIENT_STEP

        def stepped(step: float) -> Array:
            moved = np.array(values, dtype=np.float64)
            moved[k] += step
            return np.asarray(self._raw(columns, moved), dtype=np.float64)

        return numeric.richardson(stepped, h)

    # -- TF1's numerical methods -------------------------------------------------

    def _one_variable(self, what: str) -> tuple[float, float]:
        if self.dimensions != 1:
            raise UnsupportedFeatureError(
                f"{what} is TF1's, for a function of one variable, and {self.name!r} is a "
                f"{self.classname} of {self.dimensions}"
            )
        low, high = self.range
        return low, high

    def _bounds(self, low: float | None, high: float | None, what: str) -> tuple[float, float]:
        """The range to search, which ROOT takes to be the function's own unless given one."""
        own = self._one_variable(what)
        if low is None or high is None or low >= high:
            return own
        return float(low), float(high)

    def integral(self, a: float, b: float, epsrel: float = 1e-12) -> float:
        """``Integral``: from ``a`` to ``b``, either of them infinite, to ``epsrel``."""
        self._one_variable("integral")
        return numeric.integrate(self.evaluate, float(a), float(b), epsrel, epsrel)

    def derivative(self, x: Any, eps: float = DERIVATIVE_STEP) -> Any:
        """``Derivative``: ``df/dx``, Richardson's, with a step of ``eps`` times the range."""
        low, high = self._one_variable("derivative")
        h = eps * abs(high - low)
        found = numeric.derivative(self.evaluate, np.ravel(x), h if h > 0 else DERIVATIVE_STEP)
        return float(found[0]) if np.ndim(x) == 0 else found.reshape(np.shape(x))

    def _npx(self) -> int:
        return int(self._f1.get("fNpx") or 100)

    def maximum_x(self, low: float | None = None, high: float | None = None) -> float:
        """``GetMaximumX``: where the function is largest in its range, or in ``[low, high]``."""
        a, b = self._bounds(low, high, "maximum_x")
        return numeric.minimise(lambda t: -self.evaluate(t), a, b, self._npx())

    def minimum_x(self, low: float | None = None, high: float | None = None) -> float:
        """``GetMinimumX``: where the function is least in its range, or in ``[low, high]``."""
        a, b = self._bounds(low, high, "minimum_x")
        return numeric.minimise(self.evaluate, a, b, self._npx())

    def maximum(self, low: float | None = None, high: float | None = None) -> float:
        """``GetMaximum``: the largest value in its range, or in ``[low, high]``."""
        return float(self(self.maximum_x(low, high)))

    def minimum(self, low: float | None = None, high: float | None = None) -> float:
        """``GetMinimum``: the least value in its range, or in ``[low, high]``."""
        return float(self(self.minimum_x(low, high)))

    def x_at(self, y: float, low: float | None = None, high: float | None = None) -> float:
        """``GetX``: where the function is ``y``, searched for as ROOT searches, by
        minimising ``|f(x) - y|`` - so the nearest approach where it never gets there."""
        a, b = self._bounds(low, high, "x_at")
        return numeric.minimise(lambda t: np.abs(self.evaluate(t) - y), a, b, self._npx())

    # -- copying and writing -------------------------------------------------------

    def copy(self, name: str | None = None) -> Function:
        """``Clone``: the same function, sharing nothing, renamed if ``name`` is given."""
        made = type(self).__new__(type(self))
        made._adopt(self.classname, copying.deepcopy(self.members), self._model)
        if name is not None:
            made._named()["fName"] = name
            if made._formula is not None and made.classname != "TFormula":
                made._formula["TNamed"]["fName"] = name
        return made

    def written_members(self) -> dict[str, Any]:
        """The members as ``TF1::Streamer`` writes them: a function of code sampled into ``fSave``.

        ROOT writes a function it has code for, but no formula, as its values
        at ``fNpx + 1`` points over its range; a function of Python code
        here goes the same way, and reads back - here or in ROOT - as those.
        """
        if self._model is None or self.dimensions != 1:
            return self.members
        low, high = self.range
        sampled = saved.sample(self.evaluate, low, high, self._npx())
        return {**self.members, "fSave": sampled}

    def __repr__(self) -> str:
        what = self.formula if self.formula is not None else "compiled code"
        return f"<{self.classname} {self.name!r}: {what}, {self.npar} parameters>"


#: The members each axis's range is kept in, by layer.
_LIMIT_NAMES = (("fXmin", "fXmax"), ("fYmin", "fYmax"), ("fZmin", "fZmax"))


def _numbered(count: int) -> tuple[str, ...]:
    return tuple(f"p{index}" for index in range(count))


def _values(given: Any, count: int, what: str) -> np.ndarray[Any, Any]:
    """The starting parameter values: those given, or zeros, as ROOT starts them."""
    if given is None:
        return np.zeros(count)
    values = np.asarray(given, dtype=np.float64).ravel()
    if len(values) != count:
        raise ValueError(f"{what!r} has {count} parameters, and {len(values)} values were given")
    return values


def _unwrapped(f1: dict[str, Any]) -> dict[str, Any] | None:
    """The ``TFormula`` a ``TF1`` points at, as members, whichever way it was read."""
    held = f1.get("fFormula")
    if isinstance(held, Function):
        f1["fFormula"] = held.members
        return held.members
    return held if isinstance(held, dict) else None


def dress(classname: str, members: dict[str, Any]) -> Any:
    """What :mod:`xrdroot.kinds` makes of a function read from a file."""
    return Function.from_members(classname, members)
