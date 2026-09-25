"""``TF1``, ``TF2``, ``TF3`` and ``TFormula``: functions from formulas or from Python code.

``TF1("f", "[0]*exp(-x/[1])", 0, 10)`` is a :class:`xrdroot.Function` of
the formula; ``TF1("f", fn, 0, 10, 2)`` one whose values are ``fn(x, p)``,
called as PyROOT calls it - ``x[0]`` the coordinate, ``p[i]`` the
parameters - once per point. A function made is put in ``gROOT``'s list of
functions, replacing one of its name, so ``FillRandom("f")`` and
``gROOT.GetFunction("f")`` find it, as in ROOT.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np

from ...function import Function
from .objects import TAttFill, TAttLine, TAttMarker, TNamed
from .refs import store
from .wrapping import adopt, register, unwrap, wrap

__all__ = ["TF1", "TF2", "TF3", "TFormula"]

#: ``TF1``'s default number of points drawn and tabulated.
NPX = 100


def _arguments(fn: Callable[..., Any]) -> int:
    """How many arguments a Python function takes: ``(x)`` alone, or ``(x, p)``."""
    try:
        found = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return 2
    positional = [p for p in found if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return 1 if len(positional) == 1 and not any(p.kind == p.VAR_POSITIONAL for p in found) else 2


def adapted(fn: Callable[..., Any], dimensions: int) -> Callable[[Any, Any], Any]:
    """``fn(x, p)`` as PyROOT calls it, made into xrdroot's ``model(points, params)``."""
    takes = _arguments(fn)

    def model(points: Any, params: Any) -> Any:
        rows = np.asarray(points, dtype=np.float64).reshape(-1, dimensions)
        values = np.asarray(params, dtype=np.float64)
        if takes == 1:
            return np.array([float(fn(row)) for row in rows])
        return np.array([float(fn(row, values)) for row in rows])

    return model


def _ranges(numbers: list[float], dimensions: int) -> Any:
    """The range for each variable from ROOT's ``xmin, xmax[, ymin, ymax[, zmin, zmax]]``."""
    pairs = [(numbers[2 * at], numbers[2 * at + 1]) for at in range(dimensions)]
    return pairs[0] if dimensions == 1 else tuple(pairs)


def _register(function: Any) -> None:
    """Put ``function`` in ``gROOT``'s list, in place of one of its name."""
    from .troot import gROOT

    listed = gROOT.GetListOfFunctions()
    old = listed.FindObject(function.GetName())
    if old is not None:
        listed.Remove(old)
    listed.Add(function)


def standard_function(name: str) -> Any:
    """``gROOT->GetFunction("gaus")``: one of ROOT's standard functions, made when first asked."""
    from ...fillrandom import STANDARD, standard_function as made

    if name not in STANDARD:
        return None
    found = wrap(made(name))
    _register(found)
    return found


class TFormula(TNamed):
    """``TFormula``: a formula of variables and parameters, in ROOT's language."""

    CLASS_TITLE = "The Formula class"
    DIM = 1

    def __init__(self, name: Any = "", formula: Any = "", *rest: Any) -> None:
        super().__init__()
        self._xrd = Function(str(name), str(formula)) if formula else Function(str(name), "0")

    def _adopted(self, xrd: Any) -> None:
        TNamed.__init__(self)
        self._xrd = xrd

    def _named(self) -> dict[str, Any]:
        named: dict[str, Any] = self._xrd._named()
        return named

    def GetName(self) -> str:
        return str(self._named()["fName"])

    def SetName(self, name: Any) -> None:
        self._named()["fName"] = str(name)

    def GetTitle(self) -> str:
        return str(self._named()["fTitle"])

    def SetTitle(self, title: Any = "") -> None:
        self._named()["fTitle"] = str(title)

    def ClassName(self) -> str:
        return str(self._xrd.classname)

    def _bitword(self) -> int:
        return int(self._named().get("fBits", 0) or 0) & 0x00FFFFFF

    def _store_bits(self, bits: int) -> None:
        self._named()["fBits"] = int(bits)

    # -- parameters ----------------------------------------------------------------------

    def GetNpar(self) -> int:
        return int(self._xrd.npar)

    def GetNdim(self) -> int:
        return int(self._xrd.dimensions)

    def GetNumber(self) -> int:
        return int(self._xrd.number)

    def _index(self, parameter: Any) -> int:
        return int(self._xrd._index(parameter if isinstance(parameter, str) else int(parameter)))

    def SetParameter(self, parameter: Any, value: float) -> None:
        values = self._xrd.parameters.copy()
        values[self._index(parameter)] = float(value)
        self._xrd.parameters = values

    def SetParameters(self, *values: Any) -> None:
        """``SetParameters(p0, p1, ...)`` or ``SetParameters(array)``: from the first on."""
        given = list(np.ravel(values[0])) if len(values) == 1 and np.ndim(values[0]) > 0 else list(values)
        held = self._xrd.parameters.copy()
        count = min(len(given), len(held))
        held[:count] = [float(value) for value in given[:count]]
        self._xrd.parameters = held

    def GetParameter(self, parameter: Any) -> float:
        return float(self._xrd.parameters[self._index(parameter)])

    def GetParameters(self, params: Any = None) -> np.ndarray[Any, Any]:
        """``GetParameters``: every parameter; given an array, it is filled with them too."""
        values: np.ndarray[Any, Any] = self._xrd.parameters.copy()
        if params is not None:
            for at, value in enumerate(values):
                params[at] = float(value)
        return values

    def SetParName(self, parameter: int, name: Any) -> None:
        names = list(self._xrd.parameter_names)
        names[int(parameter)] = str(name)
        self._xrd.parameter_names = names

    def SetParNames(self, *names: Any) -> None:
        held = list(self._xrd.parameter_names)
        for at, name in enumerate(names[: len(held)]):
            held[at] = str(name)
        self._xrd.parameter_names = held

    def GetParName(self, parameter: int) -> str:
        return str(self._xrd.parameter_names[int(parameter)])

    def GetParNumber(self, name: Any) -> int:
        names = list(self._xrd.parameter_names)
        return names.index(str(name)) if str(name) in names else -1

    def GetExpFormula(self, option: str = "") -> str:
        """``GetExpFormula``: the expression, parameters by number as ROOT writes them."""
        import re

        return re.sub(r"\[p(\d+)\]", r"[\1]", str(self._xrd.formula or ""))

    def GetFormula(self) -> Any:
        return self

    def Eval(self, x: Any, y: Any = 0.0, z: Any = 0.0, t: Any = 0.0) -> Any:
        """``Eval(x[, y[, z]])``: the value at a point, or at each of arrays of points."""
        given = (x, y, z)[: self.GetNdim()]
        return self._xrd(*given)

    def __call__(self, *args: Any) -> Any:
        """``f(x)``, ``f(x, y)``; ``f(xs, params)`` with an array of coordinates is ``EvalPar``."""
        if len(args) == 2 and np.ndim(args[0]) > 0:
            return self.EvalPar(args[0], args[1])
        if len(args) == 1 and np.ndim(args[0]) > 0 and self.GetNdim() > 1:
            return self.Eval(*list(args[0])[: self.GetNdim()])
        return self.Eval(*args)

    def EvalPar(self, x: Any, params: Any = None) -> float:
        """``EvalPar(x, params)``: at the point ``x`` - an array - with ``params`` in place."""
        point = np.asarray(x, dtype=np.float64).reshape(1, -1)[:, : self.GetNdim()]
        column = point[:, 0] if self.GetNdim() == 1 else point
        given = None if params is None else np.asarray(params, dtype=np.float64)[: self.GetNpar()]
        return float(self._xrd.evaluate(column, given)[0])

    def Print(self, option: str = "") -> None:
        """``TFormula::Print``: the name and title, then the expression."""
        print(f" {self.GetName():>20} : {self.GetTitle()} Ndim= {self.GetNdim()}, "
              f"Npar= {self.GetNpar()}, Number= {self.GetNumber()} ")  # fmt: skip
        print(" Formula expression: ")
        print(f"\t{self.GetExpFormula()} ")
        if "V" in str(option).upper():
            self._print_parameters("Par%4d %20s =  %10f ")

    def _print_parameters(self, line: str) -> None:
        from .cformat import c_format

        if self.GetNpar() > 0:
            print("List of  Parameters: ")
            for at in range(self.GetNpar()):
                values = (at, self.GetParName(at), self.GetParameter(at))
                print(c_format(line, *(values if line.startswith("Par") else values[1:])))


class TF1(TFormula, TAttLine, TAttFill, TAttMarker):
    """``TF1``: a function of one variable, from a formula or from Python code."""

    CLASS_TITLE = "The Parametric 1-D function"
    DIM = 1

    def __init__(self, name: Any = "", source: Any = None, *rest: Any) -> None:
        TNamed.__init__(self)
        if isinstance(name, TF1):
            self._xrd = name._xrd.copy()
        else:
            self._xrd = self._made(str(name), source, rest)
        if name:
            _register(self)

    def _made(self, name: str, source: Any, rest: tuple[Any, ...]) -> Function:
        """The xrdroot function ROOT's constructor arguments describe."""
        numbers = [float(value) for value in rest if not isinstance(value, str)]
        spans = numbers[: 2 * self.DIM] + [0.0, 1.0] * (self.DIM - len(numbers) // 2)
        span = _ranges(spans, self.DIM)
        if callable(source) and not isinstance(source, str):
            extra = numbers[2 * self.DIM :]
            npar = int(extra[0]) if extra else 0
            model = adapted(source, self.DIM)
            return Function.from_callable(name, model, npar, dimensions=self.DIM, range=span)
        formula = str(source) if source is not None else "0"
        return Function(name, formula, range=span, title=formula)

    def _attribute_holder(self) -> dict[str, Any]:
        f1: dict[str, Any] = self._xrd._f1
        return f1

    # -- the range and the drawing grid ---------------------------------------------------

    def SetRange(self, *limits: float) -> None:
        """``SetRange(xmin, xmax)``; a ``TF2`` takes ``(xmin, ymin, xmax, ymax)``, as ROOT does."""
        half = len(limits) // 2
        pairs = [(float(limits[at]), float(limits[at + half])) for at in range(half)]
        self._xrd.range = pairs[0] if len(pairs) == 1 else tuple(pairs)

    def GetRange(self, *refs: Any) -> tuple[float, ...]:
        """``GetRange(xmin, xmax[, ...])``: into what is given - and handed back."""
        pairs = [self._xrd.range] if self.GetNdim() == 1 else list(self._xrd.range)
        found = tuple(pair[0] for pair in pairs) + tuple(pair[1] for pair in pairs)
        for target, value in zip(refs, found):
            store(target, float(value))
        return tuple(float(value) for value in found)

    def GetXmin(self) -> float:
        return self.GetRange()[0]

    def GetXmax(self) -> float:
        return self.GetRange()[self.GetNdim()]

    def SetNpx(self, npx: int = NPX) -> None:
        self._xrd._f1["fNpx"] = int(npx)

    def GetNpx(self) -> int:
        return int(self._xrd._f1.get("fNpx") or NPX)

    # -- parameters for fitting -------------------------------------------------------------

    def SetParError(self, parameter: Any, error: float) -> None:
        errors = self._xrd.parameter_errors.copy()
        errors[self._index(parameter)] = float(error)
        self._xrd.parameter_errors = errors

    def SetParErrors(self, errors: Any) -> None:
        self._xrd.parameter_errors = np.asarray(errors, dtype=np.float64)[: self.GetNpar()]

    def GetParError(self, parameter: Any) -> float:
        return float(self._xrd.parameter_errors[self._index(parameter)])

    def GetParErrors(self) -> np.ndarray[Any, Any]:
        errors: np.ndarray[Any, Any] = self._xrd.parameter_errors.copy()
        return errors

    def SetParLimits(self, parameter: Any, low: float, high: float) -> None:
        self._xrd.set_limits(self._index(parameter), float(low), float(high))

    def GetParLimits(self, parameter: Any, low: Any, high: Any) -> tuple[float, float]:
        at = self._index(parameter)
        found = (float(self._xrd._per_parameter("fParMin")[at]), float(self._xrd._per_parameter("fParMax")[at]))
        store(low, found[0])
        store(high, found[1])
        return found

    def FixParameter(self, parameter: Any, value: float) -> None:
        self._xrd.fix(self._index(parameter), float(value))

    def ReleaseParameter(self, parameter: Any) -> None:
        self._xrd.release(self._index(parameter))

    def IsFixed(self, parameter: Any) -> bool:
        return bool(self._xrd.fixed[self._index(parameter)])

    def GetNumberFreeParameters(self) -> int:
        return self.GetNpar() - sum(self._xrd.fixed)

    def _fit(self, key: str) -> Any:
        found = self._xrd.fit_result
        return 0 if found is None else found[key]

    def GetChisquare(self) -> float:
        return float(self._fit("chi2"))

    def GetNDF(self) -> int:
        return int(self._fit("ndf"))

    def GetNumberFitPoints(self) -> int:
        return int(self._fit("npfits"))

    def GetProb(self) -> float:
        from ...stats import prob

        return prob(self.GetChisquare(), self.GetNDF())

    def SetChisquare(self, chi2: float) -> None:
        self._xrd._f1["fChisquare"] = float(chi2)

    def SetNDF(self, ndf: int) -> None:
        self._xrd._f1["fNDF"] = int(ndf)

    def SetNormalized(self, flag: bool) -> None:
        self._xrd.normalized = bool(flag)

    def IsValid(self) -> bool:
        return True

    def Update(self) -> None:
        """``Update``: the normalising integral redone, as changing a parameter does."""
        self._xrd._renormalise()

    # -- numbers from it --------------------------------------------------------------------

    def Integral(self, a: float, b: float, *rest: float) -> float:
        """``Integral(a, b[, epsrel])``; a ``TF2`` takes ``(ax, bx, ay, by)``, a ``TF3`` six."""
        if self.GetNdim() == 1:
            return float(self._xrd.integral(float(a), float(b), *(rest[:1] or (1e-12,))))
        limits = [float(a), float(b), *(float(value) for value in rest[: 2 * self.GetNdim() - 2])]
        return _gauss_legendre(self._xrd, limits)

    def Derivative(self, x: float, params: Any = None, eps: float = 0.001) -> float:
        if params is not None:
            self.SetParameters(params)
        return float(self._xrd.derivative(float(x), eps))

    def GetMaximum(self, xmin: float = 0.0, xmax: float = 0.0, *rest: Any) -> float:
        return float(self._xrd.maximum(float(xmin), float(xmax)))

    def GetMinimum(self, xmin: float = 0.0, xmax: float = 0.0, *rest: Any) -> float:
        return float(self._xrd.minimum(float(xmin), float(xmax)))

    def GetMaximumX(self, xmin: float = 0.0, xmax: float = 0.0, *rest: Any) -> float:
        return float(self._xrd.maximum_x(float(xmin), float(xmax)))

    def GetMinimumX(self, xmin: float = 0.0, xmax: float = 0.0, *rest: Any) -> float:
        return float(self._xrd.minimum_x(float(xmin), float(xmax)))

    def GetX(self, y: float, xmin: float = 0.0, xmax: float = 0.0, *rest: Any) -> float:
        return float(self._xrd.x_at(float(y), float(xmin), float(xmax)))

    def GetRandom(self, *args: Any) -> float:
        """``GetRandom([xmin, xmax][, rng])``: a number distributed as the function, from ``gRandom``."""
        from .randoms import current_generator

        numbers = [float(value) for value in args if isinstance(value, (int, float))]
        given = [value for value in args if hasattr(value, "_xrd") or hasattr(value, "rndm")]
        generator = unwrap(given[0]) if given else current_generator()
        span = (numbers[0], numbers[1]) if len(numbers) >= 2 and numbers[0] < numbers[1] else None
        return float(self._xrd.get_random(rng=generator, range=span))

    def Moment(self, n: float, a: float, b: float, params: Any = None, epsilon: float = 1e-12) -> float:
        """``Moment(n, a, b)``: the ``n``th moment of the function over ``[a, b]``, as a density."""
        from ...function.function import _numerically

        weight = _numerically(self._xrd.evaluate, float(a), float(b), epsilon)
        shifted = _numerically(lambda x: np.power(x, n) * self._xrd.evaluate(x), float(a), float(b), epsilon)
        return float(shifted / weight) if weight else 0.0

    def CentralMoment(self, n: float, a: float, b: float, params: Any = None, epsilon: float = 1e-12) -> float:
        from ...function.function import _numerically

        mean = self.Moment(1, a, b)
        weight = _numerically(self._xrd.evaluate, float(a), float(b), epsilon)
        shifted = _numerically(lambda x: np.power(x - mean, n) * self._xrd.evaluate(x), float(a), float(b), epsilon)
        return float(shifted / weight) if weight else 0.0

    def Mean(self, a: float, b: float, params: Any = None, epsilon: float = 1e-12) -> float:
        return self.Moment(1, a, b)

    def Variance(self, a: float, b: float, params: Any = None, epsilon: float = 1e-12) -> float:
        return self.CentralMoment(2, a, b)

    # -- printing, copying, drawing ------------------------------------------------------------

    def Print(self, option: str = "") -> None:
        """``TF1::Print``: a formula's name and expression, or what code it is made of."""
        if self._xrd.formula is not None:
            print(f"Formula based function:     {self.GetName()} ")
            TFormula.Print(self, option)
            return
        print(f"Interpreted based function: {self.GetName()}(double *x, double *p).  "
              f"Ndim = {self.GetNdim()}, Npar = {self.GetNpar()}  ")  # fmt: skip
        if "V" in str(option).upper():
            self._print_parameters(" %20s =  %10f ")

    def Clone(self, newname: str = "") -> Any:
        """``Clone``: the same function, sharing nothing; the copy is not put in ``gROOT``'s list."""
        return wrap(self._xrd.copy(str(newname) if newname else None))

    def Copy(self, obj: Any) -> None:
        obj._xrd = self._xrd.copy()

    def DrawCopy(self, option: str = "") -> Any:
        made = self.Clone()
        made.Draw(option)
        return made

    def GetHistogram(self) -> Any:
        """``GetHistogram``: the function sampled at the centres of ``Npx`` bins over its range."""
        from ...hist import Histogram

        low, high = self.GetXmin(), self.GetXmax()
        edges = np.linspace(low, high, self.GetNpx() + 1)
        centres = 0.5 * (edges[1:] + edges[:-1])
        values = np.asarray(self._xrd(centres), dtype=np.float64)
        made = Histogram.new("Func", edges, values, title=self.GetTitle(), errors=np.zeros(len(values)))
        return wrap(made)


def _gauss_legendre(function: Any, limits: list[float], order: int = 48) -> float:
    """The integral of a function of two or three variables, by a Gauss-Legendre product rule."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    axes, scale = [], 1.0
    for at in range(0, len(limits), 2):
        low, high = limits[at], limits[at + 1]
        axes.append(0.5 * (high - low) * nodes + 0.5 * (high + low))
        scale *= 0.5 * (high - low)
    grids = np.meshgrid(*axes, indexing="ij")
    weight = np.ones(())
    for _ in axes:
        weight = np.multiply.outer(weight, weights)
    values = np.asarray(function(*grids), dtype=np.float64)
    return float(scale * np.sum(weight * values))


class TF2(TF1):
    """``TF2``: a function of two variables."""

    CLASS_TITLE = "The Parametric 2-D function"
    DIM = 2

    def SetNpy(self, npy: int = 30) -> None:
        self._xrd.members["fNpy"] = int(npy)

    def GetNpy(self) -> int:
        return int(self._xrd.members.get("fNpy") or 30)

    def GetYmin(self) -> float:
        return self.GetRange()[1]

    def GetYmax(self) -> float:
        return self.GetRange()[3]

    def GetRandom2(self, x: Any = None, y: Any = None, rng: Any = None) -> tuple[float, float]:
        """``GetRandom2(x, y)``: a point distributed as the function, by ROOT's grid of cells."""
        from .randoms import current_generator

        generator = unwrap(rng) if rng is not None else current_generator()
        xmin, ymin, xmax, ymax = self.GetRange()
        npx, npy = self.GetNpx(), self.GetNpy()
        dx, dy = (xmax - xmin) / npx, (ymax - ymin) / npy
        cells = [_gauss_legendre(self._xrd, [xmin + dx * i, xmin + dx * (i + 1), ymin + dy * j, ymin + dy * (j + 1)], 2)
                 for j in range(npy) for i in range(npx)]  # fmt: skip
        integral = np.concatenate([[0.0], np.cumsum(np.maximum(cells, 0.0))])
        integral /= integral[-1]
        r = float(generator.rndm())
        cell = int(np.searchsorted(integral[:-1], r, side="right")) - 1
        width = integral[cell + 1] - integral[cell]
        ddx = dx * (r - integral[cell]) / width if width > 0 else 0.0
        j, i = divmod(cell, npx)
        px, py = xmin + dx * i + ddx, ymin + dy * j + dy * float(generator.rndm())
        store(x, px)
        store(y, py)
        return px, py


class TF3(TF2):
    """``TF3``: a function of three variables."""

    CLASS_TITLE = "The Parametric 3-D function"
    DIM = 3

    def SetNpz(self, npz: int = 30) -> None:
        self._xrd.members["fNpz"] = int(npz)

    def GetNpz(self) -> int:
        return int(self._xrd.members.get("fNpz") or 30)


for _cls in (TF1, TF2, TF3, TFormula):
    register(_cls.__name__, factory=lambda xrd, cls=_cls: adopt(cls, xrd))

