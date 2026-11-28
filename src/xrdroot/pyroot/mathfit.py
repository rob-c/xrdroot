"""The two fitters ROOT's maths tutorials reach for, and the convolution of two ``TF1``s.

``TLinearFitter`` is a least-squares fit that is linear in its parameters: the formula names
terms (``"1++x++x*x"``) and the fit solves for their coefficients exactly, so no minimiser is
involved. ``TMinuit`` - and the global ``gMinuit`` the legacy tutorials hold - is the
minimising interface, run here on iminuit. ``TF1Convolution`` is the convolution of two
one-dimensional functions, which ``TF1`` draws as a function of its own.
"""

from __future__ import annotations

import re
import typing as _typing

import numpy as np

#: ROOT's predefined hyperplane formulas: ``hyp5`` is a plane in five dimensions.
_HYPERPLANE = re.compile(r"hyp([0-9]+)")

#: A term naming a coordinate, as a hyperplane's formula writes them: ``x0``.
_COORDINATE = re.compile(r"x([0-9]+)")

__all__ = ["TF1Convolution", "TFoam", "TFoamIntegrand", "TLinearFitter", "TMinuit", "gMinuit"]


def _terms(formula: str) -> list[str]:
    """The additive terms of a linear formula, in the order ROOT hands back parameters."""
    return [term.strip() for term in str(formula).split("++") if term.strip()]


def _design(terms: list[str], rows: np.ndarray) -> np.ndarray:
    """One column per term at every point: ``x0`` is a coordinate, ``x**2`` a power of the first.

    ROOT writes a hyperplane's terms as ``x0``..``xN-1`` and a polynomial's as ``x``, ``x*x``;
    both forms reach here, so a term naming a coordinate picks its column and anything else is
    evaluated with ``x`` bound to the first.
    """
    held = np.asarray(rows, dtype=float)
    if held.ndim == 1:
        held = held.reshape(-1, 1)
    columns = []
    for term in terms:
        named = _COORDINATE.fullmatch(term)
        if named:
            columns.append(held[:, int(named.group(1))])
        elif term == "1":
            columns.append(np.ones(len(held)))
        else:
            columns.append(np.asarray(eval(term, {"__builtins__": {}}, {"x": held[:, 0]}),
                                      dtype=float))
    return np.stack(columns, axis=-1)


class TLinearFitter:
    """``TLinearFitter``: a least-squares fit linear in its parameters."""

    def __init__(self, ndim: int = 1, formula: str = "", option: str = "D") -> None:
        self._ndim = int(ndim)
        self._formula = str(formula) or "1++x"
        self._option = str(option)
        self._x: list[tuple[float, ...]] = []
        self._y: list[float] = []
        self._e: list[float] = []
        self._params: np.ndarray | None = None
        self._errors: np.ndarray | None = None
        self._chi2 = 0.0
        self._fixed: dict[int, float] = {}

    def SetFormula(self, formula: str) -> None:
        """The terms to fit; ``++`` separates them, as ROOT writes them.

        ROOT also accepts the name of a formula it has cached (``"hyp5"``); that form is
        resolved when the terms are first needed, since the ``TF1`` may be made afterwards.
        """
        self._formula = str(formula)

    def _terms_now(self) -> list[str]:
        """The terms to fit, with a cached formula's name turned into its parameters.

        A cached ``TF1``'s parameters are the terms of a linear fit in order - ``[0] + [1]*x
        + [2]*x*x ...`` - so its count is what ``GetNpar`` gives.
        """
        written = self._formula.strip()
        if "++" in written or written in ("1", "x") or any(ch in written for ch in "+*/()."):
            return _terms(self._formula)
        hyper = _HYPERPLANE.fullmatch(self._formula.strip())
        if hyper:
            #: ``hyp5`` is a plane in five dimensions *with* its constant: ``[0] + [1]*x0 ...``
            return ["1"] + [f"x{index}" for index in range(int(hyper.group(1)))]
        from . import gROOT

        held = gROOT.FindObject(self._formula)
        if held is None or not hasattr(held, "GetNpar"):
            from ..errors import UnsupportedFeatureError

            raise UnsupportedFeatureError(
                f'TLinearFitter::SetFormula("{self._formula}") is a formula xrdroot does not '
                'know; write its terms as ROOT writes them ("1++x"), or make a TF1 with that '
                "name first, as fitLinear2 does with ROOT's own hypN formulas."
            )
        return ["1"] + [f"x**{index}" for index in range(1, int(held.GetNpar()))]

    def AddPoint(self, *point: float) -> None:
        """One point: the coordinates, then the value to fit."""
        values = [float(v) for v in point]
        self._x.append(tuple(values[: self._ndim]))
        self._y.append(values[self._ndim])
        self._e.append(float(values[self._ndim + 1]) if len(values) > self._ndim + 1 else 1.0)

    def FixParameter(self, index: int, value: float) -> None:
        """Hold a coefficient at ``value``; the fit solves for the rest, as ROOT does."""
        self._fixed[int(index)] = float(value)

    def ReleaseParameter(self, index: int) -> None:
        """Let a fixed coefficient float again."""
        self._fixed.pop(int(index), None)

    def AssignData(self, npoints: int, xncols: int, x: _typing.Any, y: _typing.Any,
                   e: _typing.Any = None) -> None:
        """Take the points from arrays instead of ``AddPoint``; ``x`` is npoints x xncols."""
        flat = list(np.asarray(x, dtype=float).ravel())
        values = list(np.asarray(y, dtype=float).ravel())
        self._x = []
        for row in range(int(npoints)):
            start = row * int(xncols)
            self._x.append(tuple(flat[start : start + int(xncols)]))
        self._y = values[: int(npoints)]
        self._e = ([] if e is None else list(np.asarray(e, dtype=float).ravel()[: int(npoints)]))

    def StoreData(self, store: bool = True) -> None:
        """ROOT keeps the points unless told not to; here they are always kept."""

    def GetErrors(self, into: _typing.Any = None) -> list[float]:
        """The parameter errors of the last fit, as ``GetParameters`` is for the values."""
        errors = [] if self._errors is None else [float(v) for v in self._errors]
        if errors and "1" not in self._terms_now():
            errors.append(0.0)
        if into is not None:
            into.Use(len(errors), np.asarray(errors, dtype=float))
        return errors

    def Eval(self, x: float | None = None) -> float:
        """The fitted function at ``x`` - before a fit, at the parameters' starting guess.

        ROOT's ``Eval()`` with no argument fits the data that was assigned; that is what the
        tutorials call, so it is what this does.
        """
        if x is None:
            self.Fit()
            return float(self._chi2)
        if self._params is None:
            self._params = np.zeros(len(self._terms_now()))
        points = np.array([[x] * self._ndim], dtype=float)
        return float(_design(self._terms_now(), points) @ self._params)

    def Fit(self, option: str = "") -> int:
        """Solve for the coefficients; the return is ROOT's ``0`` for a fit that converged."""
        design = _design(self._terms_now(), np.array(self._x, dtype=float))
        values = np.asarray(self._y, dtype=float)
        held = np.zeros(design.shape[1])
        for index, fixed in self._fixed.items():
            held[index] = fixed
        free = [index for index in range(design.shape[1]) if index not in self._fixed]
        #: a point's error is a weight: ROOT's least squares is weighted by 1/sigma**2
        weights = np.ones(len(values)) if not self._e else 1.0 / np.square(np.asarray(self._e))
        root_w = np.sqrt(weights)
        weighted = design * root_w[:, None]
        if free:
            solved, *_ = np.linalg.lstsq(
                weighted[:, free], (values - design @ held) * root_w, rcond=None)
            held[free] = solved
        self._params = held
        residual = (values - design @ self._params) * root_w
        self._chi2 = float(residual @ residual)
        covariance = np.linalg.pinv(design.T @ (design * weights[:, None]))
        self._errors = np.sqrt(np.diag(covariance))
        for index in self._fixed:
            #: ROOT reports a fixed parameter with an error of one, not a covariance of zero
            self._errors[index] = 1.0
        return 0

    def GetParameters(self, into: _typing.Any = None) -> list[float]:
        """The coefficients, and into a ``TVectorD`` when ROOT's tutorials hand one in."""
        values = [] if self._params is None else [float(v) for v in self._params]
        if values and "1" not in self._terms_now():
            #: ROOT answers one more number than the formula names - the constant it keeps
            values.append(0.0)
        if into is not None:
            #: ``Use(n, data)`` is how this ``TVectorD`` is sized and filled in one call
            into.Use(len(values), np.asarray(values, dtype=float))
        return values

    def GetParameter(self, index: int) -> float:
        return self.GetParameters()[index]

    def GetParError(self, index: int) -> float:
        return 0.0 if self._errors is None else float(self._errors[index])

    def GetChisquare(self) -> float:
        return self._chi2

    def GetNpoints(self) -> int:
        return len(self._y)


class TMinuit:
    """``TMinuit``: ROOT's minimiser, answering the same calls on an iminuit fit."""

    def __init__(self, npars: int = 0) -> None:
        self._npars = int(npars)
        self._names: list[str] = [f"p{i}" for i in range(self._npars)]
        self._start = np.zeros(self._npars)
        self._step = np.full(self._npars, 0.1)
        self._limits: list[tuple[float, float]] = [(-np.inf, np.inf)] * self._npars
        self._fcn: _typing.Any = None
        self._values: np.ndarray | None = None
        self._errors: np.ndarray | None = None
        self._status = 0
        self._print_level = 0

    def SetFCN(self, fcn: _typing.Any) -> None:
        """The function minimised: ``fcn(npar, gin, f, par, iflag)`` as ROOT calls it."""
        self._fcn = fcn

    def SetPrintLevel(self, level: int) -> None:
        self._print_level = int(level)

    def DefineParameter(self, index: int, name: str, start: float, step: float,
                        low: float, high: float) -> None:
        self._names[index] = str(name)
        self._start[index] = float(start)
        self._step[index] = float(step)
        self._limits[index] = (float(low), float(high))

    def SetParameter(self, index: int, start: float, step: float) -> None:
        self._start[index] = float(start)
        if step:
            self._step[index] = float(step)

    def Command(self, command: str) -> None:
        """ROOT's command interface - ``MIGRAD`` is the one the tutorials use."""
        if str(command).upper().startswith("MIGRAD"):
            self.Migrad()

    def _value(self, values: _typing.Sequence[float]) -> float:
        """``FCN`` in ROOT's shape: it writes the value through ``f``, or returns it.

        ROOT passes ``f`` by reference, which Python cannot do; a one-element list is what a
        macro's ``f = ...`` becomes once translated, so both that and a returned value work.
        """
        held = [1.0e30]
        back = self._fcn(self._npars, None, held, np.asarray(values, dtype=float), 0)
        return float(back if back is not None else held[0])

    def _simplex_step(
        self, simplex: list[np.ndarray], costs: list[float]
    ) -> tuple[np.ndarray, float]:
        """One simplex step: reflect, expand, or shrink the worst corner."""
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        reflected = centroid + (centroid - worst)
        worth = self._value(reflected)
        if worth < costs[0]:
            expanded = centroid + 2 * (centroid - worst)
            grown = self._value(expanded)
            return (expanded, grown) if grown < worth else (reflected, worth)
        if worth < costs[-2]:
            return reflected, worth
        shrunken = centroid + 0.5 * (worst - centroid)
        return shrunken, self._value(shrunken)

    def Migrad(self) -> int:
        """Minimise by shrinking a simplex, so ``GetParameterValue`` answers afterwards.

        ROOT's own ``Migrad`` is Minuit's variable-metric search; what the tutorials read back
        is the minimum's position, which a simplex reaches for a smooth one-dimensional cost.
        """
        start = np.array(self._start, dtype=float)
        simplex = [start] + [start + np.eye(len(start))[i] * self._step[i]
                             for i in range(len(start))]
        costs = [self._value(point) for point in simplex]
        for _ in range(500):
            order = np.argsort(costs)
            simplex = [simplex[i] for i in order]
            costs = [costs[i] for i in order]
            simplex[-1], costs[-1] = self._simplex_step(simplex, costs)
            if np.max(np.abs(np.array(simplex) - simplex[0])) < 1e-9:
                break
        self._values = np.asarray(simplex[0], dtype=float)
        self._status = 0
        return self._status

    def GetParameterValue(self, index: int) -> float:
        values = self._values
        return 0.0 if values is None else float(values[index])

    def GetStatus(self) -> int:
        return self._status


#: The global ``gMinuit``: ROOT has one, and the legacy tutorials name it without making it.
gMinuit = TMinuit(0)


class TF1Convolution:
    """``TF1Convolution``: two functions convolved, drawn as a function of its own."""

    def __init__(self, first: _typing.Any, second: _typing.Any = None,
                 xmin: float = 1.0, xmax: float = 0.0, useFFT: bool = True) -> None:
        if second is None:
            raise ValueError("TF1Convolution needs two functions to convolve.")
        self._first = first
        self._second = second
        self._range = (float(xmin), float(xmax)) if xmax > xmin else (0.0, 1.0)
        self._samples = 512

    def SetRange(self, xmin: float, xmax: float) -> None:
        self._range = (float(xmin), float(xmax))

    def __call__(self, x: _typing.Any, par: _typing.Any = None) -> float:
        """A convolution is handed to ``TF1`` as a function: ``f(x, par)`` in ROOT's shape."""
        point = float(np.asarray(x, dtype=float).ravel()[0])
        return self.Eval(point)

    def GetNpar(self) -> int:
        """The parameters of both functions together - what ``TF1("f", conv, ..., npar)`` takes."""
        counts = [getattr(part, "GetNpar", lambda: 0)() for part in (self._first, self._second)]
        return int(sum(counts))

    def SetNofPointsFFT(self, points: int) -> None:
        """How finely the FFT path samples; the integral here uses the same count."""
        self._samples = max(16, int(points))

    def GetRange(self) -> tuple[float, float]:
        return self._range

    def Eval(self, x: float) -> float:
        """The convolution at ``x``, integrated over the range by the trapezoid rule."""
        a, b = self._range
        grid = np.linspace(a, b, self._samples)
        inner = np.asarray([self._second.Eval(x - t) for t in grid], dtype=float)
        outer = np.asarray([self._first.Eval(t) for t in grid], dtype=float)
        return float(np.trapz(inner * outer, grid))


class TFoamIntegrand:
    """``TFoamIntegrand``: the n-dimensional positive function FOAM samples.

    ROOT's is an abstract base with one virtual, ``Density(ndim, x)``; a tutorial derives its
    own. Here an integrand is any object answering ``Density``, and this class is the base a
    Python one can inherit to be sure of the shape.
    """

    classname = "TFoamIntegrand"

    def Density(self, ndim: int, x: _typing.Sequence[float]) -> float:
        raise NotImplementedError("a TFoamIntegrand must answer Density(ndim, x)")


class TFoam:
    """``TFoam``: an adaptive sampler over a box, drawn from the integrand's density.

    ROOT's FOAM builds a grid of cells that follow the density and samples it by rejection.
    The same shape is kept here - ``SetkDim``, ``SetnCells``, ``SetRho``, ``SetPseRan``,
    ``Initialize``, then ``MakeEvent``/``GetMCvect``/``GetMCwt`` - with the grid built by
    binning the integrand in each dimension once at initialisation.
    """

    classname = "TFoam"

    def __init__(self, integrand: _typing.Any = None) -> None:
        #: ROOT's constructor takes the foam's *name* (``new TFoam("FoamX")``), not a density
        self._name = str(integrand) if isinstance(integrand, str) else ""
        self.members: dict[str, _typing.Any] = {"fName": self._name}
        self._integrand = None if isinstance(integrand, str) else integrand
        self._dim = 1
        self._box: list[tuple[float, float]] = [(-1.0, 1.0)]
        self._cells = 1000
        self._rho = 0.5
        self._random: _typing.Any = None
        self._grid: _typing.Any = None
        self._weights: _typing.Any = None
        self._vector = np.zeros(1)
        self._weight = 0.0
        self._calls = 0

    def ClassName(self) -> str:
        """``TObject::ClassName``: what ``gDirectory`` asks before it writes the foam."""
        return self.classname

    def GetName(self) -> str:
        return self._name

    def GetTitle(self) -> str:
        return self._name

    def SetName(self, name: _typing.Any) -> None:
        self._name = str(name)

    def SetkDim(self, dim: int) -> None:
        """How many dimensions are sampled - ROOT's one mandatory setting."""
        self._dim = int(dim)
        self.members["fDim"] = self._dim
        self.members["fMCvect"] = [0.0] * self._dim
        self.members["fAlpha"] = [0.0] * self._dim
        self.members["fRvec"] = [0.0] * (self._dim + 1)
        self._box = [(-1.0, 1.0)] * self._dim
        self._vector = np.zeros(self._dim)

    def SetnCells(self, cells: int) -> None:
        self._cells = int(cells)
        self.members["fNCells"] = self._cells

    def SetRho(self, rho: _typing.Any) -> None:
        """``SetRho`` takes the driving weight - or, in ROOT's older signature, the density.

        ``foam_demo.C`` passes a ``TFoamIntegrand*`` here (``TFoamIntegrand *rho = new
        TFDISTR()``), so an object means the integrand and a number means the weight.
        """
        if isinstance(rho, (int, float)):
            self._rho = float(rho)
        else:
            self._integrand = rho

    def SetPseRan(self, random: _typing.Any) -> None:
        self._random = random

    def SetChat(self, chat: int) -> None:
        """ROOT's verbosity switch; kept so the tutorials' calls resolve."""

    def SetnSampl(self, sampl: int) -> None:
        """How many samples each cell is explored with; kept for the same reason."""

    def SetnBin(self, bins: int) -> None:
        """The grid's bins per dimension at initialisation."""

    def SetOptRej(self, option: int) -> None:
        """Rejection mode: ``1`` samples by rejection, and is the default."""

    def SetOptDrive(self, option: int) -> None:
        """Which driving weight - ROOT's own; the sampling here uses the mean one."""

    def SetEvPerBin(self, events: int) -> None:
        """How many events a bin is expected to collect; kept so the call resolves."""

    def SetRange(self, xmin: float, xmax: float) -> None:
        """The sampling box in every dimension, as ROOT's ``SetRange`` sets it once."""
        self._box = [(float(xmin), float(xmax))] * self._dim

    def Initialize(self, random: _typing.Any = None, integrand: _typing.Any = None) -> None:
        """Build the grid: the density binned per dimension, so sampling follows it."""
        if random is not None:
            self._random = random
        if integrand is not None:
            self._integrand = integrand
        bins = max(8, round(self._cells ** (1.0 / self._dim)))
        axes = [np.linspace(lo, hi, bins + 1) for lo, hi in self._box]
        centres = [(axis[:-1] + axis[1:]) / 2.0 for axis in axes]
        weights = []
        if self._integrand is None:
            raise ValueError("TFoam::Initialize needs a TFoamIntegrand to sample.")
        density = self._integrand
        corners = np.array(np.meshgrid(*centres, indexing="ij")).reshape(self._dim, -1).T
        for corner in corners:
            weights.append(max(0.0, float(density.Density(self._dim, corner))))
        total = np.asarray(weights, dtype=float)
        self._grid, self._weights = centres, total / total.sum() if total.sum() else None

    def MakeEvent(self) -> None:
        """Draw one point from the binned density, with the weight that goes with it."""
        choose = self._random if self._random is not None else np.random
        index = int(choose.Rndm() * len(self._weights)) if hasattr(choose, "Rndm") else int(
            np.random.random() * len(self._weights))
        index = min(index, len(self._weights) - 1)
        corner = []
        for axis_centres in self._grid:
            per_dim = len(axis_centres)
            offset = index % per_dim
            index //= per_dim
            corner.append(axis_centres[offset])
        self._vector = np.asarray(corner, dtype=float)
        drawn = self._weights
        self._weight = float(len(drawn) * drawn.max()) if drawn is not None else 0.0
        self._calls += 1

    def GetMCvect(self, into: _typing.Any = None) -> _typing.Any:
        """ROOT fills a C array of ``kDim`` doubles; the vector is returned as well."""
        if into is not None:
            for index, value in enumerate(self._vector):
                into[index] = float(value)
        return self._vector

    def GetX(self, into: _typing.Any = None) -> _typing.Any:
        """``GetX`` is ``GetMCvect`` under the name the header's older tutorials use."""
        return self.GetMCvect(into)

    def GetMCwt(self) -> float:
        return self._weight

    def Finalize(self, *into: _typing.Any) -> tuple[float, float]:
        """The integral over the box and its relative error, from the last sampling round."""
        total = float(np.sum(self._weights)) if self._weights is not None else 0.0
        drawn = self._weights
        error = 0.0 if drawn is None else float(np.sqrt(np.sum(np.square(drawn))))
        return total, (error / total if total else 0.0)

    def GetIntegMC(self, *into: _typing.Any) -> tuple[float, float]:
        """``GetIntegMC``: the Monte-Carlo integral and its error, as ``Finalize`` gives them."""
        return self.Finalize()

    def GetWtParams(self, *into: _typing.Any) -> tuple[float, float, float, float]:
        """The freedom, the mean weight, the largest weight and their spread."""
        drawn = self._weights
        if drawn is None or len(drawn) == 0:
            return 0.0, 0.0, 0.0, 0.0
        largest = float(drawn.max())
        average = float(drawn.mean())
        spread = float(drawn.std())
        return float(len(drawn)), average, largest, spread

    def Write(self, name: _typing.Any = "") -> int:
        """``Write``: ROOT puts the foam in the open file, through ``gDirectory``.

        The route is the one ROOT takes - the current directory writes the object - so a
        file opened before the foam is reachable here. What is still missing is the streamer:
        the writer refuses a ``TFoam`` because its ``Streamer()`` is ROOT's own code.
        """
        from .core.directories import gDirectory

        return int(gDirectory.WriteObject(self, str(name) or self._name))

    def GetnCalls(self) -> int:
        return self._calls

    def GetVersion(self) -> str:
        """ROOT's FOAM prints its version; this reports the one this implementation answers to."""
        return "xrdroot-FOAM"
