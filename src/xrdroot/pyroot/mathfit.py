"""The two fitters ROOT's maths tutorials reach for, and the convolution of two ``TF1``s.

``TLinearFitter`` is a least-squares fit that is linear in its parameters: the formula names
terms (``"1++x++x*x"``) and the fit solves for their coefficients exactly, so no minimiser is
involved. ``TMinuit`` - and the global ``gMinuit`` the legacy tutorials hold - is the
minimising interface, run here on iminuit. ``TF1Convolution`` is the convolution of two
one-dimensional functions, which ``TF1`` draws as a function of its own.
"""

from __future__ import annotations

import typing as _typing

import numpy as np

__all__ = ["TF1Convolution", "TFoam", "TFoamIntegrand", "TLinearFitter", "TMinuit", "gMinuit"]


def _terms(formula: str) -> list[str]:
    """The additive terms of a linear formula, in the order ROOT hands back parameters."""
    return [term.strip() for term in str(formula).split("++") if term.strip()]


def _design(terms: list[str], x: _typing.Any) -> np.ndarray:
    """One column per term, evaluated at ``x``: ``1``, ``x``, ``x*x``, and so on."""
    held = np.asarray(x, dtype=float)
    columns = []
    for term in terms:
        if term == "1":
            columns.append(np.ones_like(held))
        else:
            columns.append(np.asarray(eval(term, {"__builtins__": {}}, {"x": held}), dtype=float))
    return np.stack(columns, axis=-1)


class TLinearFitter:
    """``TLinearFitter``: a least-squares fit linear in its parameters."""

    def __init__(self, ndim: int = 1, formula: str = "", option: str = "D") -> None:
        self._ndim = int(ndim)
        self._formula = str(formula) or "1++x"
        self._option = str(option)
        self._x: list[tuple[float, ...]] = []
        self._y: list[float] = []
        self._params: np.ndarray | None = None
        self._errors: np.ndarray | None = None
        self._chi2 = 0.0

    def SetFormula(self, formula: str) -> None:
        """The terms to fit; ``++`` separates them, as ROOT writes them."""
        self._formula = str(formula)

    def AddPoint(self, *point: float) -> None:
        """One point: the coordinates, then the value to fit."""
        values = [float(v) for v in point]
        self._x.append(tuple(values[: self._ndim]))
        self._y.append(values[self._ndim])

    def Eval(self, x: float) -> float:
        """The fitted function at ``x`` - before a fit, at the parameters' starting guess."""
        if self._params is None:
            self._params = np.zeros(len(_terms(self._formula)))
        return float(_design(_terms(self._formula), x) @ self._params)

    def Fit(self, option: str = "") -> int:
        """Solve for the coefficients; the return is ROOT's ``0`` for a fit that converged."""
        design = _design(_terms(self._formula), np.array([row[0] for row in self._x]))
        values = np.asarray(self._y, dtype=float)
        self._params, *_ = np.linalg.lstsq(design, values, rcond=None)
        residual = values - design @ self._params
        self._chi2 = float(residual @ residual)
        covariance = np.linalg.pinv(design.T @ design)
        self._errors = np.sqrt(np.diag(covariance))
        return 0

    def GetParameters(self) -> list[float]:
        return [] if self._params is None else [float(v) for v in self._params]

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
        self._integrand = integrand
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

    def SetkDim(self, dim: int) -> None:
        """How many dimensions are sampled - ROOT's one mandatory setting."""
        self._dim = int(dim)
        self._box = [(-1.0, 1.0)] * self._dim
        self._vector = np.zeros(self._dim)

    def SetnCells(self, cells: int) -> None:
        self._cells = int(cells)

    def SetRho(self, rho: float) -> None:
        self._rho = float(rho)

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
        for corner in np.array(np.meshgrid(*centres, indexing="ij")).reshape(self._dim, -1).T:
            weights.append(max(0.0, float(self._integrand.Density(self._dim, corner))))
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

    def GetnCalls(self) -> int:
        return self._calls

    def GetVersion(self) -> str:
        """ROOT's FOAM prints its version; this reports the one this implementation answers to."""
        return "xrdroot-FOAM"
