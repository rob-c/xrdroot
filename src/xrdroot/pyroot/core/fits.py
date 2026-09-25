"""``TFitResult`` and ``TFitResultPtr``: what ``Fit`` hands back, by ROOT's accessors.

Beneath is :class:`xrdroot.FitResult`, which is ROOT's fit - its options,
starting values, Minuit through iminuit, and the summary ROOT prints - so
``r.Parameter(1)``, ``r.Chi2()`` and ``r.Print("V")`` are that result's.
A ``TFitResultPtr`` is a status as a number, ``int(r) == 0`` for a good
fit, and the result itself through ``->``, which in Python is ``.``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .messages import message
from .objects import TNamed
from .wrapping import unwrap

__all__ = ["TFitResult", "TFitResultPtr", "TMatrixDSym", "TMatrixD"]


class TMatrixDSym:
    """A covariance or correlation matrix: indexed ``m[i][j]`` or ``m(i, j)``, printed as ROOT's."""

    def __init__(self, values: Any) -> None:
        self._m = np.array(values, dtype=np.float64)

    def __call__(self, i: int, j: int) -> float:
        return float(self._m[i, j])

    def __getitem__(self, i: int) -> Any:
        return self._m[i]

    def GetNrows(self) -> int:
        return int(self._m.shape[0])

    def GetNcols(self) -> int:
        return int(self._m.shape[1])

    def GetMatrixArray(self) -> np.ndarray[Any, Any]:
        return self._m.ravel()

    def Print(self, option: str = "") -> None:
        """``TMatrixTBase::Print``: ROOT's header, then five columns at a time."""
        rows, cols = self._m.shape
        print(f"\n{rows}x{cols} matrix is as follows")
        for start in range(0, cols, 5):
            columns = range(start, min(start + 5, cols))
            print("\n     |" + "".join(f"    {j:>6} |" for j in columns))
            print("-" * (7 + 12 * len(columns)))
            for i in range(rows):
                print(f"{i:>4} |" + "".join(f" {self._m[i, j]:>10.4g} " for j in columns))
        print()


TMatrixD = TMatrixDSym


class TFitResult(TNamed):
    """``TFitResult``: the parameters found, their errors, and how good the fit was."""

    CLASS_TITLE = "Class holding the result of the fit"

    def __init__(self, result: Any = None, name: str = "") -> None:
        super().__init__(name, "TFitResult")
        self._xrd = result

    def Parameter(self, i: int) -> float:
        return float(self._xrd.parameter(int(i)))

    def ParError(self, i: int) -> float:
        return float(self._xrd.error(int(i)))

    Error = ParError

    def LowerError(self, i: int) -> float:
        return float(self._xrd.lower_error(int(i)))

    def UpperError(self, i: int) -> float:
        return float(self._xrd.upper_error(int(i)))

    def Parameters(self) -> np.ndarray[Any, Any]:
        return np.array(self._xrd.parameters)

    GetParams = Parameters

    def Errors(self) -> np.ndarray[Any, Any]:
        return np.array(self._xrd.errors)

    GetErrors = Errors

    def ParName(self, i: int) -> str:
        return str(self._xrd.parameter_names[int(i)])

    def Index(self, name: Any) -> int:
        names = list(self._xrd.parameter_names)
        return names.index(str(name)) if str(name) in names else -1

    def NPar(self) -> int:
        return len(self._xrd.parameter_names)

    NTotalParameters = NPar

    def NFreeParameters(self) -> int:
        return self.NPar() - sum(self._xrd.fixed)

    def IsParameterFixed(self, i: int) -> bool:
        return bool(self._xrd.fixed[int(i)])

    def IsParameterBound(self, i: int) -> bool:
        return bool(self._xrd.bounded[int(i)])

    def Chi2(self) -> float:
        return float(self._xrd.chi2)

    def Ndf(self) -> int:
        return int(self._xrd.ndf)

    def Prob(self) -> float:
        return float(self._xrd.prob)

    def MinFcnValue(self) -> float:
        return float(self._xrd.fcn)

    def Edm(self) -> float:
        return float(self._xrd.edm)

    def NCalls(self) -> int:
        return int(self._xrd.nfev)

    def Status(self) -> int:
        return int(self._xrd.status)

    def IsValid(self) -> bool:
        return bool(self._xrd.valid)

    def IsEmpty(self) -> bool:
        return self._xrd is None

    def CovMatrixStatus(self) -> int:
        return 3 if self._xrd.valid else 1

    def MinimizerType(self) -> str:
        return str(self._xrd.minimizer)

    def CovMatrix(self, i: int, j: int) -> float:
        return float(self._xrd.covariance[int(i), int(j)])

    def Correlation(self, i: int, j: int) -> float:
        return float(self._xrd.correlation[int(i), int(j)])

    def GetCovarianceMatrix(self) -> TMatrixDSym:
        return TMatrixDSym(self._xrd.covariance)

    def GetCorrelationMatrix(self) -> TMatrixDSym:
        return TMatrixDSym(self._xrd.correlation)

    def FittedFunction(self) -> Any:
        from .wrapping import wrap

        return wrap(self._xrd.function)

    def Print(self, option: str = "") -> None:
        """``Print``: the summary ROOT prints after a fit; ``"V"`` adds the matrices."""
        print(self._xrd.summary(covariance="V" in str(option).upper()))


class TFitResultPtr:
    """``TFitResultPtr``: the fit's status as an ``int``, and its result through attributes."""

    def __init__(self, result: TFitResult | None = None, status: int = 0) -> None:
        self._result = result
        self._status = int(status if result is None or result._xrd is None else result.Status())

    def Get(self) -> TFitResult | None:
        return self._result

    def __int__(self) -> int:
        return self._status

    def __index__(self) -> int:
        return self._status

    def __eq__(self, other: object) -> bool:
        return self._status == other if isinstance(other, int) else self is other

    def __hash__(self) -> int:
        return hash(self._status)

    def __bool__(self) -> bool:
        return self._result is not None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or self._result is None:
            raise AttributeError(name)
        return getattr(self._result, name)

    def __repr__(self) -> str:
        return f"<TFitResultPtr status {self._status}>"


def _model(f1: Any) -> Any:
    """What to fit: a ``TF1``'s function, a user function by name, or a standard shape's name."""
    if not isinstance(f1, str):
        return unwrap(f1)
    from .troot import gROOT

    named = gROOT.GetListOfFunctions().FindObject(f1)
    return unwrap(named) if named is not None and named.GetName() not in ("gaus", "expo", "landau") else f1


def fit(target: Any, f1: Any, option: str, goption: str, span: tuple[float, float]) -> TFitResultPtr:
    """``Fit`` of a histogram, a graph or a multigraph, as ``TH1::Fit`` and ``TGraph::Fit`` run it."""
    low, high = float(span[0]), float(span[1])
    chosen = (low, high) if low < high else None
    try:
        found = unwrap(target).fit(_model(f1), str(option), chosen)
    except ValueError as why:
        message("Warning", "Fit", "Fit data is empty (%s)", str(why))
        return TFitResultPtr(None, -1)
    name = f"TFitResult-{target.GetName()}-{getattr(unwrap(f1), 'name', f1)}"
    return TFitResultPtr(TFitResult(found, name))
