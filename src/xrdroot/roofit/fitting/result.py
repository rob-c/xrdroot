"""``RooFitResult``: what a fit found - parameters before and after, errors, correlations, status.

``fitTo(data, Save())`` hands one back, and ``Print()`` prints it in
RooFit's table, ``Print("v")`` with the constant parameters and the
initial values beside the final ones, in ROOT's column widths and
``%12.4e`` numbers. The global correlations appear only once something has
asked for a correlation, as in ROOT, where they are made on first use.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..collections import RooArgList, RooArgSet, as_list
from ..matrix import TMatrixDSym
from ..printing import RooPrintable, g, kClassName, kName, kValue
from ..variables import RooRealVar

__all__ = ["RooFitResult"]

#: ``covQual``'s words, as the table says them.
QUALITY = {
    -1: "Unknown, matrix was externally provided", 0: "Not calculated at all",
    1: "Approximation only, not accurate", 2: "Full matrix, but forced positive-definite",
    3: "Full, accurate covariance matrix",
}  # fmt: skip


def _global_cc(cov: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Minuit's global correlation coefficients: ``sqrt(1 - 1/(V_ii (V^-1)_ii))``."""
    if cov.size == 0:
        return np.zeros(0)
    try:
        inverse = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.zeros(len(cov))
    with np.errstate(invalid="ignore", divide="ignore"):
        found = 1.0 - 1.0 / (np.diag(cov) * np.diag(inverse))
    return np.sqrt(np.clip(np.nan_to_num(found), 0.0, None))


class RooFitResult(RooPrintable):
    """The outcome of one fit."""

    def __init__(self, name: str = "", title: str = "") -> None:
        self._name = str(name)
        self._title = str(title)
        self._status = 0
        self._cov_qual = -1
        self._min_nll = 0.0
        self._edm = 0.0
        self._invalid = 0
        self._const = RooArgList()
        self._init = RooArgList()
        self._final = RooArgList()
        self._cov = np.zeros((0, 0))
        self._history: list[tuple[str, int]] = []
        self._show_global = False

    def fill(self, minimizer: Any) -> None:
        """Take everything from a minimizer that has run."""
        from .minimizer import cov_quality

        minuit = minimizer.minuit
        self._status = minimizer.status
        self._cov_qual = cov_quality(minuit.fmin)
        self._min_nll = float(minuit.fval)
        self._edm = float(minuit.fmin.edm)
        self._invalid = minimizer.invalid
        self._const = RooArgList([p.clone(p.GetName()) for p in minimizer.all_params if p.isConstant()])
        floating = [p for p in minimizer.params if not p.isConstant()]
        floating_names = {p.GetName() for p in floating}
        self._init = RooArgList([p for p in minimizer.init_params if p.GetName() in floating_names])
        self._final = RooArgList([p.clone(p.GetName()) for p in floating])
        index = [i for i, p in enumerate(minimizer.params) if not p.isConstant()]
        covariance = np.array(minuit.covariance, dtype=np.float64) if minuit.covariance is not None else (
            np.zeros((len(minimizer.params),) * 2))
        self._cov = covariance[np.ix_(index, index)]
        self._history = list(minimizer.history)

    # -- what it holds ------------------------------------------------------------

    def GetName(self) -> str:
        return self._name

    def GetTitle(self) -> str:
        return self._title

    def SetName(self, name: str) -> None:
        self._name = str(name)

    def ClassName(self) -> str:
        return "RooFitResult"

    def status(self) -> int:
        return self._status

    def covQual(self) -> int:
        return self._cov_qual

    def setCovQual(self, quality: int) -> None:
        self._cov_qual = int(quality)

    def minNll(self) -> float:
        return self._min_nll

    def edm(self) -> float:
        return self._edm

    def numInvalidNLL(self) -> int:
        return self._invalid

    def numStatusHistory(self) -> int:
        return len(self._history)

    def statusCodeHistory(self, index: int) -> int:
        return self._history[index][1]

    def statusLabelHistory(self, index: int) -> str:
        return self._history[index][0]

    def constPars(self) -> RooArgList:
        return self._const

    def floatParsInit(self) -> RooArgList:
        return self._init

    def floatParsFinal(self) -> RooArgList:
        return self._final

    def covarianceMatrix(self) -> TMatrixDSym:
        return TMatrixDSym(len(self._cov), self._cov)

    def correlationMatrix(self) -> TMatrixDSym:
        self._show_global = True
        sigma = np.sqrt(np.diag(self._cov))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = self._cov / np.outer(sigma, sigma)
        return TMatrixDSym(len(self._cov), np.nan_to_num(corr))

    def correlation(self, one: Any, two: Any = None) -> Any:
        """The correlation of two parameters, by name or by variable; of one, its row as variables."""
        self._show_global = True
        names = self._final.names()
        first = names.index(one if isinstance(one, str) else one.GetName())
        if two is None:
            return RooArgList([RooRealVar(f"C[{names[first]},{n}]", "", float(self.correlationMatrix()(first, j)))
                               for j, n in enumerate(names)])  # fmt: skip
        second = names.index(two if isinstance(two, str) else two.GetName())
        return float(self.correlationMatrix()(first, second))

    def globalCorr(self, par: Any = None) -> Any:
        self._show_global = True
        values = _global_cc(self._cov)
        if par is None:
            return RooArgList([RooRealVar(f"GC[{n}]", "", float(v)) for n, v in zip(self._final.names(), values)])
        return float(values[self._final.names().index(par if isinstance(par, str) else par.GetName())])

    def reducedCovarianceMatrix(self, params: Any) -> TMatrixDSym:
        """The covariance of ``params`` alone - the others taken as fixed at their values."""
        names = self._final.names()
        keep = [names.index(one.GetName()) for one in as_list(params)]
        inverse = np.linalg.inv(self._cov)[np.ix_(keep, keep)]
        return TMatrixDSym(len(keep), np.linalg.inv(inverse))

    def conditionalCovarianceMatrix(self, params: Any) -> TMatrixDSym:
        return self.reducedCovarianceMatrix(params)

    def randomizePars(self) -> RooArgList:
        """The final parameters drawn anew from their Gaussian covariance, with RooFit's generator."""
        from ..rng import generator

        chol = np.linalg.cholesky(self._cov)
        draws = np.array([generator().Gaus() for _ in range(len(self._cov))])
        shifted = np.array([p.getVal() for p in self._final]) + chol @ draws
        made = RooArgList([p.clone(p.GetName()) for p in self._final])
        for par, value in zip(made, shifted):
            par.setVal(float(value))
        return made

    def params(self) -> RooArgSet:
        return RooArgSet(self._final)

    # -- printing -----------------------------------------------------------------

    def printName(self) -> str:
        return self._name

    def printTitle(self) -> str:
        return self._title

    def printClassName(self) -> str:
        return "RooFitResult"

    def printValue(self) -> str:
        return f"{self._status}"

    def defaultPrintContents(self, option: Any) -> int:
        return kName | kClassName | kValue

    def defaultPrintStyle(self, option: Any) -> int:
        from ..printing import kStandard, kVerbose

        return kVerbose if "v" in str(option or "").lower() else kStandard

    def printMultiline(self, contents: int, verbose: bool, indent: str) -> str:
        """``RooFitResult::printMultiline``: the minimum, the matrix quality, the status, the table."""
        text = (f"\n{indent}  RooFitResult: minimized FCN value: {g(self._min_nll)}, estimated distance "
                f"to minimum: {g(self._edm)}\n{indent}                covariance matrix quality: "
                f"{QUALITY.get(self._cov_qual, '')}\n{indent}                Status : ")  # fmt: skip
        text += "".join(f"{label}={code} " for label, code in self._history) + "\n\n"
        text += self._verbose_table(indent) if verbose else self._table(indent)
        return text + "\n"

    def _table(self, indent: str) -> str:
        text = (f"{indent}    Floating Parameter    FinalValue +/-  Error   \n"
                f"{indent}  --------------------  --------------------------\n")  # fmt: skip
        for par in self._final:
            text += (f"{indent}  {par.GetName():>20}  {par.getVal():12.4e} +/- "
                     f"{par.getError():9.2e}\n")  # fmt: skip
        return text

    def _verbose_table(self, indent: str) -> str:
        text = self._constants(indent)
        asym = any(par.hasAsymError() for par in self._final)
        if asym:
            text += (f"{indent}    Floating Parameter  InitialValue    FinalValue (+HiError,-LoError)    "
                     f"GblCorr.\n{indent}  --------------------  ------------  "
                     "----------------------------------  --------\n")  # fmt: skip
        else:
            text += (f"{indent}    Floating Parameter  InitialValue    FinalValue +/-  Error     GblCorr.\n"
                     f"{indent}  --------------------  ------------  --------------------------  --------\n")
        correlations = _global_cc(self._cov) if self._show_global else None
        for index, (start, par) in enumerate(zip(self._init, self._final)):
            text += f"{indent}  {par.GetName():>20}{indent}  {start.getVal():12.4e}{indent}  {par.getVal():12.4e}"
            if par.hasAsymError():
                text += f" (+{par.getAsymErrorHi():8.2e},-{-par.getAsymErrorLo():8.2e})".rjust(21)
            else:
                text += ("        " if asym else "") + f" +/- {par.getError():9.2e}"
            text += f"  {correlations[index]:8.6f}" if correlations is not None else "  <none>"
            text += "\n"
        return text

    def _constants(self, indent: str) -> str:
        if not len(self._const):
            return ""
        text = (f"{indent}    Constant Parameter    Value     \n"
                f"{indent}  --------------------  ------------\n")  # fmt: skip
        for par in self._const:
            value = f"{par.getVal():12.4e}" if par.InheritsFrom("RooRealVar") else par.printValue()
            text += f"{indent}  {par.GetName():>20}  {value:>12}\n"
        return text + "\n"
