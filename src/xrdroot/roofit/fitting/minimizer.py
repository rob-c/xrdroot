"""``RooMinimizer``: Minuit driving a RooFit function, set up as RooFit sets it up.

The function's parameters are Minuit's, in RooFit's order - sorted by name
- each starting at its value with its error as the first step, or, for one
that has no error yet, a tenth of its range, trimmed if that would reach
past an end: ``RooAbsMinimizerFcn::synchronizeParameterSettings``. The
tolerance is RooFit's 1 (MIGRAD stops at an estimated distance to the
minimum of ``0.002 * 1 * errordef``), the strategy 1, the call limit 500 per
parameter, and Minuit is Minuit2 - iminuit, which is its C++ - so a fit
here takes the steps a fit in ROOT takes.

A point where the function cannot be computed gets RooFit's treatment: the
worst value seen so far plus ten times how bad it was, so MIGRAD backs away
(``RooAbsMinimizerFcn::applyEvalErrorHandling``).
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np

from ...fit.minuit import iminuit
from ..collections import RooArgList, RooArgSet, as_list
from ..messages import INFO, WARNING, log
from ..printing import PRECISION, g

__all__ = ["RooMinimizer", "first_step"]


def first_step(par: Any) -> float:
    """The first step of a parameter: its error, or RooFit's guess from its range."""
    step = par.getError()
    if step > 0:
        return float(step)
    if par.hasMin() and par.hasMax():
        low, high, value = par.getMin(), par.getMax(), par.getVal()
        step = 0.1 * (high - low)
        if high - value < 2 * step:
            step = (high - value) / 2
        elif value - low < 2 * step:
            step = (value - low) / 2
        return step if step != 0 else 0.1 * (high - low)
    return 1.0


def inside(par: Any) -> float:
    """Where Minuit2 starts a parameter: a tenth of its step inside a limit it sits on or beyond.

    ``MnUserParameterState::SetLimits`` moves a value at or past an end in by
    ``0.1 * error`` before the first call, so a fit whose parameter was
    clipped to a limit starts just off it - as iminuit does not by itself.
    """
    value = par.getVal()
    if par.isConstant():
        return float(value)
    step = first_step(par)
    if par.hasMin() and par.getMin() >= value:
        return float(par.getMin() + 0.1 * step)
    if par.hasMax() and value >= par.getMax():
        return float(par.getMax() - 0.1 * step)
    return float(value)


class RooMinimizer:
    """Minuit, minimising ``function`` over its free parameters."""

    def __init__(self, function: Any, cfg: Any = None) -> None:
        self.function = function
        params = [p for p in function.getParameters() if p.InheritsFrom("RooRealVar")]
        self.params = [p for p in params if not p.isConstant()]
        self.all_params = params
        self.init_params = RooArgList([p.clone(p.GetName()) for p in self.params])
        self.print_level = 1
        self.strategy = 1
        self.eps = 1.0
        self.errordef = float(function.defaultErrorLevel())
        self.max_calls = 500 * len(self.params)
        self.verbose = False
        self.eval_error_wall = True
        self.recover_strength = 10.0
        self.minuit: Any = None
        self.status = 0
        self.history: list[tuple[str, int]] = []
        self.minimizer_type = "Minuit2"
        self.invalid = 0
        self._max_fcn = -math.inf
        self._last: list[float] = []

    # -- settings -----------------------------------------------------------------

    def setPrintLevel(self, level: int) -> int:
        self.print_level = int(level)
        return self.print_level

    def setStrategy(self, strategy: int) -> None:
        self.strategy = int(strategy)

    def setEps(self, eps: float) -> None:
        self.eps = float(eps)

    def setErrorLevel(self, level: float) -> None:
        self.errordef = float(level)

    def setMaxFunctionCalls(self, n: int) -> None:
        self.max_calls = int(n)

    def setMaxIterations(self, n: int) -> None:
        """Minuit2 counts calls, not iterations: kept for the call's sake."""

    def setVerbose(self, flag: bool = True) -> None:
        self.verbose = bool(flag)

    def setEvalErrorWall(self, flag: bool) -> None:
        self.eval_error_wall = bool(flag)

    def setRecoverFromNaNStrength(self, strength: float) -> None:
        self.recover_strength = float(strength)

    def setMinimizerType(self, name: str) -> None:
        self.minimizer_type = str(name) or "Minuit2"

    def setOffsetting(self, flag: bool) -> None:
        self.function.offset = bool(flag)

    def setPrintEvalErrors(self, n: int) -> None:
        """How many evaluation errors to print: none are listed here."""

    def optimizeConst(self, flag: int) -> None:
        """Constant-term optimisation changes how fast, not what: nothing to do."""

    def setProfile(self, flag: bool = True) -> None:
        """Timing the steps prints times, which differ run to run: not done."""

    def getNPar(self) -> int:
        return len(self.params)

    def evalCounter(self) -> int:
        return int(self.minuit.nfcn) if self.minuit is not None else 0

    # -- the function Minuit sees -------------------------------------------------

    def _fcn(self, x: Any) -> float:
        """``RooMinimizerFcn::operator()``: set the parameters, evaluate, say so if verbose."""
        for par, value in zip(self.params, x):
            if par.getVal() != float(value):
                if self.verbose:
                    sys.stdout.write(f"{par.GetName()}={g(float(value), PRECISION[0])}, ")
                par.setVal(float(value))
        evaluate = getattr(self.function, "evaluate_nll", None)
        value = self._handled(evaluate() if evaluate is not None else float(self.function.getVal()))
        if self.verbose:
            sys.stdout.write(f"\nprevFCN = {g(value, 10)}  ")
            PRECISION[0] = 4  # RooFit leaves std::cout at four digits from here on
        return value

    def _handled(self, value: float) -> float:
        """``applyEvalErrorHandling``: a value that cannot be had becomes the worst seen, and worse."""
        if not math.isfinite(value) or value > 1e30:
            self.invalid += 1
            if self.eval_error_wall:
                badness = float(getattr(value, "badness", 0.0))
                return (self._max_fcn if math.isfinite(self._max_fcn) else 0.0) + (
                    self.recover_strength * badness
                )
            return value
        self._max_fcn = max(self._max_fcn, value)
        return value

    def _settings(self) -> Any:
        """A Minuit over the parameters as they are now: values, first steps, limits, fixed."""
        module = iminuit()
        values = [inside(p) for p in self.params]
        minuit = module.Minuit(self._fcn, np.asarray(values, dtype=np.float64),
                               name=tuple(p.GetName() for p in self.params))  # fmt: skip
        minuit.errordef = self.errordef
        minuit.tol = self.eps
        minuit.strategy = self.strategy
        minuit.print_level = 0
        minuit.errors = [first_step(p) for p in self.params]
        minuit.limits = [(p.getMin() if p.hasMin() else -np.inf, p.getMax() if p.hasMax() else np.inf)
                         for p in self.params]  # fmt: skip
        minuit.fixed = [p.isConstant() for p in self.params]
        return minuit

    # -- running Minuit -----------------------------------------------------------

    def minimize(self, type: str = "", alg: str = "") -> int:
        """``minimize(type, algorithm)``: MIGRAD - the default algorithm - from where the parameters are."""
        if type:
            self.setMinimizerType(type)
        if str(alg).lower() in ("simplex",):
            return self._run("MINIMIZE", lambda m: m.simplex(ncall=self.max_calls))
        return self._run("MINIMIZE", self._migrad)

    def migrad(self) -> int:
        return self._run("MIGRAD", self._migrad)

    def simplex(self) -> int:
        return self._run("SIMPLEX", lambda m: m.simplex(ncall=self.max_calls))

    def improve(self) -> int:
        return self._run("IMPROVE", self._migrad)

    def seek(self) -> int:
        return self._run("SEEK", self._migrad)

    def _migrad(self, minuit: Any) -> None:
        if self.print_level >= 1:
            print(f"Minuit2Minimizer: Minimize with max-calls {self.max_calls} convergence for edm < "
                  f"{g(self.eps)} strategy {self.strategy}")  # fmt: skip
        minuit.migrad(ncall=self.max_calls, iterate=1, use_simplex=False)

    def _run(self, label: str, run: Any) -> int:
        if not self.params:
            log(self, 4, "Minimization", "RooMinimizer::fitFCN(): FCN function has zero parameters")
            return -1
        log(self, INFO, "Minimization", "[fitFCN] No discrete parameters, performing continuous "
            "minimization only")  # fmt: skip
        self.minuit = self._settings()
        run(self.minuit)
        self.minuit_status = _status(self.minuit.fmin)
        self.status = self.minuit_status if self.minuit.fmin.is_valid else -1
        if self.print_level >= 1:
            self._print_results()
        self._back_propagate(minos=False)
        self.history.append((label, self.status))
        return self.status

    def hesse(self) -> int:
        """HESSE at the minimum MIGRAD found: the errors and the covariance, from second derivatives."""
        if self.minuit is None:
            log(self, WARNING, "Minimization", "RooMinimizer::hesse: Error, run Migrad before Hesse!")
            self.status = -1
            return self.status
        self.minuit.hesse(ncall=self.max_calls)
        failed = self.minuit.fmin.hesse_failed or not self.minuit.fmin.has_covariance
        self.status = getattr(self, "minuit_status", 0) + (100 if failed else 0)
        self._back_propagate(minos=False)
        self.history.append(("HESSE", self.status))
        return self.status

    def minos(self, params: Any = None) -> int:
        """MINOS for ``params``, or every free parameter: errors from where the likelihood rises by one half."""
        if self.minuit is None:
            log(self, WARNING, "Minimization", "RooMinimizer::minos: Error, run Migrad before Minos!")
            self.status = -1
            return self.status
        wanted = [p.GetName() for p in self.params if not p.isConstant()]
        if params is not None:
            names = {one.GetName() for one in as_list(params)}
            wanted = [name for name in wanted if name in names]
            if not wanted:
                return self.status
        for name in wanted:
            self._minos_one(name)
        self._back_propagate(minos=True)
        self.history.append(("MINOS", self.status))
        return self.status

    def _minos_one(self, name: str) -> None:
        index = [p.GetName() for p in self.params].index(name)
        if self.print_level >= 1:
            for side in ("LOWER", "UPPER"):
                print("*" * 102)
                print(f"Minuit2Minimizer::GetMinosError - Run MINOS {side} error for parameter #{index} : "
                      f"{name} using max-calls {self.max_calls}, tolerance {g(self.eps)}")  # fmt: skip
        try:
            self.minuit.minos(name, ncall=self.max_calls)
        except RuntimeError:  # iminuit refuses MINOS at an invalid minimum, as MnMinos does
            return
        error = self.minuit.merrors[name]
        if self.print_level >= 1:
            print(f"Minos: Lower error for parameter {name}  :  {g(error.lower)}")
            print(f"Minos: Upper error for parameter {name}  :  {g(error.upper)}")

    def _back_propagate(self, minos: bool) -> None:
        """``BackProp``: the parameters take Minuit's values and errors - and MINOS's, if it ran."""
        for index, par in enumerate(self.params):
            par.setVal(float(self.minuit.values[index]))
            par.setError(float(self.minuit.errors[index]) if not par.isConstant() else par.getError())
            error = self.minuit.merrors.get(par.GetName()) if minos else None
            if error is not None:
                par.setAsymError(float(error.lower), float(error.upper))
            elif not minos:
                par.removeAsymError()

    def _print_results(self) -> None:
        """``Minuit2Minimizer::PrintResults``."""
        fmin = self.minuit.fmin
        word = "Valid" if fmin.is_valid else "Invalid"
        print(f"Minuit2Minimizer : {word} minimum - status = {self.minuit_status}")
        print(f"FVAL  = {g(fmin.fval, 18)}")
        print(f"Edm   = {g(fmin.edm, 18)}")
        print(f"Nfcn  = {fmin.nfcn}")
        if not fmin.is_valid:
            return
        for index, par in enumerate(self.params):
            value = g(self.minuit.values[index])
            if self.minuit.fixed[index]:
                print(f"{par.GetName()}\t  = {value}\t (fixed)")
                continue
            error = g(self.minuit.errors[index])
            limited = "\t(limited)" if par.hasMin() or par.hasMax() else ""
            print(f"{par.GetName()}\t  = {value}\t +/-  {error}{limited}")

    # -- the result ---------------------------------------------------------------

    def save(self, name: Any = None, title: Any = None) -> Any:
        """``save``: a :class:`~xrdroot.roofit.fitting.result.RooFitResult` of the fit as it stands."""
        from .result import RooFitResult

        if self.minuit is None:
            log(self, WARNING, "Minimization", "RooMinimizer::save: Error, run minimization before!")
            return None
        found = RooFitResult(str(name) if name else self.function.GetName(),
                             str(title) if title else self.function.GetTitle())  # fmt: skip
        found.fill(self)
        return found

    def fitter(self) -> Any:
        return self

    def lastMinuitFit(self) -> Any:
        return self.save()

    def GetName(self) -> str:
        return "RooMinimizer"

    def ClassName(self) -> str:
        return "RooMinimizer"


def _status(fmin: Any) -> int:
    """``Minuit2Minimizer::ExamineMinimum``: the last of its checks that fails wins."""
    status = 0 if fmin.has_posdef_covar else 5
    for failed, code in ((fmin.has_made_posdef_covar, 1), (fmin.hesse_failed, 2),
                         (fmin.is_above_max_edm, 3), (fmin.has_reached_call_limit, 4)):  # fmt: skip
        if failed:
            status = code
    return 6 if status == 0 and not fmin.is_valid else status


def cov_quality(fmin: Any) -> int:
    """``Minuit2Minimizer::CovMatrixStatus``: 3 accurate, 2 forced positive, 1 approximate."""
    if fmin is None:
        return -1
    for flag, code in ((fmin.has_accurate_covar, 3), (fmin.has_made_posdef_covar, 2),
                       (fmin.has_posdef_covar, 1), (fmin.has_covariance, 0)):  # fmt: skip
        if flag:
            return code
    return -1


def as_set(items: Any) -> RooArgSet:
    return RooArgSet(as_list(items))
