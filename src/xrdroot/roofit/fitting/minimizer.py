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
from typing import Any

import numpy as np

from ...fit.minuit import iminuit
from .. import cout
from ..collections import RooArgList, RooArgSet, as_list
from ..messages import ERROR, INFO, WARNING, log, log_plain
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
        return float(step if step != 0 else 0.1 * (high - low))
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
        self.evaluations = 0
        self._offset = 0.0
        self.print_eval_errors = 10
        self._max_fcn = -math.inf
        self._last: list[float] = []
        #: ``applyCovarianceMatrix``'s matrix, which a saved result then carries instead.
        self.external_covariance: Any = None

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
        """How many evaluation errors of each object to list when a point fails; -1 for none."""
        self.print_eval_errors = int(n)

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
                    cout.write(f"{par.GetName()}={g(float(value), PRECISION[0])}, ")
                par.setVal(float(value))
        value = self._handled(self._evaluate())
        self.evaluations += 1
        if self.verbose:
            cout.write(f"\nprevFCN = {g(value, 10)}  ")
            PRECISION[0] = 4  # RooFit leaves std::cout at four digits from here on
        return value

    def _evaluate(self) -> float:
        """The function's value, its evaluation errors collected as RooFit collects them."""
        from .. import evalerrors

        evalerrors.clear()
        evalerrors.collecting(True)
        try:
            evaluate = getattr(self.function, "evaluate_nll", None)
            return evaluate() if evaluate is not None else float(self.function.getVal())
        finally:
            evalerrors.collecting(False)

    def _handled(self, value: float) -> float:
        """``applyEvalErrorHandling``: a value that cannot be had becomes the worst seen, and
        worse."""
        from .. import evalerrors

        if not math.isfinite(value) or evalerrors.count() > 0 or value > 1e30:
            self._print_errors()
            evalerrors.clear()
            self.invalid += 1
            if self.eval_error_wall:
                from ..nanpack import unpack

                badness = float(unpack(value))
                return (self._max_fcn if math.isfinite(self._max_fcn) else 0.0) + (
                    self.recover_strength * badness
                )
            return value
        if self.evaluations > 0 and self.evaluations == self.invalid:
            self._offset = -value
        value += self._offset
        self._max_fcn = max(self._max_fcn, value)
        return value

    def _print_errors(self) -> None:
        """``RooAbsMinimizerFcn::printEvalErrors``: why this point failed, as a warning."""
        from .. import evalerrors

        if self.print_eval_errors < 0:
            return
        if self.eval_error_wall:
            text = (
                "RooAbsMinimizerFcn: Minimized function has error status.\nReturning maximum FCN "
                "so "
                f"far ({g(self._max_fcn, 6)}) to force MIGRAD to back out of this region. Error "
                "log "
                "follows.\n"
            )
        else:
            text = "RooAbsMinimizerFcn: Minimized function has error status but is ignored.\n"
        text += "Parameter values: " + "".join(
            f"\t{p.GetName()}={g(p.getVal(), 6)}" for p in self.params
        )
        text += "\n" + evalerrors.text(self.print_eval_errors)
        log_plain(self, WARNING, "Minimization", text + "\n")

    def _step(self, par: Any) -> float:
        """The first step, with RooFit's word - when verbose - for a parameter that had no error."""
        step = first_step(par)
        if self.verbose and par.getError() <= 0:
            log(
                self,
                WARNING,
                "Minimization",
                "RooAbsMinimizerFcn::synchronize: WARNING: no initial "
                f"error estimate available for {par.GetName()}: using {g(step)}",
            )
        return step

    def _settings(self) -> Any:
        """A Minuit over the parameters as they are now: values, first steps, limits, fixed."""
        module = iminuit()
        values = [inside(p) for p in self.params]
        minuit = module.Minuit(
            self._fcn,
            np.asarray(values, dtype=np.float64),
            name=tuple(p.GetName() for p in self.params),
        )
        minuit.errordef = self.errordef
        minuit.tol = self.eps
        minuit.strategy = self.strategy
        minuit.print_level = 0
        minuit.errors = [self._step(p) for p in self.params]
        minuit.limits = [
            (p.getMin() if p.hasMin() else -np.inf, p.getMax() if p.hasMax() else np.inf)
            for p in self.params
        ]
        minuit.fixed = [p.isConstant() for p in self.params]
        return minuit

    # -- running Minuit -----------------------------------------------------------

    def minimize(self, type: str = "", alg: str = "") -> int:
        """``minimize(type, algorithm)``: MIGRAD - the default algorithm - from where the parameters
        are."""
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
            cout.line(
                f"Minuit2Minimizer: Minimize with max-calls {self.max_calls} convergence for edm < "
                f"{g(self.eps)} strategy {self.strategy}"
            )
        minuit.migrad(ncall=self.max_calls, iterate=1, use_simplex=False)

    def _run(self, label: str, run: Any) -> int:
        if not self.params:
            log(self, 4, "Minimization", "RooMinimizer::fitFCN(): FCN function has zero parameters")
            return -1
        self.minuit = self._settings()
        log(
            self,
            INFO,
            "Minimization",
            "[fitFCN] No discrete parameters, performing continuous minimization only",
        )
        run(self.minuit)
        self.minuit_status = _status(self.minuit.fmin)
        self.status = self.minuit_status if self.minuit.fmin.is_valid else -1
        if self.print_level >= 1:
            self._print_results()
        self._back_propagate(minos=False)
        self.history.append((label, self.status))
        return self.status

    def hesse(self) -> int:
        """HESSE at the minimum MIGRAD found: the errors and the covariance, from second
        derivatives."""
        if self.minuit is None:
            log(
                self,
                WARNING,
                "Minimization",
                "RooMinimizer::hesse: Error, run Migrad before Hesse!",
            )
            self.status = -1
            return self.status
        self.minuit.hesse(ncall=self.max_calls)
        self.status = getattr(self, "minuit_status", 0) + 100 * _hesse_code(self.minuit.fmin)
        if self.status >= 100:
            log(
                self,
                ERROR,
                "Minimization",
                "RooMinimizer::calculateHessErrors() Error when calculating Hessian",
            )
        self._back_propagate(minos=False)
        self.history.append(("HESSE", self.status))
        return self.status

    def minos(self, params: Any = None) -> int:
        """MINOS for ``params``, or every free parameter: errors from where the likelihood rises by
        one half."""
        if self.minuit is None:
            log(
                self,
                WARNING,
                "Minimization",
                "RooMinimizer::minos: Error, run Migrad before Minos!",
            )
            self.status = -1
            return self.status
        wanted = self._minos_wanted(params)
        if not wanted:
            return self.status
        for name in wanted:
            self._minos_one(name)
        self._back_propagate(minos=True)
        self.history.append(("MINOS", self.status))
        return self.status

    def _minos_wanted(self, params: Any) -> list[str]:
        """The free parameters MINOS is run for: all of them, or those of ``params``."""
        wanted = [p.GetName() for p in self.params if not p.isConstant()]
        if params is None:
            return wanted
        names = {one.GetName() for one in as_list(params)}
        return [name for name in wanted if name in names]

    def _minos_one(self, name: str) -> None:
        index = [p.GetName() for p in self.params].index(name)
        if self.print_level >= 1:
            for side in ("LOWER", "UPPER"):
                cout.line("*" * 102)
                cout.line(
                    f"Minuit2Minimizer::GetMinosError - Run MINOS {side} error for parameter "
                    f"#{index} : "
                    f"{name} using max-calls {self.max_calls}, tolerance {g(self.eps)}"
                )
        try:
            self.minuit.minos(name, ncall=self.max_calls)
        except RuntimeError:  # iminuit refuses MINOS at an invalid minimum, as MnMinos does
            return
        error = self.minuit.merrors[name]
        if self.print_level >= 1:
            cout.line(f"Minos: Lower error for parameter {name}  :  {g(error.lower)}")
            cout.line(f"Minos: Upper error for parameter {name}  :  {g(error.upper)}")

    def _back_propagate(self, minos: bool) -> None:
        """``BackProp``: the parameters take Minuit's values and errors - and MINOS's, if it ran."""
        for index, par in enumerate(self.params):
            value = float(self.minuit.values[index])
            if self.verbose and par.getVal() != value:  # SetPdfParamVal says so, as in a call
                cout.write(f"{par.GetName()}={g(value, PRECISION[0])}, ")
            par.setVal(value)
            par.setError(
                float(self.minuit.errors[index]) if not par.isConstant() else par.getError()
            )
            error = self.minuit.merrors.get(par.GetName()) if minos else None
            if error is not None:
                par.setAsymError(float(error.lower), float(error.upper))
            elif not minos:
                par.removeAsymError()

    def _print_results(self) -> None:
        """``Minuit2Minimizer::PrintResults``."""
        fmin = self.minuit.fmin
        word = "Valid" if fmin.is_valid else "Invalid"
        cout.line(f"Minuit2Minimizer : {word} minimum - status = {self.minuit_status}")
        digits = 18 if fmin.is_valid else PRECISION[0]  # an invalid minimum at cout's precision
        cout.line(f"FVAL  = {g(fmin.fval, digits)}")
        cout.line(f"Edm   = {g(fmin.edm, digits)}")
        cout.line(f"Nfcn  = {fmin.nfcn}")
        if not fmin.is_valid:
            return
        for index, par in enumerate(self.params):
            value = g(self.minuit.values[index])
            if self.minuit.fixed[index]:
                cout.line(f"{par.GetName()}\t  = {value}\t (fixed)")
                continue
            error = g(self.minuit.errors[index])
            limited = "\t(limited)" if par.hasMin() or par.hasMax() else ""
            cout.line(f"{par.GetName()}\t  = {value}\t +/-  {error}{limited}")

    def applyCovarianceMatrix(self, matrix: Any) -> None:
        """``applyCovarianceMatrix``: the floating parameters' errors, and the covariance a saved
        result carries, from ``matrix`` rather than from HESSE."""
        found = np.array([[matrix[i][j] for j in range(len(self.params))]
                          for i in range(len(self.params))], dtype=np.float64)  # fmt: skip
        self.external_covariance = found
        for index, par in enumerate(self.params):
            par.setError(math.sqrt(found[index, index]))

    # -- the result ---------------------------------------------------------------

    def save(self, name: Any = None, title: Any = None) -> Any:
        """``save``: a :class:`~xrdroot.roofit.fitting.result.RooFitResult` of the fit as it
        stands."""
        from .result import RooFitResult

        if self.minuit is None:
            log(
                self, WARNING, "Minimization", "RooMinimizer::save: Error, run minimization before!"
            )
            return None
        found = RooFitResult(
            str(name) if name else self.function.GetName(),
            str(title) if title else self.function.GetTitle(),
        )
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
    for failed, code in (
        (fmin.has_made_posdef_covar, 1),
        (fmin.hesse_failed, 2),
        (fmin.is_above_max_edm, 3),
        (fmin.has_reached_call_limit, 4),
    ):
        if failed:
            status = code
    return 6 if status == 0 and not fmin.is_valid else status


def cov_quality(fmin: Any) -> int:
    """``Minuit2Minimizer::CovMatrixStatus``: 3 accurate, 2 forced positive, 1 approximate."""
    if fmin is None:
        return -1
    for flag, code in (
        (fmin.has_accurate_covar, 3),
        (fmin.has_made_posdef_covar, 2),
        (fmin.has_posdef_covar, 1),
        (fmin.has_covariance, 0),
    ):
        if flag:
            return code
    return -1


def as_set(items: Any) -> RooArgSet:
    return RooArgSet(as_list(items))


def _hesse_code(fmin: Any) -> int:
    """``Minuit2Minimizer::Hesse``'s code for a HESSE that gave no covariance: 3 the matrix is
    not positive definite, else 1 it failed, else 4; 0 for one that gave it.

    A failed HESSE leaves Minuit2's user state without a covariance, though
    iminuit still shows the one it had. ``Hesse`` tests the matrix after the
    failure (``if failed 1; if not inverted 2; else if not positive 3``), so
    a failure over a matrix that is not positive definite scores 3.
    """
    if fmin.has_covariance and not fmin.hesse_failed:
        return 0
    if not fmin.has_posdef_covar:
        return 3
    return 1 if fmin.hesse_failed else 4
