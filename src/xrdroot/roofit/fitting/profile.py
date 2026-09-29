"""``RooProfileLL``: a likelihood profiled over its nuisance parameters, as a function of the rest.

``nll.createProfile(poi)`` is ``-log L`` minimised over every parameter
but the parameters of interest, less its absolute minimum: each value is a
MIGRAD run with the parameters of interest held at their values, started
from the absolute minimum's nuisance values, as ``RooProfileLL::evaluate``
runs it - a ``.`` of progress each time, and the messages RooFit prints the
first time, when it finds that minimum. A value is kept until a parameter
of the likelihood changes, as RooFit's dirty flags keep it, so asking again
costs no fit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..collections import RooArgSet, as_list
from ..messages import ERROR, INFO, PROGRESS, log, log_plain, service
from ..printing import g
from ..real import RooAbsReal

__all__ = ["RooProfileLL", "create_profile"]


class RooProfileLL(RooAbsReal):
    """``-log L`` minimised over the nuisance parameters, less its minimum over all of them."""

    def __init__(self, name: Any, title: Any, nll: Any, observables: Any) -> None:
        super().__init__(name, title)
        wanted = {one.GetName() for one in as_list(observables)}
        self._nll = self._proxy("input", nll)
        params = list(nll.getParameters())
        poi = [p for p in params if p.GetName() in wanted]
        self._obs = self._list_proxy("paramOfInterest", poi)
        rest = [p for p in params if p.GetName() not in wanted]
        self._par = self._list_proxy("nuisanceParam", rest)
        self._minimizer: Any = None
        self._valid = False
        self._abs_min = 0.0
        self._param_abs_min: list[Any] = []
        self._obs_abs_min: list[Any] = []
        self._param_fixed: dict[str, bool] = {}
        self._start_from_min = True
        self._cache: Any = None
        self._neval = 0
        #: A plotted copy's own copies of the nuisance parameters, which its fits never touch.
        self._detached: list[Any] | None = None

    def _nuisances(self) -> list[Any]:
        """The nuisance parameters as this profile holds them: the likelihood's, or a plotted
        copy's own."""
        return list(self._par) if self._detached is None else self._detached

    def nll(self) -> Any:
        return self._nll

    def getVariables(self, stripDisconnected: bool = True) -> RooArgSet:
        """The likelihood's parameters - constant or not - but none of the data's observables."""
        return RooArgSet(self._nll.getParameters()).sorted_copy()

    def _parameters(self, observables: Any) -> RooArgSet:
        """The likelihood's parameters: RooFit finds them through it, its value server."""
        return RooArgSet(self._nll.getParameters(observables))

    def setAlwaysStartFromMin(self, flag: bool) -> None:
        self._start_from_min = bool(flag)

    def alwaysStartFromMin(self) -> bool:
        return self._start_from_min

    def minimizer(self) -> Any:
        return self._minimizer

    def bestFitParams(self) -> RooArgSet:
        self._validate()
        return RooArgSet(self._param_abs_min)

    def bestFitObs(self) -> RooArgSet:
        self._validate()
        return RooArgSet(self._obs_abs_min)

    def createProfile(self, paramsOfInterest: Any) -> Any:
        return self._nll.createProfile(paramsOfInterest)

    def plotOn(self, frame: Any, *args: Any, **kwargs: Any) -> Any:
        """``RooAbsReal::plotOn``, which draws a deep copy of the profile - a fresh minimizer, a
        minimum to find again - with the plotted parameter put back as it was afterwards.

        The copy RooFit draws minimises the likelihood's own parameters but
        keeps copies of its nuisance parameters of its own, so what it assigns
        them from the minimum it copied does not reach the fit: each fit starts
        where the last left the nuisance parameters, and they are left where
        the last fit put them - and the parameter of interest, which the copy
        fixes, without its error.
        """
        before = [(p, p.getVal()) for p in self._obs]
        copy = self._plotted_copy()
        try:
            copy.getVal()  # RooRealIntegral's "kludge": the projection is evaluated once, as it is

            return RooAbsReal.plotOn(copy, frame, *args, **kwargs)
        finally:
            for par, value in before:
                par.setVal(value)

    def _plotted_copy(self) -> RooProfileLL:
        """``RooProfileLL``'s copy constructor as plotting makes it: the minima it knew, but not
        that they hold, and nuisance parameters of its own (:meth:`plotOn`)."""
        made = RooProfileLL(self._name, self._title, self._nll, list(self._obs))
        made._param_abs_min = [p.clone(p.GetName()) for p in self._param_abs_min]
        made._obs_abs_min = [p.clone(p.GetName()) for p in self._obs_abs_min]
        made._start_from_min = self._start_from_min
        made._detached = [p.clone(p.GetName()) for p in self._par]
        return made

    # -- values -------------------------------------------------------------------

    def compute(self, ctx: Any) -> Any:
        """The profile at the values ``ctx`` gives the parameters of interest - one fit per
        point, in order, for a column of them - or at their own values."""
        given = {one.GetName(): ctx[one.GetName()] for one in self._obs if one.GetName() in ctx}
        if not given or all(np.ndim(value) == 0 for value in given.values()):
            for name, value in given.items():
                self._obs.find(name).setVal(float(value))
            return self._value()
        size = max(np.size(value) for value in given.values())
        found = np.empty(size)
        for index in range(size):
            for name, value in given.items():
                self._obs.find(name).setVal(float(np.broadcast_to(value, (size,))[index]))
            found[index] = self._value()
        return found

    def getVal(self, nset: Any = None) -> float:
        return self._value()

    def _state(self) -> tuple[Any, ...]:
        """What the value depends on: every parameter of the likelihood, and whether it is
        free."""
        return tuple((p.GetName(), p.getVal(), p.isConstant()) for p in self._nll.getParameters())

    def _value(self) -> float:
        if self._cache is not None and self._cache[0] == self._state():
            return float(self._cache[1])
        found = self.evaluate()
        self._cache = (self._state(), found)
        return found

    def _initialize(self) -> None:
        from .minimizer import RooMinimizer

        log(self, INFO, "Minimization", f"RooProfileLL::evaluate({self._name}) Creating instance "
            "of MINUIT")  # fmt: skip
        silent = service().silentMode()
        service().setSilentMode(True)
        self._minimizer = RooMinimizer(self._nll)
        service().setSilentMode(silent)

    def evaluate(self) -> float:
        """``RooProfileLL::evaluate``: MIGRAD with the parameters of interest fixed."""
        if self._minimizer is None:
            self._initialize()
        before = [(p, p.getVal(), p.isConstant()) for p in self._obs]
        self._validate()
        for one in self._obs:
            one.setConstant(True)
        log_plain(self, PROGRESS, "Eval", ".")
        if self._start_from_min:
            RooArgSet(self._nuisances()).assign(self._param_abs_min)
        self._minimizer.zeroEvalCount()
        self._minimizer.migrad()
        self._neval = self._minimizer.evalCounter()
        for one, value, constant in before:
            one.setVal(value)
            one.setConstant(constant)
        return float(self._nll.getVal()) - self._abs_min

    def _validate(self) -> None:
        """``validateAbsMin``: the minimum over everything, found again when a nuisance
        parameter's constness has changed."""
        for par in self._nuisances() if self._valid else ():
            if self._param_fixed.get(par.GetName()) != par.isConstant():
                was = "fixed" if self._param_fixed.get(par.GetName()) else "floating"
                now = "fixed" if par.isConstant() else "floating"
                log(self, INFO, "Minimization", f"RooProfileLL::evaluate({self._name}) constant "
                    f"status of parameter {par.GetName()} has changed from {was} to {now}, "
                    "recalculating absolute minimum")  # fmt: skip
                self._valid = False
                break
        if not self._valid:
            self._find_minimum()

    def _find_minimum(self) -> None:
        log(self, INFO, "Minimization", f"RooProfileLL::evaluate({self._name}) determining minimum "
            "likelihood for current configurations w.r.t all observable")  # fmt: skip
        if self._minimizer is None:
            self._initialize()
        start = [(p, p.getVal()) for p in self._obs]
        RooArgSet(self._nuisances()).assign(self._param_abs_min)
        RooArgSet(list(self._obs)).assign(self._obs_abs_min)
        for one in self._obs:
            one.setConstant(False)
        self._minimizer.migrad()
        self._abs_min = float(self._nll.getVal())
        self._valid = True
        mine = self._nuisances()
        self._param_abs_min = [p.clone(p.GetName()) for p in mine if not p.isConstant()]
        known = {p.GetName() for p in self._obs_abs_min}  # a copy's: addClone will not add them
        for par in self._obs:
            if par.GetName() in known:
                log(None, ERROR, "InputArguments", "RooArgSet::checkForDup: ERROR argument with "
                    f"name {par.GetName()} is already in this set")  # fmt: skip
            else:
                self._obs_abs_min.append(par.clone(par.GetName()))
        self._param_fixed = {p.GetName(): p.isConstant() for p in mine}
        at = ", ".join(f"{p.GetName()}={g(p.getVal())}" for p in self._obs)
        log(self, INFO, "Minimization", f"RooProfileLL::evaluate({self._name}) minimum found at "
            f"({at})")  # fmt: skip
        for one, value in start:
            one.setVal(value)


def create_profile(nll: Any, poi: Any) -> RooProfileLL:
    """``createProfile(poi)``: named ``<nll>_Profile[a,b]``, as RooFit names it."""
    wrapper = getattr(nll, "wrapper_name", None)
    base, title = (wrapper, wrapper) if wrapper else (nll.GetName(), nll.GetTitle())
    name = f"{base}_Profile[" + ",".join(one.GetName() for one in as_list(poi)) + "]"
    return RooProfileLL(name, f"Profile of {title}", nll, poi)
