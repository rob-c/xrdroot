"""``AsymptoticCalculator``: hypothesis tests from the asymptotic formulae - no toys.

The profile likelihood ratio's distribution under each hypothesis is, for
many events, a known function of the ratio on the data and on the Asimov
data set - the data a model expects at the alternate's value of the
parameter of interest (Cowan, Cranmer, Gross, Vitells, arXiv:1007.1727).
So a test needs four fits: unconditional and conditional, on the data and
on the Asimov data. The one-sided (``qmu``), one-sided discovery (``q0``)
and two-sided (``tmu``) formulae, and their ``tilde`` variants for a
parameter bounded at the alternate, are RooStats' own, step for step.
"""

from __future__ import annotations

import math
from typing import Any

from ..roofit.cmdargs import RooCmdArg
from ..roofit.collections import RooArgSet
from ..roofit.messages import ERROR, FATAL, PROGRESS, WARNING, log, service
from ..roofit.printing import g
from .asimov import _LEVEL
from .config import CONFIG, NLLOffsetMode
from .utils import RemoveConstantParameters

__all__ = ["evaluate_nll"]

#: ``fgPrintLevel``, which the Asimov data's maker reads too: 1, the calculator's progress.
PRINT_LEVEL = _LEVEL


def _free(items: Any) -> RooArgSet:
    found = RooArgSet(list(items))
    RemoveConstantParameters(found)
    return found


def _fixed_poi(nll: Any, poi: Any) -> RooArgSet:
    """The likelihood's own copy of the parameter of interest, fixed at the value asked for."""
    fixed = RooArgSet()
    if poi is None or not len(poi):
        return fixed
    first = next(iter(poi))
    found = nll.getVariables().find(first.GetName())
    if found is not None and not found.isConstant():
        found.setVal(first.getVal())
        found.setConstant()
        fixed.add(found)
    if len(poi) > 1:
        log(None, WARNING, "InputArguments", "Model with more than one POI are not supported - "
            "ignore extra parameters, consider only first one")  # fmt: skip
    return fixed


def evaluate_nll(model: Any, data: Any, poi: Any = None) -> float:
    """``EvaluateNLL``: the likelihood minimised - with the parameter of interest fixed, if given
    - RooFit's messages held back below fatal ones unless the print level is two or more."""
    before = service().globalKillBelow()
    if PRINT_LEVEL[0] < 2:
        service().setGlobalKillBelow(FATAL)
    try:
        constrained = _free(model.GetPdf().getParameters(data))
        nll = model.createNLL(data, RooCmdArg("Constrain", constrained),
                              RooCmdArg("Offset", CONFIG.useLikelihoodOffset))  # fmt: skip
        fixed = _fixed_poi(nll, poi)
        free = _free(nll.getVariables())
        value = float(nll.getVal()) if not len(free) else _minimum(nll)
        for one in fixed:
            one.setConstant(False)
        _said(value, poi, bool(len(free)))
        return value
    finally:
        service().setGlobalKillBelow(before)


def _said(value: float, poi: Any, fitted: bool) -> None:
    if PRINT_LEVEL[0] <= 0:
        return
    text = f"AsymptoticCalculator::EvaluateNLL -  value = {g(value)}"
    if poi is not None:
        text += f" for poi fixed at = {g(next(iter(poi)).getVal())}"
    log(None, PROGRESS, "Eval", text + ("\tfit time : 0 s (real) 0 s (cpu)" if fitted else ""))


def _minimum(nll: Any) -> float:
    """The fit of ``EvaluateNLL``: MIGRAD at a tolerance of at least one, tried again - after a
    scan, with strategy one, improved - while it fails; NaN if it never succeeds."""
    from ..fit.defaults import default, minimizer_algo
    from ..roofit.fitting.minimizer import RooMinimizer

    strategy = int(default("Strategy"))
    minim = RooMinimizer(nll)
    minim.setStrategy(strategy)
    minim.setEvalErrorWall(CONFIG.useEvalErrorWall)
    tolerance = max(float(default("Tolerance")), 1.0)
    minim.setEps(tolerance)
    minim.setPrintLevel(PRINT_LEVEL[0] - 1)
    minim.optimizeConst(2)
    kind, algorithm = "", minimizer_algo()
    log(None, PROGRESS, "Eval", f"AsymptoticCalculator::EvaluateNLL  ........ using {kind} / "
        f"{algorithm} with strategy  {strategy} and tolerance {g(tolerance)}")  # fmt: skip
    status, tries = -1, 1
    while tries <= 4:
        status = minim.minimize(kind, algorithm)
        if status >= 0:
            break
        if tries == 1:
            log(None, WARNING, "Minimization", "    ----> Doing a re-scan first")
            minim.minimize(kind, "Scan")
        if tries == 2 and strategy == 0:
            log(None, WARNING, "Minimization", "    ----> trying with strategy = 1")
            minim.setStrategy(1)
        elif tries == 2:
            tries += 1
        if tries == 3:
            log(None, WARNING, "Minimization", "    ----> trying with improve")
            kind, algorithm = "Minuit", "migradimproved"
        tries += 1
    minim.optimizeConst(False)
    if status < 0:
        log(None, ERROR, "Fitting", "FIT FAILED !- return a NaN NLL ")
        return math.nan
    result = minim.save()
    return float(nll.getVal()) if NLLOffsetMode() == "initial" else float(result.minNll())


def make_asimov_data(data: Any, model: Any, values: Any, globals_out: Any, gen_poi: Any = None
                     ) -> Any:  # fmt: skip
    """``MakeAsimovData(data, model, poiValues, ...)``: the nuisance parameters at their best
    fit to ``data`` with the parameter of interest fixed at ``values``, then the Asimov data at
    those - and at ``gen_poi``, if given."""
    from ..roofit.messages import INFO
    from .utils import SetAllConstant

    poi = RooArgSet(list(model.GetParametersOfInterest()))
    poi.assign(values)
    fixed = RooArgSet()
    for par in poi:
        par.setConstant()
        if PRINT_LEVEL[0] > 0:
            log(None, INFO, "Generation", f"MakeAsimov: Setting poi {par.GetName()} to a constant "
                f"value = {g(par.getVal())}")  # fmt: skip
        fixed.add(par)
    if _has_free(model, data):
        _conditional_fit(model, data)
    SetAllConstant(fixed, False)
    params = _free(model.GetPdf().getParameters(data))
    if gen_poi is not None:
        params.assign(gen_poi)
    return make_asimov_nominal(model, params, globals_out)


def _has_free(model: Any, data: Any) -> bool:
    nuisance = model.GetNuisanceParameters()
    if nuisance is not None:
        return bool(len(_free(nuisance)))
    return any(not one.isConstant() for one in model.GetPdf().getParameters(data)
               if one.InheritsFrom("RooRealVar"))  # fmt: skip


def _conditional_fit(model: Any, data: Any) -> None:
    """The fit for the best nuisance values: no HESSE, RooFit's messages held back."""
    from ..fit.defaults import default, minimizer_algo

    level = int(default("PrintLevel"))
    if PRINT_LEVEL[0] > 0:
        log(None, PROGRESS, "Generation", "MakeAsimov: doing a conditional fit for finding best "
            "nuisance values ")  # fmt: skip
        level = PRINT_LEVEL[0]
    before = service().globalKillBelow()
    if PRINT_LEVEL[0] < 2:
        service().setGlobalKillBelow(FATAL)
    nuisance = model.GetNuisanceParameters()
    constrained = _free(nuisance) if nuisance is not None else RooArgSet()
    try:
        model.fitTo(data, RooCmdArg("Minimizer", "", minimizer_algo()),
                    RooCmdArg("Strategy", int(default("Strategy"))),
                    RooCmdArg("PrintLevel", level - 1), RooCmdArg("Hesse", False),
                    RooCmdArg("Constrain", constrained),
                    RooCmdArg("Offset", CONFIG.useLikelihoodOffset),
                    RooCmdArg("EvalErrorWall", CONFIG.useEvalErrorWall))  # fmt: skip
        if PRINT_LEVEL[0] > 0:  # said before the messages are let through again: never seen
            log(None, PROGRESS, "Generation", "fit time : 0 s (real) 0 s (cpu)")
    finally:
        if PRINT_LEVEL[0] < 2:
            service().setGlobalKillBelow(before)


def make_asimov_nominal(model: Any, values: Any, globals_out: Any) -> Any:
    """``MakeAsimovData(model, allParamValues, asimovGlobObs)``: the Asimov data at ``values``,
    and in ``globals_out`` each constraint's global observable at its nuisance parameter."""
    from ..roofit.messages import INFO
    from .asimov import GenerateAsimovData
    from .utils import SetAllConstant

    if len(values):
        model.GetPdf().getVariables().assign(values)
    asimov = GenerateAsimovData(model.GetPdf(), model.GetObservables())
    if PRINT_LEVEL[0] > 0:  # the set printed on its own line, the message where it goes
        log(None, INFO, "Generation", "Generated Asimov data for observables ")
        RooArgSet(list(model.GetObservables())).Print()
    gobs_set = model.GetGlobalObservables()
    if gobs_set is None or not len(gobs_set):
        return asimov
    gobs = RooArgSet(list(gobs_set))
    SetAllConstant(gobs, True)
    saved = gobs.snapshot()
    nuisance = RooArgSet(list(model.GetNuisanceParameters() or []))
    if not len(nuisance):
        log(None, WARNING, "Generation", "AsymptoticCalculator::MakeAsimovData: model does not "
            "have nuisance parameters but has global observables set global observables to model "
            "values ")  # fmt: skip
        globals_out.assign(gobs)
        return asimov
    from .asimovglobs import set_global_observables

    set_global_observables(model, gobs, nuisance)
    globals_out.removeAll()
    SetAllConstant(gobs, True)
    globals_out.add(gobs.snapshot())
    gobs.assign(saved)
    if PRINT_LEVEL[0] > 0:
        log(None, INFO, "Generation", "Generated Asimov data for global observables ")
        if PRINT_LEVEL[0] == 1:
            gobs.Print()
    return asimov
