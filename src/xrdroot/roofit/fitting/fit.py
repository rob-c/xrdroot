"""``RooAbsPdf::fitTo``: build the likelihood, minimise it with Minuit, say what RooFit says.

A fit is ``FitHelpers::fitTo``'s steps in order - the likelihood made from
the options that shape it (``Range``, ``Extended``, ``ConditionalObservables``,
``Constrain``...), a :class:`~.minimizer.RooMinimizer` set up from the
options that drive it (``PrintLevel``, ``Strategy``, ``Minos``,
``SumW2Error``...), MIGRAD, HESSE, MINOS if asked, and a
:class:`~.result.RooFitResult` if ``Save()`` - with each ``[#1] INFO`` line
RooFit prints on the way, so the output reads as ROOT's.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .. import copies
from ..cmdargs import Commands, commands
from ..collections import as_list
from ..messages import ERROR, INFO, WARNING, log
from .binned import binned_part
from .minimizer import RooMinimizer
from .nll import RooNLLVar

__all__ = ["fit_to", "nll_options"]

#: Whether the "which CPU library" line has been printed: RooFit says it once a session.
_SAID_LIBRARY = [False]


def _extended(pdf: Any, options: Commands) -> bool:
    """``interpretExtendedCmdArg``: extended if asked, or by default if the density can be."""
    if "Extended" in options:
        return bool(options.get("Extended"))
    if pdf.canBeExtended():
        log(
            pdf,
            INFO,
            "Minimization",
            "p.d.f. provides expected number of events, including extended term in likelihood.",
        )
        return True
    return False


def _range(options: Commands) -> Any:
    found = options.args("Range")
    if not found:
        return None
    return found[0] if isinstance(found[0], str) else None


def _constraints(
    pdf: Any, data: Any, options: Commands
) -> tuple[list[Any], frozenset[str], dict[str, float]]:
    """``createConstraintTerm``: the constraint terms - those the density carries on the
    parameters ``Constrain`` names (or on any of its parameters), and external ones - and the
    parameters they are normalised over, said as RooFit says them."""
    given = options.get("Constrain")
    params = list(as_list(given)) if given is not None else list(pdf.getParameters(data))
    observables = frozenset(one.GetName() for one in data.get())
    carried = getattr(pdf, "constraint_terms", None)
    found = carried(observables, params, given is None) if carried is not None else []
    found += list(as_list(options.get("ExternalConstraints")))
    names = sorted(one.GetName() for one in params)
    if not found:
        return found, frozenset(names), {}
    names = _without_exclusive(pdf, found, names, observables)
    log(pdf, INFO, "Minimization", " Including the following constraint terms in "
        f"minimization: ({','.join(one.GetName() for one in found)})")  # fmt: skip
    over, values = _global_observables(pdf, data, options, names)
    return found, over, values


def _without_exclusive(pdf: Any, constraints: list[Any], names: list[str],
                       observables: frozenset[str]) -> list[str]:  # fmt: skip
    """``getAllConstraints``' last step: the parameters only the constraints have - their
    nominal values, say - taken out of those the constraints are normalised over."""
    mine: set[str] = set()
    for one in constraints:
        mine |= set(one.dependents())
    shared = _reachable(pdf, constraints) - observables
    return [n for n in names if not (n in mine and n not in shared)]


def _reachable(pdf: Any, stops: list[Any]) -> set[str]:
    """The variables below ``pdf`` on a path that passes through none of ``stops``."""
    seen: set[int] = set()
    found: set[str] = set()
    todo = [pdf]
    while todo:
        node = todo.pop()
        if id(node) in seen or any(node is stop for stop in stops):
            continue
        seen.add(id(node))
        servers = node.servers()
        if not servers and node.isFundamental():
            found.add(node.GetName())
        todo.extend(servers)
    return found


def _from_data(
    pdf: Any, names: list[str] | None, values: dict[str, float]
) -> tuple[frozenset[str], dict[str, float]]:
    """Global observables the dataset carries: all of them, or those named - their values the
    dataset's."""
    if names is None:
        names = list(values)
        text = (
            "The following global observables have been automatically defined according to "
            f"the dataset which also provides their values: ({','.join(names)})"
        )
    else:
        common = ",".join(n for n in values if n in names)
        text = (
            f"The following global observables have been defined: ({','.join(names)}), with "
            f"the values of ({common}) obtained from the dataset and the other values from the "
            "model."
        )
    log(pdf, INFO, "Minimization", text)
    return frozenset(names), {k: v for k, v in values.items() if k in names}


def _global_observables(
    pdf: Any, data: Any, options: Commands, params: list[str]
) -> tuple[frozenset[str], dict[str, float]]:
    """What the constraints are normalised over - the global observables, or else the
    parameters - and the global observables' values taken from the data, said as RooFit does."""
    given = options.get("GlobalObservables")
    stored = (
        data.getGlobalObservables()
        if options.get("GlobalObservablesSource", 0, "data") == "data"
        else None
    )
    names = [one.GetName() for one in as_list(given)] if given is not None else None
    if stored:
        return _from_data(pdf, names, {one.GetName(): one.getVal() for one in stored})
    if names is not None:
        log(pdf, INFO, "Minimization", "The following global observables have been defined and "
            f"their values are taken from the model: ({','.join(names)})")  # fmt: skip
        return frozenset(names), {}
    log(pdf, INFO, "Minimization", "The global observables are not defined , normalize "
        f"constraints with respect to the parameters ({','.join(params)})")  # fmt: skip
    return frozenset(params), {}


def _fit_range_attributes(pdf: Any, data: Any, rng: Any) -> None:
    """``resetFitrangeAttributes``: ranges ``fit_nll_<pdf>_<data>`` of the fit's, for plotting
    in."""
    pdf.removeStringAttribute("fitrange")
    if not rng:
        return
    base = f"fit_nll_{pdf.GetName()}_{data.GetName()}"
    parts = [one for one in str(rng).split(",") if one]
    names = []
    for part in parts:
        name = base + (f"_{part}" if len(parts) > 1 else "")
        for var in pdf.getObservables(data):
            if var.InheritsFrom("RooRealVar"):
                var.setRange(name, var.getMin(part), var.getMax(part))
        names.append(name)
    pdf.setStringAttribute("fitrange", ",".join(names))


def _normalized_name(pdf: Any, observables: list[Any], rng: Any) -> str:
    """What the likelihood names the density: its normalised form's name - or, for a binned
    channel, which is not normalised, the binned sum's own, as RooFit takes it out of a
    product of constraints."""
    binned = binned_part(pdf)
    if binned is not None:
        return str(binned.GetName())
    if hasattr(pdf, "normalized_name"):
        return str(pdf.normalized_name(observables, rng))
    return str(pdf.GetName())


def nll_options(pdf: Any, data: Any, options: Commands) -> RooNLLVar:
    """The likelihood the options describe, with the lines RooFit prints while making it."""
    started = time.perf_counter()
    extended = _extended(pdf, options)
    rng = _range(options)
    _fit_range_attributes(pdf, data, rng)
    conditional = {one.GetName() for one in as_list(options.get("ConditionalObservables", 0, ()))}
    observables = [one for one in pdf.getObservables(data) if one.GetName() not in conditional]
    normalized = _normalized_name(pdf, observables, rng)
    observed = frozenset(one.GetName() for one in pdf.getObservables(data))
    constraints, constrained, global_values = _constraints(pdf, data, options)
    fitted = copies.copies_of(pdf, "fit", observed)
    log(
        pdf,
        INFO,
        "Fitting",
        f"RooAbsPdf::fitTo({normalized}) fixing normalization set for "
        "coefficient determination to observables in data",
    )
    if not _SAID_LIBRARY[0]:
        _SAID_LIBRARY[0] = True
        log(pdf, INFO, "Fitting", "using generic CPU library compiled with no vectorizations")
    nll = RooNLLVar(
        pdf,
        data,
        extended=extended,
        rng=rng,
        conditional=options.get("ConditionalObservables", 0, ()),
        constraints=constraints,
        constrained=constrained,
        global_values=global_values,
        name=f"nll_{normalized}_{data.GetName()}",
        offset=bool(options.get("Offset", 0, False)),
        copies=fitted,
    )
    nll.wrapper_name = "RooEvaluatorWrapper"
    elapsed = (time.perf_counter() - started) * 1000
    log(pdf, INFO, "Fitting", f"Creation of NLL object took {elapsed:g} ms")
    return nll


def fit_to(pdf: Any, data: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """``pdf.fitTo(data, options...)``: the fit, and its result if ``Save()`` was given."""
    options = commands(args, kwargs)
    options.warn_duplicates(f"fitTo({pdf.GetName()})")
    nll = nll_options(pdf, data, options)
    sumw2 = _sumw2_option(pdf, data, options)
    minimizer = RooMinimizer(nll)
    _configure(minimizer, options)
    minimizer.minimize(options.get("Minimizer", 0, ""), options.get("Minimizer", 1, ""))
    if options.get("Hesse", 0, True):
        minimizer.hesse()
    quality = _sumw2_corrected(pdf, minimizer, nll) if sumw2 == 1 and minimizer.params else None
    minos = options.get("Minos")
    if minos:
        minimizer.minos(None if minos is True else minos)
    return _saved(pdf, data, minimizer, quality) if options.get("Save", 0, False) else None


def _sumw2_option(pdf: Any, data: Any, options: Commands) -> int:
    """``SumW2Error``: 1, 0, or -1 untold - which, for weighted data, RooFit warns of."""
    sumw2 = int(options.get("SumW2Error", 0, -1))
    if data.isNonPoissonWeighted() and sumw2 == -1 and "AsymptoticError" not in options:
        log(pdf, WARNING, "InputArguments", f"RooAbsPdf::fitTo({pdf.GetName()}): {WEIGHTED}")
    return sumw2


def _saved(pdf: Any, data: Any, minimizer: RooMinimizer, quality: int | None) -> Any:
    """``Save()``'s result - its covariance's quality the correction's, if one was made."""
    found = minimizer.save(
        f"fitresult_{pdf.GetName()}_{data.GetName()}",
        f"Result of fit of p.d.f. {pdf.GetName()} to dataset {data.GetName()}",
    )
    if quality is not None:
        found.setCovQual(quality)
    return found


#: What RooFit says of a likelihood fit to weighted data told nothing of its errors.
WEIGHTED = """WARNING: a likelihood fit is requested of what appears to be weighted data.
       While the estimated values of the parameters will always be calculated taking the weights into account,
       there are multiple ways to estimate the errors of the parameters. You are advised to make an
       explicit choice for the error calculation:
           - Either provide SumW2Error(true), to calculate a sum-of-weights-corrected HESSE error matrix
             (error will be proportional to the number of events in MC).
           - Or provide SumW2Error(false), to return errors from original HESSE error matrix
             (which will be proportional to the sum of the weights, i.e., a dataset with <sum of weights> events).
           - Or provide AsymptoticError(true), to use the asymptotically correct expression
             (for details see https://arxiv.org/abs/1911.01303).\""""  # noqa: E501


def _sumw2_corrected(pdf: Any, minimizer: RooMinimizer, nll: RooNLLVar) -> int:
    """``calcSumW2CorrectedCovariance``: the covariance ``V C^-1 V`` - ``V`` HESSE's, ``C``
    HESSE's again of the likelihood with every weight squared - and its quality.

    The second HESSE is a step of the fit as any other: the status carries a
    second ``HESSE=``, and the distance to the minimum is the one it finds,
    measured on the squared-weight likelihood.
    """
    first = minimizer.save()
    nll.applyWeightSquared(True)
    log(pdf, INFO, "Fitting", f"RooAbsPdf::fitTo({pdf.GetName()}) Calculating "
        "sum-of-weights-squared correction matrix for covariance matrix")  # fmt: skip
    minimizer.hesse()
    second = minimizer.save()
    nll.applyWeightSquared(False)
    v, c = first._cov, second._cov
    try:
        np.linalg.cholesky(c)
    except np.linalg.LinAlgError:
        log(pdf, ERROR, "Fitting", f"RooAbsPdf::fitTo({pdf.GetName()}) ERROR: Cannot apply "
            "sum-of-weights correction to covariance matrix: correction matrix calculated with "
            "weight-squared is singular")  # fmt: skip
        return -1
    minimizer.applyCovarianceMatrix(v @ np.linalg.inv(c) @ v.T)
    return int(min(first.covQual(), second.covQual()))


def _configure(minimizer: RooMinimizer, options: Commands) -> None:
    """The minimizer's settings from the options that drive it."""
    minimizer.setPrintLevel(int(options.get("PrintLevel", 0, 1)))
    minimizer.setStrategy(int(options.get("Strategy", 0, 1)))
    minimizer.setVerbose(bool(options.get("Verbose", 0, False)))
    minimizer.setEvalErrorWall(bool(options.get("EvalErrorWall", 0, True)))
    minimizer.setRecoverFromNaNStrength(float(options.get("RecoverFromUndefinedRegions", 0, 10.0)))
    minimizer.setPrintEvalErrors(int(options.get("PrintEvalErrors", 0, 10)))
    if "MaxCalls" in options:
        minimizer.setMaxFunctionCalls(int(options.get("MaxCalls")))
