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

from ..cmdargs import Commands, commands
from ..collections import as_list
from ..messages import INFO, log
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
        log(pdf, INFO, "Minimization", "p.d.f. provides expected number of events, including "
            "extended term in likelihood.")  # fmt: skip
        return True
    return False


def _range(options: Commands) -> Any:
    found = options.args("Range")
    if not found:
        return None
    return found[0] if isinstance(found[0], str) else None


def _constraints(pdf: Any, data: Any, options: Commands) -> list[Any]:
    """The constraint terms: those the density carries and ``Constrain`` names, and external ones."""
    found = list(as_list(options.get("ExternalConstraints")))
    carried = getattr(pdf, "constraint_terms", None)
    if carried is not None:
        found = carried(data, options.get("Constrain")) + found
    return found


def _fit_range_attributes(pdf: Any, data: Any, rng: Any) -> None:
    """``resetFitrangeAttributes``: ranges ``fit_nll_<pdf>_<data>`` of the fit's, for plotting in."""
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


def nll_options(pdf: Any, data: Any, options: Commands) -> RooNLLVar:
    """The likelihood the options describe, with the lines RooFit prints while making it."""
    started = time.perf_counter()
    extended = _extended(pdf, options)
    rng = _range(options)
    _fit_range_attributes(pdf, data, rng)
    conditional = {one.GetName() for one in as_list(options.get("ConditionalObservables", 0, ()))}
    observables = [one for one in pdf.getObservables(data) if one.GetName() not in conditional]
    normalized = pdf.normalized_name(observables, rng) if hasattr(
        pdf, "normalized_name") else pdf.GetName()  # fmt: skip
    log(pdf, INFO, "Fitting", f"RooAbsPdf::fitTo({normalized}) fixing normalization set for "
        "coefficient determination to observables in data")  # fmt: skip
    if not _SAID_LIBRARY[0]:
        _SAID_LIBRARY[0] = True
        log(pdf, INFO, "Fitting", "using generic CPU library compiled with no vectorizations")
    nll = RooNLLVar(pdf, data, extended=extended, rng=rng,
                    conditional=options.get("ConditionalObservables", 0, ()),
                    constraints=_constraints(pdf, data, options),
                    name=f"nll_{normalized}_{data.GetName()}",
                    offset=bool(options.get("Offset", 0, False)))  # fmt: skip
    elapsed = (time.perf_counter() - started) * 1000
    log(pdf, INFO, "Fitting", f"Creation of NLL object took {elapsed:g} ms")
    return nll


def fit_to(pdf: Any, data: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """``pdf.fitTo(data, options...)``: the fit, and its result if ``Save()`` was given."""
    options = commands(args, kwargs)
    nll = nll_options(pdf, data, options)
    log(pdf, INFO, "Fitting", f"RooAddition::defaultErrorLevel({nll.GetName()}) Summation contains "
        "a RooNLLVar, using its error level")  # fmt: skip
    minimizer = RooMinimizer(nll)
    _configure(minimizer, options)
    minimizer.minimize(options.get("Minimizer", 0, ""), options.get("Minimizer", 1, ""))
    if options.get("Hesse", 0, True):
        minimizer.hesse()
    minos = options.get("Minos")
    if minos:
        minimizer.minos(None if minos is True else minos)
    if not options.get("Save", 0, False):
        return None
    return minimizer.save(f"fitresult_{pdf.GetName()}_{data.GetName()}",
                          f"Result of fit of p.d.f. {pdf.GetName()} to dataset {data.GetName()}")


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
