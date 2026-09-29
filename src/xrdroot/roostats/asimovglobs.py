"""The Asimov values of a model's global observables: each where its nuisance parameter is.

``MakeAsimovData`` goes through the constraint terms - the factors of the
nuisance density - and sets each term's one global observable to the value
of its one server that depends on the nuisance parameters: the Gaussian's
mean, the Poisson's rate - divided by the scale ``theta`` of a Gamma. A
term it cannot read so is skipped, saying why, as RooStats says it.
"""

from __future__ import annotations

from typing import Any

from ..roofit.messages import ERROR, INFO, WARNING, log

__all__ = ["set_global_observables"]

#: The constraint classes whose global observable is known to sit at a server's value.
SUPPORTED = ("RooGaussian", "RooPoisson", "RooGamma", "RooLognormal", "RooBifurGauss")

PREFIX = "AsymptoticCalculator::MakeAsimovData"


def set_global_observables(model: Any, gobs: Any, nuisance: Any) -> None:
    """Each constraint term's global observable at its nuisance parameter's server."""
    from .utils import MakeNuisancePdf

    nuispdf = MakeNuisancePdf(model, "TempNuisPdf")
    if nuispdf is None:
        log(None, 5, "Generation", f"{PREFIX}: model has nuisance parameters and global obs but "
            "no nuisance pdf ")  # fmt: skip
        return
    terms = list(nuispdf.pdfList()) if nuispdf.ClassName() == "RooProdPdf" else [nuispdf]
    for term in terms:
        if term.dependsOn(nuisance) and term.ClassName() != "RooUniform":
            _one(term, gobs, nuisance)


def _readable(term: Any, params: Any, observed: Any) -> bool:
    """One global observable and one free parameter - else the term is skipped, said."""
    from .utils import RemoveConstantParameters

    name = term.GetName()
    if len(observed) != 1:
        many = len(observed) > 1
        log(None, ERROR if many else WARNING, "Generation", f"{PREFIX}: constraint term  {name} "
            + ("has multiple global observables -cannot generate - skip it" if many else
               "has no global observables - skip it"))  # fmt: skip
        return False
    RemoveConstantParameters(params)
    if len(params) != 1:
        log(None, ERROR, "Generation", f"{PREFIX}:constraint term {name} has multiple floating "
            "params - cannot generate - skip it ")  # fmt: skip
        return False
    return True


def _kind_checked(term: Any, target: Any) -> bool:
    """The term's class warned of if unsupported, a Poisson made unrounded - and whether the
    global observable is its own server, as all but a Gamma's must be."""
    name, kind = term.GetName(), term.ClassName()
    if kind not in SUPPORTED:
        log(None, WARNING, "Generation", f"{PREFIX}:constraint term {name} of type {kind} is a "
            "non-supported type - result might be not correct ")  # fmt: skip
    if kind == "RooPoisson":
        term.setNoRounding(True)
    if term.findServer(target) is None and kind != "RooGamma":
        log(None, ERROR, "Generation", f"{PREFIX}:constraint term {name} has no direct dependence "
            "on global observable- cannot generate it ")  # fmt: skip
        return False
    return True


def _one(term: Any, gobs: Any, nuisance: Any) -> None:
    """One term: its global observable, its one free parameter, its server of the nuisance."""
    params, observed = term.getParameters(gobs), term.getObservables(gobs)
    if not _readable(term, params, observed):
        return
    target = next(iter(observed))
    if not _kind_checked(term, target):
        return
    theta = _theta(term) if term.ClassName() == "RooGamma" else None
    if not _set_from_server(term, target, nuisance, theta):
        log(None, ERROR, "Generation", f"{PREFIX} - can't find nuisance for constraint term - "
            f"global observables will not be set to Asimov value {term.GetName()}")  # fmt: skip
        log(None, ERROR, "Generation", "Parameters: ")
        params.Print("V")
        log(None, ERROR, "Generation", "Observables: ")
        observed.Print("V")


def _theta(term: Any) -> Any:
    """A Gamma's scale: its server with ``theta`` in its name - or none, a scale of one."""
    found = next((one for one in term.servers() if "theta" in one.GetName()), None)
    if found is None:
        log(None, INFO, "Generation", f"{PREFIX}:constraint term {term.GetName()} is a Gamma "
            "distribution and no server named theta is found. Assume that the Gamma scale is  1 ")
    return found


def _set_from_server(term: Any, target: Any, nuisance: Any, theta: Any) -> bool:
    """The global observable at the one server that depends on the nuisance parameters."""
    found = False
    for server in term.servers():
        if not server.dependsOn(nuisance):
            continue
        if found:
            log(None, ERROR, "Generation", f"{PREFIX}:constraint term {term.GetName()} constraint "
                "term has more server depending on nuisance- cannot generate it ")  # fmt: skip
            return False
        scaled = theta is not None and theta.getVal() > 0
        target.setVal(server.getVal() / theta.getVal() if scaled else server.getVal())
        found = True
    return found
