"""``GenerateCountingAsimovData``: the Asimov data of a counting model - one event.

A density that cannot be extended is a counting model: a Poisson, a
Gaussian, a multivariate Gaussian, or a product of them. Its Asimov data is
the one event with each observable at the value its term expects - the
Poisson's rate, unrounded, the Gaussian's mean - set from the one server of
the term that is not constant.
"""

from __future__ import annotations

from typing import Any

from ..roofit.messages import ERROR, FATAL, INFO, log

__all__ = ["counting_asimov_data"]


def _set_to_expected(servers: list[Any], observables: Any, prefix: str) -> bool:
    """``setObsToExpected``: the one observable among ``servers`` at the one free server."""
    names = {one.GetName() for one in observables}
    obs = expected = None
    for server in servers:
        if server.GetName() in names:
            if obs is not None:
                log(None, FATAL, "Generation", f"{prefix}Has two observables ?? ")
                return False
            obs = server
        elif not server.isConstant():
            if expected is not None:
                log(None, ERROR, "Generation", f"{prefix}Has two non-const arguments  ")
                return False
            expected = server
    if obs is None or expected is None:
        log(None, FATAL, "Generation", f"{prefix}No observable?")
        return False
    obs.setVal(expected.getVal())
    from .asimov import _LEVEL

    if _LEVEL[0] > 2:
        log(None, INFO, "Generation", f"SetObsToExpected : setting {obs.GetName()} to expected "
            f"value {expected.getVal():g} of {expected.GetName()}")  # fmt: skip
    return True


def _term(pdf: Any, observables: Any) -> bool:
    """One term of the counting model, its observable set to its expectation."""
    prefix = f"AsymptoticCalculator::SetObsExpected( {pdf.ClassName()} ) : "
    kind = pdf.ClassName()
    if kind == "RooProdPdf":
        return all([_term(one, observables) for one in pdf.pdfList()
                    if one.dependsOn(observables)])  # fmt: skip
    if kind == "RooMultiVarGaussian":
        return all([_set_to_expected([x, mu], observables, f"{prefix} : dim {i} ")
                    for i, (x, mu) in enumerate(zip(pdf.xVec(), pdf.muVec()))])  # fmt: skip
    if kind not in ("RooPoisson", "RooGaussian"):
        log(None, ERROR, "InputArguments", "Illegal term in counting model: the PDF "
            f"{pdf.GetName()} depends on the observables, but is not a Poisson, Gaussian or "
            "Product")  # fmt: skip
        return False
    found = _set_to_expected(list(pdf.servers()), observables, prefix)
    if kind == "RooPoisson":
        pdf.setNoRounding(True)  # the expectation is not a count
    return found


def counting_asimov_data(pdf: Any, observables: Any, category: Any = None) -> Any:
    """The one event, named ``CountingAsimovData<index>`` - or ``None``."""
    from ..roofit.data.dataset import RooDataSet
    from .asimov import _LEVEL

    if _LEVEL[0] > 1:
        log(None, INFO, "Generation", f"generate counting Asimov data for pdf of type "
            f"{pdf.ClassName()}")  # fmt: skip
    if pdf.ClassName() not in ("RooProdPdf", "RooPoisson", "RooGaussian", "RooMultiVarGaussian"):
        log(None, ERROR, "InputArguments", "A counting model pdf must be either a RooProdPdf or a "
            "RooPoisson or a RooGaussian")  # fmt: skip
        return None
    if not _term(pdf, observables):
        return None
    index = category.getCurrentIndex() if category is not None else 0
    obs = list(observables)
    made = RooDataSet(f"CountingAsimovData{index}", f"CountingAsimovData{index}", obs)
    made.add(obs)
    return made
