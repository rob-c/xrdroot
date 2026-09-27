"""Integrals of a product with conditional factors: ``RooProdPdf``'s specialised integrals.

``RooProdPdf("model", "model", {pdfErr}, Conditional({decay}, {dt}))`` is
``pdfErr(dterr) decay(dt | dterr)``, each factor normalised over its own
observables, so the product needs no normalisation of its own. Integrating
it over ``dt`` is integrating ``decay`` over its own observable - one - but
over ``dterr`` it is not: ``decay`` depends on ``dterr`` without being
normalised over it, so the factors that involve ``dterr`` are integrated
together, numerically - the integral RooFit calls
``SPECINT[pdfErr_NORM[dterr]_X_decay_NORM[dt]]_Int[dterr]`` - and the others
factor by factor.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..messages import INFO, log

__all__ = ["announce", "fraction", "joint_names", "label"]


def _own(pdf: Any, factor: Any, nset: frozenset[str]) -> frozenset[str]:
    return pdf.factor_nset(factor, nset)


def joint_names(pdf: Any, names: frozenset[str], nset: frozenset[str]) -> frozenset[str]:
    """The integrated variables some factor depends on without being normalised over - nor over all its own."""
    found: set[str] = set()
    for factor in pdf.pdfs:
        own = _own(pdf, factor, nset)
        conditioning = (factor.dependents() & names) - own
        if conditioning and not own <= names:
            found |= conditioning
    return frozenset(found)


def _group(pdf: Any, joint: frozenset[str]) -> list[Any]:
    return [factor for factor in pdf.pdfs if factor.dependents() & joint]


def label(pdf: Any, names: frozenset[str], nset: frozenset[str]) -> str:
    """``SPECINT[a_NORM[x]_X_b_NORM[y]]_Int[x]``: RooFit's name of the factors' joint integral."""
    joint = joint_names(pdf, names, nset)
    parts = []
    for factor in _group(pdf, joint):
        own = _own(pdf, factor, nset)
        order = [one.GetName() for one in factor.leaves() if one.GetName() in own]
        parts.append(f"{factor.GetName()}_NORM[{','.join(order)}]")
    over = [one.GetName() for one in pdf.leaves() if one.GetName() in joint]
    return f"SPECINT[{'_X_'.join(parts)}]_Int[{','.join(over)}]"


def announce(pdf: Any, names: frozenset[str], nset: frozenset[str]) -> bool:
    """``RooRealIntegral::init``'s line for the joint integral over ``names``, if there is one."""
    joint = joint_names(pdf, names, nset)
    if not joint:
        return False
    over = [one.GetName() for one in pdf.leaves() if one.GetName() in joint]
    method = "RooIntegrator1D" if len(over) == 1 else "RooAdaptiveIntegratorND"
    log(pdf, INFO, "NumericIntegration", f"RooRealIntegral::init({label(pdf, names, nset)}) using numeric "
        f"integrator {method} to calculate Int({','.join(over)})")  # fmt: skip
    return True


def fraction(pdf: Any, names: frozenset[str], ctx: dict[str, Any], nset: frozenset[str], rng: Any,
             norm_rng: Any = None) -> Any:  # fmt: skip
    """The integral over ``names`` of the product, its factors normalised each over its own."""
    from ..integration import numeric

    joint = joint_names(pdf, names, nset)
    group = _group(pdf, joint)
    found: Any = 1.0
    for factor in (one for one in pdf.pdfs if all(one is not member for member in group)):
        own = _own(pdf, factor, nset)
        part = names & own
        found = found * (factor.fraction(part, ctx, own, rng, norm_rng) if part
                         else factor.value(ctx, own, norm_rng))  # fmt: skip
    if not group:
        return found

    def inner(c: dict[str, Any]) -> Any:
        value: Any = 1.0
        for factor in group:
            value = value * factor.value(c, _own(pdf, factor, nset), norm_rng)
        return value

    covered = set().union(*(factor.dependents() for factor in group)) & names
    rest = [one.GetName() for one in pdf.leaves() if one.GetName() in covered]
    return found * np.asarray(numeric(pdf, rest, inner, ctx, rng))
