"""``RooProduct``'s integral: the product of its factors' integrals, grouped as RooFit groups them.

RooFit integrates a product of functions by its factors
(``RooProduct::getPartIntList``): the factors that depend on none of the
variables integrated over are multiplied in as they are, and the rest are
gathered into groups that share a variable, each group integrated over its
own variables - a factor alone by its own integral, closed form or
numerical, several together as the integral of their product. The answer
is those integrals multiplied in turn, the groups in the order they were
made. A product of a polynomial in one variable and a formula in another is
then a closed form times a one-dimensional numerical integral, not a
two-dimensional one.
"""

from __future__ import annotations

from typing import Any

__all__ = ["announce_parts", "factorises", "groups", "product_integral"]


Group = tuple[list[Any], frozenset[str]]


def _holds(group: list[Any], term: Any) -> bool:
    return any(one is term for one in group)


def groups(terms: list[Any], names: frozenset[str], order: list[str]) -> list[Group]:
    """``groupProductTerms``: the factors outside the integral first, then a group for each
    variable in ``order``, merged with the earlier groups that share a factor with it."""
    found: list[Group] = []
    outside = [term for term in terms if not term.dependents() & names]
    if outside:
        found.append((outside, frozenset()))
    for name in (one for one in order if one in names):
        members = [term for term in terms if name in term.dependents()]
        found = _joined(found, members, name)
    return found


def _joined(found: list[Group], members: list[Any], name: str) -> list[Group]:
    """``found`` with the factors of ``name`` added: one group of them and of every group that
    holds one of them, in the place of the first such group."""
    touching = [at for at, (group, over) in enumerate(found) if over and _shares(group, members)]
    merged, over = _union([found[at] for at in touching], members, name)
    kept = [one for at, one in enumerate(found) if at not in touching]
    kept.insert(touching[0] if touching else len(kept), (merged, over))
    return kept


def _union(joining: list[Group], members: list[Any], name: str) -> Group:
    """The groups joined, their factors in turn and then ``name``'s, each once."""
    merged: list[Any] = []
    over = {name}
    for group, variables in joining:
        merged += [term for term in group if not _holds(merged, term)]
        over |= variables
    merged += [term for term in members if not _holds(merged, term)]
    return merged, frozenset(over)


def _shares(group: list[Any], members: list[Any]) -> bool:
    return any(_holds(group, term) for term in members)


def factorises(product: Any, names: frozenset[str]) -> bool:
    """Whether the integral splits at all: RooFit integrates the product whole, numerically,
    when one group holds every factor."""
    terms = list(product.terms)
    return all(len(group) < len(terms) for group, _ in groups(terms, names, _order(product)))


def _integrand(owner: Any, group: list[Any]) -> Any:
    """The function a group integrates: its one factor, or the product of several (``SUBPROD_``)."""
    if len(group) == 1:
        return group[0]
    from .functions import RooProduct

    made = owner.__dict__.setdefault("_subproducts", {})
    key = tuple(id(term) for term in group)
    if key not in made:
        label = "SUBPROD_" + "_X_".join(term.GetName() for term in group)
        made[key] = RooProduct(label, label, group)
    return made[key]


def _order(product: Any) -> list[str]:
    return [one.GetName() for one in product.leaves()]


def product_integral(product: Any, names: frozenset[str], ctx: dict[str, Any], rng: Any) -> Any:
    """``RooProduct::analyticalIntegral``: each group's integral, multiplied in turn."""
    found: Any = 1.0
    for group, over in groups(list(product.terms), names, _order(product)):
        if not over:
            for term in group:
                found = found * term.compute(ctx)
            continue
        found = found * _integrand(product, group).integrate(over, ctx, rng)
    return found


def announce_parts(product: Any, names: frozenset[str], rng: Any) -> None:
    """The line each group's numerical integral prints as ``getPartIntList`` makes it - none,
    when the product does not split, and is announced whole as a numerical integral."""
    from .integration import announce

    if not factorises(product, frozenset(names)):
        return
    for group, over in groups(list(product.terms), frozenset(names), _order(product)):
        if over:
            announce(_integrand(product, group), over, rng)
