"""``RooNLLVar``: the negative log-likelihood of a density for a dataset, as RooFit sums it.

``-sum w_i log p(x_i)`` over the events in the fit range, the density
normalised over the data's observables in that range, plus - for an
extended fit - the Poisson term for the number of events, plus a
``-log`` of every constraint. The sum is taken with ``math.fsum``, as close
to RooFit's Kahan sum as a sum can be, because Minuit differences these
values and a sum that rounds differently moves where it stops.

An event where the density is not positive, or not a number, makes the
likelihood "invalid": RooFit then hands Minuit a value made worse by how
bad the events were (:class:`Invalid`), so that it backs out of the region.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..cmdargs import commands
from ..collections import RooArgSet, as_list
from ..real import RooAbsReal

__all__ = ["Invalid", "RooNLLVar", "create_nll"]


class Invalid(float):
    """A likelihood that could not be computed, carrying how bad its events were."""

    badness: float = 0.0

    @classmethod
    def of(cls, badness: float) -> Invalid:
        made = cls(math.nan)
        made.badness = badness
        return made


def _log_terms(probs: np.ndarray[Any, Any], weights: np.ndarray[Any, Any]) -> tuple[float, float]:
    """``reduceNLL``: the sum of ``-w log p``, and the badness of the events that have none."""
    probs = np.broadcast_to(probs, weights.shape)
    keep = weights != 0
    probs, weights = probs[keep], weights[keep]
    bad = ~(probs > 0)
    badness = float(np.sum(np.where(np.isnan(probs[bad]), 0.0, -probs[bad]))) if bad.any() else 0.0
    if bad.any() and badness == 0.0:
        badness = float(np.count_nonzero(bad))
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = -weights * np.log(probs)
    return math.fsum(terms.tolist()), badness


class RooNLLVar(RooAbsReal):
    """The likelihood of ``pdf`` for ``data``, a function of the density's parameters."""

    def __init__(self, pdf: Any, data: Any, *, extended: bool = False, rng: Any = None,
                 conditional: Any = (), constraints: Any = (), name: str = "",
                 offset: bool = False) -> None:  # fmt: skip
        super().__init__(name or f"nll_{pdf.GetName()}_{data.GetName()}", "-log(likelihood)")
        self.pdf = self._proxy("function", pdf)
        self.data = data
        self.extended = bool(extended)
        self.rng = rng or None
        self.constraints = [self._proxy("constraint", c) for c in as_list(constraints)]
        conditional_names = {one.GetName() for one in as_list(conditional)}
        self.nset = frozenset(
            one.GetName() for one in pdf.getObservables(data) if one.GetName() not in conditional_names
        )
        keep = data.mask(None, self.rng) & (data.weights() != 0)
        self.columns = {k: v[keep] for k, v in data.columns().items()}
        self.w = data.weights()[keep]
        self.sumw = math.fsum(self.w.tolist())
        self.offset = offset
        self._offset_value = 0.0

    def compute(self, ctx: Any) -> Any:
        return self.evaluate_nll()

    def evaluate_nll(self) -> float:
        """The likelihood at the parameters' values now, or :class:`Invalid`."""
        probs = np.asarray(self.pdf.value(dict(self.columns), self.nset, self.rng), dtype=np.float64)
        total, badness = _log_terms(probs, self.w)
        if self.extended:
            total += self.pdf.extendedTerm(self.sumw, self.pdf.expected(self.nset, self.rng))
        for constraint in self.constraints:
            found = float(np.asarray(constraint.value({}, constraint.dependents() & self._constrained(constraint))))
            total -= math.log(found) if found > 0 else math.nan
        if badness or math.isnan(total):
            return Invalid.of(badness or 1.0)
        if self.offset and self._offset_value == 0.0:
            self._offset_value = total
        return total - self._offset_value

    def _constrained(self, constraint: Any) -> frozenset[str]:
        return frozenset(one.GetName() for one in constraint.leaves())

    def defaultErrorLevel(self) -> float:
        return 0.5

    def getParameters(self, observables: Any = None, stripDisconnected: bool = True) -> RooArgSet:
        found = RooArgSet(self.pdf.getParameters(self.data))
        for constraint in self.constraints:
            found.add(constraint.getParameters(self.data))
        return RooArgSet([one for one in found if one.GetName() not in _names(observables)]).sorted_copy()

    def getVal(self, nset: Any = None) -> float:
        return float(self.evaluate_nll())


def _names(items: Any) -> set[str]:
    if items is None:
        return set()
    if hasattr(items, "numEntries"):
        return {one.GetName() for one in items.get()}
    return {one.GetName() for one in as_list(items)}


def create_nll(pdf: Any, data: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> RooNLLVar:
    """``pdf.createNLL(data, options...)``."""
    from .fit import nll_options

    options = commands(args, kwargs)
    return nll_options(pdf, data, options)
