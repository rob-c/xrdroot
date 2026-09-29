"""``RooNLLVar``: the negative log-likelihood of a density for a dataset, as RooFit sums it.

``-sum w_i log p(x_i)`` over the events in the fit range, the density
normalised over the data's observables in that range, plus - for an
extended fit - the Poisson term for the number of events, plus a
``-log`` of every constraint. The sum is taken with ``math.fsum``, as close
to RooFit's Kahan sum as a sum can be, because Minuit differences these
values and a sum that rounds differently moves where it stops.

An event where the density is not positive, or not a number, makes the
likelihood "invalid": RooFit then hands Minuit a value made worse by how
bad the events were (:mod:`..nanpack`), so that it backs out of the region.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...random import libm
from .. import kernels
from ..cmdargs import commands
from ..collections import RooArgSet, as_list
from ..real import RooAbsReal
from .binned import binned_part, binned_terms
from .kahan import Kahan

__all__ = ["RooNLLVar", "create_nll"]


def _log_terms(probs: np.ndarray[Any, Any], weights: np.ndarray[Any, Any]) -> tuple[Kahan, float]:
    """``reduceNLL``: the sum of ``-w log p``, and the badness of the events that have none.

    A value at or below zero is as bad as it is below zero, a NaN as bad as
    its payload says (:mod:`..nanpack`) - an untagged NaN not at all, so the
    sum stays NaN - which is how ``getLog`` in ``RooBatchCompute`` scores them.
    """
    from ..nanpack import unpack

    probs = np.broadcast_to(probs, weights.shape)
    keep = weights != 0
    probs, weights = probs[keep], weights[keep]
    badness = float(np.sum(np.where(probs <= 0, -probs, 0.0)) + np.sum(unpack(probs)))
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = -weights * libm.log(probs)
    return Kahan().extend(terms.tolist()), badness


def _normalised_over(pdf: Any, data: Any, conditional: Any) -> frozenset[str]:
    """The observables the density is normalised over: the data's, but the conditional ones."""
    given = {one.GetName() for one in as_list(conditional)}
    observed = [one.GetName() for one in pdf.getObservables(data)]
    return frozenset(name for name in observed if name not in given)


class RooNLLVar(RooAbsReal):
    """The likelihood of ``pdf`` for ``data``, a function of the density's parameters."""

    def __init__(
        self,
        pdf: Any,
        data: Any,
        *,
        extended: bool = False,
        rng: Any = None,
        conditional: Any = (),
        constraints: Any = (),
        constrained: Any = None,
        global_values: Any = None,
        name: str = "",
        offset: bool = False,
        copies: Any = None,
    ) -> None:
        super().__init__(name or f"nll_{pdf.GetName()}_{data.GetName()}", "-log(likelihood)")
        self.pdf = self._proxy("function", pdf)
        self.data = data
        self.extended = bool(extended)
        self.rng = rng or None
        self.constraints = [self._proxy("constraint", c) for c in as_list(constraints)]
        self._constrained_over = constrained
        self._global_values = dict(global_values or {})
        self.nset = _normalised_over(pdf, data, conditional)
        #: Whether a channel is a binned likelihood, which counts its empty bins too.
        self.binned = _any_binned(pdf)
        self._attach(data)
        self.offset = offset
        self._offset_value = 0.0
        #: The states of the fit's copy of the model, for the nodes that keep one (:mod:`..copies`).
        self._copies = copies
        self._announced = False
        #: ``applyWeightSquared``: each event counts with its weight squared, for SumW2Error.
        self._weight_squared = False
        #: What ROOT 6.40 calls the likelihood ``createNLL`` hands out - a ``RooEvaluatorWrapper``
        #: round this sum - which a profile of it is named after; ``None`` for one made directly.
        self.wrapper_name: str | None = None

    def _attach(self, data: Any) -> None:
        """The events the likelihood sums: those in range, and weighing something - or all the
        bins, for a binned channel."""
        self.data = data
        keep = data.mask(None, self.rng) & ((data.weights() != 0) | self.binned)
        self.columns = {k: v[keep] for k, v in data.columns().items()}
        self.w = data.weights()[keep]
        self.sumw = math.fsum(self.w.tolist())

    def setData(self, data: Any, cloneData: bool = True) -> bool:
        """The likelihood of the same density for other data - a toy's, as RooStats reuses it."""
        self._attach(data)
        self._offset_value = 0.0
        return True

    def applyWeightSquared(self, flag: bool) -> None:
        """Count each event with its weight squared - the likelihood whose HESSE corrects
        ``SumW2Error``'s covariance - or, again, with its weight."""
        self._weight_squared = bool(flag)

    def compute(self, ctx: Any) -> Any:
        return self.evaluate_nll()

    def evaluate_nll(self) -> float:
        """The likelihood at the parameters' values now, or a NaN carrying how bad it was."""
        from ..copies import within

        with within(self._copies or {}):
            if not self._announced:  # the fit's copy makes its normalisation integral on first use
                from ..integration import announce

                self._announced = True
                if getattr(self.pdf, "channel_terms", None) is None:  # else: per channel
                    announce(self.pdf, self.nset, self.rng)
            return self._evaluate()

    def _constraint_sum(self) -> float:
        """``RooConstraintSum``: ``-log`` of each constraint, summed on its own - the kernels'
        values, as the likelihood's evaluator computes them - before it joins the rest."""
        found = 0.0
        for constraint in self.constraints:
            nset = self._constrained(constraint)
            with kernels.likelihood():
                value = float(np.asarray(constraint.value(dict(self._global_values), nset)))
            found -= math.log(value) if value > 0 else (-math.inf if value == 0 else math.nan)
        return found

    def _evaluate(self) -> float:
        self._badness = 0.0
        channels = getattr(self.pdf, "channel_terms", None)
        total = channels(self) if channels is not None else self.channel(self.pdf, None)
        if self.constraints:
            total = total + self._constraint_sum()
        if self._badness:
            from ..nanpack import pack

            return float(pack(self._badness))
        if self.offset and self._offset_value == 0.0:
            self._offset_value = total
        return total - self._offset_value

    def channel(self, pdf: Any, keep: Any, simulated: int = 0) -> float:
        """``-sum w log p`` of the events ``keep`` selects - all, for ``None`` - and their Poisson
        term."""
        columns = self.columns if keep is None else {k: v[keep] for k, v in self.columns.items()}
        given = self.w if keep is None else self.w[keep]
        binned = binned_part(pdf)
        if binned is not None:
            return self._binned_channel(binned, columns, given, simulated)
        weights = given * given if self._weight_squared else given
        nset = self.nset & pdf.dependents() if keep is not None else self.nset
        with kernels.likelihood():
            probs = np.asarray(pdf.value(dict(columns), nset, self.rng), dtype=np.float64)
        total, badness = _log_terms(probs, weights)
        self._log_top(pdf, probs, weights, nset)
        self._badness += badness
        if self.extended and pdf.canBeExtended():
            total.total += self._extended_term(pdf, given, weights, nset)  # not onto the carry
        if simulated:
            total.add(float(math.fsum(weights.tolist())) * math.log(simulated))
        return total.total

    def _binned_channel(self, binned: Any, columns: Any, given: Any, simulated: int) -> float:
        """A binned channel: its bins' Poisson terms, and ``N log(channels)`` if simultaneous."""
        weights = given * given if self._weight_squared else given
        total, events, bad = binned_terms(binned, columns, weights)
        if bad:
            self._badness += float(bad)
        if simulated:
            total.add(events * math.log(simulated))
        return total.total

    def _extended_term(self, pdf: Any, given: Any, weights: Any, nset: frozenset[str]) -> float:
        """The Poisson term, scaled by ``sum w^2 / sum w`` when the weights are squared."""
        sumw = math.fsum(given.tolist())
        sumw2 = math.fsum(weights.tolist()) if self._weight_squared else 0.0
        expected = pdf.expected(nset, self.rng, fit=True)
        return float(pdf.extendedTerm(sumw, expected, sumw2))

    def _log_top(self, pdf: Any, probs: Any, weights: Any, nset: frozenset[str]) -> None:
        """The likelihood's own messages: each event whose density is not positive, or NaN."""
        from .. import evalerrors

        if not evalerrors.active():
            return
        probs = np.broadcast_to(probs, weights.shape)[weights != 0]
        key, origin, servers = _top_node(pdf, nset, self.rng)
        for message, number in (
            ("getLogVal() top-level p.d.f not greater than zero", probs <= 0),
            ("getLogVal() top-level p.d.f evaluates to NaN", np.isnan(probs)),
        ):
            evalerrors.record(
                key, origin, message, servers, int(np.count_nonzero(number)), top=True
            )

    def _constrained(self, constraint: Any) -> frozenset[str]:
        """What a constraint is normalised over: its variables - not its constants - among those
        the constraints are normalised over; worked out once per constraint."""
        cache = self.__dict__.setdefault("_constrained_cache", {})
        found = cache.get(id(constraint))
        if found is None:
            found = frozenset(one.GetName() for one in constraint.getVariables())
            if self._constrained_over is not None:
                found = found & self._constrained_over
            cache[id(constraint)] = found
        return found

    def defaultErrorLevel(self) -> float:
        """One half - which RooFit's likelihood, a sum with a RooNLLVar in it, says it takes."""
        from ..messages import INFO, log

        log(self, INFO, "Fitting", f"RooAddition::defaultErrorLevel({self.GetName()}) "
            "Summation contains a RooNLLVar, using its error level")  # fmt: skip
        return 0.5

    def getParameters(self, observables: Any = None, stripDisconnected: bool = True) -> RooArgSet:
        found = RooArgSet(self.pdf.getParameters(self.data))
        for constraint in self.constraints:
            found.add(constraint.getParameters(self.data))
        return RooArgSet(
            [one for one in found if one.GetName() not in _names(observables)]
        ).sorted_copy()

    def getVal(self, nset: Any = None) -> float:
        return float(self.evaluate_nll())


def _any_binned(pdf: Any) -> bool:
    """Whether the density - or one of a simultaneous density's channels - is binned."""
    channels = getattr(pdf, "channels", None)
    parts = list(channels.values()) if isinstance(channels, dict) else [pdf]
    return any(binned_part(part) is not None for part in parts)


def _top_node(pdf: Any, nset: frozenset[str], rng: Any) -> tuple[Any, Any, Any]:
    """What RooFit calls the density a likelihood takes the logarithm of, and its inputs' values."""
    if not pdf.selfNormalized():
        return (
            ("norm", id(pdf)),
            lambda: pdf.normalized_origin(nset, rng),
            lambda: pdf.normalized_servers(nset, rng),
        )
    describe = getattr(pdf, "compiled_origin", None)
    if describe is None:
        return ("pdf", id(pdf)), lambda: f"{pdf.ClassName()}::{pdf.GetName()}", lambda: ""
    return ("pdf", id(pdf)), lambda: describe(nset, rng), lambda: pdf.compiled_servers(nset, rng)


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
