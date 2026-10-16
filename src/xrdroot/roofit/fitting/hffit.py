"""Cache native histogram structure only while a synchronous Minuit fit owns the model."""
from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import numpy as np

from ...random import libm
from ...stats import log_gamma
from ..pdfs.histfactory import binned_likelihood
from ..pdfs.simultaneous import RooSimultaneous
from . import hfdata, hfmodel, hfnative
from .binned import binned_part
from .hfprogram import NotCompiled, Program
from .kahan import Kahan

#: Planning more bins costs more than the array evaluator it replaces.
MAX_BINS = 8
#: A nested or concurrent fit owns its own short-lived plan.
_ACTIVE: ContextVar[Plan | None] = ContextVar("native_histogram_fit", default=None)


class Plan:
    """Fixed data and histogram geometry, with floating parameters read on every call."""

    def __init__(self, nll: Any) -> None:
        self.nll = nll
        pdf = _channel(nll)
        root = binned_part(pdf)
        if root is None:
            raise NotCompiled
        if not _squarable(nll):
            raise NotCompiled
        weights = nll.w * nll.w if nll._weight_squared else nll.w
        self.counts = weights.tolist()
        if not all(0. <= count <= 1e100 for count in self.counts):
            raise NotCompiled
        self.factorials = [log_gamma(count + 1.) for count in self.counts]
        with binned_likelihood():
            self.predict, self.source = Program(nll.columns, len(weights)).build(root)
        from .hfconstraints import prepare

        self.constraints = prepare(nll)

    def total(self) -> float | None:
        mus = self.predict()
        if not all(0. < mu < math.inf for mu in mus):
            return None
        total = Kahan()
        for mu, count, factorial in zip(mus, self.counts, self.factorials, strict=False):
            term = 0. if abs(mu) < 1e-10 and abs(count) < 1e-10 else (
                mu - count * float(libm.log(mu)) + factorial)
            total.add(term)
        return total.total


def _squarable(nll: Any) -> bool:
    # A rejected plan must not emit an overflow warning before the ordinary evaluator does.
    return not nll._weight_squared or all(
        math.isfinite(weight) and abs(weight) <= 1e50 for weight in nll.w.tolist())


def _channel(nll: Any) -> Any:
    pdf = nll.pdf
    if type(pdf) is not RooSimultaneous:
        return pdf
    if len(pdf.channels) != 1:
        raise NotCompiled
    label, channel = next(iter(pdf.channels.items()))
    if not np.all(nll.columns[pdf.index.GetName()] == pdf.index.lookupIndex(label)):
        raise NotCompiled
    return channel


def _data(nll: Any) -> bool:
    return (hfnative.array(nll.w, floating=True) and 0 < len(nll.w) <= MAX_BINS
            and hfdata.mapping(nll.columns, lambda column: hfnative.array(column)
                               and len(column) == len(nll.w)))


def _minimizer(minimizer: Any, run: Any) -> bool:
    from .minimizer import RooMinimizer

    return (hfnative.instance(minimizer, (RooMinimizer,))
            and minimizer._optimizing and not minimizer.verbose and minimizer.print_level < 0
            and hfnative._function(getattr(run, "__func__", run)))


def _binned(minimizer: Any) -> bool:
    from .minimizer import RooMinimizer
    from .nll import RooNLLVar

    if type(minimizer) is not RooMinimizer:
        return False
    nll = object.__getattribute__(minimizer, "__dict__").get("function")
    return (type(nll) is RooNLLVar
            and object.__getattribute__(nll, "__dict__").get("binned") is True)


def _eligible(minimizer: Any, run: Any) -> bool:
    from .nll import RooNLLVar

    if not _binned(minimizer):
        return False
    if not _minimizer(minimizer, run) or np.geterr()["under"] != "ignore":
        return False
    nll = minimizer.function
    return (hfnative.instance(nll, (RooNLLVar,)) and _data(nll)
            and hfmodel.model(nll, minimizer.all_params))


def _prepare(minimizer: Any, run: Any) -> Plan | None:
    try:
        with hfnative.checking():
            return Plan(minimizer.function) if _eligible(minimizer, run) else None
    except (NotCompiled, AttributeError, TypeError, ValueError, IndexError, OverflowError):
        return None


@contextmanager
def within(minimizer: Any, run: Any) -> Iterator[None]:
    """No model references or stale histogram values survive a fit, including a failed fit."""
    token = _ACTIVE.set(_prepare(minimizer, run))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def total(nll: Any) -> float | None:
    plan = _ACTIVE.get()
    return plan.total() if plan is not None and plan.nll is nll else None


def constraints(nll: Any) -> float | None:
    plan = _ACTIVE.get()
    if plan is None or plan.nll is not nll or plan.constraints is None:
        return None
    return plan.constraints.total()
