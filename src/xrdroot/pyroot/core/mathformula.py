"""``ROOT::Math``'s functions inside a formula: ``TF1("f", "ROOT::Math::normal_pdf(x, [0])")``.

ROOT's formulas are C++, so any ``ROOT::Math`` function may be written in
one. Every function of :mod:`.rmath` - the densities, tails and quantiles,
and the special functions - is put in the formula language's table under its
``ROOT::Math::`` name when the core is imported, applied a point at a time
over the arrays a formula is evaluated on.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np

from ...function.library import CALLS, Call
from . import rmath

__all__: list[str] = []


def _vectorised(fn: Callable[..., float]) -> Callable[..., Any]:
    """``fn`` of numbers, applied over the arrays a formula passes, broadcast together."""
    each = np.vectorize(lambda *args: float(fn(*args)), otypes=[np.float64])

    def apply(*args: Any) -> Any:
        with np.errstate(all="ignore"):
            return each(*args)

    return apply


def _arity(fn: Callable[..., Any]) -> tuple[int, int]:
    """How few and how many arguments ``fn`` takes."""
    parameters = list(inspect.signature(fn).parameters.values())
    required = [p for p in parameters if p.default is inspect.Parameter.empty]
    return len(required), len(parameters)


def _functions() -> dict[str, Callable[..., Any]]:
    """Every function of :mod:`.rmath` a formula may name."""
    return {name: getattr(rmath, name) for name in rmath.__all__ if name[:1].islower()}


def register() -> None:
    """Put each function in the formula language's table under its ``ROOT::Math::`` name."""
    for name, fn in _functions().items():
        least, most = _arity(fn)
        CALLS.setdefault(f"ROOT::Math::{name}", Call(least, most, _vectorised(fn)))


register()
