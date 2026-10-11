"""``RooDerivative``: a function's first, second or third derivative by one of its variables.

RooFit takes it numerically, with ``ROOT::Math::RichardsonDerivator`` and a
step of ``eps``: central differences of the step and of half of it,
combined so their leading errors cancel; inside the variable's range that
is ROOT 6.40's to the bit. Within a step of an end ROOT's binding goes past
it in a way of its own, which agrees with this - the function evaluated
where the step puts it - to a few parts in 1e8. ``getVal`` leaves the
variable at the last point, half a step below where it was, as ROOT does.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..function.numeric import richardson, richardson2, richardson3
from .collections import as_list
from .real import Context, RooAbsReal

__all__ = ["RooDerivative", "make_derivative"]

#: Each order's Richardson combination.
ORDERS = {1: richardson, 2: richardson2, 3: richardson3}


class RooDerivative(RooAbsReal):
    """``d^n func / dx^n`` at the variables' values."""

    def __init__(self, name: Any, title: Any, func: Any, x: Any, nset: Any = None,
                 order: int = 1, eps: float = 0.001) -> None:  # fmt: skip
        super().__init__(name, title)
        if int(order) not in ORDERS:
            raise ValueError(f"RooDerivative::ctor({name}) ERROR, derivation order must be 1, 2 "
                             f"or 3")  # fmt: skip
        self.func = self._proxy("function", func)
        self.x = self._proxy("x", x)
        self._nset = frozenset(one.GetName() for one in as_list(nset or []))
        self._order, self._eps = int(order), float(eps)

    def order(self) -> int:
        return self._order

    def eps(self) -> float:
        return self._eps

    def setEps(self, eps: float) -> None:
        self._eps = float(eps)

    def compute(self, ctx: Context) -> Any:
        name = self.x.GetName()
        at = np.asarray(ctx.get(name, self.x.getVal()), dtype=np.float64)
        nset = self._nset or None

        def values(step: float) -> Any:
            moved = dict(ctx)
            moved[name] = at + step
            return np.asarray(self.func.value(moved, nset), dtype=np.float64)

        return ORDERS[self._order](values, self._eps)

    def getVal(self, nset: Any = None) -> float:
        found = super().getVal(nset)
        self.x.setVal(self.x.getVal() - self._eps / 2)  # where the binding put it last
        return found


def make_derivative(func: Any, obs: Any, args: tuple[Any, ...]) -> RooDerivative:
    """``derivative(obs, [normSet], order=1, eps=0.001)``, named as ROOT 6.40 names it."""
    nset = args[0] if args and not isinstance(args[0], (int, float)) else None
    numbers = [float(a) for a in args if isinstance(a, (int, float))]
    order, eps = (int(numbers[0]) if numbers else 1), (numbers[1] if len(numbers) > 1 else 0.001)
    name = f"{func.GetName()}_DERIV_{order}{obs.GetName()}"  # ROOT 6.40's, the order in it
    title = f"{order}Derivative of {func.GetName()} w.r.t {obs.GetName()} "
    return RooDerivative(name, title, func, obs, nset, order, eps)
