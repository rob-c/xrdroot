"""``RooEfficiency``: the probability an event of given ``x`` falls in a category's state.

``RooEfficiency("e", "e", eff, cut, "accept")`` is ``eff(x)`` - held to
``[0, 1]`` - for an event in the state ``accept`` and ``1 - eff(x)`` for
one in the other: a density of the category, normalised over it by
construction, which a fit with ``ConditionalObservables(x)`` makes a fit of
the efficiency function itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..pdf import RooAbsPdf
from ..real import Context

__all__ = ["RooEfficiency"]


class RooEfficiency(RooAbsPdf):
    """An efficiency function as a density of an accept/reject category."""

    def __init__(self, name: Any, title: Any, effFunc: Any, cat: Any, sigCatName: Any) -> None:
        super().__init__(name, title)
        self.eff = self._proxy("effFunc", effFunc)
        self.cat = self._proxy("cat", cat)
        self._accept = str(sigCatName)

    def compute(self, ctx: Context) -> Any:
        eff = np.clip(np.asarray(self.eff.compute(ctx), dtype=np.float64), 0.0, 1.0)
        state = np.asarray(self.cat.compute(ctx))
        found = np.where(state == self.cat.lookupIndex(self._accept), eff, 1.0 - eff)
        return found if found.ndim else float(found)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        """Over the category alone, its integral is one: its two states' values add up to it."""
        return frozenset(names) if names == {self.cat.GetName()} else frozenset()

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        return np.ones(np.shape(self.eff.compute(ctx))) if np.ndim(self.eff.compute(ctx)) else 1.0

    def printMetaArgs(self) -> str:
        return ""
