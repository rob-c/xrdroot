"""``RooGenericPdf``: a density written as a formula, normalised numerically.

``RooGenericPdf("g", "g", "exp(-x/tau)", [x, tau])`` has no integral in
closed form, so RooFit - and this - integrates it numerically over its
observables whenever it is normalised, announcing so once per integral.
"""

from __future__ import annotations

from typing import Any

from ..formula import RooFormula
from ..functions import _formula_args
from ..pdf import RooAbsPdf
from ..real import Context

__all__ = ["RooGenericPdf"]


class RooGenericPdf(RooAbsPdf):
    """A density whose value is a formula of its variables."""

    def __init__(self, name: Any, title: Any, *args: Any) -> None:
        super().__init__(name, title)
        expression, dependents = _formula_args(args)
        self._expression = expression or str(title)
        self.formula = RooFormula(self._expression, dependents)
        self.actual = self._list_proxy("actualVars", self.formula.actual())

    def compute(self, ctx: Context) -> Any:
        return self.formula.evaluate(ctx)

    def printMetaArgs(self) -> str:
        return f'formula="{self._expression}" '

    def expression(self) -> str:
        return self._expression
