"""RooFit's functions that are not densities: formulas, polynomials, products and sums of values.

``RooFormulaVar("f", "f", "a*x+b", [a, x, b])`` is a formula over the
variables it names (:mod:`.formula`), ``RooPolyVar`` a polynomial without
the implied constant a density has, ``RooProduct`` and ``RooAddition``
products and sums. Each is a :class:`~xrdroot.roofit.real.RooAbsReal`,
usable wherever a variable is: as a density's mean, as a fraction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import mathfuncs as mf
from .collections import as_list
from .formula import RooFormula
from .messages import INFO, log
from .real import Context, RooAbsReal

__all__ = ["RooAddition", "RooFormulaVar", "RooPolyVar", "RooProduct"]


def _formula_args(args: tuple[Any, ...]) -> tuple[str, list[Any]]:
    """``(formula, dependents)`` or ``(dependents)`` - the title then is the formula."""
    if args and isinstance(args[0], str):
        return args[0], as_list(args[1]) if len(args) > 1 else []
    return "", as_list(args[0]) if args else []


class RooFormulaVar(RooAbsReal):
    """A formula of other RooFit values."""

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

    def dependents_list(self) -> list[Any]:
        return list(self.actual)

    def state_word(self) -> str:
        return "Clean"


class RooPolyVar(RooAbsReal):
    """``RooPolyVar``: ``a0 + a1 x + a2 x^2 ...`` from ``lowestOrder`` up."""

    def __init__(
        self, name: Any, title: Any, x: Any, coefList: Any = (), lowestOrder: int = 0
    ) -> None:
        from .pdfs.basic import ref

        super().__init__(name, title)
        self.x = self._proxy("x", ref(x))
        self.coefs = self._list_proxy("coefList", [ref(c) for c in as_list(coefList)])
        self._lowest = max(int(lowestOrder), 0)

    def compute(self, ctx: Context) -> Any:
        coefs = [c.compute(ctx) for c in self.coefs]
        if not coefs:
            return 0.0
        return mf.polynomial(coefs, self._lowest, self.x.compute(ctx), False)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return frozenset([self.x.GetName()]) & names

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        coefs = [c.compute(ctx) for c in self.coefs]
        return mf.polynomial_integral(
            coefs, self._lowest, self.x.getMin(rng), self.x.getMax(rng), False
        )


class RooProduct(RooAbsReal):
    """``RooProduct``: the product of its terms."""

    def __init__(self, name: Any, title: Any, terms: Any) -> None:
        super().__init__(name, title)
        self.terms = self._list_proxy("compRSet", as_list(terms))

    def compute(self, ctx: Context) -> Any:
        found: Any = 1.0
        for term in self.terms:
            found = found * term.compute(ctx)
        return found

    def printArgs(self) -> str:
        return "[ " + " * ".join(term.GetName() for term in self.terms) + " ]"


class RooAddition(RooAbsReal):
    """``RooAddition``: the sum of its terms, or of the products of two lists."""

    def __init__(self, name: Any, title: Any, terms: Any, second: Any = None) -> None:
        super().__init__(name, title)
        self.terms = self._list_proxy("set", as_list(terms))
        self.second = self._list_proxy("set2", as_list(second)) if second is not None else None

    def compute(self, ctx: Context) -> Any:
        if self.second is None:
            values = [term.compute(ctx) for term in self.terms]
        else:
            values = [a.compute(ctx) * b.compute(ctx) for a, b in zip(self.terms, self.second)]
        return np.sum(np.broadcast_arrays(*values), axis=0) if values else 0.0

    def printArgs(self) -> str:
        if self.second is not None:
            return super().printArgs()
        return "[ " + " + ".join(term.GetName() for term in self.terms) + " ]"

    def defaultErrorLevel(self) -> float:
        """0.5 for a sum with a likelihood in it - RooFit says which - else 1."""
        from .fitting.nll import RooNLLVar

        found = next((one for one in self.getComponents() if isinstance(one, RooNLLVar)), None)
        if found is None:
            log(
                self,
                INFO,
                "Fitting",
                f"RooAddition::defaultErrorLevel({self._name}) WARNING: "
                "Summation contains neither RooNLLVar nor RooChi2Var server, "
                "using default level of 1.0",
            )
            return 1.0
        log(
            self,
            INFO,
            "Fitting",
            f"RooAddition::defaultErrorLevel({self._name}) Summation "
            "contains a RooNLLVar, using its error level",
        )
        return 0.5
