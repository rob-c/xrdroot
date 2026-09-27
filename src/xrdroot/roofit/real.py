"""``RooAbsReal``: a node with a real value, and how that value is computed over data.

Every function and density in a model is one of these. Its value is
computed by :meth:`RooAbsReal.compute` from a *context*, a mapping from
variable names to values: a variable the context names takes that value -
a whole column of a dataset at once, as a NumPy array - and one it does not
takes its own. So one call evaluates a model at every event of a dataset,
and the same call with an empty context is ``getVal()``.

Integrals are here too, because every density needs its own to be
normalised: a class that knows its integral over some of its variables in
closed form says which (:meth:`RooAbsReal.analytic_names`) and gives it
(:meth:`RooAbsReal.analytic`), exactly as RooFit's
``getAnalyticalIntegral``/``analyticalIntegral`` pair does, and whatever is
left is integrated numerically by :mod:`.integration`, as
``RooRealIntegral`` does.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .arg import RooAbsArg
from .collections import as_list
from .printing import g

__all__ = ["Context", "RooAbsReal", "value_of"]

#: A context: variable names to their values - a number, or an array of one per event.
Context = dict


def value_of(result: Any) -> float:
    """A computed value as a plain float."""
    return float(np.asarray(result, dtype=np.float64).reshape(-1)[0])


class RooAbsReal(RooAbsArg):
    """Anything with a real value: a variable, a function, a density."""

    def __init__(self, name: Any = "", title: Any = "", unit: str = "") -> None:
        super().__init__(name, title)
        self._unit = str(unit)
        self._plot_label = ""

    # -- values -------------------------------------------------------------------

    def compute(self, ctx: Context) -> Any:
        """RooFit's ``evaluate()``: the value, for the values ``ctx`` gives the variables."""
        raise NotImplementedError

    def value(self, ctx: Context, nset: Any = None, rng: Any = None) -> Any:
        """The value normalised over ``nset`` in range ``rng``: a function ignores both."""
        return self.compute(ctx)

    def getVal(self, nset: Any = None) -> float:
        """The value now, normalised over the variables in ``nset`` if it is a density."""
        return value_of(self.value({}, names_in(nset)))

    getValV = getVal

    def __float__(self) -> float:
        return self.getVal()

    def getUnit(self) -> str:
        return self._unit

    def setUnit(self, unit: str) -> None:
        self._unit = str(unit)

    def getPlotLabel(self) -> str:
        return self._plot_label or self.GetName()

    def setPlotLabel(self, label: str) -> None:
        self._plot_label = str(label)

    def getTitle(self, appendUnit: bool = False) -> str:
        title = self.GetTitle()
        if appendUnit and self._unit:
            title += f" ({self._unit})"
        return title

    def isValueDirty(self) -> bool:
        return True

    # -- integrals ----------------------------------------------------------------

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        """Which of ``names`` this class integrates over in closed form: none, by default."""
        return frozenset()

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        """The integral over ``names`` - a set :meth:`analytic_names` gave - in range ``rng``."""
        raise NotImplementedError

    def integral_code(self, names: frozenset[str]) -> int:
        """The code ``getAnalyticalIntegral`` returns for ``names``, which ROOT prints."""
        return 1

    def variable(self, name: str) -> Any:
        """The variable under this node called ``name``."""
        return next(one for one in self.leaves() if one.GetName() == name)

    def bounds(self, name: str, rng: Any) -> tuple[float, float]:
        """The ends of the range ``rng`` of the variable ``name``."""
        var = self.variable(name)
        return var.getMin(rng), var.getMax(rng)

    def integrate(self, names: Any, ctx: Context, rng: Any = None) -> Any:
        """The integral over ``names`` of the value, in ``rng``, the rest as ``ctx`` has them."""
        from .integration import integral

        return integral(self, frozenset(names), ctx, rng)

    def createIntegral(self, iset: Any, *args: Any, **kwargs: Any) -> Any:
        """``createIntegral(iset, [nset], [range])``: the integral as a function of the rest."""
        from .integral import make_integral

        return make_integral(self, iset, args, kwargs)

    # -- printing -----------------------------------------------------------------

    def printValue(self) -> str:
        return g(self.getVal())

    def printMultiline(self, contents: int, verbose: bool, indent: str) -> str:
        return (
            f"{indent}--- RooAbsReal ---\n\n{indent}  Plot label is \"{self.getPlotLabel()}\"\n"
        )


def names_in(nset: Any) -> frozenset[str] | None:
    """The names of the variables in a normalisation set, or ``None`` for none given."""
    if nset is None:
        return None
    return frozenset(one.GetName() for one in as_list(nset))
