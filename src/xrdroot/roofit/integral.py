"""``createIntegral`` and ``createCdf``: an integral of a function, as a function of the rest.

``g.createIntegral(x)`` is ``RooRealIntegral``: the integral of ``g`` over
``x``, a function of ``g``'s other variables, called ``g_Int[x]``; with
``NormSet`` it is divided by the integral over those variables' full ranges
(``g_Int[x|signal]_Norm[x]``), and with ``Range`` taken over a named range.
``createCdf`` is the integral from the lower end up to the variable's value.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .binning import RooRangeBinning
from .cmdargs import RooCmdArg, commands
from .collections import as_list
from .integration import integral_name
from .messages import INFO, log
from .printing import g
from .real import Context, RooAbsReal

__all__ = ["RooRealIntegral", "make_integral", "make_cdf"]


class RooRealIntegral(RooAbsReal):
    """The integral of a function over some of its variables."""

    def __init__(self, func: Any, names: frozenset[str], nset: frozenset[str] | None, rng: Any) -> None:
        name = integral_name(func, names, rng)
        if nset:
            order = [one.GetName() for one in func.leaves() if one.GetName() in nset]
            name += f"_Norm[{','.join(order)}]"
        super().__init__(name, f"Integral of {func.GetTitle()}")
        self.func = self._proxy("!func", func)
        self.names, self.nset, self.rng = names, nset, rng
        self._announce()

    def _announce(self) -> None:
        func, over = self.func, ",".join(sorted(self.names))
        norm = ",".join(sorted(self.nset or ()))
        log(self, INFO, "Integration", f"RooRealIntegral::ctor({self._name}) Constructing integral of "
            f"function {func.GetName()} over observables({over}) with normalization ({norm}) with "
            f"range identifier {self.rng or '<none>'}")  # fmt: skip
        closed = func.analytic_names(self.names, self.rng) & self.names
        for name in sorted(self.names):
            log(self, INFO, "Integration", f"{func.GetName()}: Observable {name} is suitable for "
                "analytical integration (if supported by p.d.f)")  # fmt: skip
        if closed:
            code = func.integral_code(closed)
            text = ",".join(sorted(closed))
            log(self, INFO, "Integration", f"{func.GetName()}: Function integrated observables ({text}) "
                f"internally with code {code}")  # fmt: skip
            log(self, INFO, "Integration", f"{func.GetName()}: Observables ({text}) are analytically "
                f"integrated with code {code}")  # fmt: skip

    def compute(self, ctx: Context) -> Any:
        if self.nset:
            return self.func.fraction(self.names, ctx, self.nset, self.rng)
        return self.func.integrate(self.names, ctx, self.rng)

    def printMetaArgs(self) -> str:
        closed = self.func.analytic_names(self.names, self.rng) & self.names
        kind = "Ana" if closed == self.names else "Num"
        return f"Int {self.func.GetName()}d[{kind}]({','.join(sorted(self.names))}) "

    def printArgs(self) -> str:
        return "[ " + self.printMetaArgs() + "]"


def make_integral(func: Any, iset: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> RooRealIntegral:
    """``func.createIntegral(iset, [nset], [range], options...)``."""
    names = frozenset(one.GetName() for one in as_list(iset))
    plain = [a for a in args if not isinstance(a, RooCmdArg)]
    options = commands([a for a in args if isinstance(a, RooCmdArg)], kwargs)
    nset = options.get("NormSet") or options.get("SupNormSet")
    rng = options.get("Range")
    for arg in plain:
        if isinstance(arg, str):
            rng = arg
        else:
            nset = arg
    norm = frozenset(one.GetName() for one in as_list(nset)) if nset is not None else None
    return RooRealIntegral(func, names, norm, rng)


class RooCdf(RooAbsReal):
    """``createCdf``: the normalised integral from the lower end up to the variable's value."""

    def __init__(self, func: Any, names: frozenset[str]) -> None:
        order = [one.GetName() for one in func.leaves() if one.GetName() in names]
        super().__init__(f"{func.GetName()}_Int[{','.join(order)}_prime]_Norm[{','.join(order)}]",
                         f"Integral of {func.GetTitle()}")  # fmt: skip
        self.func = self._proxy("!func", func)
        self.names = names

    def compute(self, ctx: Context) -> Any:
        name = next(iter(self.names))
        var = self.func.variable(name)
        values = np.atleast_1d(np.asarray(ctx.get(name, var.getVal()), dtype=np.float64))
        found = np.empty_like(values)
        for i, upper in enumerate(values):
            var._shared["_cdf_"] = RooRangeBinning(var.getMin(), float(upper), "_cdf_")
            rest = {k: v for k, v in ctx.items() if k != name}
            found[i] = float(np.asarray(self.func.fraction(self.names, rest, self.names, "_cdf_")))
        var._shared.pop("_cdf_", None)
        return found if np.ndim(ctx.get(name, 0.0)) else float(found[0])

    def printValue(self) -> str:
        return g(self.getVal())


def make_cdf(func: Any, iset: Any) -> RooCdf:
    return RooCdf(func, frozenset(one.GetName() for one in as_list(iset)))
