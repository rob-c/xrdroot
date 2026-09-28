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

    def __init__(
        self,
        func: Any,
        names: frozenset[str],
        nset: frozenset[str] | None,
        rng: Any,
        others: Any = (),
    ) -> None:
        name = integral_name(func, names, rng)
        if nset:
            order = [one.GetName() for one in func.leaves() if one.GetName() in nset]
            name += f"_Norm[{','.join(order)}]"
        super().__init__(name, f"Integral of {func.GetTitle()}")
        self.func = self._proxy("!func", func)
        self.names, self.nset, self.rng = names, nset, rng
        self._others = list(others)
        self._announce()  # the numerical part is said by who made it: see _recursive

    def announce_numeric(self) -> None:
        """``RooRealIntegral::init``'s line, if part of the integral is numerical."""
        from .integration import announce

        announce(self.func, self.names, self.rng, self._name)

    def _announce(self) -> None:
        func, over = self.func, ",".join(sorted(self.names))
        norm = ",".join(sorted(self.nset or ()))
        log(
            self,
            INFO,
            "Integration",
            f"RooRealIntegral::ctor({self._name}) Constructing integral of "
            f"function {func.GetName()} over observables({over}) with normalization ({norm}) with "
            f"range identifier {self.rng or '<none>'}",
        )
        closed = func.analytic_names(self.names, self.rng) & self.names
        for name in sorted(self.names):
            log(
                self,
                INFO,
                "Integration",
                f"{func.GetName()}: Observable {name} is suitable for "
                "analytical integration (if supported by p.d.f)",
            )
        if closed:
            code = func.integral_code(closed)
            text = ",".join(sorted(closed))
            log(
                self,
                INFO,
                "Integration",
                f"{func.GetName()}: Function integrated observables ({text}) "
                f"internally with code {code}",
            )
            log(
                self,
                INFO,
                "Integration",
                f"{func.GetName()}: Observables ({text}) are analytically "
                f"integrated with code {code}",
            )

    def _factorised(self) -> list[Any]:
        """The variables integrated over that the function does not depend on: each is a width."""
        return [one for one in self._others if one.GetName() not in self.func.dependents()]

    def compute(self, ctx: Context) -> Any:
        if self.nset:
            return self.func.fraction(self.names, ctx, self.nset, self.rng)
        found = self.func.integrate(self.names, ctx, self.rng)
        for one in self._factorised():
            found = found * (one.getMax(self.rng) - one.getMin(self.rng))
        return found

    def printMetaArgs(self) -> str:
        """``Int f_Norm(y) d[Ana](x) d[Num](y)``: the function, and how each variable is done."""
        names = self.names & self.func.dependents()
        closed = self.func.analytic_names(names, self.rng) & names
        analytic = sorted(closed | {one.GetName() for one in self._factorised()})
        numeric = sorted(names - closed)
        text = f"Int {self.func.GetName()}"
        if self.nset:
            text += f"_Norm({','.join(sorted(self.nset))}) "
        if analytic:
            text += f"d[Ana]({','.join(analytic)}) "
        if numeric:
            text += f" d[Num]({','.join(numeric)}) "
        return text

    def printArgs(self) -> str:
        return "[ " + self.printMetaArgs() + "]"


def make_integral(
    func: Any, iset: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> RooRealIntegral:
    """``func.createIntegral(iset, [nset], [range], options...)``."""
    names = frozenset(one.GetName() for one in as_list(iset))
    options = commands([a for a in args if isinstance(a, RooCmdArg)], kwargs)
    nset, rng = _norm_and_range(args, options)
    norm = frozenset(one.GetName() for one in as_list(nset)) if nset is not None else None
    others = [one for one in as_list(iset) if one.GetName() not in func.dependents()]
    return _recursive(func, names, norm, rng, others)


def _recursive(
    func: Any, names: frozenset[str], nset: Any, rng: Any, others: list[Any]
) -> RooRealIntegral:
    """``createIntObj``: integrals of integrals, when ranges are functions of other variables.

    A variable whose range in ``rng`` has ends made of other integrated
    variables - ``y`` in ``[0.1*x, 0.9*x]`` - is integrated only after those
    it is not an end of, each step integrating the last (only the innermost
    normalised): ``p_Int[z|R]_Norm[x,y,z]_Int[y|R]_Int[x|R]``. RooFit makes a
    numerical integrator when first evaluating, outermost first, so the
    steps say so in that order.
    """
    steps: list[RooRealIntegral] = []
    left = frozenset(names)
    while True:
        inner = _innermost(func, left, rng)
        if not inner or inner == left:
            inner = left
        mine = [one for one in others if one.GetName() in inner]
        steps.append(RooRealIntegral(func, inner, nset, rng, mine))
        func, nset, left = steps[-1], None, left - inner
        if not left:
            break
    for step in reversed(steps):
        step.announce_numeric()
    return steps[-1]


def _innermost(func: Any, names: frozenset[str], rng: Any) -> frozenset[str]:
    """``findInnerMostIntegration``: the variables no other's range in ``rng`` is made of."""
    if not rng:
        return names
    serving: set[str] = set()
    for leaf in func.leaves():
        if leaf.GetName() not in names or not hasattr(leaf, "getBinning"):
            continue
        binning = leaf.getBinning(rng)
        if binning.isParameterized():
            ends = binning.servers()
            serving |= set().union(*(end.dependents() for end in ends)) & names
    return names - serving


def _norm_and_range(args: tuple[Any, ...], options: Any) -> tuple[Any, Any]:
    """The normalisation set and range: as options, or as plain arguments - a set, a name."""
    nset = options.get("NormSet") or options.get("SupNormSet")
    rng = options.get("Range")
    for arg in args:
        if isinstance(arg, str):
            rng = arg
        elif not isinstance(arg, RooCmdArg):
            nset = arg
    return nset, rng


class RooCdf(RooAbsReal):
    """``createCdf``: the normalised integral from the lower end up to the variable's value."""

    def __init__(self, func: Any, names: frozenset[str]) -> None:
        order = [one.GetName() for one in func.leaves() if one.GetName() in names]
        super().__init__(
            f"{func.GetName()}_Int[{','.join(order)}_prime]_Norm[{','.join(order)}]",
            f"Integral of {func.GetTitle()}",
        )
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
