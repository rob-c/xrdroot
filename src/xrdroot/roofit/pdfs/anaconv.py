"""``RooAbsAnaConvPdf``: a density of basis functions, each convolved with a resolution.

A decay is ``sum_k coef_k * basis_k(t)``; convolved with a resolution
model it is ``sum_k coef_k * (basis_k (x) R)(t)``, and the model knows
each ``basis_k (x) R`` (:mod:`.resolution`). A subclass declares its bases
(:meth:`RooAbsAnaConvPdf.declareBasis`) and gives their coefficients
(:meth:`RooAbsAnaConvPdf.coef`); this class sums them, and integrates them
the way RooFit does: the variables no convolution depends on - the tag and
mixing categories - through the coefficients' own integrals
(``getCoefAnalyticalIntegral``, summed over the states where there is
none), the rest through the convolutions'.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from ..collections import RooArgList
from ..messages import ERROR, log
from ..pdf import RooAbsPdf
from ..real import Context, RooAbsReal
from .resolution import make_basis

__all__ = ["DECAY_TYPES", "RooAbsAnaConvPdf", "decay_type"]

#: ``DecayType``: ``SingleSided``, ``DoubleSided`` and ``Flipped``, by name.
DECAY_TYPES = {"SingleSided": 0, "DoubleSided": 1, "Flipped": 2}


def decay_type(value: Any) -> int:
    """A decay type given as ROOT's enumerator - a number - or, as PyROOT allows, by its name."""
    if isinstance(value, str):
        if value not in DECAY_TYPES:
            raise ValueError(
                f"'{value}' is not a decay type: use SingleSided, DoubleSided or Flipped."
            )
        return DECAY_TYPES[value]
    return int(value)


def _is_category(arg: Any) -> bool:
    return hasattr(arg, "lookupIndex") and arg.isFundamental()


class CoefVar(RooAbsReal):
    """``RooConvCoefVar``: one basis function's coefficient, as a function to integrate."""

    def __init__(self, pdf: Any, index: int) -> None:
        super().__init__(f"{pdf.GetName()}_coefVar_{index}", "coefVar")
        self.pdf, self.index = pdf, index
        self._list_proxy("!coefVars", pdf.coef_servers())

    def compute(self, ctx: Context) -> Any:
        return self.pdf.coef(self.index, ctx)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        cats = frozenset(one.GetName() for one in self.leaves() if _is_category(one)) & names
        return cats | frozenset(self.pdf.coef_analytic_names(self.index, names, rng))

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        """The closed form where the density has one, summed over the other categories' states."""
        closed = self.pdf.coef_analytic_names(self.index, names, rng) & names
        summed = [one for one in self.leaves() if one.GetName() in names - closed]
        total: Any = 0.0
        for states in itertools.product(*(list(one.states().values()) for one in summed)):
            point = dict(ctx)
            point.update({one.GetName(): float(state) for one, state in zip(summed, states)})
            total = total + self.pdf.coef_analytic(self.index, closed, point, rng)
        return total


class RooAbsAnaConvPdf(RooAbsPdf):
    """A density of basis functions convolved with a resolution model, each in closed form."""

    #: ``DecayType``, as the subclasses' class members.
    SingleSided, DoubleSided, Flipped = 0, 1, 2

    def __init__(self, name: Any, title: Any, model: Any, cvar: Any) -> None:
        super().__init__(name, title)
        self.model = self._proxy("!model", model)
        self._cvar = self._proxy("!convVar", cvar)
        self.convs = self._list_proxy("!convSet", [])
        self._basis_list: list[Any] = []

    def conv_var(self) -> Any:
        return self._cvar

    def convVar(self) -> Any:
        return self._cvar

    def declareBasis(self, expression: str, params: Any) -> int:
        """``declareBasis``: the basis ``expression`` of the time and ``params``; its index."""
        if not self.model.isBasisSupported(expression):
            log(
                self,
                ERROR,
                "InputArguments",
                f"RooAbsAnaConvPdf::declareBasis({self.GetName()}): resolution "
                f"model {self.model.GetName()} doesn't support basis function {expression}",
            )
            return -1
        basis = make_basis(self, expression, params)
        self._basis_list.append(basis)
        self.convs.add(self.model.convolution(basis, self))
        return len(self.convs) - 1

    def with_model(self, model: Any) -> Any:
        """``changeModel``: a copy with ``model`` - as a generator uses the truth model."""
        made = self.clone()
        made.model = model
        made.convs = RooArgList([model.convolution(basis, made) for basis in self._basis_list])
        for proxy in made._proxies:
            if proxy.name == "!model":
                proxy.target = model
            elif proxy.name == "!convSet":
                proxy.target = made.convs
        return made

    # -- the coefficients: what a subclass gives ----------------------------------

    def coef(self, index: int, ctx: Context) -> Any:
        """``coefficient(index)``: the coefficient of the basis function ``index``."""
        raise NotImplementedError

    def coefficient(self, index: int) -> float:
        return float(np.asarray(self.coef(index, {})))

    def coef_analytic_names(self, index: int, names: frozenset[str], rng: Any) -> frozenset[str]:
        """``getCoefAnalyticalIntegral``: which ``names`` the coefficients have closed forms for."""
        return frozenset()

    def coef_analytic(self, index: int, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        """``coefAnalyticalIntegral``: the coefficient integrated over ``names``, or itself."""
        return self.coef(index, ctx)

    def coef_servers(self) -> list[Any]:
        """What the coefficients are made of: every input but the model and the convolutions."""
        skip = {"!model", "!convVar", "!convSet"}
        return [one for proxy in self._proxies if proxy.name not in skip for one in proxy.args()]

    # -- values and integrals -----------------------------------------------------

    def _terms(self, ctx: Context, part: Any) -> Any:
        total: Any = 0.0
        for index, conv in enumerate(self.convs):
            weight = part(index)
            if np.any(np.asarray(weight) != 0.0):
                total = total + np.where(
                    np.asarray(weight) != 0.0, conv_value(conv, ctx) * weight, 0.0
                )
        return total

    def compute(self, ctx: Context) -> Any:
        return self._terms(ctx, lambda index: self.coef(index, ctx))

    def _split(self, names: frozenset[str]) -> tuple[frozenset[str], frozenset[str]]:
        """The variables no convolution depends on - the coefficients' - and the convolutions'."""
        conv_deps: set[str] = set()
        for conv in self.convs:
            conv_deps |= conv.dependents()
        return frozenset(names - conv_deps), frozenset(names & conv_deps)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return frozenset(names) & self.dependents()

    def integral_code(self, names: frozenset[str]) -> int:
        return 1

    def announce_inner(self, names: frozenset[str], rng: Any) -> None:
        """The convolutions' own integrals over ``names``, said the first time each is made.

        RooFit integrates the density in closed form by integrating each
        convolution over its variables (``conv->getNorm(intConvSet)``); a
        convolution without a closed form in one of them - a per-event error
        scaling the resolution - is a numerical ``RooRealIntegral`` then,
        announced once and kept in the convolution's cache.
        """
        from ..integration import announce

        conv_set = self._split(frozenset(names) & self.dependents())[1]
        for conv in self.convs if conv_set else ():
            made = conv.__dict__.setdefault("_integrals_made", set())
            if (conv_set, rng) not in made:
                made.add((conv_set, rng))
                announce(conv, conv_set, rng)

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        coef_set, conv_set = self._split(frozenset(names))
        total: Any = 0.0
        for index, conv in enumerate(self.convs):
            weight = (
                CoefVar(self, index).integrate(coef_set, ctx, rng)
                if coef_set
                else self.coef(index, ctx)
            )
            if np.any(np.asarray(weight) != 0.0):
                total = total + weight * conv.integrate(conv_set, ctx, rng)
        return total

    def normalized_name(self, observables: Any, rng: Any = None) -> str:
        """A single convolution is what RooFit's fit evaluates, so its name is the one printed."""
        if len(self.convs) == 1:
            return str(self.convs[0].normalized_name(observables, rng))
        return super().normalized_name(observables, rng)

    # -- generation ---------------------------------------------------------------

    def is_direct_gen_safe(self, name: str) -> bool:
        """``isDirectGenSafe``: the time with the truth model; else an input nothing else uses."""
        if name == self._cvar.GetName() and self.model.is_truth():
            return True
        if not any(one.GetName() == name for one in self.servers()):
            return False
        return not any(one.GetName() != name and name in one.dependents() for one in self.servers())

    def gen_context(self, names: frozenset[str], proto: Any = None) -> Any:
        from ..generation.convolution import context_for_convolution

        return context_for_convolution(self.snapshot(), names, proto)

    def snapshot(self) -> Any:
        """The copy a generator makes (``RooArgSet::snapshot``), inputs matched *by name*.

        RooFit's deep copy clones each input the first time its name is met,
        walking the inputs depth first, and connects every input of that name
        to that one clone - so two different variables that share a name
        become one, the first met. This density with its inputs matched the
        same way; itself when no two of them share a name.
        """
        first: dict[str, Any] = {self.GetName(): self}

        def walk(node: Any) -> None:
            for server in node.servers():
                if server.GetName() not in first:
                    first[server.GetName()] = server
                    walk(server)

        walk(self)
        moved = {
            id(one): first[one.GetName()]
            for one in self.servers()
            if first[one.GetName()] is not one
        }
        if not moved:
            return self
        made = self.clone()
        for proxy in made._proxies:
            if not proxy.many and id(proxy.target) in moved:
                proxy.target = moved[id(proxy.target)]
        for key, one in list(vars(made).items()):
            if id(one) in moved:
                setattr(made, key, moved[id(one)])
        return made

    def printMultiline(self, contents: int, verbose: bool, indent: str) -> str:
        text = (
            super().printMultiline(contents, verbose, indent)
            + f"{indent}--- RooAbsAnaConvPdf ---\n"
        )
        return text + "".join(conv.printMultiline(contents, verbose, indent) for conv in self.convs)


def conv_value(conv: Any, ctx: Context) -> Any:
    return conv.compute(ctx)
