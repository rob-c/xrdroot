"""Every function and constant a ``TFormula`` may name, and how each is differentiated.

ROOT hands a formula to its C++ interpreter, so in principle any C++ is
allowed; in practice formulas are written in ``TMath``, ``<cmath>`` and a few
``ROOT::Math`` densities, and those are what is here: the ``TMath`` and C
functions every ``TTree::Draw`` expression knows, ROOT's own shorthands for
them - ``sq``, ``sign``, ``binomial`` - and the densities the predefined
shapes expand into. A name that is none of these is refused by the parser,
by name.

A function that has a simple derivative carries it, as the partial
derivative with respect to each argument, so that the gradient of a fit
function with respect to its parameters comes out exact wherever the
formula is built from these; one that does not is differentiated
numerically, and only in the parameters it actually depends on.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from ..formula.functions import FUNCTIONS as TREE_FUNCTIONS
from . import special

__all__ = ["Call", "CALLS", "CONSTANTS", "PHYSICAL"]

Array = Any
#: The partial derivatives of a function with respect to each of its
#: arguments, given the arguments' values.
Partials = Callable[..., "tuple[Array, ...]"]


class Call:
    """A function's arity, implementation and, where it has one, derivative."""

    __slots__ = ("least", "most", "apply", "partials")

    def __init__(
        self, least: int, most: int, apply: Callable[..., Array], partials: Partials | None = None
    ) -> None:
        self.least = least
        self.most = most
        self.apply = apply
        self.partials = partials


def _exp(u: Array) -> tuple[Array, ...]:
    return (np.exp(u),)


def _log(u: Array) -> tuple[Array, ...]:
    return (1.0 / u,)


def _sqrt(u: Array) -> tuple[Array, ...]:
    return (0.5 / np.sqrt(u),)


def _sin(u: Array) -> tuple[Array, ...]:
    return (np.cos(u),)


def _cos(u: Array) -> tuple[Array, ...]:
    return (-np.sin(u),)


def _square(u: Array) -> tuple[Array, ...]:
    return (2.0 * u,)


def _abs(u: Array) -> tuple[Array, ...]:
    return (np.sign(u),)


def power_partials(base: Array, exponent: Array) -> tuple[Array, ...]:
    """``d a^b / da`` and ``d a^b / db``; the second is taken only where it is needed."""
    with np.errstate(all="ignore"):
        return (exponent * np.power(base, exponent - 1), np.power(base, exponent) * np.log(base))


#: The derivative each differentiable function carries, by every spelling.
_PARTIALS: dict[str, Partials] = {
    **dict.fromkeys(("exp", "TMath::Exp", "std::exp"), _exp),
    **dict.fromkeys(("log", "TMath::Log", "std::log"), _log),
    **dict.fromkeys(("sqrt", "TMath::Sqrt", "std::sqrt"), _sqrt),
    **dict.fromkeys(("sin", "TMath::Sin", "std::sin"), _sin),
    **dict.fromkeys(("cos", "TMath::Cos", "std::cos"), _cos),
    **dict.fromkeys(("sq", "TMath::Sq"), _square),
    **dict.fromkeys(("abs", "fabs", "TMath::Abs", "std::abs", "std::fabs"), _abs),
    **dict.fromkeys(("pow", "TMath::Power", "std::pow"), power_partials),
}


def _sign(a: Array, b: Array | None = None) -> Array:
    """``TMath::Sign(a, b)``; with one argument, as older formulas write it, the sign alone."""
    if b is None:
        return np.sign(a)
    return np.where(np.asarray(b) >= 0, np.abs(a), -np.abs(a))


def _binomial(n: Array, k: Array) -> Array:
    """``TMath::Binomial``: ways of choosing ``k`` of ``n``, zero where there are none."""

    def one(count: float, chosen: float) -> float:
        whole, part = int(count), int(chosen)
        if whole < 0 or part < 0 or part > whole:
            return 0.0
        return float(math.comb(whole, part))

    return np.vectorize(one, otypes=[np.float64])(n, k)


def _cheb(degree: int) -> Call:
    """``ROOT::Math::ChebyshevN``, which takes the variable and ``N + 1`` coefficients."""
    return Call(degree + 2, degree + 2, special.chebyshev)


#: The functions of ROOT's own that a predefined shape expands into, or that
#: a formula names from ``TMath`` beyond what an expression over a tree knows.
_ROOT: dict[str, Call] = {
    "TMath::Gaus": Call(1, 4, special.gaus),
    "TMath::Landau": Call(1, 4, special.landau),
    "TMath::BreitWigner": Call(1, 3, special.breit_wigner),
    "TMath::Sign": Call(1, 2, _sign),
    "sign": Call(1, 2, _sign),
    "TMath::Binomial": Call(2, 2, _binomial),
    "binomial": Call(2, 2, _binomial),
    "ROOT::Math::landau_pdf": Call(1, 3, special.landau_pdf),
    "ROOT::Math::gaussian_pdf": Call(1, 3, special.gaussian_pdf),
    "ROOT::Math::breitwigner_pdf": Call(2, 3, special.breitwigner_pdf),
    "ROOT::Math::crystalball_function": Call(4, 5, special.crystalball_function),
    "ROOT::Math::crystalball_pdf": Call(4, 5, special.crystalball_pdf),
    "ROOT::Math::bigaussian_pdf": Call(2, 7, special.bigaussian_pdf),
    **{f"ROOT::Math::Chebyshev{degree}": _cheb(degree) for degree in range(11)},
}


def _from_tree(name: str, function: Any) -> Call:
    return Call(function.least, function.most, function.apply, _PARTIALS.get(name))


#: Every function a formula may call, by every spelling ROOT accepts.
CALLS: dict[str, Call] = {
    **{name: _from_tree(name, function) for name, function in TREE_FUNCTIONS.items()},
    **_ROOT,
}

#: The constants ``TFormula`` defines, by the names a formula writes them.
CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "sqrt2": math.sqrt(2.0),
    "ln10": math.log(10.0),
    "loge": math.log10(math.e),
    "infinity": math.inf,
    "eg": 0.577215664901532860606512090082402431042,
    "c": 299792458.0,
    "true": 1.0,
    "false": 0.0,
    "kTRUE": 1.0,
    "kFALSE": 0.0,
}

#: The physical constants ``TFormula`` also defines, whose values ROOT has
#: moved between CODATA releases; a formula naming one is refused rather
#: than evaluated with a value the ROOT that wrote it may not have used.
PHYSICAL = ("g", "h", "k", "sigma", "r")
