"""``RooResolutionModel`` and ``RooTruthModel``: resolutions a decay is convolved with.

A decay's time distribution is a sum of a few *basis functions* of the
time - ``exp(-|t|/tau)``, the same times ``cos(dm t)`` - each with a
coefficient that does not depend on the time, and a resolution model knows
each basis function convolved with itself in closed form. So a
``RooDecay`` with a Gaussian resolution is ``RooGaussModel`` asked for its
value *convolved with* ``exp(-|t|/tau)``: a copy of the model called
``gm1_conv_exp(-abs(@0)/@1)_dt_tau_[decay]`` whose basis says which
closed form it is. A model with no basis is an ordinary density.

The basis is RooFit's formula string, and its code - ``basisCode`` - is the
kind (exponential, sine, cosine, linear, quadratic, cosh, sinh) times ten
plus the side: ``+1`` for positive times only, ``-1`` for negative, ``0``
for both. :class:`RooTruthModel` is the delta function: its convolutions
are the basis functions themselves.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..functions import RooFormulaVar
from ..pdf import RooAbsPdf
from ..real import Context
from .basic import ref

__all__ = [
    "BASIS_CODES",
    "GENERIC",
    "RooResolutionModel",
    "RooTruthModel",
    "basis_sign",
    "basis_type",
]

#: ``basisCode``: the basis formulas both models know - spaces removed - and their codes.
BASIS_CODES = {
    "exp(-@0/@1)": 3,
    "exp(@0/@1)": 1,
    "exp(-abs(@0)/@1)": 2,
    "exp(-@0/@1)*sin(@0*@2)": 13,
    "exp(@0/@1)*sin(@0*@2)": 11,
    "exp(-abs(@0)/@1)*sin(@0*@2)": 12,
    "exp(-@0/@1)*cos(@0*@2)": 23,
    "exp(@0/@1)*cos(@0*@2)": 21,
    "exp(-abs(@0)/@1)*cos(@0*@2)": 22,
    "(@0/@1)*exp(-@0/@1)": 33,
    "(@0/@1)*(@0/@1)*exp(-@0/@1)": 43,
    "exp(-@0/@1)*cosh(@0*@2/2)": 53,
    "exp(@0/@1)*cosh(@0*@2/2)": 51,
    "exp(-abs(@0)/@1)*cosh(@0*@2/2)": 52,
    "exp(-@0/@1)*sinh(@0*@2/2)": 63,
    "exp(@0/@1)*sinh(@0*@2/2)": 61,
    "exp(-abs(@0)/@1)*sinh(@0*@2/2)": 62,
}
#: ``genericBasis``: the truth model's code for a basis it has no closed form for.
GENERIC = 100
#: ``BasisType``: none, exp, sin, cos, lin, quad, cosh, sinh.
NONE, EXP, SIN, COS, LIN, QUAD, COSH, SINH = range(8)


def basis_type(code: int) -> int:
    return 0 if code == 0 else code // 10 + 1


def basis_sign(code: int) -> int:
    """``BasisSign``: ``+1`` for a basis of positive times, ``-1`` of negative, ``0`` of both."""
    return code - 10 * (basis_type(code) - 1) - 2


def known_code(expression: str) -> int:
    return BASIS_CODES.get(expression.replace(" ", ""), 0)


class RooResolutionModel(RooAbsPdf):
    """A resolution: a density, or - with a basis - one convolved with a basis function."""

    def __init__(self, name: Any, title: Any, x: Any) -> None:
        super().__init__(name, title)
        self.x = self._proxy("x", x)
        self._basis: Any = None
        self._basis_code = 0

    def basisCode(self, name: str) -> int:
        """The code of the basis ``name``, zero if this model has no closed form for it."""
        return known_code(name)

    def isBasisSupported(self, name: str) -> bool:
        return self.basisCode(name) != 0

    def convVar(self) -> Any:
        return self.x

    def basis(self) -> Any:
        return self._basis

    def convolution(self, basis: Any, owner: Any) -> Any:
        """``convolution``: this model convolved with ``basis``, named for it and ``owner``."""
        made = self.clone(f"{self.GetName()}_conv_{basis.GetName()}_[{owner.GetName()}]")
        made.SetTitle(f"{made.GetTitle()} convoluted with basis function {basis.GetName()}")
        made.changeBasis(basis)
        return made

    def changeBasis(self, basis: Any) -> None:
        """Convolve with ``basis`` - or nothing - from now on: its inputs become this model's."""
        self._proxies = [one for one in self._proxies if one.name != "!basis"]
        self._basis = basis
        self._basis_code = self.basisCode(basis.GetTitle()) if basis is not None else 0
        if basis is not None:
            self._list_proxy("!basis", basis.dependents_list())

    def basis_values(self, ctx: Context) -> tuple[Any, Any]:
        """The basis's lifetime and its second parameter (a frequency), zero where absent."""
        args = self._basis.dependents_list() if self._basis is not None else []
        first = args[1].compute(ctx) if len(args) > 1 else 0.0
        second = args[2].compute(ctx) if len(args) > 2 else 0.0
        return first, second

    def value(self, ctx: Context, nset: Any = None, rng: Any = None) -> Any:
        """A convolution is not normalised on its own: the density it is part of normalises it."""
        if self._basis is not None:
            return self.compute(ctx)
        return super().value(ctx, nset, rng)

    def is_direct_gen_safe(self, name: str) -> bool:
        """``isDirectGenSafe``: ``name`` is an input and no other input depends on it."""
        servers = self.servers()
        if not any(one.GetName() == name for one in servers):
            return False
        return not any(one.GetName() != name and name in one.dependents() for one in servers)

    def is_truth(self) -> bool:
        return False


def _sided(x: Any, sign: int, value: Any) -> Any:
    """``value`` where the basis lives, zero on the side of zero it does not."""
    outside = (
        (x > 0.0) if sign < 0 else (x < 0.0) if sign > 0 else np.zeros(np.shape(x), dtype=bool)
    )
    return np.where(outside, 0.0, value)


def _truth_basis(kind: int, x: Any, tau: Any, dm: Any) -> Any:
    """``computeTruthModel*Basis``: the basis itself, as RooFit's batch kernels compute it."""
    decay = np.exp(-np.abs(x) / tau)
    if kind == SIN:
        return decay * np.sin(x * dm)
    if kind == COS:
        return decay * np.cos(x * dm)
    if kind in (LIN, QUAD):
        scaled = np.abs(x) / tau
        return np.exp(-scaled) * scaled * (scaled if kind == QUAD else 1.0)
    if kind == SINH:
        return decay * np.sinh(x * dm * 0.5)
    if kind == COSH:
        return decay * np.cosh(x * dm * 0.5)
    return decay


class RooTruthModel(RooResolutionModel):
    """``RooTruthModel``: a delta function - so its convolution with a basis is the basis itself."""

    def basisCode(self, name: str) -> int:
        return known_code(name) or GENERIC

    def is_truth(self) -> bool:
        return True

    def changeBasis(self, basis: Any) -> None:
        super().changeBasis(basis)
        if self._basis_code == GENERIC:
            self._proxies = [one for one in self._proxies if one.name != "!basis"]
            self._proxy("!basis", basis)

    def compute(self, ctx: Context) -> Any:
        x = np.asarray(self.x.compute(ctx), dtype=np.float64)
        if self._basis_code == 0:
            return np.where(x == 0.0, 1.0, 0.0)
        if self._basis_code == GENERIC:
            return self._basis.compute(ctx)
        tau, dm = self.basis_values(ctx)
        with np.errstate(all="ignore"):
            found = _truth_basis(basis_type(self._basis_code), x, tau, dm)
        return _sided(x, basis_sign(self._basis_code), found)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        if self._basis_code == GENERIC:
            return frozenset()
        return frozenset([self.x.GetName()]) & names

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        if self._basis_code == 0:
            return 1.0
        from .truthint import definite

        kind = basis_type(self._basis_code)
        tau, dm = self.basis_values(ctx)
        dm = dm if kind in (SIN, COS, SINH, COSH) else math.nan
        return definite(
            kind, self.x.getMin(rng), self.x.getMax(rng), tau, dm, basis_sign(self._basis_code)
        )

    def generator_code(self, names: frozenset[str]) -> int:
        return 1 if names == frozenset([self.x.GetName()]) else 0

    def generate_event(self, code: int, rng: Any, bounds: Any = None) -> dict[str, float]:
        """``generateEvent``: the delta function's only value, zero."""
        return {self.x.GetName(): 0.0}


def make_basis(owner: Any, expression: str, params: Any) -> Any:
    """``declareBasis``'s formula: ``exp(-abs(@0)/@1)_dt_tau``, of the time and ``params``."""
    from ..collections import as_list

    args = [owner.conv_var(), *(ref(one) for one in as_list(params))]
    name = expression + "".join(f"_{one.GetName()}" for one in args)
    return RooFormulaVar(name, expression, args)
