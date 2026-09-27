"""``RooAddModel``: a sum of resolution models - ``f gm1 + (1-f) gm2`` - that is a resolution model too.

Its convolution with a basis function is the sum of its components'
convolutions with the same fractions (``convolution`` builds a new sum of
the convolved components), and it supports a basis if its first component
does and none of the others refuses it - as ``RooAddModel::basisCode``
decides. Fractions are plain numbers here: a sum of convolutions is not
normalised on its own, the decay it is part of normalises it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..collections import as_list
from ..real import Context
from .basic import ref
from .resolution import RooResolutionModel

__all__ = ["RooAddModel"]


class RooAddModel(RooResolutionModel):
    """``RooAddModel(name, title, [models], [fractions])``: the last model takes what is left."""

    def __init__(
        self, name: Any, title: Any, pdfList: Any, coefList: Any, ownPdfList: bool = False
    ) -> None:
        pdfs, coefs = as_list(pdfList), [ref(one) for one in as_list(coefList)]
        if len(pdfs) > len(coefs) + 1 or len(pdfs) < len(coefs):
            raise ValueError(f"RooAddModel::RooAddModel({name}) number of pdfs and coefficients "
                             "inconsistent, must have Npdf=Ncoef or Npdf=Ncoef+1.")  # fmt: skip
        super().__init__(name, title, pdfs[0].convVar())
        self.pdfs = self._list_proxy("!pdfs", pdfs)
        self.coefs = self._list_proxy("!coefficients", coefs)
        self._have_last = len(pdfs) == len(coefs)

    def basisCode(self, name: str) -> int:
        """``basisCode``: one if the first component has the basis and no other refuses it."""
        codes = [model.basisCode(name) for model in self.pdfs]
        return int(bool(codes[0]) and all(codes[1:]))

    def convolution(self, basis: Any, owner: Any) -> Any:
        """The sum of the components' convolutions with ``basis``, with the same fractions."""
        made = RooAddModel(f"{self.GetName()}_conv_{basis.GetName()}_[{owner.GetName()}]",
                           f"{self.GetTitle()} convoluted with basis function {basis.GetName()}",
                           [model.convolution(basis, owner) for model in self.pdfs], list(self.coefs))  # fmt: skip
        made._attributes = set(self._attributes)
        made._strings = dict(self._strings)
        made.changeBasis(basis)
        return made

    def changeBasis(self, basis: Any) -> None:
        self._basis = basis
        self._basis_code = self.basisCode(basis.GetTitle()) if basis is not None else 0

    def fractions(self, ctx: Context) -> list[Any]:
        found = [one.compute(ctx) for one in self.coefs]
        if not self._have_last:
            last: Any = 1.0
            for one in found:
                last = last - one
            found.append(last)
        return found

    def _summed(self, parts: list[Any], ctx: Context) -> Any:
        total: Any = 0.0
        for fraction, part in zip(self.fractions(ctx), parts):
            total = total + np.where(np.asarray(fraction) != 0.0, part() * fraction, 0.0)
        return total

    def compute(self, ctx: Context) -> Any:
        return self._summed([lambda m=m: m.compute(ctx) for m in self.pdfs], ctx)

    def selfNormalized(self) -> bool:
        return self._basis is None

    def value(self, ctx: Context, nset: Any = None, rng: Any = None) -> Any:
        """With a basis, the plain sum; without, a sum of densities - each normalised on its own."""
        if self._basis is not None or not nset:
            return self.compute(ctx)
        return self._summed([lambda m=m: m.value(ctx, nset, rng) for m in self.pdfs], ctx)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return frozenset(names)

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        return self._summed([lambda m=m: m.integrate(names, ctx, rng) for m in self.pdfs], ctx)

    def is_direct_gen_safe(self, name: str) -> bool:
        return all(model.is_direct_gen_safe(name) for model in self.pdfs)

    def generator_code(self, names: frozenset[str]) -> int:
        return int(all(model.generator_code(names) for model in self.pdfs))

    def generate_event(self, code: int, rng: Any, bounds: Any = None) -> dict[str, float]:
        """``RooAddGenContext``: a uniform draw picks the component, which draws the event."""
        draw, low = rng.Rndm(), 0.0
        for fraction, model in zip(self.fractions({}), self.pdfs):
            share = float(np.asarray(fraction))
            if low < draw < low + share:
                return dict(
                    model.generate_event(
                        model.generator_code(frozenset([self.x.GetName()])), rng, bounds
                    )
                )
            low += share
        return self.generate_event(code, rng, bounds)

    def printMetaArgs(self) -> str:
        text = ""
        for i, model in enumerate(self.pdfs):
            text += (" + " if i else "") + (
                f"{self.coefs[i].GetName()} * " if i < len(self.coefs) else ""
            )
            text += model.GetName()
        return text + " "
