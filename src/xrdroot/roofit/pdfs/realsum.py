"""``RooRealSumPdf``: a density that is a sum of functions times coefficients, normalised as a
whole.

Unlike a :class:`~.addpdf.RooAddPdf` of densities, the terms are functions -
amplitudes, histograms - that need not be positive nor normalised; the
sum is normalised by its integral, each term's integrated its own way.
"""

from __future__ import annotations

from typing import Any

from ..collections import as_list
from ..pdf import CAN_BE_EXTENDED, CAN_NOT_BE_EXTENDED, RooAbsPdf
from ..real import Context

__all__ = ["RooRealSumPdf"]


class RooRealSumPdf(RooAbsPdf):
    """``c1 f1 + c2 f2 + ...``, or with one coefficient fewer, the last ``1 - sum``."""

    def __init__(
        self, name: Any, title: Any, funcs: Any, coefs: Any, extended: bool = False
    ) -> None:
        super().__init__(name, title)
        self.funcs = self._list_proxy("!funcList", as_list(funcs))
        self.coefs = self._list_proxy("!coefList", as_list(coefs))
        self._extended = bool(extended)

    def _coefficients(self, ctx: Context) -> list[Any]:
        values = [c.compute(ctx) for c in self.coefs]
        if len(values) < len(self.funcs):
            values.append(1.0 - sum(values))
        return values

    def compute(self, ctx: Context) -> Any:
        total: Any = 0.0
        for coef, func in zip(self._coefficients(ctx), self.funcs, strict=False):
            total = total + coef * func.compute(ctx)
        return total

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return names

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        total: Any = 0.0
        for coef, func in zip(self._coefficients(ctx), self.funcs, strict=False):
            part = names & func.dependents()
            total = total + coef * (func.integrate(part, ctx, rng) if part else func.compute(ctx))
        return total

    def extendMode(self) -> int:
        return CAN_BE_EXTENDED if self._extended else CAN_NOT_BE_EXTENDED

    def expected(self, nset: Any, rng: Any = None, fit: bool = False) -> float:
        """``getNorm(nset)``: the integral a normalised value divides by - made, and said, the
        first time, as the normalisation cache makes it."""
        names = frozenset(nset or ())
        if names and not fit and not rng:
            self._announce_norm(names & self.dependents())
        return float(self.integrate(names, {}, rng))

    def bin_boundaries(self, name: str) -> Any:
        """``binBoundaries``: every term's boundaries, merged."""
        found: set[float] = set()
        for func in self.funcs:
            edges = getattr(func, "bin_boundaries", lambda _n: None)(name)
            found |= set(edges or ())
        return sorted(found) or None

    def isBinnedDistribution(self, obs: Any = None) -> bool:
        from .histfactory import _binned

        return _binned(self.funcs, obs)

    def printMetaArgs(self) -> str:
        parts = [
            f"{c.GetName()} * {f.GetName()}" for c, f in zip(self.coefs, self.funcs, strict=False)
        ]
        if len(self.funcs) > len(self.coefs):
            parts.append(f"[%] * {self.funcs[len(self.coefs)].GetName()}")
        return " + ".join(parts) + " "
