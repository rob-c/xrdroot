"""The everyday densities: Gaussian, exponential, polynomial, Chebychev and uniform.

Each is its formula (:meth:`compute`), the integrals it has in closed form,
over which variables and with ROOT's formula (:meth:`analytic`), and - for
the Gaussian and the uniform, which RooFit samples directly - how it draws
an event. Numbers may stand for any argument, as ``RooAbsReal::Ref`` lets
them: a number becomes a :class:`~xrdroot.roofit.variables.RooConstVar`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import mathfuncs as mf
from ..pdf import RooAbsPdf, check_range
from ..real import Context
from ..variables import RooConstVar

__all__ = ["RooChebychev", "RooExponential", "RooGaussian", "RooPolynomial", "RooUniform", "ref"]


def ref(arg: Any) -> Any:
    """``RooAbsReal::Ref``: a RooFit object as it is, a number as a constant named after it."""
    if isinstance(arg, (int, float, np.floating, np.integer)):
        from ..printing import g

        return RooConstVar(g(float(arg)), g(float(arg)), float(arg))
    return arg


class RooGaussian(RooAbsPdf):
    """``RooGaussian``: ``exp(-(x-mean)^2 / (2 sigma^2))``, integrable over ``x`` or ``mean``."""

    def __init__(self, name: Any, title: Any, x: Any, mean: Any, sigma: Any) -> None:
        super().__init__(name, title)
        self.x = self._proxy("x", ref(x))
        self.mean = self._proxy("mean", ref(mean))
        self.sigma = self._proxy("sigma", ref(sigma))
        check_range(self, [self.sigma], 0.0)

    def compute(self, ctx: Context) -> Any:
        arg = self.x.compute(ctx) - self.mean.compute(ctx)
        sig = self.sigma.compute(ctx)
        return np.exp(-0.5 * arg * arg / (sig * sig))

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        for one in (self.x, self.mean):
            if one.isFundamental() and one.GetName() in names:
                return frozenset([one.GetName()])
        return frozenset()

    def integral_code(self, names: frozenset[str]) -> int:
        return 1 if self.x.GetName() in names else 2

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        over, other = (self.x, self.mean) if self.x.GetName() in names else (self.mean, self.x)
        low, high = over.getMin(rng), over.getMax(rng)
        return mf.gaussian_integral(low, high, other.compute(ctx), self.sigma.compute(ctx))

    def generator_code(self, names: frozenset[str]) -> int:
        """``getGenerator``: 1 to draw ``x``, 2 to draw ``mean``, 0 for neither."""
        for code, one in ((1, self.x), (2, self.mean)):
            if one.isFundamental() and names == frozenset([one.GetName()]):
                return code
        return 0

    def generate_event(self, code: int, rng: Any) -> dict[str, float]:
        """``generateEvent``: a Gaussian draw, drawn again until it is inside the range."""
        target, centre = (self.x, self.mean) if code == 1 else (self.mean, self.x)
        low, high = target.getMin(), target.getMax()
        while True:
            value = float(rng.gaus(centre.getVal(), self.sigma.getVal()))
            if low < value < high:
                return {target.GetName(): value}


class RooExponential(RooAbsPdf):
    """``RooExponential``: ``exp(c x)`` - or ``exp(-c x)`` if asked to negate ``c``."""

    def __init__(self, name: Any, title: Any, x: Any, c: Any, negateCoefficient: bool = False) -> None:
        super().__init__(name, title)
        self.x = self._proxy("x", ref(x))
        self.c = self._proxy("c", ref(c))
        self._negate = bool(negateCoefficient)

    def _coef(self, ctx: Context) -> Any:
        found = self.c.compute(ctx)
        return -found if self._negate else found

    def compute(self, ctx: Context) -> Any:
        return np.exp(self._coef(ctx) * self.x.compute(ctx))

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        for one in (self.x, self.c):
            if one.isFundamental() and one.GetName() in names:
                return frozenset([one.GetName()])
        return frozenset()

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        if self.x.GetName() in names:
            low, high = self.x.getMin(rng), self.x.getMax(rng)
            return mf.exponential_integral(low, high, self._coef(ctx))
        low, high = self.c.getMin(rng), self.c.getMax(rng)
        if self._negate:
            low, high = -high, -low
        return mf.exponential_integral(low, high, self.x.compute(ctx))


class RooPolynomial(RooAbsPdf):
    """``RooPolynomial``: ``1 + a1 x + a2 x^2 ...`` - the constant term implied from ``lowestOrder``."""

    def __init__(self, name: Any, title: Any, x: Any, coefList: Any = (), lowestOrder: int = 1) -> None:
        super().__init__(name, title)
        self.x = self._proxy("x", ref(x))
        self.coefs = self._list_proxy("coefList", [ref(c) for c in _items(coefList)])
        self._lowest = max(int(lowestOrder), 0)

    def compute(self, ctx: Context) -> Any:
        coefs = [c.compute(ctx) for c in self.coefs]
        if not coefs:
            return 1.0 if self._lowest else 0.0
        return mf.polynomial(coefs, self._lowest, self.x.compute(ctx), True)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return frozenset([self.x.GetName()]) & names

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        coefs = [c.compute(ctx) for c in self.coefs]
        return mf.polynomial_integral(coefs, self._lowest, self.x.getMin(rng), self.x.getMax(rng), True)


class RooChebychev(RooAbsPdf):
    """``RooChebychev``: ``1 + a0 T1 + a1 T2 ...`` of ``x`` mapped from its range onto [-1, 1]."""

    def __init__(self, name: Any, title: Any, x: Any, coefList: Any = ()) -> None:
        super().__init__(name, title)
        self.x = self._proxy("x", ref(x))
        self.coefs = self._list_proxy("coefList", [ref(c) for c in _items(coefList)])
        self._reference: str | None = None

    def selectNormalizationRange(self, rng: Any = None, force: bool = False) -> None:
        if rng and (force or not self._reference):
            self._reference = str(rng)
        if not rng:
            self._reference = None

    def compute(self, ctx: Context) -> Any:
        coefs = [c.compute(ctx) for c in self.coefs]
        low, high = self.x.getMin(self._reference), self.x.getMax(self._reference)
        return mf.chebychev(coefs, self.x.compute(ctx), low, high)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return frozenset([self.x.GetName()]) & names

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        coefs = [c.compute(ctx) for c in self.coefs]
        low, high = self.x.getMin(self._reference), self.x.getMax(self._reference)
        return mf.chebychev_integral(coefs, low, high, self.x.getMin(rng), self.x.getMax(rng))


class RooUniform(RooAbsPdf):
    """``RooUniform``: one, over any number of observables."""

    def __init__(self, name: Any, title: Any, x: Any) -> None:
        super().__init__(name, title)
        self.xs = self._list_proxy("x", _items(x))

    def compute(self, ctx: Context) -> Any:
        return 1.0

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return frozenset(one.GetName() for one in self.xs) & names

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        total = 1.0
        for one in self.xs:
            if one.GetName() in names:
                total *= one.getMax(rng) - one.getMin(rng)
        return total

    def generator_code(self, names: frozenset[str]) -> int:
        return 1 if names and names <= frozenset(one.GetName() for one in self.xs) else 0

    def generate_event(self, code: int, rng: Any, names: frozenset[str] = frozenset()) -> dict[str, float]:
        found = {}
        for one in self.xs:
            if not names or one.GetName() in names:
                found[one.GetName()] = one.getMin() + float(rng.rndm()) * (one.getMax() - one.getMin())
        return found


def _items(items: Any) -> list[Any]:
    from ..collections import as_list

    return as_list(items)
