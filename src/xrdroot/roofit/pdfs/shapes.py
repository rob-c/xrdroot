"""The rest of RooFit's standard shapes: ARGUS, Crystal Ball, Breit-Wigner, Landau and friends.

Each value is computed as RooFit's batch kernel computes it
(``RooBatchCompute``'s ``compute*`` functions), because that is what a
fit evaluates in ROOT, and each integral in closed form is RooFit's own.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...function.analytic import landau_cdf
from ...function.special import landau_pdf
from ...random import libm
from .. import mathfuncs as mf
from ..pdf import RooAbsPdf, check_range
from ..real import Context
from .basic import ref

__all__ = [
    "RooArgusBG",
    "RooBifurGauss",
    "RooBreitWigner",
    "RooCBShape",
    "RooLandau",
    "RooLognormal",
    "RooPoisson",
]


class _Shape(RooAbsPdf):
    """A density of named inputs, declared in the order ROOT's class declares its proxies."""

    #: The inputs' names, in order.
    inputs: tuple[str, ...] = ()

    def __init__(self, name: Any, title: Any, *args: Any) -> None:
        super().__init__(name, title)
        for key, arg in zip(self.inputs, args):
            setattr(self, key, self._proxy(key, ref(arg)))

    def __getattr__(self, key: str) -> Any:
        """Only for what is not there - the inputs are set by name - so, always a refusal."""
        raise AttributeError(f"{type(self).__name__} has no input or member called {key!r}.")

    def v(self, key: str, ctx: Context) -> Any:
        return getattr(self, key).compute(ctx)

    def over(self, key: str, names: frozenset[str]) -> frozenset[str]:
        arg = getattr(self, key)
        return (
            frozenset([arg.GetName()])
            if arg.isFundamental() and arg.GetName() in names
            else frozenset()
        )


class RooArgusBG(_Shape):
    """``m (1 - (m/m0)^2)^p exp(c (1 - (m/m0)^2))`` below ``m0``: the ARGUS background."""

    inputs = ("m", "m0", "c", "p")

    def __init__(self, name: Any, title: Any, m: Any, m0: Any, c: Any, p: Any = 0.5) -> None:
        super().__init__(name, title, m, m0, c, p)

    def compute(self, ctx: Context) -> Any:
        m, m0, c, p = (self.v(k, ctx) for k in self.inputs)
        t = m / m0
        u = 1 - t * t
        with np.errstate(all="ignore"):
            found = m * libm.exp(c * u + p * libm.log(u))
        return np.where(m >= m0, 0.0, found)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return (
            self.over("m", names) if self.p.isConstant() and self.p.getVal() == 0.5 else frozenset()
        )

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        m0, c = self.v("m0", ctx), self.v("c", ctx)
        low, high = min(self.m.getMin(rng), m0), min(self.m.getMax(rng), m0)
        f1, f2 = 1.0 - (low / m0) ** 2, 1.0 - (high / m0) ** 2
        return _argus_part(m0, c, f2) - _argus_part(m0, c, f1)


def _argus_negative(c: float, f: float) -> Any:
    """The primitive's bracket for a falling exponential, ``c < 0``: with the error function."""
    root = math.sqrt(-c * f)
    return math.exp(c * f) * math.sqrt(f) / c + 0.5 / (-c) ** 1.5 * math.sqrt(math.pi) * math.erf(
        root
    )


def _argus_part(m0: float, c: float, f: float) -> Any:
    if c < 0:
        return -0.5 * m0 * m0 * _argus_negative(c, f)
    if c == 0:
        return -m0 * m0 / 3.0 * f * math.sqrt(f)
    dawson = _dawson(math.sqrt(c * f))
    return 0.5 * m0 * m0 * math.exp(c * f) / (c * math.sqrt(c)) * (dawson - math.sqrt(c * f))


def _dawson(x: float) -> float:
    """Dawson's function, ``exp(-x^2) * integral_0^x exp(t^2) dt``: ``sqrt(pi)/2 Im w(x)``."""
    ts = np.linspace(0.0, x, 2001)
    return float(np.trapezoid(libm.exp(ts * ts - x * x), ts)) if x else 0.0


class RooCBShape(_Shape):
    """``RooCBShape``: the Crystal Ball, a Gaussian with a power-law tail."""

    inputs = ("m", "m0", "sigma", "alpha", "n")

    def __init__(self, name: Any, title: Any, *args: Any) -> None:
        super().__init__(name, title, *args)
        check_range(self, [self.sigma], 0.0)
        check_range(self, [self.n], 0.0)

    def compute(self, ctx: Context) -> Any:
        m, m0, s, a, n = (self.v(k, ctx) for k in self.inputs)
        t = (m - m0) / s
        core = (a > 0) & (t >= -a) | (a < 0) & (-t >= a)
        with np.errstate(all="ignore"):
            tail = n * libm.log(n / (n - a * a - a * t)) - 0.5 * a * a
        return libm.exp(np.where(core, -0.5 * t * t, tail))

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return self.over("m", names)

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        values = [float(np.asarray(self.v(k, ctx))) for k in self.inputs[1:]]
        return mf.cb_shape_integral(self.m.getMin(rng), self.m.getMax(rng), *values)


class RooBifurGauss(_Shape):
    """A Gaussian of one width left of the mean and another right of it."""

    inputs = ("x", "mean", "sigmaL", "sigmaR")

    def compute(self, ctx: Context) -> Any:
        x, mean, left, right = (self.v(k, ctx) for k in self.inputs)
        arg = x - mean
        arg = np.where(arg < 0, arg / left, arg / right)
        return libm.exp(-0.5 * arg * arg)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return self.over("x", names)

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        mean, left, right = (float(np.asarray(self.v(k, ctx))) for k in self.inputs[1:])
        return mf.bifurgauss_integral(self.x.getMin(rng), self.x.getMax(rng), mean, left, right)


class RooBreitWigner(_Shape):
    """The non-relativistic Breit-Wigner: ``1 / ((x-mean)^2 + width^2/4)``."""

    inputs = ("x", "mean", "width")

    def compute(self, ctx: Context) -> Any:
        x, mean, width = (self.v(k, ctx) for k in self.inputs)
        arg = x - mean
        return 1 / (arg * arg + 0.25 * width * width)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return self.over("x", names)

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        mean, width = self.v("mean", ctx), self.v("width", ctx)
        c = 2.0 / width
        return c * (
            libm.atan(c * (self.x.getMax(rng) - mean)) - libm.atan(c * (self.x.getMin(rng) - mean))
        )


class RooLandau(_Shape):
    """The Landau distribution, ``landau_pdf((x-mean)/sigma)``."""

    inputs = ("x", "mean", "sigma")

    def __init__(self, name: Any, title: Any, *args: Any) -> None:
        super().__init__(name, title, *args)
        check_range(self, [self.sigma], 0.0)

    def compute(self, ctx: Context) -> Any:
        x, mean, sigma = (self.v(k, ctx) for k in self.inputs)
        return np.where(np.asarray(sigma) <= 0, 0.0, landau_pdf((x - mean) / sigma))

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return self.over("x", names)

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        mean, sigma = (
            float(np.asarray(self.v("mean", ctx))),
            float(np.asarray(self.v("sigma", ctx))),
        )
        high = landau_cdf(self.x.getMax(rng), sigma, mean)
        return sigma * (high - landau_cdf(self.x.getMin(rng), sigma, mean))

    def generator_code(self, names: frozenset[str]) -> int:
        return 1 if names == self.over("x", names) and names else 0

    def generate_event(self, code: int, rng: Any) -> dict[str, float]:
        low, high = self.x.getMin(), self.x.getMax()
        while True:
            value = rng.Landau(self.mean.getVal(), self.sigma.getVal())
            if low < value < high:
                return {self.x.GetName(): value}


class RooLognormal(_Shape):
    """The log-normal distribution of median ``m0`` and shape ``k``."""

    inputs = ("x", "m0", "k")

    def compute(self, ctx: Context) -> Any:
        x, m0, k = (self.v(key, ctx) for key in self.inputs)
        lnk = np.abs(libm.log(k))
        arg = libm.log(x / m0) / lnk
        return libm.exp(-0.5 * arg * arg) / (x * lnk * 2.506628274631000502415765284811)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return self.over("x", names)

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        m0, k = self.v("m0", ctx), self.v("k", ctx)
        lnk = abs(math.log(k)) * math.sqrt(2.0)
        high, low = self.x.getMax(rng), self.x.getMin(rng)
        return 0.5 * (math.erf(math.log(high / m0) / lnk) - math.erf(math.log(low / m0) / lnk))


class RooPoisson(_Shape):
    """The Poisson probability of ``x`` events - rounded down unless asked not to - for ``mean``."""

    inputs = ("x", "mean")

    def __init__(self, name: Any, title: Any, x: Any, mean: Any, noRounding: bool = False) -> None:
        super().__init__(name, title, x, mean)
        self._no_rounding = bool(noRounding)
        self._protect = False
        check_range(self, [self.x, self.mean], 0.0, math.inf, closed=True)

    def protectNegativeMean(self, flag: bool = True) -> None:
        self._protect = bool(flag)

    def compute(self, ctx: Context) -> Any:
        x, mean = self.v("x", ctx), self.v("mean", ctx)
        k = x if self._no_rounding else np.floor(x)
        with np.errstate(all="ignore"):
            found = libm.exp(k * libm.log(mean) - mean - mf.lgamma(np.asarray(k) + 1.0))
        found = np.where(k < 0, 0.0, np.where(k == 0, 1 / libm.exp(mean), found))
        return np.where(self._protect & (np.asarray(mean) < 0), 1e-3, found)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return self.over("x", names) or self.over("mean", names)

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        from ...stats import incomplete_gamma, incomplete_gamma_c

        mean = float(np.asarray(self.v("mean", ctx)))
        x = float(np.asarray(self.v("x", ctx)))
        if self.x.GetName() in names:
            low, high = max(0.0, self.x.getMin(rng)), self.x.getMax(rng)
            if high < 0 or high < low:
                return 0.0
            delta = 100.0 * math.sqrt(mean)
            if low < max(mean - delta, 0.0) and high > mean + delta:
                return 1.0
            first, last = int(low), int(min(high + 1, 4294967295.0))
            if first == 0:
                return incomplete_gamma_c(last, mean)
            if first <= mean:
                return incomplete_gamma_c(last, mean) - incomplete_gamma_c(first, mean)
            return incomplete_gamma(first, mean) - incomplete_gamma(last, mean)
        ix = 1 + (x if self._no_rounding else math.floor(x))
        return incomplete_gamma(ix, self.mean.getMax(rng)) - incomplete_gamma(
            ix, self.mean.getMin(rng)
        )

    def generator_code(self, names: frozenset[str]) -> int:
        return 1 if names and names == self.over("x", names) else 0

    def generate_event(self, code: int, rng: Any) -> dict[str, float]:
        low, high = self.x.getMin(), self.x.getMax()
        while True:
            value = float(rng.Poisson(self.mean.getVal()))
            if low <= value <= high:
                return {self.x.GetName(): value}
