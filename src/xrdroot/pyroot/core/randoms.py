"""``TRandom``, ``TRandom1``, ``TRandom2``, ``TRandom3`` and ``gRandom``, by ROOT's names.

Beneath each is the generator of the same name in :mod:`xrdroot.random`,
which is ROOT's to the bit: the same seed gives the same ``Rndm()``,
``Gaus()`` and ``Poisson()`` in the same order as ROOT. ``gRandom`` stands
over :data:`xrdroot.random.gRandom`, so a histogram's ``FillRandom`` and a
script's ``gRandom.Gaus()`` take their draws from one stream, as in ROOT.

Each method takes an extra ``n`` - not ROOT's - for an array of ``n`` draws
at once, the same numbers ``n`` calls would give.
"""

from __future__ import annotations

from typing import Any

from ... import random as xrandom
from .objects import TNamed
from .refs import store, store_many

__all__ = ["TRandom", "TRandom1", "TRandom2", "TRandom3", "gRandom", "current_generator"]


def _number(value: Any) -> Any:
    """A draw as a Python float, or an array of draws left as it is."""
    return float(value) if getattr(value, "ndim", 0) == 0 else value


class TRandom(TNamed):
    """``TRandom``: the distributions, over the generator beneath."""

    ENGINE: Any = xrandom.TRandom
    NAME = ("Random", "Default Random number generator")

    def __init__(self, seed: int = 65539, *, engine: Any = None) -> None:
        super().__init__(*self.NAME)
        #: The xrdroot generator beneath.
        self._xrd = engine if engine is not None else self.ENGINE(seed)

    def SetSeed(self, seed: int = 0) -> None:
        """``SetSeed``: restart the stream; zero asks for one that does not repeat."""
        self._xrd.set_seed(int(seed))

    def GetSeed(self) -> int:
        return int(self._xrd.get_seed())

    def Rndm(self, *n: Any) -> Any:
        """``Rndm``: uniform in (0, 1], never zero."""
        return _number(self._xrd.rndm(*n))

    def RndmArray(self, count: int, array: Any) -> None:
        """``RndmArray(n, array)``: ``n`` draws into ``array``."""
        store_many(array, self._xrd.rndm(int(count)))

    def Uniform(self, x1: float = 1.0, x2: Any = None, n: Any = None) -> Any:
        """``Uniform(x1)`` in (0, x1), ``Uniform(x1, x2)`` in (x1, x2)."""
        return _number(self._xrd.uniform(x1, x2, n))

    def Gaus(self, mean: float = 0.0, sigma: float = 1.0, n: Any = None) -> Any:
        return _number(self._xrd.gaus(mean, sigma, n))

    def Exp(self, tau: float, n: Any = None) -> Any:
        return _number(self._xrd.exp(tau, n))

    def Integer(self, imax: int, n: Any = None) -> Any:
        found = self._xrd.integer(imax, n)
        return int(found) if n is None else found

    def BreitWigner(self, mean: float = 0.0, gamma: float = 1.0, n: Any = None) -> Any:
        return _number(self._xrd.breit_wigner(mean, gamma, n))

    def Landau(self, mean: float = 0.0, sigma: float = 1.0, n: Any = None) -> Any:
        return _number(self._xrd.landau(mean, sigma, n))

    def Binomial(self, ntot: int, prob: float, n: Any = None) -> Any:
        found = self._xrd.binomial(int(ntot), float(prob), n)
        return int(found) if n is None else found

    def Poisson(self, mean: float, n: Any = None) -> Any:
        found = self._xrd.poisson(float(mean), n)
        return int(found) if n is None else found

    def PoissonD(self, mean: float, n: Any = None) -> Any:
        return _number(self._xrd.poisson_d(float(mean), n))

    def Rannor(self, a: Any = None, b: Any = None) -> tuple[float, float]:
        """``Rannor(a, b)``: two independent Gaussians, into ``a`` and ``b`` - and handed back."""
        x, y = (float(value) for value in self._xrd.rannor())
        store(a, x)
        store(b, y)
        return x, y

    def Circle(self, x: Any, y: Any, r: float) -> tuple[float, float]:
        """``Circle(x, y, r)``: a point on a circle of radius ``r``, into ``x`` and ``y``."""
        px, py = (float(value) for value in self._xrd.circle(r))
        store(x, px)
        store(y, py)
        return px, py

    def Sphere(self, x: Any, y: Any, z: Any, r: float) -> tuple[float, float, float]:
        """``Sphere(x, y, z, r)``: a point on a sphere of radius ``r``."""
        px, py, pz = (float(value) for value in self._xrd.sphere(r))
        for target, value in zip((x, y, z), (px, py, pz)):
            store(target, value)
        return px, py, pz

    def Print(self, option: str = "") -> None:
        print(f"Random number generator: {self.ClassName()} with seed {self.GetSeed()}")


class TRandom1(TRandom):
    """``TRandom1``: RANLUX, Lüscher's generator at ROOT's default luxury."""

    ENGINE = xrandom.TRandom1
    NAME = ("Random1", "Random number generator: RanLux")

    def __init__(self, seed: int = 4357, lux: int = 3, *, engine: Any = None) -> None:
        super().__init__(seed, engine=engine)


class TRandom2(TRandom):
    """``TRandom2``: L'Ecuyer's maximally equidistributed Tausworthe generator."""

    ENGINE = xrandom.TRandom2
    NAME = ("Random2", "Random number generator: Tausworthe")

    def __init__(self, seed: int = 1, *, engine: Any = None) -> None:
        super().__init__(seed, engine=engine)


class TRandom3(TRandom):
    """``TRandom3``: the Mersenne twister, ROOT's ``gRandom``."""

    ENGINE = xrandom.TRandom3
    NAME = ("Random3", "Random number generator: Mersenne Twister")

    def __init__(self, seed: int = 4357, *, engine: Any = None) -> None:
        super().__init__(seed, engine=engine)


#: ``gRandom``: over :data:`xrdroot.random.gRandom`, the stream the whole library shares.
gRandom = TRandom3(engine=xrandom.gRandom)


def current_generator() -> Any:
    """The xrdroot generator ``gRandom`` stands for now, even if a script replaced it."""
    import sys

    namespace = sys.modules.get("xrdroot.pyroot")
    found = getattr(namespace, "__dict__", {}).get("gRandom", gRandom)
    return getattr(found, "_xrd", xrandom.gRandom)
