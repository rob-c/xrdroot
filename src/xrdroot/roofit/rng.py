"""``RooRandom``: RooFit's own random generator, apart from ``gRandom``.

Everything RooFit draws - events, the number of events of an extended
sample, which component of a sum an event comes from - comes from one
``TRandom3`` made with the default seed the first time it is asked for, as
``RooRandom::randomGenerator()`` makes it, and not from ``gRandom``. So a
macro that seeds ``gRandom`` does not change what RooFit generates, and one
that seeds ``RooRandom::randomGenerator()`` does, exactly as in ROOT.

The generator is :class:`xrdroot.random.TRandom3`, which draws ROOT's
numbers to the bit; :class:`Generator` gives it ROOT's method names.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..random import TRandom3

__all__ = ["Generator", "RooRandom", "generator"]


class Generator:
    """A ``TRandom3`` by ROOT's method names: ``Rndm``, ``Gaus``, ``Poisson``, ``SetSeed``..."""

    def __init__(self, engine: Any = None) -> None:
        self.engine = engine if engine is not None else TRandom3()

    def Rndm(self) -> float:
        return float(self.engine.rndm())

    def RndmArray(self, n: int, out: Any = None) -> Any:
        values = np.asarray(self.engine.rndm(int(n)), dtype=np.float64)
        if out is not None:
            out[: int(n)] = values
        return values

    def Uniform(self, a: float = 1.0, b: Any = None) -> float:
        return float(self.engine.uniform(a) if b is None else self.engine.uniform(a, b))

    def Gaus(self, mean: float = 0.0, sigma: float = 1.0) -> float:
        return float(self.engine.gaus(mean, sigma))

    def Poisson(self, mean: float) -> int:
        return int(self.engine.poisson(mean))

    def Integer(self, imax: int) -> int:
        return int(self.engine.integer(imax))

    def Exp(self, tau: float) -> float:
        return float(self.engine.exp(tau))

    def Landau(self, mean: float = 0.0, sigma: float = 1.0) -> float:
        return float(self.engine.landau(mean, sigma))

    def BreitWigner(self, mean: float = 0.0, gamma: float = 1.0) -> float:
        return float(self.engine.breit_wigner(mean, gamma))

    def Binomial(self, ntot: int, prob: float) -> int:
        return int(self.engine.binomial(ntot, prob))

    def SetSeed(self, seed: int = 4357) -> None:
        self.engine.set_seed(int(seed))

    def GetSeed(self) -> int:
        return int(self.engine.get_seed())

    # the lower-case names the engine has, for code that takes either
    def rndm(self, n: Any = None) -> Any:
        return self.engine.rndm(n)

    def gaus(self, mean: float = 0.0, sigma: float = 1.0) -> float:
        return float(self.engine.gaus(mean, sigma))


_GENERATOR: list[Generator] = []


def generator() -> Generator:
    """``RooRandom::randomGenerator()``: made at the default seed the first time it is asked for."""
    if not _GENERATOR:
        _GENERATOR.append(Generator())
    return _GENERATOR[0]


class RooRandom:
    """``RooRandom``'s static functions."""

    @staticmethod
    def randomGenerator() -> Generator:
        return generator()

    @staticmethod
    def setRandomGenerator(engine: Any) -> None:
        made = engine if isinstance(engine, Generator) else Generator(getattr(engine, "_xrd", engine))
        _GENERATOR[:] = [made]

    @staticmethod
    def uniform(engine: Any = None) -> float:
        return (engine or generator()).Rndm()

    @staticmethod
    def integer(n: int, engine: Any = None) -> int:
        return (engine or generator()).Integer(n)

    @staticmethod
    def gaussian(engine: Any = None) -> float:
        return (engine or generator()).Gaus()
