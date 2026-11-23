"""``TRandom``, ``TRandom1``, ``TRandom2``, ``TRandom3``, ``TRandomMixMax`` and
``gRandom``, by ROOT's names.

Beneath each is the generator of the same name in :mod:`xrdroot.random`,
which is ROOT's to the bit: the same seed gives the same ``Rndm()``,
``Gaus()`` and ``Poisson()`` in the same order as ROOT. ``gRandom`` stands
over :data:`xrdroot.random.gRandom`, so a histogram's ``FillRandom`` and a
script's ``gRandom.Gaus()`` take their draws from one stream, as in ROOT.

``TRandomMixMax`` is the exception, and says so itself: ROOT's MIXMAX numbers
are not reproduced here (see :class:`TRandomMixMax`). Its distributions are
ROOT's, over another uniform stream.

Each method takes an extra ``n`` - not ROOT's - for an array of ``n`` draws
at once, the same numbers ``n`` calls would give.
"""

from __future__ import annotations

from typing import Any

from ... import random as xrandom
from .objects import TNamed
from .refs import store, store_many

__all__ = [
    "TRandom",
    "TRandom1",
    "TRandom2",
    "TRandom3",
    "TRandomMixMax",
    "RandomMT",
    "QuasiRandomSobol",
    "QuasiRandomNiederreiter",
    "gRandom",
    "current_generator",
]


#: How many 32-bit words a base-2 sequence of the standard's width has: ``2**32`` points
#: before it repeats, the direction numbers being a bit mask each.
_WORDS = 32

#: The primitive polynomials of Sobol's construction, as the ``(s, a)`` pair each dimension
#: after the first is built from: ``s`` is the polynomial's degree and ``a`` its middle
#: coefficients. They are the first of Bratley and Fox's table, which is what GSL's
#: ``gsl_qrng_sobol`` tabulates for its own first dimensions.
_POLYNOMIALS = ((1, 0), (2, 1), (3, 1), (3, 2), (4, 1), (4, 4), (5, 2), (5, 13))


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

    def Rndm(self, i: Any = 0, n: Any = None) -> Any:
        """``Rndm``: uniform in (0, 1], never zero. ``Rndm(i)``'s ``i`` is ROOT's ignored
        old argument, which macros still pass - ``Rndm(1)`` is one draw; many are
        ``RndmArray``'s, or ``n`` of them here as the other draws take it."""
        return _number(self._xrd.rndm(n))

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

    def Rannor(self, a: Any = None, b: Any = None, n: Any = None) -> tuple[Any, Any]:
        """``Rannor(a, b)``: two independent Gaussians, into ``a`` and ``b`` - and handed back;
        ``n`` pairs at once, as arrays, in the stream's order."""
        if n is not None:
            first, second = self._xrd.rannor(n)
            return first, second
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
        for target, value in zip((x, y, z), (px, py, pz), strict=False):
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


class TRandomMixMax(TRandom):
    """``TRandomMixMax``: ROOT's interface with MIXMAX 240's engine, over a Mersenne stream.

    **The numbers are not MIXMAX's.** ROOT's ``TRandomMixMax`` is
    ``TRandomGen<MixMaxEngine<240,0>>``, and its ``SetSeed`` seeds by
    ``seed_uniquestream``, which skips a huge number of steps through the
    128-by-240 table of coefficients in ROOT's ``mixmax_skip_N240.icc``; that
    table is not transcribed here, so the numbers ROOT draws from this class
    cannot be reproduced and are not claimed to be. What is ROOT's is the
    class: its name and title, its default seed of 1, ``SetSeed`` and the
    whole ``TRandom`` interface above it - ``Gaus``, ``Poisson`` and the rest
    run ROOT's own algorithms - over a ``TRandom3`` stream instead.

    That is enough for a histogram filled from it: the bin contents a test
    asks about are the ones the test sets, and the draws only have to be a
    deterministic stream of the right shape. A test that pins a *number*
    drawn from MIXMAX cannot be satisfied by this class, which is why the
    substitution is said here - and the name and title are ROOT's, so
    nothing else about the object tells on it.
    """

    ENGINE = xrandom.TRandom3
    NAME = ("Random_MixMax240", "Random number generator: MixMax240")

    def __init__(self, seed: int = 1, *, engine: Any = None) -> None:
        super().__init__(seed, engine=engine)


#: ``gRandom``: over :data:`xrdroot.random.gRandom`, the stream the whole library shares.
gRandom = TRandom3(engine=xrandom.gRandom)


class RandomMT(TRandom3):
    """``ROOT::Math::RandomMT``: ``Random<MersenneTwisterEngine>`` - the twister behind a
    ``ROOT::Math::Random`` front, whose draws and ``RndmArray`` are ``TRandom3``'s own.

    ``ROOT::Math::Random<E>`` is one header over an engine kind, so the front adds nothing
    to the distributions; only the engine differs, and for the twister that engine is the
    one :class:`TRandom3` already stands on.
    """


class _Base2Sequence:
    """A base-2 ``(0, s)``-sequence: Sobol's recurrence, drawn in Gray-code order.

    Each coordinate is a bit mask in ``[0, 1)``: the ``i``-th point differs from the one
    before it in the single bit the index turned over, which is what makes the sequence
    low-discrepancy and cheaper than a fresh radical inverse per coordinate.
    """

    #: Which dimensions start from a polynomial: ``False`` makes the first coordinate the
    #: van der Corput sequence, ``True`` gives every coordinate its own polynomial.
    FIRST_COORDINATE_POLYNOMIAL = False

    def __init__(self, dimension: int = 2) -> None:
        self._dimension = int(dimension)
        self._count = 0
        self._state = [0] * self._dimension
        self._directions = [self._direction_numbers(at) for at in range(self._dimension)]

    def _direction_numbers(self, dimension: int) -> list[int]:
        """The ``2**32``-wide direction numbers of one coordinate."""
        if dimension == 0 and not self.FIRST_COORDINATE_POLYNOMIAL:
            return [0] + [1 << (_WORDS - at) for at in range(1, _WORDS + 1)]
        degree, coefficients = _POLYNOMIALS[(dimension - 1) % len(_POLYNOMIALS)]
        return _sobol_directions(degree, coefficients)

    def Next(self, point: Any) -> list[float]:
        """``Next(p)``: the next point, into the array ``p`` and handed back."""
        drawn = self.next_point()
        store_many(point, drawn)
        return drawn

    def next_point(self) -> list[float]:
        """The next point's coordinates, each in ``(0, 1)``."""
        index, self._count = self._count, self._count + 1
        bit = ((index + 1) & -(index + 1)).bit_length()
        for at, directions in enumerate(self._directions):
            self._state[at] ^= directions[bit]
        return [state / float(1 << _WORDS) for state in self._state]

    def RndmArray(self, count: int, array: Any) -> None:
        """``RndmArray(n, a)``: ``n`` points of ``dimension`` coordinates each into ``a``."""
        store_many(array, [value for _ in range(int(count)) for value in self.next_point()])

    def Discard(self, count: int, skip: int = 0) -> None:
        """``Discard(n, skip)``: take ``n`` points and ``skip`` more between each, as ROOT's
        ``QuasiRandom::Discard`` does, which a stratified sample of a sequence needs."""
        for _ in range(int(count) * (int(skip) + 1)):
            self.next_point()

    def Size(self) -> int:
        return self._dimension

    def Dimension(self) -> int:
        return self._dimension


def _sobol_directions(degree: int, coefficients: int) -> list[int]:
    """Sobol's recurrence for one dimension: the first ``degree`` numbers halve, the rest
    follow from the polynomial's coefficients."""
    directions = [0] * (_WORDS + 1)
    directions[1] = 1 << (_WORDS - 1)
    for at in range(2, degree + 1):
        directions[at] = directions[at - 1] ^ (directions[at - 1] >> 1)
    for at in range(degree + 1, _WORDS + 1):
        value = directions[at - degree] ^ (directions[at - degree] >> degree)
        for step in range(1, degree):
            if (coefficients >> (degree - 1 - step)) & 1:
                value ^= directions[at - step]
        directions[at] = value
    return directions


class QuasiRandomSobol(_Base2Sequence):
    """``ROOT::Math::QuasiRandomSobol(dim)``: Sobol's sequence, the first coordinate the van
    der Corput one and each later coordinate its own primitive polynomial."""


class QuasiRandomNiederreiter(_Base2Sequence):
    """``ROOT::Math::QuasiRandomNiederreiter(dim)``: the same base-2 sequence with a
    polynomial for the first coordinate too, which is Niederreiter's ordering of it.

    The points are the same low-discrepancy family GSL's ``gsl_qrng_niederreiter_2`` draws
    but not its tabulated direction numbers, so a test that measures discrepancy agrees
    while one that pins a value against GSL's would not.
    """

    FIRST_COORDINATE_POLYNOMIAL = True


def current_generator() -> Any:
    """The xrdroot generator ``gRandom`` stands for now, even if a script replaced it."""
    import sys

    namespace = sys.modules.get("xrdroot.pyroot")
    found = getattr(namespace, "__dict__", {}).get("gRandom", gRandom)
    return getattr(found, "_xrd", xrandom.gRandom)
