"""``TRandom``: ROOT's random numbers, the distributions made of them, and the draws in hand.

Every ROOT generator is a stream of ``Rndm()`` values, and every distribution
``TRandom`` offers is a recipe for turning some of them into a number: one
for ``Exp``, two for ``Rannor``, as many as it takes for ``Gaus``. A class
here is a generator - its ``_engine_*`` methods, which make the next so many
``Rndm()`` values to the bit and say what state that leaves it in - and this
base class is everything built on top, written once.

Draws are made ahead, in runs, and held until a distribution takes them. That
is what lets a rejection algorithm look at a thousand candidates at once and
take exactly the ones ROOT's loop would have taken, leaving the rest for the
next call: nothing is thrown away, so whatever is asked for, in whatever
order, the numbers are ROOT's successive ones. Each run remembers the state
the generator was in before it was made, so the generator's state as ROOT
would report it - after exactly the draws handed out, not the ones made ahead
- can always be recovered by starting that run again and stopping at the
same place. ``state``, ``get_seed`` and pickling do that; it costs at most
one run's worth of draws.

``TRandom`` itself is ROOT's base class, a linear congruential generator ROOT
itself says not to use for anything statistical (its period is 2**31); it is
here because it is ROOT's, and because a macro that used it deserves the
same numbers. :class:`TRandom3` is the one to use, and the one ``gRandom`` is.
"""

from __future__ import annotations

import functools
import math
import operator
import uuid
from collections.abc import Callable
from typing import Any, Union

import numpy as np

from . import gauss, histogram, landau, libm, poisson

__all__ = ["Sampler", "TRandom", "nonzero", "uuid_bytes"]

#: The fewest draws made at a time, so that drawing one number at a time
#: costs a slice of an array rather than a trip to the generator.
SPARE = 1024
#: Up to how many numbers a rejection algorithm makes one by one, in Python,
#: rather than an array at a time: below this, setting up the arrays costs more.
ONE_BY_ONE = 8
#: The most draws looked at at once by a rejection algorithm, to bound memory.
BATCH = 1 << 20
#: The widest unsigned integer ROOT's ``UInt_t`` holds, plus one.
UINT = 1 << 32
#: ROOT's seeds are ``ULong_t``: sixty-four bits.
ULONG = 1 << 64
#: ``TRandom::Rndm``'s scale: ``4.6566128730774E-10``, which is 1/2**31 rounded
#: to the figures ROOT wrote down rather than 1/2**31 itself.
LCG_SCALE = 4.6566128730774e-10
#: The linear congruential generator's multiplier and increment, BSD ``rand``'s.
LCG_A, LCG_C = 1103515245, 12345
#: How many steps of it are taken per pass of arithmetic.
LCG_BLOCK = 1 << 16
#: ``TMath::PiOver2()`` and ``TMath::TwoPi()``.
HALF_PI, TWO_PI = math.pi / 2, 2 * math.pi
#: What ``Rannor`` multiplies its second draw by, as ROOT writes it.
RANNOR_TWO_PI = 6.28318530717958623

#: A decoder for a rejection algorithm: given draws and how many numbers are
#: still wanted, the numbers it could finish, column by column, and how many
#: draws those took.
Decoder = Callable[[Any, int], "tuple[list[Any], int]"]
#: What ``n`` may be: a count, or nothing for a single number.
Count = Union[int, None]


def uuid_bytes() -> bytes:
    """Sixteen unpredictable bytes, where ROOT takes the sixteen of a ``TUUID``.

    A seed of zero asks ROOT for an unrepeatable stream, and ROOT gets one from
    a UUID made from the time and the machine. What matters is only that it is
    different every time, so a random UUID does as well; a zero seed is not
    reproducible here either, and cannot be expected to give ROOT's numbers.
    """
    return uuid.uuid4().bytes


def nonzero(draw: Callable[[int], Any], count: int) -> Any:
    """``count`` 32-bit words from ``draw`` with every zero thrown away and made up.

    ``TRandom2`` and ``TRandom3`` draw again when a word comes out zero, which
    is what keeps ``Rndm()`` off zero; the replacement is simply the next word.
    """
    words = draw(count)
    while not words.all():
        words = words[words != 0]
        words = np.concatenate([words, draw(count - len(words))])
    return words


def seed_value(seed: Any) -> int:
    """A seed as ROOT's ``ULong_t`` takes it, or a refusal of one it could not be."""
    try:
        value = operator.index(seed)
    except TypeError:
        raise TypeError(f"A seed is a whole number, not {type(seed).__name__}.") from None
    if not 0 <= value < ULONG:
        raise ValueError(f"A seed is an unsigned 64-bit number, which {value} is not.")
    return value


def _how_many(n: Any) -> int:
    """``n`` as a count of numbers, or a refusal."""
    try:
        count = operator.index(n)
    except TypeError:
        raise TypeError(f"n is how many numbers to draw, a whole number, not {n!r}.") from None
    if count < 0:
        raise ValueError(f"n is how many numbers to draw, which cannot be {count}.")
    return count


def _count(n: Count, *params: Any) -> tuple[int, tuple[int, ...] | None]:
    """How many numbers to draw, and the shape to hand them back in - ``None`` for one number.

    With no ``n`` the parameters decide: arrays of them draw one number each,
    plain numbers draw a single number. With ``n``, that many are drawn and
    the parameters must go with them the way NumPy broadcasts.
    """
    if n is None and all(isinstance(param, (int, float)) for param in params):
        return 1, None
    if n is None:
        shape = _broadcast(params, "Parameters of these shapes do not go together")
        return (int(np.prod(shape)), shape) if shape else (1, None)
    count = _how_many(n)
    if _broadcast(((count,), *params), f"The parameters cannot go with n={count}") != (count,):
        raise ValueError(f"The parameters cannot go with n={count}: they are more numbers.")
    return count, (count,)


def _broadcast(params: tuple[Any, ...], refusal: str) -> tuple[int, ...]:
    """The shape parameters broadcast to, or ``refusal`` if they do not go together."""
    shapes = [param if isinstance(param, tuple) else np.shape(param) for param in params]
    try:
        return tuple(np.broadcast_shapes(*shapes))
    except ValueError:
        raise ValueError(f"{refusal}: {shapes}.") from None


def _shaped(values: Any, shape: tuple[int, ...] | None) -> Any:
    """The numbers as asked for: an array of that shape, or one Python number."""
    values = np.asarray(values)
    return values.reshape(shape) if shape is not None else values.reshape(-1)[0].item()


def _like(draws: Any, shape: tuple[int, ...] | None) -> Any:
    """Draws laid out in the shape the parameters have, so that the two go together."""
    return draws.reshape(shape) if shape else draws


def _float(value: Any) -> Any:
    """A parameter as ROOT's ``Double_t`` has it: an array of doubles, or a double."""
    return np.asarray(value, dtype=np.float64)


def _scalar(name: str, value: Any) -> float:
    """A parameter that decides how many draws a number takes, which must be one number."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim:
        raise ValueError(
            f"{name} decides how many draws each number takes, so it must be one number "
            f"for the whole call rather than an array."
        )
    return float(array)


@functools.cache
def _lcg_steps(size: int) -> tuple[Any, Any]:
    """The multipliers and increments that take the generator 1, 2, ... ``size`` steps at once.

    Step ``t`` from ``x`` is ``(a_t * x + c_t) mod 2**31``; doubling the table
    composes it with its own last entry, so the whole table is a few passes.
    """
    a = np.array([LCG_A], dtype=np.uint64)
    c = np.array([LCG_C], dtype=np.uint64)
    mask = np.uint64(0x7FFFFFFF)
    while len(a) < size:
        a_last, c_last = a[-1], c[-1]
        a = np.concatenate([a, (a * a_last) & mask])
        c = np.concatenate([c, (a[: len(c)] * c_last + c) & mask])
    return a[:size], c[:size]


class Sampler:
    """The draws of a generator made ahead, each run with the state it was made from."""

    __slots__ = ("_held", "_offset", "_runs")

    def __init__(self) -> None:
        self._runs: list[tuple[Any, Any]] = []
        self._offset = 0
        self._held = 0

    def held(self) -> int:
        """How many draws are made and not yet handed out."""
        return self._held

    def add(self, state: Any, run: Any) -> None:
        """Hold a run of draws, made starting from ``state``."""
        self._runs.append((state, run))
        self._held += len(run)

    def peek(self, count: int) -> Any:
        """The next ``count`` draws held, without handing them out; there must be that many."""
        first = self._runs[0][1] if self._runs else np.empty(0)
        if self._offset + count <= len(first):
            return first[self._offset : self._offset + count]
        rest = [run for _, run in self._runs[1:]]
        return np.concatenate([first[self._offset :], *rest])[:count]

    def pop(self) -> float | None:
        """The next draw held, handed out, or ``None`` if none is."""
        if not self._runs:
            return None
        value = float(self._runs[0][1][self._offset])
        self.advance(1)
        return value

    def advance(self, count: int) -> None:
        """Hand out the next ``count`` draws, letting go of the runs that are finished."""
        self._offset += count
        self._held -= count
        while self._runs and self._offset >= len(self._runs[0][1]):
            self._offset -= len(self._runs.pop(0)[1])

    def rewind(self) -> tuple[Any, int] | None:
        """The state the first held run was made from and how far into it the draws have gone.

        Nothing is held afterwards: the caller puts the generator back there.
        """
        if not self._runs:
            return None
        start = (self._runs[0][0], self._offset)
        self._runs, self._offset, self._held = [], 0, 0
        return start


class TRandom:
    """ROOT's ``TRandom``: the distributions every generator offers, over its own ``Rndm()``.

    As a generator it is ROOT's base class, BSD ``rand``'s linear congruence
    with ROOT's scale, and ROOT's default seed of 65539. Every method takes
    ``n`` for an array of that many numbers - the same numbers, in the same
    order, as ``n`` calls in ROOT would give - and without it gives one.

        >>> from xrdroot.random import TRandom3
        >>> r = TRandom3(4357)
        >>> r.rndm(3)            # gRandom->Rndm() three times       # doctest: +SKIP
        >>> r.gaus(0, 1, n=1000)                                     # doctest: +SKIP
    """

    #: The seed ROOT's constructor defaults to.
    DEFAULT_SEED = 65539

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self._held = Sampler()
        self._engine_init()
        self.set_seed(seed_value(seed) % UINT)

    # -- the generator: what a subclass replaces ---------------------------------------

    def _engine_init(self) -> None:
        """Make whatever the generator keeps its state in."""
        self._seed = 0

    def _engine_seed(self, seed: int) -> None:
        """``SetSeed``: zero for a seed from a UUID, else the seed cut to ``UInt_t``."""
        self._seed = int.from_bytes(uuid_bytes()[:4], "little") if seed == 0 else seed % UINT

    def _engine_fill(self, count: int) -> Any:
        """The next ``count`` values of ``Rndm()``, advancing the generator past them."""
        seeds = np.empty(0, dtype=np.uint64)
        while len(seeds) < count:
            seeds = np.concatenate([seeds, self._lcg(count - len(seeds))])
            seeds = seeds[seeds != 0]
        return seeds.astype(np.float64) * LCG_SCALE

    def _lcg(self, count: int) -> Any:
        """The next ``count`` values of ``fSeed``, zeros and all, a block of steps at a time."""
        a, c = _lcg_steps(LCG_BLOCK)
        out = []
        start = np.uint64(self._seed & 0x7FFFFFFF)
        for size in [LCG_BLOCK] * (count // LCG_BLOCK) + [count % LCG_BLOCK]:
            block = (a[:size] * start + c[:size]) & np.uint64(0x7FFFFFFF)
            out.append(block)
            start = block[-1] if size else start
        self._seed = int(start)
        return np.concatenate(out)

    def _engine_export(self) -> dict[str, Any]:
        """The generator's state, named as ROOT's data members are."""
        return {"fSeed": self._seed}

    def _engine_import(self, state: dict[str, Any]) -> None:
        """Put the generator in a state :meth:`_engine_export` gave."""
        self._seed = _member(state, "fSeed", 0, UINT)

    def _engine_get_seed(self) -> int:
        """What ROOT's ``GetSeed`` answers, which for ``TRandom`` is ``fSeed``."""
        return self._seed

    # -- state ---------------------------------------------------------------------------

    def set_seed(self, seed: int) -> None:
        """``SetSeed(seed)``: restart the stream. Zero asks for an unrepeatable seed, as in ROOT."""
        value = seed_value(seed)
        self._held = Sampler()
        self._engine_seed(value)

    def get_seed(self) -> int:
        """``GetSeed()``, whatever ROOT's class reports; for ``TRandom3``, a word of its state."""
        self._settle()
        return self._engine_get_seed()

    @property
    def state(self) -> dict[str, Any]:
        """The generator's state, named as ROOT's data members are, after the draws handed out."""
        self._settle()
        return self._engine_export()

    def set_state(self, state: dict[str, Any]) -> None:
        """Put the generator in a state :attr:`state` gave, from this object or another."""
        self._held = Sampler()
        self._engine_import(dict(state))

    def __getstate__(self) -> dict[str, Any]:
        return self.state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._held = Sampler()
        self._engine_init()
        self._engine_import(state)

    def _settle(self) -> None:
        """Put the generator where the draws handed out have left it, and hold nothing."""
        start = self._held.rewind()
        if start is not None:
            self._engine_import(start[0])
            self._engine_fill(start[1])

    # -- draws ---------------------------------------------------------------------------

    def _peek(self, count: int) -> Any:
        """The next ``count`` values of ``Rndm()``, made if need be but not yet handed out."""
        missing = count - self._held.held()
        if missing > 0:
            state = self._engine_export()
            self._held.add(state, self._engine_fill(max(missing, SPARE)))
        return self._held.peek(count)

    def _take(self, count: int) -> Any:
        """The next ``count`` values of ``Rndm()``, handed out."""
        if not self._held.held() and count >= SPARE:
            return self._engine_fill(count)
        draws = self._peek(count)
        self._held.advance(count)
        return draws

    def _next(self) -> float:
        """The next value of ``Rndm()``, for the algorithms that go one number at a time."""
        value = self._held.pop()
        if value is None:
            self._peek(1)
            value = self._held.pop()
        return value  # type: ignore[return-value]

    def _one_by_one(self, count: int, make: Callable[[Callable[[], float]], float]) -> Any:
        """``count`` numbers, each made by ROOT's loop written out, drawing one at a time."""
        return np.array([make(self._next) for _ in range(count)], dtype=np.float64)

    def _sequential(self, count: int, decode: Decoder, columns: int, per: float) -> list[Any]:
        """``count`` numbers from an algorithm that takes a varying number of draws for each.

        ``per`` is a first guess at the draws each takes; after the first pass
        the guess is what they did take. Draws looked at and not used stay
        held for whatever is asked for next.
        """
        parts: list[list[Any]] = [[np.empty(0)] for _ in range(columns)]
        ask = min(int(count * per) + 32, BATCH)
        while count:
            found, used = decode(self._peek(ask), count)
            self._held.advance(used)
            if len(found[0]):
                for column, values in zip(parts, found):
                    column.append(values)
                ask = min(int((count - len(found[0])) * used / len(found[0]) * 1.05) + 32, BATCH)
                count -= len(found[0])
            else:
                ask *= 2
        return [np.concatenate(column) for column in parts]

    # -- ROOT's distributions ------------------------------------------------------------

    def rndm(self, n: Count = None) -> Any:
        """``Rndm()``: a number in ]0, 1[, the next in the stream; ``RndmArray`` with ``n``."""
        if n is None:
            return self._next()
        return np.array(self._take(_how_many(n)))

    def uniform(self, a: Any = 1.0, b: Any = None, n: Count = None) -> Any:
        """``Uniform(a)`` in ]0, a[, or ``Uniform(a, b)`` in ]a, b[."""
        if b is None:
            count, shape = _count(n, a)
            return _shaped(_float(a) * _like(self._take(count), shape), shape)
        count, shape = _count(n, a, b)
        draws = _like(self._take(count), shape)
        return _shaped(_float(a) + (_float(b) - _float(a)) * draws, shape)

    def gaus(self, mean: Any = 0.0, sigma: Any = 1.0, n: Count = None) -> Any:
        """``Gaus(mean, sigma)``, by ROOT's acceptance-complement algorithm (see :mod:`.gauss`)."""
        count, shape = _count(n, mean, sigma)
        if shape is None:
            return float(mean) + float(sigma) * gauss.one(self._next)
        if count <= ONE_BY_ONE:
            result = self._one_by_one(count, gauss.one)
        else:
            (result,) = self._sequential(count, gauss.standard, 1, 1.5)
        return _shaped(_float(mean) + _float(sigma) * _like(result, shape), shape)

    def rannor(self, n: Count = None) -> tuple[Any, Any]:
        """``Rannor(a, b)``: two independent standard normal numbers, by Box-Muller.

        These are the ``Double_t`` overload's; the ``Float_t`` one's are these
        rounded to single precision, ``np.float32(a)``.
        """
        count, shape = _count(n)
        draws = self._take(2 * count).reshape(count, 2)
        x = draws[:, 1] * RANNOR_TWO_PI
        r = np.sqrt(-2 * libm.log(draws[:, 0]))
        return _shaped(r * libm.sin(x), shape), _shaped(r * libm.cos(x), shape)

    def exp(self, tau: Any, n: Count = None) -> Any:
        """``Exp(tau)``: a number from an exponential of mean ``tau``."""
        count, shape = _count(n, tau)
        return _shaped(-_float(tau) * libm.log(_like(self._take(count), shape)), shape)

    def integer(self, imax: Any, n: Count = None) -> Any:
        """``Integer(imax)``: a whole number in [0, imax[, as ``UInt_t(imax * Rndm())``."""
        count, shape = _count(n, imax)
        top = np.asarray(imax, dtype=np.int64) % UINT
        return _shaped((top * _like(self._take(count), shape)).astype(np.int64), shape)

    def breit_wigner(self, mean: Any = 0.0, gamma: Any = 1.0, n: Count = None) -> Any:
        """``BreitWigner(mean, gamma)``: a Cauchy number of that peak and full width."""
        count, shape = _count(n, mean, gamma)
        rval = 2 * _like(self._take(count), shape) - 1
        return _shaped(_float(mean) + 0.5 * _float(gamma) * libm.tan(rval * HALF_PI), shape)

    def landau(self, mean: Any = 0.0, sigma: float = 1.0, n: Count = None) -> Any:
        """``Landau(mean, sigma)``, from CERNLIB's quantile table (see :mod:`.landau`).

        ROOT answers a ``sigma`` of zero or less with 0, and draws nothing for it.
        """
        spread = _scalar("sigma", sigma)
        count, shape = _count(n, mean)
        if spread <= 0:
            return _shaped(np.zeros(count), shape)
        values = landau.quantile(self._take(count), spread)
        return _shaped(_float(mean) + _like(values, shape), shape)

    def circle(self, r: Any, n: Count = None) -> tuple[Any, Any]:
        """``Circle(x, y, r)``: a point drawn uniformly on a circle of radius ``r``."""
        count, shape = _count(n, r)
        phi = TWO_PI * _like(self._take(count), shape)
        radius = _float(r)
        return _shaped(radius * libm.cos(phi), shape), _shaped(radius * libm.sin(phi), shape)

    def sphere(self, r: Any, n: Count = None) -> tuple[Any, Any, Any]:
        """``Sphere(x, y, z, r)``: a point drawn uniformly on a sphere of radius ``r``.

        ROOT throws pairs of draws at a square until one lands in the circle
        inside it, so each point takes an even number of draws.
        """
        count, shape = _count(n, r)
        a, b, r2 = (
            _like(column, shape) for column in self._sequential(count, _sphere_pairs, 3, 2.6)
        )
        radius = _float(r)
        scale = 8.0 * radius * np.sqrt(0.25 - r2)
        z = radius * (-1.0 + 8.0 * r2)
        return _shaped(a * scale, shape), _shaped(b * scale, shape), _shaped(z, shape)

    def binomial(self, ntot: int, prob: float, n: Count = None) -> Any:
        """``Binomial(ntot, prob)``: how many of ``ntot`` draws came out at or below ``prob``.

        A ``prob`` outside [0, 1] is 0, and draws nothing, as in ROOT.
        """
        trials, chance = _how_many(ntot), _scalar("prob", prob)
        count, shape = _count(n)
        if chance < 0 or chance > 1 or not trials:
            return _shaped(np.zeros(count, dtype=np.int64), shape)
        rows = max(1, BATCH // trials)
        found = [
            np.count_nonzero(~(self._take(size * trials).reshape(size, trials) > chance), axis=1)
            for size in [rows] * (count // rows) + [count % rows]
        ]
        return _shaped(np.concatenate(found).astype(np.int64), shape)

    def poisson(self, mean: float, n: Count = None) -> Any:
        """``Poisson(mean)``: a count, by whichever of ROOT's three algorithms the mean asks for."""
        mu = _poisson_mean(mean)
        if mu >= 2.0**62:
            raise ValueError(f"A Poisson count of mean {mu} does not fit 64 bits; use poisson_d.")
        count, shape = _count(n)
        if mu >= 1e9:
            return _shaped(np.trunc(self._poisson_gaus(mu, count)).astype(np.int64), shape)
        return _shaped(self._poisson(mu, count).astype(np.int64), shape)

    def poisson_d(self, mean: float, n: Count = None) -> Any:
        """``PoissonD(mean)``: ``Poisson`` as a double, which above 1e9 is not rounded."""
        mu = _poisson_mean(mean)
        count, shape = _count(n)
        if mu >= 1e9:
            return _shaped(self._poisson_gaus(mu, count), shape)
        return _shaped(self._poisson(mu, count).astype(np.float64), shape)

    def _poisson(self, mean: float, count: int) -> Any:
        """Counts below a mean of 1e9, by the product of draws or by rejection."""
        if mean <= 0:
            return np.zeros(count)
        if count <= ONE_BY_ONE:
            make = poisson.product_one if mean < 25 else poisson.rejection_one
            return self._one_by_one(count, functools.partial(make, mean=mean))
        if mean < 25:
            decode = functools.partial(poisson.product, mean=mean)
            return self._sequential(count, decode, 1, mean + 1)[0]
        decode = functools.partial(poisson.rejection, mean=mean)
        return self._sequential(count, decode, 1, 3.0)[0]

    def _poisson_gaus(self, mean: float, count: int) -> Any:
        """ROOT's Gaussian approximation for a mean past 1e9, before any rounding."""
        return self.gaus(0.0, 1.0, n=count) * math.sqrt(mean) + mean + 0.5

    def from_distribution(
        self, contents: Any, edges: Any, n: Count = None, *, width: bool = False
    ) -> Any:
        """``TH1::GetRandom``: a number distributed as a histogram's bins (see :mod:`.histogram`).

        ``contents`` are the bins without their flow, ``edges`` the axis: an
        :class:`~xrdroot.Axis`, its ``nbins + 1`` edges, or ``(low, high)`` for
        even bins. ``width`` weights each bin by its width, ROOT's ``"width"``.
        A histogram with nothing in it gives 0 and draws nothing, as ROOT's does.
        """
        values = np.asarray(contents, dtype=np.float64).ravel()
        lows, widths = histogram.bins(edges, len(values))
        integral = histogram.cumulative(values, widths if width else None)
        count, shape = _count(n)
        if integral is None:
            return _shaped(np.zeros(count), shape)
        return _shaped(histogram.inverse(self._take(count), integral, lows, widths), shape)


def _poisson_mean(mean: Any) -> float:
    """A Poisson mean, refused if it is not a number ROOT's comparisons can sort."""
    mu = _scalar("mean", mean)
    if math.isnan(mu):
        raise ValueError("A Poisson mean cannot be NaN: ROOT's Poisson has no answer for one.")
    return mu


def _sphere_pairs(draws: Any, limit: int) -> tuple[list[Any], int]:
    """The pairs of draws ``Sphere`` accepts, up to ``limit`` of them, and the draws they took."""
    pairs = draws[: len(draws) // 2 * 2].reshape(-1, 2)
    a, b = pairs[:, 0] - 0.5, pairs[:, 1] - 0.5
    r2 = a * a + b * b
    hit = np.flatnonzero(r2 <= 0.25)[:limit]
    used = int(2 * (hit[-1] + 1)) if len(hit) else 0
    return [a[hit], b[hit], r2[hit]], used


def _member(state: dict[str, Any], name: str, low: int, high: int) -> int:
    """A whole-number member of a saved state, refused by name if it is missing or out of range."""
    if name not in state:
        raise ValueError(f"A saved state needs {name}, which this one does not have.")
    value = operator.index(state[name])
    if not low <= value < high:
        raise ValueError(f"{name} must be at least {low} and below {high}, not {value}.")
    return value
