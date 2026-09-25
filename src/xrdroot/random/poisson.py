"""``TRandom::Poisson``: ROOT's three roads to a count, each taking ROOT's draws.

Below a mean of 25 ROOT multiplies draws together until the product falls
to ``exp(-mean)``; the count is how many it took, less one. From 25 to 1e9 it
uses the rejection method of *Numerical Recipes*: a Lorentzian candidate
from ``tan(pi * Rndm())``, kept or thrown away by one more draw against
``LnGamma``. Above that it is a Gaussian (see :mod:`.gauss`), rounded.

The product loop is the delicate one to do an array at a time, because it
rounds at every step and the count is decided by where that rounded
product crosses a threshold. Sums of logarithms say where every number would
end cheaply, for every possible start at once; then the product each chosen
number really takes is formed in ROOT's order, draw by draw, a whole array
of numbers abreast, and a number is only kept once its own product has been
seen to cross exactly where the logarithms said. If one ever does not - the
two agree to a part in 10**13, so it is a matter of principle rather than of
practice - that number is made the slow way, draw by draw, and the fast way
resumes after it.

The rejection method's attempts follow one another regardless of which
number they belong to, so the run of attempts is followed through the draws
once (see :mod:`.chain`) and the accepted ones are the counts.
``LnGamma`` is the C library's ``lgamma``, which is what ROOT calls; Python's
own ``math.lgamma`` is a different implementation and differs from it in the
last place about half the time, so it is only the fallback where no C
library can be found. A count or two at a time is made by ROOT's loops
written out, :func:`product_one` and :func:`rejection_one`, which cost far
less than setting up the arrays would.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from . import chain, libm

__all__ = ["LGAMMA", "c_lgamma", "product", "product_one", "rejection", "rejection_one"]


def c_lgamma(library: str | None) -> Callable[[float], float]:
    """The C library's ``lgamma``, found by name, or Python's where there is none to be had."""
    path = ctypes.util.find_library(library) if library else None
    if path is None:
        return math.lgamma
    function = ctypes.CDLL(path).lgamma
    function.restype = ctypes.c_double
    function.argtypes = [ctypes.c_double]
    return function


#: ``TMath::LnGamma``, which is ``::lgamma`` from the C library.
LGAMMA = c_lgamma("m")


def product(
    draws: Any, limit: int, mean: float, guess: float | None = None
) -> tuple[list[Any], int]:
    """Up to ``limit`` counts for a mean below 25, and how many draws they took.

    ``guess`` is the threshold the logarithms are summed against, which is the
    mean itself; it is there so that the slow path can be shown to work.
    """
    draws = np.asarray(draws, dtype=np.float64)
    below = math.exp(-mean)
    starts, lengths = _proposed(draws, mean if guess is None else guess, limit)
    good = _verified(draws, starts, lengths, below)
    if good or not len(starts):
        used = int(starts[good - 1] + lengths[good - 1]) if good else 0
        return [lengths[:good] - 1], used
    return _one_by_one(draws, below)


def _proposed(draws: Any, mean: float, limit: int) -> tuple[Any, Any]:
    """Where each number would start and how many draws it would take, by sums of logarithms."""
    size = len(draws)
    total = np.cumsum(np.log(draws))
    before = np.concatenate([[0.0], total[:-1]])
    ends = np.searchsorted(-total, mean - before, side="left")
    links = np.where(ends < size, ends + 1, size + 1)
    starts = chain.follow(links, limit)
    return starts, links[starts] - starts


def _verified(draws: Any, starts: Any, lengths: Any, below: float) -> int:
    """How many of the proposed numbers, from the first, ROOT's own products agree with.

    ``pir *= Rndm()`` is formed for every number abreast, in ROOT's order,
    and a number stands only if its product is above ``exp(-mean)`` one draw
    before its end and at or below it at its end.
    """
    pir = np.ones(len(starts))
    before = np.ones(len(starts))
    for step in range(int(lengths.max()) if len(lengths) else 0):
        live = np.flatnonzero(lengths > step)
        before[live] = pir[live]
        pir[live] *= draws[starts[live] + step]
    agreed = (pir <= below) & (before > below)
    return len(agreed) if agreed.all() else int(np.argmin(agreed))


def _one_by_one(draws: Any, below: float) -> tuple[list[Any], int]:
    """The first count, the way ROOT's loop makes it, when the fast way could not vouch for it."""
    pir = 1.0
    for index, draw in enumerate(draws.tolist()):
        pir *= draw
        if pir <= below:
            return [np.array([index], dtype=np.int64)], index + 1
    return [np.empty(0, dtype=np.int64)], 0


def rejection(draws: Any, limit: int, mean: float) -> tuple[list[Any], int]:
    """Up to ``limit`` counts for a mean from 25 to 1e9, and how many draws they took."""
    draws = np.asarray(draws, dtype=np.float64)
    size = len(draws)
    sq, alxm = math.sqrt(2.0 * mean), math.log(mean)
    g = mean * alxm - LGAMMA(mean + 1.0)
    y = libm.tan(math.pi * draws)
    em = sq * y + mean
    first = chain.first_from(em >= 0)
    attempts = chain.follow(np.where(first + 1 < size, first + 2, size + 1), size)
    at = first[attempts]
    count = np.floor(em[at])
    t = 0.9 * (1.0 + y[at] * y[at]) * libm.exp(count * alxm - _lgammas(count + 1.0) - g)
    kept = np.flatnonzero(draws[at + 1] <= t)[:limit]
    used = int(at[kept[-1]] + 2) if len(kept) else 0
    return [count[kept]], used


def _lgammas(values: Any) -> Any:
    """``LnGamma`` of every value, calling the C library once for each distinct one.

    The values are whole numbers a few standard deviations either side of
    the mean, so there are only a few hundred distinct ones however many
    counts are drawn.
    """
    distinct, where = np.unique(values, return_inverse=True)
    return np.array([LGAMMA(value) for value in distinct.tolist()], dtype=np.float64)[where]


def product_one(draw: Callable[[], float], mean: float) -> int:
    """One count below a mean of 25, by ROOT's loop: multiply draws until ``exp(-mean)``."""
    below = math.exp(-mean)
    pir, n = 1.0, 0
    while True:
        pir *= draw()
        if pir <= below:
            return n
        n += 1


def rejection_one(draw: Callable[[], float], mean: float) -> int:
    """One count from 25 to 1e9, by ROOT's rejection loop."""
    sq, alxm = math.sqrt(2.0 * mean), math.log(mean)
    g = mean * alxm - LGAMMA(mean + 1.0)
    while True:
        y = math.tan(math.pi * draw())
        em = sq * y + mean
        if em < 0.0:
            continue
        em = math.floor(em)
        t = 0.9 * (1.0 + y * y) * math.exp(em * alxm - LGAMMA(em + 1.0) - g)
        if draw() <= t:
            return em
