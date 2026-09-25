"""``TRandom::Gaus``: ROOT's normal numbers, taking ROOT's draws in ROOT's order.

ROOT's ``Gaus`` is not Box-Muller. It is the acceptance-complement method of
Hörmann and Derflinger: one draw decides most numbers outright - more than
half the time nothing more is needed - a second settles most of the rest,
and what is left goes to a loop over pairs of draws in the tails that runs
until a pair is accepted. Every constant below is ROOT's, and every
comparison is made in ROOT's order with ROOT's arithmetic, so a number here
is the number ROOT gives and it uses up the draws ROOT's uses up.

For a whole array, every draw is treated as a possible first draw: what the
number starting there would be, and where the next one would start (see
:mod:`.chain`). The tail loop is answered for every possible starting pair
at once too, by finding the first accepted pair at or after each one. The
cost is a few passes of arithmetic over about 1.45 draws per number, and
twenty or so passes of the pointer chase, whatever the tails do. One number
at a time, that is a great deal of NumPy for one draw and a half, so a few
numbers are made by :func:`one` instead, which is ROOT's loop written out.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from . import chain, libm

__all__ = ["one", "standard"]

#: ROOT's constants for the central region, named as ``TRandom::Gaus`` names them.
K_C1, K_C2, K_C3 = 1.448242853, 3.307147487, 1.46754004
K_D1, K_D2, K_D3 = 1.036467755, 5.295844968, 3.631288474
K_HM, K_ZM, K_HP, K_ZP = 0.483941449, 0.107981933, 4.132731354, 18.52161694
K_PHLN, K_HM1, K_HP1 = 0.4515827053, 0.516058551, 3.132731354
K_HZM, K_HZMP = 0.375959516, 0.591923442
#: ROOT's constants for the tails.
K_AS, K_BS, K_CS, K_B = 0.8853395638, 0.2452635696, 0.2770276848, 0.5029324303
K_X0, K_YM, K_S, K_T = 0.4571828819, 0.187308492, 0.7270572718, 0.03895759111


def standard(draws: Any, limit: int) -> tuple[list[Any], int]:
    """Up to ``limit`` numbers of ``Gaus(0, 1)`` from ``draws``, and how many draws they took.

    A number whose draws run past the end is not made: it and the draws it
    would have taken are left for the caller to extend and ask again.
    """
    values, links = every_start(np.asarray(draws, dtype=np.float64))
    starts = chain.follow(links, limit)
    used = int(links[starts[-1]]) if len(starts) else 0
    return [values[starts]], used


def every_start(draws: Any) -> tuple[Any, Any]:
    """For every draw, the number ``Gaus`` would make starting there, and where the next starts."""
    size = len(draws)
    values = np.full(size, np.nan)
    links = np.full(size, size + 1, dtype=np.int64)
    at = np.arange(size)
    tails = _tails(draws)
    _outright(draws, at, values, links)
    _central(draws, at, tails, values, links)
    rest = (draws >= K_HM) & (draws <= K_HM1)
    values[rest], links[rest] = tails[0][at[rest] + 1], tails[1][at[rest] + 1]
    return values, links


def _outright(draws: Any, at: Any, values: Any, links: Any) -> None:
    """The draws that decide a number alone: above ``kHm1``, or below ``kZm``."""
    high = draws > K_HM1
    values[high] = K_HP * draws[high] - K_HP1
    low = draws < K_ZM
    rn = K_ZP * draws[low] - 1
    values[low] = np.where(rn > 0, 1 + rn, -1 + rn)
    links[high | low] = at[high | low] + 1


def _central(draws: Any, at: Any, tails: tuple[Any, Any], values: Any, links: Any) -> None:
    """The draws between ``kZm`` and ``kHm``, which take a second and may go on to the tails."""
    where = at[(draws >= K_ZM) & (draws < K_HM) & (at + 1 < len(draws))]
    value, accepted = _centre(draws[where], draws[where + 1])
    values[where] = np.where(accepted, value, tails[0][where + 2])
    links[where] = np.where(accepted, where + 2, tails[1][where + 2])


def _centre(y: Any, second: Any) -> tuple[Any, Any]:
    """ROOT's four tests of the central region, in order: the number, and whether one passed."""
    rn = second - 1 + second
    z = np.where(rn > 0, 2 - rn, -2 - rn)
    x = rn * rn
    first = (K_C1 - y) * (K_C3 + np.abs(z)) < K_C2
    then = (y + K_D1) * (K_D3 + x) < K_D2
    third = K_HZMP - y < libm.exp(-(z * z + K_PHLN) / 2)
    fourth = y + K_HZM < libm.exp(-(x + K_PHLN) / 2)
    value = np.select([first, then, third, fourth], [z, rn, z, rn], np.nan)
    return value, first | then | third | fourth


def _tails(draws: Any) -> tuple[Any, Any]:
    """For every draw, the number the tail loop makes starting there, and the draw after its last.

    Both come back two longer than the draws, so that a loop starting at or
    past the end can be looked up too; it is unfinished, as is one that
    finds no accepted pair before the draws run out.
    """
    size = len(draws)
    rn, accepted = _tail_pairs(draws)
    first = np.concatenate([chain.first_from(accepted, 2), np.full(3, len(accepted))])
    found = first < len(accepted)
    ends = np.where(found, first + 2, size + 1)
    return np.concatenate([rn, [np.nan]])[first], ends


def _tail_pairs(draws: Any) -> tuple[Any, Any]:
    """The tail loop's number for every pair of draws, and whether ROOT accepts it."""
    x, y = draws[:-1], K_YM * draws[1:]
    right = K_X0 - K_S * x - y > 0
    x = np.where(right, x, 1 - x)
    y = np.where(right, y, K_YM - y)
    ratio = 2 + y / x
    rn = np.where(right, ratio, -ratio)
    squeezed = (y - K_AS + x) * (K_CS + x) + K_BS < 0
    inside = (y < x + K_T) & (rn * rn < 4 * (K_B - libm.log(x)))
    return rn, squeezed | inside


def one(draw: Callable[[], float]) -> float:
    """One number of ``Gaus(0, 1)``, by ROOT's loop as ROOT writes it, drawing from ``draw()``."""
    y = draw()
    if y > K_HM1:
        return K_HP * y - K_HP1
    if y < K_ZM:
        rn = K_ZP * y - 1
        return 1 + rn if rn > 0 else -1 + rn
    if y < K_HM:
        value = _centre_one(y, draw())
        if value is not None:
            return value
    return _tail_one(draw)


def _centre_one(y: float, second: float) -> float | None:
    """ROOT's four tests of the central region for one number, or ``None`` if all four fail."""
    rn = second - 1 + second
    z = 2 - rn if rn > 0 else -2 - rn
    if (K_C1 - y) * (K_C3 + abs(z)) < K_C2:
        return z
    x = rn * rn
    if (y + K_D1) * (K_D3 + x) < K_D2:
        return rn
    if K_HZMP - y < math.exp(-(z * z + K_PHLN) / 2):
        return z
    if y + K_HZM < math.exp(-(x + K_PHLN) / 2):
        return rn
    return None


def _tail_one(draw: Callable[[], float]) -> float:
    """ROOT's tail loop for one number: pairs of draws until one is accepted."""
    while True:
        x = draw()
        y = K_YM * draw()
        if K_X0 - K_S * x - y > 0:
            rn = 2 + y / x
        else:
            x, y = 1 - x, K_YM - y
            rn = -(2 + y / x)
        if (y - K_AS + x) * (K_CS + x) + K_BS < 0:
            return rn
        if y < x + K_T and rn * rn < 4 * (K_B - math.log(x)):
            return rn
