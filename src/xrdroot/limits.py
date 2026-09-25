"""``THLimitsFinder``: the axis ROOT's ``TTree::Draw`` picks when it is not told one.

Asked to draw ``pt`` with no binning, ROOT looks at the values first - the
first ``TTree::GetEstimate()`` of them - and books an axis that holds them
with round numbers at its ends: ``OptimizeLimits`` widens the range a little,
``Optimize`` rounds the width of a bin up to one, two or five times a power of
ten, and the ends are moved out to the nearest whole bin. An axis whose values
are whole numbers - an integer branch, ``Entry$``, ``Length$`` - also gets bins
a whole number wide, so every integer sits in the middle of one.

This is that code, statement for statement, in the order ROOT does it,
because which axis a histogram gets decides which bin every entry lands in and
so every number drawn from it. The one part left out is the rounding of time
axes, which ``TTree::Draw`` never asks for.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

__all__ = ["Limits", "find_good_limits", "optimize", "optimize_limits"]

#: ``FLT_MAX``: a bin wider than this is not rounded, as ``Optimize`` has it.
FLT_MAX = 3.4028234663852886e38
#: ``DBL_MAX``, where ``TSelectorDraw`` starts the search for a minimum.
DBL_MAX = 1.7976931348623157e308


class Limits(NamedTuple):
    """One axis as the limits finder leaves it: how many bins, and its two ends."""

    nbins: int
    low: float
    high: float


class _Width(NamedTuple):
    """What ``Optimize`` hands back: the rounded ends and the width of a bin."""

    low: float
    high: float
    nbins: int
    width: float


def _c_min(a: float, b: float) -> float:
    """``std::min``, which keeps ``a`` unless ``b`` is below it - NaN included."""
    return b if b < a else a


def _c_max(a: float, b: float) -> float:
    """``std::max``, which keeps ``a`` unless it is below ``b``."""
    return b if a < b else a


def _truncated(value: float) -> int:
    """``Int_t(value)``: C's conversion, towards zero."""
    return int(value)


def _mantissa(sigfig: float, jlog: int) -> tuple[float, int]:
    """Round a width's leading digits up to 1, 2 or 5, or on to the next power of ten."""
    if sigfig <= 1:
        return 1.0, jlog
    if sigfig <= 2:
        return 2.0, jlog
    if sigfig <= 5:
        return 5.0, jlog
    return 1.0, jlog + 1


def _rounded_width(awidth: float) -> float | None:
    """The width of a bin rounded as ``Optimize`` rounds it, or ``None`` for its 0-to-1 fallback."""
    if not math.isfinite(awidth):  # a NaN range, which C would make INT_MIN of
        return None
    jlog = _truncated(math.log10(awidth))
    if jlog < -200 or jlog > 200:
        return None
    if awidth <= 1:
        jlog -= 1
    # ROOT takes 1e-10 off so a width of exactly 2 or 5 is not rounded past.
    sigfig = awidth * math.pow(10, -jlog) - 1e-10
    siground, jlog = _mantissa(sigfig, jlog)
    return siground * math.pow(10, jlog)


def _placed(al: float, ah: float, width: float) -> tuple[float, float, int]:
    """The ends at whole multiples of ``width`` either side of ``al`` and ``ah``."""
    alb = al / width
    lwid = _truncated(alb)
    if alb < 0:
        lwid -= 1
    alb = ah / width + 1.00001
    kwid = _truncated(alb)
    if alb < 0:
        kwid -= 1
    return width * lwid, width * kwid, kwid - lwid


def _settled(al: float, ah: float, found: _Width) -> _Width:
    """``Optimize``'s last step: an end a whole bin past the range is brought back in."""
    low, high, nbins, width = found
    atest = width * 0.0001
    if al - low >= atest:
        low += width
        nbins -= 1
    if high - ah >= atest:
        high -= width
        nbins -= 1
    if low >= high:  # which can happen with five bins or fewer
        return found
    return _Width(low, high, nbins, width)


def _try(al: float, ah: float, nold: int, ntemp: int, nbins: int) -> _Width | None:
    """One pass of ``Optimize`` from ``L20``: ``None`` asks for the pass to be run again."""
    awidth = (ah - al) / float(ntemp)
    if awidth >= FLT_MAX or awidth <= 0:
        return _Width(0.0, 0.0, nbins, 0.0)  # what the caller started them at
    width = _rounded_width(awidth)
    if width is None:
        return _Width(0.0, 1.0, 100, math.nan)  # marked as ROOT's fallback of 0 to 1
    if abs(al / width) > 1e9:
        # ROOT caps nbins here at nold, which it already is: nothing has counted it yet.
        return _Width(al, ah, nbins, -width)  # marked to skip the settling
    return _counted(nold, _Width(*_placed(al, ah, width), width))


def _counted(nold: int, found: _Width) -> _Width | None:
    """Keep what a pass found, but for one bin asked for and half the bins found."""
    if nold <= 5:
        if not (nold > 1 or found.nbins == 1):  # one bin asked for is the hard case
            return _Width(found.low, found.high, 1, found.width * 2)
        return found
    return None if 2 * found.nbins == nold else found


def _search(al: float, ah: float, nold: int) -> _Width:
    """``Optimize`` from ``L20`` to ``LOK``: the width, and the ends it puts the range between.

    A pass that finds half the bins asked for is run again with one more,
    as ROOT's ``goto L20`` runs it.
    """
    ntemp = max(nold, 2)
    found = _try(al, ah, nold, ntemp, nold)
    while found is None:
        ntemp += 1
        found = _try(al, ah, nold, ntemp, nold // 2)
    return found


def optimize(a1: float, a2: float, nold: int) -> tuple[float, float, int, float]:
    """``THLimitsFinder::Optimize``: round ends and a round width for ``nold`` bins.

        >>> optimize(0.0, 108.9, 100)
        (0.0, 108.0, 54, 2.0)

    The low end, the high end, how many bins that makes and the width of one.
    A width that could not be rounded - a range of nothing, or of infinity -
    comes back as zero.
    """
    al, ah = _c_min(a1, a2), _c_max(a1, a2)
    if al == ah:
        ah = al + 1
    found = _search(al, ah, nold)
    if math.isnan(found.width):
        return 0.0, 1.0, 100, 0.01
    if found.width < 0:  # the bins would be too far from zero to count them
        return found.low, found.high, found.nbins, -found.width
    return tuple(_settled(al, ah, found))  # type: ignore[return-value]


def _whole_ends(xmin: float, xmax: float) -> tuple[float, float]:
    """The ends made whole numbers, outwards, and at least one apart."""
    dxmin, dxmax = float(_truncated(xmin)), float(_truncated(xmax))
    xmin = dxmin - 1 if xmin < 0 and xmin != dxmin else dxmin
    if xmax > 0 and xmax != dxmax:
        xmax = dxmax + 1
    elif xmax == 0 and xmax == dxmax:
        xmax = 1.0
    else:
        xmax = dxmax
    return xmin, (xmin + 1 if xmin >= xmax else xmax)


def _whole(nbins: int, xmin: float, xmax: float, umin: float, umax: float) -> Limits:
    """``OptimizeLimits`` for integers: whole ends, and bins a whole number wide."""
    xmin, xmax = _whole_ends(xmin, xmax)
    bw = _truncated((xmax - xmin) / nbins) or 1
    nbins = _truncated((xmax - xmin) / bw)
    if xmin + nbins * bw < umax:
        nbins += 1
        xmax = xmin + nbins * bw
    if xmin > umin:
        nbins += 1
        xmin = xmax - nbins * bw
    return Limits(nbins, xmin, xmax)


def _widened(nbins: int, xmin: float, xmax: float, integer: bool) -> tuple[float, float]:
    """The range ``OptimizeLimits`` rounds: a tenth wider, never past zero from one side."""
    dx = 5 * (xmax - xmin) / nbins if integer else 0.1 * (xmax - xmin)
    umin, umax = xmin - dx, xmax + dx
    if umin < 0 and xmin >= 0:
        umin = 0.0
    if umax > 0 and xmax <= 0:
        umax = 0.0
    return umin, umax


def optimize_limits(nbins: int, xmin: float, xmax: float, integer: bool = False) -> Limits:
    """``THLimitsFinder::OptimizeLimits``: the axis for values from ``xmin`` to ``xmax``.

        >>> optimize_limits(100, 0.0, 99.0)
        Limits(nbins=100, low=-0.99, high=108.0)

    The range is widened by a tenth - never past zero from one side of it -
    and rounded by :func:`optimize`; the ends are those, or a hundredth past
    the values, whichever is further out. The number of bins stays what it
    was asked to be, unless the values are ``integer`` and the bins are made
    a whole number wide.
    """
    umin, umax = _widened(nbins, xmin, xmax, integer)
    low, high, _count, width = optimize(umin, umax, nbins)
    if width <= 0 or width > 1e39:
        xmin, xmax = -1.0, 1.0
    else:
        delta = 0.01 * (xmax - xmin)
        xmin, xmax = _c_min(low, xmin - delta), _c_max(high, xmax + delta)
    if integer:
        return _whole(nbins, xmin, xmax, umin, umax)
    return Limits(nbins, xmin, xmax)


def find_good_limits(
    nbins: int, xmin: float, xmax: float, integer: bool = False
) -> Limits:
    """``THLimitsFinder::FindGoodLimits`` for one axis: a range of nothing is made one wide.

        >>> find_good_limits(100, -1.467, 2.344)       # ROOT's own example
        Limits(nbins=100, low=-1.8, high=2.7)

    ``FindGoodLimitsXY`` and ``FindGoodLimitsXYZ`` are this for each axis in
    turn, which is what :func:`good_axes` does.
    """
    if xmin >= xmax:
        xmin, xmax = xmin - 1, xmax + 1
    return optimize_limits(nbins, xmin, xmax, integer)


def good_axes(
    nbins: Sequence[int], ranges: Sequence[tuple[float, float]], integers: Sequence[bool]
) -> list[Limits]:
    """Every axis of a histogram found as :func:`find_good_limits` finds one."""
    return [
        find_good_limits(count, low, high, integer)
        for count, (low, high), integer in zip(nbins, ranges, integers)
    ]
