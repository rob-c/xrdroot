"""The sums SNIP clipping is made of, element by element in ROOT's order, over whole arrays.

Every clipping filter in ``TSpectrum`` estimates each channel from others
some distance away - the mean of the two at ``j - i`` and ``j + i``, or a
fourth-, sixth- or eighth-order combination of more - and each such
estimate is a short sum ROOT adds term by term. These are those sums, over
every channel of a pass at once, with each channel's terms added in ROOT's
order from ROOT's first term, so that each is ROOT's to the last bit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from typing import TypeAlias

__all__ = ["Array", "combination", "shifted_mean", "terms"]

#: A spectrum, or any array of ``Double_t``.
Array: TypeAlias = "np.ndarray[Any, np.dtype[np.float64]]"

#: ``(sign, coefficient, reach)`` of the terms of each filter order's estimate,
#: with the number they are divided by and the step ``i // step`` a reach is in:
#: the binomial weights ``TSpectrum::Background`` writes out, which for the
#: smoothed eighth order - as ROOT has it - take ``f8`` with the wrong sign.
ORDERS: dict[int, tuple[int, int, tuple[tuple[int, int, int], ...]]] = {
    4: (2, 6, ((-1, 1, -2), (1, 4, -1), (1, 4, 1), (-1, 1, 2))),
    6: (3, 20, ((1, 1, -3), (-1, 6, -2), (1, 15, -1), (1, 15, 1), (-1, 6, 2), (1, 1, 3))),
    8: (4, 70, ((-1, 1, -4), (1, 8, -3), (-1, 28, -2), (1, 56, -1),
                (1, 56, 1), (-1, 28, 2), (1, 8, 3), (-1, 1, 4))),
}  # fmt: skip

#: The smoothed eighth order's terms as ROOT adds them: ``- 56 * f8`` where the
#: unsmoothed filter, and the binomial weights, have ``+ 56``.
SMOOTHED_EIGHTH = ((-1, 1, -4), (1, 8, -3), (-1, 28, -2), (1, 56, -1),
                   (-1, 56, 1), (-1, 28, 2), (1, 8, 3), (-1, 1, 4))  # fmt: skip


def terms(order: int, smoothed: bool) -> tuple[int, int, tuple[tuple[int, int, int], ...]]:
    """The step, divisor and terms of ``order``'s estimate, smoothed or not."""
    step, divisor, found = ORDERS[order]
    return step, divisor, SMOOTHED_EIGHTH if smoothed and order == 8 else found


def shifted_mean(values: Array, centres: Any, half: int) -> Array:
    """The mean of ``values`` within ``half`` channels of each centre, inside the array.

    ``half`` 0 is the value at the centre itself. The channels are added from
    the lowest up, from zero, those outside left out and not counted - which
    adding 0.0 in their place does to the last bit.
    """
    size = values.shape[0]
    total = np.zeros(len(centres))
    count = np.zeros(len(centres))
    for offset in range(-half, half + 1):
        where = centres + offset
        inside = (where >= 0) & (where < size)
        total = total + np.where(inside, values[np.clip(where, 0, size - 1)], 0.0)
        count = count + inside
    return total / count if half else total


def combination(
    means: list[Array], signs: tuple[tuple[int, int, int], ...], divisor: int, each: bool
) -> Array:
    """``means`` weighted by ``signs``, divided by ``divisor``, added as ROOT adds them.

    ``each`` is the unsmoothed filters' way: ``c = 0; c -= x / 6; c += 4 * x / 6``,
    every term divided on its own. Otherwise it is the smoothed way,
    ``(-b4 + 4 * c4 + 4 * d4 - e4) / 6``: the first term negated, the others
    added, and the sum divided once.
    """
    if each:
        total = np.zeros(means[0].shape[0])
        for (sign, weight, _), mean in zip(signs, means, strict=False):
            term = weight * mean / divisor
            total = total + term if sign > 0 else total - term
        return total
    first_sign, first_weight, _ = signs[0]
    total = first_weight * means[0] if first_sign > 0 else -(first_weight * means[0])
    for (sign, weight, _), mean in zip(signs[1:], means[1:], strict=False):
        total = total + weight * mean if sign > 0 else total - weight * mean
    return total / divisor
