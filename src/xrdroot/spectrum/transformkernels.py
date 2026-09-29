"""The fast transforms themselves - Haar, Walsh, Fourier - on ROOT's working space.

Each is ``TSpectrumTransform::Haar``, ``Walsh``, ``BitReverse`` and
``Fourier`` over the same ``working_space`` the C++ passes around: the
spectrum in its first ``num`` places, a scratch or imaginary half after it.
A stage's butterflies touch disjoint places, so a stage is done at once over
every butterfly, each worked out as ROOT works it out; what the scratch half
holds afterwards is what ROOT leaves there too, since a later step may read
it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..random import libm
from .transformtypes import FORWARD, INVERSE, PI, stages

__all__ = ["bit_reversed", "bit_reverse", "fourier", "haar", "walsh"]


def bit_reversed(num: int) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Where each of ``num`` places goes when its index is written backwards in binary."""
    bits = stages(num)
    index = np.arange(num)
    reversed_index = np.zeros(num, dtype=np.intp)
    for bit in range(bits):
        reversed_index |= ((index >> bit) & 1) << (bits - 1 - bit)
    return reversed_index


def _haar_forward(ws: Any, num: int, levels: int) -> None:
    """The sums and differences of neighbours, level by level down to one sum."""
    for m in range(1, levels + 1):
        half = 2 ** (levels - m)
        ws[num : num + 2 * half] = ws[: 2 * half]
        even, odd = ws[num : num + 2 * half : 2], ws[num + 1 : num + 2 * half : 2]
        ws[half : 2 * half] = even - odd
        ws[:half] = even + odd


def _haar_inverse(ws: Any, num: int, levels: int) -> None:
    """Sums and differences back into neighbours, from the coarsest level up."""
    for m in range(1, levels + 1):
        half = int(math.pow(2.0, m - 1))
        ws[num : num + 2 * half] = ws[: 2 * half]
        low, high = ws[num : num + half], ws[num + half : num + 2 * half]
        ws[1 : 2 * half : 2] = low - high
        ws[0 : 2 * half : 2] = low + high


def _haar_scale(ws: Any, levels: int) -> None:
    """Each level's coefficients divided by the square root of the width it sums."""
    whole = math.sqrt(math.pow(2.0, levels))
    ws[0] = ws[0] / whole
    ws[1] = ws[1] / whole
    for level in range(2, levels + 1):
        weight = 1 / math.sqrt(math.pow(2.0, levels - level + 1))
        ws[2 ** (level - 1) : 2**level] *= weight


def haar(ws: Any, num: int, direction: int) -> None:
    """``Haar``: the forward transform, or - scaled first, as ROOT scales - the inverse."""
    ws[num : 2 * num] = 0.0
    levels = stages(num)
    if direction == FORWARD:
        _haar_forward(ws, num, levels)
    _haar_scale(ws, levels)
    if direction == INVERSE:
        _haar_inverse(ws, num, levels)


def walsh(ws: Any, num: int) -> None:
    """``Walsh``: the Walsh-Hadamard butterflies, in natural order, over ``sqrt(num)``."""
    ws[num : 2 * num] = 0.0
    blocks = 1
    for m in range(1, stages(num) + 1):
        blocks = 1 if m == 1 else blocks * 2
        width = num // blocks
        low = (np.arange(blocks)[:, None] * width + np.arange(width // 2)[None, :]).ravel()
        high = low + width // 2
        first, second = ws[low], ws[high]
        ws[low + num] = first + second
        ws[high + num] = first - second
        ws[:num] = ws[num : 2 * num]
    ws[:num] = ws[:num] / math.sqrt(float(num))


def bit_reverse(ws: Any, num: int) -> None:
    """``BitReverse``: each place moved to its index written backwards."""
    ws[num : 2 * num] = ws[:num]
    ws[bit_reversed(num)] = ws[num : 2 * num]


def _fourier_stage(ws: Any, num: int, span: int, sign: float) -> None:
    """One radix-2 stage: every butterfly ``span`` wide, with its twiddle factor."""
    half = span // 2
    step = np.arange(half)
    arg = step.astype(np.float64) * (PI / float(half))
    blocks = num // span
    wr, wi = np.repeat(libm.cos(arg), blocks), np.repeat(sign * libm.sin(arg), blocks)
    j1 = (np.arange(0, num, span)[None, :] + step[:, None]).ravel()
    j2 = j1 + half
    a, b, c, d = ws[j1], ws[j2], ws[j1 + num], ws[j2 + num]
    tr, ti = a - b, c - d
    ws[j1], ws[j1 + num] = a + b, c + d
    ws[j2] = tr * wr - ti * wi
    ws[j2 + num] = ti * wr + tr * wi


def _fourier_scale(ws: Any, num: int, hartley: int) -> None:
    """Both parts over ``sqrt(num)`` - or, for Hartley's, their sum, the imaginary part cleared."""
    root = math.sqrt(float(num))
    if hartley == 0:
        ws[:num] = ws[:num] / root
        ws[num : 2 * num] = ws[num : 2 * num] / root
        return
    ws[:num] = (ws[:num] + ws[num : 2 * num]) / root
    ws[num : 2 * num] = 0.0


def fourier(ws: Any, num: int, hartley: int, direction: int, zt_clear: int) -> None:
    """``Fourier``: the FFT of the real and imaginary halves, or Hartley's transform.

    A forward transform clears the imaginary half first unless told the
    caller has put one there (``zt_clear``); the inverse Hartley transform
    is the forward one read backwards, as ROOT reads it.
    """
    if direction == FORWARD and zt_clear == 0:
        ws[num : 2 * num] = 0.0
    sign = 1.0 if direction == INVERSE else -1.0
    span = num
    for _ in range(stages(num)):
        _fourier_stage(ws, num, span, sign)
        span //= 2
    order = bit_reversed(num)
    ws[:num], ws[num : 2 * num] = ws[:num][order], ws[num : 2 * num][order]
    _fourier_scale(ws, num, hartley)
    if hartley == 1 and direction == INVERSE:
        ws[:num] = ws[:num][(-np.arange(num)) % num]
        ws[num : 2 * num] = 0.0
