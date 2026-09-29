"""The mixed transforms' butterflies: ``GeneralExe``, ``GeneralInv`` and ``BitReverseHaar``.

Morhac's mixed transforms - Fourier-Walsh, Cosine-Haar and the rest - run
the first ``degree`` stages of one transform and the rest of another, over
a real part at the start of the working space and an imaginary part at
``2 * num``; the stage's results go to ``num`` and ``3 * num`` and are
copied back, as in ROOT. Every butterfly of a stage is independent, so a
stage is worked out at once, each butterfly by ROOT's own expression.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..random import libm
from .transformkernels import bit_reversed
from .transformtypes import HAAR_MIXED, HALF_SQRT2, PI, WALSH_HAAR, stages

__all__ = ["bit_reverse_haar", "general_exe", "general_inv"]


def bit_reverse_haar(ws: Any, shift: int, num: int, start: int) -> None:
    """``BitReverseHaar``: ``num`` places from ``start`` bit-reversed, real and imaginary alike."""
    real, imag = slice(start, start + num), slice(start + 2 * shift, start + 2 * shift + num)
    ws[start + shift : start + shift + num] = ws[real]
    ws[start + 3 * shift : start + 3 * shift + num] = ws[imag]
    order = bit_reversed(num)
    ws[start + order] = ws[start + shift : start + shift + num]
    ws[start + 2 * shift + order] = ws[start + 3 * shift : start + 3 * shift + num]


def _twiddles(num: int, blocks: int, ring: int, kind: int) -> tuple[Any, Any]:
    """Each block's ``wr`` and ``wi``: a Fourier root of unity, or 1 and 0 for Walsh-Haar."""
    if kind == WALSH_HAAR:
        return np.ones(blocks), np.zeros(blocks)
    pom = np.arange(blocks) % ring
    angle = np.zeros(blocks)
    for bit in range(stages(num) - 1):
        angle = angle + np.where(pom & (1 << bit), float((num // 4) >> bit), 0.0)
    arg = angle * (2.0 * PI / float(num))
    return libm.cos(arg), libm.sin(arg)


def _pairs(num: int, blocks: int, step: int) -> tuple[Any, ...]:
    """The butterflies of a stage: each's two places, block, and its ``a0r``, ``b0r`` weights."""
    width = num // blocks
    offset = np.arange(width // 2)
    low = (np.arange(blocks)[:, None] * width + offset[None, :]).ravel()
    block = np.repeat(np.arange(blocks), width // 2)
    weighted = np.tile(offset % step == 0, blocks)
    a0r = np.where(weighted, HALF_SQRT2, 1.0)
    b0r = np.where(weighted, HALF_SQRT2, 0.0)
    return low, low + width // 2, block, a0r, b0r


def _copy_back(ws: Any, num: int) -> None:
    """A stage's results, at ``num`` and ``3 * num``, made the working parts again."""
    ws[:num] = ws[num : 2 * num]
    ws[2 * num : 3 * num] = ws[3 * num : 4 * num]


def _stage(ws: Any, num: int, blocks: int, ring: int, step: int, kind: int, fly: Any) -> None:
    """One stage: every butterfly of ``fly`` over its pair, the results copied back."""
    low, high, block, a0r, b0r = _pairs(num, blocks, step)
    wr, wi = (part[block] for part in _twiddles(num, blocks, ring, kind))
    a, b, c, d = ws[low], ws[high], ws[low + 2 * num], ws[high + 2 * num]
    ws[num + low], ws[3 * num + low], ws[num + high], ws[3 * num + high] = fly(
        a, b, c, d, (a0r, b0r, wr, wi)
    )
    _copy_back(ws, num)


def _exe_fly(a: Any, b: Any, c: Any, d: Any, weights: tuple[Any, ...]) -> tuple[Any, ...]:
    """``GeneralExe``'s butterfly: the pair's weighted sum, and its rotated difference."""
    a0r, b0r, wr, wi = weights
    return (
        a * a0r + b * b0r,
        c * a0r + d * b0r,
        a * b0r * wr - c * b0r * wi - b * a0r * wr + d * a0r * wi,
        c * b0r * wr + a * b0r * wi - d * a0r * wr - b * a0r * wi,
    )


def general_exe(ws: Any, zt_clear: int, num: int, degree: int, kind: int) -> None:
    """``GeneralExe``: the forward mixed transform of degree ``degree``."""
    if zt_clear == 0:
        ws[2 * num : 3 * num] = 0.0
    levels = stages(num)
    blocks, step, ring = num, 1, num
    for _ in range(levels - degree):
        ring //= 2
    for m in range(1, levels + 1):
        blocks //= 2
        if m > degree and kind in HAAR_MIXED:
            step *= 2
        if ring > 1:
            ring //= 2
        _stage(ws, num, blocks, ring, step, kind, _exe_fly)


def _inv_fly(a: Any, b: Any, c: Any, d: Any, weights: tuple[Any, ...]) -> tuple[Any, ...]:
    """``GeneralInv``'s butterfly: ``GeneralExe``'s undone."""
    a0r, b0r, wr, wi = weights
    return (
        a * a0r + b * wr * b0r + d * wi * b0r,
        c * a0r + d * wr * b0r - b * wi * b0r,
        a * b0r - b * wr * a0r - d * wi * a0r,
        c * b0r - d * wr * a0r + b * wi * a0r,
    )


def general_inv(ws: Any, num: int, degree: int, kind: int) -> None:
    """``GeneralInv``: the inverse mixed transform of degree ``degree``."""
    levels = stages(num)
    step = int(math.pow(2.0, levels - degree)) if kind in HAAR_MIXED and levels > degree else 1
    ring, blocks = 1, 1
    for m in range(1, levels + 1):
        blocks = 1 if m == 1 else blocks * 2
        if m > levels - degree + 1:
            ring *= 2
        _stage(ws, num, blocks, ring, step, kind, _inv_fly)
        if m <= levels - degree and kind in HAAR_MIXED:
            step //= 2
