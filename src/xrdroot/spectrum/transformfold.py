"""The folding around the mixed transforms: mirrored blocks, corrections, and reordering.

A cosine-Walsh or sine-Haar transform of ``n`` numbers is a Walsh or Haar
transform of ``2 n``: each block of ``j`` numbers is laid beside its mirror
image (negated, for a sine), transformed, and the cosine's half-sample shift
divided out. Every mixed transform then deals its coefficients out
``k`` apart. These are those steps, shared by ``TSpectrumTransform`` and
``TSpectrum2Transform``, which do them in the same places of their working
space: each moves numbers only between places its loop in ROOT never both
reads and writes, so it is done at once.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..random import libm
from .transformtypes import PI, SQRT2, WALSH_HAAR, power2

__all__ = [
    "correct_cos", "correct_sin", "deal", "gather", "half_block", "load_mirrored",
    "undeal", "unfold_cos", "unfold_sin",
]  # fmt: skip


def half_block(degree: int) -> int:
    """``j = (Int_t) TMath::Power(2, degree) / 2``: the length of a mirrored block's half."""
    return power2(degree) // 2


def _blocks(n: int, half: int) -> tuple[Any, Any, Any]:
    """For each ``i < n``: its place in its half-block, where its block starts, and the angle."""
    index = np.arange(n)
    rest = index % half
    start = 2 * (index // half) * half
    angle = PI * rest.astype(np.float64) / float(2 * half)
    return rest, start, angle


def load_mirrored(ws: Any, values: Any, degree: int, sign: float) -> None:
    """Each half-block of ``values`` beside its mirror image, times ``sign`` for a sine's."""
    rest, start, _ = _blocks(len(values), half_block(degree))
    ws[start + rest] = values
    ws[start + 2 * half_block(degree) - 1 - rest] = values if sign > 0 else -values


def correct_cos(ws: Any, n: int, degree: int, clear: int) -> None:
    """The cosine's shift divided out of each coefficient, and ``clear`` onwards zeroed."""
    rest, start, angle = _blocks(n, half_block(degree))
    b = ws[start + rest]
    ws[:n] = np.where(rest == 0, b / SQRT2, b / libm.cos(angle))
    ws[clear : clear + n] = 0.0


def correct_sin(ws: Any, n: int, degree: int, clear: int) -> None:
    """The sine's shift divided out, each coefficient laid back into its half-block."""
    half = half_block(degree)
    rest, start, angle = _blocks(n, half)
    b = ws[half + start + rest]
    ws[half + start // 2 - rest - 1] = np.where(rest == 0, b / SQRT2, b / libm.cos(angle))
    ws[clear : clear + n] = 0.0


def _spacing(n: int, degree: int, kind: int) -> tuple[int, Any]:
    """``k`` and where each coefficient goes: ``l + i / j``, ``l`` stepping by ``k`` round ``n``."""
    k = power2(degree - 1) if kind > WALSH_HAAR else power2(degree)
    index = np.arange(n)
    return k, (index * k) % n + index // (n // k)


def deal(ws: Any, n: int, degree: int, kind: int, imag: int) -> None:
    """The coefficients dealt out ``k`` apart, the imaginary ones at ``imag`` too."""
    _, source = _spacing(n, degree, kind)
    ws[n : 2 * n] = ws[source]
    ws[n + imag : 2 * n + imag] = ws[source + imag]
    ws[:n] = ws[n : 2 * n]
    ws[imag : imag + n] = ws[n + imag : 2 * n + imag]


def undeal(ws: Any, n: int, degree: int, kind: int, imag: int) -> int:
    """The coefficients gathered back from ``k`` apart; ``k`` itself, which is needed next."""
    k, target = _spacing(n, degree, kind)
    ws[n + target] = ws[:n]
    ws[n + target + imag] = ws[imag : imag + n]
    ws[:n] = ws[n : 2 * n]
    ws[imag : imag + n] = ws[n + imag : 2 * n + imag]
    return k


def gather(ws: Any, n: int, degree: int, offset: int = 0) -> Any:
    """A cosine or sine mixed transform's ``n`` numbers, out of their mirrored blocks."""
    rest, start, _ = _blocks(n, half_block(degree))
    return ws[start + rest + offset].copy()


def _unfold(ws: Any, n: int, values: Any, at: Any, rest: Any, angle: Any) -> None:
    """``values`` put back at ``2 n + at`` with the cosine's shift, the sine part at ``6 n``."""
    real = 2 * n + at
    head = rest == 0
    ws[real] = np.where(head, values * SQRT2, values * libm.cos(angle))
    ws[real + 4 * n] = np.where(head, 0.0, -values * libm.sin(angle))


def unfold_cos(ws: Any, n: int, degree: int) -> None:
    """The inverse's unfolding of a cosine mixed transform, ready for ``GeneralInv``."""
    half = half_block(degree)
    rest, start, angle = _blocks(n, half)
    _unfold(ws, n, ws[:n].copy(), start + rest, rest, angle)
    head = rest == 0
    ws[2 * n + start[head] + half] = 0.0
    ws[6 * n + start[head] + half] = 0.0
    src, dst = 2 * n + (start + rest)[~head], 2 * n + (start + 2 * half - rest)[~head]
    ws[dst] = ws[src]
    ws[dst + 4 * n] = -ws[src + 4 * n]
    _to_front(ws, n)


def unfold_sin(ws: Any, n: int, degree: int) -> None:
    """The inverse's unfolding of a sine mixed transform, ready for ``GeneralInv``."""
    half = half_block(degree)
    rest, start, angle = _blocks(n, half)
    values = ws[half + start // 2 - rest - 1].copy()
    _unfold(ws, n, values, start + half + rest, rest, angle)
    head = rest == 0
    ws[2 * n + start[head]] = 0.0
    ws[6 * n + start[head]] = 0.0
    dst, src = 2 * n + (start + rest)[~head], 2 * n + (start + 2 * half - rest)[~head]
    ws[dst] = ws[src]
    ws[dst + 4 * n] = -ws[src + 4 * n]
    _to_front(ws, n)


def _to_front(ws: Any, n: int) -> None:
    """The unfolded ``2 n`` numbers and their sine parts moved to where ``GeneralInv`` reads."""
    ws[: 2 * n] = ws[2 * n : 4 * n]
    ws[4 * n : 6 * n] = ws[6 * n : 8 * n]
