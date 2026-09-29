"""``TSpectrum2Transform::General2``: a mixed transform of every column, then every row.

The steps are the 1-D mixed transform's, but in ``General2``'s order, which
is not ``TSpectrumTransform``'s: a Fourier-, Walsh-Haar line runs its
butterflies before bit-reversing its blocks, and undoes them after; and a
cosine or sine line keeps its imaginary part at ``4 n``, where the
butterflies of its doubled length look for it.
"""

from __future__ import annotations

from typing import Any

from .transformfold import (
    correct_cos,
    correct_sin,
    deal,
    gather,
    load_mirrored,
    undeal,
    unfold_cos,
    unfold_sin,
)
from .transformmixed import bit_reverse_haar, general_exe, general_inv
from .transformtypes import COS_MIXED, FORWARD, PLAIN_MIXED, power2

__all__ = ["general2"]


def _imag(kind: int, n: int) -> int:
    """Where a mixed line's imaginary part is: ``2 n``, or ``4 n`` for a cosine's or sine's."""
    return 2 * n if kind in PLAIN_MIXED else 4 * n


def _forward_line(wv: Any, real: Any, imag: Any, n: int, kind: int, degree: int) -> None:
    """One line's forward mixed transform; ``imag`` is ``None`` along x, where there is none."""
    zt_clear = int(imag is not None)
    if kind in PLAIN_MIXED:
        wv[:n] = real
        if imag is not None:
            wv[2 * n : 3 * n] = imag
        general_exe(wv, zt_clear, n, degree, kind)
        k = power2(degree)
        for i in range(n // k):
            bit_reverse_haar(wv, n, k, i * k)
    else:
        sign = 1.0 if kind in COS_MIXED else -1.0
        load_mirrored(wv, real, degree, sign)
        if imag is not None:
            load_mirrored(wv[4 * n :], imag, degree, sign)
        m = power2(degree)
        for i in range(2 * n // m):
            bit_reverse_haar(wv, 2 * n, m, i * m)
        general_exe(wv, zt_clear, 2 * n, degree, kind)
        (correct_cos if kind in COS_MIXED else correct_sin)(wv, n, degree, 4 * n)
    deal(wv, n, degree, kind, _imag(kind, n))


def _inverse_line(wv: Any, real: Any, imag: Any, n: int, kind: int, degree: int) -> Any:
    """One line's inverse mixed transform: its real part, and its imaginary part."""
    at = _imag(kind, n)
    wv[:n], wv[at : at + n] = real, imag
    k = undeal(wv, n, degree, kind, at)
    if kind in PLAIN_MIXED:
        for i in range(n // k):
            bit_reverse_haar(wv, n, k, i * k)
        general_inv(wv, n, degree, kind)
        return wv[:n].copy(), wv[2 * n : 3 * n].copy()
    (unfold_cos if kind in COS_MIXED else unfold_sin)(wv, n, degree)
    general_inv(wv, 2 * n, degree, kind)
    m = power2(degree)
    for i in range(2 * n // m):
        bit_reverse_haar(wv, 2 * n, m, i * m)
    return gather(wv, n, degree), gather(wv, n, degree, 4 * n)


def general2(matrix: Any, wv: Any, direction: int, kind: int, degree: int) -> None:
    """``General2``: the mixed transform of every column and every row, or its inverse."""
    nx, ny = matrix.shape[0], matrix.shape[1] // 2
    if direction == FORWARD:
        for j in range(ny):
            _forward_line(wv, matrix[:, j], None, nx, kind, degree)
            at = _imag(kind, nx)
            matrix[:, j], matrix[:, j + ny] = wv[:nx], wv[at : at + nx]
        for i in range(nx):
            _forward_line(wv, matrix[i, :ny], matrix[i, ny:], ny, kind, degree)
            at = _imag(kind, ny)
            matrix[i, :ny], matrix[i, ny:] = wv[:ny], wv[at : at + ny]
        return
    for i in range(nx):
        matrix[i, :ny], matrix[i, ny:] = _inverse_line(
            wv, matrix[i, :ny], matrix[i, ny:], ny, kind, degree
        )
    for j in range(ny):
        matrix[:, j] = _inverse_line(wv, matrix[:, j], matrix[:, j + ny], nx, kind, degree)[0]
