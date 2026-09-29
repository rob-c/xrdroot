"""``TSpectrum2Transform``'s ``HaarWalsh2`` and ``FourCos2``: one transform along x, then y.

A 2-D transform here is ROOT's: the 1-D transform of every column, then of
every row, through one working vector that each line is copied into and
out of - forward x then y, inverse y then x. The working matrix keeps each
line's real part in its first ``sizey`` columns and its imaginary part in
the next ``sizey``, and the vector is the one ROOT allocates once per call,
so whatever a line leaves in it is there for the next.
"""

from __future__ import annotations

from typing import Any

from .transform import cos_forward, cos_inverse, forward, inverse, sin_forward, sin_inverse
from .transformkernels import fourier
from .transformtypes import COS, FORWARD, FOURIER, HARTLEY, INVERSE, SIN

__all__ = ["four_cos2", "haar_walsh2", "imaginary_offset"]


def _haar_walsh_line(wv: Any, line: Any, n: int, direction: int, kind: int) -> Any:
    """One line's Haar or Walsh transform, forward or inverse."""
    if direction == FORWARD:
        forward(wv, line, n, kind, 0)
        return wv[:n]
    wv[:n] = line
    return inverse(wv, n, kind, 0)


def haar_walsh2(matrix: Any, wv: Any, direction: int, kind: int) -> None:
    """``HaarWalsh2``: the Haar or Walsh transform of every column and every row."""
    nx, ny = matrix.shape[0], matrix.shape[1] // 2
    for axis in (0, 1) if direction == FORWARD else (1, 0):
        n = nx if axis == 0 else ny
        for index in range(ny if axis == 0 else nx):
            line = (slice(None), index) if axis == 0 else (index, slice(0, ny))
            matrix[line] = _haar_walsh_line(wv, matrix[line], n, direction, kind)


def imaginary_offset(kind: int, n: int) -> int:
    """Where a line's imaginary part sits in the vector: after it for Fourier's, else ``2 n``."""
    return n if kind == FOURIER else 2 * n


def _fourier_line(wv: Any, n: int, direction: int, kind: int, zt_clear: int) -> None:
    """One line's cosine, sine, Fourier or Hartley transform, the line already in ``wv``."""
    if kind == COS:
        (cos_forward(wv, wv[:n].copy(), n) if direction == FORWARD else cos_inverse(wv, n))
    elif kind == SIN:
        (sin_forward(wv, wv[:n].copy(), n) if direction == FORWARD else sin_inverse(wv, n))
    elif direction == FORWARD:
        fourier(wv, n, int(kind == HARTLEY), FORWARD, zt_clear if kind == FOURIER else 0)
    else:
        fourier(wv, n, int(kind == HARTLEY), INVERSE, 1)


def _pass(matrix: Any, wv: Any, axis: int, direction: int, kind: int, imag_in: bool) -> None:
    """Every line along ``axis`` through ``_fourier_line``, its imaginary part in and out."""
    nx, ny = matrix.shape[0], matrix.shape[1] // 2
    n = nx if axis == 0 else ny
    at = imaginary_offset(kind, n)
    for index in range(ny if axis == 0 else nx):
        real = (slice(None), index) if axis == 0 else (index, slice(0, ny))
        imag = (slice(None), index + ny) if axis == 0 else (index, slice(ny, 2 * ny))
        wv[:n] = matrix[real]
        if imag_in:
            wv[at : at + n] = matrix[imag]
        _fourier_line(wv, n, direction, kind, int(imag_in))
        matrix[real] = wv[:n]
        if not (axis == 0 and direction == INVERSE):
            matrix[imag] = wv[at : at + n]


def four_cos2(matrix: Any, wv: Any, direction: int, kind: int) -> None:
    """``FourCos2``: the cosine, sine, Fourier or Hartley transform of every column and row."""
    if direction == FORWARD:
        _pass(matrix, wv, 0, FORWARD, kind, False)
        _pass(matrix, wv, 1, FORWARD, kind, True)
    else:
        _pass(matrix, wv, 1, INVERSE, kind, True)
        _pass(matrix, wv, 0, INVERSE, kind, True)
