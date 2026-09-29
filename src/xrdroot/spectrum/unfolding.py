"""``TSpectrum::Unfolding``: a spectrum unfolded through a response matrix by Gold's method.

Where deconvolution has one response shifted along the spectrum, unfolding
has a column of the matrix for each channel of the answer: ``y = A x``, ``A``
``sizex`` by ``sizey``. ROOT normalises each column to unit area, squares the
normal equations twice - solving ``(A'A)'(A'A) x = (A'A)'A'y`` - and runs
Gold's multiplicative iteration on that, boosted between repetitions.
"""

from __future__ import annotations

import numpy as np

from ..random import libm
from .clipping import Array
from .sums import ordered_sum

__all__ = ["unfold"]


def _refusal(sizex: int, sizey: int, iterations: int) -> str | None:
    """ROOT's message for arguments it will not unfold with, or ``None``."""
    if sizex <= 0 or sizey <= 0:
        return "Wrong Parameters"
    if sizex < sizey:
        return "Sizex must be greater than sizey)"
    if iterations <= 0:
        return "Number of iterations must be positive"
    return None


def normalised(matrix: Array) -> Array | str:
    """Each row of ``matrix`` (a column of ``A``) over its area - refused at an all-zero one."""
    rows = []
    for row in matrix:
        if not np.any(row != 0):
            return "ZERO COLUMN IN RESPONSE MATRIX"
        rows.append(row / ordered_sum(row))
    return np.array(rows)


def products(left: Array, right: Array) -> Array:
    """``sum_k left[i][k] * right[j][k]`` for every ``i``, ``j``, over ``k`` in order."""
    return np.array([ordered_sum(row[None, :] * right, axis=1) for row in left])


def unfold(
    source: Array, matrix: Array, iterations: int, repetitions: int, boost: float
) -> Array | str:
    """``source`` unfolded through ``matrix`` (``matrix[j]`` the response of channel ``j``).

    The answer has ``source``'s length, zero past ``sizey``; a refusal is ROOT's message.
    """
    sizey, sizex = matrix.shape
    refused = _refusal(sizex, sizey, iterations)
    if refused is not None:
        return refused
    response = normalised(matrix)
    if isinstance(response, str):
        return response
    square = products(response, response)
    projected = ordered_sum(response * source[None, :sizex], axis=1)
    fourth = products(square, square)
    target = ordered_sum(square * projected[None, :], axis=1)
    x = np.ones(sizey)
    for repetition in range(repetitions):
        if repetition:
            x = np.asarray(libm.power(x, boost), dtype=np.float64)
        for _ in range(iterations):
            denominator = ordered_sum(fourth * x[None, :], axis=1)
            with np.errstate(all="ignore"):
                safe = np.where(denominator != 0, denominator, 1.0)
                ratio = np.where(denominator != 0, target / safe, 0.0)
            x = ratio * x
    out = np.zeros(sizex)
    out[:sizey] = x
    return out
