"""Sums of products added term by term, as ROOT's inner loops add them, for every output at once.

A deconvolution's inner loop is ``lda = lda + ldb * ldc`` over a kernel, from
zero, for each channel in turn. Stacking each channel's products as a column
and letting NumPy accumulate down the columns adds them in the same order
from the same first term - ``np.add.accumulate`` is sequential, where
``np.sum`` pairs terms up - so each channel's sum is ROOT's to the last bit,
at array speed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .clipping import Array

__all__ = ["gather", "ordered_sum", "padded"]


def padded(values: Array) -> Array:
    """``values`` with a zero after the last, which an index of ``len(values)`` picks."""
    return np.concatenate([values, [0.0]])


def gather(size: int, rows: Any, columns: Any) -> np.ndarray[Any, np.dtype[np.intp]]:
    """``rows + columns`` as indices into an array of ``size``, those outside it at ``size``.

    An index at ``size`` picks the zero :func:`padded` adds: a term the C++
    leaves out, which adding 0.0 in its place does to the last bit.
    """
    found = np.add.outer(np.asarray(rows), np.asarray(columns))
    return np.where((found >= 0) & (found < size), found, size).astype(np.intp)


def ordered_sum(products: Array, axis: int = 0) -> Array:
    """The sum along ``axis`` of ``products``, added in order from the first."""
    if products.shape[axis] == 0:
        shape = list(products.shape)
        del shape[axis]
        return np.zeros(shape)
    return np.asarray(np.take(np.add.accumulate(products, axis=axis), -1, axis=axis))
