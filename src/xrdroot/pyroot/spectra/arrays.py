"""The ``Double_t *`` and ``Double_t **`` a spectrum method reads and writes back into.

``s->Background(source, 256, ...)`` reads 256 numbers from ``source`` and
leaves the background in their place. A translated macro's ``Double_t
source[256]`` is a NumPy array, its ``Double_t **`` a list of rows (or a
vector's ``data()``), and a Python script may hand over any of those, an
``array.array`` or a plain list: each is read as C reads it, from its start,
and filled in place.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["matrix_in", "matrix_out", "vector_in", "vector_out"]


def vector_in(source: Any, size: int) -> np.ndarray[Any, np.dtype[np.float64]]:
    """The first ``size`` numbers of ``source``, as ``Double_t`` - a copy to work on."""
    return np.array(np.asarray(source, dtype=np.float64).ravel()[: int(size)], dtype=np.float64)


def vector_out(target: Any, values: Any) -> None:
    """Write ``values`` into ``target`` from its start, as the C++ writes through the pointer."""
    if isinstance(target, np.ndarray):
        target.flat[: len(values)] = values
        return
    for index, value in enumerate(np.asarray(values, dtype=np.float64).tolist()):
        target[index] = value


def matrix_in(source: Any, sizex: int, sizey: int) -> np.ndarray[Any, np.dtype[np.float64]]:
    """``source[i][j]`` for ``i < sizex``, ``j < sizey``, as a ``(sizex, sizey)`` array."""
    rows = [vector_in(source[i], sizey) for i in range(int(sizex))]
    return np.array(rows, dtype=np.float64).reshape(int(sizex), int(sizey))


def matrix_out(target: Any, values: Any) -> None:
    """Write each row of ``values`` into the row of ``target`` it came from."""
    for i, row in enumerate(np.asarray(values, dtype=np.float64)):
        vector_out(target[i], row)
