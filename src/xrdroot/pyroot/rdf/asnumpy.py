"""What PyROOT's ``AsNumpy`` gives: NumPy arrays that know their result, or a lazy result.

ROOT hands back each column as its own ``ndarray`` - NumPy's, subclassed - carrying
``result_ptr``, the result whose memory the array is a view of, so that memory
lives as long as the array or any slice of it does; ``result_ptr.GetValue()`` is
that memory, written through by the array. ``AsNumpy(lazy=True)`` books the read
and hands back an ``AsNumpyResult`` whose ``GetValue`` runs it, once; ROOT's
``_clone_asnumpyresult`` books the same read again, as a fresh result.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["AsNumpyResult", "clone", "column", "ndarray"]


class ndarray(np.ndarray):  # type: ignore[misc]
    """A column ``AsNumpy`` read: NumPy's array, with ``result_ptr`` - kept by its slices."""

    result_ptr: Any = None

    def __array_finalize__(self, obj: Any) -> None:
        self.result_ptr = getattr(obj, "result_ptr", None)


class _Read:
    """A column's ``result_ptr``: ``GetValue`` is the memory the column is a view of."""

    def __init__(self, values: Any) -> None:
        self._values = values

    def GetValue(self) -> Any:
        return self._values


def column(values: Any) -> ndarray:
    """``values`` as the column ``AsNumpy`` hands back: a view, knowing what it views."""
    made: ndarray = values.view(ndarray)
    made.result_ptr = _Read(values)
    return made


class AsNumpyResult:
    """``AsNumpy(lazy=True)``: the columns as a dict of arrays, read when first asked for.

    ``book`` books the read on the frame and gives back the booked result -
    what distributed ``RDataFrame`` runs a range of, and adds up - and
    ``present`` turns the dict of columns it gives into ``AsNumpy``'s own.
    """

    def __init__(
        self, book: Callable[[], Any], present: Callable[[Any], dict[str, Any]]
    ) -> None:
        self._book = book
        self._present = present
        self._read = book()
        self._value: dict[str, Any] | None = None

    def GetValue(self) -> dict[str, Any]:
        if self._value is None:
            self._value = self._present(self._read.GetValue())
        return self._value


def clone(result: AsNumpyResult) -> AsNumpyResult:
    """``_clone_asnumpyresult``: the same columns booked again - over the frame's range now."""
    return AsNumpyResult(result._book, result._present)
