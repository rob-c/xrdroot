"""``TVectorD`` and ``TArrayD``: a vector of doubles with its own bounds, and a plain array.

A ``TVectorD`` counts from its lower bound - 0 unless it is made with
``TVectorD(lwb, upb)`` - so ``v[i]``, ``v(i)`` and ``GetUpb`` are all by
that count; its values are a NumPy array, and it is one wherever NumPy
wants one, so ``TGraph(x, y)`` takes two of them. A ``TArrayD`` is ROOT's
array of doubles, read with ``At`` and ``GetAt`` or by index.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.objects import TObject

__all__ = ["TArrayD", "TVectorD"]


def _values(args: tuple[Any, ...]) -> tuple[int, np.ndarray[Any, Any]]:
    """A vector's lower bound and values from ``TVectorD``'s constructors' arguments."""
    if not args:
        return 0, np.zeros(0)
    first = args[0]
    if isinstance(first, TVectorD):
        return first._lwb, first._v.copy()
    if np.ndim(first) > 0:
        return 0, np.array(first, dtype=np.float64)
    if len(args) == 1:
        return 0, np.zeros(int(first))
    if np.ndim(args[1]) > 0:  # TVectorD(n, values)
        return 0, np.array(np.asarray(args[1], dtype=np.float64)[:int(first)])
    lwb, upb = int(first), int(args[1])  # TVectorD(lwb, upb[, values])
    given = np.asarray(args[2], dtype=np.float64)[:upb - lwb + 1] if len(args) > 2 else None
    return lwb, np.zeros(upb - lwb + 1) if given is None else np.array(given)


class TVectorD(TObject):
    """``TVectorD``: doubles counted from a lower bound."""

    def __init__(self, *args: Any) -> None:
        super().__init__()
        self._lwb, self._v = _values(args)

    def GetNoElements(self) -> int:
        return len(self._v)

    GetNrows = GetNoElements

    def GetLwb(self) -> int:
        return self._lwb

    def GetUpb(self) -> int:
        return self._lwb + len(self._v) - 1

    def GetMatrixArray(self) -> np.ndarray[Any, Any]:
        return self._v

    def __len__(self) -> int:
        return len(self._v)

    def __getitem__(self, i: int) -> float:
        return float(self._v[int(i) - self._lwb])

    def __setitem__(self, i: int, value: float) -> None:
        self._v[int(i) - self._lwb] = float(value)

    __call__ = __getitem__

    def __setcall__(self, i: int, value: float) -> None:
        """``v(i) = value``."""
        self[i] = value

    def __iter__(self) -> Any:
        return iter(self._v.tolist())

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray[Any, Any]:
        return np.asarray(self._v, dtype=dtype)

    def Sum(self) -> float:
        return float(self._v.sum())

    def Norm2Sqr(self) -> float:
        return float(np.dot(self._v, self._v))

    def Print(self, option: str = "") -> None:
        """``TVectorT::Print``: a one-column sheet, every value by ``%g``."""
        print(f"\nVector ({len(self._v)}) {option} is as follows\n\n     |        1  |")
        print("------------------")
        for i, value in enumerate(self._v.tolist()):
            print(f"{i + self._lwb:4d} |{value:g} ")
        print()


class TArrayD(TObject):
    """``TArrayD``: an array of doubles."""

    def __init__(self, n: Any = 0, values: Any = None) -> None:
        super().__init__()
        size = len(n) if np.ndim(n) > 0 else int(n)
        source = n if np.ndim(n) > 0 else values
        self._v = (np.zeros(size) if source is None else
                   np.array(np.asarray(source, dtype=np.float64)[:size]))  # fmt: skip

    def At(self, i: int) -> float:
        return float(self._v[int(i)])

    GetAt = At
    __getitem__ = At

    def SetAt(self, value: float, i: int) -> None:
        self._v[int(i)] = float(value)

    def GetSize(self) -> int:
        return len(self._v)

    def __len__(self) -> int:
        return len(self._v)

    def GetArray(self) -> np.ndarray[Any, Any]:
        return self._v

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray[Any, Any]:
        return np.asarray(self._v, dtype=dtype)
