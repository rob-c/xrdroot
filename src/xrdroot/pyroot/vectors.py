"""``TVectorD`` and ``TArrayD``: a vector of doubles with its own bounds, and a plain array.

A ``TVectorD`` counts from its lower bound - 0 unless it is made with
``TVectorD(lwb, upb)`` - so ``v[i]``, ``v(i)`` and ``GetUpb`` are all by
that count; its values are a NumPy array, and it is one wherever NumPy
wants one, so ``TGraph(x, y)`` takes two of them. A ``TArrayD`` is ROOT's
array of doubles, read with ``At`` and ``GetAt`` or by index.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from .core.objects import TObject

__all__ = ["TArrayD", "TArrayF", "TArrayI", "TArrayL", "TArrayS", "TArrayC", "TVectorD"]


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
        """The values, as NumPy asks for them: a copy when it asks for one."""
        made = np.asarray(self._v, dtype=dtype)
        return made.copy() if copy else made

    def Sum(self) -> float:
        return float(self._v.sum())

    def Use(self, *args: Any) -> TVectorD:
        """``Use(n, data)`` or ``Use(lwb, upb, data)``: the vector made of ``data`` itself, which
        it reads and writes, not a copy of it."""
        data = np.asarray(args[-1])
        lwb, size = (0, int(args[0])) if len(args) == 2 else (int(args[0]), int(args[1]) - int(
            args[0]) + 1)  # fmt: skip
        self._lwb, self._v = lwb, data[:size]
        return self

    def _assign(self, value: Any) -> None:
        """``v = other``: the other's elements copied in; ``v = 0.`` every element set."""
        if np.ndim(value):
            self._v = np.array(np.asarray(value), dtype=np.float64)
        else:
            self._v[:] = float(value)

    def ResizeTo(self, n: Any) -> TVectorD:
        grown = np.zeros(int(n))
        kept = min(int(n), len(self._v))
        grown[:kept] = self._v[:kept]
        self._v = grown
        return self

    def Zero(self) -> TVectorD:
        self._v[:] = 0.0
        return self

    def Abs(self) -> TVectorD:
        self._v = np.abs(self._v)
        return self

    def Max(self) -> float:
        return float(self._v.max())

    def Min(self) -> float:
        return float(self._v.min())

    def Norm1(self) -> float:
        return float(np.abs(self._v).sum())

    def NormInf(self) -> float:
        return float(np.abs(self._v).max())

    def __imul__(self, other: Any) -> TVectorD:
        """``v *= a`` scales; ``v *= M`` is ``v = M v``, as ROOT's ``operator*=`` of a matrix."""
        self._v = self._v * float(other) if np.ndim(other) == 0 else np.asarray(other) @ self._v
        return self

    def __mul__(self, other: Any) -> Any:
        """``v * a`` scaled; ``v1 * v2`` their dot product, as ROOT's ``operator*``."""
        if np.ndim(other) == 0:
            return TVectorD(self._v * float(other))
        return float(np.dot(self._v, np.asarray(other)))

    def __rmul__(self, other: Any) -> TVectorD:
        return TVectorD(float(other) * self._v)

    def __add__(self, other: Any) -> TVectorD:
        return TVectorD(self._v + np.asarray(other))

    def __sub__(self, other: Any) -> TVectorD:
        return TVectorD(self._v - np.asarray(other))

    def __neg__(self) -> TVectorD:
        return TVectorD(-self._v)

    def Norm2Sqr(self) -> float:
        return float(np.dot(self._v, self._v))

    def Print(self, option: str = "") -> None:
        """``TVectorT::Print``: a one-column sheet, every value by ``%g``."""
        print(f"\nVector ({len(self._v)}) {option} is as follows\n\n     |        1  |")
        print("------------------")
        for i, value in enumerate(self._v.tolist()):
            print(f"{i + self._lwb:4d} |{value:g} ")
        print()


class _TArray(TObject):
    """``TArray``: a fixed number of values of one type, read and written by index."""

    #: The values' type, as NumPy names it.
    DTYPE = np.dtype(np.float64)

    def __init__(self, n: Any = 0, values: Any = None) -> None:
        super().__init__()
        size = len(n) if np.ndim(n) > 0 else int(n)
        source = n if np.ndim(n) > 0 else values
        self._v = (np.zeros(size, dtype=self.DTYPE) if source is None else
                   np.array(np.asarray(source)[:size], dtype=self.DTYPE))  # fmt: skip

    def At(self, i: Any) -> Any:
        return self._v[int(i)].item()

    GetAt = At
    __getitem__ = At

    def __setitem__(self, i: Any, value: Any) -> None:
        self._v[int(i)] = value

    def SetAt(self, value: Any, i: Any) -> None:
        self._v[int(i)] = value

    AddAt = SetAt

    def Set(self, n: Any, values: Any = None) -> None:
        """``Set(n[, values])``: ``n`` values - those given, or the old ones kept and zeros."""
        if values is not None:
            self._v = np.array(np.asarray(values)[:int(n)], dtype=self.DTYPE)
            return
        grown = np.zeros(int(n), dtype=self.DTYPE)
        kept = min(int(n), len(self._v))
        grown[:kept] = self._v[:kept]
        self._v = grown

    def Reset(self, value: Any = 0) -> None:
        self._v[:] = value

    def GetSum(self) -> float:
        return float(self._v.sum())

    def GetSize(self) -> int:
        return len(self._v)

    def __len__(self) -> int:
        return len(self._v)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._v.tolist())

    def GetArray(self) -> np.ndarray[Any, Any]:
        return self._v

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray[Any, Any]:
        """The values, as NumPy asks for them: a copy when it asks for one."""
        made = np.asarray(self._v, dtype=dtype)
        return made.copy() if copy else made


class TArrayD(_TArray):
    """``TArrayD``: an array of doubles."""


class TArrayF(_TArray):
    """``TArrayF``: an array of floats."""

    DTYPE = np.dtype(np.float32)


class TArrayI(_TArray):
    """``TArrayI``: an array of ints."""

    DTYPE = np.dtype(np.int32)


class TArrayL(_TArray):
    """``TArrayL``: an array of longs."""

    DTYPE = np.dtype(np.int64)


class TArrayS(_TArray):
    """``TArrayS``: an array of shorts."""

    DTYPE = np.dtype(np.int16)


class TArrayC(_TArray):
    """``TArrayC``: an array of chars, as small numbers."""

    DTYPE = np.dtype(np.int8)
