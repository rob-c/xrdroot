"""``TMVA::Experimental::RTensor``: an n-dimensional array, row or column major.

An ``RTensor`` is a NumPy array underneath - made of zeros for a shape, or
over the macro's own ``float data[]`` or ``std::vector`` without copying -
with TMVA's methods: ``(i, j)`` to read and write an element,
``GetShape``, ``Reshape``, ``Squeeze``, ``Slice``, and the ``{ { 1, 2 } { 3, 4 } }``
printing of ``operator<<``. ``AsTensor<T>(df, columns)`` fills one from an
``RDataFrame``, an event per row.

From Python, as PyROOT has it, ``RTensor("float")`` is the class ``RTensor<float>``,
``AsRTensor(array)`` - or ``RTensor<T>(array)`` - adopts a NumPy array's memory in
its layout, and ``numpy.asarray(tensor)`` is a view of the tensor's, which
``[i, j]`` reads and ``[:, j]`` slices.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np

from .tools import CxxVector

__all__ = ["AsRTensor", "AsTensor", "MemoryLayout", "RTensor", "tensor_type"]

#: The element types a tensor can be, by their C++ names.
DTYPES = {
    "float": np.float32,
    "double": np.float64,
    **dict.fromkeys(("int", "int32_t"), np.int32),
    **dict.fromkeys(("long", "long long", "Long64_t", "int64_t"), np.int64),
    **dict.fromkeys(("unsigned int", "uint32_t"), np.uint32),
    **dict.fromkeys(
        ("unsigned long", "unsigned long long", "ULong64_t", "uint64_t"), np.uint64),
    "bool": np.bool_,
}


class MemoryLayout(IntEnum):
    """``MemoryLayout``: C's order, or Fortran's."""

    RowMajor = 1
    ColumnMajor = 2


def _array(data: Any, dtype: Any) -> Any:
    """The macro's buffer as a NumPy array that shares its memory, if it can."""
    for name in ("_array", "_values", "data"):
        inner = getattr(data, name, None)
        if isinstance(inner, np.ndarray):
            return inner
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(list(data), dtype=dtype)


class _Template(type):
    """``RTensor("float")``: PyROOT's call of a template with a type - ``RTensor<float>``."""

    def __call__(cls, *args: Any) -> Any:
        if cls is RTensor and len(args) == 1 and isinstance(args[0], str):
            return tensor_type(args[0])
        return super().__call__(*args)


class RTensor(metaclass=_Template):
    """``RTensor<T>(shape)``, ``(data, shape)`` or ``(vector, shape)``, and ``RTensor[T]``.

    A NumPy array of two dimensions or more alone, ``RTensor<T>(array)``, is
    adopted, as ``AsRTensor`` adopts it.
    """

    dtype: Any = np.float32

    def __class_getitem__(cls, kind: Any) -> type[RTensor]:
        return tensor_type(kind)

    def __new__(cls, *args: Any) -> Any:
        if len(args) == 1 and isinstance(args[0], np.ndarray) and args[0].ndim > 1:
            return AsRTensor(args[0].astype(cls.dtype, copy=False))
        return super().__new__(cls)

    def __init__(self, *args: Any) -> None:
        if len(args) == 1 and isinstance(args[0], np.ndarray) and args[0].ndim > 1:
            return  # adopted by ``__new__``
        layout, args = _layout(args)
        order = "C" if layout == MemoryLayout.RowMajor else "F"
        if len(args) == 1:
            shape = tuple(int(n) for n in args[0])
            self.array = np.zeros(shape, dtype=self.dtype, order=order)
        else:
            flat = _array(args[0], self.dtype)
            shape = tuple(int(n) for n in args[1])
            self.array = flat.reshape(shape, order=order)
        self.layout = layout

    @classmethod
    def wrap(cls, array: Any, layout: MemoryLayout = MemoryLayout.RowMajor) -> RTensor:
        made: RTensor = object.__new__(cls)
        made.array, made.layout = array, layout
        return made

    def __call__(self, *index: Any) -> Any:
        return self.array[tuple(int(i) for i in index)].item()

    def __setcall__(self, *args: Any) -> None:
        *index, value = args
        self.array[tuple(int(i) for i in index)] = value

    def __getitem__(self, index: Any) -> Any:
        """``t[i]`` of the elements in memory order, ``t[i, j]`` one element, and a slice -
        ``t[:, 0]`` - a tensor, which drops the dimensions left one long, as ``Slice`` does."""
        if not isinstance(index, tuple):
            return self.array.reshape(-1)[int(index)].item()
        if not any(isinstance(i, slice) for i in index):
            return self.array[tuple(int(i) for i in index)].item()
        kept = tuple(i if isinstance(i, slice) else _one(int(i), n)
                     for i, n in zip(index, self.array.shape, strict=False))
        return self.wrap(self.array[kept], self.layout).Squeeze()

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        """NumPy's view of the tensor's memory - a copy only when asked for one."""
        view = self.array.view() if dtype is None else self.array.astype(dtype)
        return view.copy() if copy else view

    def GetStrides(self) -> CxxVector:
        """How many elements apart each index's neighbours are, as TMVA counts strides."""
        return CxxVector(int(n) // self.array.itemsize for n in self.array.strides)

    def Transpose(self) -> RTensor:
        flipped = MemoryLayout.ColumnMajor if self.layout == MemoryLayout.RowMajor else (
            MemoryLayout.RowMajor)
        return self.wrap(self.array.T, flipped)

    def ExpandDims(self, axis: int) -> RTensor:
        return self.wrap(np.expand_dims(self.array, int(axis)), self.layout)

    def GetShape(self) -> CxxVector:
        return CxxVector(int(n) for n in self.array.shape)

    def GetSize(self) -> int:
        return int(self.array.size)

    def GetData(self) -> Any:
        return self.array

    def GetMemoryLayout(self) -> MemoryLayout:
        return self.layout

    def Reshape(self, shape: Any) -> RTensor:
        order = "C" if self.layout == MemoryLayout.RowMajor else "F"
        return self.wrap(self.array.reshape(tuple(int(n) for n in shape), order=order), self.layout)

    def Squeeze(self) -> RTensor:
        squeezed = self.array.squeeze()
        return self.wrap(squeezed.reshape(1) if squeezed.ndim == 0 else squeezed, self.layout)

    def Slice(self, ranges: Any) -> RTensor:
        index = tuple(slice(int(low), int(high)) for low, high in ranges)
        # TMVA's slice drops the dimensions it leaves one long, as ``Squeeze`` does.
        return self.wrap(self.array[index], self.layout).Squeeze()

    def __str__(self) -> str:
        return _printed(self.array)

    def __repr__(self) -> str:
        return f"<RTensor {self.array.shape} {_printed(self.array)}>"


def _layout(args: tuple[Any, ...]) -> tuple[MemoryLayout, tuple[Any, ...]]:
    """The layout a constructor's last argument names - row major if none does - and the
    arguments before it."""
    last = args[-1] if len(args) > 1 else None
    if isinstance(last, (MemoryLayout, int)) and not isinstance(last, bool):
        return MemoryLayout(int(last)), args[:-1]
    return MemoryLayout.RowMajor, args


def _one(index: int, size: int) -> slice:
    """Index ``index`` of ``size`` - negative from the end - as a slice one long."""
    start = index + size if index < 0 else index
    return slice(start, start + 1)


def tensor_type(kind: Any) -> type[RTensor]:
    """``RTensor<T>``: the tensor class of elements of C++ type ``kind`` - ``float`` if unknown."""
    name = str(kind).replace("Float_t", "float").replace("Double_t", "double")
    return type(f"RTensor<{name}>", (RTensor,), {"dtype": DTYPES.get(name, np.float32)})


def AsRTensor(array: Any) -> RTensor:
    """``TMVA::Experimental::AsRTensor``: a tensor over a NumPy array's own memory, in its
    layout - column major when it is in Fortran's order."""
    found = np.asarray(array)
    layout = MemoryLayout.RowMajor if found.flags.c_contiguous else MemoryLayout.ColumnMajor
    return tensor_type(_DTYPE_NAMES.get(found.dtype.type, "float")).wrap(found, layout)


def _number(value: Any) -> str:
    """One element as ``std::ostream`` writes it, six significant figures."""
    return format(float(value), "g") if not isinstance(value, (bool, np.bool_)) else str(int(value))


def _printed(array: Any) -> str:
    """``operator<<``: ``{ 1, 2 }`` for one dimension, ``{ { 1, 2 } { 3, 4 } }`` for two."""
    if array.ndim <= 1:
        return "{ " + ", ".join(_number(v) for v in np.ravel(array)) + " }"
    return "{ " + " ".join(_printed(row) for row in array) + " }"


class _AsTensor:
    """``AsTensor<T>(dataframe, columns = {}, layout = RowMajor)``: an event per row."""

    def __init__(self, dtype: Any = np.float32) -> None:
        self.dtype = dtype

    def __getitem__(self, kind: Any) -> _AsTensor:
        return _AsTensor(tensor_type(kind).dtype)

    def __call__(
        self, frame: Any, columns: Any = (), layout: Any = MemoryLayout.RowMajor
    ) -> RTensor:
        names = [str(c) for c in columns] or [str(c) for c in frame.GetColumnNames()]
        found = frame.AsNumpy(names)
        table = np.column_stack([np.asarray(found[name], dtype=self.dtype) for name in names])
        order = "C" if int(layout) == MemoryLayout.RowMajor else "F"
        made = tensor_type(_DTYPE_NAMES.get(self.dtype, "float"))
        return made.wrap(np.asarray(table, order=order), MemoryLayout(int(layout)))


#: Each element type's C++ name.
_DTYPE_NAMES = {dtype: name for name, dtype in reversed(DTYPES.items())}
#: ``TMVA::Experimental::AsTensor``, templated on the element type.
AsTensor = _AsTensor()
