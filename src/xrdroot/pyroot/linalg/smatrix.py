"""``ROOT::Math``'s fixed-size vectors and matrices: ``SVector<T, D>``, ``SMatrix<T, D1, D2>``.

A C++ macro writes the shape into the type, ``SVector<float, 3>``, which a Python class cannot:
the translator turns that into a subscript, ``SVector['float', 3]``, and subscripting a class
here makes - once, cached - a subclass carrying the element type and the dimensions. What is
stored is one flat list of numbers, so the arithmetic is plain Python arithmetic and no NumPy
dtype decides what ``float`` means for a matrix ROOT would have kept in single precision.

``SMatrix<float, 3>`` with one dimension is ROOT's symmetric matrix, and an explicit
``MatRepSym``/``MatRepStd`` fourth template argument says which representation, exactly as
ROOT defaults it.
"""

from __future__ import annotations

import typing as _typing
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["MatRepStd", "MatRepSym", "SMatrix", "SVector", "Dot"]


def _text(value: Any) -> str:
    """One number as a C++ stream writes it at its default precision, which is what ROOT's
    ``operator<<`` for these classes prints."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):g}"


def _is_number(value: Any) -> bool:
    """Whether ``value`` is a number rather than something a vector can be filled from."""
    return isinstance(value, (int, float, np.integer, np.floating))


class _Shape:
    """``MatRepStd<T, D1, D2>`` and ``MatRepSym<T, D>``: only the shape they name is kept."""

    _kind = "std"

    def __init__(self, rows: int, cols: int) -> None:
        self.rows, self.cols = rows, cols

    @classmethod
    def __class_getitem__(cls, spec: Any) -> _Shape:
        _, *dims = spec if isinstance(spec, tuple) else (None, spec)
        return cls(int(dims[0]), int(dims[-1]))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.rows}, {self.cols})"


class MatRepStd(_Shape):
    """``MatRepStd<T, D1, D2>``: a matrix that stores every element."""


class MatRepSym(_Shape):
    """``MatRepSym<T, D>``: a matrix that stores one triangle and mirrors the other."""

    _kind = "sym"


#: The classes subscripting has made so far, by ``(base name, element, dimensions)``.
_SHAPES: dict[tuple[Any, ...], type] = {}


def _vector(element: Any, size: int) -> Any:
    """The ``SVector`` class of that element type and size, made once."""
    return _shaped(SVector, element, (int(size),))


def _matrix(element: Any, rows: int, cols: int, symmetric: bool = False) -> Any:
    """The ``SMatrix`` class of that shape, made once."""
    return _shaped(SMatrix, element, (int(rows), int(cols), int(symmetric)))


def _shaped(base: Any, element: Any, dims: tuple[int, ...]) -> Any:
    """The subclass of ``base`` for this element type and shape, made once and reused so that
    ``isinstance`` works whatever a macro wrote the type out as each time."""
    key = (base.__name__, getattr(element, "__name__", element), dims)
    made: Any = _SHAPES.get(key)
    if made is None:
        made = type(f"{base.__name__}_{'_'.join(str(d) for d in dims)}", (base,), {})
        element_type = float if element in (float, "float", "double") else int
        made._element = element_type
        made._dims = dims
        _SHAPES[key] = made
    return made


class SVector:
    """``SVector<T, D>``: a vector of ``D`` numbers, written ``SVector['float', 3]``."""

    _element: Callable[[Any], Any] = float
    _dims: tuple[int, ...] = (0,)

    def __class_getitem__(cls, spec: Any) -> Any:
        element, size = spec if isinstance(spec, tuple) else (float, spec)
        return _vector(element, int(size))

    def __init__(self, *values: Any) -> None:
        self._values = [self._element(v) for v in self._given(values)]

    @property
    def kSize(self) -> int:
        """How many elements the vector holds, as ROOT's ``SVector::kSize`` does."""
        return self._dims[0]

    def _given(self, values: tuple[Any, ...]) -> list[Any]:
        """The numbers a constructor was handed: none for a zero vector, one sequence to
        copy, or one number per element."""
        if not values:
            return [0] * self.kSize
        if len(values) == 1 and not _is_number(values[0]):
            given = list(values[0])
        else:
            given = list(values)
        if len(given) != self.kSize:
            raise TypeError(
                f"an SVector of {self.kSize} elements cannot be built from {len(given)} values"
            )
        return given

    def __len__(self) -> int:
        return self.kSize

    def __iter__(self) -> _typing.Iterator[Any]:
        return iter(self._values)

    def __getitem__(self, index: int) -> Any:
        return self._values[index]

    def __setitem__(self, index: int, value: Any) -> None:
        self._values[index] = self._element(value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SVector) and list(self._values) == list(other._values)

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__: Any = None  # mutable, as ROOT's vector is

    def _combined(self, other: Any, op: Callable[[Any, Any], Any]) -> Any:
        """``op`` element by element against another vector, or against a number."""
        if isinstance(other, SVector):
            if other._values and len(other._values) != self.kSize:
                raise TypeError(
                    f"SVector sizes differ: {self.kSize} and {other.kSize}, so they cannot be "
                    "combined element by element"
                )
            paired = zip(self._values, other._values, strict=False)
            return type(self)(*[op(a, b) for a, b in paired])
        return type(self)(*[op(a, other) for a in self._values])

    def __add__(self, other: Any) -> Any:
        return self._combined(other, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:
        return self._combined(other, lambda a, b: a - b)

    def __rsub__(self, other: Any) -> Any:
        return self._combined(other, lambda a, b: b - a)

    def __mul__(self, other: Any) -> Any:
        return self._combined(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Any:
        return self._combined(other, lambda a, b: a / b)

    def __neg__(self) -> Any:
        return type(self)(*[-a for a in self._values])

    def Place_at(self, vector: SVector, at: int) -> None:
        """Write ``vector`` into this one starting at index ``at``, as ``SVector::Place_at``
        does for a window of a longer vector."""
        for offset, value in enumerate(vector):
            self._values[at + offset] = self._element(value)

    def Dot(self, other: SVector) -> Any:
        """The scalar product with another vector of the same size."""
        return Dot(self, other)

    def Mag2(self) -> Any:
        """The squared length, which ROOT keeps in the element type."""
        return self.Dot(self)

    def __str__(self) -> str:
        return ", ".join(_text(v) for v in self._values)


def Dot(one: SVector, two: SVector) -> Any:
    """``Dot(a, b)``: the scalar product, in the element type of ``one``."""
    if len(one._values) != len(two._values):
        raise TypeError(
            f"a scalar product needs vectors of one size, not {one.kSize} and {two.kSize}"
        )
    total = sum(a * b for a, b in zip(one._values, two._values, strict=True))
    return one._element(total)


class SMatrix:
    """``SMatrix<T, D1, D2, R>``: ``D1`` by ``D2`` numbers, written ``SMatrix['float', 4, 3]``;
    one dimension gives ROOT's symmetric matrix."""

    _element: Callable[[Any], Any] = float
    _dims: tuple[int, ...] = (0, 0)

    def __class_getitem__(cls, spec: Any) -> Any:
        if not isinstance(spec, tuple):
            raise TypeError(f"SMatrix needs an element type and its dimensions, not {spec!r}")
        element, *dims = spec
        first, second = int(dims[0]), dims[1] if len(dims) > 1 else None
        if isinstance(second, _Shape):
            if second._kind == "sym" and second.rows != second.cols:
                raise TypeError(
                    f"MatRepSym needs a square matrix, and {second.rows} by {second.cols} is not "
                    "one"
                )
            rows, cols, symmetric = second.rows, second.cols, second._kind == "sym"
        else:
            rows, cols, symmetric = first, first if second is None else int(second), second is None
        return _matrix(element, rows, cols, symmetric)

    def __init__(self, *values: Any) -> None:
        self._values = [0] * (self.kRows * self.kCols)
        if values:
            single = len(values) == 1 and not _is_number(values[0])
            self._filled(list(values[0]) if single else list(values))

    def _filled(self, given: list[Any]) -> None:
        """Fill from the elements in row order, and from one triangle's worth when the matrix
        is symmetric, which is what ROOT's iterators hand over."""
        if len(given) != len(self._values):
            raise TypeError(
                f"an SMatrix of {self.kRows} by {self.kCols} cannot be built from "
                f"{len(given)} values"
            )
        for index, value in enumerate(given):
            self._values[index] = self._element(value)
        if self._symmetric:
            self._mirrored()

    @property
    def _symmetric(self) -> bool:
        return bool(self._dims[2])

    @property
    def kRows(self) -> int:
        """How many rows the matrix has, as ROOT's ``SMatrix::kRows`` does."""
        return self._dims[0]

    @property
    def kCols(self) -> int:
        """How many columns the matrix has, as ROOT's ``SMatrix::kCols`` does."""
        return self._dims[1]

    @property
    def kSize(self) -> int:
        """How many elements are stored: all of them, or one triangle of a symmetric one."""
        return self.kRows * self.kCols

    def _mirrored(self) -> None:
        """Copy the upper triangle onto the lower one, which is how a symmetric matrix is
        filled from the triangle ROOT's iterators hand over."""
        for i in range(self.kCols):
            for j in range(i):
                self._values[i * self.kCols + j] = self._values[j * self.kCols + i]

    def __call__(self, row: int, col: int) -> Any:
        return self._values[row * self.kCols + col]

    def __setcall__(self, row: int, col: int, value: Any) -> None:
        """``m(i, j) = value``, which the translator hands over rather than indexing."""
        self.__setitem__((row, col), value)

    def __getitem__(self, cell: tuple[int, int]) -> Any:
        return self(*cell)

    def __setitem__(self, cell: tuple[int, int], value: Any) -> None:
        row, col = cell
        self._values[row * self.kCols + col] = self._element(value)
        if self._symmetric:
            self._values[col * self.kCols + row] = self._element(value)

    def _array(self) -> Any:
        return np.array(self._values, dtype=float).reshape(self.kRows, self.kCols)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SMatrix) and list(self._values) == list(other._values)

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__: Any = None  # mutable, as ROOT's matrix is

    def _same_shape(self, other: SMatrix) -> None:
        if (other.kRows, other.kCols) != (self.kRows, self.kCols):
            raise TypeError(
                "SMatrix shapes differ: "
                f"{self.kRows} by {self.kCols} and {other.kRows} by {other.kCols}, so they "
                "cannot be combined element by element"
            )

    def __add__(self, other: Any) -> Any:
        if isinstance(other, SMatrix):
            self._same_shape(other)
            return type(self)([a + b for a, b in zip(self._values, other._values, strict=False)])
        return type(self)([a + other for a in self._values])

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, SMatrix):
            self._same_shape(other)
            return type(self)([a - b for a, b in zip(self._values, other._values, strict=False)])
        return type(self)([a - other for a in self._values])

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, SVector):
            return self._times_vector(other)
        if isinstance(other, SMatrix):
            return self._times_matrix(other)
        made = type(self)([a * other for a in self._values])
        return made

    def __rmul__(self, other: Any) -> Any:
        return self * other

    def __truediv__(self, other: Any) -> Any:
        return type(self)([a / other for a in self._values])

    def __neg__(self) -> Any:
        return type(self)([-a for a in self._values])

    def _times_matrix(self, other: SMatrix) -> Any:
        """The matrix product, which exists once the shapes meet: (n by m) by (m by p)."""
        if self.kCols != other.kRows:
            raise TypeError(
                f"cannot multiply a {self.kRows} by {self.kCols} matrix by a "
                f"{other.kRows} by {other.kCols} one: the inner dimensions differ"
            )
        product = self._array() @ other._array()
        made = _matrix(type(self)._element, self.kRows, other.kCols)
        return made(product.ravel().tolist())

    def _times_vector(self, other: SVector) -> SVector:
        """The matrix applied to a vector, which needs as many elements as there are columns."""
        if self.kCols != other.kSize:
            raise TypeError(
                f"cannot apply a {self.kRows} by {self.kCols} matrix to a vector of "
                f"{other.kSize} elements: the dimensions differ"
            )
        applied = self._array() @ np.array(list(other), dtype=float)
        wide = _vector(type(other)._element, self.kRows)
        return _typing.cast(SVector, wide(*applied.tolist()))

    def Transpose(self) -> Any:
        """The transpose, of the mirrored shape."""
        transposed = self._array().T
        made = _matrix(float, self.kCols, self.kRows)(transposed.ravel().tolist())
        return _typing.cast(SMatrix, made)

    def Trace(self) -> Any:
        """The sum of the diagonal, in the element type."""
        diagonal = [self._values[i * self.kCols + i] for i in range(min(self.kRows, self.kCols))]
        return self._element(sum(diagonal))

    def Determinant(self) -> Any:
        """The determinant of a square matrix, which a symmetric one always is."""
        if self.kRows != self.kCols:
            raise TypeError(
                f"only a square matrix has a determinant, and this one is {self.kRows} by "
                f"{self.kCols}"
            )
        return self._element(np.linalg.det(self._array()))

    def Invert(self) -> bool:
        """Replace a square matrix by its inverse in place, as ROOT's ``Invert`` does, and say
        whether it could: a singular matrix is left as it was and ``False`` comes back."""
        if self.kRows != self.kCols:
            return False
        try:
            inverse = np.linalg.inv(self._array())
        except np.linalg.LinAlgError:
            return False
        self._values = [self._element(v) for v in inverse.ravel().tolist()]
        return True

    def Place_in_row(self, vector: SVector, row: int, col: int) -> None:
        """Write ``vector`` into the row starting at ``(row, col)``."""
        for offset, value in enumerate(vector):
            self[row, col + offset] = value

    def Place_in_col(self, vector: SVector, row: int, col: int) -> None:
        """Write ``vector`` into the column starting at ``(row, col)``."""
        for offset, value in enumerate(vector):
            self[row + offset, col] = value

    def Place_at(self, other: SMatrix, row: int, col: int) -> None:
        """Write ``other`` into this matrix with its top left corner at ``(row, col)``."""
        for i in range(other.kRows):
            for j in range(other.kCols):
                self[row + i, col + j] = other(i, j)

    def __iter__(self) -> _typing.Iterator[Any]:
        return iter(self._values)

    def __str__(self) -> str:
        """ROOT's ``operator<<``: ``[ `` then each element right aligned in twelve columns, one
        row a line, and `` ]`` after the last."""
        rows = [
            "".join(f"{_text(self._values[r * self.kCols + c]):>12}" for c in range(self.kCols))
            for r in range(self.kRows)
        ]
        return "[ " + "\n  ".join(rows) + " ]"
