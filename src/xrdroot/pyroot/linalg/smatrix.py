"""``ROOT::Math``'s fixed-size vectors and matrices: ``SVector<T, D>``, ``SMatrix<T, D1, D2>``.

A C++ macro writes the shape into the type, ``SVector<float, 3>``, which a Python class cannot:
the translator turns that into a subscript, ``SVector['float', 3]``, and subscripting a class
here makes - once, cached - a subclass carrying the element type and the dimensions. What is
stored is one flat list of numbers, so the arithmetic is plain Python arithmetic and no NumPy
dtype decides what ``float`` means for a matrix ROOT would have kept in single precision.

``SMatrix<float, 3>`` with one dimension is ROOT's symmetric matrix, and an explicit
``MatRepSym``/``MatRepStd`` fourth template argument says which representation, exactly as
ROOT defaults it.

The free functions a macro calls unqualified in ``ROOT::Math`` - ``Transpose``, ``Similarity``,
``Mag``, ``Cross``, ``Square``, ``Times`` - are here beside the classes, because what they do
is what the classes hold and nothing above this layer decides it.
"""

from __future__ import annotations

import copy
import math
import typing as _typing
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

__all__ = [
    "MatRepStd",
    "MatRepSym",
    "SMatrix",
    "SMatrixIdentity",
    "SVector",
    "AssignSym",
    "Cross",
    "Dot",
    "Mag",
    "Mag2",
    "Similarity",
    "SimilarityT",
    "Square",
    "Times",
    "Transpose",
    "Unit",
]


def _text(value: Any) -> str:
    """One number as a C++ stream writes it at its default precision, which is what ROOT's
    ``operator<<`` for these classes prints."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):g}"


def _is_number(value: Any) -> bool:
    """Whether ``value`` is a number rather than something a vector can be filled from."""
    return isinstance(value, (int, float, np.integer, np.floating))


def _is_dimension(value: Any) -> bool:
    """Whether ``value`` is a dimension a subscript gave, rather than the element type or the
    representation - which are names, and travel as C++ text."""
    return isinstance(value, (int, np.integer))


def _triangle_size(size: int) -> int:
    """How many values the upper triangle of a symmetric ``size`` by ``size`` matrix holds:
    what its iterators hand over, where a matrix that stores every element hands over all
    ``size * size`` of them."""
    return size * (size + 1) // 2


def _upto(first: Any, last: Any) -> list[Any]:
    """The numbers from one cursor up to another of the same container, stepped the way C++
    steps one: ``*it`` reads and ``++it`` moves on, which is what a ``std::vector``'s iterator
    and the one here both do."""
    found: list[Any] = []
    at = copy.copy(first)  # the caller's cursor keeps its own position
    while at != last:
        found.append(at.__deref__())
        at += 1
    return found


def _counted(first: Any, last: Any) -> list[Any]:
    """The numbers between two C pointers: ``SMatrix<T,D>(a, size)`` hands over the array and a
    count, ``SMatrix<T,D>(begin, end)`` two pointers into one array - which the translator
    writes as two slices, ``d`` and ``d + 9`` - and ``std::vector<T>(v.begin(), v.end())`` two
    iterators of another container, which are stepped."""
    if _is_number(last):
        return list(first)[: int(last)]
    if not (hasattr(first, "__len__") and hasattr(last, "__len__")):
        return _upto(first, last)
    return _spanning(list(first), list(last))


def _spanning(whole: list[Any], rest: list[Any]) -> list[Any]:
    """The head of one array a second slice of the same array ends: ``d`` and ``d + 9`` differ
    by what is left after the nine elements they name."""
    if len(rest) >= len(whole) or whole[len(whole) - len(rest) :] != rest:
        raise TypeError(
            "an SVector or SMatrix is built from a sequence, from a pointer and a count, or "
            "from two pointers into one array, and these two arguments are none of those"
        )
    return whole[: len(whole) - len(rest)]


def _selected(values: tuple[Any, ...]) -> list[Any]:
    """The numbers a constructor's arguments name: one sequence to copy - also when it is one
    half of a ``(pointer, count)`` or ``(begin, end)`` pair - or one number per element."""
    if len(values) == 1 and not _is_number(values[0]):
        return list(values[0])
    if len(values) == 2 and not _is_number(values[0]):
        return _counted(values[0], values[1])
    return list(values)


class _Templated:
    """A method a macro may also name with a result type, ``m.SubRow<SVector<double,2>>(0, 1)``:
    the translator writes that as a subscript on the method, ``m.SubRow[SVector['double', 2]]
    (0, 1)``, so the attribute has to take one before it is called. The type is not passed on -
    these methods take their shape from where they start, the remainder of the row or column -
    and an ordinary ``m.Row(0)`` reaches the same call through ``__call__``."""

    def __init__(self, method: Callable[..., Any]) -> None:
        self._method = method
        self.__doc__ = method.__doc__
        self.__name__ = method.__name__

    def __get__(self, instance: Any, owner: Any = None) -> _Templated:
        return type(self)(self._method.__get__(instance, owner))

    def __getitem__(self, result_type: Any) -> Callable[..., Any]:
        return self._method

    def __call__(self, *args: Any) -> Any:
        return self._method(*args)


class _Cursor:
    """A position in a vector or matrix, which ``(begin, last)`` builds another container from:
    the range form ``std::vector<T>(a.begin(), a.end())`` compares its two cursors and steps
    one, so this is what both an equal comparison and ``next()`` mean here. Two cursors are
    equal at one position over the same values, which is what a range over a symmetric matrix
    asks of the two lists a triangle is read out into."""

    def __init__(self, values: list[Any], at: int = 0) -> None:
        self._values, self._at = values, at

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _Cursor)
            and self._at == other._at
            and self._values == other._values
        )

    def __iadd__(self, step: int) -> _Cursor:
        """``++it``: the next position, which walking a range moves to."""
        self._at += step
        return self

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__: Any = None  # a position, compared as one

    def __next__(self) -> Any:
        taken = self._values[self._at]
        self._at += 1
        return taken

    def __deref__(self) -> Any:
        """``*it``: the element the cursor is at, which a range reads at each step."""
        return self._values[self._at]

    def __iter__(self) -> Iterator[Any]:
        while self._at < len(self._values):
            yield next(self)


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


#: The representations a fourth template argument names, by the C++ name a macro writes.
_REPRESENTATIONS: dict[str, type[_Shape]] = {"MatRepSym": MatRepSym, "MatRepStd": MatRepStd}


def _declared_shape(dims: list[Any]) -> _Shape:
    """The shape a subscript's dimensions name: ``[3]`` ROOT's symmetric 3 by 3, ``[2, 3]``
    a full 2 by 3, and either followed by the ``MatRepSym``/``MatRepStd`` a macro's type gives
    as its fourth template argument."""
    sizes, representation = _dimensions_and_representation(dims)
    fitted = _representation(representation, sizes[0], sizes[-1])
    if fitted is not None:
        return fitted
    return MatRepSym(sizes[0], sizes[-1]) if len(sizes) == 1 else MatRepStd(sizes[0], sizes[-1])


def _dimensions_and_representation(dims: list[Any]) -> tuple[list[int], Any]:
    """ROOT's ``SMatrix<T, D1, D2, R>`` read off a subscript, whose dimensions come as numbers
    and whose representation comes as the C++ text of a type: ``'MatRepStd<double,2,3>'``."""
    sizes: list[int] = []
    named: list[Any] = []
    for each in dims:
        if _is_dimension(each):
            sizes.append(int(each))
        else:
            named.append(each)
    if not 1 <= len(sizes) <= 2 or len(named) > 1:
        raise TypeError(
            "an SMatrix is named by one or two dimensions and at most one MatRepSym or "
            f"MatRepStd, not {list(dims)!r}"
        )
    return sizes, named[0] if named else None


def _representation(named: Any, rows: int, cols: int) -> _Shape | None:
    """The ``MatRepSym``/``MatRepStd`` a fourth template argument names, or nothing when the
    type leaves it to ROOT's default; a symmetric one has to be square, as ROOT refuses to
    compile one that is not."""
    if named is None:
        return None
    fitted = named if isinstance(named, _Shape) else _named_representation(named, rows, cols)
    if fitted._kind == "sym" and fitted.rows != fitted.cols:
        raise TypeError(
            f"MatRepSym needs a square matrix, and {fitted.rows} by {fitted.cols} is not one"
        )
    return fitted


def _named_representation(text: Any, rows: int, cols: int) -> _Shape:
    """The representation the C++ text of a fourth template argument names, which is how a
    translated macro hands its type over - ``'MatRepSym<double,3>'`` - carrying the size it
    names, or else the shape the rest of the subscript already gave."""
    name, _, inside = str(text).partition("<")
    made = _REPRESENTATIONS.get(name.rpartition(":")[2].strip())
    if made is None:
        raise TypeError(
            "an SMatrix's fourth template argument is MatRepSym or MatRepStd, and "
            f"{text!r} is neither"
        )
    sizes = [int(part) for part in inside.partition(">")[0].split(",") if part.strip().isdigit()]
    return made(sizes[0], sizes[-1]) if sizes else made(rows, cols)


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
        given = [0] * self.kSize if not values else _selected(values)
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

    def begin(self) -> _Cursor:
        """A cursor at the first element, so that ``std::vector<T>(v.begin(), v.end())`` - and
        every other range a macro builds from a vector - can read it."""
        return _Cursor(self._values)

    def end(self) -> _Cursor:
        """A cursor past the last element, which is where a range ends."""
        return _Cursor(self._values, self.kSize)

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
        shape = _declared_shape(dims)
        return _matrix(element, shape.rows, shape.cols, shape._kind == "sym")

    def __init__(self, *values: Any) -> None:
        self._values = [0] * (self.kRows * self.kCols)
        if values:
            self._filled(_selected(values))

    def _filled(self, given: list[Any]) -> None:
        """Fill from the elements in row order, and from the upper triangle when the matrix is
        symmetric, which is all ``MatRepSym`` stores and so all its iterators hand over."""
        if self._symmetric:
            self._filled_triangle(given)
            return
        if len(given) != len(self._values):
            raise TypeError(
                f"an SMatrix of {self.kRows} by {self.kCols} cannot be built from "
                f"{len(given)} values"
            )
        for index, value in enumerate(given):
            self._values[index] = self._element(value)

    def _filled_triangle(self, given: list[Any]) -> None:
        """A symmetric matrix from its upper triangle taken column by column, which is the
        order ``SMatrix<T,N>(v.begin(), v.end())`` copies - ROOT's ``MatRepSym`` keeps that
        triangle and mirrors the other one. Anything past the triangle is not read, as ROOT
        does not read it either."""
        stored = _triangle_size(self.kRows)
        if len(given) < stored:
            raise TypeError(
                f"a symmetric SMatrix of {self.kRows} by {self.kCols} is built from the "
                f"{stored} values of its upper triangle, and {len(given)} is not that many"
            )
        at = 0
        for col in range(self.kCols):
            for row in range(col + 1):
                self[row, col] = given[at]
                at += 1

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

    def Row(self, row: int) -> SVector:
        """One row as a vector, as ``SMatrix::Row`` makes it."""
        return self._vector_of([self._values[row * self.kCols + c] for c in range(self.kCols)])

    def Col(self, col: int) -> SVector:
        """One column as a vector, as ``SMatrix::Col`` makes it."""
        return self._vector_of([self._values[r * self.kCols + col] for r in range(self.kRows)])

    def Diagonal(self) -> SVector:
        """The diagonal as a vector, which is as long as the shorter side."""
        return self._vector_of(
            [self._values[i * self.kCols + i] for i in range(min(self.kRows, self.kCols))]
        )

    def _vector_of(self, numbers: list[Any]) -> SVector:
        """A vector of this matrix's element type holding ``numbers``."""
        made = _vector(type(self)._element, len(numbers))
        return _typing.cast(SVector, made(numbers))

    @_Templated
    def SubRow(self, row: int, col: int) -> SVector:
        """The rest of row ``row`` from column ``col``, as ``SubRow<M>(row, col)`` copies."""
        return self._vector_of(
            [self._values[row * self.kCols + c] for c in range(col, self.kCols)]
        )

    @_Templated
    def SubCol(self, row: int, col: int) -> SVector:
        """The rest of column ``col`` from row ``row``, as ``SubCol<M>(row, col)`` copies."""
        return self._vector_of(
            [self._values[r * self.kCols + col] for r in range(row, self.kRows)]
        )

    @_Templated
    def Sub(self, row: int, col: int) -> SMatrix:
        """The corner from ``(row, col)`` to the bottom right, as ``Sub<M>(row, col)`` copies:
        the shape a macro names in the angle brackets is the remainder, which the translator
        does not pass on."""
        made = _matrix(type(self)._element, self.kRows - row, self.kCols - col)
        cells = [self(r, c) for r in range(row, self.kRows) for c in range(col, self.kCols)]
        return _typing.cast(SMatrix, made(cells))

    def UpperBlock(self) -> SVector:
        """The upper triangle in row order, which is what a symmetric matrix stores."""
        cells = [
            self._values[r * self.kCols + c]
            for r in range(self.kRows)
            for c in range(r, self.kCols)
        ]
        return self._vector_of(cells)

    def LowerBlock(self) -> SVector:
        """The lower triangle in row order, the same numbers as the upper one when the matrix
        is symmetric."""
        cells = [
            self._values[r * self.kCols + c] for r in range(self.kRows) for c in range(r + 1)
        ]
        return self._vector_of(cells)

    def _stored_values(self) -> list[Any]:
        """What a range over this matrix hands over: every element in row order, or one
        triangle of a symmetric one, which is all it keeps."""
        if not self._symmetric:
            return self._values
        return [self(row, col) for col in range(self.kCols) for row in range(col + 1)]

    def begin(self) -> _Cursor:
        """A cursor at the first stored element, so a range built from
        ``(m.begin(), m.end())`` reads the matrix as ROOT's iterators hand it over."""
        return _Cursor(self._stored_values())

    def end(self) -> _Cursor:
        """A cursor past the last stored element."""
        return _Cursor(self._stored_values(), len(self._stored_values()))

    def _product_by_element(self, other: Any) -> Any:
        """Each element times the matching one, or times a number: ``Times(a, b)``, which is
        not the matrix product ``a * b`` is."""
        if isinstance(other, SMatrix):
            self._same_shape(other)
            return type(self)([a * b for a, b in zip(self._values, other._values, strict=True)])
        return type(self)([a * other for a in self._values])

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


class AssignSym:
    """``AssignSym::Evaluate(sym, product)``: a symmetric matrix given the one triangle a
    product ROOT knows is symmetric - what ``Similarity`` returns and a Kalman fit writes as
    ``H*Cp*Transpose(H)``, which is only symmetric up to rounding, so ROOT fills the triangle
    it stores and mirrors it rather than checking the other half."""

    @staticmethod
    def Evaluate(lhs: Any, rhs: Any) -> None:
        """Write ``rhs``'s lower triangle into ``lhs``, which is the triangle ROOT's
        ``MatRepSym`` keeps, and mirror it - the rest of ``rhs`` is not looked at."""
        if not isinstance(lhs, SMatrix) or not lhs._symmetric:
            raise TypeError(
                "AssignSym::Evaluate takes a symmetric SMatrix, SMatrix<T,N,MatRepSym<T,N>> "
                "or SMatrix<T,N> of one dimension, as its result; pass a matrix of that shape "
                "or assign element by element"
            )
        for row in range(lhs.kRows):
            for col in range(row + 1):
                lhs[row, col] = rhs[row, col]


def Transpose(matrix: SMatrix) -> SMatrix:
    """``Transpose(A)``: the mirrored matrix, which is what ``A.Transpose()`` makes too."""
    return _typing.cast(SMatrix, matrix.Transpose())


def Mag2(vector: SVector) -> Any:
    """``Mag2(x)``: the squared length, ``Dot(x, x)``, kept in the element type."""
    return Dot(vector, vector)


def Mag(vector: SVector) -> Any:
    """``Mag(x)``: the length, in the element type - which ROOT truncates for an integer one."""
    return vector._element(math.sqrt(Dot(vector, vector)))


def Unit(vector: SVector) -> SVector:
    """``Unit(x)``: the same direction, one long, as ``x / Mag(x)``."""
    return _typing.cast(SVector, vector / Mag(vector))


def Cross(one: SVector, other: SVector) -> SVector:
    """``Cross(a, b)``: the vector product of two three-element vectors."""
    if one.kSize != 3 or other.kSize != 3:
        raise TypeError(
            f"a vector product is of two vectors of three elements, not {one.kSize} and "
            f"{other.kSize}"
        )
    a, b = list(one), list(other)
    return type(one)(
        a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]
    )


def Times(one: Any, two: Any) -> Any:
    """``Times(a, b)``: the element-by-element product - of two matrices, of a matrix and a
    number, or of two vectors - which ROOT keeps apart from the matrix product."""
    if isinstance(one, SMatrix):
        return one._product_by_element(two)
    return one._combined(two, lambda a, b: a * b)


def Square(one: Any) -> Any:
    """``Square(x)``: each element multiplied by itself, which is ``Times(x, x)``."""
    return Times(one, one)


def Similarity(one: Any, other: Any) -> Any:
    """``Similarity(x, A)``: ``x^T A x`` for a vector and a matrix, and ``A B A^T`` for two
    matrices, which is the sandwich a Kalman fit is written with."""
    if isinstance(one, SVector):
        return Dot(one, other * one)
    return one * other * one.Transpose()


def SimilarityT(one: Any, other: Any) -> Any:
    """``SimilarityT(A, B)``: ``A^T B A``, the same sandwich with the first matrix transposed
    on the other side."""
    return one.Transpose() * other * one


def SMatrixIdentity(like: Any = None) -> Any:
    """``SMatrixIdentity<T,D1,D2>()``: the identity matrix of the shape the declaration it is
    assigned to names, which is where C++ reads the template arguments from - ``like`` is that
    declared class, and without it the shape is not written anywhere in the call."""
    if like is None:
        raise TypeError(
            "SMatrixIdentity() takes its shape from the variable it initialises, and the "
            "translator has to name that variable's class here for the identity to have a size"
        )
    identity = like()
    for at in range(min(identity.kRows, identity.kCols)):
        identity[at, at] = 1
    return identity
