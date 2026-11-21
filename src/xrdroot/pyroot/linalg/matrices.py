"""``TMatrixT``: ROOT's dense matrices - general, symmetric, Hilbert's - over a NumPy array.

A matrix is made by size, ``TMatrixD(3, 4)``; with lower bounds,
``TMatrixD(1, 3, 1, 4)``; from its elements, row by row; as a copy; from
an operation, ``TMatrixD(A, TMatrixD::kMult, B)`` or ``TMatrixDSym(kAtA,
A)``; or from a NumPy array of rows. It is indexed as C++ indexes it -
``m(i, j)`` and ``m[i][j]`` from the lower bounds - and does what ROOT's
matrices do in place - ``Invert``, ``Transpose``, ``T()``, ``Abs()`` -
giving itself back, so that calls chain as they do in C++. ``Print`` lays
it out as ``TMatrixTBase::Print`` does.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from ..core.objects import TObject

__all__ = ["TMatrix", "TMatrixT", "TMatrixTSym", "TMatrixD", "TMatrixF", "TMatrixDSym",
           "TMatrixFSym", "TMatrixTLazy", "TMatrixTSymLazy", "THilbertMatrixD",
           "THilbertMatrixDSym", "THilbertMatrixF", "hand_back"]  # fmt: skip

#: The one-matrix operations a matrix can be made by: ``EMatrixCreatorsOp1``.
kZero, kUnit, kTransposed, kInverted, kAtA = range(5)
#: The two-matrix ones: ``EMatrixCreatorsOp2``.
kMult, kTransposeMult, kInvMult, kMultTranspose, kPlus, kMinus = range(6)

#: How each two-matrix operation makes its matrix.
OPERATIONS = {
    kMult: lambda a, b: a @ b,
    kTransposeMult: lambda a, b: a.T @ b,
    kInvMult: lambda a, b: np.linalg.inv(a) @ b,
    kMultTranspose: lambda a, b: a @ b.T,
    kPlus: lambda a, b: a + b,
    kMinus: lambda a, b: a - b,
}


def _one(op: int, a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """The matrix a one-matrix operation makes of ``a``."""
    made = {kZero: lambda: np.zeros_like(a), kUnit: lambda: np.eye(*a.shape),
            kTransposed: lambda: a.T.copy(), kInverted: lambda: np.linalg.inv(a),
            kAtA: lambda: a.T @ a}  # fmt: skip
    return np.array(made[op](), dtype=np.float64)


def values_of(given: Any) -> np.ndarray[Any, Any]:
    """The elements of a matrix, vector or array, as a NumPy array."""
    return given.values if isinstance(given, TMatrixT) else np.asarray(given, dtype=np.float64)


def hand_back(det: Any, value: Any) -> None:
    """Hand a value back through a pointer or reference - ``Double_t *det``, ``Bool_t &ok`` -
    when the caller gave one."""
    if det is not None and hasattr(det, "value"):
        det.value = value


#: The element type a matrix has unless its name says ``Float_t``.
FLOAT64 = np.dtype(np.float64)


def _counted(args: tuple[Any, ...]) -> bool:
    return all(isinstance(each, (int, np.integer)) and not isinstance(each, bool) for each in args)


def _made(kind: type[TMatrixT], args: tuple[Any, ...]) -> tuple[np.ndarray[Any, Any], int, int]:
    """A matrix's elements and lower bounds, from any of its constructors' arguments."""
    if not args:
        return np.zeros((0, 0)), 0, 0
    operated = _operated(args)
    if operated is not None:
        return operated
    if len(args) >= 4 and _counted(args[:4]):  # (row_lwb, row_upb, col_lwb, col_upb[, data])
        rows, cols = int(args[1]) - int(args[0]) + 1, int(args[3]) - int(args[2]) + 1
        return _filled(rows, cols, args[4:]), int(args[0]), int(args[2])
    return _sized(kind, args), 0, 0


def _operated(args: tuple[Any, ...]) -> tuple[np.ndarray[Any, Any], int, int] | None:
    """``(A, op, B)`` and ``(op, A)``: the matrix an operation makes, or ``None`` for neither."""
    if len(args) == 3 and isinstance(args[0], TMatrixT) and _counted(args[1:2]):
        return OPERATIONS[int(args[1])](args[0].values, values_of(args[2])), 0, 0
    if len(args) == 2 and _counted(args[:1]) and isinstance(args[1], TMatrixT):
        return _one(int(args[0]), args[1].values), args[1].rowlwb, args[1].collwb
    return None


def _sized(kind: type[TMatrixT], args: tuple[Any, ...]) -> np.ndarray[Any, Any]:
    """``(n)``, ``(nrows, ncols[, data, option])``, a copy, or rows given whole."""
    first = args[0]
    if isinstance(first, TMatrixT):
        return first.values.copy()
    if not _counted(args[:1]):
        return np.array(values_of(first), dtype=np.float64)
    if len(args) == 1 or not _counted(args[1:2]) or kind.SYMMETRIC:
        return _filled(int(first), int(first), args[1:2])
    return _filled(int(first), int(args[1]), args[2:])


def _filled(rows: int, cols: int, data: tuple[Any, ...]) -> np.ndarray[Any, Any]:
    """A ``rows`` by ``cols`` matrix of zeros, or of the elements given - by column for ``F``."""
    if not data or data[0] is None:
        return np.zeros((rows, cols))
    flat = np.asarray(data[0], dtype=np.float64)[:rows * cols]
    by_column = len(data) > 1 and "F" in str(data[1]).upper()
    return flat.reshape((cols, rows)).T.copy() if by_column else flat.reshape(rows, cols).copy()


class TMatrixTLazy(TObject):
    """``TMatrixTLazy<Element>``: a matrix promised, and made only when one is needed.

    Its bounds are the matrix it stands for - ``TMatrixTLazy(nrows, ncols)`` or
    ``TMatrixTLazy(row_lwb, row_upb, col_lwb, col_upb)`` - and ``FillIn``, which a macro's
    own class overrides, is what writes the elements into the matrix built from it. A
    ``TMatrixD m = promise(...)`` builds ``m`` and calls ``promise.FillIn(m)``.
    """

    def __init__(self, *args: Any) -> None:
        super().__init__()
        row_lwb, row_upb, col_lwb, col_upb = _promised(type(self), args)
        self.fRowLwb, self.fRowUpb = row_lwb, row_upb
        self.fColLwb, self.fColUpb = col_lwb, col_upb

    def GetRowLwb(self) -> int:
        return self.fRowLwb

    def GetRowUpb(self) -> int:
        return self.fRowUpb

    def GetColLwb(self) -> int:
        return self.fColLwb

    def GetColUpb(self) -> int:
        return self.fColUpb

    def FillIn(self, matrix: Any) -> None:
        raise UnsupportedFeatureError(
            f"{type(self).__name__} was forced into a matrix with no FillIn, which is the "
            "method ROOT calls to write the matrix's elements; give the class a FillIn."
        )


class TMatrixTSymLazy(TMatrixTLazy):
    """``TMatrixTSymLazy<Element>``: the same promise, of a symmetric matrix."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)

    def FillIn(self, matrix: Any) -> None:
        raise UnsupportedFeatureError(
            f"{type(self).__name__} was forced into a symmetric matrix with no FillIn, which "
            "is the method ROOT calls to write its elements; give the class a FillIn."
        )


def _promised(kind: type[TMatrixTLazy], args: tuple[Any, ...]) -> tuple[int, int, int, int]:
    """A promise's four bounds: ``(nrows, ncols)``, or the four it is given exactly."""
    if kind is TMatrixTSymLazy or len(args) < 3:
        rows, cols = int(args[0]), int(args[1] if len(args) > 1 else args[0])
        return 0, rows - 1, 0, cols - 1
    row_lwb, row_upb, col_lwb, col_upb = (int(each) for each in args[:4])
    return row_lwb, row_upb, col_lwb, col_upb


class TMatrixT(TObject):
    """``TMatrixT<double>``: a dense matrix, counted from its lower bounds."""

    #: Whether this is a symmetric matrix, ``TMatrixTSym``, made square.
    SYMMETRIC = False
    #: How ROOT keeps the elements: ``Double_t``, or ``Float_t`` for the ``F`` matrices.
    DTYPE: Any = FLOAT64
    #: The instance of this template over ``Float_t``, which ``TMatrixT<float>`` is.
    FLOAT: type[TMatrixT]
    CPP = "TMatrixT<double>"
    kZero, kUnit, kTransposed, kInverted, kAtA = kZero, kUnit, kTransposed, kInverted, kAtA
    kMult, kTransposeMult, kInvMult = kMult, kTransposeMult, kInvMult
    kMultTranspose, kPlus, kMinus = kMultTranspose, kPlus, kMinus

    def __init__(self, *args: Any) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], TMatrixTLazy):
            self._promised_by(args[0])
            return
        self.values, self.rowlwb, self.collwb = _made(type(self), args)
        self.values = self.values.astype(self.DTYPE)

    def _promised_by(self, lazy: TMatrixTLazy) -> None:
        """``TMatrixT(const TMatrixTLazy &)``: the bounds promised, then the promise forced."""
        rows = lazy.GetRowUpb() - lazy.GetRowLwb() + 1
        cols = lazy.GetColUpb() - lazy.GetColLwb() + 1
        self.values = np.zeros((rows, cols), dtype=self.DTYPE)
        self.rowlwb, self.collwb = lazy.GetRowLwb(), lazy.GetColLwb()
        lazy.FillIn(self)

    def __class_getitem__(cls, kind: Any) -> type:
        """``TMatrixT<double>``, or ``TMatrixT<float>`` - the ``F`` matrices of 32-bit floats."""
        return cls.FLOAT if str(kind) in _FLOAT_KINDS else cls

    def ClassName(self) -> str:
        return self.CPP

    def __repr__(self) -> str:
        return f"<{self.CPP} {self.GetNrows()}x{self.GetNcols()}>"

    # -- shape --------------------------------------------------------------------------------

    def GetNrows(self) -> int:
        return int(self.values.shape[0])

    def GetNcols(self) -> int:
        return int(self.values.shape[1])

    def GetNoElements(self) -> int:
        return int(self.values.size)

    def GetRowLwb(self) -> int:
        return self.rowlwb

    def GetColLwb(self) -> int:
        return self.collwb

    def GetRowUpb(self) -> int:
        return self.rowlwb + self.GetNrows() - 1

    def GetColUpb(self) -> int:
        return self.collwb + self.GetNcols() - 1

    def __len__(self) -> int:
        return self.GetNrows()

    def ResizeTo(self, nrows: Any, ncols: Any = None) -> TMatrixT:
        """``ResizeTo(n, m)``, or to another matrix's shape: what fits kept, the rest zero."""
        rows, cols = (nrows.GetNrows(), nrows.GetNcols()) if isinstance(nrows, TMatrixT) else (
            int(nrows), int(nrows if ncols is None else ncols))  # fmt: skip
        grown = np.zeros((rows, cols), dtype=self.DTYPE)
        keep = (min(rows, self.GetNrows()), min(cols, self.GetNcols()))
        grown[: keep[0], : keep[1]] = self.values[: keep[0], : keep[1]]
        self.values = grown
        return self

    # -- elements -----------------------------------------------------------------------------

    def __call__(self, i: Any, j: Any) -> float:
        return float(self.values[int(i) - self.rowlwb, int(j) - self.collwb])

    def __setcall__(self, i: Any, j: Any, value: Any) -> None:
        """``m(i, j) = value``."""
        self.values[int(i) - self.rowlwb, int(j) - self.collwb] = float(value)

    def __getitem__(self, i: Any) -> Any:
        """``m[i]``, a row - which ``m[i][j]`` indexes - or ``m[i, j]``, an element."""
        if isinstance(i, tuple):
            return float(self.values[int(i[0]) - self.rowlwb, int(i[1]) - self.collwb])
        return self.values[int(i) - self.rowlwb]

    def __setitem__(self, i: Any, value: Any) -> None:
        """``m[i, j] = value`` - PyROOT's - or a whole row."""
        if isinstance(i, tuple):
            self.values[int(i[0]) - self.rowlwb, int(i[1]) - self.collwb] = value
        else:
            self.values[int(i) - self.rowlwb] = value

    def matrix(self) -> np.ndarray[Any, Any]:
        """The elements, as the engine takes them."""
        return self.values

    def GetMatrixArray(self) -> np.ndarray[Any, Any]:
        return self.values.ravel()

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray[Any, Any]:
        """The elements, as NumPy asks for them: a copy when it asks for one."""
        made = np.asarray(self.values, dtype=dtype)
        return made.copy() if copy else made

    def _assign(self, value: Any) -> None:
        """``m = other``: the other's elements copied in; ``m = 0.`` every element set.

        A promise - a ``TMatrixTLazy``, or a macro's class derived from one - is forced, as
        ``TMatrixT::operator=(const TMatrixTLazy &)`` forces it, into this matrix.
        """
        if isinstance(value, TMatrixTLazy):
            value.FillIn(self)
            return
        self.values = values_of(value).astype(self.DTYPE) if np.ndim(value) else (
            np.full_like(self.values, float(value)))  # fmt: skip

    def Print(self, option: Any = "") -> None:
        """``TMatrixTBase::Print``: sheets of five columns (ten for a narrow ``f=`` format),
        each headed by its columns' numbers set in a bar as wide as ROOT makes it."""
        from ...roofit.matrix import matrix_text

        found = str(option).find("f=")
        print(matrix_text(self.values, *([str(option)[found + 2:]] if found >= 0 else [])), end="")

    # -- in place, giving itself back -------------------------------------------------------

    def _became(self, values: Any) -> TMatrixT:
        self.values = np.array(values, dtype=self.DTYPE)
        return self

    def Invert(self, det: Any = None) -> TMatrixT:
        """``Invert(&det)``: the inverse, by LU decomposition; its determinant handed back."""
        hand_back(det, float(np.linalg.det(self.values)))
        return self._became(np.linalg.inv(self.values))

    InvertFast = Invert

    def Transpose(self, source: Any = None) -> TMatrixT:
        """``Transpose(m)``: this the transpose of ``m`` - of itself, when none is given."""
        return self._became(values_of(self if source is None else source).T)

    def T(self) -> TMatrixT:
        """``T()``: transposed in place."""
        return self.Transpose()

    def Abs(self) -> TMatrixT:
        return self._became(np.abs(self.values))

    def Sqr(self) -> TMatrixT:
        return self._became(self.values**2)

    def Sqrt(self) -> TMatrixT:
        return self._became(np.sqrt(self.values))

    def Zero(self) -> TMatrixT:
        return self._became(np.zeros_like(self.values))

    def UnitMatrix(self) -> TMatrixT:
        return self._became(np.eye(*self.values.shape))

    def Apply(self, action: Any) -> TMatrixT:
        """``Apply(f)``: ``f.Operation`` run on every element, row by row, giving itself back.

        A ``TElementPosActionT`` is told where each element is, in ``fI`` and ``fJ``, and
        writes the element it is given; a symmetric matrix acts on its upper triangle only
        and mirrors the result, as ``TMatrixTSym::Apply`` does.
        """
        from .actions import applied, applied_symmetric

        (applied_symmetric if self.SYMMETRIC else applied)(self, action)
        return self

    def GetSub(self, row_lwb: Any, row_upb: Any, col_lwb: Any, col_upb: Any,
               option: Any = "") -> TMatrixT:  # fmt: skip
        """``GetSub(rl, ru, cl, cu[, opt])``: the submatrix given by those bounds.

        Its lower bounds are the ones asked for, or ``0`` with ``"S"`` in the option, which
        shifts them to the origin.
        """
        rows = slice(int(row_lwb) - self.rowlwb, int(row_upb) - self.rowlwb + 1)
        cols = slice(int(col_lwb) - self.collwb, int(col_upb) - self.collwb + 1)
        found = type(self)(self.values[rows, cols])
        shift = "S" in str(option).upper()
        found.rowlwb, found.collwb = (0, 0) if shift else (int(row_lwb), int(col_lwb))
        return found

    def SetSub(self, row_lwb: Any, col_lwb: Any, source: Any) -> TMatrixT:
        """``SetSub(rl, cl, source)``: ``source``'s elements written in from there."""
        given = values_of(source)
        row, col = int(row_lwb) - self.rowlwb, int(col_lwb) - self.collwb
        rows, cols = given.shape
        self.values[row: row + rows, col: col + cols] = given
        return self

    # -- what it is ---------------------------------------------------------------------------

    def Determinant(self, d1: Any = None, d2: Any = None) -> float:
        """``Determinant()``, or ``Determinant(d1, d2)``: ``d1 * 2**d2``, ``d1`` in ``[0.5, 1)``."""
        found = float(np.linalg.det(self.values))
        mantissa, exponent = np.frexp(found)
        hand_back(d1, float(mantissa))
        hand_back(d2, float(exponent))
        return found

    def Max(self) -> float:
        return float(self.values.max())

    def Min(self) -> float:
        return float(self.values.min())

    def Sum(self) -> float:
        return float(self.values.sum())

    def E2Norm(self) -> float:
        return float((self.values**2).sum())

    def NormInf(self) -> float:
        """The largest sum of a row's absolute values."""
        return float(np.abs(self.values).sum(axis=1).max())

    RowNorm = NormInf
    """``RowNorm``: the norm induced by the infinity vector norm, which is ``NormInf``."""

    def Norm1(self) -> float:
        """The largest sum of a column's absolute values."""
        return float(np.abs(self.values).sum(axis=0).max())

    ColNorm = Norm1
    """``ColNorm``: the norm induced by the 1 vector norm, which is ``Norm1``."""

    def IsSymmetric(self) -> bool:
        return bool(self.values.shape[0] == self.values.shape[1] and
                    np.array_equal(self.values, self.values.T))  # fmt: skip

    # -- arithmetic -----------------------------------------------------------------------------

    def _with(self, values: Any) -> Any:
        """A matrix of these elements; a vector, ``TVectorD``, for a column of them."""
        if np.ndim(values) == 1:
            from ..vectors import TVectorD

            return TVectorD(values)
        made = np.asarray(values)
        square = made.shape[0] == made.shape[1]
        kept = self.SYMMETRIC and square and np.allclose(made, made.T)  # symmetric still
        return (type(self) if kept else TMatrixD)(made)

    def __mul__(self, other: Any) -> Any:
        if np.ndim(other) == 0:
            return self._with(self.values * float(other))
        return self._with(self.values @ values_of(other))

    def __rmul__(self, other: Any) -> Any:
        return self._with(float(other) * self.values)

    def __add__(self, other: Any) -> Any:
        return self._with(self.values + values_of(other))

    def __sub__(self, other: Any) -> Any:
        return self._with(self.values - values_of(other))

    def __neg__(self) -> Any:
        return self._with(-self.values)

    def __imul__(self, other: Any) -> TMatrixT:
        changed = self.values * float(other) if np.ndim(other) == 0 else self.values @ values_of(
            other)  # fmt: skip
        return self._became(changed)

    def __iadd__(self, other: Any) -> TMatrixT:
        return self._became(self.values + values_of(other))

    def __isub__(self, other: Any) -> TMatrixT:
        return self._became(self.values - values_of(other))

    def __itruediv__(self, other: Any) -> TMatrixT:
        return self._became(self.values / float(other))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TMatrixT) and np.array_equal(self.values, other.values)

    __hash__ = None  # type: ignore[assignment]


class TMatrixTSym(TMatrixT):
    """``TMatrixTSym<double>``: a symmetric matrix, ``n`` by ``n``."""

    SYMMETRIC = True
    CPP = "TMatrixTSym<double>"


#: The template arguments that name ``Float_t``: ``TMatrixT<float>`` is ``TMatrixF``.
_FLOAT_KINDS = frozenset({"float", "Float_t", "float32", "np.float32"})


class TMatrixTF(TMatrixT):
    """``TMatrixT<Float_t>``: the matrix whose elements are ``Float_t``, kept as 32-bit floats.

    ROOT's ``Real_t`` and ``Float_t`` are the same 32-bit float, and a matrix of them stores
    what fits in one - ``m(i, j) = x`` rounds ``x`` on the way in, and so does an element
    action's ``element = ...``, which is what makes the two agree.
    """

    DTYPE = np.dtype(np.float32)
    CPP = "TMatrixT<float>"


class TMatrixTFSym(TMatrixTSym):
    """``TMatrixTSym<Float_t>``: the symmetric matrix of 32-bit floats."""

    DTYPE = np.dtype(np.float32)
    CPP = "TMatrixTSym<float>"


#: ``TMatrixD`` and ``TMatrixF``, and their symmetric kinds: ``F`` keeps ``Float_t``.
TMatrixD = TMatrixT
TMatrixF = TMatrixTF
TMatrixDSym = TMatrixTSym
TMatrixFSym = TMatrixTFSym
TMatrixT.FLOAT = TMatrixTF
TMatrixTSym.FLOAT = TMatrixTFSym

#: ``TMatrix``, ROOT's ``TMatrixT<Float_t>``: the default element type of ``TMatrix.h``.
TMatrix = TMatrixTF

#: ``TMatrixTLazy<Double_t>`` and its kind: one promise, of the one number Python has.
TMatrixDLazy = TMatrixTLazy
TMatrixFLazy = TMatrixTLazy
TMatrixDSymLazy = TMatrixTSymLazy
TMatrixFSymLazy = TMatrixTSymLazy


def haar_values(no_rows: int, no_cols: int, kind: Any = FLOAT64) -> np.ndarray[Any, Any]:
    """The ``2**order`` by ``no_cols`` Haar (sub)matrix, as ``MakeHaarMat`` builds it.

    The columns are Haar's functions: the first is the constant, and each family after it a
    step of one width, half as wide as the family before. They are orthogonal and each of
    unit length, which is what the test of them checks to 1e-15.
    """
    if no_rows < no_cols or no_cols <= 0:
        raise ValueError(f"a Haar matrix of {no_rows} rows cannot have {no_cols} columns")
    transposed = np.zeros((no_cols, no_rows), dtype=kind)
    element = kind.type
    norm = element(1.0 / np.sqrt(no_rows))
    transposed[0, :] = norm
    written = 1
    step = no_rows // 2
    while written < no_cols and step > 0:
        for position in range(0, no_rows, 2 * step):
            if written == no_cols:
                break
            transposed[written, position: position + step] = norm
            transposed[written, position + step: position + 2 * step] = -norm
            written += 1
        step //= 2
        norm = element(norm * np.sqrt(2.0))
    return transposed.T.copy()


class THaarMatrixD(TMatrixTLazy):
    """``THaarMatrixD(order[, no_cols])``: the Haar matrix of ``2**order`` rows, promised.

    The matrix is made when one is built from this, as ROOT's lazy matrix is, so that a macro
    that only passes it on pays nothing for it.
    """

    #: The element type the Haar matrix is made in, ROOT's ``Element``.
    DTYPE: Any = FLOAT64

    def __init__(self, order: Any, no_cols: Any = 0) -> None:
        rows = 1 << int(order)
        super().__init__(rows, int(no_cols) if int(no_cols) else rows)
        self._rows, self._cols = rows, int(no_cols) if int(no_cols) else rows

    def FillIn(self, matrix: Any) -> None:
        matrix._became(haar_values(self._rows, self._cols, self.DTYPE))


class THaarMatrixF(THaarMatrixD):
    """``THaarMatrixF``: the same Haar matrix of ``Float_t``."""

    DTYPE = np.dtype(np.float32)


THaarMatrixF.__name__ = "THaarMatrixF"


def _hilbert(rows: int, cols: int) -> np.ndarray[Any, Any]:
    """Hilbert's matrix: ``1 / (i + j + 1)``."""
    i, j = np.indices((rows, cols))
    return 1.0 / (i + j + 1.0)


class THilbertMatrixD(TMatrixT):
    """``THilbertMatrixD(nrows, ncols)``: Hilbert's matrix, the classic ill-conditioned one."""

    def __init__(self, nrows: Any, ncols: Any = None) -> None:
        super().__init__(_hilbert(int(nrows), int(nrows if ncols is None else ncols)))


THilbertMatrixF = THilbertMatrixD


class THilbertMatrixDSym(TMatrixTSym):
    """``THilbertMatrixDSym(n)``: the symmetric Hilbert matrix."""

    def __init__(self, n: Any) -> None:
        super().__init__(_hilbert(int(n), int(n)))
