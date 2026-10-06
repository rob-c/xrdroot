"""ROOT's matrix decompositions - LU, Cholesky, Bunch-Kaufman, SVD, Householder QR - on NumPy.

``TDecompLU lu(A)`` decomposes ``A``; ``lu.Solve(b)`` solves ``A x = b`` in
place, ``lu.Solve(b, ok)`` hands the solution back and says in ``ok``
whether there was one, ``lu.Invert()`` is the inverse and ``lu.Det(d1,
d2)`` the determinant as ``d1 * 2**d2``. The factors are LAPACK's, through
NumPy; the singular values and the QR factors have the signs LAPACK gives,
which are ROOT's. An SVD of a matrix with more rows than columns solves in
the least-squares sense, and its inverse is the pseudo-inverse.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.objects import TObject
from ..vectors import TVectorD
from .matrices import TMatrixD, TMatrixDSym, TMatrixT, hand_back, values_of

__all__ = ["TDecompLU", "TDecompChol", "TDecompBK", "TDecompSVD", "TDecompQRH", "NormalEqn"]


class _Decomp(TObject):
    """What every decomposition does: decompose, solve, invert, and give the determinant."""

    def __init__(self, matrix: Any = None, tol: Any = None) -> None:
        super().__init__()
        self._a = np.zeros((0, 0)) if matrix is None else values_of(matrix).copy()
        self._tol = 2.2e-16 if tol is None else float(tol)
        self._done = False

    def SetMatrix(self, matrix: Any) -> None:
        self._a, self._done = values_of(matrix).copy(), False

    def GetNrows(self) -> int:
        return int(self._a.shape[0])

    def GetNcols(self) -> int:
        return int(self._a.shape[1])

    def Decompose(self) -> bool:
        """``Decompose()``: the factors made - ``false`` for a matrix that has none."""
        try:
            self._factor()
        except np.linalg.LinAlgError:
            return False
        self._done = True
        return True

    def _ready(self) -> None:
        if not self._done and not self.Decompose():
            raise np.linalg.LinAlgError(f"{type(self).__name__}: the matrix cannot be decomposed")

    def _factor(self) -> None:
        np.linalg.inv(self._a)  # a square matrix with an inverse decomposes

    def _solved(self, b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.linalg.solve(self._a, b)

    def _inverse(self) -> np.ndarray[Any, Any]:
        return np.linalg.inv(self._a)

    def _det(self) -> float:
        return float(np.linalg.det(self._a))

    def Solve(self, b: Any, ok: Any = None) -> Any:
        """``Solve(b)``: ``b`` made the solution, true if there was one; ``Solve(b, ok)``: the
        solution, ``b`` left as it was, and ``ok`` whether there was one."""
        try:
            self._ready()
            found = self._solved(values_of(b))
        except np.linalg.LinAlgError:
            hand_back(ok, False)
            return TVectorD(np.zeros(len(values_of(b)))) if ok is not None else False
        hand_back(ok, True)
        if ok is not None:
            return TVectorD(found)
        b._assign(found)
        return True

    def Invert(self, inverse: Any = None) -> Any:
        """``Invert()``: the inverse; ``Invert(m)``: ``m`` made the inverse, true if it was."""
        self._ready()
        found = self._inverse()
        if inverse is None:
            return self._made(found)
        inverse._assign(found)
        return True

    def _made(self, values: np.ndarray[Any, Any]) -> TMatrixT:
        return TMatrixD(values)

    def Det(self, d1: Any = None, d2: Any = None) -> float:
        """``Det(d1, d2)``: the determinant as ``d1 * 2**d2``, ``d1`` in ``[0.5, 1)``."""
        self._ready()
        found = self._det()
        mantissa, exponent = np.frexp(found)
        hand_back(d1, float(mantissa))
        hand_back(d2, float(exponent))
        return found

    def Condition(self) -> float:
        """The condition number in the 1-norm, which ROOT's ``GetCondition`` estimates."""
        return float(np.linalg.cond(self._a, 1))

    GetCondition = Condition


class TDecompLU(_Decomp):
    """``TDecompLU``: ``P A = L U`` with partial pivoting, the factors kept as ROOT keeps them -
    ``L`` below the diagonal, its ones understood, and ``U`` on and above it."""

    def _factor(self) -> None:
        lu = self._a.copy()
        size = lu.shape[0]
        for k in range(size):
            pivot = k + int(np.argmax(np.abs(lu[k:, k])))
            if lu[pivot, k] == 0:
                raise np.linalg.LinAlgError("TDecompLU: the matrix is singular")
            lu[[k, pivot]] = lu[[pivot, k]]
            lu[k + 1:, k] /= lu[k, k]
            lu[k + 1:, k + 1:] -= np.outer(lu[k + 1:, k], lu[k, k + 1:])
        self._lu = lu

    def GetLU(self) -> TMatrixT:
        self._ready()
        return TMatrixD(self._lu)


class TDecompChol(_Decomp):
    """``TDecompChol``: ``A = U^T U`` of a symmetric positive-definite matrix."""

    def _factor(self) -> None:
        self._u = np.linalg.cholesky(self._a).T

    def GetU(self) -> TMatrixT:
        self._ready()
        return TMatrixD(self._u)

    def _made(self, values: np.ndarray[Any, Any]) -> TMatrixT:
        return TMatrixDSym(values)


class TDecompBK(_Decomp):
    """``TDecompBK``: Bunch-Kaufman's decomposition of a symmetric, perhaps indefinite, matrix."""

    def _made(self, values: np.ndarray[Any, Any]) -> TMatrixT:
        return TMatrixDSym(values)


class TDecompSVD(_Decomp):
    """``TDecompSVD``: ``A = U S V^T`` of a matrix with at least as many rows as columns."""

    def _factor(self) -> None:
        self._u, self._sig, vt = np.linalg.svd(self._a)
        self._v = vt.T

    def _solved(self, b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return self._inverse() @ b

    def _inverse(self) -> np.ndarray[Any, Any]:
        """The pseudo-inverse, ``V S^-1 U^T``, the inverse of a square matrix."""
        count = len(self._sig)
        return (self._v / self._sig) @ self._u[:, :count].T

    def _det(self) -> float:
        """The product of the singular values, as ROOT's ``TDecompSVD::Det`` takes it."""
        return float(np.prod(self._sig))

    def GetU(self) -> TMatrixT:
        self._ready()
        return TMatrixD(self._u)

    def GetV(self) -> TMatrixT:
        self._ready()
        return TMatrixD(self._v)

    def GetSig(self) -> TVectorD:
        self._ready()
        return TVectorD(self._sig)


class TDecompQRH(_Decomp):
    """``TDecompQRH``: ``A = Q R`` by Householder reflections, ``R`` upper triangular."""

    def _factor(self) -> None:
        self._q, self._r = np.linalg.qr(self._a)

    def _solved(self, b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.linalg.solve(self._r, self._q.T @ b)

    def _inverse(self) -> np.ndarray[Any, Any]:
        return np.linalg.solve(self._r, self._q.T)

    def _det(self) -> float:
        return float(np.prod(np.diag(self._r)) * np.linalg.det(self._q))

    def GetOrthogonalMatrix(self) -> TMatrixT:
        self._ready()
        return TMatrixD(self._q)

    GetQ = GetOrthogonalMatrix

    def GetR(self) -> TMatrixT:
        self._ready()
        return TMatrixD(self._r)


def NormalEqn(a: Any, b: Any, std: Any = None) -> TVectorD:
    """``NormalEqn(A, b[, sigma])``: the ``x`` that minimises ``|W (A x - b)|``, ``W`` the
    inverse errors."""
    design, target = values_of(a), values_of(b)
    weights = np.ones(len(target)) if std is None else 1.0 / values_of(std) ** 2
    normal = design.T @ (weights[:, None] * design)
    return TVectorD(np.linalg.solve(normal, design.T @ (weights * target)))
