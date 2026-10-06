"""ROOT's linear algebra: dense matrices, views of them, decompositions, and the checks on them.

``TMatrixD``, ``TMatrixDSym`` and ``THilbertMatrixD`` (:mod:`.matrices`), the
row, column and diagonal views (:mod:`.views`), and ``TDecompLU``,
``TDecompChol``, ``TDecompBK``, ``TDecompSVD`` and ``TDecompQRH``
(:mod:`.decompositions`), all over NumPy. ``VerifyVectorIdentity`` and
``VerifyMatrixIdentity`` say whether two agree to within a deviation, and
with ``verbose`` print the largest, as ROOT's do.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.messages import message
from .decompositions import NormalEqn, TDecompBK, TDecompChol, TDecompLU, TDecompQRH, TDecompSVD
from .matrices import (
    THilbertMatrixD,
    THilbertMatrixDSym,
    THilbertMatrixF,
    TMatrixD,
    TMatrixDSym,
    TMatrixF,
    TMatrixFSym,
    TMatrixT,
    TMatrixTSym,
    values_of,
)
from .views import (
    TMatrixDColumn,
    TMatrixDColumn_const,
    TMatrixDDiag,
    TMatrixDDiag_const,
    TMatrixDRow,
    TMatrixDRow_const,
    TMatrixFColumn,
    TMatrixFDiag,
    TMatrixFRow,
    TMatrixTColumn,
    TMatrixTDiag,
    TMatrixTRow,
)

__all__ = [
    "TMatrixT", "TMatrixTSym", "TMatrixD", "TMatrixF", "TMatrixDSym", "TMatrixFSym",
    "THilbertMatrixD", "THilbertMatrixDSym", "THilbertMatrixF", "TMatrixDRow", "TMatrixDColumn",
    "TMatrixDDiag", "TMatrixFRow", "TMatrixFColumn", "TMatrixFDiag", "TMatrixDRow_const",
    "TMatrixDColumn_const", "TMatrixDDiag_const", "TMatrixTRow", "TMatrixTColumn",
    "TMatrixTDiag", "TDecompLU", "TDecompChol", "TDecompBK", "TDecompSVD", "TDecompQRH",
    "NormalEqn", "VerifyVectorIdentity", "VerifyMatrixIdentity",
]  # fmt: skip


def _verified(where: str, one: Any, two: Any, verbose: Any, allowed: Any) -> bool:
    """Whether two agree element by element to within ``allowed``; with ``verbose``, the worst
    deviation printed, and an error if it is too large."""
    first, second = values_of(one), values_of(two)
    if first.shape != second.shape:
        message("Error", where, "objects have different shapes")
        return False
    deviation = np.abs(first - second)
    worst = np.unravel_index(int(np.argmax(deviation)), deviation.shape) if deviation.size else ()
    largest = float(deviation[worst]) if deviation.size else 0.0
    if int(verbose):
        at = ",".join(str(int(i)) for i in worst)
        print(f"Largest dev for ({at}); dev = |{first[worst]:g} - {second[worst]:g}| = {largest:g}")
        if largest > float(allowed):
            message("Error", where, "Deviation > %g", float(allowed))
    return largest <= float(allowed)


def VerifyVectorIdentity(one: Any, two: Any, verbose: Any = 1, allowed: Any = 0.0) -> bool:
    """``VerifyVectorIdentity(v1, v2, verbose, maxDevAllow)``: do they agree to within it?"""
    return _verified("VerifyVectorIdentity", one, two, verbose, allowed)


def VerifyMatrixIdentity(one: Any, two: Any, verbose: Any = 1, allowed: Any = 0.0) -> bool:
    """``VerifyMatrixIdentity(m1, m2, verbose, maxDevAllow)``: do they agree to within it?"""
    return _verified("VerifyMatrixIdentity", one, two, verbose, allowed)
