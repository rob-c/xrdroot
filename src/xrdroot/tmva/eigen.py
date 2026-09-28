"""``TMatrixDSymEigen``: the eigenvectors of a symmetric matrix, as ROOT finds them.

An eigenvector is only defined up to its sign, and which sign comes out
depends on the algorithm. ROOT's is JAMA's - Householder reduction to a
tridiagonal matrix (``tred2``), then the QL algorithm with implicit shifts
(``tql2``), then the eigenvalues sorted from the largest down - and TMVA's
principal components are the data projected on those vectors, so a
component's sign is the one ROOT's algorithm gives it. This is that
algorithm, line for line, so that ``VarTransform=PCA`` gives the values
TMVA gives rather than their mirror images.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["inverse_square_root", "symmetric"]

#: ``TMath::Power(2.0, -52.0)``, the QL iteration's tolerance.
EPS = 2.0**-52


def _reduce_row(v: Any, d: Any, e: Any, i: int, scale: float) -> float:
    """One step of the Householder reduction, for row ``i``; its ``h``."""
    d[:i] /= scale
    h = float(np.dot(d[:i], d[:i]))
    f = d[i - 1]
    g = -math.sqrt(h) if f > 0 else math.sqrt(h)
    e[i] = scale * g
    h -= f * g
    d[i - 1] = f - g
    e[:i] = 0.0
    for j in range(i):
        f = d[j]
        v[j, i] = f
        g = e[j] + v[j, j] * f
        g += float(np.dot(v[j + 1 : i, j], d[j + 1 : i]))
        e[j + 1 : i] += v[j + 1 : i, j] * f
        e[j] = g
    e[:i] /= h
    hh = float(np.dot(e[:i], d[:i])) / (h + h)
    e[:i] -= hh * d[:i]
    for j in range(i):
        v[j:i, j] -= d[j] * e[j:i] + e[j] * d[j:i]
        d[j] = v[i - 1, j]
        v[i, j] = 0.0
    return h


def _accumulate(v: Any, d: Any, e: Any, n: int) -> None:
    """The Householder transformations gathered into the eigenvector matrix."""
    for i in range(n - 1):
        v[n - 1, i] = v[i, i]
        v[i, i] = 1.0
        h = d[i + 1]
        if h != 0.0:
            d[: i + 1] = v[: i + 1, i + 1] / h
            for j in range(i + 1):
                g = float(np.dot(v[: i + 1, i + 1], v[: i + 1, j]))
                v[: i + 1, j] -= g * d[: i + 1]
        v[: i + 1, i + 1] = 0.0
    d[:] = v[n - 1, :]
    v[n - 1, :] = 0.0
    v[n - 1, n - 1] = 1.0
    e[0] = 0.0


def _tridiagonal(v: Any, d: Any, e: Any) -> None:
    """``MakeTridiagonal``: JAMA's ``tred2``."""
    n = len(d)
    d[:] = v[n - 1, :]
    for i in range(n - 1, 0, -1):
        scale = float(np.sum(np.abs(d[:i])))
        h = 0.0
        if scale == 0.0:
            e[i] = d[i - 1]
            d[:i] = v[i - 1, :i]
            v[i, :i] = 0.0
            v[:i, i] = 0.0
        else:
            h = _reduce_row(v, d, e, i, scale)
        d[i] = h
    _accumulate(v, d, e, n)


def _rotate(v: Any, d: Any, e: Any, lo: int, m: int) -> float:
    """One implicit QL step on the block from ``lo`` to ``m``; the shift it made."""
    g = d[lo]
    p = (d[lo + 1] - g) / (2.0 * e[lo])
    r = math.hypot(p, 1.0)
    if p < 0:
        r = -r
    d[lo] = e[lo] / (p + r)
    d[lo + 1] = e[lo] * (p + r)
    dl1 = d[lo + 1]
    h = g - d[lo]
    d[lo + 2 :] -= h
    shift = h
    p = d[m]
    c = c2 = c3 = 1.0
    el1 = e[lo + 1]
    s = s2 = 0.0
    for i in range(m - 1, lo - 1, -1):
        c3, c2, s2 = c2, c, s
        g = c * e[i]
        h = c * p
        r = math.hypot(p, e[i])
        e[i + 1] = s * r
        s, c = e[i] / r, p / r
        p = c * d[i] - s * g
        d[i + 1] = h + s * (c * g + s * d[i])
        column = v[:, i + 1].copy()
        v[:, i + 1] = s * v[:, i] + c * column
        v[:, i] = c * v[:, i] - s * column
    p = -s * s2 * c3 * el1 * e[lo] / dl1
    e[lo], d[lo] = s * p, c * p
    return float(shift)


def _ql(v: Any, d: Any, e: Any) -> None:
    """``MakeEigenVectors``: JAMA's ``tql2``, then the eigenvalues sorted, the largest first."""
    n = len(d)
    e[:-1] = e[1:].copy()
    e[-1] = 0.0
    f = tst1 = 0.0
    for lo in range(n):
        tst1 = max(tst1, abs(d[lo]) + abs(e[lo]))
        m = lo
        while m < n - 1 and abs(e[m]) > EPS * tst1:
            m += 1
        if m > lo:
            for _ in range(30):
                f += _rotate(v, d, e, lo, m)
                if abs(e[lo]) <= EPS * tst1:
                    break
        d[lo] += f
        e[lo] = 0.0
    _sort(v, d)


def _sort(v: Any, d: Any) -> None:
    """The eigenvalues in decreasing order, their vectors swapped with them, as ROOT sorts."""
    for i in range(len(d) - 1):
        k = i + int(np.argmax(d[i:]))
        if d[k] > d[i]:
            d[i], d[k] = d[k], d[i]
            v[:, [i, k]] = v[:, [k, i]]


def symmetric(matrix: Any) -> tuple[Any, Any]:
    """The eigenvalues, largest first, and the eigenvectors as columns, as ROOT gives them."""
    v = np.array(matrix, dtype=np.float64)
    n = len(v)
    d, e = np.zeros(n), np.zeros(n)
    if n:
        _tridiagonal(v, d, e)
        _ql(v, d, e)
    return d, v


def inverse_square_root(matrix: Any) -> Any:
    """``Tools::GetSQRootMatrix``: the inverse of the matrix's square root."""
    values, vectors = symmetric(matrix)
    root = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    return np.linalg.inv(root)
