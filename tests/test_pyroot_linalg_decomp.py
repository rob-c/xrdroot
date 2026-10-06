"""ROOT's decompositions, matrix views, vectors and arrays, by ROOT's names."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.cint.runtime import Cell

A = [[4.0, 2.0, 0.6], [2.0, 5.0, 1.0], [0.6, 1.0, 3.0]]


def _matrix() -> object:
    return ROOT.TMatrixD(np.array(A))


@pytest.mark.parametrize("kind", ["TDecompLU", "TDecompChol", "TDecompBK", "TDecompSVD",
                                  "TDecompQRH"])  # fmt: skip
def test_every_decomposition_solves_inverts_and_gives_the_determinant(kind: str) -> None:
    decomp = getattr(ROOT, kind)(_matrix())
    assert decomp.Decompose() and (decomp.GetNrows(), decomp.GetNcols()) == (3, 3)
    b = ROOT.TVectorD([1.0, 2.0, 3.0])
    ok = Cell(False)
    solution = decomp.Solve(b, ok)
    assert ok.value and np.allclose(np.array(A) @ np.asarray(solution), [1.0, 2.0, 3.0])
    assert decomp.Solve(b) is True and np.allclose(np.asarray(b), np.asarray(solution))
    inverse = ROOT.TMatrixD(3, 3)
    assert decomp.Invert(inverse) and np.allclose(inverse.values @ np.array(A), np.eye(3))
    assert np.allclose(decomp.Invert().values, inverse.values)
    d1, d2 = Cell(0.0), Cell(0.0)
    assert decomp.Det(d1, d2) == pytest.approx(np.linalg.det(A))
    assert 0.5 <= d1.value < 1 and decomp.Condition() == decomp.GetCondition() > 1


def test_a_decomposition_of_what_has_none_says_so() -> None:
    singular = ROOT.TDecompLU(ROOT.TMatrixD(2, 2))
    assert not singular.Decompose()
    ok = Cell(True)
    assert list(singular.Solve(ROOT.TVectorD([1.0, 1.0]), ok)) == [0.0, 0.0] and not ok.value
    assert singular.Solve(ROOT.TVectorD([1.0, 1.0])) is False
    with pytest.raises(np.linalg.LinAlgError, match="cannot be decomposed"):
        singular.Invert()
    with pytest.raises(np.linalg.LinAlgError):
        ROOT.TDecompLU(ROOT.TMatrixD(2, 2, [0.0, 1.0, 0.0, 1.0])).GetLU()
    empty = ROOT.TDecompLU()
    empty.SetMatrix(_matrix())
    assert empty.GetNrows() == 3


def test_the_factors_are_the_ones_roots_decompositions_keep() -> None:
    lu = ROOT.TDecompLU(_matrix()).GetLU()
    lower, upper = np.tril(lu.values, -1) + np.eye(3), np.triu(lu.values)
    assert np.allclose(lower @ upper, np.array(A)[[0, 1, 2]])  # no row swapped for this one
    u = ROOT.TDecompChol(_matrix()).GetU()
    assert np.allclose(u.values.T @ u.values, A)
    svd = ROOT.TDecompSVD(ROOT.THilbertMatrixD(4, 3))
    sig, left, right = svd.GetSig(), svd.GetU(), svd.GetV()
    assert left.GetNrows() == 4 and right.GetNrows() == 3 and len(sig) == 3
    assert np.allclose(svd.Invert().values @ ROOT.THilbertMatrixD(4, 3).values, np.eye(3))
    qr = ROOT.TDecompQRH(ROOT.TMatrixD(3, 3, [12, -51, 4, 6, 167, -68, -4, 24, -41]))
    assert qr.GetR()(0, 0) == pytest.approx(-14.0) and qr.GetQ()(0, 0) == pytest.approx(-6 / 7)
    assert type(ROOT.TDecompChol(_matrix()).Invert()) is ROOT.TMatrixDSym


def test_a_view_reads_and_writes_its_part_of_the_matrix() -> None:
    m = ROOT.TMatrixD(2, 3)
    column, row, diag = ROOT.TMatrixDColumn(m, 0), ROOT.TMatrixDRow(m, 1), ROOT.TMatrixDDiag(m)
    column._assign(1.0)
    row._assign(ROOT.TVectorD([4.0, 5.0, 6.0]))
    assert m.values.tolist() == [[1.0, 0.0, 0.0], [4.0, 5.0, 6.0]]
    diag._assign(0.0)
    assert (m(0, 0), m(1, 1), len(diag), diag.GetNdim()) == (0.0, 0.0, 2, 2)
    row[2] = 9.0
    row.__setcall__(0, 7.0)
    assert (row(2), row[0], row.Sum()) == (9.0, 7.0, 16.0)
    row *= 2.0
    row += [1.0, 1.0, 1.0]
    assert list(row * 0.5) == [7.5, 0.5, 9.5] and list(0.5 * row) == [7.5, 0.5, 9.5]
    assert list(row + 1) == [16.0, 2.0, 20.0] and np.asarray(column).tolist() == [0.0, 15.0]


def test_identities_are_checked_to_within_a_deviation(capsys) -> None:
    one, two = ROOT.TVectorD([1.0, 2.0]), ROOT.TVectorD([1.0, 2.5])
    assert ROOT.VerifyVectorIdentity(one, one, 0, 0.0)
    assert not ROOT.VerifyVectorIdentity(one, two, 1, 0.1)
    assert ROOT.VerifyMatrixIdentity(ROOT.TMatrixD(2, 2), ROOT.TMatrixD(2, 2), 1)
    assert not ROOT.VerifyMatrixIdentity(ROOT.TMatrixD(2, 2), ROOT.TMatrixD(1, 2), 0)
    assert ROOT.VerifyVectorIdentity(ROOT.TVectorD(), ROOT.TVectorD(), 0)
    captured = capsys.readouterr()
    assert "Largest dev for (1); dev = |2 - 2.5| = 0.5" in captured.out
    assert "Error in <VerifyVectorIdentity>: Deviation > 0.1" in captured.err


def test_the_normal_equations_fit_a_line_with_and_without_errors() -> None:
    design = ROOT.TMatrixD(np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]))
    y = ROOT.TVectorD([1.0, 3.0, 5.0])
    assert np.allclose(np.asarray(ROOT.NormalEqn(design, y)), [1.0, 2.0])
    weighted = ROOT.NormalEqn(design, y, ROOT.TVectorD([1.0, 1.0, 1.0]))
    assert np.allclose(np.asarray(weighted), [1.0, 2.0])


def test_a_vector_uses_an_array_scales_and_is_multiplied_by_a_matrix() -> None:
    data = np.array([1.0, -2.0, 3.0])
    v = ROOT.TVectorD().Use(3, data)
    data[0] = 5.0
    assert v[0] == 5.0 and ROOT.TVectorD().Use(1, 2, data).GetLwb() == 1
    v._assign(0.5)
    assert list(v) == [0.5, 0.5, 0.5]
    v._assign([1.0, -2.0])
    assert (v.Max(), v.Min(), v.Norm1(), v.NormInf()) == (1.0, -2.0, 3.0, 2.0)
    assert list(v.Abs()) == [1.0, 2.0] and list(v.ResizeTo(3)) == [1.0, 2.0, 0.0]
    assert list(v.Zero()) == [0.0, 0.0, 0.0]



def test_a_vector_is_multiplied_by_a_matrix_and_by_numbers() -> None:
    w = ROOT.TVectorD([1.0, 2.0])
    w *= ROOT.TMatrixD(np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
    assert list(w) == [2.0, 1.0, 3.0]
    w *= 2
    assert list(w) == [4.0, 2.0, 6.0] and w * w == 56.0 and list(w * 0.5) == [2.0, 1.0, 3.0]
    assert list(0.5 * w) == [2.0, 1.0, 3.0] and list(w + w - w) == [4.0, 2.0, 6.0]
    assert list(-w) == [-4.0, -2.0, -6.0]


def test_every_array_kind_holds_its_type_and_is_iterated_and_set() -> None:
    floats = ROOT.TArrayF(3)
    floats[1] = 2.5
    floats.AddAt(1.5, 2)
    assert list(floats) == [0.0, 2.5, 1.5] and floats.GetSum() == 4.0
    assert floats.GetArray().dtype == np.float32 and ROOT.TArrayI([1, 2]).At(1) == 2
    shorts, longs, chars = ROOT.TArrayS(2), ROOT.TArrayL(1), ROOT.TArrayC(1)
    assert (shorts.GetArray().dtype, longs.GetArray().dtype, chars.GetArray().dtype) == (
        np.int16, np.int64, np.int8)  # fmt: skip
    floats.Set(4)
    assert list(floats) == [0.0, 2.5, 1.5, 0.0]
    floats.Set(2, [7.0, 8.0])
    floats.Reset(1.0)
    assert list(floats) == [1.0, 1.0] and len(floats) == 2
    assert np.array(floats, copy=True) is not floats.GetArray()
