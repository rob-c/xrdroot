"""ROOT's dense matrices by ROOT's names: made, indexed, changed in place, multiplied, printed."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.cint.runtime import Cell


def test_a_matrix_is_made_by_size_bounds_elements_copy_or_rows() -> None:
    assert ROOT.TMatrixD().GetNrows() == 0
    wide = ROOT.TMatrixD(2, 3)
    assert (wide.GetNrows(), wide.GetNcols(), wide.GetNoElements()) == (2, 3, 6)
    bounded = ROOT.TMatrixD(1, 2, 1, 3)
    assert (bounded.GetRowLwb(), bounded.GetRowUpb(), bounded.GetColLwb(), bounded.GetColUpb()) == (
        1, 2, 1, 3)  # fmt: skip
    bounded.__setcall__(2, 3, 5.0)
    assert bounded(2, 3) == 5.0 and bounded[2][2] == 5.0 and bounded[2, 3] == 5.0
    rows = ROOT.TMatrixT["double"](2, 2, np.array([1.0, 2.0, 3.0, 4.0]))
    by_column = ROOT.TMatrixD(2, 2, [1.0, 2.0, 3.0, 4.0], "F")
    assert rows(0, 1) == 2.0 and by_column(0, 1) == 3.0
    assert ROOT.TMatrixD(1, 2, 0, 1, [7.0, 8.0, 9.0, 10.0])(2, 1) == 10.0
    copied = ROOT.TMatrixD(rows)
    copied[0, 0] = 9.0
    assert rows(0, 0) == 1.0 and ROOT.TMatrixD([[1, 2]]).GetNcols() == 2
    assert "TMatrixT<double> 2x2" in repr(rows) and rows.ClassName() == "TMatrixT<double>"


def test_a_matrix_is_made_by_an_operation_on_two_others() -> None:
    a = ROOT.TMatrixD(2, 2, [1.0, 2.0, 3.0, 4.0])
    b = ROOT.TMatrixD(2, 2, [0.0, 1.0, 1.0, 0.0])
    expected = [a.values @ b.values, a.values.T @ b.values, np.linalg.inv(a.values) @ b.values,
                a.values @ b.values.T, a.values + b.values, a.values - b.values]  # fmt: skip
    for op, wanted in enumerate(expected):
        assert np.allclose(ROOT.TMatrixD(a, op, b).values, wanted)
    assert ROOT.TMatrixD.kMult == 0 and ROOT.TMatrixD.kMinus == 5


def test_a_matrix_is_made_by_an_operation_on_one_other() -> None:
    a = ROOT.TMatrixD(2, 2, [1.0, 2.0, 3.0, 4.0])
    expected = [np.zeros((2, 2)), np.eye(2), a.values.T, np.linalg.inv(a.values),
                a.values.T @ a.values]  # fmt: skip
    for op, wanted in enumerate(expected):
        assert np.allclose(ROOT.TMatrixD(op, a).values, wanted)
    sym = ROOT.TMatrixDSym(ROOT.TMatrixDSym.kAtA, a)
    assert sym.ClassName() == "TMatrixTSym<double>" and sym.IsSymmetric()


def test_a_matrix_changes_in_place_and_gives_itself_back() -> None:
    a = ROOT.TMatrixD(2, 2, [4.0, 7.0, 2.0, 6.0])
    det = Cell(0.0)
    assert a.Invert(det) is a and det.value == pytest.approx(10.0)
    assert np.allclose(a.values, np.linalg.inv([[4.0, 7.0], [2.0, 6.0]]))
    back = ROOT.TMatrixD(2, 2, [1.0, -2.0, 3.0, -4.0])
    assert back.T()(0, 1) == 3.0 and back.Transpose(ROOT.TMatrixD(2, 2, [1.0] * 4))(1, 1) == 1.0
    signed = ROOT.TMatrixD(1, 2, [-4.0, 9.0])
    assert signed.Abs().Sqrt().Sqr().Max() == pytest.approx(9.0)
    assert signed.Zero().Sum() == 0.0 and signed.ResizeTo(2, 2).GetNrows() == 2
    assert ROOT.TMatrixD(2, 2).UnitMatrix().IsSymmetric()
    assert ROOT.TMatrixD(1, 1).ResizeTo(ROOT.TMatrixD(1, 3)).GetNcols() == 3
    assert not ROOT.TMatrixD(1, 2).IsSymmetric() and ROOT.TMatrixD(1, 1).InvertFast is not None


def test_a_matrix_says_what_it_is() -> None:
    a = ROOT.TMatrixD(2, 2, [1.0, -2.0, 3.0, 4.0])
    d1, d2 = Cell(0.0), Cell(0.0)
    assert a.Determinant(d1, d2) == pytest.approx(10.0)
    assert d1.value * 2**d2.value == pytest.approx(10.0)
    assert (a.Min(), a.E2Norm(), a.NormInf(), a.Norm1()) == (-2.0, 30.0, 7.0, 6.0)
    assert len(a) == 2 and list(a.GetMatrixArray()) == [1.0, -2.0, 3.0, 4.0]


def test_a_matrix_does_arithmetic_with_numbers_vectors_and_matrices() -> None:
    a = ROOT.TMatrixD(2, 2, [1.0, -2.0, 3.0, 4.0])
    v = a * ROOT.TVectorD([1.0, 1.0])
    assert isinstance(v, ROOT.TVectorD) and list(v) == [-1.0, 7.0]
    assert (2 * a)(1, 1) == 8.0 and (a * 2)(0, 0) == 2.0 and (a + a - a)(0, 1) == -2.0
    assert (-a)(0, 0) == -1.0 and (a * a).ClassName() == "TMatrixT<double>"
    sym = ROOT.TMatrixDSym(2, [2.0, 1.0, 1.0, 2.0])
    assert type(sym + sym) is ROOT.TMatrixDSym and ROOT.TMatrixT["float"] is ROOT.TMatrixD


def test_a_matrix_changes_by_arithmetic_in_place() -> None:
    a = ROOT.TMatrixD(2, 2, [1.0, -2.0, 3.0, 4.0])
    a *= 2
    a += ROOT.TMatrixD(2, 2)
    a -= ROOT.TMatrixD(2, 2)
    a /= 2
    a *= ROOT.TMatrixD(2, 2).UnitMatrix()
    assert a == ROOT.TMatrixD(2, 2, [1.0, -2.0, 3.0, 4.0]) and a != "a"


def test_a_matrix_is_set_whole_printed_and_given_to_numpy(capsys) -> None:
    a = ROOT.TMatrixD(2, 2)
    a._assign(3.0)
    assert a.Sum() == 12.0
    a._assign(ROOT.THilbertMatrixD(2, 2))
    assert a(1, 1) == pytest.approx(1 / 3)
    assert ROOT.THilbertMatrixDSym(3)(2, 2) == pytest.approx(0.2)
    a[1] = [5.0, 6.0]
    assert list(a[1]) == [5.0, 6.0] and np.asarray(a, dtype=np.float32).dtype == np.float32
    assert np.array(a) is not a.values and a.matrix() is a.values
    a.Print()
    a.Print("f=%5.1f ")
    out = capsys.readouterr().out
    assert "2x2 matrix is as follows" in out and "  5.0" in out
