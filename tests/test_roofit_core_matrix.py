"""``TMatrixDSym`` and ``TVectorD``: a fit result's matrices, read and printed as ROOT's are.

The printed layouts came from ROOT 6.40's ``TMatrixDSym::Print()`` on the
same matrices: five ``%11.4g`` columns a sheet under a numbered bar.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.roofit.matrix import TMatrixDSym, TVectorD, matrix_text

SMALL = (
    "\n2x2 matrix is as follows\n"
    "\n"
    "     |      0    |      1    |\n"
    "-------------------------------\n"
    "   0 |          4         1.5 \n"
    "   1 |        1.5           2 \n"
    "\n"
)


def small() -> TMatrixDSym:
    return TMatrixDSym(2, [[4.0, 1.5], [1.5, 2.0]])


def test_a_matrix_is_read_by_call_or_by_row_and_knows_its_shape() -> None:
    """Macros read a covariance as ``m(i, j)`` or ``m[i][j]``; both must give the element."""
    m = small()
    assert m(0, 1) == 1.5
    assert m[1][1] == 2.0
    assert (m.GetNrows(), m.GetNcols()) == (2, 2)
    assert m.ClassName() == "TMatrixTSym<double>"
    assert m.GetName() == ""
    np.testing.assert_array_equal(np.asarray(m), [[4.0, 1.5], [1.5, 2.0]])
    assert np.asarray(m, dtype=np.float32).dtype == np.float32


def test_a_matrix_made_with_a_size_alone_is_zero() -> None:
    """``TMatrixDSym(n)`` is an n by n matrix of zeros, filled in afterwards."""
    np.testing.assert_array_equal(TMatrixDSym(3).values, np.zeros((3, 3)))
    assert TMatrixDSym().GetNrows() == 0


def test_a_copy_of_a_matrix_is_its_own() -> None:
    """``TMatrixDSym(other)`` copies, so inverting the copy leaves the original alone."""
    m = small()
    made = TMatrixDSym(m)
    made.values[0, 0] = 9.0
    assert m(0, 0) == 4.0
    assert made.GetNrows() == 2


def test_the_determinant_and_inverse_match_root() -> None:
    """A correlation study inverts the covariance; ``Invert`` works in place and returns it."""
    m = small()
    assert m.Determinant() == pytest.approx(5.75, rel=1e-15)
    assert m.Invert() is m
    assert m(0, 0) == pytest.approx(0.34782608695652173, rel=1e-15)
    assert m(0, 1) == pytest.approx(-0.2608695652173913, rel=1e-15)
    assert m(1, 1) == pytest.approx(0.6956521739130435, rel=1e-15)


def test_a_small_matrix_prints_as_root_prints_it(capsys: pytest.CaptureFixture[str]) -> None:
    """``Print()`` is what a tutorial's output shows, so the layout is ROOT's to the space."""
    small().Print()
    assert capsys.readouterr().out == SMALL


def test_an_empty_matrix_prints_only_its_heading() -> None:
    """With no columns there is no bar to number, and ROOT prints the heading alone."""
    assert matrix_text(np.zeros((0, 0))) == "\n0x0 matrix is as follows\n"


def test_a_narrow_format_puts_ten_columns_on_a_sheet() -> None:
    """ROOT fits ten columns a sheet when each is at most eight characters wide.

    The lines are those of ROOT's ``Print("f=%4.1f ")`` on an 11 by 11 identity.
    """
    lines = matrix_text(np.eye(11), "%4.1f ").splitlines()
    assert lines[3] == "     |" + "".join(f"    {j} |" for j in range(10))
    assert lines[4] == "-" * 75
    assert lines[5] == "   0 | 1.0  " + "0.0  " * 8 + "0.0 "
    assert lines[18] == "     |   10 |"
    assert lines[-2:] == ["  10 | 1.0 ", ""]


def test_a_vector_is_read_by_call_or_by_index() -> None:
    """``TVectorD`` hands back its elements as floats either way."""
    v = TVectorD([1.0, 2.5])
    assert v[1] == 2.5
    assert v(0) == 1.0
    assert v.GetNrows() == 2
    assert TVectorD().GetNrows() == 0


BIG = (
    "\n"
    "12x12 matrix is as follows\n"
    "\n"
    "     |       0    |       1    |       2    |       3    |       4    |\n"
    "----------------------------------------------------------------------\n"
    "   0 |      1e+05       0.002       0.003       0.004       0.005 \n"
    "   1 |      0.002       1e+05       0.006       0.008        0.01 \n"
    "   2 |      0.003       0.006       1e+05       0.012       0.015 \n"
    "   3 |      0.004       0.008       0.012       1e+05        0.02 \n"
    "   4 |      0.005        0.01       0.015        0.02       1e+05 \n"
    "   5 |      0.006       0.012       0.018       0.024        0.03 \n"
    "   6 |      0.007       0.014       0.021       0.028       0.035 \n"
    "   7 |      0.008       0.016       0.024       0.032        0.04 \n"
    "   8 |      0.009       0.018       0.027       0.036       0.045 \n"
    "   9 |       0.01        0.02        0.03        0.04        0.05 \n"
    "  10 |      0.011       0.022       0.033       0.044       0.055 \n"
    "  11 |      0.012       0.024       0.036       0.048        0.06 \n"
    "\n"
    "\n"
    "     |       5    |       6    |       7    |       8    |       9    |\n"
    "----------------------------------------------------------------------\n"
    "   0 |      0.006       0.007       0.008       0.009        0.01 \n"
    "   1 |      0.012       0.014       0.016       0.018        0.02 \n"
    "   2 |      0.018       0.021       0.024       0.027        0.03 \n"
    "   3 |      0.024       0.028       0.032       0.036        0.04 \n"
    "   4 |       0.03       0.035        0.04       0.045        0.05 \n"
    "   5 |      1e+05       0.042       0.048       0.054        0.06 \n"
    "   6 |      0.042       1e+05       0.056       0.063        0.07 \n"
    "   7 |      0.048       0.056       1e+05       0.072        0.08 \n"
    "   8 |      0.054       0.063       0.072       1e+05        0.09 \n"
    "   9 |       0.06        0.07        0.08        0.09       1e+05 \n"
    "  10 |      0.066       0.077       0.088       0.099        0.11 \n"
    "  11 |      0.072       0.084       0.096       0.108        0.12 \n"
    "\n"
    "\n"
    "     |      10    |      11    |\n"
    "----------------------------------------------------------------------\n"
    "   0 |      0.011       0.012 \n"
    "   1 |      0.022       0.024 \n"
    "   2 |      0.033       0.036 \n"
    "   3 |      0.044       0.048 \n"
    "   4 |      0.055        0.06 \n"
    "   5 |      0.066       0.072 \n"
    "   6 |      0.077       0.084 \n"
    "   7 |      0.088       0.096 \n"
    "   8 |      0.099       0.108 \n"
    "   9 |       0.11        0.12 \n"
    "  10 |      1e+05       0.132 \n"
    "  11 |      0.132       1e+05 \n"
    "\n"
)


def test_a_twelve_by_twelve_matrix_prints_in_three_sheets_of_five_columns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A fit with a dozen parameters prints its covariance in sheets, as ROOT does."""
    values = [
        [0.001 * (i + 1) * (j + 1) + (1e5 if i == j else 0.0) for j in range(12)] for i in range(12)
    ]
    TMatrixDSym(12, values).Print()
    assert capsys.readouterr().out == BIG
