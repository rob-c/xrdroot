"""``ROOT::Math``'s ``SVector`` and ``SMatrix``: the shapes a macro subscripts, and what ROOT
prints for them."""

import pytest

from xrdroot.pyroot.linalg import Dot, MatRepStd, MatRepSym, SMatrix, SVector


def test_a_subscripted_vector_carries_its_size_and_element_type():
    three = SVector["float", 3]
    assert three.kSize if isinstance(three, SVector) else True  # a class, not an instance
    assert three is SVector["float", 3]
    assert three is not SVector["float", 4]
    assert three is not SVector["double", 3]
    assert list(three(4, 5, 6)) == [4.0, 5.0, 6.0]
    assert SVector[3](4, 5, 6)[0] == 4
    assert SVector[3]._element is float


def test_a_vector_is_built_from_numbers_a_sequence_or_nothing():
    assert list(SVector["double", 2](1.5, -2.25)) == [1.5, -2.25]
    assert list(SVector["double", 2]([1.5, -2.25])) == [1.5, -2.25]
    assert list(SVector["double", 2](SVector["double", 2](1, 2))) == [1.0, 2.0]
    assert list(SVector["double", 2]()) == [0.0, 0.0]
    assert list(SVector["int", 1](5)) == [5]
    with pytest.raises(TypeError, match="an SVector of 3 elements cannot be built from 2"):
        SVector["double", 3](1, 2)


def test_a_vector_combines_element_by_element_and_with_a_number():
    x, y = SVector["double", 3](4, 5, 6), SVector["double", 3](1, 2, 3)
    assert list(x + y) == [5.0, 7.0, 9.0]
    assert list(y + x) == [5.0, 7.0, 9.0]
    assert list(x - y) == [3.0, 3.0, 3.0]
    assert list(10 - y) == [9.0, 8.0, 7.0]


def test_a_vector_scales_divides_negates_and_compares():
    x, y = SVector["double", 3](4, 5, 6), SVector["double", 3](1, 2, 3)
    assert list(x * 2) == [8.0, 10.0, 12.0]
    assert list(2 * x) == [8.0, 10.0, 12.0]
    assert list(x / 2) == [2.0, 2.5, 3.0]
    assert list(-x) == [-4.0, -5.0, -6.0]
    assert x != y and x == SVector["double", 3](4, 5, 6)
    assert x != 3
    with pytest.raises(TypeError, match="SVector sizes differ: 3 and 2"):
        x + SVector["double", 2](1, 2)


def test_a_vectors_scalar_product_is_in_its_element_type():
    x = SVector["double", 3](4, 5, 6)
    assert Dot(x, x) == 77.0
    assert x.Dot(x) == 77.0
    assert x.Mag2() == 77.0
    assert SVector["int", 2](3, 4).Mag2() == 25
    with pytest.raises(TypeError, match="vectors of one size, not 3 and 2"):
        Dot(x, SVector["double", 2](1, 2))


def test_a_vector_writes_elements_and_windows():
    x = SVector["double", 3](4, 5, 6)
    assert str(x) == "4, 5, 6"
    assert str(SVector["double", 2](1.5, -2.25)) == "1.5, -2.25"
    assert len(x) == 3
    x[1] = 7
    assert x[1] == 7.0
    wide = SVector["double", 4](1, 2, 3, 4)
    wide.Place_at(SVector["double", 2](8, 9), 1)
    assert list(wide) == [1.0, 8.0, 9.0, 4.0]


def test_a_subscripted_matrix_carries_its_shape_and_representation():
    sym = SMatrix["double", 3]
    std = SMatrix["double", 2, 3]
    assert (sym().kRows, sym().kCols) == (3, 3)
    assert (std().kRows, std().kCols) == (2, 3)
    assert sym is SMatrix["double", 3]
    assert sym is not SMatrix["double", 3, MatRepStd["double", 3, 3]]
    assert SMatrix["double", 3, MatRepSym["double", 3]] is sym
    assert repr(MatRepSym["double", 3]) == "MatRepSym(3, 3)"
    assert repr(MatRepStd[float, 3, 4]) == "MatRepStd(3, 4)"
    with pytest.raises(TypeError, match="needs an element type and its dimensions"):
        SMatrix[3]
    with pytest.raises(TypeError, match="MatRepSym needs a square matrix, and 3 by 4 is not"):
        SMatrix["double", 3, MatRepSym[float, 3, 4]]


def test_a_matrix_is_filled_in_row_order_or_by_element():
    assert SMatrix["double", 2, 3]([1, 2, 3, 4, 5, 6])(1, 2) == 6.0
    assert SMatrix["double", 2, 3](1, 2, 3, 4, 5, 6)(1, 0) == 4.0
    assert SMatrix["double", 2, 2]()(0, 0) == 0.0
    square = SMatrix["double", 2, 2]()
    square[0, 1] = 3
    square.__setcall__(1, 0, 4)
    assert square[0, 1] == 3.0 and square(1, 0) == 4.0
    assert square.kSize == 4
    with pytest.raises(TypeError, match="an SMatrix of 2 by 3 cannot be built from 2 values"):
        SMatrix["double", 2, 3](1, 2)


def test_a_symmetric_matrix_mirrors_what_is_set():
    square = SMatrix["double", 3]()
    square.__setcall__(0, 2, 5)
    square[2, 1] = 7
    assert square(2, 0) == 5.0 and square(1, 2) == 7.0
    filled = SMatrix["double", 2]([1, 2, 3, 4])
    assert [filled(i, j) for i in range(2) for j in range(2)] == [1.0, 2.0, 2.0, 4.0]


def test_a_matrix_adds_subtracts_and_scales():
    a = SMatrix["double", 2, 2](1, 2, 3, 4)
    b = SMatrix["double", 2, 2](10, 20, 30, 40)
    assert [int(v) for v in (a + b)] == [11, 22, 33, 44]


def test_a_matrix_scales_divides_negates_and_compares():
    a = SMatrix["double", 2, 2](1, 2, 3, 4)
    b = SMatrix["double", 2, 2](10, 20, 30, 40)
    assert [int(v) for v in (b - a)] == [9, 18, 27, 36]
    assert [int(v) for v in (a - 1)] == [0, 1, 2, 3]
    assert [int(v) for v in (a + 1)] == [2, 3, 4, 5]
    assert [int(v) for v in (a * 2)] == [2, 4, 6, 8]
    assert [int(v) for v in (2 * a)] == [2, 4, 6, 8]
    assert list(a / 2) == [0.5, 1.0, 1.5, 2.0]
    assert [int(v) for v in -a] == [-1, -2, -3, -4]
    assert a == SMatrix["double", 2, 2](1, 2, 3, 4)
    assert a != b and a != 3
    with pytest.raises(TypeError, match="shapes differ: 2 by 2 and 2 by 3"):
        a + SMatrix["double", 2, 3]()



def test_a_matrix_refuses_to_combine_shapes_that_differ():
    a = SMatrix["double", 2, 2](1, 2, 3, 4)
    with pytest.raises(TypeError, match="shapes differ: 2 by 2 and 2 by 3"):
        a + SMatrix["double", 2, 3]()
    with pytest.raises(TypeError, match="shapes differ: 2 by 2 and 2 by 3"):
        a - SMatrix["double", 2, 3]()


def test_a_matrix_multiplies_by_a_matrix_and_by_a_vector():
    a = SMatrix["double", 2, 3](1, 2, 3, 4, 5, 6)
    product = a * SMatrix["double", 3, 2]([7, 8, 9, 10, 11, 12])
    assert (product.kRows, product.kCols) == (2, 2)
    assert [product(i, j) for i in range(2) for j in range(2)] == [58.0, 64.0, 139.0, 154.0]
    applied = a * SVector["double", 3](1, 0, 0)
    assert (applied.kSize, list(applied)) == (2, [1.0, 4.0])
    with pytest.raises(TypeError, match="the inner dimensions differ"):
        a * SMatrix["double", 2, 2]()
    with pytest.raises(TypeError, match="the dimensions differ"):
        a * SVector["double", 2](1, 2)


def test_a_matrix_transposes_traces_and_solves():
    a = SMatrix["double", 2, 3](1, 2, 3, 4, 5, 6)
    transposed = a.Transpose()
    assert (transposed.kRows, transposed.kCols) == (3, 2)
    assert transposed(1, 0) == 2.0
    assert a.Trace() == 6.0


def test_a_matrix_solves_and_refuses_nonsense():
    a = SMatrix["double", 2, 3](1, 2, 3, 4, 5, 6)
    with pytest.raises(TypeError, match="only a square matrix has a determinant"):
        a.Determinant()
    square = SMatrix["double", 2, 2](4, 7, 2, 6)
    assert square.Determinant() == pytest.approx(10.0)
    assert square.Invert() is True
    assert square(0, 0) == pytest.approx(0.6)
    assert square(0, 1) == pytest.approx(-0.7)
    singular = SMatrix["double", 2, 2](1, 2, 2, 4)
    assert singular.Invert() is False
    assert singular(0, 0) == 1.0
    assert a.Invert() is False


def test_a_matrix_places_vectors_and_submatrices():
    a = SMatrix["double", 4, 3]()
    a.Place_in_col(SVector["double", 3](4, 5, 6), 1, 0)
    a.Place_in_row(SVector["double", 2](2, 3), 0, 1)
    a.Place_at(SMatrix["double", 2, 2]([9, 8, 7, 6]), 2, 1)
    assert [a(i, 0) for i in range(4)] == [0.0, 4.0, 5.0, 6.0]
    assert [a(0, j) for j in range(3)] == [0.0, 2.0, 3.0]
    assert [a(2, 1), a(2, 2), a(3, 1), a(3, 2)] == [9.0, 8.0, 7.0, 6.0]
    lines = str(a).split("\n")
    assert [len(line) for line in lines] == [38, 38, 38, 40]
    assert lines[0].startswith("[ ") and lines[0].endswith("3")
    assert lines[-1].endswith("6 ]")


def test_a_vector_or_matrix_is_built_from_a_pointer_and_a_count():
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert list(SVector["double", 3](numbers, 3)) == [1.0, 2.0, 3.0]
    assert SMatrix["double", 3](numbers, 9)(2, 2) == 9.0
    with pytest.raises(TypeError, match="these two arguments are none of those"):
        SMatrix["double", 2, 2]("abc", "ab")


def test_a_vector_or_matrix_is_built_from_two_pointers_into_one_array():
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert SMatrix["double", 3](numbers, numbers[9:])(0, 1) == 2.0
    assert list(SVector["double", 4](numbers, numbers[4:])) == [1.0, 2.0, 3.0, 4.0]


def test_a_range_of_a_vector_or_matrix_is_read_by_its_cursors():
    square = SMatrix["double", 2, 2]([1, 2, 3, 4])
    assert list(square.begin()) == [1.0, 2.0, 3.0, 4.0]
    assert square.begin() != square.end() and square.end() == square.end()
    assert next(square.begin()) == 1.0
    assert list(SVector["double", 4](square.begin(), square.end())) == [1.0, 2.0, 3.0, 4.0]
    three = SVector["double", 3](1, 2, 3)
    assert list(three.begin()) == [1.0, 2.0, 3.0] and list(three.end()) == []
    assert list(SVector["double", 3](three.begin(), three.end())) == [1.0, 2.0, 3.0]


def test_a_matrix_reads_a_row_a_column_and_its_diagonal_as_vectors():
    square = SMatrix["double", 2, 2]([1, 2, 3, 4])
    assert list(square.Row(0)) == [1.0, 2.0]
    assert list(square.Col(1)) == [2.0, 4.0]
    assert list(square.Diagonal()) == [1.0, 4.0]
    flat = SMatrix["double", 2, 3]([1, 2, 3, 4, 5, 6])
    assert list(flat.Diagonal()) == [1.0, 5.0]
    assert isinstance(square.Row(0), SVector) and square.Row(0).kSize == 2


def test_a_matrix_copies_the_rest_of_a_row_a_column_or_itself():
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    square = SMatrix["double", 3](numbers, 9)
    assert list(square.SubRow(0, 1)) == [2.0, 3.0]
    assert list(square.SubCol(0, 1)) == [2.0, 5.0, 6.0]
    corner = square.Sub(1, 1)
    assert (corner.kRows, corner.kCols) == (2, 2)
    assert [corner(i, j) for i in range(2) for j in range(2)] == [5.0, 6.0, 6.0, 9.0]
    assert list(square.SubRow[SVector["double", 2]](0, 1)) == [2.0, 3.0]
    assert list(square.SubCol[SVector["double", 2]](0, 1)) == [2.0, 5.0, 6.0]
    assert square.Sub[SMatrix["double", 2, 2]](1, 1)(1, 1) == 9.0


def test_a_symmetric_matrix_hands_over_its_triangles():
    square = SMatrix["double", 3]([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert list(square.UpperBlock()) == [1.0, 2.0, 3.0, 5.0, 6.0, 9.0]
    assert list(square.LowerBlock()) == [1.0, 2.0, 5.0, 3.0, 6.0, 9.0]


def test_the_free_functions_of_a_matrix_are_the_ones_root_has():
    from xrdroot.pyroot.linalg.smatrix import (
        Similarity,
        SimilarityT,
        SMatrixIdentity,
        Square,
        Times,
        Transpose,
    )

    square = SMatrix["double", 2, 2](1, 2, 3, 4)
    assert Transpose(square)(0, 1) == 3.0
    assert Times(square, square)(0, 1) == 4.0
    assert Times(square, 2)(1, 0) == 6.0
    assert Square(square)(1, 1) == 16.0
    identity = SMatrixIdentity(SMatrix["float", 2, 3])
    assert [identity(i, j) for i in range(2) for j in range(3)] == [1.0, 0, 0, 0, 1.0, 0]
    three = SMatrix["double", 3, 3]([1, 2, 3, 4, 5, 6, 7, 8, 9])
    x = SVector["double", 3](1, 2, 3)
    assert Similarity(x, three) == pytest.approx(228.0)
    assert Similarity(square, square)(0, 1) == (square * square * square.Transpose())(0, 1)
    assert SimilarityT(square, square)(0, 0) == pytest.approx(
        (square.Transpose() * square * square)(0, 0)
    )
    assert Times(SVector["double", 2](1, 2), SVector["double", 2](3, 4))[1] == 8.0


def test_the_free_functions_of_a_vector_are_the_ones_root_has():
    from xrdroot.pyroot.linalg.smatrix import Cross, Mag, Mag2, Unit

    x = SVector["double", 3](1, 2, 3)
    assert Mag2(x) == 14.0 and Mag(x) == pytest.approx(14.0**0.5)
    assert list(Cross(x, SVector["double", 3](4, 5, 6))) == [-3.0, 6.0, -3.0]
    assert list(Unit(x)) == pytest.approx([1 / 14.0**0.5 * n for n in (1, 2, 3)])
    assert Mag(SVector["int", 2](3, 4)) == 5


def test_a_vector_of_the_wrong_width_is_refused_by_name():
    from xrdroot.pyroot.linalg.smatrix import Cross, SMatrixIdentity

    x = SVector["double", 3](1, 2, 3)
    with pytest.raises(TypeError, match="two vectors of three elements, not 2 and 3"):
        Cross(SVector["double", 2](1, 2), x)
    with pytest.raises(TypeError, match="has to name that variable's class here"):
        SMatrixIdentity()
