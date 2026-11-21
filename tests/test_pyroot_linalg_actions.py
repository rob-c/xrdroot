"""ROOT's element actions and lazy matrices: a macro's own ``Operation`` and ``FillIn``.

Both are what a ROOT macro writes a class for: an element action fills a matrix with
a function of the position, and a lazy matrix promises one it makes only when needed.
``TElementPosActionD``, ``TElementActionD`` and their ``F`` kinds are what such a class
derives from, and ``Apply`` is what calls it back.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pycint import translate
from xrdroot.pycint.execute import run_source


def test_an_element_action_fills_every_element_in_row_order() -> None:
    """``Apply`` of a ``TElementActionT``: the action sees each element, row by row, and
    writes it - by reference, which is what the macro's ``element =`` stores into."""

    class Times2(ROOT.TElementActionD):
        def Operation(self, element) -> None:
            element.value = element.value * 2

    matrix = ROOT.TMatrixD(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert matrix.Apply(Times2()) is matrix
    assert list(matrix.GetMatrixArray()) == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


def test_a_position_action_is_told_the_row_and_column_of_each_element() -> None:
    """``fI`` and ``fJ`` are ROOT's own members, counted from the matrix's lower bound."""

    class Filled(ROOT.TElementPosActionD):
        def Operation(self, element) -> None:
            element.value = 10 * self.fI + self.fJ

    matrix = ROOT.TMatrixD(1, 2, 4, 5)
    matrix.Apply(Filled())
    assert list(matrix.GetMatrixArray()) == [14.0, 15.0, 24.0, 25.0]


def test_a_symmetric_matrix_acts_on_its_upper_triangle_and_mirrors_the_answer() -> None:
    """``TMatrixTSym::Apply`` hands the action a fresh zero and fills both halves with what it
    gives back, so a symmetric matrix stays symmetric whatever the action does."""

    class Position(ROOT.TElementPosActionD):
        def Operation(self, element) -> None:
            element.value = self.fI + self.fJ

    sym = ROOT.TMatrixDSym(3)
    sym.Apply(Position())
    assert np.array_equal(sym.values, np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))
    seen = []

    class WhatWasThere(ROOT.TElementActionD):
        def Operation(self, element) -> None:
            seen.append(element.value)
            element.value = 1.0

    sym.Apply(WhatWasThere())
    assert seen == [0.0] * 6  # the upper triangle's elements, each handed over as zero


def test_a_vector_apply_tells_a_position_action_the_index_it_has_reached() -> None:
    class Squares(ROOT.TElementPosActionD):
        def Operation(self, element) -> None:
            element.value = float(self.fI * self.fI)

    vector = ROOT.TVectorD(2, 4)
    assert vector.Apply(Squares()) is vector
    assert list(vector) == [4.0, 9.0, 16.0]


def test_an_action_with_no_operation_is_refused_as_roofs_pure_virtual_is() -> None:
    with pytest.raises(UnsupportedFeatureError, match="with no Operation"):
        ROOT.TMatrixD(2, 2).Apply(ROOT.TElementActionD())
    with pytest.raises(UnsupportedFeatureError, match="with no Operation"):
        ROOT.TVectorD(2).Apply(ROOT.TElementActionF())
    assert ROOT.TElementActionD.fElementType == "Double_t"
    assert ROOT.TElementPosActionT["double"] is ROOT.TElementPosActionT


def test_a_macros_own_action_class_is_what_apply_calls_back(capsys) -> None:
    """``class FillMatrix : public TElementPosActionD`` of ROOT's ``test/vlazy.cxx``: the base
    is the library's, ``fI``/``fJ`` are read from it, and ``element`` is written through."""
    source = """
    #include <cstdio>
    class Fill : public TElementPosActionD {
       void Operation(Double_t &element) const override { element = 10*fI + fJ; }
    public:
       Fill() {}
    };
    void t() {
       TMatrixD m(1,2,4,5);
       Fill f;
       m.Apply(f);
       printf("%g %g %g %g\\n", m(1,4), m(1,5), m(2,4), m(2,5));
    }
    """
    assert "class Fill(ROOT.TElementPosActionD):" in translate(source, "t.C")
    run_source(source, "t.C", root=ROOT)
    assert capsys.readouterr().out == "14 15 24 25\n"


def test_a_lazy_matrix_is_filled_in_when_a_matrix_is_built_from_it() -> None:
    """``TMatrixD m = THaarMatrixD(order)``: the promise's bounds are the matrix's, and its
    ``FillIn`` writes the elements - here Haar's, each column of unit length."""

    class Doubling(ROOT.TMatrixDLazy):
        def __init__(self) -> None:
            super().__init__(2, 2)

        def FillIn(self, m) -> None:
            m._become_ones()

    assert ROOT.TMatrixDLazy is ROOT.TMatrixTLazy
    assert ROOT.TMatrixFLazy is ROOT.TMatrixTLazy
    assert ROOT.TMatrixDSymLazy is ROOT.TMatrixTSymLazy
    assert ROOT.TMatrixFSymLazy is ROOT.TMatrixTSymLazy
    promise = ROOT.TMatrixTLazy(0, 2, 0, 1)
    assert (promise.GetRowLwb(), promise.GetRowUpb(), promise.GetColLwb(), promise.GetColUpb()) == (
        0, 2, 0, 1)  # fmt: skip
    square = ROOT.TMatrixTSymLazy(3)
    assert (square.GetRowLwb(), square.GetRowUpb()) == (0, 2)
    assert (square.GetColLwb(), square.GetColUpb()) == (0, 2)
    forced = ROOT.TMatrixD(ROOT.THilbertMatrixD(2, 2))
    assert forced(1, 1) == pytest.approx(1 / 3)


def test_a_promise_with_no_fill_in_is_refused_in_a_full_sentence() -> None:
    with pytest.raises(UnsupportedFeatureError, match="no FillIn"):
        ROOT.TMatrixD(ROOT.TMatrixTLazy(2, 2))
    with pytest.raises(UnsupportedFeatureError, match="symmetric matrix with no FillIn"):
        ROOT.TMatrixDSym(ROOT.TMatrixTSymLazy(2))


def test_a_lazy_matrix_forced_into_a_matrix_that_is_already_one_fills_that_one_in() -> None:
    """``m = promise(...)`` is ``TMatrixT::operator=(const TMatrixTLazy &)``: the matrix keeps
    its own bounds and the promise writes into it."""

    class Rows(ROOT.TMatrixDLazy):
        def __init__(self) -> None:
            super().__init__(2, 2)

        def FillIn(self, m) -> None:
            m[0] = [1.0, 2.0]
            m[1] = [3.0, 4.0]

    matrix = ROOT.TMatrixD(2, 2)
    matrix._assign(Rows())
    assert list(matrix.GetMatrixArray()) == [1.0, 2.0, 3.0, 4.0]


def test_haars_matrix_is_orthonormal_and_lazy_in_both_element_types() -> None:
    """``THaarMatrixD``/``THaarMatrixF`` are promises of ROOT's Haar matrix: the columns are
    Haar's functions, each of unit length and orthogonal to the others."""
    assert ROOT.THaarMatrixF.__name__ == "THaarMatrixF"
    for kind, dtype in ((ROOT.THaarMatrixD, np.dtype(np.float64)),
                        (ROOT.THaarMatrixF, np.dtype(np.float32))):  # fmt: skip
        matrix = ROOT.TMatrixD(kind(3))
        assert matrix.values.dtype == np.dtype(np.float64) or dtype == np.dtype(np.float32)
        assert (matrix.GetNrows(), matrix.GetNcols()) == (8, 8)
        assert np.allclose(matrix.values.T @ matrix.values, np.eye(8), atol=1e-6)
    wide = ROOT.THaarMatrixD(3, 5)
    assert (wide.GetRowUpb(), wide.GetColUpb()) == (7, 4)
    small = ROOT.TMatrixF(ROOT.THaarMatrixF(3))
    assert small.values.dtype == np.float32
    assert small(0, 1) == pytest.approx(8**-0.5, rel=1e-6) and small(0, 2) == 0.5
    with pytest.raises(ValueError, match="cannot have 9 columns"):
        ROOT.TMatrixD(ROOT.THaarMatrixD(2, 9))


def test_an_f_matrix_keeps_its_elements_as_32_bit_floats() -> None:
    """ROOT's ``TMatrixT<Float_t>`` stores ``Float_t``: what is written is rounded on the way
    in, which is why ``TMatrixF`` and ``TMatrixD`` of the same numbers differ in the last
    digits."""
    assert ROOT.TMatrixF is ROOT.TMatrixT["float"]
    assert ROOT.TMatrixF.DTYPE == np.dtype(np.float32)
    assert ROOT.TMatrixFSym.DTYPE == np.dtype(np.float32)
    third = ROOT.TMatrixF(1, 1)
    third.__setcall__(0, 0, 1 / 3)
    assert third(0, 0) == float(np.float32(1 / 3))
    assert ROOT.TMatrixD(1, 1).DTYPE == np.dtype(np.float64)
    assert ROOT.TMatrixF(2, 2, [1.0, 2.0, 3.0, 4.0]).Max() == 4.0
    assert ROOT.TMatrixF(1, 2).ResizeTo(2, 2).values.dtype == np.float32
    assert ROOT.TMatrixF(1, 1)._became(np.ones((1, 1))).values.dtype == np.float32
    assigned = ROOT.TMatrixF(1, 1)
    assigned._assign(np.ones((1, 1)))
    assert assigned.values.dtype == np.float32


def test_a_submatrix_is_taken_and_put_back_by_its_bounds() -> None:
    """``GetSub`` copies the bounds asked for, ``SetSub`` writes a matrix in from its corner -
    with ``"S"`` shifting the copy's own lower bounds to zero."""
    matrix = ROOT.TMatrixD(3, 3, list(range(9)))
    part = matrix.GetSub(1, 2, 0, 1, "")
    assert (part.GetRowLwb(), part.GetColLwb(), part.GetNrows(), part.GetNcols()) == (1, 0, 2, 2)
    assert list(part.GetMatrixArray()) == [3.0, 4.0, 6.0, 7.0]
    shifted = matrix.GetSub(1, 2, 0, 1, "S")
    assert (shifted.GetRowLwb(), shifted.GetColLwb()) == (0, 0)
    target = ROOT.TMatrixD(2, 2)
    assert target.SetSub(0, 0, part) is target
    assert list(target.GetMatrixArray()) == [3.0, 4.0, 6.0, 7.0]
    assert (matrix.RowNorm(), matrix.ColNorm()) == (matrix.NormInf(), matrix.Norm1())


def test_a_nested_class_is_the_class_a_type_names_both_ways(capsys) -> None:
    """``Outer::Inner`` is a class of the macro's like any other, written at the top level under
    its own name; a member of its type is made with it, not with a name the library has not."""
    source = """
    struct Outer { struct Inner { int v; }; Outer::Inner i; };
    void t() { Outer o; o.i.v = 7; printf("%d\\n", o.i.v); }
    """
    python = translate(source, "t.C")
    assert "self.i = Inner()" in python
    run_source(source, "t.C", root=ROOT)
    assert capsys.readouterr().out == "7\n"


def test_a_matrix_is_built_from_a_class_it_is_not_a_copy_of(capsys) -> None:
    """``TMatrixF orig_m = THaarMatrixF(4, 6)``: C++ copy-initialisation of one class type from
    another is that class's converting constructor, which is what forces a lazy matrix; and
    ``m = promise(...)`` writes into ``m``, as ``operator=`` does."""
    source = """
    #include <cstdio>
    class Fixed : public TMatrixDLazy {
       void FillIn(TMatrixD &m) const override { for (int i=m.GetRowLwb(); i<=m.GetRowUpb(); i++)
          for (int j=m.GetColLwb(); j<=m.GetColUpb(); j++) m(i,j) = 10*i+j; }
    public:
       Fixed(int n) : TMatrixDLazy(n,n) {}
    };
    void t() {
       TMatrixD made = Fixed(2);
       printf("%g %g\\n", made(0,1), made(1,0));
       TMatrixD kept(2,2);
       kept = Fixed(2);
       printf("%g\\n", kept(1,1));
       TMatrixD lazy = THaarMatrixD(2);
       printf("%g\\n", lazy(0,2));
    }
    """
    python = translate(source, "t.C")
    assert "ROOT.TMatrixD(Fixed(2))" in python
    assert "assign_into(kept, Fixed(2))" in python
    run_source(source, "t.C", root=ROOT)
    assert capsys.readouterr().out == "1 10\n11\n0.707107\n"


def test_a_qualified_type_is_the_nested_class_only_when_the_macro_declares_it() -> None:
    """``Outer::Inner`` is the macro's own class when ``Outer`` declares an ``Inner``; a
    qualifier naming a namespace or a class the macro does not declare is the library's."""
    nested = translate("struct Outer { struct Inner {}; }; Outer::Inner v;", "t.C")
    assert "v = Inner()" in nested
    elsewhere = translate("struct Outer { struct Other {}; }; Outer::Inner v;", "t.C")
    assert "v = ROOT.Outer.Inner()" in elsewhere
    library = translate("Outside::Inner v;", "t.C")
    assert "v = ROOT.Outside.Inner()" in library


def test_a_class_of_the_macros_is_copied_into_one_of_its_own_bases_not_built() -> None:
    """``Base b = Derived()`` is a copy - ``Derived`` is a ``Base`` already - where a class
    that is no ``Base`` at all is one the library's constructor makes."""
    source = "struct Base { int n; }; struct Derived : Base {}; void t() { Base b = Derived(); }"
    assert "b = Derived()" in translate(source, "t.C")
    other = "struct Thing { int n; }; void t() { TMatrixD m = Thing(); }"
    assert "m = ROOT.TMatrixD(Thing())" in translate(other, "t.C")


def test_an_element_action_is_a_template_root_names() -> None:
    """``TElementActionT<double>`` is a template like ROOT's, so a macro's use of it parses."""
    text = translate("void t() { TElementActionT<double> *a = nullptr; }", "t.C")
    assert "a = None" in text
