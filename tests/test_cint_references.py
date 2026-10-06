"""Assigning through what a call returns by reference, ``m(i, j) = v``, and through ``*p``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cintfake import fake
from xrdroot.cint import Refusal, translate
from xrdroot.cint.execute import run_source
from xrdroot.cint.runtime import Cell, assign_call, assign_into, assign_method, store_through


def body(source: str) -> str:
    text = translate("void t() {\n" + source + "\n}\n", "t.C")
    compile(text, "t.C", "exec")
    return text


class Matrix:
    """A stand-in for ``TMatrixD``: ``m(i, j)`` reads, ``__setcall__`` writes."""

    def __init__(self, rows: int = 2, cols: int = 2) -> None:
        self.data = np.zeros((rows, cols))

    def __call__(self, *index: int) -> float:
        return float(self.data[index])

    def __setcall__(self, *args: Any) -> None:
        *index, value = args
        self.data[tuple(index)] = value


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("m(0, 1) = 2;", "assign_call(ROOT.m, (0, 1), 2)"),
        ("v(i) += 1;", "assign_call(ROOT.v, (ROOT.i,), ROOT.v(ROOT.i) + 1)"),
        ("v(i + 1) -= 2;", "assign_call(ROOT.v, (ROOT.i + 1,), ROOT.v(ROOT.i + 1) - 2)"),
        ("a(0, 0) = b(1, 1) = 3;", "assign_call(ROOT.a, (0, 0), assign_call(ROOT.b, (1, 1), 3))"),
        ("(*sqrtMat)(i, i) = 1;", "assign_call(deref(ROOT.sqrtMat), (ROOT.i, ROOT.i), 1)"),
        ("p.X() = 6;", "assign_method(ROOT.p, 'X', (), 6)"),
        ("c.ParSettings(i) = s;", "assign_method(ROOT.c, 'ParSettings', (ROOT.i,), ROOT.s)"),
        ("items.at(k - 1) = x;", "set_item(ROOT.items, ROOT.k - 1, ROOT.x)"),
        ("TMatrixDColumn(A, 0) = 1.0;", "assign_into(ROOT.TMatrixDColumn(ROOT.A, 0), 1.0)"),
        ("TMatrixD A(2, 2); TMatrixDDiag d(A); d = 0.0;", "assign_into(d, 0.0)"),
        ("TDecompQRH q(A); auto Q = q.GetOrthogonalMatrix(); auto QT = Q;", "QT = value_copy(Q)"),
        ("TDecompLU lu(A); Double_t d1, d2; lu.Det(d1, d2);", "lu.Det(d1, d2)"),
        ("auto pt = model->MakeField<float>(\"pt\"); *pt = 1;", "pt[0] = f32(1)"),
        ("auto pt = make(); *pt = 1;", "store_through(pt, 1)"),
        ("TH1F *h; *h = *g;", "store_through(h, deref(ROOT.g))"),
        ("int n = (*id = 3);", "n = int(store_through(ROOT.id, 3))"),
    ],
)
def test_an_assignment_through_a_reference_goes_through_the_runtime(
    source: str, fragment: str
) -> None:
    assert fragment in body(source)


def test_a_field_an_rntuple_model_makes_is_a_pointer_to_its_type() -> None:
    source = """auto n = model->MakeField<float>("n"); auto s = model->MakeField<std::string>("s");
    std::istringstream in("2 x"); in >> *n >> *s; auto *q = get(); in >> *q; *n = 1;"""
    text = body(source)
    assert "n[0] = in_.extract('float')" in text
    assert "store_through(s, in_.extract('std::string'))" in text
    assert "store_through(q, in_.extract('double'))" in text
    assert "    n[0] = f32(1)\n" in text


def test_reading_through_a_shared_pointer_to_a_number_goes_through_the_runtime() -> None:
    text = body('std::shared_ptr<int> a = model->MakeField<int>("a"); h->Fill(*a); *a = 3;'
                'std::shared_ptr<TH1F> g; g->Fill(*a);')  # fmt: skip
    assert "Fill(deref(a))" in text
    assert "store_through(a, 3)" in text
    assert "g.Fill(deref(a))" in text
    skip = body('auto p = model->MakeField<std::uint16_t>("skip"); (*p)++;')
    assert "p[0] = u16(p[0] + 1)" in skip


def test_a_reference_the_macros_own_class_returns_is_refused() -> None:
    source = """struct V { double x;
      double &X() { return x; } double &operator()(int) { return x; } };
    void t() { V v; v.X() = 1; }"""
    with pytest.raises(Refusal, match="the macro's class V returns from its method X"):
        translate(source, "t.C")
    with pytest.raises(Refusal, match="returns from its operator"):
        translate(source.replace("v.X() = 1", "v(0) = 1"), "t.C")


def test_a_matrix_element_is_written_through_its_setcall(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = fake()
    root.TMatrixD = Matrix
    source = """{ TMatrixD m(2, 2); m(0, 1) = 2; m(0, 1) *= 3; m(1, 0) = m(0, 1) = 5;
    printf("%g %g\\n", m(0, 1), m(1, 0)); }"""
    run_source(source, "t.C", root=root)
    assert capsys.readouterr().out == "5 5\n"


def test_a_value_made_by_the_library_is_stored_through_its_cell(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = fake()
    root.make = lambda: Cell(0.0, "float")
    source = """{ auto pt = make(); *pt = 1.5; *pt += 1; printf("%g\\n", *pt); }"""
    run_source(source, "t.C", root=root)
    assert capsys.readouterr().out == "2.5\n"


def test_the_runtime_assigns_into_what_it_is_given() -> None:
    items = {"a": 1}
    assert assign_call(items, ("b",), 2) == 2 and items == {"a": 1, "b": 2}
    grid = np.zeros((2, 2))
    assign_call(grid, (1, 1), 4.0)
    assert grid[1, 1] == 4.0
    with pytest.raises(TypeError, match="neither __setcall__ nor __setitem__"):
        assign_call(object(), (0,), 1)
    listed, found = [1, 2], {1}
    assert assign_into(listed, (3,)) == (3,) and listed == [3]
    assign_into(found, {7})
    assert found == {7}
    assign_into(grid, 1.0)
    assert grid.sum() == 4.0
    with pytest.raises(TypeError, match="a int returned by value cannot be assigned to"):
        assign_into(3, 4)


def test_objects_are_assigned_by_their_own_operator_or_by_copying_their_state() -> None:
    class Own:
        def __init__(self) -> None:
            self.got: Any = None

        def _assign(self, other: Any) -> None:
            self.got = other

    class Plain:
        def __init__(self, n: int) -> None:
            self.n = n

        def SetN(self, n: int) -> None:
            self.n = n

        def Get(self, k: int) -> Plain:
            return self

    own, plain = Own(), Plain(1)
    store_through(own, 5)
    assert own.got == 5
    store_through(plain, Plain(7))
    assert plain.n == 7
    assign_method(plain, "N", (), 9)
    assert plain.n == 9
    assign_method(plain, "Get", (0,), Plain(2))
    assert plain.n == 2
    values = [0, 0]
    store_through(values, 8)
    assert values == [8, 0]
