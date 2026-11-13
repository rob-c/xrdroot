"""``v(i++) += 1``: a compound assignment to a call evaluates the call's arguments once."""

from __future__ import annotations

import types

import pytest

import xrdroot.pyroot as ROOT
from cintfake import fake
from xrdroot.pycint import Refusal, translate
from xrdroot.pycint.execute import run_source
from xrdroot.pycint.runtime import CallRef, HeldRef


def output(capsys: pytest.CaptureFixture[str], source: str) -> str:
    run_source(source, "t.C", root=ROOT)
    return capsys.readouterr().out


def test_a_vectors_element_is_read_and_written_with_its_index_worked_out_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    int calls = 0;
    int pick(int i) { calls++; return i; }
    void t() {
      TVectorD v(4);
      for (int i = 0; i < 4; i++) v(i) = i;
      v(v.GetUpb()) += 1;
      int k = 0;
      v(k++) += 10;
      v(pick(2)) *= 3;
      TMatrixD m(2, 2);
      m(pick(0), pick(1)) -= 2;
      double r = (v(pick(1)) += 5);
      printf("%g %g %g %g k=%d calls=%d m01=%g r=%g\\n", v(0), v(1), v(2), v(3), k, calls,
             m(0, 1), r);
    }
    """
    text = translate(source, "t.C")
    assert "store_reference(1, CallRef(v, (v.GetUpb(),)), " in text
    assert output(capsys, source) == "10 6 6 4 k=1 calls=4 m01=-2 r=6\n"


def test_a_method_returning_an_element_is_called_once_for_each_compound_assignment(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    int calls = 0;
    int pick(int i) { calls++; return i; }
    void t() {
      std::vector<int> w(3, 1);
      w.at(pick(1)) += 5;
      TMatrixD m(2, 2);
      TMatrixDRow(m, pick(1)) += 2;
      printf("%d %d %g %g calls=%d\\n", w[0], w[1], m(1, 0), m(0, 0), calls);
    }
    """
    text = translate(source, "t.C")
    assert "ItemRef(w, pick(1))" in text
    assert "HeldRef(ROOT.TMatrixDRow(m, pick(1)))" in text
    assert output(capsys, source) == "1 6 2 0 calls=2\n"


def test_a_call_reference_reads_by_calling_and_writes_back_through_the_runtime() -> None:
    class Grid:
        def __init__(self) -> None:
            self.cells = {(1, 2): 3}

        def __call__(self, *at: int) -> int:
            return self.cells[at]

        def __setcall__(self, *args: int) -> None:
            *at, value = args
            self.cells[tuple(at)] = value

        def Cell(self, *at: int) -> int:
            return self.cells[at]

        def SetCell(self, value: int) -> None:
            self.cells[(0, 0)] = value

    grid = Grid()
    called = CallRef(grid, (1, 2))
    called.value = called.value + 4
    assert grid.cells[(1, 2)] == 7
    method = CallRef(grid, (1, 2), "Cell")
    assert method.value == 7
    setter = CallRef(grid, (), "Cell")
    setter.value = 9
    assert grid.cells[(0, 0)] == 9
    held = HeldRef([1, 2])
    held.value = [3, 4]
    assert held.value == [3, 4]


def test_a_compound_assignment_through_a_macros_own_reference_is_still_refused() -> None:
    source = """
    struct Acc {
      double x = 0;
      double &At(int) { return x; }
      double &operator()(int) { return x; }
    };
    int pick(int i) { return i; }
    void t() { Acc a; %s }
    """
    for statement in ("a.At(pick(1)) += 2;", "a(pick(1)) += 2;"):
        with pytest.raises(Refusal, match="assigning to what the macro's class Acc returns"):
            translate(source % statement, "t.C")


def test_an_unchanging_argument_is_still_written_in_place(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = "void t() { TVectorD v(2); int i = 1; v(i) += 2; printf(\"%g\\n\", v(1)); }"
    assert "assign_call(v, (i,), v(i) + 2)" in translate(source, "t.C")
    assert output(capsys, source) == "2\n"
    assert run_source(source, "t.C", root=fake(), call=False) is not None


class _Row:
    """What a library method returns a reference to: read by adding, written by ``_assign``."""

    def __init__(self, owner: _Grid, at: int) -> None:
        self.owner, self.at = owner, at

    def __add__(self, other: int) -> int:
        return int(self.owner.rows[self.at] + other)

    def _assign(self, value: int) -> None:
        self.owner.rows[self.at] = value


class _Grid:
    def __init__(self) -> None:
        self.rows = [0, 10, 20]

    def Row(self, at: int) -> _Row:
        return _Row(self, at)


def test_a_library_method_returning_a_reference_is_written_back_into_what_it_returns() -> None:
    source = """
    int calls = 0;
    int pick(int i) { calls++; return i; }
    int t() { Grid g; g.Row(pick(1)) += 5; return g.Row(1) + 100 * calls; }
    """
    assert "CallRef(g, (pick(1),), 'Row')" in translate(source, "t.C")
    assert run_source(source, "t.C", root=types.SimpleNamespace(Grid=_Grid)) == 115
