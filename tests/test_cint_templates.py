"""Function templates with value parameters: ``template <unsigned N>``, given or deduced."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.pycint import translate
from xrdroot.pycint.execute import run_source


def output(capsys: pytest.CaptureFixture[str], source: str) -> str:
    run_source(source, "t.C", root=fake())
    return capsys.readouterr().out


def test_a_value_parameter_is_a_keyword_the_explicit_template_argument_gives(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    template <typename T, int N> T times(T x) { return N * x; }
    void t() { printf("%d %g\\n", times<int, 3>(2), times<double, 2>(1.25)); }
    """
    assert "def times(x, *, N=None):" in translate(source, "t.C")
    assert "times__int_3(2)" in translate(source, "t.C")
    assert "def times__int_3(x):\n    return 3 * x" in translate(source, "t.C")
    assert output(capsys, source) == "6 2.5\n"


def test_a_value_parameter_is_deduced_from_the_size_of_an_array_passed_by_reference(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    template <unsigned N> unsigned fill(short (&out)[N]) {
      short tmp[N] = {};
      for (unsigned i = 0; i < N; ++i) tmp[i] = 10 + i;
      std::copy(tmp, tmp + N, out);
      return N;
    }
    void t() { short a[3]; unsigned n = fill(a); printf("%u %d %d\\n", n, a[0], a[2]); }
    """
    assert output(capsys, source) == "3 10 12\n"


def test_a_value_parameter_keeps_its_default_after_a_variadic_function() -> None:
    text = translate("template <int N = 4> int f(int a, ...) { return N; }", "t.C")
    assert "def f(a, *varargs, N=4):" in text
    typed = translate("template <typename T> T same(T x) { return x; } int n = same<int>(2);", "")
    assert "n = same__int(2)" in typed
    built = translate("template <typename T> struct Box { T v; }; auto b = Box<int>();", "")
    assert "b = Box__int()" in built


def test_std_copy_into_an_array_roots_method_gives_writes_it_from_its_start() -> None:
    source = "void t() { std::vector<int> r; std::copy(r.begin(), r.end(), gd.GetData()); }"
    assert "    transformed(r, 0, None, ROOT.gd.GetData(), 0, None)\n" in translate(source, "t.C")
    offset = source.replace("gd.GetData()", "gd.GetData() + n")
    assert "    transformed(r, 0, None, ROOT.gd.GetData(), ROOT.n, None)\n" in translate(offset, "")
    begun = source.replace("gd.GetData()", "v.begin() + 1 + k")
    assert "v = transformed(r, 0, None, ROOT.v, 1 + ROOT.k, None)" in translate(begun, "")
    member = source.replace("gd.GetData()", "box.items.begin()")
    assert "box.items = transformed(r, 0, None, ROOT.box.items, 0, None)" in translate(member, "")


def test_std_copy_without_a_destination_is_refused() -> None:
    from xrdroot.pycint import Refusal

    with pytest.raises(Refusal, match="std::copy without a range and a place to copy it to"):
        translate("void t() { int a[2]; std::copy(a, a + 2); }", "t.C")


def test_a_static_member_of_a_class_template_is_reached_through_its_instance() -> None:
    """``T::size`` of ``T`` an instance the translator made: the instance's own class member."""
    source = """
    template <typename T> struct Box { static int size; int v; };
    template <typename T> int size_of() { return T::size; }
    int f() { return size_of<Box<int>>(); }
    """
    python = translate(source, "t.C")
    assert "def size_of__Box_int():\n    return Box__int.size" in python


def test_a_reference_parameter_of_an_instance_takes_what_writes_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A variable the caller keeps by value goes into a cell of its own, given back after."""
    bumped = """
    template <typename T> void bump(T &e) { e += 1; }
    void t() { double x = 1.5; bump(x); printf("%d\\n", (int) x); }
    """
    python = translate(bumped, "t.C")
    assert "def bump__double(e):\n    e.value += 1" in python
    assert "written_after(bump__double(ref_1 := Cell(x)), x := ref_1.value)" in python
    assert output(capsys, bumped) == "2\n"
    element = """
    template <typename T> void bump(T &e) { e += 1; }
    void t() { double a[2] = {1.0, 2.0}; bump(a[0]); printf("%d %d\\n", (int) a[0], (int) a[1]); }
    """
    assert "bump__double(ItemRef(a, 0))" in translate(element, "t.C")
    assert output(capsys, element) == "2 2\n"
