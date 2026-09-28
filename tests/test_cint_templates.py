"""Function templates with value parameters: ``template <unsigned N>``, given or deduced."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.cint import translate
from xrdroot.cint.execute import run_source


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
    assert "times(2, N=3)" in translate(source, "t.C")
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
    assert "n = int(same(2))" in typed
    built = translate("template <typename T> struct Box { T v; }; auto b = Box<int>();", "")
    assert "b = Box()" in built


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
    from xrdroot.cint import Refusal

    with pytest.raises(Refusal, match="std::copy without a range and a place to copy it to"):
        translate("void t() { int a[2]; std::copy(a, a + 2); }", "t.C")
