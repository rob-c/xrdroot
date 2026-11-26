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


def test_a_template_parameter_stands_for_the_class_template_the_call_gives(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``template <template <class T> class Cont>`` gives ``Cont`` a template and not a type, so
    ``Cont<double>`` in the body is that template written with ``double`` - ROOT's own
    ``roottest/root/io/cpp11Containers/commonUtils.h`` writes exactly this and passes
    ``std::forward_list``, ``std::list`` and the rest."""
    source = """
    #include <vector>
    template <template <typename T> class Cont>
    void fill(const char* name) {
       Cont<double> c {1., 2.};
       printf("%s %d\\n", name, (int)c.size());
    }
    void t() { fill<std::vector>("vec"); }
    """
    assert "std.vector['double']" in translate(source, "t.C")
    assert output(capsys, source) == "vec 2\n"
    two = """
    #include <vector>
    #include <list>
    template <template <typename T> class Cont>
    void fill(const char* name) {
      Cont<double> c {1., 2.}; printf("%s %d\\n", name, (int)c.size());
    }
    void t() { fill<std::vector>("vec"); fill<std::list>("list"); }
    """
    translated = translate(two, "t.C")
    first, second = translated.split("def fill__std_list")
    assert "std.vector['double']" in first and "std.list['double']" in second


def test_a_template_parameter_written_as_an_expression_is_the_template_it_stands_for() -> None:
    """``C<int> make() { return C<int>(); }`` with ``C`` a template parameter: the return type and
    the expression both name the template the call gave, so the instance answers ``Box<int>``."""
    source = """
    template <class T> struct Box { int v = 1; };
    template <template <class T> class C> C<int> make() { return C<int>(); }
    void t() { Box<int> b = make<Box>(); (void)b; }
    """
    translated = translate(source, "t.C")
    instance = translated[translated.index("def make__Box") :]
    assert instance.startswith("def make__Box():\n    return Box__int()")


def test_a_template_argument_naming_a_constexpr_is_the_number_it_stands_for() -> None:
    """``constexpr int N = 5;`` then ``SMatrix<double, N, N>``: C++ folds an integral constant into
    the argument, so the matrix is named by 5 and 5 - and a plain ``int N = 5`` is no constant
    expression, so a value the translator does not work out stays the name it was written as."""
    fold = "constexpr int N = 5;\nvoid t(){ SMatrix<double, N, N> m; (void)m; }"
    const = "const int N = 5;\nvoid t(){ SMatrix<double, N, N> m; (void)m; }"
    plain = "int N = 5;\nvoid t(){ SMatrix<double, N, N> m; (void)m; }"
    assert "SMatrix['double', 5, 5]" in translate(fold, "t.C")
    assert "SMatrix['double', 5, 5]" in translate(const, "t.C")
    assert "SMatrix['double', N, N]" in translate(plain, "t.C")


def test_a_template_argument_naming_an_enumerator_is_its_number() -> None:
    """An enumerator is a constant too: one written as a number is that number, and one without is
    the one before it plus one - the first is zero."""
    written = "enum { kA = 3, kB };\nvoid t(){ SMatrix<double, kA, kB> m; (void)m; }"
    counted = "enum Color { kRed, kGreen };\nvoid t(){ SMatrix<double, kRed, kGreen> m; (void)m; }"
    assert "SMatrix['double', 3, 4]" in translate(written, "t.C")
    assert "SMatrix['double', 0, 1]" in translate(counted, "t.C")


def test_a_value_parameter_naming_a_template_argument_is_the_value_it_was_given() -> None:
    """``SMatrix<double, Rows, Cols>`` inside ``template <unsigned Rows, unsigned Cols>``: a value
    parameter is a value in its own body, so the argument is the number the call gives and not a
    name a library is asked about."""
    source = """
    template <unsigned int Rows, unsigned int Cols>
    void fill() { SMatrix<double, Rows, Cols> m; (void)m; }
    void t() { fill<3, 5>(); }
    """
    assert "SMatrix['double', 3, 5]" in translate(source, "t.C")


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
