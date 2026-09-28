"""The edges: names that clash, types known only by spelling, loops that are not counted."""

from __future__ import annotations

import io

import pytest

from cintfake import fake
from xrdroot.cint import Refusal, translate
from xrdroot.cint import runtime as rt
from xrdroot.cint.execute import run_source


def printed(capsys: pytest.CaptureFixture[str], source: str) -> str:
    run_source(source, "t.C", root=fake())
    return capsys.readouterr().out


def body(source: str) -> str:
    text = translate("void t() {\n" + source + "\n}\n", "t.C")
    compile(text, "t.C", "exec")
    return text


@pytest.mark.parametrize(
    ("source", "text"),
    [
        ('void t() { int len = 1; int len_ = 2; printf("%d %d\\n", len, len_); }', "1 2\n"),
        (
            "void t() { double d = 3; d /= 2; int i = 7; i %= 4; i <<= 1;"
            ' printf("%g %d\\n", d, i); }',
            "1.5 6\n",
        ),
        (
            "struct F { int x; }; void t() { auto f = F(); f.x = 7; int y = f.x / 2;"
            ' printf("%d\\n", y); }',
            "3\n",
        ),
        (
            "void t() { auto m = std::max(7, 2); int q = m / 2; auto a = abs(-7) / 2;"
            ' printf("%d %d\\n", q, a); }',
            "3 -3\n" if False else "3 3\n",
        ),
        (
            "void t() { int n = 4; int k = 0; for (int i = 0; i < n; i++) { n--; k++; }"
            ' printf("%d\\n", k); }',
            "2\n",
        ),
        (
            "void t() { std::vector<int> v; v.push_back(1); int k = 0;"
            " for (int i = 0; i < v.size(); i++) { if (v.size() < 3) v.push_back(i); k++; }"
            ' printf("%d\\n", k); }',
            "3\n",
        ),
        ('void t() { int x = 1; switch (x) { case 1: {} case 2: printf("fell\\n"); } }', "fell\n"),
        (
            "int f(int x) { switch (x) { case 1: return 5; default: return 6; } }"
            'void t() { printf("%d %d\\n", f(1), f(2)); }',
            "5 6\n",
        ),
        (
            "void t() { int n = 3; int k = 0; for (int i = 0; n > i; i++) k++;"
            ' printf("%d\\n", k); }',
            "3\n",
        ),
        ('void t() { struct P { int a; } p; p.a = 4; printf("%d\\n", p.a); }', "4\n"),
        (
            'void g(double *p) { *p = 2; } void f(double x) { g(&x); printf("%g\\n", x); }'
            "void t() { f(1); }",
            "2\n",
        ),
        (
            "int k(TH1F *h) { return 1; } int k(int n) { return 2; }"
            'void t() { printf("%d %d\\n", k(new TH1F("h", "h", 1, 0, 1)), k(3)); }',
            "1 2\n",
        ),
        (
            "struct A { int n = 5; int &r; A(int &x) : r(x) {} }; void t() { int v = 1; A a(v);"
            ' printf("%d\\n", a.n); }',
            "5\n",
        ),
        ('void f(int a); void f(int) {}\nvoid t() { f(1); printf("ok\\n"); }', "ok\n"),
        ('int Nope::counter = 3; void t() { printf("%d\\n", counter); }', "3\n"),
        (
            'void t() { const char * const p = "x"; enum class Mode m = Mode::kA;'
            ' printf("%s\\n", p); }'
            "enum class Mode { kA };",
            "x\n",
        ),
        ('void t() { int *x = new int(1); (*x) = 3; printf("%d\\n", *x); }', "3\n"),
        ('struct A { int a;; }; void t() { A x; printf("%d\\n", x.a); }', "0\n"),
        ('void t() { printf("%d|%*d|\\n", "A", -3, 1); }', "65|1  |\n"),
    ],
)
def test_the_edges_behave_as_cpp_does(
    capsys: pytest.CaptureFixture[str], source: str, text: str
) -> None:
    assert printed(capsys, source) == text


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("auto *v = (ROOT::Math::XYZVector *)p;", "v = ROOT.p"),
        ("auto s = (std::string)x;", "s = cstr(ROOT.x)"),
        ("auto r = (TMVA::Reader *)q;", "r = ROOT.q"),
        ("auto z = (Foo) 3;", "z = 3"),
        ("auto w = (Foo *) &x;", "w = ROOT.x"),
        ("std::array<int, N + 1> a;", "ROOT.std.array['int', ROOT.N + 1]()"),
        ("auto f = (double (*)(double))g;", "f = ROOT.g"),
        ("std::function<double (*)(double)> h;", "h = None"),
        ("bool same = nullptr == p;", "same = ROOT.p is None"),
        ("double a[3]; int n = sizeof(TH1F *) + sizeof(a);", "n = 8 + 24"),
        ("auto x = Klass::Missing();", "ROOT.Klass.Missing()"),
    ],
)
def test_the_edges_are_written_as_python(source: str, fragment: str) -> None:
    prefix = "struct Klass { Klass(int) {} };\n" if "Klass" in source else ""
    text = translate(prefix + "void t() {\n" + source + "\n}\n", "t.C")
    compile(text, "t.C", "exec")
    assert fragment in text


def test_a_second_unnamed_block_goes_on_the_first(capsys: pytest.CaptureFixture[str]) -> None:
    run_source('{ printf("a\\n"); }\n{ printf("b\\n"); }\n', "u.C", root=fake())
    assert capsys.readouterr().out == "a\nb\n"


def test_a_class_deriving_from_itself_reaches_nothing_of_roots() -> None:
    text = translate("class A : public A { void f() { fX = 1; } };", "t.C")
    assert "ROOT.fX = 1" in text


def test_out_of_class_members_of_a_class_template_are_its_methods() -> None:
    source = """
    template <typename T> struct Box { T v; void set(T x); };
    template <typename T> void Box<T>::set(T x) { v = x; }
    """
    assert "def set(self, x):" in translate(source, "t.C")


def test_a_specialisation_of_a_class_template_is_refused() -> None:
    with pytest.raises(Refusal, match="a specialisation of the class template Box"):
        translate("template <typename T> struct Box {};\ntemplate <> struct Box<int> {};", "t.C")


@pytest.mark.parametrize(
    ("source", "why"),
    [
        ("void t() {", "an expression was expected, and the end of the macro is there"),
        ("struct A { A operator/() { return *this; } };", r"the operator / with this many"),
        ("void t() { int n = sizeof(double[n]); }", "sizeof of a type whose size"),
    ],
)
def test_the_edges_that_have_no_python_are_refused(source: str, why: str) -> None:
    with pytest.raises(Refusal, match=why):
        translate(source, "t.C")


def test_streams_write_to_a_file_given_them_and_format_anything() -> None:
    target = io.StringIO()
    out = rt.ostream(target)
    out << object.__new__(type("Thing", (), {"__str__": lambda self: "thing"})) << rt.endl
    assert target.getvalue() == "thing\n"
    strings = rt.ostringstream()
    strings << "a" << rt.endl
    assert strings.str() == "a\n"


@pytest.mark.parametrize(
    ("source", "text"),
    [
        (
            'class B : public TObject { public: int x; }; void t() { B b; printf("%d\\n", b.x); }',
            "0\n",
        ),
        (
            "void t() { for (int i = 0; i < 3; i++) {"
            " switch (i) { case 1: continue; default: break; }"
            ' printf("%d", i); } printf("\\n"); }',
            "02\n",
        ),
        (
            'struct S { char buf[8]; std::string s; }; void t() { S o; sprintf(o.buf, "%d", 5);'
            ' o.s.append("x"); std::istringstream in("l"); std::getline(in, o.s);'
            ' printf("%s %s\\n", o.buf, o.s.c_str()); }',
            "5 l\n",
        ),
        ('void t() { int a[2] = {1, 2}; std::swap(a[0], a[1]); printf("%d\\n", a[0]); }', "2\n"),
        (
            "void t() { int y = 0; auto f = [&]() { int z = 0; z = 2; y = z; }; f();"
            ' printf("%d\\n", y); }',
            "2\n",
        ),
        (
            "int gInit = 4; int next() { static int n = gInit; return n++; }"
            'void t() { next(); printf("%d\\n", next()); }',
            "5\n",
        ),
        (
            "int f(int) { return 1; } void t() { int k = 0; for (int i = 0; i < f(i); i++) k++;"
            ' printf("%d\\n", k); }',
            "1\n",
        ),
        ('void t() { int x = 1; { int x = 2; printf("%d", x); } printf("%d\\n", x); }', "21\n"),
        (
            'void f(int a = 1); void f(int a = 1) { printf("%d\\n", a); } void f(int);'
            "void t() { f(); }",
            "1\n",
        ),
        (
            "void g(int &a, int &b) { a = 1; } void t() { int x = 0, arr[1] = {0}; g(x, arr[0]);"
            ' printf("%d\\n", x); }',
            "1\n",
        ),
        (
            "int f() noexcept { return 1; } int g() throw() { return 2; } enum E {};"
            'void t() { printf("%d%d\\n", f(), g()); }',
            "12\n",
        ),
    ],
)
def test_more_edges_behave_as_cpp_does(
    capsys: pytest.CaptureFixture[str], source: str, text: str
) -> None:
    assert printed(capsys, source) == text


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("auto *q = (Outer::TInner *)p;", "q = ROOT.p"),
        ("struct P { int a; }; P p; auto m = p.missing;", "m = p.missing"),
        ("auto x = f<>(1); auto y = df.Take<float, int>(2); int z = g((Int_t));", "f(1)"),
        ("auto [a, b] = pairs();", "a, b = ROOT.pairs()"),
        ("int (*arr)[3] = nullptr;", "arr = None"),
    ],
)
def test_more_edges_are_written_as_python(source: str, fragment: str) -> None:
    text = translate("void t() {\n" + source + "\n}\n", "t.C")
    compile(text, "t.C", "exec")
    assert fragment in text


def test_members_declared_after_their_definitions_and_types_named_through_classes() -> None:
    source = """
    struct A { void f() {} };
    void A::f();
    Outer::Inner var;
    struct K { K(int) {} };
    void t() { auto k = K::K(1); }
    """
    text = translate(source, "t.C")
    assert "var = ROOT.Outer.Inner()" in text
    assert "k = K(1)" in text


@pytest.mark.parametrize(
    ("source", "why"),
    [
        ("static_assert(1", "the macro ends in the middle"),
        ("void t() { int (*p) = nullptr; }", "assigning to something that is not a variable"),
        ("template <int N> struct Fixed { int a[N]; };", "N, a value a class template is given"),
        ("template <typename T> T half = T(1) / 2;", "a variable template"),
    ],
)
def test_more_edges_that_have_no_python_are_refused(source: str, why: str) -> None:
    with pytest.raises(Refusal, match=why):
        translate(source, "t.C")


@pytest.mark.parametrize(
    ("source", "text"),
    [
        ('char gBuf[8]; void t() { sprintf(gBuf, "%d", 3); printf("%s\\n", gBuf); }', "3\n"),
        (
            "int gOther = 0; void g(int &a, int &b = gOther) { a = 5; }"
            'void t() { int x = 0; g(x); printf("%d\\n", x); }',
            "5\n",
        ),
        ('void t() { { int x = 1; { int x = 2; printf("%d", x); } printf("%d\\n", x); } }', "21\n"),
        ('int n = 2; int value(n), m;\nvoid t() { printf("%d %d\\n", value, m); }', "2 0\n"),
    ],
)
def test_the_last_edges_behave_as_cpp_does(
    capsys: pytest.CaptureFixture[str], source: str, text: str
) -> None:
    assert printed(capsys, source) == text


def test_a_member_of_one_of_roots_classes_has_no_type_the_macro_knows() -> None:
    text = body("TH1F *h = nullptr; int x = h->fN / 2;")
    assert "x = int(div(h.fN, 2))" in text


def test_a_string_written_into_a_buffer_of_roots_is_assigned_there() -> None:
    assert "ROOT.extBuf = cformat('%d', 1)" in body('sprintf(extBuf, "%d", 1);')


def test_a_variable_initialised_from_a_name_is_not_a_function_declaration() -> None:
    text = translate("int value(unknownThing), m;", "t.C")
    assert "value = int(ROOT.unknownThing)" in text


def test_assigning_to_an_operators_result_is_refused() -> None:
    with pytest.raises(Refusal, match="assigning to something that is not a variable"):
        body("int x = 0; -x = 3;")
