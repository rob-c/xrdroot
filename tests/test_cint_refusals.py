"""The C++ the translator refuses by name, and the corners of the grammar it still reads."""

from __future__ import annotations

import pytest

from xrdroot.pycint import Refusal, translate


def body(source: str) -> str:
    text = translate("void t() {\n" + source + "\n}\n", "t.C")
    compile(text, "t.C", "exec")
    return text


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("int *m = new int[3][4];", "m = array('int', (3, 4))"),
        ("int a = 1; ROOT::Internal::AtomicInc(&a);", "ROOT.Internal.AtomicInc(a)"),
        ('TH1F *h = new TH1F{"h"};', "h = ROOT.TH1F('h')"),
        ('::printf("x"); double r = ::gRandom->Rndm();', "r = ROOT.gRandom.Rndm()"),
        ("int v = obj.operator()(1);", "ROOT.obj.__call__(1)"),
        ("auto s = X::operator+(a, b);", "ROOT.X.__add__(ROOT.a, ROOT.b)"),
        ("int q = a < TH1F > +1;", "q = int((ROOT.a < ROOT.TH1F) > +1)"),
        ("double y = (TString) - x;", "y = ROOT.TString - ROOT.x"),
        ("auto z = (MyType) w;", "z = ROOT.w"),
        ("if (x = f(); x > 0) {}", "ROOT.x = ROOT.f()"),
        ("struct Foo *p = nullptr;", "p = None"),
        ("if (pass) pass = 0;", "if getattr(ROOT, 'pass'):\n        setattr(ROOT, 'pass', 0)"),
    ],
)
def test_the_corners_of_the_grammar_are_read(source: str, fragment: str) -> None:
    text = translate("void t() {\n" + source + "\n}\n", "t.C")
    assert fragment in text


@pytest.mark.parametrize(
    ("source", "why"),
    [
        ("goto end; end: ;", "a goto forward to a label, or across a loop's edge"),
        ("char buf[8]; TH1F *h = new (buf) TH1F();", "placement new"),
        ("auto a = alignof(TNone);", "alignof of a type whose size this translator does not know"),
        ("bool b = noexcept(f());", "noexcept is not something"),
        ("int x; auto f = [&n = x]() { return n; };", r"the lambda capture &n = \.\.\."),
        ("auto f = [n = 1]() mutable { return ++n; };", "the mutable lambda's capture n"),
        ("Point p = {.x = 1};", r"a designated initialiser, \.name = value"),
        ("int x = );", "an expression was expected"),
        ("unknown x y;", "';' was expected"),
        ("switch (x) case 1: break;", "a switch whose body is not a block"),
        ("switch (x) { case 1 ... 3: break; }", "a case range"),
        ('asm("nop");', "inline assembly"),
        ("typedef enum { kA } E;", "a typedef of an enum defined in place"),
        ("struct { int a; };", "an unnamed struct, which has no name"),
        ("int n = sizeof...(args);", "sizeof of a type whose size"),
        ("int x = 10_km;", "10_km is a user-defined literal"),
    ],
)
def test_constructs_with_no_python_are_refused_by_name_at_their_line(source: str, why: str) -> None:
    with pytest.raises(Refusal, match=why) as refused:
        body(source)
    assert refused.value.where is not None and refused.value.where.line == 2


@pytest.mark.parametrize(
    ("source", "why"),
    [
        ("struct A { void *operator new(size_t n); };", "a class's own operator new or delete"),
        ("template <typename... Args> void f(Args... args) {}", "a variadic template"),
        ("template <template <typename> int C> void f() {}",
         "a template template parameter that names a value"),
        ("auto f = [](auto... xs) { return 0; };", "a parameter pack, which a variadic"),
        ("int f() try { return 1; } catch (...) { return 0; }", "a function-try-block"),
        ("struct A { int bits : 3; };", "the bit-field bits"),
        (
            "template void f<int>(int);\nstruct B { B operator,(B o) { return *this; } };",
            "the operator , defined for a class",
        ),
        (
            "struct V { int x; };\nV operator&&(V a, V b) { return a; }",
            "the operator && defined outside a class",
        ),
        ("int main() { return 0; } }", "C\\+\\+ this translator cannot read"),
    ],
)
def test_declarations_with_no_python_are_refused_by_name(source: str, why: str) -> None:
    with pytest.raises(Refusal, match=why):
        translate(source, "t.C")
