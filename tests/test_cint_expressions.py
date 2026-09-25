"""C++ expressions, as the Python each becomes: pointers, addresses, stores, casts."""

from __future__ import annotations

import pytest

from xrdroot.cint import Refusal, translate


def body(source: str) -> str:
    text = translate("void t() {\n" + source + "\n}\n", "t.C")
    compile(text, "t.C", "exec")
    return text


def method(source: str) -> str:
    text = translate("struct S { int fN; S *fNext; S &Self() {\n" + source + "\n} };", "t.C")
    compile(text, "t.C", "exec")
    return text


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("double a[3]; double *p = a + 1;", "p = a[1:]"),
        ("int *q = nullptr; int *r = q + 2;", "r = q[2:]"),
        ('std::string s = "a" + name;', "s = 'a' + cstr(ROOT.name)"),
        ("bool n = p == NULL;", "n = ROOT.p is None"),
        ("double r = x / y; int m = x % y;", "r = div(ROOT.x, ROOT.y)"),
        ("double f = 7.5; double g = f % 2;", "fmod(f, 2)"),
        ("auto v = *it;", "v = deref(ROOT.it)"),
        ("struct P { int x; TH1F h; }; P o; int *px = &o.x; TH1F *ph = &o.h;", "ph = o.h"),
        ('int *p = nullptr; int *same = &*p; auto *s = &"abc";', "same = p"),
        ("TVirtualPad **pad = &gPad;", "pad = ROOT.gPad"),
        ("int *e = &v[0];", "e = ItemRef(ROOT.v, 0)"),
        (
            "int i = 0; int x = a[i]++; int y = o.n++; int *p = nullptr; int z = (*p)++;",
            "z = postinc(ItemRef(p, 0), 1)",
        ),
        ("int x = a[i]++;", "x = int(postinc(ItemRef(ROOT.a, ROOT.i), 1))"),
        ("int y = o.n--;", "y = int(postinc(AttrRef(ROOT.o, 'n'), -1))"),
        (
            "int *p = nullptr; int x = 0; x = p[0] = 1; x = *p = 3;",
            "x = set_attr(ItemRef(p, 0), 'value', 3)",
        ),
        ("int x = 0; x = a[0] = 2; x = o.n = 4;", "x = int(set_attr(ROOT.o, 'n', 4))"),
        ("w.import(x);", "getattr(ROOT.w, 'import')(ROOT.x)"),
        ('std::string s = "abc"; int c = s[0];', "c = char_at(s, 0)"),
        ("TH1F *h = dynamic_cast<TH1F *>(o);", "h = dynamic_cast(ROOT.TH1F, ROOT.o)"),
        ("const char *c = (const char *)ts;", "c = cstr(ROOT.ts)"),
        ("double d = Double_t{2};", "d = 2.0"),
        ("int x = y ? 1 : throw 2;", "throw(2)"),
        ('std::string *s = new std::string("x");', "s = str('x')"),
        ("bool b; float f; double d; char c; TH1F *h;", "b = False"),
        ("int k = 5; int m = -k + ~k + +k;", "m = -k + ~k + +k"),
        ("unsigned u = 3; unsigned v = u << 2; unsigned w = u >> 1;", "v = u << 2"),
        ("bool ok = !h; bool bad = !x;", "bad = not ROOT.x"),
        ("TH1F *h = nullptr; bool ok = !h; if (h && !h->IsZombie()) {}", "if h is not None and"),
    ],
)
def test_expressions_become_the_python_that_does_the_same(source: str, fragment: str) -> None:
    assert fragment in body(source)


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("return *this;", "return self"),
        ("S *me = this; return *me;", "me = self"),
        ("int *p = &fN; return *this;", "p = AttrRef(self, 'fN')"),
        ("int x = fN++; return *this;", "x = postinc(AttrRef(self, 'fN'), 1)"),
        ("int y = 0; y = fN = 2; return *this;", "y = set_attr(AttrRef(self, 'fN'), 'value', 2)"),
        ("fNext->fN = 1; return *fNext;", "self.fNext.fN = 1"),
    ],
)
def test_a_methods_members_are_its_objects(source: str, fragment: str) -> None:
    assert fragment in method(source)


@pytest.mark.parametrize(
    ("source", "why"),
    [
        ("int x = a.*p;", r"t.C:2: the operator \.\*"),
        ("int x = a <=> b;", "the operator <=>"),
        ('const char *s = "abc"; const char *t = s + 1;', r"pointer arithmetic on a char\*"),
        ("double a[3]; double *p = a - 1;", r"pointer arithmetic on a double\[\]"),
        ("TH1F *hs[2]; TH1F **p = hs + 1;", r"pointer arithmetic on a TH1F\* is"),
        ("std::ifstream in; if (in >> x) {}", "reading from a stream with >> inside"),
        ("int x = f()++;", "changing this in the middle of an expression"),
        ("int x = gCount++;", "changing this in the middle of an expression"),
        ("h->~TH1F();", "calling the destructor ~TH1F by hand"),
        ("int n = sizeof(TH1F);", "sizeof of a type whose size"),
        ("int a = (delete p, 0);", "delete inside an expression"),
        ("std::string s; s[0] = 'x';", "changing one character of a string in place"),
        ("TH1F *h; *h = *g;", "assigning a whole object through a pointer"),
        ("m(0, 1) = 2;", r"assigning to what a call returns by reference"),
    ],
)
def test_what_has_no_python_that_does_the_same_is_refused(source: str, why: str) -> None:
    with pytest.raises(Refusal, match=why):
        body(source)
