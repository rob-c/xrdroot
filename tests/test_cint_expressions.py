"""C++ expressions, as the Python each becomes: pointers, addresses, stores, casts."""

from __future__ import annotations

import pytest

from xrdroot.pycint import Refusal, translate


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
        ("TTreeReaderValue<float> px(r, \"px\"); double s = *px + *px;",
         "s = f32(px.__deref__() + px.__deref__())"),
        ("ROOT::RVecD p{1., 2.}; auto g = p[p > 1 && abs(p) < 2];",
         "g = p[(p > 1) & (cabs(p) < 2)]"),
        ("RVecF p; auto g = p > 1 || p < 0;", "g = (p > 1) | (p < 0)"),
        ("ROOT::RVecD p; double h = 1; auto k = p * h; bool ok = h > 1 && h < 2;",
         "ok = bool(h > 1 and h < 2)"),
        ("ROOT::RVecD p; std::cout << p << std::endl;", "cout << p << endl"),
        ("struct P { int x; TH1F h; }; P o; int *px = &o.x; TH1F *ph = &o.h;", "ph = o.h"),
        ('int *p = nullptr; int *same = &*p; auto *s = &"abc";', "same = p"),
        ("TVirtualPad **pad = &gPad;", "pad = ROOT.gPad"),
        ("int *e = &v[0];", "e = ItemRef(ROOT.v, 0)"),
        ("int *f = &v.front();", "f = ItemRef(ROOT.v, 0)"),
        ("int *g = &v.at(2);", "g = ItemRef(ROOT.v, 2)"),
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
        ("const char *c = (const char *)ts;", "c = char_pointer(ROOT.ts)"),
        ("double d = Double_t{2};", "d = 2.0"),
        ("int x = y ? 1 : throw 2;", "throw(2)"),
        ('std::string *s = new std::string("x");', "s = str('x')"),
        ("bool b; float f; double d; char c; TH1F *h;", "b = False"),
        ("int k = 5; int m = -k + ~k + +k;", "m = -k + ~k + +k"),
        ("unsigned u = 3; unsigned v = u << 2; unsigned w = u >> 1;", "v = u << 2"),
        ("bool ok = !h; bool bad = !x;", "bad = not ROOT.x"),
        ("TH1F *h = nullptr; bool ok = !h; if (h && !h->IsZombie()) {}", "if h is not None and"),
        # 8 bytes into an object is no object here: refused as it runs unless the offset is 0
        ("TH1F *h = nullptr; h = (TH1F *)(((char *)h) - 8);", "h = byte_offset(h, -(8))"),
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
        ('const char *s = "abc"; const char *t = s - 1;', r"pointer arithmetic on a char\*"),
        ("double a[3]; double *p = a - 1;", r"pointer arithmetic on a double\[\]"),
        ("TH1F *hs[2]; TH1F **p = hs + 1;", r"pointer arithmetic on a TH1F\* is"),
        # the bytes of numbers, unlike an object's, are not one Python object to stand on
        ("double *d = nullptr; char *c = ((char *)d) - 8;", r"pointer arithmetic on a char\*"),
        ("std::ifstream in[2]; if (in[0] >> x) {}", "from a stream that is not a variable"),
        ("int x = f()++;", "changing this in the middle of an expression"),
        ("int x = gCount++;", "changing this in the middle of an expression"),
        ("h->~TH1F();", "calling the destructor ~TH1F by hand"),
        ("int n = sizeof(TUnlaidOut);", "sizeof of a type whose size"),
        ("int a = (delete p, 0);", "delete inside an expression"),
        ("char *p = text(); p[0] *= 2;", r"changing a character through a char\* with \*="),
        ("char *p = text(); char c = *p--;", r"moving a char\* back with --"),
        ("char *p = text(); p = p - 1;", r"pointer arithmetic on a char\*"),
    ],
)
def test_what_has_no_python_that_does_the_same_is_refused(source: str, why: str) -> None:
    with pytest.raises(Refusal, match=why):
        body(source)


def test_a_character_of_a_string_passed_by_value_is_refused() -> None:
    with pytest.raises(Refusal, match="changing one character of a string in place"):
        translate("void f(std::string s) { s[0] = 'x'; }", "t.C")


def test_address_of_compound_assignment_borrows_the_cell() -> None:
    import xrdroot.pyroot as ROOT
    from xrdroot.pycint.execute import run_source

    cpp = """
    int f() {
        int a = 1;
        int* r = &a;
        int* p = &(a += 2);
        return *r + *p + a;
    }
    """
    assert run_source(cpp, "f.C", (), root=ROOT) == 9


def test_address_of_prefix_increment_borrows_the_cell() -> None:
    import xrdroot.pyroot as ROOT
    from xrdroot.pycint.execute import run_source

    cpp = """
    int f() {
        int a = 1;
        int* r = &a;
        int* q = &(++a);
        return *r + *q + a;
    }
    """
    assert run_source(cpp, "f.C", (), root=ROOT) == 6


def test_address_of_an_unaddressable_update_is_refused() -> None:
    with pytest.raises(Refusal, match="destination is addressable"):
        body("int a = 1; int* p = &(a += 2); return *p;")


def test_address_of_dereference_compound_assignment_borrows_the_pointer() -> None:
    import xrdroot.pyroot as ROOT
    from xrdroot.pycint.execute import run_source

    cpp = """
    int f() {
        int b = 1;
        int* p = &b;
        int* q = &(*p += 2);
        return *q + *p;
    }
    """
    assert run_source(cpp, "f.C", (), root=ROOT) == 6


def test_address_of_an_unaddressable_prefix_increment_is_refused() -> None:
    with pytest.raises(Refusal, match="destination is addressable"):
        body("int a = 1; int* q = &(++a); return *q;")


def test_address_of_class_pointer_compound_assignment_is_refused() -> None:
    cpp = (
        "struct S { int v; S& operator+=(int o) { v += o; return *this; } };\n"
        "void f() { S s; s.v = 1; S* p = &s; S* q = &(*p += 5); }\n"
    )
    with pytest.raises(Refusal, match="destination is addressable"):
        translate(cpp, "t.C")


def test_a_standard_typedef_written_as_a_call_is_the_scalar_cast_it_is():
    """``std::size_t(3.7)`` is C++'s conversion to the type the typedef names - 3, and not a
    call of a name the runtime has no ``size_t`` for - and ``uint8_t(300)`` is 44."""
    assert "= u64(3.7)" in body("double d = std::size_t(3.7);")
    assert "= 44" in body("double b = uint8_t(300);")


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        # Two arguments is no cast, so the name stays the runtime's namespace's.
        ("std::size_t a = std::size_t(1, 2);", "ROOT.std.size_t(1, 2)"),
        # A template argument list is no cast either.
        ("std::size_t<int> a = std::size_t<int>(1);", "ROOT.std.size_t['int'](1)"),
        # And a name that is no typedef of the standard's is the runtime's own call.
        ("double a = std::sqrt(4.0);", "sqrt(4.0)"),
    ],
)
def test_a_call_of_something_that_is_no_standard_typedef_stays_a_call(
    source: str, fragment: str
) -> None:
    assert fragment in body(source)
