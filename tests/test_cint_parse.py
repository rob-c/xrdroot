"""The C++ the parser reads: each snippet is translated, and the Python it gives checked.

These are about reading - that a declaration is taken for a declaration, a
cast for a cast, a template argument list for one - so each asserts a line
or two of the translation rather than running it.
"""

from __future__ import annotations

import pytest

from xrdroot.cint import translate


def python(source: str) -> str:
    text = translate(source, "t.C")
    compile(text, "t.C", "exec")
    return text


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        # namespaces, extern "C", aliases and templates at the top of a macro
        ("namespace a::b { int f() { return 1; } }\nint t() { return a::b::f(); }", "return f()"),
        ("namespace x = std;\nint t() { return 1; }", "def t():"),
        (
            'extern "C" { int f(int); }\nextern "C" int g(int);\n'
            "extern int h;\nint t() { return 0; }",
            "def t():",
        ),
        ('extern "C" double f(double x) { return x; }', "def f(x):"),
        ("inline namespace v1 { int f() { return 2; } }", "def f():"),
        ("inline int f() { return 2; }", "def f():"),
        ("template class std::vector<int>;\nint t() { return 0; }", "def t():"),
        ("template <class T> struct Box;\nint t() { return 0; }", "def t():"),
        (
            "template <typename T = double> T first(T x) { return x * 3; }",
            "def first(x):",
        ),
        ("template <typename> int count() { return 0; }", "def count():"),
        (
            "template <typename T> struct Box { T v; };\nvoid t() { Box<int> *b = nullptr; }",
            "b = None",
        ),
        ("using Real = double;\nReal t(Real x) { return x / 2; }", "return x / 2"),
        ("typedef unsigned int UI;\nUI t(UI a) { return a / 2; }", "return idiv(a, 2)"),
        ("typedef int A, *PA;\nint t() { return 0; }", "def t():"),
        ("[[nodiscard]] int t() { return 1; }", "def t():"),
        (";\nint t() { return 1; }", "def t():"),
        # functions and their declarations
        ("int f(int);\nint f(int a) { return a; }", "def f(a):"),
        ("int f(int a = 3);\nint f(int b) { return b; }", "def f(b=3):"),
        ("void f(void) {}", "def f():"),
        ("void f(int, double) {}", "def f(arg0, arg1):"),
        ("void f(double x[], int n) {}", "def f(x, n):"),
        ("void f(double (*g)(double)) { g(1); }", "g(1)"),
        ("int printfish(const char *fmt, ...) { return 0; }", "def printfish(fmt, *varargs):"),
        ("auto f() -> int { return 1; }", "def f():"),
        ("int f() noexcept(true) { return 1; }\nint g() throw() { return 2; }", "def g():"),
        ("void f() const volatile & {}", "def f():"),
        ('TH1F h("h", "t", 10, 0, 1);', "h = ROOT.TH1F('h', 't', 10, 0, 1)"),
        ("TCanvas c(a, b);\nvoid t() {}", "c = ROOT.TCanvas(ROOT.a, ROOT.b)"),
        ("int values[] = {1, 2}, *p = nullptr, n(3);", "n = 3"),
        ("double (*fp)(double) = nullptr;", "fp = None"),
        ("static const int kN = 4;", "kN = 4"),
        # classes
        ("struct P; class Q;\nstruct P { int a; };", "class P:"),
        (
            "class A final : public TObject, private virtual B { public: A() {} };",
            "class A(ROOT.TObject, ROOT.B):",
        ),
        (
            "class A { friend class B; friend int f(A &a) { return 1; } using T = int; };",
            "class A:",
        ),
        ("class A { public slots: void Go() {} signals: void Done(); };", "def Go(self):"),
        ("class A { template <typename T> T get(T x) { return x; } };", "def get(self, x):"),
        (
            "class A { enum { kOne = 1 }; struct In { int x; }; int f() { return kOne; } };",
            "return A.kOne",
        ),
        ('class A { typedef double R; R r; static_assert(1, "x"); };', "self.r = 0.0"),
        ("struct S { int a; } s1, *s2;", "s1 = S()"),
        ("struct A { A(); ~A(); int n; };\nA::A() : n(2) {}\nA::~A() {}", "def _destruct(self):"),
        ("struct A { explicit A(int x) : n{x} {} int n; };", "self.n = x"),
        ("struct A { virtual void f() = 0; void g() = delete; };", "class A:"),
        (
            "struct V { double x; V operator-() const { V v; v.x = -x; return v; } };",
            "def __neg__(self):",
        ),
        ("struct V { double x; operator double() const { return x; } };", "def __float__(self):"),
        (
            "struct V { double d[3]; double operator[](int i) const { return d[i]; } };",
            "def __getitem__(self, i):",
        ),
        ("struct V { int n; bool operator()(int a) { return a > n; } };", "def __call__(self, a):"),
        ("enum class E : short { kA, kB = kA + 2 };", "class E:"),
        ("enum E { kA = -1, kB, kC = 'c', kD };", "kD = 100"),
        ("enum Forward : int;\nint t() { return 0; }", "def t():"),
        ("enum { kX = 1 << 2, kY };", "kY = kX + 1"),
        ("enum Shade { kLight } shade;", "shade = 0"),
        ("union U { int i; float f; };", "class U:"),
    ],
)
def test_declarations_are_read_as_cpp_declares_them(source: str, fragment: str) -> None:
    assert fragment in python(source)


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        # declarations and expressions a statement may start with
        ("int a = 1; a * 2;", "a * 2"),
        ("TH1F *h; h->Draw();", "h.Draw()"),
        ("std::vector<double> v(3);", "v = ROOT.std.vector['double'](3)"),
        ("std::map<std::string, int> m;", "m = ROOT.std.map['std::string', 'int']()"),
        ("std::vector<std::vector<int>> vv;", "vv = ROOT.std.vector['std::vector<int>']()"),
        ("vector<int> v; string s;", "s = ''"),
        ("auto [a, b] = f();", "a, b = ROOT.f()"),
        ('static_assert(sizeof(int) == 4, "x");', "pass"),
        ("unsigned long long u = 1; signed char c = 1; long double d = 1; short s = 2;", "d = 1.0"),
        ("const auto &r = h; auto *p = new TH1F(); decltype(p) q = p;", "q = p"),
        ("int x = (int)3.7 + (Int_t)(2.5) + int(1.5) + int{2} + (double)1;", "x = int("),
        ("double d = (TH1F *)0 == nullptr; TObject *o = (TObject *)h;", "o = ROOT.h"),
        ("int y = (x) * 2 + (z);", "y = int(ROOT.x * 2 + ROOT.z)"),
        ("Float_t f = (Float_t)-1;", "f = f32("),
        ("int n = sizeof(double); int m = sizeof n; int k = sizeof(int[4]);", "k = 16"),
        ("TH1F *h = new TH1F; double *d = new double[3]; int *i = new int(4);", "d = array("),
        (
            "TH1F *hs = new TH1F[2]; TH1F **hp = new TH1F *[2]; delete [] hs; delete hp;",
            "hp = [None] * 2",
        ),
        ("TH1F *h = ::new TH1F(); ::delete h;", "delete(h)"),
        ("auto f = [&, x](int a) mutable -> int { return a; };", "def lambda_"),
        ("auto f = [=, this]() { return 1; }; auto g = [*this] { return 2; };", "def lambda_"),
        ("int a[2][3] = {{1, 2}, {3}}; a[1][2] = a[0][1];", "a[1][2] = a[0][1]"),
        ('std::string s = "a" "b"; std::string t = "c"s;', "s = 'ab'"),
        ("char c = 'ab';", "c = 24930"),
        ("bool b = true or false and not true; int x = 1 bitand 2 bitor 3 xor 4;", "x = "),
        ("int i = 0; i not_eq 2;", "i != 2"),
        ("TMath::Pi(); ROOT::Math::XYZVector v(1, 2, 3);", "v = ROOT.Math.XYZVector(1, 2, 3)"),
        ('auto h = df.Histo1D<double>("x");', "h = ROOT.df.Histo1D['double']('x')"),
        ('auto t = df.template Take<float>("x");', "ROOT.df.Take['float']('x')"),
        ("auto x = std::get<0>(tup);", "ROOT.std.get[0](ROOT.tup)"),
        ("bool c = a < b && c > (d);", "ROOT.a < ROOT.b and"),
        ("auto v = std::vector<int>{1, 2};", "ROOT.std.vector['int'](1, 2)"),
        ("TLorentzVector v = TLorentzVector(1, 2, 3, 4);", "v = ROOT.TLorentzVector(1, 2, 3, 4)"),
        ("int a, b; a = b = 3; a += 2, b -= 1;", "a = (b := 3)"),
        ("x ? y : z;", "ROOT.y if ROOT.x else ROOT.z"),
        ("int k = 0; k = k ? 1 : k ? 2 : 3;", "k = 1 if k else 2 if k else 3"),
        ('double w = v[0] + m["k"];', "ROOT.v[0] + ROOT.m['k']"),
        ("if (auto *h = f(); h) {}", "h = ROOT.f()"),
        ("if (TH1 *h = (TH1 *)f()) h->Draw();", "if h is not None:"),
        ("if constexpr (true) {} else if (false) {} else {}", "elif False:"),
        ("while (TObject *o = next()) o->Print();", "while True:"),
        ("for (;;) break;", "while True:"),
        ("for (auto &[k, v] : m) {}", "for k, v in iterate(ROOT.m):"),
        ("for (int x : {1, 2, 3}) {}", "for x in [1, 2, 3]:"),
        ("try { f(); } catch (std::exception &) { } catch (...) { }", "except Exception:"),
        ("using namespace std; using std::cout; using V = std::vector<int>;", "pass"),
        ("typedef std::vector<int> V; V v;", "v = ROOT.std.vector['int']()"),
        ("struct Local { int a; }; Local l;", "l = Local()"),
        ("enum Local { kA, kB }; int b = kB;", "b = kB"),
        ("enum class Scoped { kA }; Scoped s = Scoped::kA;", "s = Scoped.kA"),
        ("[[maybe_unused]] int unused = 1;", "unused = 1"),
        ("switch (x) { case 1: { int y = 2; break; } default: break; }", "if switch_1 == 1:"),
        ("do x++; while (x < 3);", "while True:"),
    ],
)
def test_statements_are_read_as_cpp_reads_them(body: str, fragment: str) -> None:
    text = translate("void t() {\n" + body + "\n}\n", "t.C")
    compile(text, "t.C", "exec")
    assert fragment in text
