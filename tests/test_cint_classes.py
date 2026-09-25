"""Functions, classes, references and the rest of C++ a macro declares, run and read back."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.cint import Refusal
from xrdroot.cint.execute import run_source


def output(capsys: pytest.CaptureFixture[str], source: str, *args: object) -> str:
    run_source(source, "t.C", tuple(args), root=fake())
    return capsys.readouterr().out


def test_references_and_addresses_write_back_to_the_callers_variable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    void twice(double &x) { x *= 2; }
    void set(int *p) { *p = 7; }
    void fill(double *a, int n) { for (int i = 0; i < n; i++) a[i] = i * 0.5; }
    void t() {
      double v = 1.5; int k = 0; double arr[3];
      twice(v); set(&k); fill(arr, 3); fill(&arr[1], 1);
      int a[4] = {1, 2, 3, 4}; int &first = a[0]; first = 9;
      Float_t px, py; gRandom->Rannor(px, py);
      printf("%g %d %g %g %d %d\\n", v, k, arr[1], arr[2], a[0], px != py);
    }
    """
    assert output(capsys, source) == "3 7 0 1 9 1\n"


def test_overloads_are_chosen_by_count_and_then_by_type(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    int f(int a) { return 1; }
    int f(double a) { return 2; }
    int f(int a, int b = 3) { return 10 + b; }
    int g(const char *s) { return 100; }
    int g(int n) { return 200; }
    void t() { printf("%d %d %d %d %d\\n", f(1), f(1.5), f(1, 2), g("x"), g(4)); }
    """
    assert output(capsys, source) == "1 2 12 100 200\n"


def test_a_class_is_built_and_used_as_cpp_builds_and_uses_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    class Counter {
    public:
      Counter(int start) : fN(start), fStep(1) {}
      Counter() : Counter(0) {}
      void Add() { fN += fStep; ++fTotal; }
      int Get() const { return fN; }
      static int Total() { return fTotal; }
      Counter operator+(const Counter &o) const { return Counter(fN + o.fN); }
      bool operator<(const Counter &o) const { return fN < o.fN; }
    private:
      int fN;
      int fStep;
      static int fTotal;
    };
    int Counter::fTotal = 0;
    struct Point { double x; double y; };
    struct Noisy { ~Noisy() { printf("gone\\n"); } };
    void t() {
      Counter *a = new Counter(5); a->Add(); Counter *b = new Counter(); b->Add(); b->Add();
      Counter *c = new Counter((*a + *b).Get());
      Point p = {1.5, 2};
      printf("%d %d %d %d %d %g\\n", a->Get(), b->Get(), c->Get(), Counter::Total(),
             b->Get() < a->Get(), p.x + p.y);
      Noisy *n = new Noisy(); delete n; delete b;
    }
    """
    assert output(capsys, source) == "6 2 8 3 1 3.5\ngone\n"


def test_a_local_object_whose_destructor_does_something_is_refused() -> None:
    source = 'struct Noisy { ~Noisy() { printf("gone"); } };\nvoid t() { Noisy n; }'
    with pytest.raises(Refusal, match=r"t.C:2: the local Noisy n, whose destructor C\+\+ runs"):
        run_source(source, "t.C", root=fake())


def test_inheritance_calls_the_base_and_overrides_the_virtual(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    class Shape { public: Shape(const char *n) : fName(n) {} virtual double Area() const { return 0; }
      const char *Name() const { return fName; } protected: const char *fName; };
    class Square : public Shape { public: Square(double s) : Shape("square"), fSide(s) {}
      double Area() const override { return fSide * fSide; } private: double fSide; };
    void t() {
      Shape *shapes[2] = {new Shape("dot"), new Square(3)};
      for (auto *s : shapes) printf("%s %g\\n", s->Name(), s->Area());
    }
    """
    assert output(capsys, source) == "dot 0\nsquare 9\n"


def test_lambdas_capture_by_value_and_by_reference(capsys: pytest.CaptureFixture[str]) -> None:
    source = """
    void t() {
      int n = 1; int total = 0;
      auto byValue = [n](int x) { return x + n; };
      auto byRef = [&](int x) { total += x; };
      n = 100;
      byRef(byValue(2)); byRef(5);
      std::vector<int> v; v.push_back(3); v.push_back(1); v.push_back(2);
      std::sort(v.begin(), v.end(), [](int a, int b) { return a > b; });
      printf("%d %d %d %d\\n", total, v[0], v[1], v[2]);
    }
    """
    assert output(capsys, source) == "8 3 2 1\n"


def test_templates_enums_namespaces_and_globals(capsys: pytest.CaptureFixture[str]) -> None:
    source = """
    namespace util { template <typename T> T twice(T x) { return x + x; } }
    enum Color { kNone, kSome = 5, kMore };
    enum class Mode { kA = 2, kB };
    int gCount = 0;
    const int N = 3;
    void bump() { gCount++; }
    int counter() { static int calls = 0; return ++calls; }
    void t() {
      for (int i = 0; i < N; i++) bump();
      counter(); counter();
      Mode m = Mode::kB;
      printf("%d %g %d %d %d %d %d\\n", util::twice(2), util::twice(1.25), kMore, (int)m,
             gCount, counter(), sizeof(double) * N);
    }
    """
    assert output(capsys, source) == "4 2.5 6 3 3 3 24\n"


def test_exceptions_are_thrown_and_caught(capsys: pytest.CaptureFixture[str]) -> None:
    source = """
    double safe(double x) { if (x < 0) throw std::runtime_error("negative"); return x; }
    void t() {
      try { safe(-1); } catch (const std::exception &e) { printf("caught %s\\n", e.what()); }
      try { throw 3; } catch (...) { printf("anything\\n"); }
    }
    """
    assert output(capsys, source) == "caught negative\nanything\n"


def test_casts_ternaries_commas_and_pointers_to_null(capsys: pytest.CaptureFixture[str]) -> None:
    source = """
    void t() {
      double d = 2.75; TH1F *h = nullptr; TH1F *g = new TH1F("g", "g", 10, 0, 1);
      int i = (int)d; int j = static_cast<int>(-d); float f = float(d);
      int k = (i > 1) ? i * 10 : -1;
      int c = (i++, i + 100);
      if (!h && g) printf("%d %d %g %d %d %d\\n", i, j, f, k, c, h == 0);
      unsigned int big = 1u << 31; long long huge = 1LL << 40;
      printf("%u %lld %d\\n", big, huge, 'a' - '0');
    }
    """
    assert output(capsys, source) == "3 -2 2.75 20 103 1\n2147483648 1099511627776 49\n"


def test_arguments_given_to_the_macro_reach_its_function(capsys: pytest.CaptureFixture[str]) -> None:
    source = 'void t(int n = 2, const char *what = "x") { printf("%d %s\\n", n * 2, what); }'
    assert output(capsys, source) == "4 x\n"
    assert output(capsys, source, 5, "y") == "10 y\n"
