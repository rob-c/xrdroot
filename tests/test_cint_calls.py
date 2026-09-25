"""Calls into the standard library, ROOT and the macro itself, as the Python they become."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.cint import Refusal, translate
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
        ('auto h = std::make_unique<TH1F>("h", "h", 2, 0, 1); printf("%s\\n", h->GetName());',
         "h\n"),
        ('std::shared_ptr<TH1F> s = std::make_shared<TH1F>("s", "s", 1, 0, 1);'
         'TH1F *raw = s.get(); printf("%s\\n", raw->GetName());', "s\n"),
        ("std::unique_ptr<TH1F> e = std::unique_ptr<TH1F>(); printf(\"%d\\n\", e == nullptr);",
         "1\n"),
        ('std::unique_ptr<TH1F> r(new TH1F("r", "r", 1, 0, 1)); TH1F *x = r.release();'
         'printf("%s\\n", x->GetName());', "r\n"),
        ('printf("%g %d %g\\n", std::numeric_limits<double>::max(),'
         " std::numeric_limits<int>::min(), std::numeric_limits<float>::epsilon());",
         "1.79769e+308 -2147483648 1.19209e-07\n"),
        ('double a[4] = {3, 1, 2, 0}; std::sort(a, a + 3); printf("%g %g\\n", a[0], a[3]);',
         "1 0\n"),
        ("std::vector<int> v; v.push_back(3); v.push_back(1); v.push_back(2);"
         'std::sort(v.begin() + 1, v.end()); printf("%d %d %d\\n", v[0], v[1], v[2]);',
         "3 1 2\n"),
        ('std::vector<int> v; v.push_back(1); v.push_back(2); std::reverse(v.begin(), v.end());'
         'printf("%d\\n", v[0]);', "2\n"),
        ('std::vector<double> v; v.push_back(1.5); v.push_back(2);'
         'printf("%g\\n", std::accumulate(v.begin(), v.end(), 0.0));', "3.5\n"),
        ('double w[3] = {1, 2, 3}; printf("%g\\n", std::accumulate(w, w + 2, 10.0));', "13\n"),
        ('printf("%d %d %d\\n", abs(-3), std::max(2, 5), std::min(1.5, 0.5) < 1);', "3 5 1\n"),
        ('long big = 5; unsigned char c = 260; printf("%ld %d\\n", big, c);', "5 4\n"),
        ('int x; x = {3}; printf("%d\\n", x);', "3\n"),
        ('std::string s = "a,b"; printf("%d\\n", s.find(",") != std::string::npos);', "1\n"),
        ('printf("%d %d\\n", kTRUE, kFALSE);', "1 0\n"),
        ('assert(1 + 1 == 2); printf("ok\\n");', "ok\n"),
    ],
)
def test_library_calls_do_what_the_library_does(
    capsys: pytest.CaptureFixture[str], source: str, text: str
) -> None:
    assert printed(capsys, "void t() {\n" + source + "\n}\n") == text


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("auto f = Format(\"x\");", "f = ROOT.Format('x')"),
        ("auto g = std::make_unique(3);", "g = ROOT.std.make_unique(3)"),
        ("auto a = (*fp)(2); auto b = f()();", "b = ROOT.f()()"),
        ("std::vector<std::array<int, 3>> v;", "ROOT.std.vector['std::array<int,3>']()"),
        ("fprintf(stderr, \"x\");", "fprintf('stderr', 'x')"),
    ],
)
def test_calls_are_written_as_their_python(source: str, fragment: str) -> None:
    assert fragment in body(source)


@pytest.mark.parametrize(
    ("source", "why"),
    [
        ("auto x = std::numeric_limits<long double>::max();", r"numeric_limits<long double>::max"),
        ("std::sort(v.begin());", "std::sort without a range"),
        ("auto s = std::accumulate(v.begin(), v.end());", "std::accumulate with an operation"),
        ("std::sort(v.begin(), w.end());", "a range whose two ends are in different containers"),
        ("std::sort(it, other);", "an iterator this translator cannot follow back"),
        ("std::unique_ptr<TH1F> p; long n = p.use_count();", r"the smart pointer's use_count\(\)"),
    ],
)
def test_calls_with_no_python_that_does_the_same_are_refused(source: str, why: str) -> None:
    with pytest.raises(Refusal, match=why):
        body(source)


def test_a_method_calls_its_classs_other_members_through_self(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    class Pair { public: static int count; static int Count() { return count; }
      int Twice() { return 2 * One(); } int One() { return 1; } };
    int Pair::count = 7;
    void t() { Pair p; printf("%d %d %d\\n", p.Twice(), Pair::Count(), Pair::count); }
    """
    assert printed(capsys, source) == "2 7 7\n"


def test_a_class_deriving_from_roots_reaches_roots_members_through_self() -> None:
    source = """
    class MySelector : public TSelector { public:
      Bool_t Process(Long64_t entry) { fChain->GetEntry(entry); return kTRUE; } };
    class Deeper : public MySelector { public: void Go() { fInput->Print(); } };
    class Plain { public: int fN; void Go() { fN = 1; fOther = 2; } };
    """
    text = translate(source, "t.C")
    assert "self.fChain.GetEntry(entry)" in text
    assert "self.fInput.Print()" in text
    assert "ROOT.fOther = 2" in text
