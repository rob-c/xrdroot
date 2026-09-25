"""Statements run as C++ runs them: strings written into, loops, switches, jumps, exceptions."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.cint import Refusal, translate
from xrdroot.cint.execute import run_source


def printed(capsys: pytest.CaptureFixture[str], body: str) -> str:
    run_source("void t() {\n" + body + "\n}\n", "t.C", root=fake())
    return capsys.readouterr().out


@pytest.mark.parametrize(
    ("body", "text"),
    [
        ('char b[8]; snprintf(b, 4, "%d", 123456); printf("%s\\n", b);', "123\n"),
        ('char b[8]; strcpy(b, "abc"); strncpy(b, "wxyz", 2); printf("%s\\n", b);', "wx\n"),
        ('char b[8] = "ab"; strncat(b, "cdef", 2); printf("%s\\n", b);', "abcd\n"),
        ('int a = 1, b = 2; std::swap(a, b); printf("%d %d\\n", a, b);', "2 1\n"),
        (
            'std::string s = "a"; s.append("b"); s.push_back(\'c\'); printf("%s\\n", s.c_str());',
            "abc\n",
        ),
        ('std::string s = "a"; s.clear(); printf("[%s]\\n", s.c_str());', "[]\n"),
        ('TString s; s.Form("n=%d", 3); printf("%s\\n", s.Data());', "n=3\n"),
        (
            'std::unique_ptr<TH1F> p; p.reset(new TH1F("h", "h", 1, 0, 1)); p.reset();'
            'printf("%d\\n", p == nullptr);',
            "1\n",
        ),
        (
            'std::istringstream in("one two"); std::string w; std::getline(in, w);'
            'printf("%s\\n", w.c_str());',
            "one two\n",
        ),
        (
            'int v[4] = {1, 2, 3, 4}; for (auto &x : v) x *= 10; printf("%d %d\\n", v[0], v[3]);',
            "10 40\n",
        ),
        ('std::vector<int> w; w.push_back(1); for (int x : w) printf("%d\\n", x);', "1\n"),
        ('for (int i = 5; i != 0; i--) if (i == 3) printf("%d\\n", i);', "3\n"),
        ('for (int i = 0; i <= 2; i++) printf("%d", i); printf("\\n");', "012\n"),
        ('for (int i = 3; i > 0; i -= 2) printf("%d", i); printf("\\n");', "31\n"),
        (
            'for (int i = 0; i < 10; i += 3) { if (i == 3) continue; printf("%d", i); }'
            'printf("\\n");',
            "069\n",
        ),
        ('for (int i = 0; i < 4; i += 2) { i = i; printf("%d", i); } printf("\\n");', "02\n"),
        ('int n = 3; int *p = &n; for (int i = 0; i < n; ++i) *p = 1; printf("%d\\n", n);', "1\n"),
        ('for (double x = 0; x < 1; x += 0.5) printf("%g ", x); printf("\\n");', "0 0.5 \n"),
        ('int k = 0; for (int i = 0; i < 2.5; i++) k++; printf("%d\\n", k);', "3\n"),
        ('int i = 0; do { if (++i < 3) continue; } while (i < 5); printf("%d\\n", i);', "5\n"),
        (
            'int x = 2; switch (x) { case 1: printf("one"); case 2: printf("two"); '
            'default: printf("dflt"); } printf("\\n");',
            "twodflt\n",
        ),
        ('int x = 9; switch (x + 0) { case 1: break; default: printf("d\\n"); break; }', "d\n"),
        ('switch (3) { default: printf("only\\n"); }', "only\n"),
        (
            'int x = 0; switch (x) { case 1: printf("a"); break; case 0: '
            'if (x == 0) { printf("z"); } break; } printf("\\n");',
            "z\n",
        ),
        (
            'try { throw std::invalid_argument("bad"); } catch (std::exception &e) '
            '{ printf("%s\\n", e.what()); }',
            "bad\n",
        ),
        (
            'try { try { throw 1; } catch (...) { throw; } } catch (...) { printf("again\\n"); }',
            "again\n",
        ),
        ('if (0) printf("a"); else if (1) printf("b"); else printf("c"); printf("\\n");', "b\n"),
        (
            'int x = 1; if (x == 0) {} else if (auto f = [](){ return 1; }; f()) printf("f\\n");',
            "f\n",
        ),
    ],
)
def test_statements_do_what_cpp_does(
    capsys: pytest.CaptureFixture[str], body: str, text: str
) -> None:
    assert printed(capsys, body) == text


def test_a_void_function_returns_nothing_and_a_function_stored_in_one_returns_its_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    void quiet(int x) { if (x) return; printf("no\\n"); }
    auto make() { return [](int a) { return a * 2; }; }
    void t() { quiet(1); quiet(0); auto f = make(); printf("%d\\n", f(4)); }
    """
    run_source(source, "t.C", root=fake())
    assert capsys.readouterr().out == "no\n8\n"


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ("void inner() {}", "a function defined inside another function"),
        ("switch (x) { int y = 1; case 1: break; }", "a statement in a switch before its first"),
        ("switch (x) { case 1: { case 2: break; } }", "a case label outside the top of a switch"),
        (
            'for (;;) { switch (x) { case 1: continue; case 2: printf("a"); case 3: break; } }',
            "a continue inside a switch whose cases fall through",
        ),
        ("try {} catch (int) {} catch (double) {}", "several catch clauses for different types"),
        ("char b[4]; sprintf(b);", "a C string function with too few arguments"),
        ("std::swap(a);", "std::swap of other than two things"),
        ("std::getline(in);", "std::getline without a string to read into"),
        (
            "auto x = std::max_element(v.begin(), v.end());",
            r"std::max_element\(\) where its result",
        ),
        ("std::string s; s.reserve(3);", r"std::string::reserve\(\)"),
    ],
)
def test_statements_without_python_that_does_the_same_are_refused(body: str, why: str) -> None:
    source = "void t() {\n" + body + "\n}\n"
    with pytest.raises(Refusal, match=why):
        translate(source, "t.C")


def test_sprintf_into_a_buffer_the_caller_passed_is_refused() -> None:
    with pytest.raises(Refusal, match="writing a string into a buffer the caller passed"):
        translate('void t(char *out) { sprintf(out, "%d", 1); }', "t.C")


@pytest.mark.parametrize(
    ("body", "text"),
    [
        ('short s = 32767; s += 1; printf("%d\\n", s);', "-32768\n"),
        ('int x = 3; -x; printf("%d\\n", x);', "3\n"),
        (
            'int x = 0; if (x) {} else if ([](int a) { return a; }(1)) printf("lambda\\n");',
            "lambda\n",
        ),
        ('for (int i = 1; i < 9; i *= 2) printf("%d", i); printf("\\n");', "1248\n"),
        ('for (int i = 0; i < 2; i++) { int *p = &i; printf("%d", *p); } printf("\\n");', "01\n"),
        ('for (int i = 0; i != 6; i += 2) printf("%d", i); printf("\\n");', "024\n"),
        ('for (int i = 0; i > 5; i++) printf("never"); printf("done\\n");', "done\n"),
        (
            'auto g = [](int a) { printf("%d\\n", a); }; auto h = [&]() { return g; }; h()(5);',
            "5\n",
        ),
    ],
)
def test_loops_and_branches_that_python_writes_differently(
    capsys: pytest.CaptureFixture[str], body: str, text: str
) -> None:
    assert printed(capsys, body) == text
