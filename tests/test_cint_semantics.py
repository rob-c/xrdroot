"""Translated macros do what the C++ does: each test runs a macro and reads what it printed.

The expected text is what the macro prints under ROOT (checked against a
real ROOT while these were written), so every test here is a small claim
that the translation keeps C++'s meaning - integer division, conversions
on assignment, the order of evaluation of loops, switch fallthrough.
"""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.cint.execute import run_source


def output(capsys: pytest.CaptureFixture[str], source: str, *args: object) -> str:
    """What ``source`` prints, run as the macro ``t.C`` (its function ``t``, or its block)."""
    run_source(source, "t.C", tuple(args), root=fake())
    return capsys.readouterr().out


def block(capsys: pytest.CaptureFixture[str], body: str) -> str:
    return output(capsys, "{\n" + body + "\n}\n")


def test_integers_divide_and_take_remainders_as_c_does(capsys: pytest.CaptureFixture[str]) -> None:
    text = block(capsys, 'int a = -7, b = 2; printf("%d %d %g %d\\n", a / b, a % b, a / 2.0, 7/2);')
    assert text == "-3 -1 -3.5 3\n"


def test_stores_convert_as_c_converts(capsys: pytest.CaptureFixture[str]) -> None:
    source = """
    int i = 3.9; int j = -3.9; unsigned int u = -1; unsigned char c = 300; short s = 40000;
    float f = 0.1; double d = 1; bool b = 5; char ch = 'a' + 1;
    printf("%d %d %u %d %d %.10f %g %d %c\\n", i, j, u, c, s, f, d / 2, b, ch);
    i += 2.7; u += 2; d += 1;
    printf("%d %u %g\\n", i, u, d);
    """
    assert block(capsys, source) == "3 -3 4294967295 44 -25536 0.1000000015 0.5 1 b\n5 1 2\n"


def test_logical_operators_give_zero_or_one_where_their_value_is_used(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = 'int a = 5, b = 3; int c = a && b; int d = a || 0; int e = !a; printf("%d %d %d\\n", c, d, e);'
    assert block(capsys, source) == "1 1 0\n"


def test_increments_inside_expressions_are_evaluated_in_place(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    int i = 0; int a[3] = {10, 20, 30};
    int x = a[i++]; int y = a[++i]; int z = i--;
    printf("%d %d %d %d\\n", x, y, z, i);
    """
    assert block(capsys, source) == "10 30 2 1\n"


def test_for_loops_keep_c_semantics_whether_or_not_they_become_range(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    int n = 0;
    for (int i = 0; i < 5; ++i) n += i;
    for (int i = 10; i >= 0; i -= 5) printf("%d ", i);
    int k;
    for (k = 0; k < 3; k++) { if (k == 1) continue; printf("k%d ", k); }
    printf("after %d ", k);
    for (int i = 0, j = 10; i < j; i++, j--) if (j - i < 4) break;
    int m = 4;
    for (int i = 0; i < m; i++) { if (i == 0) m = 2; printf("m%d ", i); }
    printf("%d\\n", n);
    """
    assert block(capsys, source) == "10 5 0 k0 k2 after 3 m0 m1 10\n"


def test_while_and_do_while_test_where_c_tests(capsys: pytest.CaptureFixture[str]) -> None:
    source = """
    int i = 0;
    do { i++; if (i < 3) continue; printf("d%d ", i); } while (i < 4);
    do printf("once "); while (0);
    while (i > 0) { i -= 2; }
    printf("%d\\n", i);
    """
    assert block(capsys, source) == "d3 d4 once 0\n"


def test_a_switch_takes_its_case_and_falls_through_where_c_does(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    void t() {
      for (int i = 0; i < 5; i++) {
        switch (i) {
          case 0: printf("zero "); break;
          case 1: case 2: printf("small "); break;
          default: printf("big ");
        }
        switch (i) {
          case 3: printf("three ");
          case 4: printf("four "); break;
          case 0: if (i == 0) break; printf("never ");
        }
      }
      printf("\\n");
    }
    """
    text = output(capsys, source)
    assert text == "zero small small big three four big four \n"


def test_printf_and_cout_print_what_cpp_prints(capsys: pytest.CaptureFixture[str]) -> None:
    source = """
    double x = 1.0 / 3; float f = 2.5f; int n = 42; char c = 'z'; bool ok = true;
    std::cout << x << " " << f << " " << n << " " << c << " " << ok << std::endl;
    std::cout << std::fixed << std::setprecision(3) << x << " " << std::setw(6) << n << "|" << std::endl;
    printf("%5.2f|%-4d|%e\\n", x, n, 1234.5);
    Printf("%s", Form("h%d", n));
    """
    assert block(capsys, source) == "0.333333 2.5 42 z 1\n0.333     42|\n 0.33|42  |1.234500e+03\nh42\n"


def test_strings_are_written_and_read_as_c_and_std_string_are(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    char buf[32]; sprintf(buf, "h%02d", 7); strcat(buf, "_x");
    std::string s = "abc"; s += "def"; std::string t = s.substr(1, 3);
    TString name = "n"; const char *p = "lit";
    printf("%s %d %s %d %s %d\\n", buf, (int)strlen(buf), t.c_str(), (int)s.size(), p, s.empty());
    printf("%s\\n", ("x" + std::to_string(3)).c_str());
    """
    assert block(capsys, source) == "h07_x 5 bcd 6 lit 0\nx3\n"
