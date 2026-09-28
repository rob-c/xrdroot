"""Reads whose result is used: ``while (std::getline(in, line))``, ``while (in >> a >> b)``."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.cint import Refusal, translate
from xrdroot.cint.execute import run_source
from xrdroot.cint.runtime import find_if


def block(capsys: pytest.CaptureFixture[str], body: str) -> str:
    run_source("{\n" + body + "\n}\n", "t.C", root=fake())
    return capsys.readouterr().out


def test_getline_in_a_condition_reads_each_line_until_the_stream_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = r"""std::istringstream in("one\n\nthree"); std::string line; int n = 0;
    while (std::getline(in, line)) { n++; printf("[%s]", line.c_str()); }
    printf(" %d\n", n);"""
    assert block(capsys, source) == "[one][][three] 3\n"


def test_getline_splits_at_the_delimiter_it_is_given(capsys: pytest.CaptureFixture[str]) -> None:
    source = r"""std::istringstream f("a/bc/d"); std::string s;
    while (getline(f, s, '/')) printf("%s;", s.c_str());
    std::istringstream g("x/y"); getline(g, s, '/'); printf("%s\n", s.c_str());"""
    assert block(capsys, source) == "a;bc;d;x\n"


def test_reading_pairs_in_a_condition_stops_at_the_first_failed_read(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = r"""std::istringstream in("1 2.5 3 4 5"); double a, b; double sum = 0;
    struct P { double x; } p; double c[1];
    while (in >> a >> b) sum += a * b;
    std::istringstream again("7 8"); if (again >> p.x >> c[0]) sum += p.x + c[0];
    printf("%g\n", sum);"""
    assert block(capsys, source) == "29.5\n"


def test_getline_with_its_result_used_needs_a_string_to_read_into() -> None:
    with pytest.raises(Refusal, match="std::getline without a string to read into"):
        translate("void t() { std::istringstream in; if (std::getline(in)) {} }", "t.C")


def test_find_if_over_roots_iterators_gives_what_it_found() -> None:
    text = translate("void t() { auto found = find_if(it.Begin(), it.End(), pred); }", "t.C")
    assert "found = find_if(ROOT.it.Begin(), ROOT.it.End(), ROOT.pred)" in text
    stop = object()
    assert find_if([1, 4, 9], None, lambda x: x > 3) == 4
    assert find_if([1], None, lambda x: False) is None
    assert find_if([1, stop, 9], stop, lambda x: x > 3) is None


def test_find_if_over_a_container_is_refused() -> None:
    source = "void t() { std::vector<int> v; auto it = std::find_if(v.begin(), v.end(), f); }"
    with pytest.raises(Refusal, match="std::find_if over a container"):
        translate(source, "t.C")
