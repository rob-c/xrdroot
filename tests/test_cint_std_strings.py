"""``std::string``'s members that change or search a string, and ``std::transform`` over one."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.cint import Refusal, translate
from xrdroot.cint.execute import run_source
from xrdroot.cint.runtime import (
    find_first_not_of,
    find_first_of,
    find_last_not_of,
    find_last_of,
    npos,
    resize,
    tolower,
    toupper,
    transformed,
)


def block(capsys: pytest.CaptureFixture[str], body: str) -> str:
    run_source("{\n" + body + "\n}\n", "t.C", root=fake())
    return capsys.readouterr().out


def test_resize_and_assign_change_the_string_they_are_called_on(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """std::string s = "abcdef"; s.resize(3); std::string t = "x"; t.assign(s);
    t.resize(5, '-'); printf("%s %s %d\\n", s.c_str(), t.c_str(), (int)t.size());"""
    assert block(capsys, source) == "abc abc-- 5\n"


def test_the_find_of_family_searches_for_any_of_a_set_of_characters(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = r"""std::string p = "/a/b\\c.C";
    printf("%d %d %d %d\n", (int)p.find_last_of("\\/"), (int)p.find_first_of("/"),
           (int)p.find_first_not_of("/a"), (int)p.find_last_not_of("C."));"""
    assert block(capsys, source) == "4 0 3 5\n"


def test_transform_rewrites_a_string_one_character_at_a_time(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """std::string n = "Cable";
    std::transform(n.begin(), n.end(), n.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    printf("%s\\n", n.c_str());"""
    assert block(capsys, source) == "cable\n"


def test_transform_writes_an_array_through_a_pointer_to_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """double a[3] = {1, 2, 3}; double b[3];
    std::transform(a, a + 3, b, [](double x) { return 2 * x; });
    printf("%g %g\\n", b[0], b[2]);
    std::transform(a, a + 2, b + 1, [](double x) { return -x; });
    printf("%g %g %g\\n", b[0], b[1], b[2]);"""
    assert block(capsys, source) == "2 6\n2 -1 -2\n"


def test_transform_writes_a_container_in_place() -> None:
    items = [1, 2, 3]
    assert transformed(items, 0, None, items, 0, lambda x: x * 10) is items
    assert items == [10, 20, 30]
    assert transformed("abc", 1, None, "xyz", 1, toupper) == "xBC"


def test_transform_is_refused_with_a_second_input_range() -> None:
    source = """void t() { std::vector<int> a, b;
    std::transform(a.begin(), a.end(), b.begin(), b.begin(), f); }"""
    with pytest.raises(Refusal, match="std::transform of two ranges into a third"):
        translate(source, "t.C")


def test_the_runtime_string_searches_give_npos_when_nothing_is_found() -> None:
    assert find_first_of("abc", "xyz") == npos and find_last_of("abc", "xyz") == npos
    assert find_first_not_of("aaa", "a") == npos and find_last_not_of("aaa", "a") == npos
    assert find_last_of("a/b/c", "/", 2) == 1
    assert resize("ab", 4) == "ab\0\0"
    assert tolower(ord("Q")) == ord("q") and tolower(200) == 200 and toupper(200) == 200


def test_c_str_of_a_string_whose_type_is_not_known_is_the_string() -> None:
    text = translate("void t() { printf(\"%s\", ss.str().c_str()); p->c_str(); }", "t.C")
    assert "printf('%s', cstr(ROOT.ss.str()))" in text and "ROOT.p.c_str()" in text
