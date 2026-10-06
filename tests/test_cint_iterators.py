"""A class's own iterator: ``operator++`` both ways, ``operator*``, and a range-for over it."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.cint import translate
from xrdroot.cint.execute import run_source
from xrdroot.cint.runtime import iterate

SOURCE = """
struct Counter {
  int at;
  Counter(int a) : at(a) {}
  Counter operator++(int) { Counter r = *this; ++(*this); return r; }
  Counter &operator++() { ++at; return *this; }
  Counter &operator--() { --at; return *this; }
  int operator*() { return at * 10; }
  bool operator!=(const Counter &o) const { return at != o.at; }
};
struct Range {
  int n;
  Range(int k) : n(k) {}
  Counter begin() { return Counter(0); }
  Counter end() { return Counter(n); }
};
"""


def test_a_classs_increments_and_dereference_call_its_operators(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = SOURCE + """
    void t() {
      Counter c(1); Counter old = c++; ++c; --c; int v = *c;
      printf("%d %d %d\\n", old.at, c.at, v);
    }
    """
    text = translate(source, "t.C")
    assert "def _postinc(self, arg0):" in text and "def _preinc(self):" in text
    assert "old = c._postinc(0)" in text and "    c._preinc()\n" in text
    assert "v = int(c._deref())" in text
    run_source(source, "t.C", root=fake())
    assert capsys.readouterr().out == "1 2 20\n"


def test_a_range_for_over_a_class_walks_its_begin_to_its_end(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = SOURCE + """
    void t() { int sum = 0; for (auto v : Range(3)) sum += v; printf("%d\\n", sum); }
    """
    run_source(source, "t.C", root=fake())
    assert capsys.readouterr().out == "30\n"


def test_an_iterator_or_a_plain_object_is_iterated_as_it_is() -> None:
    items = [1, 2]
    assert iterate(items) is items


def test_a_range_for_over_a_class_template_of_the_macros_walks_its_begin_to_its_end(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = SOURCE + """
    template <typename T> struct Span { int n; Span(int k) : n(k) {}
      Counter begin() { return Counter(0); } Counter end() { return Counter(n); } };
    void t() { Span<int> s(2); int sum = 0; for (auto v : s) sum += v; printf("%d\\n", sum); }
    """
    run_source(source, "t.C", root=fake())
    assert capsys.readouterr().out == "10\n"


def test_a_library_container_with_a_begin_is_iterated_by_python() -> None:
    class Listed(list):  # type: ignore[type-arg]
        def begin(self) -> None:
            raise AssertionError("a container Python iterates is not walked")

    items = Listed([1, 2])
    assert iterate(items) is items
