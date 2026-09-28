"""A macro's own ``ostream &operator<<(ostream &, const T &)``: what ``cout << t`` writes."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.cint import translate
from xrdroot.cint.execute import run_source
from xrdroot.cint.runtime import cout, stream_formatter


def output(capsys: pytest.CaptureFixture[str], source: str) -> str:
    run_source(source, "t.C", root=fake())
    return capsys.readouterr().out


def test_an_enums_writer_is_called_where_the_type_says_it_is_that_enum(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    enum class EMode { kNaive, kInformed };
    std::ostream &operator<<(std::ostream &os, const EMode &e) {
      switch (e) { case EMode::kNaive: os << "naive"; break; default: os << "informed"; }
      return os;
    }
    void show(EMode m) { std::cout << "mode " << m << "!" << std::endl; }
    void t() { show(EMode::kNaive); show(EMode::kInformed); }
    """
    assert "def ostream_EMode(os, e):" in translate(source, "t.C")
    assert output(capsys, source) == "mode naive!\nmode informed!\n"


def test_a_classs_writer_is_called_even_where_the_type_is_not_known(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    struct Task { unsigned long id = 7; };
    ostream &operator<<(ostream &s, const Task &t) { s << "task " << t.id; return s; }
    void t() { Task one; cout << one << endl; }
    """
    assert "stream_formatter(Task, ostream_Task)" in translate(source, "t.C")
    assert output(capsys, source) == "task 7\n"
    namespace = run_source(source, "t.C", root=fake(), call=False)
    cout << namespace["Task"]() << "\n"
    assert capsys.readouterr().out == "task 7\n"


def test_a_writer_of_a_pointer_or_another_operator_is_still_refused() -> None:
    from xrdroot.cint import Refusal

    with pytest.raises(Refusal, match="the operator << defined outside a class"):
        translate("struct T {}; ostream &operator<<(ostream &o, T *t) { return o; }", "t.C")
    with pytest.raises(Refusal, match="the operator << defined outside a class"):
        translate("struct T {}; int operator<<(int a, const T &t) { return a; }", "t.C")


def test_the_runtime_writer_is_found_through_the_class_a_value_derives_from(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Base:
        pass

    class Derived(Base):
        pass

    stream_formatter(Base, lambda os, value: os << "a base")
    cout << Derived() << "\n"
    assert capsys.readouterr().out == "a base\n"
