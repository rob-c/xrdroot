"""Static locals: one variable for the whole run, initialised the first time it is reached."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.cint import translate
from xrdroot.cint.execute import run_source
from xrdroot.cint.runtime import Static


def output(capsys: pytest.CaptureFixture[str], source: str) -> str:
    run_source(source, "t.C", root=fake())
    return capsys.readouterr().out


def test_a_static_local_in_a_method_is_shared_by_every_object(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    struct A { int f() { static int n = 0; return ++n; } };
    void t() { A a, b; a.f(); b.f(); printf("%d\\n", a.f()); }
    """
    assert output(capsys, source) == "3\n"


def test_a_static_local_in_a_lambda_keeps_its_value_between_calls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    void t() {
      auto next = []() { static unsigned long i = 0; return i++; };
      next(); next();
      printf("%lu\\n", next());
    }
    """
    assert output(capsys, source) == "2\n"


def test_a_static_local_is_initialised_once_from_a_local_when_first_reached(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    int seen(int x) { static int first = x; return first; }
    void t() { seen(4); printf("%d\\n", seen(9)); }
    """
    assert output(capsys, source) == "4\n"


def test_a_static_object_is_built_once_and_its_address_is_the_object(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    struct Box { int n = 0; };
    Box *box() { static Box b; b.n++; return &b; }
    void t() { box(); Box *p = box(); printf("%d\\n", p->n); }
    """
    assert output(capsys, source) == "2\n"


def test_a_constant_static_array_is_made_before_anything_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    int days(int m) { static int maxdays[3] = {31, 28, 31}; return maxdays[m]; }
    void t() { printf("%d\\n", days(1)); }
    """
    text = translate(source, "t.C")
    assert "days_maxdays_1 = Static(array('int', 3, [31, 28, 31]))" in text
    assert "if not" not in text
    pointers = translate("void f() { static TH1F *hs[2]; }", "t.C")
    assert "f_hs_1 = Static(array('TH1F*', 2))" in pointers
    assert output(capsys, source) == "28\n"


def test_a_static_local_is_filed_under_its_class_and_method() -> None:
    text = translate("struct A { int operator()() { static int n = 1; return n; } };", "t.C")
    assert "A_operator_n_1 = Static(1, 'int')" in text


def test_a_static_local_of_a_class_that_is_not_a_static_nested_in_a_local_class() -> None:
    source = "{ struct In { struct Deep { int v; }; }; static int k = 3; printf(\"%d\", k); }"
    text = translate(source, "t.C")
    assert text.index("t_k_1 = Static(3, 'int')") < text.index("def t():")


def test_a_static_that_throws_initialising_is_tried_again() -> None:
    cell = Static(0, "int", ready=False)
    assert not cell.ready and cell.value == 0 and Static().ready
