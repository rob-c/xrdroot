"""One translation unit for the session: the prompt's lines, ``ProcessLine`` and ``.x`` share it.

Every expected value here is what ROOT 6.40.04's ``root.exe`` gives for the
same input (``.undo``, which crashes that build's cling on macOS, excepted).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from pyrootsupport import fresh
from xrdroot.cppstd.cells import address_of, pointer_at
from xrdroot.pycint.execute import process_line
from xrdroot.pycint.interactive import Interpreter
from xrdroot.pycint.interactive import lines as session
from xrdroot.pycint.interactive.rerun import merged, revised
from xrdroot.pycxx import cache
from xrdroot.pycxx.errors import MacroError, Refusal
from xrdroot.pycxx.runtime.namespace import DECLARED


@pytest.fixture(autouse=True)
def _fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(cache.ENVIRONMENT, str(tmp_path / "cache"))
    monkeypatch.setattr(session, "_UNITS", [])
    yield from fresh(tmp_path)


def typed(prompt: Interpreter, *lines: str) -> list[str]:
    return [said.text for line in lines for said in prompt.feed(line)]


def test_what_one_process_line_declares_the_next_one_sees(
    capsys: pytest.CaptureFixture[str],
) -> None:
    process_line("int k = 3;")
    process_line("int twice(int n) { return 2 * n; }")
    process_line('printf("%d\\n", twice(k));')
    assert process_line("k + 1;") == 4
    assert process_line("  ") is None
    assert capsys.readouterr().out == "6\n"


def test_a_pointer_printed_with_percent_p_is_the_object_cast_back_from_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """{
       TNamed *n = new TNamed("named", "a title");
       gROOT->ProcessLine(TString::Format("TNamed *m = (TNamed*)%p;", n));
       gROOT->ProcessLine("printf(\\"%s %d\\\\n\\", m->GetTitle(), (Long_t)m == (Long_t)m);");
    }"""
    from xrdroot.pycint.execute import run_source

    run_source(source, "p.C")
    assert capsys.readouterr().out == "a title 1\n"


def test_an_address_no_object_was_seen_at_is_refused_and_null_is_null() -> None:
    marker = object()
    assert pointer_at(address_of(marker)) is marker
    assert address_of(None) == 0 and pointer_at(0) is None
    with pytest.raises(ValueError, match="no address of an object"):
        pointer_at(1)


def test_a_pointer_to_a_number_made_from_an_integer_is_refused() -> None:
    with pytest.raises(Refusal, match="a pointer to int made from an integer"):
        process_line("long a = 8; int *p = (int*)a;")


def test_a_line_that_fails_stops_the_macro_with_its_own_words() -> None:
    with pytest.raises(MacroError, match=r"<ProcessLine>:1: .*NoSuchThing"):
        process_line("NoSuchThing::Make();")


def test_a_macro_run_with_x_sees_the_prompts_variables(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "usek.C").write_text('void usek() { printf("%d\\n", k); }\n')
    prompt = Interpreter()
    assert typed(prompt, "int k = 4;", "int *p = &k;") == []
    assert DECLARED["k"] == 4  # a cell's value, not the cell
    assert typed(prompt, f".x {tmp_path / 'usek.C'}") == []
    assert capsys.readouterr().out == "4\n"


def test_a_line_process_line_runs_while_the_prompt_runs_one_runs_by_itself(
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt = Interpreter()
    assert typed(prompt, 'gROOT->ProcessLine("int z = 2; printf(\\"%d\\\\n\\", z);");') == []
    assert capsys.readouterr().out == "2\n"


def test_a_later_overload_and_a_method_defined_outside_its_class_are_there() -> None:
    prompt = Interpreter()
    lines = ["int f(int a) { return a; }", "f(2)", "int f(double a) { return 10; }",
             "f(2.5)", "f(2)", "class A { public: int g(); };", "A a;",
             "int A::g() { return 7; }", "a.g()"]  # fmt: skip
    assert typed(prompt, *lines) == ["(int) 2", "(int) 10", "(int) 2", "(int) 7"]


def test_the_address_of_an_earlier_variable_writes_through_to_it() -> None:
    prompt = Interpreter()
    assert typed(prompt, "int i = 3;", "int *p = &i;", "*p = 5;", "i") == ["(int) 5"]


def test_undo_takes_back_inputs_and_destroys_what_they_made() -> None:
    prompt = Interpreter()
    lines = ['struct Marker { ~Marker() { printf("dest\\n"); } };', "Marker m1;", "int i = 3;"]
    assert typed(prompt, *lines) == []
    assert typed(prompt, ".undo 2") == []
    assert "i" not in prompt.unit.namespace and "m1" not in prompt.unit.namespace
    assert typed(prompt, "int i = 4;", "i") == ["(int) 4"]
    assert typed(prompt, ".undo x") == [
        "error: .undo takes a number of inputs to take back, not 'x'"
    ]


def test_process_line_runs_undo_and_include_on_the_sessions_unit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from xrdroot.pyroot.core.system import gSystem

    monkeypatch.setattr(gSystem, "_include_path", "")  # what .I adds to, for ACLiC
    (tmp_path / "seven.h").write_text("int seven() { return 7; }\n")
    assert process_line(f".I {tmp_path}") == 0
    process_line('#include "seven.h"')
    process_line("int q = 1;")
    process_line(".undo")
    assert process_line(".I") == 0
    assert capsys.readouterr().out == f"-I\n{tmp_path}\n"
    with pytest.raises(MacroError, match="ROOT has q"):
        process_line("q;")


def test_process_line_runs_the_prompts_commands_but_not_leaving(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``.O 0`` is kept for the next ``.O``; ``.pwd`` hands back ``1``, as ROOT's
    ``ProcessLine`` does for its own commands; ``.q`` and an unknown command are refused."""
    assert process_line(".O 0") == 0
    assert process_line(".O") == 0
    assert process_line(".pwd") == 1
    out = capsys.readouterr().out
    assert out.startswith("Current cling optimization level: 0\nCurrent directory: ")
    for refused in (".q", ".typedef X", ". odd"):
        with pytest.raises(ValueError, match="is not a command gROOT->ProcessLine runs here"):
            process_line(refused)


def test_a_statement_whose_python_did_not_change_is_not_revised() -> None:
    import ast

    statement = ast.parse("x = f(1)").body[0]
    assert revised(statement, None) is None
    assert revised(statement, ast.dump(statement)) is None
    assert revised(statement, "other") is None
    celled = ast.parse("x = Cell(3)").body[0]
    assert ast.unparse(revised(celled, "other")) == (  # type: ignore[arg-type]
        "x = x if isinstance(x, Cell) else Cell(x, None)"
    )
    assert revised(ast.parse("x = y = Cell(3)").body[0], "other") is None
    assert revised(ast.parse("x.a = Cell(3)").body[0], "other") is None
    assert revised(ast.parse("x = g.Cell(3)").body[0], "other") is None


def test_a_class_defined_again_is_merged_into_the_class_it_replaces() -> None:
    class Old:
        pass

    class New:
        def g(self) -> int:
            return 7

    namespace = {"A": New, "B": 3}
    merged(namespace, {"A": Old, "B": Old})
    assert namespace["A"] is Old and Old().g() == 7  # type: ignore[attr-defined]
    merged(namespace, {"A": Old})
    assert namespace["A"] is Old
