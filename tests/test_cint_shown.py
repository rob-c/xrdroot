"""What ``root -b -q macro.C`` prints of the value the macro's function gives back.

Every expected line is what ROOT 6.40.04's cling printed for a one-line
macro of that return type, run as ``root -b -q -l file.C``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cintfake import fake
from xrdroot.cint import cache
from xrdroot.cint.execute import run
from xrdroot.cint.shown import shown


@pytest.fixture(autouse=True)
def private_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    where = tmp_path / "cache"
    monkeypatch.setenv(cache.ENVIRONMENT, str(where))
    return where


#: A macro's body, and the line ROOT printed after running it.
ROOTS = [
    ("int f() { return 0; }", "(int) 0"),
    ("Int_t f() { return -3; }", "(int) -3"),
    ("double f() { return 0.1; }", "(double) 0.10000000"),
    ("double f() { return 1234567.891; }", "(double) 1234567.9"),
    ("float f() { return 2.5; }", "(float) 2.50000f"),
    ("bool f() { return true; }", "(bool) true"),
    ('const char *f() { return "hi"; }', '(const char *) "hi"'),
    ("Long64_t f() { return 12; }", "(long long) 12"),
    ("unsigned int f() { return 12; }", "(unsigned int) 12"),
    ("char f() { return 65; }", "(char) 'A'"),
    ("TH1F *f() { return nullptr; }", "(TH1F *) nullptr"),
    ('std::string f() { return "x"; }', '(std::string) "x"'),
]


@pytest.mark.parametrize(("body", "line"), ROOTS)
def test_the_value_the_macro_gives_back_is_printed_as_cling_prints_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], body: str, line: str
) -> None:
    path = tmp_path / "f.C"
    path.write_text(body)
    run(path, root=fake(), show=True)
    assert capsys.readouterr().out == line + "\n"


def test_nothing_is_printed_unless_asked_or_for_a_function_giving_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only the macro ``root`` was told to run shows its value; ``void`` has none to show."""
    given = tmp_path / "f.C"
    given.write_text("int f() { return 1; }")
    nothing = tmp_path / "g.C"
    nothing.write_text("void g() { }")
    unnamed = tmp_path / "h.C"
    unnamed.write_text("{ int a = 1; }")
    assert run(given, root=fake()) == 1
    run(nothing, root=fake(), show=True)
    run(unnamed, root=fake(), show=True)
    assert capsys.readouterr().out == ""


def test_a_pointer_to_an_object_is_its_address_and_a_missing_value_nothing() -> None:
    """``(TCanvas *) 0x7f...``: an address, which the tutorials' comparison masks."""
    assert shown("TCanvas *", object()).startswith("(TCanvas *) 0x")
    assert shown("char *", "text") == '(char *) "text"'
    assert shown("char", "B") == "(char) 'B'"
    assert shown("int", None) is None
