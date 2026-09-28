"""What ``root -q`` makes of a macro's returned value: cling's line, and the exit status.

Every expectation is ROOT 6.40.04's own, from a one-line macro returning
that value run under ``root -b -q -l``: the line cling printed after the
macro's output, and the status the process exited with.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from cintfake import fake
from xrdroot.cint import cache
from xrdroot.cint.ctype import CType
from xrdroot.cint.returned import shown, spelling, status
from xrdroot.cli import main


@pytest.fixture(autouse=True)
def fake_pyroot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("xrdroot.pyroot")
    module.__dict__.update(vars(fake()))
    monkeypatch.setitem(sys.modules, "xrdroot.pyroot", module)
    monkeypatch.setenv(cache.ENVIRONMENT, str(tmp_path / "cache"))


#: A returned value, the type it was returned as, cling's line and ROOT's exit status.
ROOT_RUNS: list[tuple[Any, CType, str, int]] = [
    (3, CType("int"), "(int) 3", 3),
    (0, CType("int"), "(int) 0", 0),
    (300, CType("int"), "(int) 300", 255),
    (-1, CType("int"), "(int) -1", 255),
    (9, CType("int"), "(int) 9", 9),  # Int_t, which cling spells int
    (7, CType("long long"), "(long long) 7", 7),  # Long64_t
    (4294967301, CType("long"), "(long) 4294967301", 255),
    (4, CType("unsigned int"), "(unsigned int) 4", 4),
    (6, CType("short"), "(short) 6", 6),
    (65, CType("char"), "(char) 'A'", 65),
    ("A", CType("char"), "(char) 'A'", 65),
    (True, CType("bool"), "(bool) true", 1),
    (False, CType("bool"), "(bool) false", 0),
    (2.7, CType("double"), "(double) 2.7000000", 2),
    (1e20, CType("double"), "(double) 1.0000000e+20", 255),
    (0.1, CType("double"), "(double) 0.10000000", 0),
    (123456.789, CType("double"), "(double) 123456.79", 255),
    (-3.5, CType("double"), "(double) -3.5000000", 255),
    (255.9, CType("double"), "(double) 255.90000", 255),
    (-0.5, CType("double"), "(double) -0.50000000", 0),
    (1e-5, CType("double"), "(double) 1.0000000e-05", 0),
    (2.5, CType("float"), "(float) 2.50000f", 2),
    ("abc", CType("char", pointer=1, const=True), '(const char *) "abc"', 255),
    ("abc", CType("TString"), '(TString) "abc"[3]', 255),
    ("abc", CType("std::string"), '(std::string) "abc"', 255),
    ("abc", CType("string"), '(std::string) "abc"', 255),
    (None, CType("TObject", pointer=1), "(TObject *) nullptr", 0),
]


@pytest.mark.parametrize(("value", "returns", "line", "code"), ROOT_RUNS)
def test_a_returned_value_is_printed_and_exited_with_as_root_does(
    value: Any, returns: CType, line: str, code: int
) -> None:
    spelled = spelling(returns)
    assert shown(value, spelled) == line
    assert status(value, spelled) == code


def test_a_pointer_to_an_object_prints_its_address_and_exits_255() -> None:
    held = object()
    spelled = spelling(CType("TH1F", pointer=1))
    assert shown(held, spelled) == f"(TH1F *) 0x{id(held):x}"
    assert status(held, spelled) == 255


def test_what_cling_is_not_known_to_print_prints_nothing() -> None:
    for returns in (
        CType("void"),
        CType("long double"),
        CType("TH1F"),
        CType("TH1F", reference=True),
        CType("int", dims=[3]),
        CType("std::vector", args=[CType("int")]),
    ):
        assert spelling(returns) is None
    assert shown(4, None) is None
    assert shown(None, "int") is None  # an int function that fell off its end
    assert shown("text", "char *") == '(char *) "text"'
    # An object held by value has an address, as a pointer does; nothing is none.
    assert status(object(), None) == 255 and status(None, None) == 0


def test_a_number_no_integer_holds_exits_255() -> None:
    assert status(float("nan"), "double") == 255
    assert status(float("inf"), "double") == 255


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_run_exits_with_what_the_macro_returned_cached_or_not(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = write(tmp_path, "text.C", 'const char* text() { printf("hi\\n"); return "abc"; }')
    none = write(tmp_path, "none.C", "TObject* none() { return nullptr; }")
    nothing = write(tmp_path, "nothing.C", "void nothing() { }")
    assert main(["run", str(text)]) == 255
    assert main(["run", str(text)]) == 255  # the kept translation remembers the type
    assert main(["run", str(none)]) == 0
    assert main(["run", str(nothing)]) == 0
    printed = '(const char *) "abc"\n'
    assert capsys.readouterr().out == f"hi\n{printed}hi\n{printed}(TObject *) nullptr\n"
