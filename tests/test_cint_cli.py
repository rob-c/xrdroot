"""``xrdroot run`` and the session's ``.x`` and ``ProcessLine``, for C++ macros."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from cintfake import fake
from xrdroot.cint import cache
from xrdroot.cli import main
from xrdroot.session import Session


@pytest.fixture(autouse=True)
def fake_pyroot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """``xrdroot.pyroot`` as the fake ROOT, so the default binding finds the fake's names."""
    module = types.ModuleType("xrdroot.pyroot")
    module.__dict__.update(vars(fake()))
    monkeypatch.setitem(sys.modules, "xrdroot.pyroot", module)
    monkeypatch.setenv(cache.ENVIRONMENT, str(tmp_path / "cache"))
    return module


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_run_runs_a_macro_with_its_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(
        tmp_path, "count.C", 'int count(int n = 2) { printf("%d\\n", n * kRed); return 7; }'
    )
    # What the macro returns cling prints, and root -q exits with.
    assert main(["run", str(path)]) == 7
    assert main(["run", f"{path}(3)", "--no-cache"]) == 7
    assert capsys.readouterr().out == "1264\n(int) 7\n1896\n(int) 7\n"


#: ``tutorials/rootlogon.C`` and ``rootlogoff.C``, as ROOT 6.40 ships them.
LOGON = """{
   printf("\\nWelcome to the ROOT tutorials\\n\\n");
   printf("\\nType \\".x demos.C\\" to get a toolbar from which to execute the demos\\n");
   printf("\\nType \\".x demoshelp.C\\" to see the help window\\n\\n");
   printf("==> Many tutorials use the file hsimple.root produced by hsimple.C\\n");
   printf("==> It is recommended to execute hsimple.C before any other script\\n\\n");
}
"""
LOGOFF = '{\n   printf("\\nTaking a break from ROOT? Hope to see you back!\\n\\n");\n}\n'


def test_run_runs_the_working_directorys_logon_and_logoff_macros_around_a_macro_as_root_does(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write(tmp_path, "quiet.C", "void quiet() {}")
    write(tmp_path, "rootlogon.C", LOGON)
    write(tmp_path, "rootlogoff.C", LOGOFF)
    write(tmp_path, "script.py", "print('no logon for PyROOT')\n")
    assert main(["run", "quiet.C"]) == 0
    # What `root -b -q -l demoshelp.C` prints in ROOT's tutorials directory.
    assert capsys.readouterr().out == (
        "\nWelcome to the ROOT tutorials\n\n"
        '\nType ".x demos.C" to get a toolbar from which to execute the demos\n'
        '\nType ".x demoshelp.C" to see the help window\n\n'
        "==> Many tutorials use the file hsimple.root produced by hsimple.C\n"
        "==> It is recommended to execute hsimple.C before any other script\n\n"
        "\nTaking a break from ROOT? Hope to see you back!\n\n"
    )
    assert main(["run", "script.py"]) == 0
    assert capsys.readouterr().out == "no logon for PyROOT\n"
    assert main(["run", "-n", "quiet.C"]) == 0  # as `root -n`, which hsimple.C runs under
    assert capsys.readouterr().out == ""


def test_run_prints_and_exits_with_a_returned_value_between_the_logon_macros(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write(tmp_path, "rootlogon.C", LOGON)
    write(tmp_path, "rootlogoff.C", LOGOFF)
    # hsimple.C returns its file: under -n no logon speaks, and root -q exits 255.
    write(tmp_path, "gives.C", 'TH1F *gives() { printf("made\\n"); return nullptr; }')
    write(tmp_path, "file.C", "int file() { return 300; }")
    assert main(["run", "-n", "file.C"]) == 255
    assert capsys.readouterr().out == "(int) 300\n"
    assert main(["run", "gives.C"]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("\nWelcome") and printed.endswith(
        "made\n(TH1F *) nullptr\n\nTaking a break from ROOT? Hope to see you back!\n\n"
    )


def test_run_prints_the_translation_when_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path, "show.C", "void show() { int a = 7 / 2; }")
    assert main(["run", str(path), "--python"]) == 0
    assert "    a = idiv(7, 2)\n" in capsys.readouterr().out


def test_run_runs_a_python_script_with_root_as_pyroot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_pyroot: types.ModuleType
) -> None:
    path = write(tmp_path, "script.py", "import ROOT\nprint(ROOT.kRed, __name__)\n")
    assert main(["run", str(path)]) == 0
    assert capsys.readouterr().out == "632 __main__\n"
    assert sys.modules["ROOT"] is fake_pyroot


def test_run_refuses_a_construct_in_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path, "jump.C", "void jump() {\n  goto end;\n}\n")
    assert main(["run", str(path)]) == 2
    assert capsys.readouterr().err == (
        f"xrdroot run: {path}:2: goto, which jumps to a label, and Python has no jump\n"
    )


def test_the_session_runs_a_cpp_macro_with_x(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path, "hello.C", 'void hello(int n) { printf("hello %d\\n", n); }')
    session = Session()
    session.macro(f"{path}+", 4)
    session.ProcessLine('printf("%d\\n", 9 / 2)')
    assert capsys.readouterr().out == "hello 4\n4\n"
    assert ".x macro.C(a)" in session.help()
