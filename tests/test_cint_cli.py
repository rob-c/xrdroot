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
    assert main(["run", str(path)]) == 0
    assert main(["run", f"{path}(3)", "--no-cache"]) == 0
    # As root -b -q does, the value the function gives back is printed after it runs.
    assert capsys.readouterr().out == "1264\n(int) 7\n1896\n(int) 7\n"


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
