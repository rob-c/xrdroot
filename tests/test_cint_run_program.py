"""``xrdroot run`` starting a C++ program: its ``main``, several files, or nothing to run."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from cintfake import fake
from xrdroot.cli import main
from xrdroot.pycint.execute import run_and_quit
from xrdroot.pycxx import cache


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


STRESS = """
int main(int argc, const char *argv[]) {
   int n = 10;
   if (argc > 1) n = atoi(argv[1]);
   printf("%s ran %d with %d arguments\\n", argv[0], n, argc);
   return n;
}
"""


def test_a_program_runs_its_main_with_the_arguments_after_the_double_dash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path, "stress.cxx", STRESS)
    assert main(["run", "-n", str(path)]) == 10
    assert main(["run", "-n", str(path), "--", "300", "--verbose"]) == 300 % 256
    assert capsys.readouterr().out == (
        "stress ran 10 with 1 arguments\nstress ran 300 with 3 arguments\n"
    )


def test_a_main_without_parameters_is_called_without_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path, "plain.cxx", 'int main() { printf("plain\\n"); return 0; }\n')
    assert main(["run", "-n", str(path), "--", "ignored"]) == 0
    assert capsys.readouterr().out == "plain\n"


def test_a_main_returning_nothing_exits_0(tmp_path: Path) -> None:
    path = write(tmp_path, "quiet.cxx", "void main() {}\n")
    assert main(["run", "-n", str(path)]) == 0


def test_several_files_are_one_program_started_as_the_first_says(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = write(tmp_path, "prog.cxx", "int twice(int);\nint main() { return twice(21); }\n")
    second = write(tmp_path, "lib.cxx", "int twice(int n) { return 2 * n; }\n")
    assert main(["run", "-n", str(first), str(second)]) == 42
    assert main(["run", "-n", str(first), str(second), "--python"]) == 0
    printed = capsys.readouterr().out
    assert "def main():" in printed and "def twice(n):" in printed


def test_a_file_with_nothing_to_run_is_refused_in_a_sentence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path, "library.cxx", "int helper() { return 3; }\n")
    assert main(["run", "-n", str(path)]) == 2
    assert capsys.readouterr().err == (
        "xrdroot run: library.cxx has nothing to run: it defines no function library(), no "
        "main(), no GoogleTest TEST, and it is no unnamed macro { ... }, which are what xrdroot "
        "run looks for, in that order\n"
    )


def test_an_error_in_main_is_placed_at_its_cpp_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path, "broken.cxx", "int main() {\n   return NoSuchClass::Make();\n}\n")
    assert main(["run", "-n", str(path)]) == 2
    assert capsys.readouterr().err.startswith(f"xrdroot run: {path}:2: AttributeError")


def test_a_python_script_is_given_the_arguments_after_the_double_dash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path, "args.py", "import sys\nprint(sys.argv[1:])\n")
    assert main(["run", str(path), "--", "-v", "x"]) == 0
    assert capsys.readouterr().out == "['-v', 'x']\n"


def test_quitting_after_a_macro_still_prints_and_exits_with_what_it_returned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path, "seven.C", "int seven() { return 7; }\n")
    assert run_and_quit(f"{path}+") == 7
    assert capsys.readouterr().out == "(int) 7\n"

def test_a_program_s_output_is_written_a_line_at_a_time(monkeypatch):
    """``print`` and the kit's own writes share one stream, as a C++ program's ``std::cout`` is.

    Under a pipe - which is how a harness always runs a program - Python buffers ``print``'s
    lines until the process ends, so a tutorial's own output arrives after lines the kit wrote
    first, in an order the program never wrote. Measured on ``rf613_global_observables.py``:
    with this, its output is byte-identical to ROOT's; without it, the same lines reordered.
    """
    from xrdroot.cli.run import ordered_output

    seen: list[dict[str, bool]] = []

    class Pipe:
        def isatty(self) -> bool:
            return False

        def reconfigure(self, **options: bool) -> None:
            seen.append(dict(options))

    class Terminal(Pipe):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdout", Pipe())
    ordered_output()
    assert seen == [{"line_buffering": True}]

    monkeypatch.setattr("sys.stdout", Terminal())
    ordered_output()
    assert seen == [{"line_buffering": True}]  # a terminal already writes a line at a time
