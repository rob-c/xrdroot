"""The C++ prompt's dot commands, its terminal, and ``xrdroot root`` on the command line."""

from __future__ import annotations

import io
import re
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

from pyrootsupport import fresh
from xrdroot.cli import main
from xrdroot.pycint.interactive import Interpreter, interact, terminal
from xrdroot.pycint.interactive.terminal import Completer, say
from xrdroot.pycxx import cache

#: A ROOT file of the test data, with one tree in it.
SIMPLE = Path(__file__).parent / "data" / "simple.root"

#: A pointer as cling prints one: an address, which differs from run to run.
ADDRESS = r"0x[0-9a-f]+"


@pytest.fixture(autouse=True)
def _fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(cache.ENVIRONMENT, str(tmp_path / "cache"))
    monkeypatch.setattr(terminal, "HISTORY", tmp_path / "history")
    yield from fresh(tmp_path)


def typed(prompt: Interpreter, *lines: str) -> list[str]:
    return [said.text for line in lines for said in prompt.feed(line)]


def write(name: str, text: str) -> Path:
    path = Path(name)
    path.write_text(text)
    return path


def piped(text: str) -> io.StringIO:
    return io.StringIO(text)


class Terminal(io.StringIO):
    """Standard input that says it is a terminal."""

    def isatty(self) -> bool:
        return True


# -- running macros --------------------------------------------------------------------


def test_dot_x_runs_a_macro_and_prints_what_it_returns(capsys: pytest.CaptureFixture[str]) -> None:
    write("fill.C", 'double fill(int n = 3) { printf("filled %d\\n", n); return n * 1.5; }')
    prompt = Interpreter()
    assert typed(prompt, ".x fill.C(4)", ".X fill.C+") == []
    assert capsys.readouterr().out == "filled 4\n(double) 6.0000000\nfilled 3\n(double) 4.5000000\n"


def test_dot_x_reports_a_missing_macro_as_root_does_and_a_failing_one_at_its_line() -> None:
    write("bad.C", "int bad() { return 1 / 0; }")
    write("unnamed.C", "{ }")
    prompt = Interpreter()
    assert typed(prompt, ".x", ".x nothere.C") == [
        "error: .x needs a macro to run: .x file.C, or .x file.C(arguments)",
        "Error in <TApplication::ExecuteFile>: macro nothere.C not found in path .",
    ]
    (failed,) = typed(prompt, ".x bad.C")
    assert failed.startswith("bad.C:1:1: error: ZeroDivisionError")
    (refused,) = typed(prompt, ".x unnamed.C(1)")
    assert refused == "error: unnamed.C is an unnamed macro, which takes no arguments"


def test_dot_l_declares_a_macros_functions_and_dot_u_forgets_them() -> None:
    write("lib.C", "int twice(int x) { return 2 * x; }")
    prompt = Interpreter()
    assert typed(prompt, ".L lib.C+", "twice(21)") == ["(int) 42"]
    assert typed(prompt, ".U lib.C", "twice(1)") == [
        "ROOT_prompt_3:1:1: error: use of undeclared identifier 'twice'"
    ]  # fmt: skip
    assert typed(prompt, ".U lib.C", ".U") == [
        "error: lib.C was not loaded with .L, so there is nothing to unload",
        "error: nothing was not loaded with .L, so there is nothing to unload",
    ]


def test_dot_l_reports_a_missing_or_refused_macro() -> None:
    write("broken.C", "int broken() { return ; ; }}")
    prompt = Interpreter()
    assert typed(prompt, ".L", ".L gone.C") == [
        "Error in <TRint::ProcessLine>: macro (none) not found in path .",
        "Error in <TRint::ProcessLine>: macro gone.C not found in path .",
    ]
    (refused,) = typed(prompt, ".L broken.C")
    assert refused.startswith("broken.C:1:1: error: ")


# -- the session's files -------------------------------------------------------------------


def test_a_root_file_is_attached_as_file0_and_listed_by_dot_ls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt = Interpreter()
    (attached,) = [said.text for said in prompt.given(str(SIMPLE))]
    assert capsys.readouterr().out == f"Attaching file {SIMPLE} as _file0...\n"
    assert re.fullmatch(rf"\(TFile \*\) {ADDRESS}", attached)
    assert prompt.status == 255  # a pointer's value, as root -q exits with it
    assert typed(prompt, ".ls", "_file0->GetListOfKeys()->GetSize()") == ["(int) 1"]
    assert "KEY: TTree\ttree;1\tfake data" in capsys.readouterr().out
    assert typed(prompt, ".files") == ["Files open:", f"  {SIMPLE}", "Macros loaded:"]


def test_dot_cd_and_dot_pwd_move_through_the_current_file(
    capsys: pytest.CaptureFixture[str],
) -> None:
    dirs = Path(__file__).parent / "data" / "dirs-6.14.00.root"
    prompt = Interpreter()
    prompt.attach(str(dirs))
    capsys.readouterr()
    assert typed(prompt, ".cd dir1", ".pwd", ".cd", ".pwd") == [
        f"Current directory: {dirs}:/dir1", "Current style:     Modern",
        "Current directory: PyROOT:/", "Current style:     Modern",
    ]  # fmt: skip
    assert prompt.status == 1  # what ROOT's ProcessLine hands back for .pwd


def test_dot_files_lists_the_macros_loaded() -> None:
    write("lib.C", "int one() { return 1; }")
    prompt = Interpreter()
    typed(prompt, ".L lib.C")
    assert typed(prompt, ".files")[-1] == "  lib.C"


# -- the other commands -------------------------------------------------------------------


def test_dot_bang_runs_a_shell_command() -> None:
    prompt = Interpreter()
    assert prompt.feed(".! echo out; echo err 1>&2") == [("out", False), ("err", True)]


def test_dot_greater_sends_standard_output_to_a_file_until_it_is_given_alone() -> None:
    prompt = Interpreter()
    before = sys.stdout
    typed(prompt, ".> out.txt", 'printf("kept\\n");')
    assert sys.stdout is not before
    typed(prompt, ".>")
    assert sys.stdout is before
    assert Path("out.txt").read_text() == "kept\n"
    typed(prompt, ".> again.txt")
    prompt.close()
    assert sys.stdout is before


def test_dot_help_lists_the_commands() -> None:
    said = typed(Interpreter(), ".help")
    assert said[0].startswith("This is xrdroot's C++ prompt")
    assert any(line.lstrip().startswith(".x file.C") for line in said)
    assert typed(Interpreter(), ".?") == said


def test_dot_q_and_its_spellings_end_the_session() -> None:
    for command in (".q", ".qqq", ".qqqqq", ".qqqqqqq", ".exit", ".quit"):
        prompt = Interpreter()
        typed(prompt, command)
        assert prompt.finished


def test_the_raw_input_and_optimization_settings_are_kept() -> None:
    prompt = Interpreter()
    assert typed(prompt, ".rawInput", ".rawInput 0", ".rawInput 1") == [
        "Using raw input", "Not using raw input", "Using raw input"
    ]  # fmt: skip
    assert typed(prompt, ".O", ".O 3", ".O", ".O fast") == [
        "Current cling optimization level: 2",
        "Current cling optimization level: 3",
        "error: .O takes a level, a number such as 2, not 'fast'",
    ]


def test_dot_g_lists_the_variables_declared_at_the_prompt() -> None:
    prompt = Interpreter()
    typed(prompt, "int i = 1;", "auto r = gRandom->Gaus();")
    assert typed(prompt, ".g", ".g i", ".g nope") == [
        "int i", "auto r", "int i", "error: no variable nope was declared at the prompt"
    ]  # fmt: skip


def test_dot_class_shows_a_classes_bases_and_methods_or_says_there_is_none() -> None:
    prompt = Interpreter()
    said = typed(prompt, ".class TH1F")
    assert said[0].startswith("class TH1F : TH1, ")
    assert "Fill" in " ".join(said[2:])
    typed(prompt, "class Mine { public: int get() { return 1; } };")
    assert typed(prompt, ".class Mine")[:2] == ["class Mine", "  1 public methods:"]
    assert typed(prompt, ".class Nothing", ".class") == [
        "error: there is no class Nothing here: .class shows the classes declared at the "
        "prompt and those xrdroot.pyroot provides, by their ROOT names",
        "error: there is no class (none given) here: .class shows the classes declared at the "
        "prompt and those xrdroot.pyroot provides, by their ROOT names",
    ]


def test_an_unknown_command_is_named() -> None:
    assert typed(Interpreter(), ".frobnicate") == [
        "error: .frobnicate is not a command this prompt knows; .help lists the ones it does"
    ]  # fmt: skip


def test_dot_at_with_nothing_to_cancel_does_nothing() -> None:
    assert typed(Interpreter(), ".@") == []


# -- macros and files given on the command line ---------------------------------------------


def test_a_macro_given_is_processed_and_its_status_kept(capsys: pytest.CaptureFixture[str]) -> None:
    write("seven.C", "int seven() { return 7; }")
    prompt = Interpreter()
    assert prompt.given("seven.C") == []
    assert capsys.readouterr().out == "Processing seven.C...\n(int) 7\n"
    assert prompt.status == 7 and not prompt.failed


def test_a_macro_given_that_is_missing_or_fails_is_reported() -> None:
    write("bad.C", "int bad() { return 1 / 0; }")
    write("unnamed.C", "{ }")
    prompt = Interpreter()
    assert [said.text for said in prompt.given("gone.C")] == [
        "Error in <TApplication::ExecuteFile>: macro gone.C not found in path ."
    ]  # fmt: skip
    (failed,) = prompt.given("bad.C")
    assert failed.text.startswith("bad.C:1:1: error: ZeroDivisionError")
    (refused,) = prompt.given("unnamed.C(2)")
    assert refused.text == "error: unnamed.C is an unnamed macro, which takes no arguments"
    assert prompt.status == 1


# -- the terminal ---------------------------------------------------------------------


def test_say_writes_errors_to_standard_error(capsys: pytest.CaptureFixture[str]) -> None:
    from xrdroot.pycint.interactive import Said

    say([Said("out"), Said("err", True)])
    assert capsys.readouterr() == ("out\n", "err\n")


def test_piped_lines_run_without_prompts_until_the_input_ends(
    capsys: pytest.CaptureFixture[str],
) -> None:
    lines = 'TH1F h("h","h",10,0,1); h.Fill(.5); h.GetEntries()\nnope\n'
    assert interact(Interpreter(), piped(lines)) == 0
    assert capsys.readouterr() == (
        "(double) 1.0000000\n",
        "ROOT_prompt_1:1:1: error: use of undeclared identifier 'nope'\n",
    )


def test_piped_lines_stop_at_dot_q() -> None:
    prompt = Interpreter()
    assert interact(prompt, piped(".q\nnever\n")) == 0
    assert prompt.number == 1


def test_at_a_terminal_lines_are_read_with_the_prompt_and_history_is_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    history: list[str] = []
    readline = types.SimpleNamespace(
        read_history_file=history.append,
        write_history_file=history.append,
        set_history_length=lambda length: None,
        set_completer=lambda completer: history.append(type(completer).__name__),
        parse_and_bind=lambda text: None,
    )
    monkeypatch.setattr(terminal, "_readline", lambda: readline)
    (tmp_path / "history").write_text("")
    asked: list[str] = []
    answers = iter(["1 + 1", "{", "KeyboardInterrupt", "2 + 2"])

    def answer(prompt: str) -> str:
        asked.append(prompt)
        given = next(answers, None)
        if given is None:
            raise EOFError
        if given == "KeyboardInterrupt":
            raise KeyboardInterrupt
        return given

    monkeypatch.setattr("builtins.input", answer)
    assert interact(Interpreter(), Terminal()) == 0
    assert asked == ["root [0] ", "root [1] ", "root (cont'ed, cancel with .@) [2] ",
                     "root [2] ", "root [3] "]  # fmt: skip
    assert capsys.readouterr().out == "(int) 2\n\n(int) 4\n"
    assert history == [str(tmp_path / "history"), "Completer", str(tmp_path / "history")]


def test_a_terminal_without_readline_reads_lines_plainly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(terminal, "_readline", lambda: None)

    def leave(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", leave)
    assert interact(Interpreter(), Terminal()) == 0


def test_a_terminal_without_a_history_file_starts_one(monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[str] = []
    readline = types.SimpleNamespace(
        read_history_file=lambda path: pytest.fail("there is no history to read"),
        write_history_file=written.append,
        set_history_length=lambda length: None,
        set_completer=lambda completer: None,
        parse_and_bind=lambda text: None,
    )
    monkeypatch.setattr(terminal, "_readline", lambda: readline)
    monkeypatch.setattr("builtins.input", lambda prompt: ".q")
    interact(Interpreter(), Terminal())
    assert len(written) == 1


def test_readline_is_python_s_or_none_where_python_has_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert terminal._readline() is sys.modules.get("readline", terminal._readline())
    monkeypatch.setitem(sys.modules, "readline", None)
    assert terminal._readline() is None


def test_tab_completes_the_prompts_names_then_roots() -> None:
    prompt = Interpreter()
    typed(prompt, "int THx = 1;")
    complete = Completer(prompt)
    found = [complete("TH1", state) for state in range(3)]
    assert found[:2] == ["TH1", "TH1C"]
    assert complete("THx", 0) == "THx"
    assert complete("THx", 1) is None


# -- the command line -------------------------------------------------------------------


def run(*argv: str, stdin: str = "") -> int:
    sys.stdin = piped(stdin)
    try:
        return main(list(argv))
    finally:
        sys.stdin = sys.__stdin__


def test_piped_input_prints_what_cling_would(capsys: pytest.CaptureFixture[str]) -> None:
    script = 'TH1F h("h","h",10,0,1); h.Fill(.5); h.GetEntries()\n'
    assert run("root", "-l", "-b", stdin=script) == 0
    assert capsys.readouterr().out == "(double) 1.0000000\n"


def test_cint_is_the_same_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("cint", "-l", stdin="2 * 21\n") == 0
    assert capsys.readouterr().out == "(int) 42\n"


def test_without_l_a_banner_says_this_is_xrdroot(capsys: pytest.CaptureFixture[str]) -> None:
    run("root", "-q")
    out = capsys.readouterr().out
    assert out.startswith("xrdroot ") and "This is not ROOT" in out


def test_the_banner_names_no_release_when_xrdroot_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib import metadata

    from xrdroot.cli.root import banner

    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    assert banner().startswith("xrdroot (not installed): ")


def test_q_runs_the_files_and_lines_then_leaves_with_the_last_macros_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("seven.C", "int seven() { return 7; }")
    status = run("root", "-l", "-b", "-q", "-e", "int i = 6", "-e", "i * 7", str(SIMPLE),
                 "seven.C", stdin="never\n")  # fmt: skip
    out = capsys.readouterr().out
    assert status == 7
    assert out.startswith("(int) 6\n(int) 42\nAttaching file ")
    assert out.endswith("\nProcessing seven.C...\n(int) 7\n")


def test_x_leaves_at_the_first_failure_as_it_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """``-x`` is ROOT's exit on an exception: C++ failing as it runs, not code that did not
    compile."""
    assert run("root", "-l", "-x", "-e", "nope", "-e", "throw 3;", "-e", "1 + 1",
               stdin="2 + 2\n") == 1  # fmt: skip
    assert capsys.readouterr().out == "\n"


def test_the_logon_and_logoff_macros_run_around_the_session_unless_n(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("rootlogon.C", '{ printf("on\\n"); }')
    write("rootlogoff.C", '{ printf("off\\n"); }')
    assert run("root", "-l", stdin="1\n") == 0
    assert capsys.readouterr().out == "on\n(int) 1\noff\n"
    assert run("root", "-l", "-n", stdin="1\n") == 0
    assert capsys.readouterr().out == "(int) 1\n"


def test_exit_in_a_line_given_ends_the_session_before_the_prompt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("seven.C", "int seven() { return 7; }")
    assert run("root", "-l", "-e", "exit(4);", "-e", "1", "seven.C", stdin="2\n") == 4
    assert capsys.readouterr().out == "\n"  # as root does before what the command line runs


def test_cint_names_the_root_subcommand_and_a_file_name_still_starts_the_shell() -> None:
    from xrdroot.cli import _as_shell

    assert _as_shell(["cint", "-l"]) == ["root", "-l"]
    assert _as_shell(["f.root"]) == ["shell", "f.root"]


# -- root's command line, as root.exe takes it (each checked against ROOT 6.40.04) ----------


def test_root_takes_its_flags_anywhere_and_its_lines_and_files_in_their_order() -> None:
    from xrdroot.cli.root import options

    given = options(["-l", "a.C", "-e", "x", "-b", "-q", "-x", "-n", "-t", "-a", "-memstat",
                     "-config", "c", "--web=off", "b.root", "-e"])  # fmt: skip
    assert given.items == [("file", "a.C"), ("e", "x"), ("file", "b.root"), ("e", "")]
    assert (given.no_banner, given.batch, given.quit, given.exit_on_error) == (True,) * 4
    assert (given.no_logon, given.threads, given.splash, given.memstat) == (True,) * 4
    assert options(["-?"]).help and options(["--version"]).version


def test_lines_and_macros_run_in_command_line_order_with_the_last_value_the_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("five.C", "int five() { return 5; }")
    write("v.C", 'void v() { printf("in v\\n"); }')
    status = run("root", "-l", "-b", "-q", "five.C()", "-e", 'printf("e1\\n")', "v.C",
                 "-e", 'printf("e2\\n")')  # fmt: skip
    assert status == 3  # printf's value, as root -q exits with it
    assert capsys.readouterr().out == (
        "Processing five.C()...\n(int) 5\ne1\n(int) 3\nProcessing v.C...\nin v\ne2\n(int) 3\n"
    )


@pytest.mark.parametrize(
    ("lines", "status"),
    [(["int x = 4;"], 4), (["3;"], 3), (["double q = 2.5;"], 2), (["int a[2] = {1, 2};"], 0),
     (['printf("ab");'], 2), (["{ int a = 1; }"], 0), (["nonsense_x"], 0),
     (["syntax error here"], 1), ([".ls"], 1), ([".L nothing.C"], 0), ([".q"], 0),
     (["int y = 1;", "[]{}();"], 0), (['TH1F *h = new TH1F("h", "h", 1, 0, 1);'], 255)],
)  # fmt: skip
def test_q_exits_with_the_value_root_s_process_line_gives_the_last_line(
    lines: list[str], status: int
) -> None:
    argv = [word for line in lines for word in ("-e", line)]
    assert run("root", "-l", "-b", "-q", *argv) == status


def test_a_missing_file_ends_root_before_anything_runs(capsys: pytest.CaptureFixture[str]) -> None:
    write("five.C", "int five() { return 5; }")
    assert run("root", "-l", "-b", "-q", "five.C", "gone.C") == 2
    assert capsys.readouterr() == ("", "Error in <TApplication::GetOptions>: macro gone.C not "
                                   "found\n")  # fmt: skip
    assert run("root", "-l", "-b", "-q", "gone.root") == 2
    assert capsys.readouterr().err == (
        "Warning in <TApplication::GetOptions>: file gone.root not found\n"
        "root: unrecognized option 'gone.root'\nTry 'root --help' for more information.\n"
    )


def test_a_macro_that_fails_ends_the_command_line_and_a_line_that_fails_does_not(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("bad.C", "int bad() { return 1 / 0; }")
    write("five.C", "int five() { return 5; }")
    assert run("root", "-l", "-b", "-q", "bad.C", "five.C") == 1
    assert "Processing five.C" not in capsys.readouterr().out
    assert run("root", "-l", "-b", "-q", "-e", "nope", "five.C") == 5
    assert run("root", "-l", "-b", "-q", "-e", ".x gone.C", "five.C") == 1


def test_gapplication_terminate_and_exit_in_a_macro_end_root_with_their_status() -> None:
    write("term.C", "void term() { gApplication->Terminate(7); }")
    write("ex.C", "void ex() { exit(9); }")
    assert run("root", "-l", "-b", "-q", "term.C") == 7
    assert run("root", "-l", "-b", "-q", "ex.C", "term.C") == 9
    assert run("root", "-l", "-b", "-q", "-e", "gApplication->Terminate()") == 0


def test_the_banner_shows_only_without_l_and_with_nothing_to_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run("root", "-b", "-q") == 0
    assert capsys.readouterr().out.startswith("xrdroot ")
    assert run("root", "-b", "-q", "-e", "1") == 1
    assert capsys.readouterr().out == "(int) 1\n"


def test_root_s_help_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("root", "-h") == 0
    assert capsys.readouterr().out.startswith("usage: root ")
    assert run("root", "--version") == 0
    assert capsys.readouterr().out.startswith("ROOT Version: 6.40.04")


def test_g_root_is_rint_in_a_root_session_and_py_root_again_after(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import xrdroot.pyroot as ROOT

    write("pw.C", 'void pw() { printf("%s %s %d\\n", gROOT->GetName(), gApplication->GetName(), '
          "gApplication->Argc()); }")  # fmt: skip
    assert run("root", "-l", "-b", "-q", "pw.C", "-e", ".pwd") == 1
    assert capsys.readouterr().out == (
        "Processing pw.C...\nRint TRint 7\nCurrent directory: Rint:/\nCurrent style:     Modern\n"
    )
    assert ROOT.gROOT.GetName() == "PyROOT"


def test_a_macro_compiled_with_aclic_on_the_command_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    write("sq.C", "int sq(int n) { return n * n; }")
    write("bad.C", "int bad() { return ; ; }}")
    assert run("root", "-l", "-b", "-q", "sq.C+(3)") == 9
    out = capsys.readouterr()
    assert out.out == "Processing sq.C+(3)...\n(int) 9\n" and "creating shared library" in out.err
    assert run("root", "-l", "-b", "-q", "-e", ".L sq.C+", "-e", "sq(4)") == 16
    assert run("root", "-l", "-b", "-q", "bad.C+") == 1
    assert capsys.readouterr().err.endswith("Error in <ACLiC>: Compilation failed!\n")
