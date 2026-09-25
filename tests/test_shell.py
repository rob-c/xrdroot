"""The shell: ROOT's dot-commands as Python, the prompt, and ``%load_ext xrdroot`` in IPython.

The translation is tested as the function it is; the prompt is fed lines as
if they were typed, and IPython - which is not a dependency - is stood in for
by the few methods of a shell the extension calls.
"""

from __future__ import annotations

import builtins
import pathlib
import sys
import types

import pytest

import xrdroot
from xrdroot import gROOT
from xrdroot.cli import main
from xrdroot.cli.dot import transform, translate
from xrdroot.cli.magics import load, magics
from xrdroot.cli.shell import Console, banner, interact, namespace

DATA = pathlib.Path(__file__).parent / "data"
SIMPLE = str(DATA / "simple.root")
GRAPHS = str(DATA / "graphs.root")


@pytest.fixture(autouse=True)
def _a_clean_session():
    """Leave the one session as it was found: nothing held open, nothing in memory, at the top."""
    yield
    gROOT.close_all()
    for name in gROOT.keys():
        gROOT.remove(name)


def typed(monkeypatch, *lines: str) -> None:
    """Have the prompt read ``lines``, then the end of input."""
    waiting = iter(lines)

    def reading(prompt: str = "") -> str:
        try:
            return next(waiting)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(builtins, "input", reading)


# -- the dot-commands ------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "python"),
    [
        (".ls", "print(gROOT.ls())"),
        (".ls dir1", "print(gROOT.ls('dir1'))"),
        (".pwd", "print(gROOT.pwd())"),
        (".cd", "gROOT.cd()"),
        (".cd dir1/dir11", "gROOT.cd('dir1/dir11')"),
        ('.cd "my dir"', "gROOT.cd('my dir')"),
        (".cd 'f.root:/d'", "gROOT.cd('f.root:/d')"),
        (".x fill.py", "gROOT.macro('fill.py')"),
        (".x fill.py(1000, 'a')", "gROOT.macro('fill.py', 1000, 'a')"),
        (".x fill.py()", "gROOT.macro('fill.py')"),
        (".q", "exit()"),
        (".qqq", "exit()"),
        (".help", "print(gROOT.help())"),
        (".?", "print(gROOT.help())"),
    ],
)
def test_each_dot_command_is_the_python_it_stands_for(line, python):
    assert translate(line) == python


def test_dot_x_without_a_macro_says_what_it_needs():
    assert translate(".x") == (
        "print('.x needs a macro to run: .x macro.py, or .x macro.py(arguments)')"
    )


def test_an_unknown_command_says_so_rather_than_being_a_syntax_error():
    assert translate(".L macro.C") == (
        "print('.L is not a command here; .help lists the ones there are')"
    )


@pytest.mark.parametrize(
    "line",
    ["h = gROOT['h']", "    .Filter('x > 1')", ".5 * x", "print('.ls')", ""],
)
def test_python_is_left_as_it_is(line):
    assert translate(line) == line


def test_a_cell_has_each_command_line_translated_and_its_line_endings_kept():
    assert transform([".ls\n", "x = 1\n", ".pwd"]) == [
        "print(gROOT.ls())\n",
        "x = 1\n",
        "print(gROOT.pwd())",
    ]


# -- the shell ---------------------------------------------------------------------


def test_the_files_given_are_underscore_file_n_and_the_session_is_in_the_last(monkeypatch):
    names = namespace([SIMPLE, GRAPHS])
    assert names["_file0"].name == SIMPLE
    assert names["_file1"].name == GRAPHS
    assert gROOT.pwd() == f"{GRAPHS}:/"
    assert names["np"].__name__ == "numpy"
    assert names["Histogram"] is xrdroot.Histogram
    assert names["gROOT"] is gROOT


def test_the_banner_says_what_this_is_and_which_files_it_opened(monkeypatch):
    names = namespace([SIMPLE])
    monkeypatch.setattr("importlib.metadata.version", lambda name: "9.9")
    said = banner(names)
    assert said.startswith("xrdroot 9.9: ROOT files in Python")
    assert f"  _file0 = {SIMPLE}" in said.splitlines()


def test_the_banner_of_an_uninstalled_copy_says_so(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", missing)
    assert banner({}).startswith("xrdroot (not installed):")


def test_the_prompt_reads_dot_commands_and_python_alike(monkeypatch, capsys):
    typed(monkeypatch, ".pwd", "x = 40 + 2", "print(x)", ".cd tree")
    names = namespace([SIMPLE])
    interact(names, plain=True)
    out, err = capsys.readouterr()
    assert f"{SIMPLE}:/" in out
    assert "42" in out
    assert "NotADirectoryError: 'tree' is a TTree, not a directory" in err


def test_leaving_the_prompt_by_raising_system_exit_is_leaving(monkeypatch, capsys):
    typed(monkeypatch, "raise SystemExit", "print('not reached')")
    interact({}, plain=True)
    assert "not reached" not in capsys.readouterr().out


def test_the_console_translates_every_line_it_is_pushed():
    console = Console({"gROOT": gROOT})
    assert console.push(".cd") is False


def test_ipython_is_the_prompt_when_it_is_installed(monkeypatch, capsys):
    started = []
    fake = types.ModuleType("IPython")
    fake.start_ipython = lambda **given: started.append(given)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", fake)
    names = {"a": 1}
    interact(names)
    assert started == [{"argv": ["--no-banner", "--ext=xrdroot"], "user_ns": names}]
    assert capsys.readouterr().out.startswith("xrdroot ")


def test_without_ipython_the_prompt_is_the_standard_librarys(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "IPython", None)
    typed(monkeypatch, "print('plain')")
    interact({})
    assert "plain" in capsys.readouterr().out


def test_the_shell_subcommand_opens_files_runs_macros_and_quits(capsys, tmp_path):
    macro = tmp_path / "hello.py"
    macro.write_text("def hello():\n    print('hello from', gROOT.pwd())\n")
    assert main(["shell", "-q", SIMPLE, str(macro)]) == 0
    assert capsys.readouterr().out == f"hello from {SIMPLE}:/\n"
    assert gROOT.pwd() == "Rint:/"  # and what it opened is closed again


def test_the_shell_subcommand_prompts_unless_told_to_quit(monkeypatch, capsys):
    typed(monkeypatch, ".ls")
    assert main(["shell", "--plain", GRAPHS]) == 0
    assert "KEY: TGraph\ttg;1\tgraph without errors" in capsys.readouterr().out


# -- the IPython extension -------------------------------------------------------------


class FakeIPython:
    """The few parts of an IPython shell the extension touches."""

    def __init__(self) -> None:
        self.user_ns: dict = {}
        self.input_transformers_cleanup: list = []
        self.magics: dict = {}

    def push(self, names: dict) -> None:
        self.user_ns.update(names)

    def register_magic_function(self, function, magic_kind, magic_name) -> None:
        self.magics[magic_name] = (magic_kind, function)


def test_loading_the_extension_brings_the_names_the_commands_and_the_magics():
    shell = FakeIPython()
    xrdroot.load_ipython_extension(shell)
    load(shell)  # loading twice adds the translation once
    assert shell.user_ns["gROOT"] is gROOT
    assert shell.input_transformers_cleanup == [transform]
    assert {name: kind for name, (kind, _) in shell.magics.items()} == {
        "root_ls": "line",
        "root_open": "line",
        "root_macro": "cell",
    }


def test_root_open_opens_the_next_underscore_file_and_root_ls_lists_it(capsys):
    shell = FakeIPython()
    shell.user_ns["_file0"] = "taken"
    found = magics(shell)
    opened = found["root_open"][1](f" {GRAPHS} ")
    assert shell.user_ns["_file1"] is opened
    assert capsys.readouterr().out == f"_file1 = {GRAPHS}\n"
    found["root_ls"][1]("")
    assert "KEY: TGraph\ttg;1" in capsys.readouterr().out
    found["root_ls"][1](f"{GRAPHS}:")
    assert "KEY: TGraphErrors\ttge;1" in capsys.readouterr().out


def test_root_macro_runs_its_cell_with_the_names_preloaded():
    found = magics(FakeIPython())
    names = found["root_macro"][1]("", "h = Histogram.book('h', (2, 0.0, 1.0))\nn = np.int64(3)")
    assert names["h"].classname == "TH1D"
    assert names["n"] == 3
