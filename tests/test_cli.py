"""The ``xrdroot`` command: its registry of subcommands, ``ls``, ``info``, ``scan`` and ``draw``.

Every subcommand is run in this process, through :func:`xrdroot.cli.main`
exactly as the console script runs it, over the go-hep files in
``tests/data``; what is checked is the lines that carry the information,
not every space between them.
"""

from __future__ import annotations

import argparse
import importlib
import pathlib

import pytest

import xrdroot.cli
from clisupport import FakeMatplotlib
from xrdroot.cli import COMMANDS, main
from xrdroot.cli.info import compression
from xrdroot.cli.target import location
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.file import Directory
from xrdroot.hist import Histogram

DATA = pathlib.Path(__file__).parent / "data"


def data(name: str) -> str:
    return str(DATA / name)


@pytest.fixture
def fake_matplotlib(monkeypatch):
    return FakeMatplotlib().install(monkeypatch)


def run(capsys, *argv: str) -> tuple[int, list[str], str]:
    """The status, the lines written to standard output, and standard error."""
    status = main(list(argv))
    out, err = capsys.readouterr()
    return status, out.splitlines(), err


# -- the registry ------------------------------------------------------------------


def test_every_registered_subcommand_is_a_module_with_a_parser_and_a_runner():
    for name in COMMANDS:
        module = importlib.import_module(f"xrdroot.cli.{name}")
        assert callable(module.add_parser)
        assert callable(module.run)


def test_the_parser_has_a_subcommand_for_every_registered_name():
    parser = xrdroot.cli.build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert list(choices) == COMMANDS


def test_a_subcommand_that_says_nothing_about_its_status_succeeded(monkeypatch, capsys):
    monkeypatch.setattr(importlib.import_module("xrdroot.cli.info"), "run", lambda args: None)
    assert main(["info", data("simple.root")]) == 0


def test_a_refusal_is_one_line_on_standard_error_and_status_two(capsys):
    status, out, err = run(capsys, "ls", f"{data('simple.root')}:nothing")
    assert status == 2
    assert out == []
    assert err.startswith("xrdroot ls: 'nothing' is not in /; there is tree")


def test_no_arguments_at_all_start_the_shell(monkeypatch):
    seen = []
    monkeypatch.setattr("xrdroot.cli.shell.run", lambda args: seen.append(args) or 0)
    assert main([]) == 0
    assert seen[0].command == "shell"
    assert seen[0].files == []


def test_a_file_first_starts_the_shell_with_it_as_root_l_does(monkeypatch):
    seen = []
    monkeypatch.setattr("xrdroot.cli.shell.run", lambda args: seen.append(args) or 0)
    main([data("simple.root"), "--plain"])
    assert seen[0].files == [data("simple.root")]
    assert seen[0].plain


def test_the_arguments_come_from_the_command_line_unless_given(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["xrdroot", "info", data("simple.root")])
    assert main() == 0
    assert "ROOT version:      60600 (6.06/00)" in capsys.readouterr().out


def test_python_dash_m_is_the_same_command():
    module = importlib.import_module("xrdroot.__main__")
    assert module.main is main


# -- FILE[:path] ---------------------------------------------------------------------


def test_a_location_without_a_path_is_the_file_and_dash_k_names_the_path():
    assert location("root://host:1094//f.root:dir/h") == ("root://host:1094//f.root", "dir/h")
    assert location("https://host/data?id=7") == ("https://host/data?id=7", "")
    assert location("https://host/data?id=7", "/dir/h") == ("https://host/data?id=7", "dir/h")


# -- ls ------------------------------------------------------------------------------


def test_ls_lists_every_key_with_its_class_title_and_cycle(capsys):
    status, out, _ = run(capsys, "ls", data("graphs.root"))
    assert status == 0
    assert out[:2] == [f"=== [{data('graphs.root')}] ===", "version: 60806"]
    assert out[2].split() == ["TGraph", "tg", "graph", "without", "errors", "(cycle=1)"]
    assert len(out) == 5


def test_ls_walks_directories_down_indenting_each_level(capsys):
    _, out, _ = run(capsys, "ls", data("dirs-6.14.00.root"))
    assert out[2].startswith("TDirectory ")
    assert out[3].startswith("  TDirectory dir11")
    assert out[4].startswith("    TH1F")
    assert out[4].split()[1:3] == ["h1", "h1"]


def test_ls_of_a_path_lists_that_directory_or_that_key(capsys):
    _, out, _ = run(capsys, "ls", f"{data('dirs-6.14.00.root')}:dir1/dir11")
    assert [line.split()[0] for line in out[2:]] == ["TH1F"]
    _, out, _ = run(capsys, "ls", data("graphs.root"), "-k", "tge")
    assert [line.split()[1] for line in out[2:]] == ["tge"]


def test_ls_long_adds_what_each_record_costs(capsys):
    _, out, _ = run(capsys, "ls", "-l", data("dirs-6.14.00.root"))
    assert "(cycle=1, 345 bytes on disk, 936 uncompressed, x3.04, 2018-07-03 11:08:55)" in out[4]


def test_ls_trees_lists_each_column_with_its_type(capsys):
    _, out, _ = run(capsys, "ls", "-t", data("small-flat-tree.root"))
    assert out[2].endswith("(cycle=1, entries=100)")
    columns = {line.split()[0]: line.split()[1:] for line in out[3:]}
    assert columns["ArrayInt32"] == ['"ArrayInt32[10]"', "int32"]
    assert columns["Str"] == ['"Str"', "str"]
    assert len(columns) == 20


def test_ls_trees_long_adds_each_columns_baskets(capsys):
    _, out, _ = run(capsys, "ls", "-tl", data("small-flat-tree.root"))
    assert out[3].split()[-4:] == ["(1", "baskets,", "244", "bytes)"]


def test_ls_trees_lists_an_rntuples_fields_with_their_cxx_types(capsys):
    _, out, _ = run(capsys, "ls", "-t", data("rntuple/ntpl001_staff_rntuple_v1-0-0-0.root"))
    assert out[2].split()[:2] == ["ROOT::RNTuple", "Staff"]
    assert out[2].endswith("entries=3354)")
    assert out[-1].split() == ["Nation", "std::string", "str"]


def test_ls_trees_says_why_a_tree_would_not_open(monkeypatch, capsys):
    real = Directory.__getitem__

    def refusing(self, name):
        if name.startswith("tree"):
            raise UnsupportedFeatureError("this tree is made of something else")
        return real(self, name)

    monkeypatch.setattr(Directory, "__getitem__", refusing)
    _, out, _ = run(capsys, "ls", "-t", data("simple.root"))
    assert out[2].endswith("(cycle=1, unreadable: this tree is made of something else)")


def test_ls_of_several_files_puts_a_blank_line_between_them(capsys):
    _, out, _ = run(capsys, "ls", data("simple.root"), data("graphs.root"))
    assert out[3] == ""
    assert out[4] == f"=== [{data('graphs.root')}] ==="


# -- info ------------------------------------------------------------------------------


def test_info_says_what_the_header_says(capsys):
    status, out, _ = run(capsys, "info", data("gauss-h1.root"))
    assert status == 0
    said = {key: value.strip() for key, value in (line.split(":", 1) for line in out[:11])}
    assert said == {
        "file": data("gauss-h1.root"),
        "ROOT version": "60806 (6.08/06)",
        "format": "32-bit seeks",
        "size": "5310 bytes",
        "compression": "zlib, level 1 (1)",
        "UUID": "ba001cf4-0d75-11e7-84f8-5e789e86beef",
        "first record": "100",
        "free segments": "1 at 5253 (57 bytes)",
        "streamer info": "at 2180 (3073 bytes)",
        "keys at the top": "4",
        "classes described": "15",
    }
    assert "  TH1D (2 members)" in out


def test_info_of_several_files_puts_a_blank_line_between_them(capsys):
    _, out, _ = run(capsys, "info", data("simple.root"), data("gauss-h1.root"))
    assert "" in out


def test_compression_codes_are_said_in_words():
    assert compression(0) == "none (0)"
    assert compression(505) == "zstd, level 5 (505)"
    assert compression(207) == "lzma, level 7 (207)"
    assert compression(901) == "algorithm 9, level 1 (901)"


def test_info_says_an_uncompressed_file_is_uncompressed(capsys):
    name = "rntuple/rntviewer-testfile-uncomp-single-rntuple-v1-0-0-0.root"
    _, out, _ = run(capsys, "info", data(name))
    assert "compression:       none (0)" in out


# -- scan ------------------------------------------------------------------------------


def test_scan_prints_roots_table_for_the_rows_the_cut_keeps(capsys):
    status, out, _ = run(capsys, "scan", f"{data('simple.root')}:tree", "one:three", "one > 2")
    assert status == 0
    assert out[1] == "*    Row   *       one *     three *"
    assert out[3] == "*        2 *         3 *      tres *"
    assert out[-1] == "==> 2 selected entries"


def test_scan_of_a_file_with_one_tree_needs_no_tree_named(capsys):
    _, out, _ = run(capsys, "scan", data("simple.root"), "-n", "2", "--first", "1", "--width", "6")
    assert out[3] == "*        1 *      2 *    2.2 *    dos *"
    assert len(out) == 6


def test_scan_takes_roots_precision(capsys):
    _, out, _ = run(capsys, "scan", data("simple.root"), "two", "--precision", "3", "-n", "1")
    assert out[3] == "*        0 *       1.1 *"


# -- draw ------------------------------------------------------------------------------


def test_draw_without_a_picture_file_draws_in_characters(capsys):
    status, out, _ = run(
        capsys, "draw", data("small-flat-tree.root"), "Int32>>h(10, 0, 100)", "Int32 >= 50"
    )
    assert status == 0
    assert out[-1] == "==> 50 selected"
    assert len(out) == 11


def test_draw_into_a_picture_file_saves_the_figure(monkeypatch, capsys, tmp_path, fake_matplotlib):
    saved = []
    monkeypatch.setattr(Histogram, "plot", lambda self, **style: fake_matplotlib.axes(saved, style))
    target = tmp_path / "pt.png"
    status, out, _ = run(
        capsys,
        "draw",
        f"{data('small-flat-tree.root')}:tree",
        "Float64",
        "--style",
        "color=red",
        "-o",
        str(target),
    )
    assert status == 0
    assert out == [f"wrote {target}", "==> 100 selected"]
    assert saved == [(str(target), {"color": "red"})]


def test_the_subcommands_modules_are_what_main_imports():
    assert isinstance(xrdroot.cli.build_parser(), argparse.ArgumentParser)
