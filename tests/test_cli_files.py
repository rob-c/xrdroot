"""``xrdroot rm``, ``mkdir`` and ``mv``: ROOT's ``rootrm``, ``rootmkdir`` and ``rootmv``.

The messages and statuses expected here are the ones ROOT 6.40's tools gave
for the same command lines; what the files hold afterwards is checked by
reading them back, and by ``xrdroot ls`` - which is ``rootls``.
"""

from __future__ import annotations

import io
import shutil

import numpy as np
import pytest

from xrdroot import Histogram, create, open_root
from xrdroot.cli import main
from xrdroot.cli import mkdir as mkdir_command
from xrdroot.cli import mv as mv_command
from xrdroot.cli import rm as rm_command
from xrdroot.cli.tooling import Flag, OptionError, Options, raw_parser
from xrdroot.writer import WritableDirectory

DATA = __import__("pathlib").Path(__file__).parent / "data"


def histogram(name: str = "h") -> Histogram:
    return Histogram.new(name, np.arange(4.0), np.asarray([1.0, 2.0, 3.0]))


@pytest.fixture
def made(tmp_path, monkeypatch):
    """``test.root`` in a working directory of its own: hpx, hprof, a tree, dir/h0 and dir/sub."""
    monkeypatch.chdir(tmp_path)
    with create("test.root") as out:
        out["hpx"] = histogram("hpx")
        out["hprof"] = histogram("hprof")
        out["events"] = {"x": np.arange(3, dtype=np.int32)}
        out["dir/h0"] = histogram("h0")
        out["dir/h1"] = histogram("h1")
        out.mkdir("dir/sub")
    return tmp_path


def run(capsys, *argv: str) -> tuple[int, str, str]:
    status = main(list(argv))
    out, err = capsys.readouterr()
    return status, out, err


def names(path: str = "test.root", where: str = "") -> list[str]:
    with open_root(path) as f:
        return (f[where] if where else f).keys()


# -- the option parser -------------------------------------------------------------------


FLAGS = (
    Flag(("-a", "--all")),
    Flag(("-n", "--name"), takes_value=True),
    Flag(("-v",), repeatable=True),
)


def test_short_flags_group_and_a_value_is_the_next_word_or_after_an_equals():
    parsed = Options(FLAGS).parse(["-av", "x", "--name=y", "-v", "--", "-z"])
    assert parsed.args == ["x", "-z"] and parsed.value("n") == "y" and parsed.count("v") == 2
    assert Options(FLAGS).parse(["-n", "z", "w"]).args == ["w"]
    assert Options(FLAGS).parse(["--all", "a"]).has("a")
    assert Options(FLAGS).value("n") is None


@pytest.mark.parametrize(
    ("argv", "said"),
    [
        (["-x"], "Unknown flag: -x"),
        (["-"], "Unknown flag: -"),
        (["-aa"], "Flag -a appeared more than once"),
        (["-n", "1", "--name", "2"], "Flag --name appeared more than once with the value: 1"),
        (["-n"], "Missing argument for flag -n"),
        (["--name", "-a"], "Missing argument for flag --name"),
        (["--all=3"], "Flag --all does not expect an argument"),
    ],
)
def test_a_command_line_root_would_refuse_is_refused_in_its_words(argv, said):
    with pytest.raises(OptionError, match=f"^{said}$"):
        Options(FLAGS).parse(argv)


def test_a_long_single_dash_flag_makes_single_dash_words_whole_names():
    flags = (Flag(("-v",)), Flag(("-vv",)))
    assert Options(flags).parse(["-vv", "-v"]).has("vv")
    with pytest.raises(OptionError, match="Unknown flag: -vx"):
        Options(flags).parse(["-vx"])


def test_a_raw_subcommand_keeps_its_whole_command_line():
    import argparse

    parser = argparse.ArgumentParser()
    raw_parser(parser.add_subparsers(dest="command"), "tool", "a tool")
    assert parser.parse_args(["tool", "-x", "--", "a:b"]).argv == ["-x", "--", "a:b"]
    sys_argv = __import__("sys").argv
    sys_argv_before = list(sys_argv)
    sys_argv[1:] = ["-q"]
    try:
        assert parser._subparsers._group_actions[0].choices["tool"].parse_args().argv == ["-q"]
    finally:
        sys_argv[:] = sys_argv_before


# -- the writer under them -----------------------------------------------------------------


def test_a_directory_being_written_forgets_a_key_and_says_what_a_name_holds(tmp_path):
    with create(io.BytesIO()) as out:
        out["h"] = histogram()
        out["h"] = histogram()
        out.mkdir("d")
        assert out.classname_of("h") == "TH1D" and out.classname_of("nothing") is None
        assert out.delete("h", 1) == 1 and out._cycles["h"] == 2
        assert out.delete("h") == 1 and "h" not in out._cycles
        assert out.delete("d") == 1 and "d" not in out._subdirs
        assert out.delete("nothing") == 0
        out["h"] = histogram()
        assert out._cycles["h"] == 1


# -- rm ------------------------------------------------------------------------------------


def test_rm_removes_every_cycle_of_what_it_matches(capsys, made):
    assert run(capsys, "rm", "-v", "test.root:hp*") == (
        0,
        "",
        "Info in <rootrm>: removed 'test.root:hprof'\nInfo in <rootrm>: removed 'test.root:hpx'\n",
    )
    assert names() == ["dir", "events"]


def test_rm_takes_a_directory_or_the_file_only_with_r(capsys, made):
    assert run(capsys, "rm", "test.root:dir", "test.root") == (
        0,
        "",
        (
            "Error in <rootrm>: cannot remove 'dir': is a directory. Use -r to remove it.\n"
            "Error in <rootrm>: cannot remove 'test.root': is a ROOT file. Use -r to remove it.\n"
        ),
    )
    assert run(capsys, "rm", "-r", "test.root:dir/sub")[0] == 0
    assert names(where="dir") == ["h0", "h1"]
    assert run(capsys, "rm", "-r", "test.root", "test.root:hpx") == (0, "", "")
    assert not made.joinpath("test.root").exists()


def test_rm_dry_run_says_what_would_go_and_takes_nothing(capsys, made):
    assert run(capsys, "rm", "-n", "test.root:hpx", "test.root:dir/h*")[2] == (
        "Info in <rootrm>: removed 'test.root:hpx'\n"
        "Info in <rootrm>: removed 'test.root:dir/h0'\n"
        "Info in <rootrm>: removed 'test.root:dir/h1'\n"
    )
    assert run(capsys, "rm", "-n", "-r", "test.root")[0] == 0
    assert names() == ["hpx", "hprof", "dir", "events"]


def test_rm_very_verbose_says_where_each_key_was(capsys, made):
    with open_root("test.root") as f:
        key = f.key("hpx")
    status, out, _ = run(capsys, "rm", "-vv", "test.root:hpx")
    assert status == 0
    assert out == f"Deleting key: hpx at address {key.seek_key}, nbytes = {key.nbytes}\n"


def test_rm_asks_first_with_i_and_takes_only_a_yes(capsys, made, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("y\nn\n"))
    status, out, _ = run(capsys, "rm", "-i", "test.root:hpx", "test.root:hprof")
    assert status == 0
    assert out == ("remove 'hpx' from 'test.root' ? (y/n) remove 'hprof' from 'test.root' ? (y/n) ")
    assert names() == ["hprof", "dir", "events"]
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert run(capsys, "rm", "-i", "-r", "test.root")[1] == "remove 'test.root' ? (y/n) "
    assert made.joinpath("test.root").exists()


def test_rm_stops_at_the_first_source_that_matches_nothing_unless_forced(capsys, made):
    status, _, err = run(capsys, "rm", "test.root:nothing", "test.root:hpx")
    assert status == 1 and err == "Error in <rootrm>: 'test.root:nothing' matches no objects.\n"
    assert run(capsys, "rm", "-f", "test.root:nothing", "none.root:x", "test.root:hpx")[0] == 0
    assert names() == ["hprof", "dir", "events"]


def test_rm_refuses_a_file_it_cannot_write(capsys, made):
    made.joinpath("test.root").chmod(0o444)
    assert run(capsys, "rm", "test.root:hpx") == (
        1,
        "",
        "Error in <rootrm>: File 'test.root' does not exist or is not writable.\n",
    )


def test_rm_says_why_a_file_would_not_go(capsys, made, monkeypatch):
    def failing(path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(rm_command.os, "unlink", failing)
    assert run(capsys, "rm", "-r", "test.root")[2] == (
        "Error in <rootrm>: failed to remove 'test.root': Permission denied\n"
    )
    assert run(capsys, "rm", "-r", "-f", "test.root")[2] == ""


def test_rm_reads_its_command_line_as_rootrm_does(capsys, made):
    usage = rm_command.USAGE
    assert run(capsys, "rm") == (1, "", usage)
    assert run(capsys, "rm", "-x") == (1, "", f"Unknown flag: -x\n{usage}")
    assert run(capsys, "rm", "-i", "-f", "test.root:x") == (
        1,
        "",
        f"Error in <rootrm>: -i and -f flags are mutually exclusive.\n{usage}",
    )
    assert run(capsys, "rm", "-h") == (0, "", usage + rm_command.HELP)
    assert run(capsys, "rm", "ftp://h/x.root") == (
        1,
        "",
        "Error in <rootrm>: unknown file protocol\n",
    )


# -- mkdir ---------------------------------------------------------------------------------


def test_mkdir_adds_a_directory_and_with_p_its_parents_and_the_file(capsys, made):
    assert run(capsys, "mkdir", "test.root:new") == (0, "", "")
    assert run(capsys, "mkdir", "-p", "-v", "test.root:a/b", "fresh.root:x") == (
        0,
        "",
        (
            "Info in <rootmkdir>: created directory 'test.root:a'\n"
            "Info in <rootmkdir>: created directory 'test.root:a/b'\n"
            "Info in <rootmkdir>: created file 'fresh.root'\n"
            "Info in <rootmkdir>: created directory 'fresh.root:x'\n"
        ),
    )
    assert names()[-2:] == ["new", "a"] and names("fresh.root") == ["x"]
    assert run(capsys, "mkdir", "-p", "test.root:a/b", "test.root:dir/sub") == (0, "", "")
    assert run(capsys, "mkdir", "empty.root") == (0, "", "") and names("empty.root") == []


@pytest.mark.parametrize(
    ("argv", "said"),
    [
        (["test.root:dir"], "error adding 'dir': a directory already exists at that path."),
        (
            ["test.root:hpx"],
            "error adding 'hpx': path is already taken by an object of type 'TH1D'",
        ),
        (
            ["-p", "test.root:hpx/x"],
            "error adding 'hpx/x': path 'hpx' is already taken by an object of type 'TH1D'",
        ),
        (
            ["test.root:a/b"],
            "cannot create directory 'a/b': parent directory 'a' does not "
            "exist. If you want to create the entire hierarchy, use the -p "
            "flag.",
        ),
        (
            ["new.root:x"],
            "cannot create directory 'new.root:x': file does not exist. Use the "
            "-p flag if you want to create the file alongside the directories.",
        ),
        (["test.root"], "cannot create file 'test.root': already exists."),
        (["junk.root"], "cannot create file 'junk.root': already exists."),
        (["-p", "junk.root:x"], "failed to open 'junk.root' for writing."),
    ],
)
def test_mkdir_says_what_is_in_the_way_as_rootmkdir_says_it(capsys, made, argv, said):
    made.joinpath("junk.root").write_bytes(b"junk")
    assert run(capsys, "mkdir", *argv) == (1, "", f"Error in <rootmkdir>: {said}\n")


def test_mkdir_checks_every_path_first_and_then_makes_what_it_can(capsys, made):
    assert run(capsys, "mkdir", "test.root:x", "test.root:a;1") == (
        1,
        "",
        "Error in <rootmkdir>: cannot specify cycle for the directory to create.\n",
    )
    assert run(capsys, "mkdir", "test.root:dir", "test.root:y")[0] == 1
    assert "y" in names()
    usage = mkdir_command.USAGE
    assert run(capsys, "mkdir") == (1, "", usage)
    assert run(capsys, "mkdir", "-x") == (1, "", f"Unknown flag: -x\n{usage}")
    assert run(capsys, "mkdir", "--help") == (0, "", usage + mkdir_command.HELP)
    assert (
        run(capsys, "mkdir", "ftp://h/x.root")[2] == "Error in <rootmkdir>: unknown file protocol\n"
    )


# -- mv ------------------------------------------------------------------------------------


def test_mv_renames_and_moves_into_a_directory(capsys, made):
    assert run(capsys, "mv", "test.root:hpx", "test.root:histo") == (0, "", "")
    assert names() == ["hprof", "dir", "events", "histo"]
    assert run(capsys, "mv", "test.root:histo", "test.root:dir") == (0, "", "")
    assert names(where="dir") == ["h0", "h1", "sub", "histo"]
    assert run(capsys, "mv", "test.root:hprof", "test.root:events") == (0, "", "")
    with open_root("test.root") as f:
        assert f.keys() == ["dir", "events"] and f.key("events").classname == "TH1D"


def test_mv_moves_a_directory_whole_into_another(capsys, made):
    assert run(capsys, "mv", "test.root:dir", "test.root:moved") == (0, "", "")
    assert names() == ["hpx", "hprof", "events", "moved"]
    assert names(where="moved") == ["h0", "h1", "sub"]
    with create("other.root") as out:
        out.mkdir("inside")
    assert run(capsys, "mv", "test.root:moved", "other.root:inside") == (0, "", "")
    assert names("other.root", "inside") == ["moved"] and "moved" not in names()


def test_mv_takes_patterns_and_files_to_another_file(capsys, made):
    assert run(capsys, "mv", "test.root:hp*", "out.root") == (0, "", "")
    assert names("out.root") == ["hpx", "hprof"]
    shutil.copy("out.root", "copy.root")
    assert run(capsys, "mv", "--recreate", "copy.root", "again.root") == (0, "", "")
    assert names("again.root") == ["hpx", "hprof"] and not made.joinpath("copy.root").exists()
    assert run(capsys, "mv", "test.root:events", "test.root:dir/h0", "out.root")[0] == 0
    assert names("out.root") == ["hpx", "hprof", "h0", "events"]


def test_mv_keeps_only_the_newest_of_several_cycles(capsys, made):
    with create("cycles.root") as out:
        out["h"] = histogram()
        out["h"] = histogram()
    status, _, err = run(capsys, "mv", "cycles.root:h", "moved.root")
    assert status == 0 and err == (
        "WARNING: Several versions of 'h' are present in "
        "'cycles.root'. Only the most recent will be considered.\n"
    )
    assert names("cycles.root") == [] and names("moved.root") == ["h"]


@pytest.mark.parametrize(
    ("argv", "status", "said"),
    [
        (["test.root:nothing", "out.root"], 0, "WARNING: can't find nothing in test.root\n"),
        (
            ["test.root:hpx", "out.txt"],
            1,
            "WARNING: out.txt: No such file (or extension not supported)\n",
        ),
        (["none.root:x", "out.root"], 1, ""),  # no file the pattern globs to: nothing to say
        (
            ["-c", "505", "test.root:hpx", "test.root:y"],
            1,
            "WARNING: can't change compression settings on existing file\n",
        ),
        (
            ["test.root:dir", "test.root:hpx"],
            1,
            "WARNING: cannot overwrite non-directory 'hpx' with directory 'dir'\n"
            "WARNING: error during copy of dir, it is not removed from test.root\n",
        ),
        (
            ["test.root:hp*", "test.root:events"],
            2,
            "WARNING: target 'events' is not a directory\n"
            "WARNING: error during copy of hpx, it is not removed from test.root\n"
            "WARNING: target 'events' is not a directory\n"
            "WARNING: error during copy of hprof, it is not removed from test.root\n",
        ),
    ],
)
def test_mv_says_what_went_wrong_as_rootmv_warns_it(capsys, made, argv, status, said):
    assert run(capsys, "mv", *argv) == (status, "", said)


def test_mv_reads_its_command_line_with_rootmvs_parser(capsys, made):
    assert run(capsys, "mv")[0] == 2
    status, out, _ = run(capsys, "mv", "-h")
    assert status == 0 and out.startswith("usage: rootmv [-h] [-c COMPRESS] [-i] [--recreate]")


def test_mv_asks_first_with_i(capsys, made, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    status, out, _ = run(capsys, "mv", "-i", "test.root:hpx", "out.root")
    assert status == 0
    assert out == "remove 'hpx' from 'test.root' ? (y/n) : " and "hpx" in names()
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    assert run(capsys, "mv", "-i", "test.root", "out2.root")[1] == "remove 'test.root' ? (y/n) : "


def test_mv_compresses_a_new_destination_as_told(capsys, made):
    assert run(capsys, "mv", "-c", "505", "test.root:hpx", "new.root") == (0, "", "")
    with open_root("new.root") as f:
        assert f.compression == 505


def test_mv_names_a_url_and_its_path_as_rootmv_does():
    assert mv_command._sources("root://host//f.root:h", wildcards=False) == [
        ("root://host//f.root", [["h"]])
    ]
    assert mv_command._sources("root://host//f.root") == [("root://host//f.root", [[]])]


def test_mv_says_when_a_destination_will_not_open_or_a_source_turns_unreadable(
    capsys, made, monkeypatch
):
    def refusing(*args, **kwargs):
        raise OSError("no")

    real = mv_command._destination
    monkeypatch.setattr(mv_command, "_destination", refusing)
    assert run(capsys, "mv", "test.root:hpx", "x.root") == (
        1,
        "",
        "WARNING: File x.root does not exist\n",
    )
    monkeypatch.setattr(mv_command, "_destination", real)
    move = mv_command._Move(False)
    with create("d.root") as move.dest:
        assert move._file("none.root", [["h"]], [], True) == 1
    assert capsys.readouterr().err == "WARNING: File none.root does not exist\n"


def test_mv_reports_a_key_that_went_before_it_could_be_deleted(capsys, made):
    move = mv_command._Move(False)
    with create("d.root") as out:
        assert move._delete("d.root", out, ["nothing"]) == 1
        assert move._delete("d.root", out, ["dir", "nothing"]) == 1
    assert capsys.readouterr().err == (
        "WARNING: object nothing was not existing, so it is not deleted\n" * 2
    )


def test_a_writable_directory_is_what_mkdir_and_mv_walk(made):
    with create("w.root") as out:
        assert isinstance(out.mkdir("a"), WritableDirectory)
        out["a/h"] = histogram()
        assert mkdir_command._kind(out, "a") == "TDirectory"
        move = mv_command._Move(False, out)
        assert move._kind(["a", "h"]) == "TH1D" and move._kind(["h", "x"]) is None


def test_mv_will_not_put_a_directory_where_an_object_of_its_name_is(capsys, made):
    with create("other.root") as out:
        out["inside/dir"] = histogram("dir")
        out["into/dir/sub"] = histogram("sub")
    assert run(capsys, "mv", "test.root:dir", "other.root:inside")[2] == (
        "WARNING: cannot overwrite non-directory 'dir' with directory 'dir'\n"
        "WARNING: error during copy of dir, it is not removed from test.root\n"
    )
    assert run(capsys, "mv", "test.root:dir", "other.root:into")[2] == (
        "WARNING: cannot overwrite non-directory 'sub' with directory 'sub'\n"
        "WARNING: error during copy of dir, it is not removed from test.root\n"
    )


def test_mv_warns_of_a_file_that_globs_but_will_not_open_and_paths_through_objects(capsys, made):
    made.joinpath("junk.root").write_bytes(b"junk")
    assert run(capsys, "mv", "junk.root:h", "out.root") == (  # rootmv itself raises here
        1,
        "",
        "WARNING: File junk.root does not exist\n" * 2,
    )
    assert run(capsys, "mv", "test.root:hpx/x", "out.root") == (
        0,
        "",
        "WARNING: can't find hpx/x in test.root\n",
    )
    with open_root("test.root") as f:
        assert mv_command._directory(f, ["nothing"]) is None


def test_mv_moves_after_a_yes(capsys, made, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
    assert run(capsys, "mv", "-i", "test.root:hpx", "out.root")[0] == 0
    assert "hpx" not in names() and names("out.root") == ["hpx"]


def test_mv_keeps_the_newest_cycle_whichever_order_the_keys_are_listed_in(monkeypatch):
    class Key:
        def __init__(self, cycle):
            self.name, self.cycle = "h", cycle

    class Listing:
        def all_keys(self):
            return [Key(2), Key(1)]

    monkeypatch.setattr(mv_command, "_directory", lambda reading, path: Listing())
    assert [key.cycle for key in mv_command._newest(None, [])] == [2]
