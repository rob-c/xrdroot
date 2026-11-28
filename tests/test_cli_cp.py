"""``xrdroot cp``: ROOT's ``rootcp``, its destinations, refusals and messages.

What is expected is what ROOT 6.40's ``rootcp`` printed and left in the
file for the same command lines; ``--cut``, ``--columns`` and ``--slow``
are xrdroot's own, ``TTree::CopyTree`` on the way.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from xrdroot import Histogram, create, open_root
from xrdroot.cli import cp as cp_command
from xrdroot.cli import main

DATA = pathlib.Path(__file__).parent / "data"
HALF = str(DATA / "chain.flat.1.root")


def histogram(name: str = "h") -> Histogram:
    return Histogram.new(name, np.arange(4.0), np.asarray([1.0, 2.0, 3.0]))


@pytest.fixture
def made(tmp_path, monkeypatch):
    """``test.root`` to copy from: hpx, hprof, note, dir/h0 and dir/sub."""
    monkeypatch.chdir(tmp_path)
    with create("test.root") as out:
        out["hpx"] = histogram("hpx")
        out["hprof"] = histogram("hprof")
        out["note"] = "text"
        out["dir/h0"] = histogram("h0")
        out.mkdir("dir/sub")
    return tmp_path


def run(capsys, *argv: str) -> tuple[int, str, str]:
    status = main(["cp", *argv])
    out, err = capsys.readouterr()
    return status, out, err


def names(path: str, where: str = "") -> list[str]:
    with open_root(path) as f:
        return (f[where] if where else f).keys()


def test_cp_copies_what_each_source_names_compressed_as_the_first_is(capsys, made):
    assert run(capsys, "test.root:hpx", f"{HALF}:tree", "out.root") == (0, "", "")
    with open_root("out.root") as f:
        assert f.keys() == ["hpx", "tree"]
    assert run(capsys, "test.root", "all.root") == (
        0,
        "",
        (
            "Warning in <rootcp>: Directory 'dir' will not be copied. Use the -r option if you "
            "need a recursive copy.\n"
        ),
    )
    assert names("all.root") == ["hprof", "hpx", "note"]  # in name order, as rootcp goes


def test_cp_recursive_copies_directories_and_says_what_it_does(capsys, made):
    status, _, err = run(capsys, "-vv", "-r", "test.root:dir", "out.root")
    assert status == 0
    assert err == (
        "Info in <rootcp>: cp test.root:dir -> out.root:dir\n"
        'Info in <rootcp>: read object "dir" of type TDirectoryFile\n'
        "Info in <rootcp>: cp test.root:dir/h0 -> out.root:dir/h0\n"
        "Info in <rootcp>: mkdir dir\n"
        'Info in <rootcp>: read object "dir/h0" of type TH1D\n'
        "Info in <rootcp>: cp test.root:dir/sub -> out.root:dir/sub\n"
        "Info in <rootcp>: mkdir dir\n"
        'Info in <rootcp>: read object "dir/sub" of type TDirectoryFile\n'
    )
    assert names("out.root", "dir") == ["h0", "sub"]


def test_cp_renames_into_a_new_name_its_record_moved_to_its_new_key(capsys, made):
    assert run(capsys, "-v", "test.root:hpx", "test.root:a_longer_name")[2] == (
        "Info in <rootcp>: cp test.root:hpx -> test.root:a_longer_name\n"
    )
    with open_root("test.root") as f:
        assert f["a_longer_name"].values().tolist() == f["hpx"].values().tolist()
    assert run(capsys, "test.root:hpx", "test.root:dir") == (0, "", "")
    assert names("test.root", "dir") == ["h0", "sub", "hpx"]


def test_cp_replaces_only_when_told(capsys, made):
    assert run(capsys, "test.root:hpx", "test.root:hprof") == (
        1,
        "",
        (
            'Error in <rootcp>: destination path "test.root:hprof" already exists (as an object of '
            "type TH1D). Use the --replace flag to overwrite it.\n"
        ),
    )
    assert run(capsys, "--replace", "test.root:hpx", "test.root:hprof") == (0, "", "")
    with open_root("test.root") as f:
        assert [key.cycle for key in f.all_keys() if key.name == "hprof"] == [1]
    assert run(capsys, "test.root:hpx", "out.root") == (0, "", "")
    assert run(capsys, "test.root:hpx", "out.root") == (
        0,
        "",
        (
            "Error in <rootcp>: an object of type 'TH1D' already exists at out.root:hpx. Use the "
            "--replace flag to overwrite existing objects.\n"
        ),
    )
    assert run(capsys, "--replace", "test.root:hpx", "out.root") == (0, "", "")


def test_cp_will_not_copy_a_thing_onto_itself_nor_recreate_a_source(capsys, made):
    assert run(capsys, "-r", "test.root:dir", "test.root") == (
        0,
        "",
        ("Error in <rootcp>: test.root:dir: source and destination cannot be the same\n"),
    )
    assert run(capsys, "--recreate", "test.root:hpx", "test.root") == (
        1,
        "",
        ("Error in <rootcp>: cannot recreate destination file if this is also a source file\n"),
    )


def test_cp_refuses_several_things_for_one_new_name(capsys, made):
    assert run(capsys, "test.root:hpx", "test.root:hprof", "out.root:x") == (
        1,
        "",
        (
            'Error in <rootcp>: multiple sources were specified, but destination path "out.root:x" '
            "is not a directory.\n"
        ),
    )
    assert run(capsys, "test.root:hp*", "out.root:x") == (
        1,
        "",
        (
            'Error in <rootcp>: multiple sources were specified but destination path "out.root:x" '
            "is not a directory.\n"
        ),
    )
    assert not made.joinpath("out.root").exists()  # a failed run takes its destination away


def test_cp_says_where_it_found_nothing_and_stops(capsys, made):
    assert run(capsys, "test.root:nothing", "test.root:") == (
        1,
        "",
        ("Error in <rootcp>: 'test.root:nothing' matches no objects.\n"),
    )
    assert run(capsys, "none.root", "out.root") == (
        1,
        "",
        (
            f"Error in <TFile::TFile>: file {made}/none.root does not exist\n"
            "Error in <rootcp>: Failed to open file 'none.root'\n"
        ),
    )


def test_cp_takes_compression_only_for_a_new_file_as_rootcp_does_not(capsys, made):
    # sic: rootcp refuses -c for the file it would make, and takes it for one already there
    assert run(capsys, "-c", "505", "test.root:hpx", "new.root") == (
        1,
        "",
        ("Error in <rootcp>: can't change compression settings on existing file new.root\n"),
    )
    assert run(capsys, "test.root:hpx", "old.root") == (0, "", "")
    assert run(capsys, "--recreate", "-c", "505", "test.root:hpx", "old.root") == (0, "", "")
    with open_root("old.root") as f:
        assert f.compression == 505


def test_cp_says_a_destination_in_no_folder_does_not_exist(capsys, made):
    assert run(capsys, "test.root:hpx", "nodir/x.root") == (
        1,
        "",
        (
            f"SysError in <TFile::TFile>: file {made}/nodir/x.root can not be opened No such file "
            "or directory\nError in <rootcp>: File nodir/x.rootdoes not exist.\n"
        ),
    )
    made.joinpath("junk.root").write_bytes(b"junk")
    assert run(capsys, "test.root:hpx", "junk.root") == (
        1,
        "",
        "Error in <rootcp>: File junk.rootdoes not exist.\n",
    )


def test_cp_says_each_tree_copied_twice_is_already_there(capsys, made):
    with create("cycles.root") as out:
        out["t"] = {"x": np.arange(2, dtype=np.int32)}
        out["t"] = {"x": np.arange(3, dtype=np.int32)}
    assert run(capsys, "cycles.root", "out.root")[2] == (
        "Error in <rootcp>: an object of type 'TTree' already exists at out.root:t. Use the "
        "--replace flag to overwrite existing objects.\n"
    )


def test_cp_says_what_it_could_not_copy(capsys, made):
    assert run(capsys, f"{DATA / 'chain.1.root'}:tree", "out.root")[2].startswith(
        'Error in <rootcp>: failed to read object "tree". '
    )


def test_cp_compresses_a_new_destination_as_a_source_it_cannot_read_would_not_say(made):
    assert cp_command._compression(None, [("none.root", "")]) == {
        "compression": None,
        "level": None,
    }
    made.joinpath("junk.root").write_bytes(b"junk")
    assert cp_command._compression(None, [("junk.root", "")])["compression"] is None


def test_cp_cuts_and_picks_columns_as_copytree_does(capsys, made):
    argv = ["--cut", "N > 2", "--columns", "N,F32", "--slow", f"{HALF}:tree", "out.root"]
    assert run(capsys, *argv) == (0, "", "")
    with open_root("out.root") as f:
        assert f["tree"].keys() == ["N", "F32"] and f["tree"].num_entries == 2


def test_cp_reads_its_command_line_as_rootcp_does(capsys, made):
    usage = cp_command.USAGE
    assert run(capsys, "test.root") == (1, "", usage)
    assert run(capsys, "-c101", "a.root", "b.root") == (1, "", f"Unknown flag: -c101\n{usage}")
    assert run(capsys, "--help") == (0, "", usage + cp_command.HELP)
    assert run(capsys, "ftp://h/a.root", "b.root") == (
        1,
        "",
        "Error in <rootcp>: unknown file protocol\n",
    )


def test_a_path_through_an_object_names_nothing_in_the_destination(made):
    with create("w.root") as out:
        out["h"] = histogram()
        out["d/x"] = histogram("x")
        assert cp_command._kind(out, "h/x") is None
        assert cp_command._kind(out, "d/x") == "TH1D"


def test_cp_copies_into_a_directory_already_there_without_replace(capsys, made):
    assert run(capsys, "-r", "test.root:dir", "out.root") == (0, "", "")
    assert run(capsys, "-r", "test.root:dir/sub", "out.root:dir") == (0, "", "")
    assert names("out.root", "dir") == ["h0", "sub"]


def test_a_plain_remote_pair_is_a_whole_file_copy_and_an_object_path_is_not():
    """``cp root://host//f.root here.root`` means the bytes, not the objects inside."""
    from xrdroot.cli.cp import _whole_file_copy

    class Options:
        def __init__(self, args):
            self.args = args

    remote = "root://eospublic.cern.ch//eos/root-eos/h1/dstarmb.root"
    assert _whole_file_copy(Options([remote, "here.root"])) == (remote, "here.root")
    assert _whole_file_copy(Options([remote + ":hist", "here.root"])) is None
    assert _whole_file_copy(Options(["local.root", "here.root"])) is None
    assert _whole_file_copy(Options([remote, "here.root", "third.root"])) is None


def test_a_whole_file_copy_streams_the_bytes_into_the_destination(tmp_path, monkeypatch):
    """``_streamed`` writes what the remote stream gives, and says 0 when it worked."""
    import io as _io

    import xrdclient

    from xrdroot.cli.cp import _streamed

    monkeypatch.setattr(xrdclient, "open", lambda _url: _io.BytesIO(b"rootpayload"))
    dest = tmp_path / "copied.root"
    assert _streamed("root://host//f.root", str(dest), None) == 0
    assert dest.read_bytes() == b"rootpayload"


def test_a_whole_file_copy_that_cannot_open_says_so(tmp_path, monkeypatch, capsys):
    """A source that will not open is reported in ROOT's words, and the status is 1."""
    import xrdclient

    from xrdroot.cli.cp import _streamed

    def refuse(_url):
        raise OSError("no such file")

    monkeypatch.setattr(xrdclient, "open", refuse)
    assert _streamed("root://host//f.root", str(tmp_path / "x.root"), None) == 1
    assert "can not be opened" in capsys.readouterr().err


def test_a_copy_that_opens_but_fails_while_reading_is_reported(tmp_path, monkeypatch, capsys):
    """A stream that breaks part way is the same refusal as one that will not open."""
    import xrdclient

    from xrdroot.cli.cp import _streamed

    class Broken:
        def __enter__(self):
            return self

        def __exit__(self, *why):
            return False

        def read(self, _size):
            raise OSError("connection reset")

    monkeypatch.setattr(xrdclient, "open", lambda _url: Broken())
    assert _streamed("root://host//f.root", str(tmp_path / "y.root"), None) == 1
    assert "can not be opened" in capsys.readouterr().err


def test_the_command_streams_a_plain_remote_pair(tmp_path, monkeypatch):
    """``xrdroot cp root://host//f.root here.root`` runs the streamed copy, not rootcp."""
    import io as _io

    import xrdclient

    from xrdroot.cli import cp as cp_cli

    monkeypatch.setattr(xrdclient, "open", lambda _url: _io.BytesIO(b"payload"))
    dest = tmp_path / "streamed.root"
    args = type("Args", (), {"argv": ["root://host//f.root", str(dest)]})()
    assert cp_cli.run(args) == 0
    assert dest.read_bytes() == b"payload"
