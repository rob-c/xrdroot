"""``xrdroot merge`` and ``xrdroot cp``: ``hadd`` and ``rootcp`` on the command line.

The flags are ``hadd``'s, spelled as ``hadd`` spells them - ``-f505`` and
``-fk`` run together - and each is checked to reach :func:`xrdroot.merge`
as the argument it stands for. Failures are a message and a status of one,
never a traceback.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pytest

from xrdroot import Histogram, create, open_root
from xrdroot.cli import COMMANDS, main
from xrdroot.cli import cp as cp_command
from xrdroot.cli import merge as merge_command

DATA = pathlib.Path(__file__).parent / "data"
HALVES = [str(DATA / "chain.flat.1.root"), str(DATA / "chain.flat.2.root")]


def parsed(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="xrdroot")
    subparsers = parser.add_subparsers(dest="command")
    merge_command.add_parser(subparsers)
    cp_command.add_parser(subparsers)
    return parser.parse_args(list(argv))


def histogram_file(path, name="h"):
    with create(str(path)) as out:
        out[name] = Histogram.new(name, np.arange(4.0), np.asarray([1.0, 2.0, 3.0]))
        out["note"] = "text"
    return str(path)


def test_the_subcommands_are_the_ones_the_package_names():
    assert COMMANDS == ["merge", "cp"]


def test_merge_merges_and_says_what_it_made(tmp_path, capsys):
    out = str(tmp_path / "out.root")
    assert main(["merge", out, *HALVES]) == 0
    said = capsys.readouterr().out
    assert "from 2 sources" in said and "70 copied as they were" in said
    with open_root(out) as f:
        assert f["tree"].num_entries == 10


@pytest.mark.parametrize(
    ("flag", "force", "compression", "keep", "first"),
    [
        ("-f", True, None, False, False),
        ("-fk", True, None, True, False),
        ("-ff", True, None, False, True),
        ("-ffk", True, None, True, True),
        ("-f505", True, 505, False, False),
        ("-fk404", True, 404, True, False),
        ("-f0", True, 0, False, False),
    ],
)
def test_the_f_family_is_spelled_as_hadd_spells_it(flag, force, compression, keep, first):
    args = parsed("merge", flag, "out.root", "in.root")
    assert (args.force, args.compression, args.keep_compression, args.first_compression) == (
        force,
        compression,
        keep,
        first,
    )
    assert args.target == "out.root" and args.sources == ["in.root"]


def test_f_alone_does_not_take_the_target_for_its_argument():
    args = parsed("merge", "-f", "200.root", "in.root")
    assert args.target == "200.root" and args.force


def test_a_setting_hadd_does_not_take_is_refused_by_the_parser(capsys):
    with pytest.raises(SystemExit):
        parsed("merge", "-f510", "out.root", "in.root")


def test_keep_and_first_reach_merge_as_what_they_mean(tmp_path):
    out = str(tmp_path / "out.root")
    assert main(["merge", "-fk505", out, *HALVES]) == 0
    with open_root(out) as f:
        assert f.compression == 505
    assert merge_command._keep(parsed("merge", "-ff", "o", "i")) is False
    assert merge_command._keep(parsed("merge", "o", "i")) is None


def test_the_other_flags_reach_merge(tmp_path, capsys):
    source = histogram_file(tmp_path / "a.root")
    out = str(tmp_path / "out.root")
    assert main(["merge", "-v", "0", out, source]) == 0
    assert capsys.readouterr().out == ""
    assert main(["merge", "-a", "-v", "3", out, source]) == 0
    said = capsys.readouterr()
    assert "h: merged" in said.out and "warning: 'note' is a string" in said.err
    with open_root(out) as f:
        assert f["h"].values().tolist() == [2.0, 4.0, 6.0]
    missing = str(tmp_path / "missing.root")
    assert main(["merge", "-f", "-k", "-T", "-O", "-j", "4", "-n", "2", out, missing, source]) == 0
    assert "passed over" in capsys.readouterr().err


def test_a_list_of_objects_is_skipped_or_taken_alone(tmp_path):
    source = histogram_file(tmp_path / "a.root")
    listed = tmp_path / "list.txt"
    listed.write_text("# the histograms\nh extra words\n\n")
    out = str(tmp_path / "out.root")
    assert main(["merge", "-f", "-L", str(listed), "-Ltype", "OnlyListed", out, source]) == 0
    with open_root(out) as f:
        assert f.keys() == ["h"]
    assert main(["merge", "-f", "-L", str(listed), "-Ltype", "SkipListed", out, source]) == 0
    with open_root(out) as f:
        assert f.keys() == ["note"]


def test_a_list_without_its_type_is_refused_with_a_message(tmp_path, capsys):
    source = histogram_file(tmp_path / "a.root")
    assert main(["merge", "-L", "list.txt", str(tmp_path / "o.root"), source]) == 1
    assert "-L and -Ltype go together" in capsys.readouterr().err


def test_a_merge_that_fails_says_why_and_returns_one(tmp_path, capsys):
    source = histogram_file(tmp_path / "a.root")
    out = tmp_path / "out.root"
    out.write_bytes(b"there")
    assert main(["merge", str(out), source]) == 1
    assert "xrdroot merge: " in capsys.readouterr().err and out.read_bytes() == b"there"


# -- cp ---------------------------------------------------------------------------


def test_a_source_is_a_file_and_after_its_name_what_to_take_from_it():
    assert cp_command.split("in.root") == ("in.root", None)
    assert cp_command.split("in.root:hists/*") == ("in.root", "hists/*")
    assert cp_command.split("root://host//store/in.root:h") == ("root://host//store/in.root", "h")
    assert cp_command.split("in.root:") == ("in.root", None)


def test_cp_copies_what_each_source_names_into_the_destination(tmp_path):
    source = histogram_file(tmp_path / "a.root")
    out = str(tmp_path / "out.root")
    assert main(["cp", f"{source}:h", f"{HALVES[0]}:tree", out]) == 0
    with open_root(out) as f:
        assert f.keys() == ["h", "tree"] and f.compression == 106  # as the first source is
    assert main(["cp", source, f"{out}:more"]) == 0  # added to, into a directory
    with open_root(out) as f:
        assert f["more"].keys() == ["h", "note"]


def test_cp_cuts_and_picks_columns_as_copytree_does(tmp_path):
    out = str(tmp_path / "out.root")
    argv = ["cp", "--cut", "N > 2", "--columns", "N,F32", "--slow", f"{HALVES[0]}:tree", out]
    assert main(argv) == 0
    with open_root(out) as f:
        assert f["tree"].keys() == ["N", "F32"] and f["tree"].num_entries == 2


def test_cp_can_write_the_destination_anew_and_compressed_as_told(tmp_path):
    source = histogram_file(tmp_path / "a.root")
    out = str(tmp_path / "out.root")
    assert main(["cp", source, out]) == 0
    assert main(["cp", "--recreate", "-c", "505", f"{source}:h", out]) == 0
    with open_root(out) as f:
        assert f.keys() == ["h"] and f.compression == 505
    assert main(["cp", "-c", "404", "-r", f"{source}:note", out]) == 0
    with open_root(out) as f:
        assert f.keys() == ["h", "note"]


def test_a_copy_that_fails_says_why_and_returns_one(tmp_path, capsys):
    source = histogram_file(tmp_path / "a.root")
    assert main(["cp", f"{source}:nothing", str(tmp_path / "out.root")]) == 1
    assert "xrdroot cp: " in capsys.readouterr().err
