"""``xrdroot hadd``: ROOT's ``hadd``, its command line, its words and its statuses.

Each expectation is what ROOT 6.40's ``hadd`` printed for the same command
line - roottest's ``root/io/hadd`` cases among them - on standard output
(``hadd Source file``, ``hadd Target path``) and standard error (``Info in
<hadd>:`` and the rest).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from xrdroot import Histogram, create, open_root
from xrdroot.cli import hadd as hadd_command
from xrdroot.cli import main

DATA = pathlib.Path(__file__).parent / "data"


def histogram(name: str = "h") -> Histogram:
    return Histogram.new(name, np.arange(4.0), np.asarray([1.0, 2.0, 3.0]))


@pytest.fixture
def made(tmp_path, monkeypatch):
    """``in.root``, compressed as ROOT's 101: a histogram and a tree."""
    monkeypatch.chdir(tmp_path)
    with create("in.root", compression="zlib", level=1) as out:
        out["h"] = histogram()
        out["t"] = {"x": np.arange(3, dtype=np.int32)}
    return tmp_path


def run(capsys, *argv: str) -> tuple[int, str, str]:
    status = main(["hadd", *argv])
    out, err = capsys.readouterr()
    return status, out, err


TARGET = "Info in <hadd>: target file: {}\n"
SETTING = "Info in <hadd>: compression setting for all output: {}\n"
UNEXPECTED = "the argument after '-{}' flag was not of the expected type."


def test_hadd_merges_saying_its_target_setting_sources_and_path(capsys, made):
    assert run(capsys, "-n0", "out.root", "in.root", "in.root") == (
        0,
        "hadd Source file 1: in.root\nhadd Source file 2: in.root\nhadd Target path: out.root:/\n",
        TARGET.format("out.root") + SETTING.format(101),
    )
    with open_root("out.root") as f:
        assert f["t"].num_entries == 6 and f.compression == 101


def test_hadd_verbosity_hides_or_adds_to_what_it_says(capsys, made):
    assert run(capsys, "o1.root", "in.root", "-v1") == (0, "", "")
    status, _, err = run(capsys, "-v=3", "o2.root", "in.root")
    assert status == 0
    assert err.endswith("Info in <hadd>: merged 1 (1) input (partial) files into o2.root\n")
    assert run(capsys, "-v", "o3.root", "in.root")[0] == 0


def test_hadd_says_which_compression_and_warns_when_it_differs(capsys, made):
    assert run(capsys, "-f0", "--", "--out.root", "in.root")[2] == (
        TARGET.format("--out.root") + SETTING.format(0) + "Warning in <hadd>: Sources and "
        "Target have different compression settings\nhadd merging will be slower\n"
    )
    assert run(capsys, "-fk", "o.root", "in.root")[2].endswith(
        "compression setting for meta data: 101\n"
    )
    assert run(capsys, "-ff", "-fk505", "o2.root", "in.root")[0] == 1
    assert run(capsys, "-fk505", "o3.root", "in.root")[2].endswith("meta data: 505\n")
    status, _, err = run(capsys, "-f5", "o4.root", "in.root")
    assert status == 0
    assert err.startswith(
        "Warning in <hadd>: interpreting 5 as 105. This behavior is "
        "deprecated, please use the full compression settings.\n"
    )
    with open_root("o4.root") as f:
        assert f.compression == 105


def test_hadd_takes_the_first_inputs_compression_when_asked(capsys, made):
    with create("in505.root", compression="zstd", level=5) as out:
        out["h"] = histogram()
    assert run(capsys, "-ff", "o.root", "in505.root")[2].endswith(SETTING.format(505))
    with open_root("o.root") as f:
        assert f.compression == 505
    made.joinpath("junk.root").write_bytes(b"junk")
    assert run(capsys, "-ff", "o2.root", "junk.root")[2].startswith(
        TARGET.format("o2.root") + SETTING.format(101)
    )


def test_hadd_parallel_flags_are_said_and_run_here_one_after_another(capsys, made):
    assert run(capsys, "-ff", "o.root", "in.root", "in.root", "-dbg", "-j", "2")[2] == (
        "Info in <hadd>: parallelizing  with 2 processes.\n"
        + TARGET.format("o.root")
        + SETTING.format(101)
        + "Info in <hadd>: each process should handle at least 3 files "
        "for efficiency. Setting the number of processes to: 1\n"
    )
    many = ["in.root"] * 6
    assert run(capsys, "-j", "2", "-d", ".", "o2.root", *many)[0] == 0


def test_hadd_filters_with_a_list_and_skips_or_keeps_only_what_it_lists(capsys, made):
    made.joinpath("list.txt").write_text("# a comment\nh\n\n")
    made.joinpath("empty.txt").write_text("# nothing\n")
    assert run(capsys, "-L", "list.txt", "-Ltype", "SkipListed", "o.root", "in.root")[2].endswith(
        "Info in <hadd>: added 1 object from filter file 'list.txt'\n"
    )
    with open_root("o.root") as f:
        assert f.keys() == ["t"]
    assert run(capsys, "-L=list.txt", "-Ltype=OnlyListed", "o2.root", "in.root")[0] == 0
    with open_root("o2.root") as f:
        assert f.keys() == ["h"]
    assert run(capsys, "-L", "empty.txt", "-Ltype", "OnlyListed", "o3.root", "in.root")[2].endswith(
        "Warning in <hadd>: no objects were added from filter file 'empty.txt'\n"
    )
    assert run(capsys, "-L", "none.txt", "-Ltype", "OnlyListed", "o4.root", "in.root")[2].endswith(
        "Error in <hadd>: error opening filter file 'none.txt'\n"
    )


def test_hadd_leaves_out_trees_and_appends_when_told(capsys, made):
    assert run(capsys, "-T", "-O", "o.root", "in.root")[0] == 0
    with open_root("o.root") as f:
        assert f.keys() == ["h"]
    assert run(capsys, "-a", "o.root", "in.root")[0] == 0
    with open_root("o.root") as f:
        assert [key.cycle for key in f.all_keys() if key.name == "h"] == [1, 2]


def test_hadd_validates_its_inputs_and_passes_bad_ones_over_with_k(capsys, made):
    made.joinpath("junk.txt").write_text("hi")
    made.joinpath("list.txt").write_text("in.root\n\njunk.txt\n")
    assert run(capsys, "-f", "o.root", "none.root") == (
        1,
        "",
        (
            TARGET.format("o.root") + 'Error in <hadd>: could not validate argument "none.root" as '
            "input file \n"
        ),
    )
    assert run(capsys, "-f", "o.root", "@list.txt") == (
        1,
        "hadd Source file 1: in.root\nhadd Source file 2: junk.txt\n",
        TARGET.format("o.root")
        + SETTING.format(101)
        + "Error in <TFile::ReadBuffer>: error reading all requested bytes from file junk.txt, "
        "got 2 of 300\nError in <TFile::Init>: junk.txt failed to read the file type data.\n"
        "Error in <TFileMerger::AddFile>: cannot open file junk.txt\n"
        "Error in <hadd>: exiting due to error in junk.txt\n"
        "Error in <hadd>: failure during the merge of 2 (1) input (partial) files into o.root\n",
    )
    status, out, err = run(
        capsys, "-k", "-f", "o.root", "none.root", "junk.txt", "@no.txt", "@list.txt", "in.root"
    )
    assert status == 0 and out.endswith("hadd Target path: o.root:/\n")
    assert "Warning in <hadd>: skipping file with error: junk.txt\n" in err
    assert "Error in <hadd>: could not open indirect file no.txt\n" in err
    made.joinpath("bad.txt").write_text("none.root\n")
    assert run(capsys, "-f", "o.root", "@bad.txt")[2].endswith(
        'Error in <hadd>: could not validate the file name "none.root" within indirect file '
        "bad.txt\n"
    )
    assert run(capsys, "-f", "o.root", "@no.txt")[2].endswith(
        "Error in <hadd>: could not open indirect file no.txt\n"
    )
    assert run(capsys, "-k", "-f", "o.root", "none.root")[2].endswith(
        "Error in <hadd>: could not find any valid input file \n"
    )


def test_hadd_says_a_file_that_is_no_root_file_will_not_open(made):
    made.joinpath("big.bin").write_bytes(b"x" * 400)
    assert hadd_command._unreadable("big.bin") == (
        "Error in <TFileMerger::AddFile>: cannot open file big.bin\n"
    )


def test_hadd_will_not_write_over_a_target_without_f_or_where_it_cannot(capsys, made):
    assert run(capsys, "o.root", "in.root")[0] == 0
    assert run(capsys, "o.root", "in.root") == (
        1,
        "",
        (
            TARGET.format("o.root")
            + SETTING.format(101)
            + f"Error in <TFile::TFile>: file {made}/o.root already exists\n"
            "Error in <TFileMerger::OutputFile>: cannot open the MERGER output file \n"
            "Error in <hadd>: error opening target file (does o.root exist?).\n"
            'pass "-f" argument to force re-creation of output file.\n'
        ),
    )
    assert run(capsys, "-f", "in.root", "in.root")[2].endswith(
        "Error in <hadd>: file in.root cannot be both the target and an input!\n"
    )
    for flags, status, said in (
        (["-a"], 2, "error opening target file for update :nodir/x.root."),
        (["-f"], 1, "error opening target file (does nodir/x.root exist?)."),
    ):
        assert run(capsys, *flags, "nodir/x.root", "in.root") == (
            status,
            "",
            (
                TARGET.format("nodir/x.root")
                + SETTING.format(101)
                + f"SysError in <TFile::TFile>: file {made}/nodir/x.root can not be opened No such "
                "file or directory\nError in <TFileMerger::OutputFile>: cannot open the MERGER "
                f"output file \nError in <hadd>: {said}\n"
            ),
        )


def test_hadd_says_why_a_merge_failed(capsys, made, monkeypatch):
    def failing(*args, **kwargs):
        raise ValueError("it went wrong")

    monkeypatch.setattr(hadd_command, "merge", failing)
    assert run(capsys, "o.root", "in.root")[2].endswith(
        "Error in <hadd>: it went wrong\n"
        "Error in <hadd>: failure during the merge of 1 (1) input (partial) files into o.root\n"
    )


@pytest.mark.parametrize(
    ("argv", "said"),
    [
        (["-na", "o.root", "in.root"], UNEXPECTED.format("n")),
        (["-n", "o.root", "in.root"], UNEXPECTED.format("n")),
        (["-jfa", "o.root", "in.root"], UNEXPECTED.format("j")),
        (["-j", "10a", "o.root", "in.root"], "error parsing integer argument '10a'"),
        (["o.root", "in.root", "-n"], "expected argument after '-n' flag."),
        (
            ["o.root", "in.root", "-L", "x", "-Ltype", "onlylisted"],
            "invalid argument for -Ltype: 'onlylisted'. Can only be 'SkipListed' or 'OnlyListed' "
            "(case matters).",
        ),
        (["o.root", "in.root", "-L", "x"], "-L must always be passed along with -Ltype."),
        (
            ["o.root", "in.root", "-Ltype", "SkipListed"],
            "-L must always be passed along with -Ltype.",
        ),
        (
            ["-cachesize", "10x", "o.root", "in.root"],
            "could not parse the cache size passed after -cachesize: '10x'",
        ),
        (["-f511", "o.root", "in.root"], "511 is not a supported compression settings."),
        (["-fx", "o.root", "in.root"], "invalid flag: fx"),
        (
            ["-ff", "-f505", "o.root", "in.root"],
            "cannot specify both -ff and -f[0-9]. Either use the first input compression or "
            "specify it.",
        ),
        (["-f101", "-f101", "o.root", "in.root"], "cannot specify -f[0-9] multiple times!"),
        (
            ["-ff505", "o.root", "in.root"],
            "cannot specify both -ff and -f[0-9]. Either use the first input compression or "
            "specify it.",
        ),
        (
            ["-d", "/nonexistent", "o.root", "in.root"],
            "could not access the directory specified: /nonexistent.",
        ),
        (
            ["o.root", "--", "in.root"],
            "found `--`, but we've already parsed (or are still parsing) a sequence of positional "
            "arguments!",
        ),
        ([], f"missing output file.\n{hadd_command.USAGE}"),
        (["o.root"], f"missing input file.\n{hadd_command.USAGE}"),
        (
            ["-ff", "o.root", "-k", "in.root"],
            "seen a positional argument 'in.root' after some flags. Positional arguments were "
            "already parsed at this point (from 'o.root' onwards), and you can only have one "
            "sequence of them, so you cannot pass more. Please group your positional arguments all "
            "together so that hadd works as you expect.\nCmdline: hadd -ff o.root -k in.root ",
        ),
    ],
)
def test_hadd_refuses_a_command_line_in_its_words(capsys, made, argv, said):
    assert run(capsys, *argv) == (1, "", f"Error in <hadd>: {said}\n")


def test_hadd_says_the_other_order_of_ff_and_a_setting_its_other_way(capsys, made):
    assert run(capsys, "-f505", "-ff", "o.root", "in.root") == (
        1,
        "",
        (
            "[err] Cannot specify both -ff and -f[0-9]. Either use the first input compression or "
            "specify it.\n"
        ),
    )


def test_hadd_warns_of_flags_it_does_not_know_or_was_given_twice(capsys, made):
    status, _, err = run(capsys, "-x", "-T", "-T", "-ff", "-ff", "-fkk",
                         "-experimental-io-features", "foo,kGenerateOffsetMap",
                         "-cachesize", "1Ki", "o.root", "in.root")  # fmt: skip
    assert status == 0
    assert err.startswith(
        "Warning in <hadd>: unknown flag: -x\nWarning in <hadd>: duplicate flag: T\n"
        "Warning in <hadd>: duplicate flag: -ff\nWarning in <hadd>: duplicate flag: -fk\n"
        "Error in <Set>: Could not locate foo in TBasket::EIOBits\n"
        "Warning in <hadd>: ignoring unknown feature request: foo\n"
        + TARGET.format("o.root")
        + "Info in <hadd>: Using cachesize=1Ki\n"
    )


def test_hadd_help_is_its_usage_and_options_on_standard_error(capsys, made):
    for flag in ("-h", "-help", "--help"):
        assert run(capsys, "-n0", "o.root", "in.root", flag) == (
            0,
            "",
            hadd_command.USAGE + hadd_command.HELP,
        )


def test_hadd_takes_a_lone_dash_as_a_name_and_skips_empty_words(capsys, made):
    assert run(capsys, "-f", "", "-", "in.root")[1].endswith("hadd Target path: -:/\n")


def test_a_url_input_is_taken_as_readable(made):
    hadd = hadd_command._Hadd(hadd_command.parse(["o.root", "in.root"]), None)
    assert hadd._valid("root://host//f.root", "o.root", "never") == ["root://host//f.root"]


def test_an_input_that_stops_the_merge_leaves_the_target_made_and_empty_as_root_does(
    capsys, made
):
    made.joinpath("junk.txt").write_text("")
    status, _, err = run(capsys, "-f", "o.root", "junk.txt", "in.root")
    assert status == 1
    assert err.endswith("failure during the merge of 2 (0) input (partial) files into o.root\n")
    with open_root("o.root") as f:
        assert f.keys() == [] and f.compression == 101
    run(capsys, "-f", "kept.root", "in.root")
    status, _, _ = run(capsys, "-a", "kept.root", "in.root", "junk.txt")
    with open_root("kept.root") as f:
        assert status == 1 and f.keys() == ["h", "t"]


def test_a_tntuple_merged_stays_a_tntuple(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with create("n.root") as out:
        tree = out.tree("nt", {"a": "f", "b": "f"}, classname="TNtuple")
        tree.extend({"a": np.arange(3, dtype=np.float32), "b": np.ones(3, dtype=np.float32)})
    assert run(capsys, "-f", "m.root", "n.root", "n.root")[0] == 0
    with open_root("m.root") as f:
        assert f.classnames()["nt"] == "TNtuple" and f["nt"].num_entries == 6
