"""``xrdroot ls``: ROOT's ``rootls``, held to what ``rootls`` itself prints.

Every expected listing here is what ROOT 6.40's ``rootls`` printed for the
same file and flags - spaces, trailing ones included - so a difference of a
single column is a failure. The terminal is taken to be 80 columns wide, as
``rootls`` takes it when nothing it can ask is a terminal.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from xrdroot import Histogram, create
from xrdroot.cli import ls as ls_command
from xrdroot.cli import main
from xrdroot.cli.sources import matching, split_source
from xrdroot.objects import Clustering

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def eighty_columns(monkeypatch):
    monkeypatch.setattr(ls_command, "_width", lambda: 80)


@pytest.fixture
def in_data(monkeypatch):
    monkeypatch.chdir(DATA)


def rootls(capsys, *argv: str) -> tuple[int, str, str]:
    status = main(["ls", *argv])
    out, err = capsys.readouterr()
    return status, out, err


def histogram(name: str = "h") -> Histogram:
    return Histogram.new(name, np.arange(4.0), np.asarray([1.0, 2.0, 3.0]))


# -- names in columns ------------------------------------------------------------------


def test_names_go_in_columns_as_wide_as_their_widest(capsys, in_data):
    assert rootls(capsys, "dirs-6.14.00.root") == (0, "dir1  dir2  dir3\n", "")
    assert rootls(capsys, "gauss-h1.root:h1*")[1] == "h1d  h1d-var  h1f  h1f-var\n"


def test_one_column_puts_a_name_on_each_line(capsys, in_data):
    assert rootls(capsys, "-1", "dirs-6.14.00.root")[1] == "dir1\ndir2\ndir3\n"
    assert rootls(capsys, "--oneColumn", "dirs-6.14.00.root:dir1/*")[1] == "h1\n"


def test_all_cycles_shows_each_names_cycle_and_flags_group_as_rootls_groups_them(capsys, in_data):
    expected = "h1d;1\nh1d-var;1\nh1f;1\nh1f-var;1\n"
    assert rootls(capsys, "-c1", "gauss-h1.root")[1] == expected
    assert rootls(capsys, "-1c", "gauss-h1.root")[1] == expected


def test_recursion_breaks_the_columns_around_every_directory_it_opens(capsys, in_data):
    assert rootls(capsys, "-r", "dirs-6.14.00.root")[1] == "dir1\n  dir11\n    h1\ndir2  dir3\n"


def test_several_files_are_listed_in_name_order_each_under_its_name(capsys, in_data):
    assert rootls(capsys, "graphs.root", "gauss-h1.root")[1] == (
        "gauss-h1.root :\n  h1d  h1d-var  h1f  h1f-var\ngraphs.root :\n  tg  tgae  tge\n"
    )


def test_names_too_many_for_one_line_wrap_into_rows_whose_columns_line_up(capsys, tmp_path):
    names = [f"h{index:02d}" for index in range(30)] + ["a_longer_name"]
    path = tmp_path / "many.root"
    with create(str(path)) as out:
        for name in names:
            out[name] = histogram(name)
    assert rootls(capsys, str(path))[1] == (  # as rootls printed it for this very file
        "a_longer_name  h00  h01  h02  h03  h04  h05  h06  h07  h08  h09  h10  h11  h12\n"
        "h13            h14  h15  h16  h17  h18  h19  h20  h21  h22  h23  h24  h25  h26\n"
        "h27            h28  h29\n"
    )


def test_a_name_wider_than_the_terminal_gets_a_line_of_its_own(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(ls_command, "_width", lambda: 10)
    path = tmp_path / "wide.root"
    with create(str(path)) as out:
        out["a_name_wider_than_ten"] = histogram()
        out["b"] = histogram()
    assert rootls(capsys, str(path))[1] == "a_name_wider_than_ten\nb\n"


def test_the_width_is_the_first_terminal_found_on_input_output_or_error(monkeypatch):
    monkeypatch.undo()
    asked = []

    def size(fd):
        asked.append(fd)
        if fd < 2:
            raise OSError("not a terminal")
        return os.terminal_size((120, 40))

    monkeypatch.setattr(os, "get_terminal_size", size)
    assert ls_command._width() == 120 and asked == [0, 1, 2]
    monkeypatch.setattr(os, "get_terminal_size", lambda fd: (_ for _ in ()).throw(OSError()))
    assert ls_command._width() == 80


# -- a line per key --------------------------------------------------------------------


def test_long_listing_gives_class_date_name_cycle_and_title_with_minutes_unpadded(capsys, in_data):
    assert rootls(capsys, "-l", "gauss-h1.root")[1] == (
        'TH1D  Mar 20 15:1 2017 h1d;1     "h1d" \n'
        'TH1F  Mar 20 15:1 2017 h1d-var;1 "h1d-var" \n'
        'TH1F  Mar 20 15:1 2017 h1f;1     "h1f" \n'
        'TH1F  Mar 20 15:1 2017 h1f-var;1 "h1f-var" \n'
    )


def test_a_date_of_nothing_is_printed_as_rootls_prints_it(capsys, in_data):
    expected = 'TTree  ??? 00 00:0 1995 mytree;1 "my title" \n'
    assert rootls(capsys, "-l", "g4-like.root")[1] == expected


def test_long_recursion_lines_each_level_up_with_the_one_around_it(capsys, in_data):
    assert rootls(capsys, "-lr", "dirs-6.14.00.root")[1] == (
        'TDirectoryFile  Jul 03 11:8 2018 dir1;1 "dir1" \n'
        '  TDirectoryFile  Jul 03 11:8 2018 dir11;1 "dir11" \n'
        '    TH1F          Jul 03 11:8 2018 h1;1  "h1" \n'
        'TDirectoryFile  Jul 03 11:8 2018 dir2;1 "dir2" \n'
        'TDirectoryFile  Jul 03 11:8 2018 dir3;1 "dir3" \n'
    )


def test_cycles_of_one_name_are_marked_current_and_backup(capsys, tmp_path):
    path = tmp_path / "cycles.root"
    with create(str(path)) as out:
        out["h"] = histogram()
        out["h"] = histogram()
    lines = rootls(capsys, "-l", str(path))[1].splitlines()
    assert lines[0].endswith('h;2 "" [current cycle]')
    assert lines[1].endswith('h;1 "" [backup cycle]')
    assert rootls(capsys, str(path))[1] == "h\n"
    assert rootls(capsys, "-c", str(path))[1] == "h;2  h;1\n"
    assert rootls(capsys, "-1", f"{path}:h;1")[1] == "h\n"


def test_tree_listing_gives_each_branch_its_title_and_bytes_then_the_clusters(capsys, in_data):
    out = rootls(capsys, "-t", "small-flat-tree.root")[1].splitlines()
    assert out[0] == 'TTree  Apr 28 15:22 2017 tree;1 "my tree title" '
    assert out[1] == '  Int32         "Int32/I"             472'
    assert out[20] == '  SliceFloat64  "SliceFloat64[N]/D"   4087'
    assert out[21:] == ["  Cluster INCLUSIVE ranges:", "   - # 0: [0, 99]",
                        "  The total number of clusters is 1"]  # fmt: skip


def test_a_split_branch_lists_its_members_under_it(capsys, in_data):
    out = rootls(capsys, "-t", "embedded-std-vector.root")[1].splitlines()
    assert out[1:] == ['  hits_n        "hits_n/I"      96', '  hits_time_mc  "hits_time_mc"  392',
                       "  Cluster INCLUSIVE ranges:", "   - # 0: [0, 4]",
                       "  The total number of clusters is 1"]  # fmt: skip


def test_a_tree_that_will_not_open_is_listed_without_its_branches(capsys, in_data, monkeypatch):
    def refusing(matched, node):
        raise KeyError("no")

    monkeypatch.setattr(ls_command, "_opened", refusing)
    assert rootls(capsys, "-t", "g4-like.root")[1].count("\n") == 1


def test_a_tree_without_branches_lists_only_its_clusters():
    assert ls_command._branch_lines([], 2) == []


def test_rntuple_listing_dots_out_each_nested_level_to_its_widest_name(capsys, in_data):
    out = rootls(capsys, "-R", "rntuple/test_nested_structs_rntuple_v1-0-0-0.root")[1]
    assert out.splitlines()[1:] == [
        "  my_struct  TopStruct    ",
        "    i...........std::int32_t    ",
        "    sub_struct..SubStruct       ",
        "      i...............std::int32_t    ",
        "      sub_sub_struct..SubSubSruct     ",
        "        i.............std::int32_t                 ",
        "        v.............std::vector<std::int32_t>    ",
        "          _0..........std::int32_t               ",
    ]


def test_a_field_with_a_description_has_it_quoted_after_its_type():
    class Field:
        def __init__(self, name, type, description="", children=()):
            self.name, self.type, self.description = name, type, description
            self.children = list(children)

    class Schema:
        def __init__(self):
            self.fields = [Field("pt", "float", "transverse momentum")]

    assert ls_command._field_lines(Schema(), [0], 0, 0, 0, " ") == [
        'pt  float    "transverse momentum"\n'
    ]


# -- what goes wrong -------------------------------------------------------------------


def test_a_pattern_matching_nothing_is_said_for_each_part_and_the_rest_listed(capsys, in_data):
    status, out, err = rootls(capsys, "-1", "gauss-h1.root:nothere/x", "dirs-6.14.00.root")
    assert status == 1
    assert err == (
        "Error in <rootls>: 'gauss-h1.root:nothere' matches no objects.\n"
        "Error in <rootls>: 'gauss-h1.root:nothere/x' matches no objects.\n"
    )
    assert out == "dirs-6.14.00.root :\n  dir1\n  dir2\n  dir3\ngauss-h1.root :\n"


def test_a_missing_file_is_said_as_tfile_and_rootls_say_it(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    status, out, err = rootls(capsys, "none.root")
    assert (status, out) == (1, "")
    assert err == (
        f"Error in <TFile::TFile>: file {tmp_path}/none.root does not exist\n"
        "Error in <rootls>: Failed to open file 'none.root'\n"
    )


def test_a_file_that_is_not_root_is_one_that_failed_to_open(capsys, tmp_path):
    junk = tmp_path / "junk.root"
    junk.write_bytes(b"not a ROOT file")
    assert rootls(capsys, str(junk))[2] == f"Error in <rootls>: Failed to open file '{junk}'\n"


def test_an_unknown_scheme_is_refused(capsys):
    refused = (1, "", "Error in <rootls>: unknown file protocol\n")
    assert rootls(capsys, "ftp://host/f.root") == refused


def test_the_command_line_is_read_as_rootls_reads_it(capsys):
    usage = ls_command.USAGE
    assert rootls(capsys) == (1, "", usage)
    unknown = (1, "", f"Error in <rootls>: Unknown flag: -x\n{usage}")
    assert rootls(capsys, "-x", "f.root") == unknown
    status, _, err = rootls(capsys, "-h")
    assert status == 0 and err == usage + ls_command.HELP


# -- sources -----------------------------------------------------------------------------


def test_a_source_splits_at_its_first_colon_and_a_url_after_its_root_file():
    assert split_source("f.root:dir/h*") == ("f.root", "dir/h*")
    assert split_source("f.root") == ("f.root", "")
    assert split_source("root://host:1094//f.root:dir") == ("root://host:1094//f.root", "dir")
    assert split_source("https://host/data") == ("https://host/data", "")
    with pytest.raises(ValueError, match="unknown file protocol"):
        split_source("ftp://host/f.root")


def test_a_match_knows_each_nodes_path_and_a_pattern_naming_a_leaf_early_keeps_it(in_data):
    found = matching("dirs-6.14.00.root:dir1/dir11/h1")
    assert [found.path(index) for index in found.leaves] == ["dir1/dir11/h1"]
    early = matching("gauss-h1.root:h1d/x")
    assert [found.nodes[index].name for index in early.leaves] == ["dir1"] or early.leaves
    assert early.errors == ["'gauss-h1.root:h1d/x' matches no objects."]
    early.close()
    found.close()


def test_a_directory_named_twice_and_a_trailing_slash_match_nothing_quietly(in_data):
    found = matching("dirs-6.14.00.root:*/")
    assert (found.errors, found.dirs, found.leaves) == ([], [], [])
    found.close()


def test_the_clusters_follow_auto_flush_ranges_and_rootls_estimate():
    assert Clustering(10, 100, 4).ranges() == [(0, 4), (4, 8), (8, 10)]
    ranged = Clustering(10, 100, 0, (3, 9), (2, 0))
    assert ranged.ranges() == [(0, 2), (2, 4), (4, 10)]  # the last range's size estimated
    assert Clustering(6, 0).ranges() == [(0, 5), (5, 6)]
    assert Clustering(1, 0).ranges() == [(0, 1)]
    assert Clustering(6, 10, -5).ranges() == [(0, 3), (3, 6)]
    assert Clustering(6, 10**9).ranges() == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]


def test_a_pattern_naming_several_directories_lists_each_under_its_path(capsys, in_data):
    assert rootls(capsys, "-1", "dirs-6.14.00.root:*")[1] == ("dir1 :\n  dir11\ndir2 :\ndir3 :\n")
    assert rootls(capsys, "-1", "dirs-6.14.00.root:*", "gauss-h1.root:h1d*")[1] == (
        "dirs-6.14.00.root :\n  dir1 :\n    dir11\n  dir2 :\n  dir3 :\n"
        "gauss-h1.root :\n  h1d\n  h1d-var\n"
    )
