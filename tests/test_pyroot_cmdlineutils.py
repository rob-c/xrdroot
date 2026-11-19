"""``cmdLineUtils``: the pattern and stream helpers ROOT's command-line tools are written over.

The patterns are ROOT's own: ``file.root``, ``file.root:obj``, ``file.root:dir/h*``,
each part a shell pattern matched against the keys at that depth. What is
asserted here is the shape the tools rely on - a list of ``(fileName,
pathSplitList)``, the paths split at ``/`` and taken in the order the keys were
written - and that a pattern naming nothing is warned about rather than silently
matching everything.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pytest

from xrdroot import Histogram, create
from xrdroot.pyroot import cmdlineutils
from xrdroot.pyroot.cmdlineutils import (
    fileNameListMatch,
    pathSplitListMatch,
    patternToFileNameAndPathSplitList,
    patternToPathSplitList,
    stderrRedirected,
    stdoutRedirected,
    streamRedirected,
)


def histogram(name: str) -> Histogram:
    return Histogram.new(name, np.arange(4.0), np.asarray([1.0, 2.0, 3.0]))


@pytest.fixture
def sample(tmp_path, monkeypatch):
    """A file of two histograms and two directories, worked in from the directory holding it."""
    with create(str(tmp_path / "f.root")) as out:
        out["h1"] = histogram("h1")
        out["h2"] = histogram("h2")
        out["dir/a"] = histogram("a")
        out["other/b"] = histogram("b")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def warned(caplog):
    caplog.set_level(logging.WARNING, logger="cmdLineUtils")
    return caplog


# -- the module's name -----------------------------------------------------------------


def test_the_module_is_importable_under_the_name_roots_tools_use():
    assert sys.modules["cmdLineUtils"] is cmdlineutils


def test_a_star_import_from_it_gives_every_name_the_tools_call():
    public = {name for name in dir(cmdlineutils) if not name.startswith("_")}
    assert set(cmdlineutils.__all__) <= public


def test_a_helper_root_has_and_this_does_not_says_where_the_work_is_done():
    with pytest.raises(AttributeError) as missing:
        cmdlineutils.copyRootObject  # noqa: B018 - looked up on purpose
    assert "cmdLineUtils.copyRootObject is not here" in str(missing.value)
    assert "xrdroot ls, cp, mv, rm or print" in str(missing.value)


def test_a_dunder_is_a_missing_attribute_as_python_expects():
    with pytest.raises(AttributeError):
        cmdlineutils.__no_such_dunder__  # type: ignore[attr-defined]  # noqa: B018


# -- patterns naming a file ------------------------------------------------------------


def test_a_pattern_naming_a_file_alone_is_that_file_and_nothing_in_it(sample):
    assert patternToFileNameAndPathSplitList("f.root") == [("f.root", [[]])]


def test_a_file_pattern_names_every_file_it_matches(sample):
    with create(str(sample / "g.root")) as out:
        out["h1"] = histogram("h1")
    named = patternToFileNameAndPathSplitList("*.root")
    assert sorted(name for name, _ in named) == ["f.root", "g.root"]
    assert all(paths == [[]] for _, paths in named)


def test_a_pcm_pattern_is_read_as_a_file_of_objects_too(tmp_path):
    assert patternToFileNameAndPathSplitList("x.pcm", wildcards=False) == [("x.pcm", [[]])]


def test_a_pattern_that_does_not_end_like_a_root_file_names_nothing(warned):
    assert patternToFileNameAndPathSplitList("notes.txt") == []
    assert warned.messages == ["notes.txt: No such file (or extension not supported)"]


# -- patterns naming objects in a file -------------------------------------------------


def test_an_object_pattern_names_the_keys_matching_it_in_the_order_they_were_written(sample):
    assert patternToFileNameAndPathSplitList("f.root:h*") == [("f.root", [["h1"], ["h2"]])]


def test_a_pattern_walks_into_every_directory_a_part_matches(sample):
    assert patternToFileNameAndPathSplitList("f.root:*/*") == [
        ("f.root", [["dir", "a"], ["other", "b"]])
    ]


def test_a_path_naming_a_histogram_is_not_walked_into(sample, warned):
    assert patternToFileNameAndPathSplitList("f.root:h1/*") == [("f.root", [])]
    assert warned.messages == ["can't find h1/* in f.root"]


def test_a_part_matching_nothing_ends_the_walk_there(sample, warned):
    assert patternToFileNameAndPathSplitList("f.root:nothing*") == [("f.root", [])]
    assert warned.messages == ["can't find nothing* in f.root"]


def test_without_wildcards_the_pattern_is_the_answer_and_no_file_is_opened(tmp_path):
    assert patternToFileNameAndPathSplitList("gone.root:tof/hpx", wildcards=False) == [
        ("gone.root", [["tof", "hpx"]])
    ]
    assert pathSplitListMatch("gone.root", "/tof//hpx/", wildcards=False) == [["tof", "hpx"]]
    assert fileNameListMatch("gone.root", wildcards=False) == ["gone.root"]


def test_a_file_pattern_matching_no_file_names_nothing_and_says_nothing(tmp_path, warned):
    assert patternToFileNameAndPathSplitList("nope.root:*") == []
    assert warned.messages == []


def test_a_file_that_will_not_open_is_warned_about_and_names_nothing(tmp_path, warned):
    assert patternToPathSplitList(str(tmp_path / "nope.root"), "*") == []
    (tmp_path / "notroot.root").write_text("this is not a ROOT file")
    assert patternToPathSplitList(str(tmp_path / "notroot.root"), "*") == []
    missing, truncated = warned.messages
    assert missing.startswith("[Errno 2] No such file or directory: ")
    assert missing.endswith("nope.root'")
    assert truncated.endswith(
        "notroot.root ends at 23, where 100 bytes were expected from 0: the file is truncated"
    )


# -- URLs ------------------------------------------------------------------------------


def test_a_url_is_the_one_file_it_names_and_is_never_globbed():
    assert patternToFileNameAndPathSplitList("root://server//f.root") == [
        ("root://server//f.root", [[]])
    ]


def test_a_url_with_an_object_pattern_names_a_path_in_that_one_file():
    assert patternToFileNameAndPathSplitList("root://server//f.root:dir/h*", wildcards=False) == [
        ("root://server//f.root", [["dir", "h*"]])
    ]


# -- the streams the tools hide --------------------------------------------------------


def test_what_a_block_writes_goes_where_the_block_sends_it(tmp_path):
    kept, hidden = tmp_path / "kept.txt", tmp_path / "hidden.txt"
    with open(kept, "wb") as source:
        with streamRedirected(source, str(hidden)) as inside:
            assert inside is source
            source.write(b"during\n")
        source.write(b"after\n")
    assert hidden.read_bytes() == b"during\n"
    assert kept.read_bytes() == b"after\n"


def test_the_shortcuts_hide_standard_output_and_standard_error():
    with stdoutRedirected() as out:
        assert out is sys.stdout
        print("hidden from the terminal")
    with stderrRedirected() as err:
        assert err is sys.stderr
        sys.stderr.write("hidden from the terminal\n")
