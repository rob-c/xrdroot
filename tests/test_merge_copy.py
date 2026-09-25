"""Copying objects between files, as ``rootcp`` does - and trees cut on the way.

A copy is the record it was wherever it can be: the same bytes behind a new
key, the classes described in the new file as the old one described them,
so that an object of a class this library has no layout for reads back
exactly as it read before. A tree goes across as its baskets, and with a cut
or computed columns as ``TTree::CopyTree`` would write it.
"""

from __future__ import annotations

import io
import pathlib

import numpy as np
import pytest

from xrdroot import UnsupportedFeatureError, copy, create, open_root
from xrdroot import writer as writing
from xrdroot.file import Key
from xrdroot.merging.copying import _outermost
from xrdroot.merging.records import copy_record

DATA = pathlib.Path(__file__).parent / "data"

#: Carrying over what is not merged is said in a warning, which the tests that
#: are about it ask for by name; everywhere else it is only noise.
pytestmark = pytest.mark.filterwarnings("ignore::xrdroot.MergeWarning")


def test_everything_in_a_file_is_copied_by_default_directories_and_all(tmp_path):
    out = tmp_path / "out.root"
    written = copy(DATA / "dirs-6.14.00.root", out)
    assert written == ["dir1", "dir1/dir11", "dir1/dir11/h1", "dir2", "dir3"]
    with open_root(str(out)) as f, open_root(str(DATA / "dirs-6.14.00.root")) as g:
        assert f.classnames() == g.classnames()
        assert f["dir1/dir11/h1"].values().tolist() == g["dir1/dir11/h1"].values().tolist()


def test_an_object_of_a_class_this_library_does_not_write_reads_back_as_it_was(tmp_path):
    out = tmp_path / "out.root"
    copy(DATA / "tformula.root", out, ["fconv", "fnorm", "func*"])
    with open_root(str(out)) as f, open_root(str(DATA / "tformula.root")) as g:
        for name in g.keys():
            assert repr(f[name]) == repr(g[name]), name
        assert "TF1Convolution" in f._source.streamers()


def test_a_copied_record_is_the_very_bytes_it_was(tmp_path):
    out = tmp_path / "out.root"
    copy(DATA / "tgme.root", out, "gme")
    with open_root(str(out)) as f, open_root(str(DATA / "tgme.root")) as g:
        new, old = f._key("gme"), g._key("gme")
        assert f._source.read(new.seek_key + new.keylen, new.nbytes - new.keylen) == (
            g._source.read(old.seek_key + old.keylen, old.nbytes - old.keylen)
        )
        assert f["gme"].y.tolist() == g["gme"].y.tolist()


def test_a_path_can_be_given_a_new_one_and_a_name_that_is_there_gets_a_new_cycle(tmp_path):
    out = tmp_path / "out.root"
    copy(DATA / "gauss-h1.root", out, {"h1d": "copies/h1d", "h1f": "h1f"})
    copy(DATA / "gauss-h1.root", out, "h1f")
    with open_root(str(out)) as f:
        assert f.keys() == ["copies", "h1f"]
        assert sorted(key.cycle for key in f._keys if key.name == "h1f") == [1, 2]
        assert f["copies"].keys() == ["h1d"]


def test_an_object_renamed_to_a_name_of_another_length_is_written_again(tmp_path):
    out = tmp_path / "out.root"
    copy(DATA / "gauss-h1.root", out, {"h1d": "a_longer_name"})
    with open_root(str(out)) as f, open_root(str(DATA / "gauss-h1.root")) as g:
        assert f["a_longer_name"].values().tolist() == g["h1d"].values().tolist()


def test_an_object_that_cannot_be_written_again_is_not_renamed_to_another_length(tmp_path):
    refusal = "not a class this writer carries a layout for to write it again"
    with pytest.raises(UnsupportedFeatureError, match=refusal):
        copy(DATA / "tformula.root", tmp_path / "out.root", {"fconv": "longer"})


def test_a_record_whose_key_would_grow_is_refused_rather_than_misplaced(tmp_path, monkeypatch):
    with open_root(str(DATA / "tformula.root")) as f, create(io.BytesIO()) as out:
        key = f._key("fconv")
        monkeypatch.setattr(writing, "BIG", 0)
        with pytest.raises(UnsupportedFeatureError, match=r"would need a longer key here"):
            copy_record(out, "fconv", key, f._source, {})
        assert isinstance(key, Key)


def test_a_pattern_that_matches_nothing_is_refused_naming_what_there_is(tmp_path):
    with pytest.raises(KeyError, match=r"'nothing\*' is nothing in .*there is h1d, h1f"):
        copy(DATA / "gauss-h1.root", tmp_path / "out.root", "nothing*")


def test_a_directory_matched_with_what_is_in_it_is_copied_once():
    assert _outermost(["a", "a/b", "c", "a/b/d", "c"]) == ["a", "c"]
    assert _outermost(["ab", "a/x"]) == ["ab", "a/x"]


def test_a_path_through_something_that_is_not_a_directory_is_refused(tmp_path):
    with pytest.raises(KeyError, match=r"'h1d' in .* is not a directory"):
        copy(DATA / "gauss-h1.root", tmp_path / "out.root", {"h1d/x": "x"})


def test_a_copy_goes_into_a_directory_of_a_file_being_written(tmp_path):
    out = tmp_path / "out.root"
    with create(str(out)) as written:
        place = written.mkdir("from/elsewhere")
        assert copy(DATA / "gauss-h1.root", place, "h1f") == ["h1f"]
        assert not written.closed  # left for its owner to close
    with open_root(str(out)) as f:
        assert f["from/elsewhere"].keys() == ["h1f"]


def test_a_new_file_is_compressed_as_the_source_unless_told(tmp_path):
    copy(DATA / "gauss-h1.root", tmp_path / "a.root")
    copy(DATA / "gauss-h1.root", tmp_path / "b.root", compression=404)
    with open_root(str(tmp_path / "a.root")) as a, open_root(str(tmp_path / "b.root")) as b:
        assert a.compression == 101 and b.compression == 404


def test_a_copy_that_goes_wrong_leaves_no_file_behind(tmp_path):
    out = tmp_path / "out.root"
    with pytest.raises(UnsupportedFeatureError):
        copy(DATA / "tformula.root", out, {"func1": "f", "fconv": "longer"})
    assert not out.read_bytes()


def test_a_source_already_open_is_used_and_left_open(tmp_path):
    with open_root(str(DATA / "gauss-h1.root")) as f:
        copy(f, tmp_path / "out.root", "h1d")
        assert f["h1d"].entries > 0


def test_trees_are_copied_as_their_baskets_unless_a_cut_says_otherwise(tmp_path):
    out = tmp_path / "out.root"
    copy(DATA / "chain.flat.1.root", out, "tree", columns=["SliI32"], fast=True)
    with open_root(str(out)) as f, open_root(str(DATA / "chain.flat.1.root")) as g:
        assert f["tree"].keys() == ["N", "SliI32"]  # the counter comes with its run
        assert f["tree"]["SliI32"].array() == g["tree"]["SliI32"].array()


def test_a_tree_filter_says_which_trees_the_cut_is_for(tmp_path):
    out = tmp_path / "out.root"
    source = tmp_path / "two.root"
    with create(str(source)) as written:
        written["kept"] = {"x": np.arange(5.0)}
        written["cut"] = {"x": np.arange(5.0)}
    copy(source, out, tree_filter=lambda path: path == "cut", cut="x > 2", columns="x")
    with open_root(str(out)) as f:
        assert f["kept"].num_entries == 5
        assert f["cut"]["x"].array().tolist() == [3.0, 4.0]


def test_an_rntuple_is_copied_entry_by_entry(tmp_path):
    source = tmp_path / "nt.root"
    with create(str(source)) as written:
        written.write("nt", {"pt": np.arange(4, dtype=np.float32)}, rntuple=True)
    copy(source, tmp_path / "out.root")
    with open_root(str(tmp_path / "out.root")) as f:
        assert f["nt"]["pt"].array().tolist() == [0.0, 1.0, 2.0, 3.0]
