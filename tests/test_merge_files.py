"""Merging whole files, as ``hadd`` does: every object, in every directory.

Each kind of object is merged the way ROOT's own ``Merge`` for it merges:
histograms and profiles add up bin by bin, sums and entries alike;
efficiencies add up what passed and what was tried; graphs and multigraphs
gather their points and their graphs. What ROOT has no ``Merge`` for is
carried over from every input, a cycle each, said once in a warning. Names
only some files hold are taken up, directories are walked all the way down,
and what is asked to be left out is. Every refusal names the object and the
file.
"""

from __future__ import annotations

import io
import pathlib
import shutil
import warnings

import numpy as np
import pytest

from xrdroot import (
    Efficiency,
    FormatError,
    Graph,
    Histogram,
    MergeWarning,
    MultiGraph,
    create,
    merge,
    open_root,
)
from xrdroot.merging.files import codes_of, resolve_compression

DATA = pathlib.Path(__file__).parent / "data"

#: Carrying over what is not merged is said in a warning, which the tests that
#: are about it ask for by name; everywhere else it is only noise.
pytestmark = pytest.mark.filterwarnings("ignore::xrdroot.MergeWarning")


def merged(tmp_path, inputs, **options):
    out = tmp_path / "out.root"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MergeWarning)
        result = merge(out, inputs, force=True, **options)
    return open_root(str(out)), result


def histogram_file(path, name="h", counts=(1.0, 2.0, 3.0), edges=(0.0, 1.0, 2.0, 3.0), **extra):
    with create(str(path)) as out:
        out[name] = Histogram.new(name, np.asarray(edges), np.asarray(counts))
        for key, value in extra.items():
            out[key] = value
    return path


# -- what is added up -----------------------------------------------------------


def test_histograms_add_up_bins_sums_and_entries_as_th1_merge_does(tmp_path):
    source = DATA / "gauss-h1.root"
    with open_root(str(source)) as f:
        before = {name: f[name] for name in f.keys()}
    with merged(tmp_path, [source, source, source])[0] as f:
        for name, histogram in before.items():
            again = f[name]
            assert again.classname == histogram.classname
            assert np.allclose(again.values(flow=True), 3 * histogram.values(flow=True))
            assert np.allclose(again.variances(flow=True), 3 * histogram.variances(flow=True))
            assert again.entries == 3 * histogram.entries
            assert again.mean() == pytest.approx(histogram.mean())


def test_profiles_add_up_what_went_into_each_bin(tmp_path):
    source = DATA / "tprofile.root"
    with open_root(str(source)) as f:
        profile = f["p1d"]
    with merged(tmp_path, [source, source])[0] as f:
        again = f["p1d"]
        assert np.allclose(again.bin_entries(True), 2 * profile.bin_entries(True))
        assert np.allclose(again.values(), profile.values())
        assert again.entries == 2 * profile.entries


def test_efficiencies_add_up_what_passed_and_what_was_tried(tmp_path):
    source = DATA / "tefficiency.root"
    with open_root(str(source)) as f:
        before = f["eff1"]
    with merged(tmp_path, [source, source])[0] as f:
        again = f["eff1"]
        assert isinstance(again, Efficiency)
        assert np.allclose(again.passed.values(), 2 * before.passed.values())
        assert np.allclose(again.total.values(), 2 * before.total.values())
        assert np.allclose(again.values(), before.values(), equal_nan=True)


def test_graphs_gather_their_points_and_their_bars_as_tgraph_merge_does(tmp_path):
    source = DATA / "graphs.root"
    with open_root(str(source)) as f:
        before = {name: f[name] for name in f.keys()}
    with merged(tmp_path, [source, source])[0] as f:
        for name, graph in before.items():
            again = f[name]
            assert again.classname == graph.classname
            assert again.x.tolist() == [*graph.x, *graph.x]
            assert again.y.tolist() == [*graph.y, *graph.y]
            for bars in ("xerr", "yerr"):
                if getattr(graph, bars) is not None:
                    low, high = getattr(again, bars)
                    assert low.tolist() == 2 * getattr(graph, bars)[0].tolist()
                    assert high.tolist() == 2 * getattr(graph, bars)[1].tolist()


def test_graphs_of_different_classes_are_refused_as_hadd_refuses_them(tmp_path):
    plain, even = tmp_path / "plain.root", tmp_path / "even.root"
    with create(str(plain)) as out:
        out["g"] = Graph.new("g", [1.0], [2.0])
    with create(str(even)) as out:
        out["g"] = Graph.new("g", [3.0], [4.0], xerr=[0.5], yerr=[0.25])
    with pytest.raises(ValueError, match=r"'g' is a TGraph in .*plain.root and a TGraphErrors"):
        merged(tmp_path, [even, plain])


def test_multigraphs_gather_their_graphs_as_tmultigraph_merge_does(tmp_path):
    source = DATA / "tgme.root"
    with open_root(str(source)) as f:
        before = f["mg"]
    f, result = merged(tmp_path, [source, source])
    with f:
        again = f["mg"]
        assert isinstance(again, MultiGraph)
        assert [graph.classname for graph in again] == 2 * [graph.classname for graph in before]
        assert [graph.x.tolist() for graph in again] == 2 * [graph.x.tolist() for graph in before]
        assert [fit.name for fit in again.functions] == ["pol1"]
    assert result.objects["mg"] == "merged"


# -- what is carried over -------------------------------------------------------


def test_what_hadd_does_not_merge_is_carried_over_from_every_input_a_cycle_each(tmp_path):
    source = DATA / "tformula.root"
    out = tmp_path / "out.root"
    with pytest.warns(MergeWarning, match=r"'func1' is a TF1, which is not merged"):
        result = merge(out, [source, source])
    assert result.objects["func1"] == "copied"
    with open_root(str(out)) as f, open_root(str(source)) as g:
        assert sorted(key.cycle for key in f._keys if key.name == "func1") == [1, 2]
        assert f["func1;1"].parameters.tolist() == g["func1"].parameters.tolist()
        assert repr(f["fconv"]) == repr(g["fconv"])  # a class this library has no layout for


def test_strings_are_carried_over_as_they_were(tmp_path):
    first = histogram_file(tmp_path / "a.root", note="first")
    second = histogram_file(tmp_path / "b.root", note="second")
    with merged(tmp_path, [first, second])[0] as f:
        assert f["note;1"] == "first" and f["note;2"] == "second"
        assert f["h"].values().tolist() == [2.0, 4.0, 6.0]


def test_every_cycle_of_an_object_that_is_not_merged_is_carried(tmp_path):
    source = tmp_path / "cycles.root"
    with create(str(source)) as out:
        out["note"] = "old"
        out["note"] = "new"
    with merged(tmp_path, [source])[0] as f:
        assert [f["note;1"], f["note;2"]] == ["old", "new"]


# -- directories, and names only some inputs hold --------------------------------


def test_directories_are_walked_all_the_way_down_and_kept_when_empty(tmp_path):
    source = DATA / "dirs-6.14.00.root"
    with merged(tmp_path, [source, source])[0] as f, open_root(str(source)) as g:
        assert f.classnames() == g.classnames()
        assert f["dir2"].keys() == [] and f["dir3"].keys() == []
        once = g["dir1/dir11/h1"]
        assert np.allclose(f["dir1/dir11/h1"].values(), 2 * once.values())


def test_a_name_only_a_later_input_holds_is_taken_up_too(tmp_path):
    first = histogram_file(tmp_path / "a.root")
    second = histogram_file(tmp_path / "b.root", name="later")
    with merged(tmp_path, [first, second, first])[0] as f:
        assert f.keys() == ["h", "later"]
        assert f["h"].values().tolist() == [2.0, 4.0, 6.0]
        assert f["later"].values().tolist() == [1.0, 2.0, 3.0]


def test_skipped_names_and_patterns_are_left_out(tmp_path):
    source = DATA / "gauss-h1.root"
    with merged(tmp_path, [source], skip_keys=["h1d", "*-var"])[0] as f:
        assert f.keys() == ["h1f"]
    with merged(tmp_path, [DATA / "dirs-6.14.00.root"], skip_keys="dir1")[0] as f:
        assert f.keys() == ["dir2", "dir3"]


def test_only_the_names_listed_are_taken_and_a_listed_directory_is_taken_whole(tmp_path):
    with merged(tmp_path, [DATA / "gauss-h1.root"], only_keys=["h1f"])[0] as f:
        assert f.keys() == ["h1f"]
    with merged(tmp_path, [DATA / "dirs-6.14.00.root"], only_keys=["dir1"])[0] as f:
        assert f.keys() == ["dir1"] and f["dir1/dir11"].keys() == ["h1"]
    with merged(tmp_path, [DATA / "dirs-6.14.00.root"], only_keys=["h1"])[0] as f:
        assert f.keys() == ["dir1"] and f["dir1/dir11"].keys() == ["h1"]


def test_trees_can_be_left_out(tmp_path):
    source = tmp_path / "both.root"
    shutil.copy(DATA / "string-example.root", source)
    f, result = merged(tmp_path, [source], trees=False)
    with f:
        assert f.keys() == ["FileSummaryRecord"]
    assert "Refs" not in result.objects


def test_rntuples_hold_every_input_s_entries_one_after_another(tmp_path):
    source = DATA / "rntuple" / "test_int_vfloat_tlv_vtlv_rntuple_v1-0-0-0.root"
    written = tmp_path / "written.root"
    with create(str(written)) as out:
        out.write("ntuple", {"n": np.arange(3, dtype=np.int32), "name": ["a", "b", "c"]},
                  rntuple=True)
    with merged(tmp_path, [written, written])[0] as f:
        ntuple = f["ntuple"]
        assert ntuple["n"].array().tolist() == [0, 1, 2, 0, 1, 2]
        assert ntuple["name"].array() == ["a", "b", "c", "a", "b", "c"]
    del source


def test_rntuples_with_different_fields_are_refused_naming_the_fields(tmp_path):
    first, second = tmp_path / "a.root", tmp_path / "b.root"
    with create(str(first)) as out:
        out.write("nt", {"n": np.arange(2, dtype=np.int32)}, rntuple=True)
    with create(str(second)) as out:
        out.write("nt", {"n": np.arange(2.0)}, rntuple=True)
    with pytest.raises(ValueError, match=r"RNTuple 'nt' in .*b.root .*\(n differ\)"):
        merged(tmp_path, [first, second])


def test_an_rntuple_with_no_entries_is_still_an_rntuple(tmp_path):
    empty = tmp_path / "empty.root"
    with create(str(empty)) as out:
        out.write("nt", {"n": np.zeros(0, np.int32)}, rntuple=True)
    with merged(tmp_path, [empty])[0] as f:
        assert len(f["nt"]) == 0 and f["nt"].keys() == ["n"]


# -- refusals, and inputs that will not open -------------------------------------


def test_histograms_binned_differently_are_refused_naming_the_object_and_file(tmp_path):
    first = histogram_file(tmp_path / "a.root")
    other = histogram_file(tmp_path / "b.root", edges=(0.0, 1.0, 2.0, 4.0))
    with pytest.raises(ValueError, match=r"'h' in .*b.root cannot be merged .*binned differently"):
        merge(tmp_path / "out.root", [first, other])
    assert not (tmp_path / "out.root").read_bytes()  # nothing half-written left behind


def test_one_name_holding_different_classes_is_refused(tmp_path):
    first = histogram_file(tmp_path / "a.root")
    other = tmp_path / "b.root"
    with create(str(other)) as out:
        out["h"] = "a string, not a histogram"
    with pytest.raises(ValueError, match=r"'h' is a string in .*b.root and a TH1D in the files"):
        merge(tmp_path / "out.root", [first, other])
    folder = tmp_path / "c.root"
    with create(str(folder)) as out:
        out.mkdir("h")
    with pytest.raises(ValueError, match=r"'h' is a TDirectory in .*c.root and a TH1D"):
        merge(tmp_path / "again.root", [first, folder])


def test_an_output_already_there_is_refused_unless_forced(tmp_path):
    source = histogram_file(tmp_path / "a.root")
    out = tmp_path / "out.root"
    out.write_bytes(b"not yet a root file")
    with pytest.raises(FileExistsError, match=r"force=True\) - hadd -f"):
        merge(out, [source])
    merge(out, [source], force=True)
    with open_root(str(out)) as f:
        assert f.keys() == ["h"]


def test_the_output_is_never_one_of_the_inputs(tmp_path):
    source = histogram_file(tmp_path / "a.root")
    with pytest.raises(ValueError, match=r"is the output and one of the inputs"):
        merge(source, [source], force=True)


def test_a_merge_can_go_into_a_file_object_as_create_can(tmp_path):
    source = histogram_file(tmp_path / "a.root")
    target = io.BytesIO()
    merge(target, [source, source])
    with open_root(io.BytesIO(target.getvalue())) as f:
        assert f["h"].values().tolist() == [2.0, 4.0, 6.0]


def test_a_merge_needs_something_to_merge(tmp_path):
    with pytest.raises(ValueError, match=r"at least one input"):
        merge(tmp_path / "out.root", [])


def test_an_input_that_will_not_open_stops_the_merge_or_is_passed_over(tmp_path):
    good = histogram_file(tmp_path / "a.root")
    bad = tmp_path / "bad.root"
    bad.write_bytes(b"<html>not found</html>" * 10)
    with pytest.raises(FormatError):
        merge(tmp_path / "out.root", [good, bad])
    with pytest.warns(MergeWarning, match=r"bad.root is passed over"):
        result = merge(tmp_path / "out.root", [bad, good, tmp_path / "missing.root"],
                       skip_errors=True, force=True)
    assert result.skipped == [str(bad), str(tmp_path / "missing.root")]
    with open_root(str(tmp_path / "out.root")) as f:
        assert f["h"].values().tolist() == [1.0, 2.0, 3.0]


def test_inputs_that_all_fail_to_open_make_an_empty_output(tmp_path):
    bad = tmp_path / "bad.root"
    bad.write_bytes(b"nothing")
    with pytest.warns(MergeWarning):
        result = merge(tmp_path / "out.root", [bad], skip_errors=True)
    assert result.objects == {}
    with pytest.raises(OSError):
        merge(tmp_path / "again.root", [tmp_path / "missing.root"])


def test_files_already_open_are_merged_and_left_open(tmp_path):
    source = histogram_file(tmp_path / "a.root")
    with open_root(str(source)) as one:
        merge(tmp_path / "out.root", [one, one])
        assert one["h"].values().tolist() == [1.0, 2.0, 3.0]  # still open
    with open_root(str(tmp_path / "out.root")) as f:
        assert f["h"].values().tolist() == [2.0, 4.0, 6.0]


# -- appending, and compression --------------------------------------------------


def test_appending_merges_into_what_the_output_already_holds(tmp_path):
    out = histogram_file(tmp_path / "out.root", note="kept")
    more = histogram_file(tmp_path / "more.root", note="more")
    merge(out, [more], append=True)
    with open_root(str(out)) as f:
        assert f["h"].values().tolist() == [2.0, 4.0, 6.0]
        assert f["note;1"] == "kept" and f["note;2"] == "more"


def test_appending_to_a_file_not_there_yet_makes_it(tmp_path):
    more = histogram_file(tmp_path / "more.root")
    merge(tmp_path / "new.root", [more], append=True)
    with open_root(str(tmp_path / "new.root")) as f:
        assert f.keys() == ["h"]


def test_the_output_is_compressed_as_the_first_input_unless_told(tmp_path):
    source = DATA / "gauss-h1.root"
    with merged(tmp_path, [source])[0] as f:
        assert f.compression == 101
    with merged(tmp_path, [source], compression=505)[0] as f:
        assert f.compression == 505
    with merged(tmp_path, [source], compression="lzma")[0] as f:
        assert f.compression == 206
    with merged(tmp_path, [source], compression=("zlib", 9))[0] as f:
        assert f.compression == 109
    with merged(tmp_path, [source], compression=0)[0] as f:
        assert f.compression == 0


def test_every_way_of_spelling_a_compression_setting_is_read_as_root_reads_it():
    assert resolve_compression(None, 1) == ("zlib", 1)
    assert resolve_compression(404, 0) == ("lz4", 4)
    assert resolve_compression(7, 0) == ("zlib", 7)
    assert resolve_compression("zstd", 0) == ("zstd", None)
    assert codes_of("zstd", None) == 503 and codes_of("zlib", 1) == 101 and codes_of(None, 5) == 0
    for wrong in (True, 1.5, ["zlib"]):
        with pytest.raises(ValueError, match=r"compression is"):
            resolve_compression(wrong, 0)
    with pytest.raises(ValueError, match=r"compression must be one of"):
        resolve_compression("snappy", 0)
