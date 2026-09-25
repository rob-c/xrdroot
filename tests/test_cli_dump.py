"""``xrdroot dump``: every key's bins, points, entries and members, as go-hep prints them."""

from __future__ import annotations

import pathlib

import numpy as np

from xrdroot import Jagged
from xrdroot.cli import main
from xrdroot.cli.dump import text
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.file import Directory

DATA = pathlib.Path(__file__).parent / "data"


def dumped(capsys, name: str, *more: str) -> list[str]:
    assert main(["dump", str(DATA / name), *more]) == 0
    return capsys.readouterr().out.splitlines()


def test_a_tree_is_dumped_an_entry_and_a_column_at_a_time_as_go_hep_does(capsys):
    out = dumped(capsys, "simple.root")
    assert out[:5] == [
        f">>> file[{DATA / 'simple.root'}]",
        'key[000]: tree;1 "fake data" (TTree)',
        "[000][one]: 1",
        "[000][two]: 1.1",
        "[000][three]: uno",
    ]
    assert out[-1] == "[003][three]: quatro"


def test_entries_limits_how_much_of_each_tree_is_dumped(capsys):
    out = dumped(capsys, "small-flat-tree.root", "-n", "2")
    assert "[001][ArrayInt32]: [1 1 1 1 1 1 1 1 1 1]" in out
    assert "[001][SliceInt64]: [1]" in out
    assert not any(line.startswith("[002]") for line in out)


def test_an_rntuple_is_dumped_like_a_tree(capsys):
    out = dumped(capsys, "rntuple/test_stl_containers_rntuple_v1-0-0-0.root", "-n", "1")
    assert "[000][vector_vector_string]: [[one]]" in out
    assert "[000][lorentz_vector]: {pt: 1.0, eta: 1.0, phi: 1.0, mass: 1.0}" in out


def test_a_trees_unreadable_columns_are_named_with_the_reason(monkeypatch, capsys):
    from xrdroot.tree import TTree

    real = TTree.readable

    def fewer(self):
        self.unreadable["three"] = "a reason of its own"
        return [name for name in real(self) if name != "three"]

    monkeypatch.setattr(TTree, "readable", fewer)
    out = dumped(capsys, "simple.root")
    assert "  unreadable column three: a reason of its own" in out
    assert not any("[three]" in line for line in out)


def test_a_histogram_is_dumped_bin_by_bin_with_its_edges_and_errors(capsys):
    out = dumped(capsys, "dirs-6.14.00.root")
    assert 'key[002]: dir1/dir11/h1;1 "h1" (TH1F)' in out
    assert "  entries: 5.0  sum: 5.0" in out
    assert "  x: 100 bins in [0, 100)  mean: 1.37823  std: 1.42501" in out
    assert "  outside the axes: 0.0" in out
    assert "  bin 0 [0, 1): 3 ± 1.73205" in out
    assert 'key[003]: dir2;1 "dir2" (TDirectory)' in out


def test_a_two_dimensional_histogram_names_each_cell_by_both_bins(capsys):
    out = dumped(capsys, "gauss-h2.root")
    assert "  bin (0, 1) [0, 1) x [1, 2): 488 ± 22.0907" in out
    assert "  y: 3 bins in [0, 3)  mean: 0.894202  std: 1.83003" in out


def test_a_profile_has_no_sum_because_its_bins_are_means(capsys):
    out = dumped(capsys, "tprofile.root")
    assert "  entries: 24999.0" in out
    assert not any("sum:" in line or "outside" in line for line in out)


def test_a_graph_is_dumped_point_by_point_with_its_bars(capsys):
    out = dumped(capsys, "graphs.root")
    assert "  point 0: (1, 2)" in out
    assert "  point 3: (4, 8) x-0.4 x+0.4 y-0.8 y+0.8" in out
    assert "  point 0: (1, 2) x-0.1 x+0.2 y-0.3 y+0.4" in out


def test_a_graph_of_layered_bars_has_every_layer_and_a_multigraph_each_graph(capsys):
    out = dumped(capsys, "tgme.root")
    assert "  point 0: (0, 0) x-0.3 x+0.3 y-1 y+0.5 y-0.5 y+0.6" in out
    assert "  [2]: <TGraphAsymmErrors 'Graph' of 5 points>" in out


def test_an_efficiency_is_passed_over_total_and_its_interval(capsys):
    out = dumped(capsys, "tefficiency.root")
    assert "  method: clopper-pearson  level: 0.682689" in out
    assert "  bin 0: 47/111 = 0.423423 -0.0504015 +0.051905" in out
    assert any(line.startswith("  bin (0, 0, 0): 0/0 = 0") for line in out)


def test_a_function_is_its_formula_range_and_parameters(capsys):
    out = dumped(capsys, "tformula.root")
    assert "  formula: [p0]+[p1]*x" in out
    assert "  range: [0.0 10.0]" in out
    assert "  [1] p1 = 20 ± 0" in out
    assert any(line.startswith("  {TF1AbsComposition: {TObject: ") for line in out)


def test_strings_dates_and_containers_of_objects_are_dumped_as_they_read(capsys):
    assert "  2006-01-02 15:04:05" in dumped(capsys, "tdatime.root")
    clones = dumped(capsys, "tclonesarray-no-streamerbypass.root")
    assert "  [0]: {'TObject': {'fUniqueID': 0, 'fBits': 50331648}, 'fString': 'Elem-0'}" in clones
    record = dumped(capsys, "string-example.root")
    assert record[2].startswith('  {"LumiCounter.eventsByRun"')


def test_a_path_dumps_just_that_key(capsys):
    out = dumped(capsys, "graphs.root", "-k", "tg")
    assert out[1] == 'key[000]: tg;1 "graph without errors" (TGraph)'
    assert len(out) == 6
    inner = dumped(capsys, "dirs-6.14.00.root:dir1")
    assert inner[1] == 'key[000]: dir1/dir11;1 "dir11" (TDirectory)'


def test_a_key_that_will_not_read_says_why_and_the_dump_goes_on(monkeypatch, capsys):
    real = Directory.__getitem__

    def refusing(self, name):
        if name.startswith("tge"):
            raise UnsupportedFeatureError("tge is refused here")
        return real(self, name)

    monkeypatch.setattr(Directory, "__getitem__", refusing)
    out = dumped(capsys, "graphs.root")
    assert "  unreadable: tge is refused here" in out
    assert "  point 0: (1, 2) x-0.1 x+0.2 y-0.3 y+0.4" in out


def test_values_are_written_as_go_hep_writes_them():
    assert text(np.array([1, 2, 3])) == "[1 2 3]"
    assert text(np.float32(1.1)) == "1.1"
    assert text(np.array(5)) == "5"
    assert text([np.array([1.5]), "a"]) == "[[1.5] a]"
    assert text(Jagged(np.array([1, 2, 3]), np.array([0, 1, 3]))) == "[[1] [2 3]]"
    assert text({"a": np.array([1, 2]), "b": {"c": 1}}) == "{a: [1 2], b: {c: 1}}"
