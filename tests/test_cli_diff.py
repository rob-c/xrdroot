"""``xrdroot diff``: two files compared key by key and value by value, with diff's status."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from xrdroot import Graph, Histogram, create, open_root
from xrdroot.cli import main
from xrdroot.cli.diff import Tolerance, _column, compare
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.file import Directory

DATA = pathlib.Path(__file__).parent / "data"


def data(name: str) -> str:
    return str(DATA / name)


def diffed(capsys, *argv: str) -> tuple[int, list[str], str]:
    status = main(["diff", *argv])
    out, err = capsys.readouterr()
    return status, out.splitlines(), err


def booked(name: str, counts: list[float]) -> Histogram:
    h = Histogram.book(name, (len(counts), 0.0, float(len(counts))))
    h.fill(np.arange(len(counts)) + 0.5, weight=np.asarray(counts, dtype=float))
    return h


@pytest.fixture
def pair(tmp_path):
    """Two files alike but for what each test writes into them."""

    def write(one: dict, two: dict) -> tuple[str, str]:
        paths = []
        for index, content in enumerate((one, two)):
            path = tmp_path / f"side{index}.root"
            with create(str(path)) as out:
                for name, value in content.items():
                    out[name] = value
            paths.append(str(path))
        return paths[0], paths[1]

    return write


def test_a_file_is_the_same_as_itself_and_says_nothing(capsys):
    for name in (
        "graphs.root",
        "dirs-6.14.00.root",
        "tefficiency.root",
        "small-evnt-tree-fullsplit.root",
    ):
        assert diffed(capsys, data(name), data(name)) == (0, [], "")


def test_names_in_one_file_only_are_said_with_the_file_they_are_in(capsys):
    status, out, _ = diffed(capsys, data("gauss-h1.root"), data("gauss-h2.root"))
    assert status == 1
    assert f"only in {data('gauss-h1.root')}: h1d" in out
    assert f"only in {data('gauss-h2.root')}: h2d-var" in out


def test_keys_picks_the_names_compared_and_names_one_in_neither(capsys):
    status, out, _ = diffed(capsys, data("gauss-h1.root"), data("gauss-h1.root"), "-k", "h1d, nope")
    assert status == 1
    assert out == ["in neither: nope"]


def test_quiet_says_nothing_and_the_status_says_it_all(capsys):
    assert diffed(capsys, data("gauss-h1.root"), data("gauss-h2.root"), "-q") == (1, [], "")


def test_one_name_of_two_classes_is_a_difference_of_class(capsys, pair):
    one, two = pair({"x": booked("x", [1, 2])}, {"x": Graph.new("x", [1, 2], [3, 4])})
    _, out, _ = diffed(capsys, one, two)
    assert out == [f"x: a TH1D in {one}, a TGraph in {two}"]


def test_directories_are_compared_all_the_way_down(capsys, pair):
    one, two = pair(
        {"a/b/h": booked("h", [1, 2]), "a/c": booked("c", [1])},
        {"a/b/h": booked("h", [1, 3]), "a/c": booked("c", [1])},
    )
    status, out, _ = diffed(capsys, one, two)
    assert status == 1
    assert out == [
        "a/b/h: contents differ in 1 of 4 places, first at 2: 2.0 and 3.0",
        "a/b/h: errors differ in 1 of 4 places, first at 2: 4.0 and 9.0",
    ]
    assert diffed(capsys, f"{one}:a/c", f"{two}:a/c")[0] == 0
    assert diffed(capsys, f"{one}:a", f"{two}:a")[1][0].startswith("a/b/h: contents differ")


def test_histograms_differ_by_bins_edges_errors_and_entries(capsys, pair):
    wider = Histogram.book("h", (2, 0.0, 4.0))
    wider.fill([0.5, 1.5])
    one, two = pair(
        {"h": booked("h", [1, 2]), "shape": booked("shape", [1, 2])},
        {"h": wider, "shape": booked("shape", [1, 2, 3])},
    )
    _, out, _ = diffed(capsys, one, two)
    assert "shape: (2,) bins and (3,) bins" in out
    assert "h: edges of axis 0 differ in 2 of 3 places, first at 1: 1.0 and 2.0" in out
    assert any(line.startswith("h: contents differ") for line in out)
    assert any(line.startswith("h: errors differ") for line in out)


def test_the_tolerance_makes_floats_and_doubles_the_same(capsys):
    one, two = f"{data('gauss-h1.root')}:h1d", f"{data('gauss-h1.root')}:h1f"
    status, out, _ = diffed(capsys, one, two)
    assert status == 1
    assert out[0].startswith("h1d: contents differ in 10 of 12 places, first at 1: 6.6 and 6.59")
    assert diffed(capsys, one, two, "--rtol", "1e-4")[0] == 0
    assert diffed(capsys, one, two, "--atol", "1")[0] == 0


def test_graphs_differ_by_their_points_and_bars(capsys):
    status, out, _ = diffed(capsys, f"{data('graphs.root')}:tge", f"{data('graphs.root')}:tgae")
    assert status == 1
    assert out == [
        "tge: x errors differ in 4 of 8 places, first at (1, 0): 0.1 and 0.2",
        "tge: y errors differ in 8 of 8 places, first at (0, 0, 0): 0.2 and 0.3",
    ]
    _, plain, _ = diffed(capsys, f"{data('graphs.root')}:tg", f"{data('graphs.root')}:tge")
    assert plain[0] == "tg: x errors have shapes (0,) and (2, 4)"


def test_graphs_of_different_lengths_and_points_differ(capsys, pair):
    one, two = pair(
        {"g": Graph.new("g", [1, 2], [3, 4]), "h": Graph.new("h", [1, 2], [3, 4])},
        {"g": Graph.new("g", [1, 2, 3], [3, 4, 5]), "h": Graph.new("h", [1, 2], [3, 5])},
    )
    _, out, _ = diffed(capsys, one, two)
    assert out == [
        "g: 2 points and 3 points",
        "h: y differ in 1 of 2 places, first at 1: 4.0 and 5.0",
    ]


def test_efficiencies_differ_by_what_passed_and_the_total(capsys):
    eff = data("tefficiency.root")
    _, out, _ = diffed(capsys, f"{eff}:eff1", f"{eff}:eff2")
    assert out == [
        "eff1 (passed): (10,) bins and (10, 10) bins",
        "eff1 (total): (10,) bins and (10, 10) bins",
    ]


def test_functions_differ_by_formula_and_parameters(capsys):
    functions = data("tformula.root")
    _, out, _ = diffed(capsys, f"{functions}:func1", f"{functions}:func3")
    assert out == [
        "func1: '[p0]+[p1]*x' and None",
        "func1: parameters have shapes (2,) and (4,)",
    ]


def test_members_of_a_class_are_compared_member_by_member(capsys):
    status, out, _ = diffed(capsys, data("tformula.root"), data("tformula-v14.root"))
    assert status == 1
    assert out == [
        "fconv.TF1AbsComposition.TObject.fBits: values 50331648 and 0",
        "fnorm.TF1AbsComposition.TObject.fBits: values 50331648 and 0",
    ]


def test_trees_differ_by_entries_columns_and_each_columns_first_difference(capsys):
    status, out, _ = diffed(capsys, data("chain.flat.1.root"), data("chain.flat.2.root"))
    assert status == 1
    assert "tree.I32: entries differ in 5 of 5 places, first at 0: 0 and -5" in out
    assert "tree.Str: entries differ, first at 0: 'str-0' and 'str-5'" in out
    assert "tree.ArrI8: entries differ in 50 of 50 places, first at (0, 0): 0 and -5" in out
    assert "tree.SliI8: row lengths differ in 5 of 5 places, first at 0: 0 and 5" in out
    assert "tree.SliI8: elements have shapes (10,) and (35,)" in out


def test_trees_of_different_lengths_or_columns_say_so(capsys):
    _, out, _ = diffed(capsys, data("simple.root"), data("small-flat-tree.root"))
    assert out == ["tree: 4 entries and 100 entries"]
    _, out, _ = diffed(capsys, data("std-map-split0.root"), data("std-map-split1.root"))
    assert f"only in {data('std-map-split1.root')}: tree.mi32" in out
    _, out, _ = diffed(capsys, data("std-map-split1.root"), data("std-map-split0.root"))
    assert f"only in {data('std-map-split1.root')}: tree.mi32" in out


def test_objects_per_entry_are_compared_whole(capsys):
    _, out, _ = diffed(capsys, data("chain.1.root"), data("chain.2.root"))
    assert out[0].startswith("tree.evt: entries differ, first at 0: {'Beg': 'beg-000'")


def test_a_later_batch_counts_its_entries_from_where_it_starts(capsys, pair, monkeypatch):
    monkeypatch.setattr("xrdroot.cli.diff.STEP", 3)
    x = np.arange(7.0)
    one, two = pair({"t": {"x": x}}, {"t": {"x": np.where(x > 4, -1.0, x)}})
    _, out, _ = diffed(capsys, one, two)
    assert out == ["t.x: entries differ in 1 of 3 places, first at 5: 5.0 and -1.0"]


def test_an_object_neither_side_reads_is_said_on_standard_error_and_not_counted(
    capsys, monkeypatch
):
    real = Directory.__getitem__

    def refusing(self, name):
        if name == "tg":
            raise UnsupportedFeatureError("tg is refused here")
        return real(self, name)

    monkeypatch.setattr(Directory, "__getitem__", refusing)
    status, out, err = diffed(capsys, data("graphs.root"), data("graphs.root"))
    assert (status, out) == (0, [])
    assert err == "xrdroot diff: tg not compared: tg is refused here\n"


def test_values_of_other_kinds_are_compared_as_they_are():
    tolerance = Tolerance()
    assert list(compare("a", 1, tolerance)) == ["the objects: a str and a int"]
    assert list(compare("a", "b", tolerance)) == ["the objects: 'a' and 'b'"]
    assert list(compare(True, True, tolerance)) == []
    assert list(compare([1, 2], [1], tolerance, path="p")) == ["p: 2 items and 1 items"]
    assert list(compare([1, 2], [1, 3], tolerance, path="p")) == ["p[1]: values 2 and 3"]
    assert list(compare({"a": 1}, {"b": 1}, tolerance, path="p")) == [
        "p.a: only in the first",
        "p.b: only in the second",
    ]
    assert list(compare(np.array(["a", "b"]), np.array(["a", "c"]), tolerance, path="s")) == [
        "s: values differ in 1 of 2 places, first at 1: b and c"
    ]


def test_a_column_of_objects_in_an_array_is_compared_an_entry_at_a_time():
    one = np.array([{"a": 1}, {"a": 2}], dtype=object)
    two = np.array([{"a": 1}, {"a": 3}], dtype=object)
    assert list(_column(one, two, "t.c", Tolerance(), 10)) == [
        "t.c: entries differ, first at 11: {'a': 2} and {'a': 3}"
    ]
    assert list(_column(one, one, "t.c", Tolerance(), 0)) == []


def test_two_open_files_compare_the_same_way_the_command_does():
    with open_root(data("graphs.root")) as one, open_root(data("graphs.root")) as two:
        assert list(compare(one, two, Tolerance())) == []
