"""``ROOT.RDataFrame``, ``ROOT.RDF``, ``ROOT.RVec`` and ``ROOT.VecOps`` as PyROOT scripts use them.

The frame is xrdroot's; what is checked here is the PyROOT side of it - a
frame made from a ``TTree`` of :mod:`xrdroot.pyroot.trees` or a dict of
arrays, results handed back through ``core``'s wrapper, ``AsNumpy`` giving
its dict at once with an ``RVec`` per entry of a collection - and that
``VecOps`` of one ``RVec`` gives what the batched functions give.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from xrdroot.pyroot.rdf.frame import RResultPtr
from xrdroot.pyroot.trees import hooks


def _frame():
    return ROOT.RDF.FromNumpy({"x": np.arange(10.0), "n": np.arange(10, dtype=np.int32)})


def test_a_display_prints_every_column_by_name_with_roots_digits(capsys):
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.float64)
    ROOT.RDF.FromNumpy({"x": x, "y": y}).Define("z", "x + y").Display().Print()
    # What df032_RDFFromNumpy.py prints under ROOT 6.40, trailing blanks and all.
    assert capsys.readouterr().out == (
        "+-----+---+----------+----------+\n"
        "| Row | x | y        | z        | \n"
        "+-----+---+----------+----------+\n"
        "| 0   | 1 | 4.000000 | 5.000000 | \n"
        "+-----+---+----------+----------+\n"
        "| 1   | 2 | 5.000000 | 7.000000 | \n"
        "+-----+---+----------+----------+\n"
        "| 2   | 3 | 6.000000 | 9.000000 | \n"
        "+-----+---+----------+----------+\n"
    )


def test_a_display_quotes_strings_and_spells_booleans_as_cpp_does():
    words = np.array(["hi", "yo"])
    df = ROOT.RDF.FromNumpy({"s": words, "b": np.array([True, False]), "f": np.array([1e20, 0])})
    shown = df.Display(["s", "b", "f"], 1).AsString()
    # ROOT's box for std::string, bool and a double of 1e20.
    assert shown.splitlines()[3] == '| 0   | "hi" | true | 100000000000000000000.000000 | '
    assert df.Display(["b"], 2).AsString().splitlines()[5] == "| 1   | false | "
    raw = xrdroot.rdf.Display({"c": np.array([b"ab"])}, 1, 10).AsString()
    assert raw.splitlines()[3] == '| 0   | "ab" | '


def test_a_frame_of_numpy_arrays_defines_filters_and_books_as_root_does():
    df = _frame()
    selected = df.Define("z", "x * n").Filter("z > 10", "big")
    h = selected.Histo1D(ROOT.RDF.TH1DModel("h", "z", 10, 0.0, 100.0), "z")
    count = selected.Count()
    assert isinstance(count, RResultPtr) and not count.IsReady()
    assert count.GetValue() == 6 and int(count) == 6 and float(count) == 6.0
    assert count.IsReady() and str(count) == "6" and "Result" in repr(count)
    assert h.GetValue().GetName() == "h" and h.GetPtr().GetEntries() == 6 and h.GetEntries() == 6
    assert ROOT.RDF.MakeNumpyDataFrame({"a": np.ones(3)}).Count().GetValue() == 3
    assert "RNode" in repr(selected) and selected.GetFilterNames() == ["big"]
    with pytest.raises(AttributeError):
        count._hidden  # noqa: B018
    with pytest.raises(AttributeError):
        df._hidden  # noqa: B018


def test_results_are_handed_back_through_the_wrapper_core_installs(monkeypatch):
    monkeypatch.setattr(hooks, "wrap", lambda obj: ("wrapped", obj))
    assert _frame().Count().GetValue() == ("wrapped", 10)
    taken = _frame().Take("x")
    assert len(taken) == 2 and taken[0] == "wrapped" and next(iter(taken)) == "wrapped"


def test_as_numpy_gives_its_columns_at_once_with_an_rvec_per_collection():
    t = ROOT.TTree("t", "")
    v = ROOT.std.vector["float"]()
    t.Branch("v", v)
    for i in range(3):
        v.assign(range(i))
        t.Fill()
    got = ROOT.RDataFrame(t).Define("s", "Sum(v)").AsNumpy(ROOT.std.vector["string"](["s", "v"]))
    assert got["s"].tolist() == [0.0, 0.0, 1.0] and got["v"].dtype == object
    assert (type(got["v"][2]), list(got["v"][2])) == (np.ndarray, [0.0, 1.0])
    assert list(_frame().AsNumpy(["x"], exclude=None)) == ["x"]


def test_a_frame_is_made_from_a_file_empty_entries_and_run_together(tmp_path):
    path = tmp_path / "rdf.root"
    with xrdroot.create(str(path)) as f:
        f["events"] = {"pt": np.arange(5.0)}
    df = ROOT.RDataFrame("events", str(path))
    means = df.Mean("pt")
    empty = ROOT.RDataFrame(4).Define("one", "1")
    total = empty.Sum("one")
    assert ROOT.RDF.RunGraphs([means, total]) == 2
    assert means.GetValue() == 2.0 and total.GetValue() == 4
    assert df.GetColumnNames() == ["pt"] and df.GetNSlots() >= 1
    written = df.Snapshot("copy", str(tmp_path / "copy.root"))
    assert isinstance(written, ROOT.RDataFrame) and written.Count().GetValue() == 5
    lazy = df.Snapshot("again", str(tmp_path / "again.root"), lazy=True)
    assert isinstance(lazy, RResultPtr) and lazy.GetValue().Count().GetValue() == 5
    written.close()
    lazy.GetValue().close()
    df.close()


def test_rdf_models_are_the_tuples_xrdroot_books_from_and_the_rest_is_refused():
    assert ROOT.RDF.TH2DModel("h", "", 2, 0, 1, np.array([0.0, 1.0])) == ("h", "", 2, 0, 1, [0, 1])
    assert ROOT.RDF.TProfile1DModel("p", "", 2, 0.0, 1.0) == ("p", "", 2, 0.0, 1.0)
    assert ROOT.RDF.RNode is ROOT.RDataFrame and repr(ROOT.RDF) == "<namespace ROOT::RDF>"
    with pytest.raises(AttributeError, match=r"ROOT has RDF\.MakeTrivialDataFrame; xrdroot"):
        ROOT.RDF.MakeTrivialDataFrame  # noqa: B018
    ROOT.EnableImplicitMT(2)
    assert ROOT.IsImplicitMTEnabled() and ROOT.GetThreadPoolSize() == 2
    ROOT.DisableImplicitMT()
    assert not ROOT.IsImplicitMTEnabled()


def test_an_rvec_is_a_vector_that_does_numpy_arithmetic_and_is_indexed_by_a_mask():
    v = ROOT.RVec["double"]([1.0, 2.0, 3.0])
    assert ROOT.RVecD is ROOT.RVec["double"] is ROOT.VecOps.RVec["Double_t"]
    doubled = (v * 2)[v > 1]
    assert isinstance(doubled, ROOT.RVecD) and list(doubled) == [4.0, 6.0]
    assert list(v[[0, 2]]) == [1.0, 3.0] and v[1] == 2.0 and list(v[1:]) == [2.0, 3.0]
    assert list(v == 2.0) == [False, True, False] and list(v != 2.0) == [True, False, True]
    assert list(np.sqrt(ROOT.RVecF([4, 9]))) == [2.0, 3.0] and np.sum(v) == 6.0
    grown = ROOT.RVecI()
    grown.push_back(3)
    assert grown.size() == 1 and "RVec<int>" in repr(grown) and ROOT.RVecB([1]).dtype == bool
    assert ROOT.RVecL([1]).dtype == np.int64
    with pytest.raises(TypeError, match="an RVec here holds numbers"):
        ROOT.RVec["string"]


def test_an_rvec_prints_as_roots_stream_writes_it_and_its_functions_are_named_bare():
    assert str(ROOT.RVecF([1.0, 2.0, 2.0 / 3])) == "{ 1, 2, 0.666667 }"
    assert str(ROOT.RVecB([1, 0])) == "{ 1, 0 }" and str(ROOT.RVecI([])) == "{  }"
    assert ROOT.Any(ROOT.RVecB([0, 1])) and ROOT.Mean(ROOT.RVecD([1.0, 3.0])) == 2.0
    with pytest.raises(AttributeError, match="ROOT has NoSuchName"):
        ROOT.NoSuchName  # noqa: B018
    v = ROOT.RVecD([3.0, 1.0, 2.0])
    v[0:2] = [5.0, 4.0]
    assert list(v) == [5.0, 4.0, 2.0]


def _listed(value):
    return list(value) if hasattr(value, "__len__") else value


def test_vecops_of_one_rvec_gives_what_rdataframes_vecops_gives():
    v = ROOT.RVecF([3.0, 1.0, 2.0])
    ops = ROOT.VecOps
    got = [
        ops.Sum(v),
        ops.Max(v),
        ops.Mean(v),
        ops.Sort(v),
        ops.Argsort(v),
        ops.ArgMax(v),
        ops.Take(v, 2),
        ops.Take(v, [2, 0]),
        ops.Nonzero(ROOT.RVecI([0, 4, 0])),
        ops.Sum(ROOT.RVecF()),
    ]
    assert [_listed(each) for each in got] == [
        6.0,
        3.0,
        2.0,
        [1, 2, 3],
        [1, 2, 0],
        0,
        [3, 1],
        [2, 3],
        [1],
        0.0,
    ]
    first, second = ops.Combinations(v, 2)
    assert (list(first), list(second)) == ([0, 0, 1], [1, 2, 2])
    assert np.isnan(ops.Max(ROOT.RVecF()))
    assert ops.DeltaPhi(0.1, 3.0) == pytest.approx(2.9)
    mass = ops.InvariantMass(
        ROOT.RVecF([10, 20]), ROOT.RVecF([0, 1]), ROOT.RVecF([0, 2]), ROOT.RVecF([0.1, 0.1])
    )
    assert mass == pytest.approx(27.995, abs=1e-3)


def test_vecops_maps_and_filters_with_python_functions_and_refuses_what_it_lacks():
    v = ROOT.RVecF([3.0, 1.0, 2.0])
    assert list(ROOT.VecOps.Map(v, lambda x: x * x)) == [9.0, 1.0, 4.0]
    assert list(ROOT.VecOps.Map(v, v, lambda a, b: a + b)) == [6.0, 2.0, 4.0]
    assert list(ROOT.VecOps.Filter(v, lambda x: x > 1)) == [3.0, 2.0]
    assert (ROOT.VecOps.Sum.__name__, repr(ROOT.VecOps)) == ("Sum", "<namespace ROOT::VecOps>")
    with pytest.raises(AttributeError, match=r"ROOT has VecOps\.Nope; xrdroot\.pyroot does not"):
        ROOT.VecOps.Nope  # noqa: B018
    with pytest.raises(AttributeError, match=r"ROOT has VecOps\._rows"):
        ROOT.VecOps._rows  # noqa: B018


def test_the_namespace_refuses_a_name_root_has_and_it_does_not():
    with pytest.raises(AttributeError, match=r"ROOT has TFoo; xrdroot\.pyroot does not yet"):
        ROOT.TFoo  # noqa: B018
    assert {"TTree", "RDataFrame", "std", "RVec", "EnableImplicitMT"} <= set(ROOT.__all__)


def test_a_result_is_an_index_and_a_frame_hands_back_what_is_not_a_method():
    from xrdroot.pyroot.rdf import rvec

    count = _frame().Count()
    assert [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10][count] == 10
    frame = _frame()
    frame._inner.extra = "kept"
    assert frame.extra == "kept"
    assert rvec._made(3) == 3 and rvec._unbatched(2.5) == 2.5
    assert list(rvec._unbatched(np.zeros(0))) == []
