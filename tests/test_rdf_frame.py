"""RDataFrame's transformations and actions, with go-hep's ``rdf`` tests ported.

The expected values are go-hep's, worked out by hand over its two trees: the
``xyn`` tree it writes itself (``x`` running 0 to 99, ``y = -x``), and
``small-flat-tree.root``, in whose entry ``i`` ``N = i % 10``, a slice of
``N`` copies of ``i`` and an array of ten. go-hep spells a collection's size
``Length$(...)``, as ``TTree::Draw`` does; an ``RDataFrame`` expression is
C++, where it is ``.size()``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import xrdroot
from frames import DATA, write_xyn
from xrdroot import FormulaError, Histogram, Jagged, Profile, RDataFrame, UnsupportedFeatureError
from xrdroot.rdf import CutFlowReport, Result, vecops

FLAT = str(DATA / "small-flat-tree.root")


@pytest.fixture
def xyn(tmp_path):
    with RDataFrame("tree", write_xyn(tmp_path / "xyn.root")) as df:
        yield df


@pytest.fixture
def flat():
    with RDataFrame("tree", FLAT, step=7) as df:
        yield df


def _slice_totals() -> tuple[int, float]:
    return sum(i % 10 for i in range(100)), float(sum((i % 10) * i for i in range(100)))


def test_count_and_filter(xyn):
    counts = [
        xyn.Count(),
        xyn.Filter("x >= 50").Count(),
        xyn.Filter("x >= 50").Filter("x < 60").Count(),
    ]
    assert [int(count) for count in counts] == [100, 50, 10]


def test_every_result_comes_from_one_pass(xyn):
    counts = [xyn.Count() for _ in range(10)]
    assert [count.GetValue() for count in counts] == [100] * 10
    assert xyn.GetNRuns() == 1
    assert counts[0].GetValue() == 100 and xyn.GetNRuns() == 1  # asking again runs nothing


def test_define_builds_on_earlier_defines(xyn):
    df = xyn.Define("z", "2*x + 1")
    total, mean, shifted = df.Sum("z"), df.Mean("z"), df.Define("w", "z - 1").Sum("w")
    assert (total.GetValue(), mean.GetValue(), shifted.GetValue()) == (10000.0, 100.0, 9900.0)


def test_a_define_after_a_filter_is_computed_only_for_what_passed(xyn):
    df = xyn.Filter("x >= 90").Define("z", "x*x")
    assert df.Sum("z").GetValue() == sum(i * i for i in range(90, 100))
    assert df.Count().GetValue() == 10


def test_histograms_of_one_and_two_columns_and_weights(xyn):
    h1 = xyn.Histo1D(("h1", "", 100, 0, 100), "x")
    h2 = xyn.Histo2D(("h2", "", 10, 0, 100, 10, -100, 0), "x", "y")
    hw = xyn.Define("two", "2.").Histo1D(("hw", "", 100, 0, 100), "x", "two")
    assert (h1.entries, h1.mean()) == (100, 49.5)
    # y = 0 is the upper edge of y's axis, so ROOT puts entry 0 in the overflow and
    # leaves it out of the moments; go-hep, which does not, says 49.5 and -49.5.
    assert (h2.GetValue().mean(0), h2.GetValue().mean(1)) == (50.0, -50.0)
    assert hw.GetValue().sum() == 200.0
    assert isinstance(h1.GetValue(), Histogram) and h1.GetValue().name == "h1"


def test_the_report_counts_what_each_named_filter_saw_and_kept(xyn):
    df = xyn.Filter("x >= 50", "half").Filter("x < 60", "narrow")
    report, count = df.Report(), df.Count()
    cuts = list(report.GetValue())
    assert [(cut.name, cut.all, cut.passed) for cut in cuts] == [
        ("half", 100, 50),
        ("narrow", 50, 10),
    ]
    assert cuts[0].GetEff() == 50.0 and count.GetValue() == 10
    assert str(report.GetValue()).splitlines() == [
        "half      : pass=50         all=100        -- eff=50.00 % cumulative eff=50.00 %",
        "narrow    : pass=10         all=50         -- eff=20.00 % cumulative eff=10.00 %",
    ]


def test_frames_are_immutable_so_one_tree_feeds_several_branches(xyn):
    everything, low, high = xyn.Count(), xyn.Filter("x < 10").Count(), xyn.Filter("x >= 90").Count()
    assert (everything.GetValue(), low.GetValue(), high.GetValue()) == (100, 10, 10)


def test_min_max_std_dev_and_take(xyn):
    assert (xyn.Min("x").GetValue(), xyn.Max("x").GetValue()) == (0.0, 99.0)
    wanted = math.sqrt(sum((i - 49.5) ** 2 for i in range(100)) / 99)
    assert xyn.StdDev("x").GetValue() == pytest.approx(wanted, abs=1e-12)
    assert xyn.Filter("x < 4").StdDev("x").GetValue() == pytest.approx(math.sqrt(5 / 3))
    assert xyn.Filter("x == 7").StdDev("x").GetValue() == 0.0
    assert xyn.Take("x").GetValue().tolist() == list(range(100))
    assert xyn.Filter("x < 5").Take("y").GetValue().tolist() == [0, -1, -2, -3, -4]


def test_a_graph_is_a_point_per_entry_in_order(xyn):
    graph = xyn.Filter("x < 5").Graph("x", "y").GetValue()
    assert graph.x.tolist() == [0, 1, 2, 3, 4] and graph.y.tolist() == [0, -1, -2, -3, -4]


def test_a_result_asked_for_runs_the_loop_without_being_told(xyn):
    assert xyn.Filter("x < 25").Count().GetValue() == 25


def test_a_range_is_counted_from_where_it_starts(xyn):
    df = xyn.Range(10, 30)
    assert (df.Count().GetValue(), df.Min("x").GetValue(), df.Max("x").GetValue()) == (
        20,
        10.0,
        29.0,
    )
    assert xyn.Range(5).Take("n").GetValue().tolist() == [0, 1, 2, 3, 4]
    assert xyn.Range(0, 10, 3).Take("n").GetValue().tolist() == [0, 3, 6, 9]
    assert xyn.Filter("n % 2 == 1").Range(2, 5).Take("n").GetValue().tolist() == [5, 7, 9]
    assert xyn.Range(95, 0).Count().GetValue() == 5


def test_mistakes_are_refused_when_they_are_made(xyn):
    with pytest.raises(FormulaError, match="nosuch"):
        xyn.Filter("nosuch > 0")
    with pytest.raises(FormulaError, match="could not be parsed"):
        xyn.Define("z", "x +")


def test_actions_over_collections_run_once_per_element(flat):
    number, total = _slice_totals()
    arrays, slices, count = flat.Sum("ArrayFloat64"), flat.Sum("SliceFloat64"), flat.Count()
    assert arrays.GetValue() == sum(10 * i for i in range(100))
    assert slices.GetValue() == total and count.GetValue() == 100
    h = flat.Histo1D(("h", "", 100, -1, 100), "SliceFloat64").GetValue()
    assert (h.entries, h.mean()) == (number, pytest.approx(total / number))


def test_a_defined_collection_behaves_like_a_branch_holding_one(flat):
    df = flat.Define("twice", "ArrayFloat64 * 2")
    assert df.Sum("twice").GetValue() == sum(20 * i for i in range(100))
    assert df.Histo1D(("h", "", 100, -1, 200), "twice").GetValue().entries == 1000


def test_a_reduced_collection_is_one_value_per_entry(flat):
    df = flat.Define("n", "SliceFloat64.size()")
    assert (df.Sum("n").GetValue(), df.Count().GetValue()) == (_slice_totals()[0], 100)


def test_a_filter_on_a_reduction_selects_entries(flat):
    df = flat.Filter("SliceFloat64.size() >= 5", "at least five")
    count, total, report = df.Count(), df.Sum("SliceFloat64"), df.Report()
    kept = [i for i in range(100) if i % 10 >= 5]
    assert (count.GetValue(), total.GetValue()) == (len(kept), sum((i % 10) * i for i in kept))
    assert [(cut.all, cut.passed) for cut in report.GetValue()] == [(100, 50)]


def test_a_filter_over_a_collection_is_refused_with_what_to_do_instead(flat):
    with pytest.raises(ValueError, match=r"Any\(\.\.\.\), All\(\.\.\.\), Sum"):
        flat.Filter("SliceFloat64 > 5").Count().GetValue()


def test_mean_and_extremes_over_collections(flat):
    assert flat.Mean("ArrayFloat64").GetValue() == pytest.approx(49.5)
    assert (flat.Min("ArrayFloat64").GetValue(), flat.Max("ArrayFloat64").GetValue()) == (0.0, 99.0)


def test_a_number_per_entry_goes_with_every_element(flat):
    assert flat.Sum("ArrayFloat64 - Float64").GetValue() == 0.0


def test_collections_that_do_not_line_up_are_refused(flat):
    with pytest.raises(ValueError, match="different sizes in entry 0, 0 and 10"):
        flat.Sum("SliceFloat64 + ArrayFloat64").GetValue()


def test_profiles_average_one_column_in_bins_of_another(flat):
    p = flat.Profile1D(("p", "", 10, 0, 100), "Float64", "Float64").GetValue()
    assert isinstance(p, Profile) and p.entries == 100
    assert p.values().tolist() == [10 * k + 4.5 for k in range(10)]
    assert (
        flat.Filter("Float64 >= 50").Profile1D(("p", "", 10, 0, 100), "Float64", "Float64").entries
        == 50
    )
    each = flat.Profile1D(("p", "", 10, 0, 100), "ArrayFloat32", "ArrayFloat64")
    assert each.GetValue().entries == 1000
    p2 = flat.Profile2D(
        ("p2", "", 10, 0, 100, 10, 0, 100), "Float64", "Float64", "Float64"
    ).GetValue()
    assert p2.entries == 100 and np.diag(p2.values()).tolist() == [10 * k + 4.5 for k in range(10)]


def test_models_come_as_tuples_or_booked_objects(xyn):
    edges = xyn.Histo1D(("e", "t;x;n", [0, 10, 50, 100]), "x").GetValue()
    counted = xyn.Histo1D(("c", "", 3, [0, 10, 50, 100]), "x").GetValue()
    assert counted.values().tolist() == edges.values().tolist()
    assert edges.values().tolist() == [10, 40, 50] and edges.title == "t"
    booked = Histogram.book("b", (10, 0, 100), kind="F")
    booked.fill([1, 2, 3])
    made = xyn.Histo1D(booked, "x").GetValue()
    assert made.classname == "TH1F" and made.entries == 100 and booked.entries == 3
    anonymous = xyn.Histo1D((10, 0, 100), "x").GetValue()
    assert anonymous.name == "x"
    three = xyn.Histo3D(("h3", "", 2, 0, 100, 2, -100, 0, 2, 0, 100), "x", "y", "n").GetValue()
    assert three.entries == 100 and three.values().sum() == 99  # y = 0 is overflow
    profiled = xyn.Profile1D(("p", "", 10, 0, 100, -50, 0, "s"), "x", "y").GetValue()
    assert profiled.entries == 51 and profiled.error_mode == "s"
    by_keyword = xyn.Histo1D("x", model=("k", "", 4, 0, 100), weight="n").GetValue()
    assert by_keyword.sum() == sum(range(100))


def test_bad_models_are_refused_by_name(xyn):
    with pytest.raises(ValueError, match="number of bins"):
        xyn.Histo1D(("h", "", 10, 0), "x")
    with pytest.raises(ValueError, match="left over"):
        xyn.Histo1D(("h", "", 10, 0, 1, 7), "x")
    with pytest.raises(ValueError, match="books a histogram of 2 axes"):
        xyn.Histo2D(Histogram.book("h", (10, 0, 1)), "x", "y")
    with pytest.raises(ValueError, match="averages and its error option"):
        xyn.Profile1D(("p", "", 10, 0, 1, 5), "x", "y")
    with pytest.raises(TypeError, match="needs a model"):
        xyn.Histo2D("x", "y")
    with pytest.raises(TypeError, match="2 column names"):
        xyn.Histo2D(("h", "", 1, 0, 1, 1, 0, 1), "x")


def test_a_histogram_without_a_model_spans_what_it_was_filled_with(xyn):
    h = xyn.Histo1D("x").GetValue()
    assert (len(h.values()), h.entries, h.axes[0].edges()[0]) == (128, 100, 0.0)
    assert h.values(flow=True)[-1] == 0  # the largest value is inside, not in the overflow
    weighted = xyn.Histo1D("x", "n").GetValue()
    assert weighted.sum() == sum(range(100))
    single = xyn.Filter("x == 3").Histo1D("x").GetValue()
    assert (single.axes[0].edges()[0], single.axes[0].edges()[-1]) == (2.0, pytest.approx(4.0))
    nothing = xyn.Filter("x < 0").Histo1D("x").GetValue()
    assert nothing.entries == 0


def test_expressions_are_columns_to_every_action(xyn):
    assert xyn.Sum("x * 2").GetValue() == 9900.0
    assert xyn.Histo1D(("h", "", 10, 0, 200), "x * 2").GetValue().mean() == 99.0


def test_integers_sum_exactly_and_nothing_sums_to_zero(xyn):
    total = xyn.Sum("n")
    assert total.GetValue() == 4950 and isinstance(total.GetValue(), int)
    assert xyn.Filter("n < 0").Sum("x").GetValue() == 0.0
    assert xyn.Filter("n < 0").Mean("x").GetValue() == 0.0
    assert xyn.Filter("n < 0").Min("n").GetValue() == 2**31 - 1
    assert xyn.Filter("n < 0").Max("x").GetValue() == -np.finfo(np.float64).max
    assert xyn.Define("b", "n > 3").Filter("n < 0").Min("b").GetValue() == 2**31 - 1
    assert xyn.Sum("n > 49").GetValue() == 50
    assert xyn.Define("u", "(unsigned int)n").Sum("u").GetValue() == 4950


def test_callables_see_a_whole_batch_at_a_time(xyn):
    sizes = []

    def square(x):
        sizes.append(len(x))
        return x * x

    df = xyn.Define("x2", square).Filter(lambda x2: x2 > 100, name="big")
    assert df.Count().GetValue() == 89 and sizes == [100]
    assert xyn.Define("r", np.hypot, ["x", "y"]).Max("r").GetValue() == pytest.approx(
        99 * math.sqrt(2)
    )
    assert xyn.Define("twice", np.add, ["x", "x"]).Sum("twice").GetValue() == 9900.0
    rows = xyn.Define(
        "v", lambda n: Jagged(np.repeat(n, n % 3), np.concatenate([[0], np.cumsum(n % 3)]))
    )
    assert rows.Sum("v").GetValue() == sum(i * (i % 3) for i in range(100))


def test_callables_that_do_not_give_a_value_per_entry_are_refused(xyn):
    with pytest.raises(ValueError, match="gave a float"):
        xyn.Define("z", lambda x: float(x[0])).Sum("z").GetValue()
    with pytest.raises(ValueError, match="gave 3 values"):
        xyn.Filter(lambda x: x[:3] > 0).Count().GetValue()
    with pytest.raises(TypeError, match="give columns="):
        xyn.Define("z", lambda *args: args[0])
    with pytest.raises(TypeError, match="give columns="):
        xyn.Define("z", max)
    with pytest.raises(KeyError, match="'nosuch'"):
        xyn.Define("z", lambda nosuch: nosuch)
    with pytest.raises(TypeError, match="give no columns"):
        xyn.Define("z", "x", ["x"])
    with pytest.raises(TypeError, match="an expression or a callable"):
        xyn.Define("z", 3)


def test_names_a_define_may_not_make(xyn):
    for name, message in [
        ("x", "Redefine replaces one"),
        ("rdfentry_", "ROOT's own"),
        ("rdfthing", "ROOT's own"),
        ("a b", "not a C\\+\\+ name"),
        ("class", "not a C\\+\\+ name"),
    ]:
        with pytest.raises(ValueError, match=message):
            xyn.Define(name, "1")


def test_redefine_alias_and_the_special_columns(xyn):
    df = xyn.Redefine("x", "x + 1").Alias("ex", "x")
    assert df.Sum("ex").GetValue() == 5050.0 and xyn.Sum("x").GetValue() == 4950.0
    assert xyn.Define("e", "rdfentry_").Take("e").GetValue().dtype == np.uint64
    assert xyn.Sum("rdfslot_").GetValue() == 0
    with pytest.raises(ValueError, match="none called 'nosuch'"):
        xyn.Redefine("nosuch", "1")
    with pytest.raises(KeyError, match="Alias names 'nosuch'"):
        xyn.Alias("a", "nosuch")


def test_define_per_sample_is_one_value_per_file_of_a_chain():
    files = [str(DATA / "chain.flat.1.root"), str(DATA / "chain.flat.2.root")]
    with RDataFrame("tree", files, step=3) as df:
        tagged = df.DefinePerSample("second", lambda info: info.Contains("flat.2"))
        assert tagged.Take("second").GetValue().tolist() == [False] * 5 + [True] * 5
        ranges = df.DefinePerSample("first", lambda info: info.EntryRange()[0]).Take("first")
        assert ranges.GetValue().tolist() == [0] * 5 + [5] * 5
    with pytest.raises(UnsupportedFeatureError, match="callable of the sample"):
        RDataFrame(3).DefinePerSample("s", "1")


def test_an_empty_frame_is_numbered_entries_for_define_to_fill(xyn):
    df = RDataFrame(10).Define("x", "rdfentry_ * 2.")
    assert df.Take("x").GetValue().tolist() == [2.0 * i for i in range(10)]
    assert RDataFrame(0).Define("x", "1").Take("x").GetValue().tolist() == []
    assert RDataFrame(0).Count().GetValue() == 0
    with pytest.raises(ValueError, match="number of entries"):
        RDataFrame(-1)


def test_reduce_aggregate_and_foreach(xyn):
    assert xyn.Reduce(np.add, "n").GetValue() == 4950
    assert xyn.Reduce(max, "x", -1.0).GetValue() == 99.0
    seen = []
    xyn.Foreach(lambda x: seen.append(len(x)))
    assert seen == [100]
    slots = []
    xyn.ForeachSlot(lambda slot, x: slots.append(slot))
    xyn.ForeachSlot(lambda slot, values: slots.append(len(values)), ["n"])
    assert slots == [0, 100]
    xyn.Foreach(lambda values: seen.append(values.sum()), ["n"])
    assert seen[-1] == 4950


def _collect(acc: list, values: np.ndarray) -> list:
    return [*acc, float(values.sum())]


def test_aggregate_folds_each_batch_into_a_copy_of_init(tmp_path):
    with RDataFrame("tree", write_xyn(tmp_path / "xyn.root"), step=30) as df:
        parts = df.Aggregate(_collect, lambda a, b: a + b, "x", []).GetValue()
    assert len(parts) == 4 and sum(parts) == 4950.0


def test_as_numpy_gives_every_column_or_those_asked_for(flat):
    everything = flat.Define("twice", "Int32 * 2").AsNumpy().GetValue()
    assert list(everything)[:2] == ["twice", "Int32"] and len(everything) == 21
    picked = flat.AsNumpy(["Int32", "SliceInt32"]).GetValue()
    assert picked["SliceInt32"].tolist()[:3] == [[], [1], [2, 2]]
    assert list(flat.AsNumpy("^Array.*32$").GetValue()) == [
        "ArrayInt32",
        "ArrayUInt32",
        "ArrayFloat32",
    ]
    assert "Str" not in flat.AsNumpy(exclude=["Str"]).GetValue()
    assert flat.AsNumpy(["Int32"])["Int32"][:2].tolist() == [0, 1]


def test_the_frame_hands_its_columns_to_other_libraries(flat):
    assert flat.to_numpy(["Int32"])["Int32"].sum() == 4950
    pytest.importorskip("pandas")
    assert flat.to_pandas(["Int32", "Float64"]).shape == (100, 2)
    pytest.importorskip("awkward")
    assert len(flat.to_awkward(["SliceInt32"])) == 100
    pytest.importorskip("pyarrow")
    assert flat.to_arrow(["Int32"]).num_rows == 100
    pytest.importorskip("polars")
    assert flat.to_polars(["Int32"]).height == 100


def test_the_display_is_a_box_of_the_first_entries(flat):
    shown = flat.Display(["Int32", "SliceFloat64", "Str"], 3).GetValue()
    lines = str(shown).splitlines()
    assert lines[:3] == [
        "+-----+-------+--------------+---------+",
        "| Row | Int32 | SliceFloat64 | Str     |",
        "+-----+-------+--------------+---------+",
    ]
    assert lines[7:9] == [
        "| 2   | 2     | 2.0          | evt-002 |",
        "|     |       | 2.0          |         |",
    ]
    assert lines[-1] == "..." and shown.AsString() == shown.as_string()
    few = flat.Range(2).Display(["ArrayInt32"], 5, elements=2).GetValue()
    assert str(few).splitlines()[3:6] == [
        "| 0   | 0          |",
        "|     | 0          |",
        "|     | ...        |",
    ]
    assert not str(few).endswith("...")
    everything = flat.Display(rows=1).GetValue()
    assert "SliceUInt64" in str(everything)


def test_display_and_report_print_themselves(flat, capsys):
    flat.Display(["Int32"], 1).GetValue().Print()
    flat.Filter("Int32 > 1", "positive").Report().GetValue().Print()
    printed = capsys.readouterr().out
    assert "| Int32 |" in printed and printed.rstrip().endswith("cumulative eff=98.00 %")


def test_a_report_at_the_head_lists_every_named_filter(xyn):
    xyn.Filter("x > 10", "a").Filter("x > 20")  # a filter with no name is not reported
    xyn.Filter("x < 5", "b")
    report = xyn.Report().GetValue()
    assert isinstance(report, CutFlowReport) and [cut.name for cut in report] == ["a", "b"]
    assert (report["a"].GetAll(), report["a"].GetPass(), report.At("b").GetName()) == (100, 89, "b")
    assert (
        report[1].passed == 5
        and len(report) == 2
        and repr(report["b"]) == "<CutInfo 'b': 5 of 100>"
    )
    with pytest.raises(KeyError, match=r"no filter called 'c'.*'a', 'b'"):
        report["c"]
    assert str(RDataFrame(1).Report().GetValue()) == ""
    with pytest.raises(KeyError, match="none"):
        RDataFrame(1).Report().GetValue().At("x")


def test_a_filter_nothing_reaches_has_no_efficiency(xyn):
    report = xyn.Filter("x < 0", "none").Filter("x > 1", "after").Report().GetValue()
    assert str(report).splitlines()[1].endswith("eff=nan % cumulative eff=0.00 %")
    assert str(RDataFrame(0).Filter("rdfentry_ > 0", "c").Report().GetValue()).endswith("eff=nan %")


def test_stats_are_a_tstatistic_of_a_column(xyn):
    stats = xyn.Stats("x").GetValue()
    assert (stats.GetN(), stats.GetMean(), stats.GetMin(), stats.GetMax()) == (100, 49.5, 0.0, 99.0)
    assert stats.GetRMS() == pytest.approx(np.std(np.arange(100), ddof=1))
    assert stats.GetMeanErr() == pytest.approx(stats.GetRMS() / 10)
    weighted = xyn.Stats("x", "n").GetValue()
    assert weighted.GetW() == 4950 and weighted.GetW2() == sum(i * i for i in range(100))
    assert weighted.GetMean() == pytest.approx(sum(i * i for i in range(100)) / 4950)
    empty = xyn.Filter("x < 0").Stats("x").GetValue()
    assert (empty.GetVar(), empty.GetMean(), empty.GetMeanErr(), empty.GetMin()) == (
        0.0,
        0.0,
        0.0,
        math.inf,
    )
    assert "n=100" in repr(stats)


def test_what_a_frame_can_say_about_itself(flat):
    df = flat.Define("twice", "ArrayFloat64 * 2").Define("half", "Int32 / 2.f").Alias("i", "Int32")
    assert df.GetColumnNames()[:3] == ["twice", "half", "i"] and df.columns == df.GetColumnNames()
    assert df.GetDefinedColumnNames() == ["twice", "half"]
    assert df.HasColumn("i") and not df.HasColumn("rdfentry") and df.GetNSlots() == 1
    types = {
        name: df.GetColumnType(name)
        for name in ("twice", "half", "Int32", "Str", "rdfentry_", "rdfslot_")
    }
    assert types == {
        "twice": "ROOT::VecOps::RVec<double>",
        "half": "float",
        "Int32": "Int_t",
        "Str": "std::string",
        "rdfentry_": "ULong64_t",
        "rdfslot_": "unsigned int",
    }
    assert df.Filter("Int32 > 3", "cut").Filter("Int32 < 9").GetFilterNames() == ["cut"]
    with pytest.raises(KeyError, match="no column called 'nosuch'"):
        df.GetColumnType("nosuch")
    assert repr(df) == "<RDataFrame over Chain 'tree' of " + FLAT + ", 23 columns>"
    assert repr(df.Filter("Int32 > 3")).startswith("<RNode over")


def test_describe_lists_every_column_and_where_it_comes_from(flat):
    text = (
        flat.Define("pair", "Combinations(SliceInt32, 2)")
        .Define("fixed", "ArrayInt32 > 1")
        .Describe()
    )
    lines = text.splitlines()
    assert lines[0] == f"Dataframe from Chain 'tree' of {FLAT}"
    assert "Columns from defines    2" in lines
    assert any(
        line.split() == ["pair", "ROOT::VecOps::RVec<ROOT::VecOps::RVec<ULong64_t>>", "Define"]
        for line in lines
    )
    assert any(line.split() == ["fixed", "ROOT::VecOps::RVec<int>", "Define"] for line in lines)
    assert any(
        line.split() == ["ArrayInt32", "ROOT::VecOps::RVec<Int_t>", "Dataset"] for line in lines
    )


def test_results_are_promises_that_behave_like_what_they_promise(xyn):
    count = xyn.Count()
    assert repr(count) == "<Result of Count, not yet computed>" and not count.IsReady()
    assert int(count) == 100 and float(count) == 100.0 and count.value == 100 and count.IsReady()
    assert (
        repr(count) == "<Result of Count: 100>"
        and str(count) == "100"
        and [10, 20][xyn.Range(1).Count()] == 20
    )
    taken = xyn.Take("n")
    assert len(taken) == 100 and list(taken)[:2] == [0, 1] and taken[3] == 3
    assert np.asarray(taken).sum() == 4950 and taken.get_value() is taken.GetValue()
    assert xyn.Histo1D(("h", "", 1, 0, 100), "x").mean() == 49.5  # attributes of the value itself
    with pytest.raises(AttributeError):
        _ = count.__nothing__
    assert isinstance(count, Result)


def test_snake_case_spells_everything_too(xyn):
    df = xyn.define("z", "x * 2").filter("z > 10", "cut").alias("zz", "z").redefine("z", "z + 1")
    assert df.count().get_value() == 94 and df.sum("zz").value == sum(2 * i for i in range(6, 100))
    assert df.histo1d(("h", "", 10, 0, 200), "z").entries == 94
    assert df.get_column_names()[:2] == ["z", "zz"] and df.get_filter_names() == ["cut"]
    assert df.as_numpy(["n"])["n"][0] == 6 and df.report().GetValue()["cut"].passed == 94


def test_ways_of_making_a_frame_that_are_not_are_refused(xyn):
    with pytest.raises(TypeError, match="a number of empty entries"):
        RDataFrame()
    with pytest.raises(UnsupportedFeatureError, match="not from a dict"):
        RDataFrame({})
    with pytest.raises(ValueError, match="step must be"):
        RDataFrame(5, step=0)
    with pytest.raises(ValueError, match="workers must be"):
        RDataFrame(5, workers=0)
    with pytest.raises(TypeError, match="name of one column"):
        xyn.Sum(["x"])
    with pytest.raises(ValueError, match="is not a span"):
        xyn.Range(5, 2)
    with pytest.raises(ValueError, match="is not a span"):
        xyn.Range(0, 5, 0)


def test_a_frame_over_a_tree_it_was_handed_leaves_it_open(tmp_path):
    with xrdroot.open_root(write_xyn(tmp_path / "xyn.root")) as f:
        tree = f["tree"]
        with RDataFrame(tree) as df:
            assert df.Sum("x").GetValue() == 4950.0
        assert tree["x"].array(0, 2).tolist() == [0.0, 1.0]


def test_columns_defined_in_every_way_and_where_they_are_computed(tmp_path):
    with RDataFrame("tree", write_xyn(tmp_path / "xyn.root"), step=30) as df:
        rooted = df.Define("r", np.sqrt, "x").Define("t", 'n > 50 ? "high" : "low"')
        assert rooted.GetColumnType("t") == "std::string" and rooted.Max(
            "r"
        ).GetValue() == pytest.approx(99**0.5)
        deeper = df.Filter("x > 10").Define("z", "x * 10").Filter("z > 500").Sum("z")
        assert deeper.GetValue() == sum(10 * i for i in range(51, 100))
        assert df.Filter("x < 0").StdDev("x").GetValue() == 0.0
        sampled = df.DefinePerSample("s", lambda info: 1)
        assert "s" in sampled.Describe().split("DefinePerSample")[0].splitlines()[-1]


def test_what_combinations_gives_is_a_column_of_several_collections():
    df = RDataFrame(6).Define("v", "Range(rdfentry_ % 3)")
    pairs = df.Define("pairs", lambda v: vecops.Combinations(v, 2))
    first, second = pairs.Take("pairs").GetValue()
    assert (first.tolist(), second.tolist()) == (
        [[], [], [0], [], [], [0]],
        [[], [], [1], [], [], [1]],
    )
    assert "| 2   | [0]   |" in str(pairs.Display(["pairs"], 3).GetValue())
    with RDataFrame(6, step=4) as split:
        again = split.Define("v", "Range(rdfentry_ % 3)").Define("p", "Combinations(v, 2)")
        assert again.Take("p").GetValue()[1].tolist() == second.tolist()
    kept = pairs.Filter("v.size() > 1 && pairs[0].size() > 0").Count()
    assert kept.GetValue() == 2
    every = pairs.Filter("v.size() >= 0 && v.size() < 5").Count()
    assert every.GetValue() == 6
    with pytest.raises(ValueError, match="several collections"):
        pairs.Sum("pairs").GetValue()
