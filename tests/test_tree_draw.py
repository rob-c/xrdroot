"""``TTree::Draw``: histograms, profiles and graphs filled from a tree by expression.

The expected numbers are go-hep's ``rdraw`` tests where they apply - its own
tree of ``x = 0..99``, ``y = -x``, an integer ``n`` and a bool ``ok``, and
``small-flat-tree.root``, whose ``i``-th entry holds ``i`` everywhere, a
slice of ``i % 10`` copies of it and an array of ten - and ROOT's own code
where go-hep and ROOT part: ROOT pairs collections of different lengths up
to the shorter, and books the axes ``THLimitsFinder`` finds, which each test
works out by hand from ``TSelectorDraw``, ``THLimitsFinder`` and
``TH1::ExtendAxis``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from xrdroot import FormulaError, Graph, Histogram, Profile, chain, create, open_root
from xrdroot.treedraw import DrawnHistogram, is_integer

DATA = pathlib.Path(__file__).parent / "data"
ENTRY = np.arange(100)


@pytest.fixture(scope="module")
def simple(tmp_path_factory):
    """go-hep's tree: x runs 0..99, y is -x, n is x as an int32, ok says x is even."""
    path = tmp_path_factory.mktemp("draw") / "tree.root"
    with create(str(path)) as f:
        f["tree"] = {
            "x": ENTRY.astype(np.float64),
            "y": -ENTRY.astype(np.float64),
            "n": ENTRY.astype(np.int32),
            "ok": ENTRY % 2 == 0,
        }
    with open_root(str(path)) as f:
        yield f["tree"]


@pytest.fixture
def flat():
    with open_root(str(DATA / "small-flat-tree.root")) as f:
        yield f["tree"]


def _sumw(h):
    return float(h.values(flow=True).sum())


# -- go-hep's TestH1D and friends ------------------------------------------


@pytest.mark.parametrize(
    ("expr", "selection", "keywords", "entries", "sumw", "mean"),
    [
        ("x", "", {}, 100, 100, 49.5),
        ("2*x", "", {"bins": (100, 0, 200)}, 100, 100, 99),
        ("x", "x >= 50", {}, 50, 50, 74.5),
        ("x", "x >= 20 && x < 30", {}, 10, 10, 24.5),
        ("x", "ok", {}, 50, 50, 49),
        ("n", "", {}, 100, 100, 49.5),
        ("abs(y)", "", {}, 100, 100, 49.5),
        ("TMath::Abs(y)", "", {}, 100, 100, 49.5),
        ("x", "", {"weight": 2}, 100, 200, 49.5),
        ("x", "", {"weight": "2"}, 100, 200, 49.5),
        ("x", "", {"entries": 10}, 10, 10, 4.5),
        ("x", "", {"first_entry": 90}, 10, 10, 94.5),
    ],
)
def test_go_hep_s_one_dimensional_draws_count_weigh_and_average_as_root(
    simple, expr, selection, keywords, entries, sumw, mean
):
    h = simple.draw(expr, selection, bins=keywords.pop("bins", (100, 0, 100)), **keywords)
    assert h.classname == "TH1F" and h.name == "htemp"
    assert h.entries == entries and h.selected == entries
    assert _sumw(h) == sumw
    assert h.mean() == pytest.approx(mean)


def test_y_colon_x_puts_y_on_the_vertical_axis_as_root_does(simple):
    h = simple.draw("y:x", bins=[(10, 0, 100), (10, -100, 0)])
    assert h.classname == "TH2F" and h.entries == 100
    # y = 0 is the upper edge of y's axis, so ROOT - unlike go-hep - leaves entry 0
    # out of the moments of both axes, and the means are those of the other 99.
    assert h.mean(0) == pytest.approx(50) and h.mean(1) == pytest.approx(-50)
    assert [axis.title for axis in h.axes] == ["x", "y"]


def test_z_colon_y_colon_x_fills_a_th3f_with_z_last(simple):
    h = simple.draw("n:y:x", bins=[(10, 0, 100), (10, -100, 1), (10, 0, 100)])
    assert h.classname == "TH3F" and h.entries == 100
    assert h.mean(2) == pytest.approx(49.5)
    assert [axis.title for axis in h.axes] == ["x", "y", "n"]


def test_how_many_parts_an_expression_has_is_how_many_axes_it_fills(simple):
    assert [simple.draw(e).classname for e in ("x", "y:x", "n:y:x")] == ["TH1F", "TH2F", "TH3F"]


@pytest.mark.parametrize(
    ("expr", "selection", "error", "match"),
    [
        ("nosuch", "", FormulaError, "nosuch"),
        ("nosuchfct(x)", "", FormulaError, "nosuchfct"),
        ("x +", "", FormulaError, "could not be parsed"),
        ("x", "x +", FormulaError, "could not be parsed"),
        ("x:y:n:x", "", ValueError, "4 parts"),
        ("x::", "", FormulaError, "could not be parsed"),
        ("x: ", "", ValueError, "empty part"),
    ],
)
def test_what_cannot_be_drawn_is_refused_by_name(simple, expr, selection, error, match):
    with pytest.raises(error, match=match):
        simple.draw(expr, selection)


# -- collections, from small-flat-tree.root -----------------------------------


def test_a_fixed_array_is_drawn_ten_fills_an_entry(flat):
    h = flat.draw("ArrayFloat64", bins=(100, -1, 100))
    assert h.entries == 1000 and _sumw(h) == 1000
    assert h.mean() == pytest.approx(np.repeat(ENTRY, 10).mean())


def test_a_slice_is_drawn_as_many_fills_as_it_has_elements(flat):
    h = flat.draw("SliceFloat64", bins=(100, -1, 100))
    lengths = ENTRY % 10
    assert h.entries == lengths.sum()
    assert h.mean() == pytest.approx((lengths * ENTRY).sum() / lengths.sum())


def test_an_index_draws_one_element_an_entry(flat):
    h = flat.draw("ArrayFloat64[0]", bins=(100, -1, 100))
    assert h.entries == 100 and h.mean() == flat.draw("Float64", bins=(100, -1, 100)).mean()


def test_the_reducers_go_back_to_one_fill_an_entry(flat):
    lengths = ENTRY % 10
    assert flat.draw("Length$(SliceFloat64)").mean() * 100 == pytest.approx(lengths.sum())
    total = flat.draw("Sum$(SliceFloat64)", bins=(100, -1, 1000))
    assert total.entries == 100 and total.mean() * 100 == pytest.approx((lengths * ENTRY).sum())
    peak = flat.draw("Max$(SliceFloat64)", bins=(100, -1, 100))
    assert peak.mean() * 100 == pytest.approx(ENTRY[lengths > 0].sum())


def test_a_cut_over_elements_keeps_the_elements_that_pass(flat):
    h = flat.draw("ArrayFloat64", "ArrayFloat64 > 50", bins=(100, -1, 100))
    assert h.entries == 490 and h.selected == 490


def test_a_cut_over_the_entry_keeps_every_element_of_it(flat):
    h = flat.draw("SliceFloat64", "Length$(SliceFloat64) >= 5", bins=(100, -1, 100))
    lengths = ENTRY % 10
    assert h.entries == lengths[lengths >= 5].sum()
    empty = flat.draw("SliceFloat64", "Length$(SliceFloat64) == 0", bins=(10, -1, 100))
    assert empty.entries == 0 and empty.selected == 0


def test_a_number_per_entry_goes_with_every_element(flat):
    h = flat.draw("ArrayFloat64 - Float64", bins=(21, -10.5, 10.5))
    assert h.entries == 1000 and h.mean() == 0


def test_two_axes_over_one_collection_go_element_by_element(flat):
    h = flat.draw("ArrayFloat32:ArrayFloat64", bins=[(20, -1, 100), (20, -1, 100)])
    assert h.entries == 1000 and h.mean(0) == pytest.approx(h.mean(1))


def test_collections_of_different_lengths_pair_up_to_the_shorter_as_root_pairs_them(flat):
    # go-hep refuses this; ROOT's TTreeFormulaManager runs the loop to the shorter.
    h = flat.draw("SliceFloat64 + ArrayFloat64", bins=(10, 0, 1000))
    assert h.entries == (ENTRY % 10).sum()


def test_an_axis_and_a_cut_over_different_collections_share_one_loop(flat):
    # The slice's elements pair with the array's first i % 10, all of which are i.
    h = flat.draw("SliceFloat64", "ArrayFloat64 > 50", bins=(100, -1, 100))
    lengths = ENTRY % 10
    assert h.entries == lengths[ENTRY > 50].sum()


def test_a_number_per_entry_cut_by_a_collection_is_filled_once_per_passing_element(flat):
    h = flat.draw("Int32", "SliceFloat64 > 50", bins=(100, 0, 100))
    lengths = ENTRY % 10
    assert h.entries == lengths[ENTRY > 50].sum()


def test_entry_and_iteration_say_where_the_loop_has_got_to(flat):
    h = flat.draw("Entry$ - Int32", bins=(3, -1.5, 1.5))
    assert h.entries == 100 and h.mean() == 0
    h = flat.draw("Iteration$", "ArrayFloat64 > -1", bins=(10, -0.5, 9.5))
    assert h.entries == 1000 and h.mean() == pytest.approx(4.5)


def test_a_weight_is_evaluated_for_every_element_too(flat):
    h = flat.draw("ArrayFloat64", weight="2", bins=(100, -1, 100))
    assert h.entries == 1000 and _sumw(h) == 2000
    h = flat.draw("ArrayFloat64", weight="Iteration$", bins=(100, -1, 100))
    assert h.entries == 900 and _sumw(h) == 100 * 45  # the zero weights are not filled


def test_an_index_past_the_end_of_a_collection_fills_nothing(flat):
    h = flat.draw("SliceFloat64[5]", bins=(100, -1, 100))
    assert h.entries == ((ENTRY % 10) > 5).sum()


def test_the_selection_s_value_is_a_weight(flat):
    h = flat.draw("Float64", "N", bins=(100, 0, 100))
    assert h.entries == 90 and _sumw(h) == (ENTRY % 10).sum()
    assert h.mean() == pytest.approx(((ENTRY % 10) * ENTRY).sum() / (ENTRY % 10).sum())
    h = flat.draw("Float64", "N", weight=0.5, bins=(100, 0, 100))
    assert _sumw(h) == (ENTRY % 10).sum() / 2


def test_strings_are_selected_on_but_not_drawn(flat):
    assert flat.draw("Int32", 'Str == "evt-042"').mean() == 42
    with pytest.raises(ValueError, match="strings"):
        flat.draw("Str")


# -- THLimitsFinder: the axes of a draw given none ------------------------------


def test_a_draw_given_no_binning_books_the_axis_root_s_limits_finder_finds(simple):
    # OptimizeLimits(100, 0, 99): widened to 0..108.9, rounded to bins of 2 ending at
    # 108, and then a hundredth of the range past the values: -0.99 to 108.
    h = simple.draw("x")
    assert (h.axes[0].nbins, h.axes[0].low, h.axes[0].high) == (100, -0.99, 108.0)
    assert h.axes[0].title == "x" and h.title == "x"
    assert h.values(flow=True)[[0, -1]].tolist() == [0, 0]


def test_an_integer_branch_gets_bins_a_whole_number_wide(simple):
    # The same, with kIsInteger: widened by 5*99/100 to 103.95, the ends made whole,
    # -1 and 102, and bins of one added until the axis reaches 103.95: 104 of them.
    h = simple.draw("n")
    assert (h.axes[0].nbins, h.axes[0].low, h.axes[0].high) == (104, -1.0, 103.0)
    assert (h.axes[0].high - h.axes[0].low) / h.axes[0].nbins == 1


def test_brackets_giving_only_a_count_still_find_the_ends(simple):
    # 0..108.9 over 50 is 2.178 a bin, rounded up to 5: bins end at 110, brought
    # back to 105 because the last would be empty.
    h = simple.draw("x>>h(50)")
    assert (h.axes[0].nbins, h.axes[0].low, h.axes[0].high) == (50, -0.99, 105.0)


def test_a_two_dimensional_draw_finds_both_axes_with_forty_bins_each(simple):
    # Forty bins over 108.9 is 2.7 a bin, rounded to 5, so x ends at 105 and y at -105.
    h = simple.draw("y:x")
    assert [(a.nbins, a.low, a.high) for a in h.axes] == [(40, -0.99, 105.0), (40, -105.0, 0.99)]


def test_three_dimensions_have_twenty_bins_an_axis(simple):
    h = simple.draw("x:y:x")
    assert [a.nbins for a in h.axes] == [20, 20, 20]


def test_one_axis_given_and_another_not_finds_them_both_as_root_does(simple):
    # A two-dimensional '>>h(10,0,100)' leaves y's ends at zero, so the histogram
    # can extend and FindGoodLimitsXY finds both axes, the ones given too.
    # Ten bins of x are 10.89 wide, rounded to 20, ending at 120 and brought back to 100.
    h = simple.draw("y:x>>h(10,0,100)")
    assert [(a.nbins, a.low, a.high) for a in h.axes] == [(10, -0.99, 100.0), (40, -105.0, 0.99)]


def test_the_ends_are_found_over_elements_not_entries(flat):
    h = flat.draw("ArrayFloat64")
    assert h.entries == 1000 and h.axes[0].low <= 0 and h.axes[0].high > 99


def test_counts_root_keeps_are_whole_numbers_and_arithmetic_is_not(flat):
    from xrdroot import compile_formula
    from xrdroot.formula.select import TreeNames

    def compiled(text):
        return compile_formula(text, TreeNames(flat))

    whole = ["Int32", "Entry$", "Length$(SliceFloat64)", "Alt$(Int32, 0)", "SliceInt32[0]",
             "SliceFloat64.size()"]
    for text in whole:
        assert is_integer(compiled(text), flat), text
    for text in ["Float64", "Int32 + 1", "(int)Float64", "Sum$(SliceInt32)", "Str"]:
        assert not is_integer(compiled(text), flat), text


def test_nothing_selected_books_the_axis_the_limits_finder_makes_of_nothing(simple):
    h = simple.draw("x", "x > 1000")
    assert (h.axes[0].nbins, h.axes[0].low, h.axes[0].high) == (100, -1.0, 1.0)
    assert h.entries == 0 and h.selected == 0
    assert simple.draw("n", "x > 1000").axes[0].nbins == 2


# -- past the estimate: the axis doubles -----------------------------------------


@pytest.fixture(scope="module")
def long(tmp_path_factory):
    """Two thousand entries of x = the entry number, and one NaN at the end."""
    path = tmp_path_factory.mktemp("draw") / "long.root"
    x = np.arange(2000.0)
    with create(str(path)) as f:
        f["tree"] = {"x": x, "w": np.where(x == 1999, np.nan, x)}
    with open_root(str(path)) as f:
        yield f["tree"]


def test_values_past_the_estimate_double_the_axis_until_it_holds_them(long):
    # The first thousand find -9.99 to 1080; 1080 itself doubles the range once,
    # to -9.99 + 2 * 1089.99 = 2169.99, which holds everything after.
    h = long.draw("x", estimate=1000, step=300)
    axis = h.axes[0]
    assert (axis.nbins, axis.low) == (100, -9.99)
    assert axis.high == pytest.approx(2169.99)
    assert h.entries == 2000 and h.values(flow=True)[[0, -1]].tolist() == [0, 0]
    fresh = Histogram.book("fresh", (100, axis.low, axis.high), kind="F")
    fresh.fill(np.arange(2000.0))
    assert h.values().tolist() == fresh.values().tolist()
    assert h.mean() == pytest.approx(999.5) and h.extendable


def test_a_value_below_the_axis_doubles_it_downwards(long):
    h = long.draw("1000 - x", estimate=1000)
    assert h.axes[0].low < -999 and h.values(flow=True)[0] == 0 and h.entries == 2000


def test_a_nan_stops_a_histogram_extending_and_goes_to_the_overflow(long):
    h = long.draw("w", "x >= 1998 || x < 10", estimate=10)
    assert not h.extendable and h.values(flow=True)[-1] == 1  # the NaN
    assert h.axes[0].high > 1998  # 1998 came first and was taken in
    after = long.draw("x < 1998 ? x : (x == 1998 ? sqrt(-1) : 5000)", "x < 10 || x >= 1998",
                      estimate=10)
    assert after.axes[0].high == 9.9 and after.values(flow=True)[-1] == 2  # NaN, then 5000


def test_an_infinity_is_never_reached_and_goes_to_the_flow_bin(long):
    h = long.draw("x < 10 ? x : 1/(x - x)", "x < 12", estimate=10)
    assert h.values(flow=True)[-1] == 2 and h.extendable


def test_a_profile_extends_too_and_a_nan_does_not_stop_it(long):
    p = long.draw("x:w", "x < 10 || x > 1990", "prof", estimate=10)
    assert isinstance(p, Profile) and p.extendable
    assert p.axes[0].high > 1990 and p.entries == 19


def test_a_histogram_that_extended_can_be_drawn_into_and_extends_again(long):
    kept = {}
    long.draw("x>>h", "x < 100", histograms=kept, estimate=10)
    first = kept["h"].axes[0].high
    again = long.draw("x>>+h", "x >= 100", histograms=kept)
    assert again is kept["h"] and again.axes[0].high > first and again.entries == 2000


def test_flow_on_an_extending_axis_is_lost_as_root_warns(long):
    kept = {}
    long.draw("x>>h", "x < 20", histograms=kept, estimate=5)
    h = kept["h"]
    from xrdroot.extending import extend_axis, new_limits

    cells = h._cells()
    cells[0] = 3.0  # something in the underflow, as a fill with no extending would put
    assert extend_axis(h, 0, 1e6) and h.values(flow=True)[0] == 0
    assert new_limits(0.0, 1.0, 2.5) == (0.0, 4.0)
    assert new_limits(0.0, 1.0, float("inf")) is None and new_limits(1.0, 1.0, 5) is None
    assert not extend_axis(h, 0, float("inf"))


# -- profiles --------------------------------------------------------------------


def test_a_profile_holds_the_mean_of_y_in_bins_of_x(flat):
    p = flat.draw("Float64:Float64", "", "prof", bins=(10, 0, 100))
    assert p.classname == "TProfile" and p.entries == 100
    assert p.values().tolist() == [10 * k + 4.5 for k in range(10)]
    assert p.bin_entries().tolist() == [10] * 10
    spread = np.sqrt(np.mean((np.arange(10) - 4.5) ** 2))  # ROOT's spread has no Bessel
    assert p.errors() == pytest.approx(np.full(10, spread / np.sqrt(10)))
    s = flat.draw("Float64:Float64", "", "profs", bins=(10, 0, 100))
    assert s.error_mode == "s" and s.errors() == pytest.approx(np.full(10, spread))
    assert flat.draw("Float64:Float64", "", "profi").error_mode == "i"
    assert flat.draw("Float64:Float64", "", "profg").error_mode == "g"


def test_a_profile_over_collections_goes_element_by_element(flat):
    p = flat.draw("ArrayFloat32:ArrayFloat64", "", "prof", bins=(10, 0, 100))
    assert p.entries == 1000
    assert p.values() == pytest.approx([10 * k + 4.5 for k in range(10)])


def test_a_profile_finds_its_axis_with_a_hundred_bins(flat):
    p = flat.draw("Float64:Float64", "", "prof")
    assert p.entries == 100 and p.axes[0].nbins == 100 and p.axes[0].low <= 0
    assert p.axes[0].title == "" and p.title == "Float64:Float64"


def test_a_profile_takes_the_cut_and_the_weight(flat):
    p = flat.draw("Float64:Float64", "Float64 >= 50", "prof", bins=(10, 0, 100))
    assert p.entries == 50 and p.bin_entries().tolist() == [0] * 5 + [10] * 5


def test_prof_on_one_expression_is_ignored_as_root_ignores_it(flat):
    assert flat.draw("Float64", "", "prof").classname == "TH1F"


def test_three_parts_and_prof_make_a_tprofile2d(flat):
    p = flat.draw("Float64:Float64:Float64", "", "prof", bins=[(10, 0, 100), (10, 0, 100)])
    assert p.classname == "TProfile2D" and p.entries == 100
    means = p.values()
    assert np.diag(means).tolist() == [10 * k + 4.5 for k in range(10)]
    assert p.bin_entries().sum() == np.trace(p.bin_entries())
    auto = flat.draw("Float64:Float64:Float64", "", "prof")
    assert [a.nbins for a in auto.axes] == [20, 20]


# -- graphs ------------------------------------------------------------------------


def test_points_or_lines_make_the_scatter_root_draws_a_graph_of(flat):
    g = flat.draw("Float64:Int64", "", "p")
    assert isinstance(g, Graph) and g.name == "Graph" and g.title == "Graph"
    assert g.x.tolist() == list(range(100)) and g.y.tolist() == list(range(100))
    assert g.selected == 100
    cut = flat.draw("Float64:Int64", "Int64 < 5", "l", title="first five")
    assert len(cut) == 5 and cut.title == "first five"
    assert flat.draw("Float64:Int64", "", "col").classname == "TH2F"
    assert len(flat.draw("Float64:Int64", "Int64 < 0", "*")) == 0


def test_a_named_scatter_fills_the_histogram_it_names(flat):
    assert flat.draw("Float64:Int64>>h", "", "p").classname == "TH2F"
    g = flat.draw("Float64:Int64", "", "p", name="scatter", title=None)
    assert g.name == "scatter" and g.title == "Float64:Int64"
    with pytest.raises(ValueError, match="no bins"):
        flat.draw("Float64:Int64", "", "p", bins=[10, 10])


# -- >>name, >>+name and histograms= ----------------------------------------------


def test_brackets_after_the_name_bin_every_axis(simple):
    h = simple.draw("y:x>>h2(10,0,100,20,-100,0)")
    assert h.name == "h2" and h.title == "y:x"
    assert [(a.nbins, a.low, a.high) for a in h.axes] == [(10, 0, 100), (20, -100, 0)]
    assert [a.title for a in h.axes] == ["", ""]  # a named histogram keeps no axis titles
    assert not h.extendable


def test_brackets_leave_out_what_they_do_not_give(simple):
    h = simple.draw("x>>h(, 0, 50)")
    assert (h.axes[0].nbins, h.axes[0].low, h.axes[0].high) == (100, 0, 50)
    assert h.values(flow=True)[-1] == 50


def test_the_histograms_dict_is_gdirectory(simple):
    kept = {}
    first = simple.draw("x>>h(10,0,100)", "x < 50", histograms=kept)
    assert kept["h"] is first and first.entries == 50
    added = simple.draw("x>>+h", "x >= 50", histograms=kept)
    assert added is first and first.entries == 100 and added.selected == 50
    again = simple.draw("x>>h", "x < 10", histograms=kept)
    assert again is first and first.entries == 10  # '>>h' starts it again
    rebinned = simple.draw("x>>h(5,0,100)", histograms=kept)
    assert rebinned is not first and kept["h"] is rebinned
    simple.draw("x", histograms=kept)
    assert kept["htemp"].entries == 100


def test_a_histogram_from_elsewhere_is_filled_and_its_entries_say_what_was_selected(simple):
    mine = Histogram.book("mine", (10, 0, 100))
    kept = {"mine": mine}
    assert simple.draw("x>>+mine", histograms=kept) is mine and mine.entries == 100
    assert not isinstance(mine, DrawnHistogram)
    profile = Profile.book("p", (10, 0, 100))
    simple.draw("y:x>>+p", histograms={"p": profile})
    assert profile.entries == 100 and profile.values()[0] == -4.5


@pytest.mark.parametrize(
    ("varexp", "keywords", "error", "match"),
    [
        (">>elist", {}, ValueError, "entry list"),
        ("x>>+missing", {}, KeyError, "none"),
        ("x>>h(10,0", {}, ValueError, "one pair of brackets"),
        ("x>>h(1)(2)", {}, ValueError, "one pair of brackets"),
        ("x>>h)1(", {}, ValueError, "one pair of brackets"),
        ("x>>h(10,0,1,5)", {}, ValueError, "4 numbers for 1 axis"),
        ("y:x>>h(1,2,3,4,5,6,7)", {}, ValueError, "7 numbers for 2 axes"),
        ("x>>h(ten)", {}, ValueError, "not a number"),
        ("x>>", {}, ValueError, "names no histogram"),
        ("x>>h(10,0,1)", {"bins": 10}, ValueError, "twice"),
        ("x>>h", {"name": "other"}, ValueError, "twice"),
        ("x>>+thing", {"histograms": {"thing": 3}}, TypeError, "not a histogram"),
        ("y:x>>+h1", {"histograms": {"h1": Histogram.book("h1", (1, 0, 1))}}, ValueError, "TH1D"),
        ("x>>+p", {"histograms": {"p": Profile.book("p", (1, 0, 1))}}, ValueError, "profile of 0"),
        ("x>>+h1", {"histograms": {"h1": Histogram.book("h1", (1, 0, 1))}, "bins": 3},
         ValueError, "already binned"),
        ("y:x>>+h2", {"histograms": {"h2": Histogram.book("h2", (1, 0, 1), (1, 0, 1))},
                      "option": "prof"}, ValueError, "fills a profile"),
    ],
)
def test_what_follows_the_arrows_is_refused_when_it_cannot_be_meant(
    simple, varexp, keywords, error, match
):
    with pytest.raises(error, match=match):
        simple.draw(varexp, **keywords)


# -- bins=, titles and names ----------------------------------------------------------


def test_bins_takes_counts_ends_and_edges_for_each_axis(simple):
    assert simple.draw("x", bins=50).axes[0].nbins == 50
    edges = simple.draw("x", bins=[0, 10, 50, 100]).axes[0]
    assert edges.edges().tolist() == [0, 10, 50, 100] and not edges.even
    h = simple.draw("y:x", bins=[None, (5, -100, 0)])
    assert [a.nbins for a in h.axes] == [40, 5]  # an axis left to be found finds both
    both = simple.draw("y:x", bins=[np.array([0.0, 50, 100]), (5, -100, 0)])
    assert [a.nbins for a in both.axes] == [2, 5] and both.entries == 100


@pytest.mark.parametrize(
    ("bins", "match"),
    [([10], "1 axes for a draw that bins along 2"), ([[0, 1, 2], None], "edges for one axis")],
)
def test_bins_that_do_not_fit_the_draw_are_refused(simple, bins, match):
    with pytest.raises(ValueError, match=match):
        simple.draw("y:x", bins=bins)


def test_the_title_is_the_expression_and_the_selection_in_braces(simple):
    h = simple.draw("x", "x >= 50")
    assert h.title == "x {x >= 50}" and h.axes[0].title == "x"
    assert simple.draw("x", "1").title == "x"  # ROOT adds a selection of one character never
    named = simple.draw("x", name="pt", title="p_{T};GeV;events")
    assert named.name == "pt" and named.title == "p_{T}"
    assert [named.axes[0].title] == ["GeV"]


# -- options -------------------------------------------------------------------------


def test_option_e_keeps_the_squares_of_the_weights_from_the_start(simple):
    assert simple.draw("x", "", "e")._sumw2() is not None
    assert simple.draw("x")._sumw2() is None


def test_option_norm_scales_to_a_sum_of_weights_of_one(simple):
    h = simple.draw("x", "", "norm", bins=(10, 0, 100))
    assert h.values().sum() == pytest.approx(1.0)
    empty = simple.draw("x", "x > 1000", "norm", bins=(10, 0, 100))
    assert empty.values().sum() == 0
    with pytest.raises(ValueError, match="means"):
        simple.draw("y:x", "", "prof norm")


def test_goff_and_same_change_nothing_and_rarer_pictures_are_refused(simple):
    plain = simple.draw("x").values().tolist()
    assert simple.draw("x", "", "goff").values().tolist() == plain
    assert simple.draw("x", "", "SAME").values().tolist() == plain
    for option in ("para", "candle", "gl5d", "entrylist"):
        with pytest.raises(ValueError, match=option):
            simple.draw("y:x", "", option)


def test_an_ax_is_drawn_on(simple):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    ax = pyplot.subplots()[1]
    h = simple.draw("x", ax=ax)
    assert isinstance(h, Histogram) and ax.patches
    pyplot.close("all")


def test_the_range_and_the_batch_are_checked(simple):
    with pytest.raises(ValueError, match="negative"):
        simple.draw("x", first_entry=-1)
    with pytest.raises(ValueError, match="negative"):
        simple.draw("x", entries=-1)
    with pytest.raises(ValueError, match="at least one"):
        simple.draw("x", step=0)
    assert simple.draw("x", step=7).values().tolist() == simple.draw("x").values().tolist()


def test_a_copy_of_what_was_drawn_is_still_a_drawn_histogram(simple):
    h = simple.draw("x")
    assert isinstance(h.copy(), DrawnHistogram) and h.copy().selected == 0


# -- chains and friends -----------------------------------------------------------------


def test_a_chain_draws_across_its_files():
    files = [str(DATA / "chain.flat.1.root"), str(DATA / "chain.flat.2.root")]
    with chain("tree", files) as events:
        h = events.draw("Entry$", "LocalEntry$ < 3")
        assert h.entries == 6 and h.mean() == pytest.approx((0 + 1 + 2 + 5 + 6 + 7) / 6)
        assert h.axes[0].nbins != 100  # Entry$ is a whole number, and binned as one


def test_a_friend_s_columns_are_drawn_beside_the_tree_s_own():
    with open_root(str(DATA / "join1.root")) as one, open_root(str(DATA / "join2.root")) as two:
        tree = one["j1"]
        tree.add_friend(two["j2"], "second")
        h = tree.draw("second.b20 - b10", bins=(10, 0, 200))
        assert h.mean() == 100 and h.entries == len(tree)


# -- THLimitsFinder, statement by statement ------------------------------------------


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        # 1000 over 100 is exactly 10 a bin, which is not rounded past.
        ((0, 1000, 100), (0.0, 1000.0, 100, 10.0)),
        # A range of NaN, and one too small to take the logarithm of, fall back to 0..1.
        ((float("nan"), float("nan"), 100), (0.0, 1.0, 100, 0.01)),
        ((0, 1e-248, 100), (0.0, 1.0, 100, 0.01)),
        # Below zero both ends count down from it.
        ((-100, -50, 100), (-100.0, -50.0, 100, 0.5)),
        # Bins a billion widths from zero are not placed: the ends are kept.
        ((1e12, 1e12 + 1, 100), (1e12, 1e12 + 1, 100, 0.01)),
        # Five bins for ten asked for is tried again with eleven, which makes ten of 2.
        ((0, 20.5, 10), (0.0, 20.0, 10, 2.0)),
        # A range of nothing is made one wide.
        ((5, 5, 10), (5.0, 6.000000000000001, 10, 0.1)),
        # One bin asked for is ROOT's hard case: double the width, and settle.
        ((0, 1, 1), (0.0, 0.5, 0, 1.0)),
        ((0, 1, 3), (0.0, 1.0, 2, 0.5)),
    ],
)
def test_optimize_rounds_as_root_s_optimize_does(args, expected):
    from xrdroot.limits import optimize

    assert optimize(*args) == expected


def test_the_limits_of_whole_numbers_below_zero_end_at_one(simple):
    from xrdroot.limits import Limits, find_good_limits, good_axes

    # Widened by 5*50/2 to -225..0, rounded to -200..0, made whole to -200..1 with
    # bins of 100, and one more bin added below because -200 is above -225.
    assert find_good_limits(2, -100, -50, True) == Limits(3, -299.0, 1.0)
    assert good_axes([10, 20], [(0, 1), (5, 5)], [False, True]) == [
        Limits(10, -0.01, 1.01),
        Limits(4, 3.0, 7.0),
    ]


def test_a_draw_of_no_entries_still_books_its_axes(simple, flat):
    assert simple.draw("x", entries=0).axes[0].nbins == 100
    assert len(flat.draw("Float64:Int64", "", "p", entries=0)) == 0


def test_a_second_axis_extends_while_the_first_stays(long):
    h = long.draw("x:1", estimate=10)
    assert h.axes[0].high == 2.2 and h.axes[1].high > 1999 and h.entries == 2000


def test_a_limit_is_not_found_below_an_infinity():
    from xrdroot.extending import new_limits

    assert new_limits(0.0, 1.0, float("-inf")) is None
