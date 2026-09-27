"""RooFit's unbinned datasets - filled, imported, cut, reduced and printed - held to ROOT 6.40.

The printed text, the counts and the sums below were printed by ROOT itself
through PyROOT, for the same datasets built with ROOT's classes; where the
engine is known to part from ROOT the test says so and pins only what the two
share.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from xrdroot.roofit.categories import RooCategory, RooThresholdCategory
from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.data.dataset import RooDataSet, as_set
from xrdroot.roofit.data.selection import in_range, selected
from xrdroot.roofit.data.store import RooAbsData, is_arg, value_text
from xrdroot.roofit.functions import RooFormulaVar
from xrdroot.roofit.messages import service
from xrdroot.roofit.variables import RooRealVar


@pytest.fixture(autouse=True)
def _messages() -> Iterator[None]:
    """Every test starts from the message streams RooFit starts with."""
    service().reset()
    yield
    service().reset()


class Tree:
    """What a dataset imports from: named columns and a length, as a ``TTree`` offers them."""

    def __init__(self, **columns: list[float]) -> None:
        self._columns = {k: np.asarray(v, dtype=np.float64) for k, v in columns.items()}

    def arrays(self, names: list[str]) -> dict[str, Any]:
        return {name: self._columns[name] for name in names}

    def __len__(self) -> int:
        return len(next(iter(self._columns.values())))


def xy() -> tuple[RooRealVar, RooRealVar]:
    return RooRealVar("x", "x", 0, 10), RooRealVar("y", "y", -5, 5)


def filled() -> tuple[RooDataSet, RooRealVar, RooRealVar, RooCategory]:
    """Ten events of ``x``, ``y`` and a category ``c`` taking ``A`` and ``B`` in turn."""
    x, y = xy()
    c = RooCategory("c", "c")
    c.defineType("A", 0)
    c.defineType("B", 1)
    d = RooDataSet("d", "d title", RooArgSet([x, y, c]))
    for i in range(10):
        x.setVal(i + 0.5)
        y.setVal(i - 4.5)
        c.setIndex(i % 2)
        d.add(RooArgSet([x, y, c]))
    return d, x, y, c


def tree() -> Tree:
    """Twelve events, the first six of them outside ``x``'s range ``[0, 10]``."""
    return Tree(
        x=[i - 6.0 for i in range(12)],
        y=[0.5 * i - 2 for i in range(12)],
        w=[0.5 + i for i in range(12)],
    )


def test_a_filled_dataset_prints_its_line_and_its_store_as_root_does(capsys: Any) -> None:
    """``Print`` and ``Print("v")`` are how a macro shows a dataset: the text is ROOT's."""
    d, *_ = filled()
    d.get(9)
    d.Print()
    d.Print("v")
    assert capsys.readouterr().out == (
        "RooDataSet::d[x,y,c] = 10 entries\n"
        "DataStore d (d title)\n"
        "  Contains 10 entries\n"
        "  Observables: \n"
        '    1)  x = 9.5  L(0 - 10)  "x"\n'
        '    2)  y = 4.5  L(-5 - 5)  "y"\n'
        "    3)  c = B(idx = 1)\n"
        '  "c"\n'
    )


def test_the_standard_print_of_a_store_lists_its_observables_on_one_line(capsys: Any) -> None:
    """``Print("s")`` is the short form of the store: its observables' names in one line."""
    d, *_ = filled()
    d.get(0)
    d.Print("s")
    assert capsys.readouterr().out == (
        "DataStore d (d title)\n  Contains 10 entries\n  Observables (x,y,c)\n"
    )


def test_a_dataset_answers_what_a_tobject_is_asked() -> None:
    """Names, titles, class and inheritance are what a macro and the plotting code ask first."""
    d, *_ = filled()
    d.SetName("renamed")
    d.SetTitle("new title")
    assert (d.GetName(), d.GetTitle(), d.ClassName()) == ("renamed", "new title", "RooDataSet")
    assert (d.printName(), d.printTitle(), d.printClassName()) == (
        "renamed",
        "new title",
        "RooDataSet",
    )
    assert d.InheritsFrom("RooAbsData") and d.InheritsFrom(RooAbsData)
    assert not d.InheritsFrom("RooDataHist")
    assert (len(d), repr(d)) == (10, "<RooDataSet::renamed 10 entries>")


def test_an_event_is_read_back_by_setting_the_datasets_own_variables() -> None:
    """``get(i)`` loads the event into the copies, leaving the caller's variables alone."""
    d, x, *_ = filled()
    x.setVal(3.0)
    row = d.get(4)
    assert (row.find("x").getVal(), row.find("y").getVal(), row.find("c").getIndex()) == (
        4.5,
        -0.5,
        0,
    )
    assert x.getVal() == 3.0
    assert d.get() is row
    assert (d.weight(), d.weightSquared()) == (1.0, 1.0)
    d.get(99)  # past the end: the copies keep the last event loaded
    assert row.find("x").getVal() == 4.5


def test_an_empty_dataset_weighs_one_and_has_no_entries() -> None:
    """Asking an empty dataset for its event's weight is not an error; ROOT answers one."""
    x, _ = xy()
    d = RooDataSet("e", "e", x)
    assert (d.numEntries(), d.weight(), d.weightSquared(), d.sumEntries()) == (0, 1.0, 1.0, 0.0)
    assert RooAbsData("bare").numEntries() == 0


def test_the_datasets_columns_are_its_events_values() -> None:
    """The columns are what every likelihood and plot reads, so they must be the events."""
    d, *_ = filled()
    assert list(d.column("y")[:3]) == [-4.5, -3.5, -2.5]
    assert sorted(d.columns()) == ["c", "x", "y"]
    assert list(d.column("c")) == [0.0, 1.0] * 5
    assert not d.isWeighted() and not d.isNonPoissonWeighted()
    assert (d.sumEntriesW2(), list(d.weights_squared())) == (10.0, [1.0] * 10)


def test_the_datasets_variable_can_be_found_by_name() -> None:
    """``variable`` hands back the dataset's own copy, not the caller's variable."""
    d, x, *_ = filled()
    assert d.variable("x").GetName() == "x" and d.variable("x") is not x
    assert d.variable("nothing") is None


def test_a_tree_import_skips_what_the_variables_cannot_hold_saying_so_as_root_does(
    capsys: Any,
) -> None:
    """Out-of-range events are dropped with ROOT's words: four named, then ``...``, then a count."""
    x, y = xy()
    w = RooRealVar("w", "w", 0, 100)
    d = RooDataSet("d", "d", RooArgSet([x, y, w]), Import=tree(), WeightVar="w")
    lines = [
        f"[#1] INFO:DataHandling -- RooTreeDataStore::loadValues(d) Skipping event #{i} because x "
        f"cannot accommodate the value {i - 6}\n"
        for i in range(4)
    ]
    assert capsys.readouterr().out == "".join(lines) + (
        "[#1] INFO:DataHandling -- RooTreeDataStore::loadValues(d) Skipping ...\n"
        "[#0] WARNING:DataHandling -- RooTreeDataStore::loadValues(d) Ignored 6 out-of-range "
        "events\n"
    )
    d.get(0)
    d.Print()
    d.Print("v")
    assert capsys.readouterr().out == (
        "RooDataSet::d[x,y,weight:w] = 6 entries (54 weighted)\n"
        "DataStore d (d)\n"
        "  Contains 6 entries\n"
        "  Observables: \n"
        '    1)  x = 0  L(0 - 10)  "x"\n'
        '    2)  y = 1  L(-5 - 5)  "y"\n'
        '  Dataset variable "w" is interpreted as the event weight\n'
    )
    assert (d.numEntries(), d.sumEntries(), d.sumEntriesW2()) == (6, 54.0, 503.5)
    assert d.isWeighted() and d.isNonPoissonWeighted()


def test_sums_of_weights_honour_a_cut_a_named_range_and_both() -> None:
    """``sumEntries(cut, range)`` is how yields in a region are counted: ROOT's sums."""
    x, y = xy()
    w = RooRealVar("w", "w", 0, 100)
    d = RooDataSet("d", "d", RooArgSet([x, y, w]), Import=tree(), WeightVar=w)
    x.setRange("lo", 0, 4)
    assert (d.sumEntries("y>0"), d.sumEntries("", "lo"), d.sumEntries("y>0", "lo")) == (
        54.0,
        42.5,
        42.5,
    )
    assert d.sumEntries(None, ",lo,") == 42.5


def test_a_cut_and_a_cut_range_at_construction_keep_only_the_selected_events() -> None:
    """``Import(tree), Cut(...)`` and ``Import(data), CutRange(...)`` select as ROOT selects."""
    x, y = xy()
    d2 = RooDataSet("d2", "d2", RooArgSet([x, y]), Import=tree(), Cut="y>0")
    x.setRange("lo", 0, 4)
    d4 = RooDataSet("d4", "d4", RooArgSet([x, y]), Import=d2, CutRange="lo")
    assert (d2.numEntries(), d4.numEntries()) == (6, 5)
    assert list(d4.column("x")) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_a_weighted_dataset_imported_keeps_its_weights() -> None:
    """Importing weighted data, with or without naming the weight, keeps each event's weight."""
    x = RooRealVar("x", "x", 0, 10)
    w = RooRealVar("w", "w", 0, 10)
    d = RooDataSet("d", "d", RooArgSet([x, w]), WeightVar="w")
    for value, weight in ((1, 2.0), (2, 0.5), (3, 1.5)):
        x.setVal(value)
        d.add(RooArgSet([x]), weight)
    u = RooDataSet("u", "u", x, Import=d)
    kept = RooDataSet("k", "k", RooArgSet([x, w]), Import=d, WeightVar="w")
    assert (u.isWeighted(), u.sumEntries(), list(u.weights())) == (True, 4.0, [2.0, 0.5, 1.5])
    assert (kept.sumEntries(), kept.printArgs()) == (4.0, "[x,weight:w]")


def test_a_weight_variable_the_dataset_was_not_given_is_made_for_it() -> None:
    """``WeightVar("ww")`` naming nothing among the variables still gives weighted data."""
    x = RooRealVar("x", "x", 0, 10)
    nw = RooDataSet("nw", "nw", RooArgSet([x]), RooCmdArg("WeightVar", "ww"))
    x.setVal(1)
    nw.add(RooArgSet([x]), 4.0)
    assert (nw.printArgs(), nw.printValue()) == ("[x,weight:ww]", "1 entries (4 weighted)")
    assert not nw.isNonPoissonWeighted()


def test_a_weight_given_to_unweighted_data_is_ignored(capsys: Any) -> None:
    """ROOT ignores the weight - and says so - when the dataset has no weight variable."""
    x, y = xy()
    d = RooDataSet("d", "d", x, y)
    d.add(RooArgSet([x, y]), 2.0)
    d.add(RooArgSet([x, y]), 1.0, 0.5)
    assert (d.numEntries(), d.sumEntries(), d.isWeighted()) == (2, 2.0, False)
    assert capsys.readouterr().out.count("The weight will be ignored.") == 2


def test_an_event_added_from_a_row_missing_a_variable_takes_the_datasets_own_value() -> None:
    """A row need not hold every variable: the missing ones keep the copy's current value."""
    x, y = xy()
    d = RooDataSet("d", "d", RooArgSet([x, y]))
    x.setVal(2.0)
    d.add(RooArgSet([x]))
    d.add(object())
    assert (d.column("x")[0], list(d.column("y"))) == (2.0, [0.0, 0.0])


def test_appending_and_adding_columns_grow_the_events_and_their_weights() -> None:
    """``append`` puts another dataset's events after these, weighing one if they had none."""
    x = RooRealVar("x", "x", 0, 10)
    w = RooRealVar("w", "w", 0, 10)
    d = RooDataSet("d", "d", RooArgSet([x, w]), WeightVar="w")
    x.setVal(1)
    d.add(RooArgSet([x]), 3.0)
    a = RooDataSet("a", "a", x)
    a.add_columns({"x": [5.0, 6.0]})
    b = RooDataSet("b", "b", x)
    b.append(a)
    d.append(a)
    assert (a.isWeighted(), b.numEntries(), list(b.column("x"))) == (False, 2, [5.0, 6.0])
    assert (d.numEntries(), list(d.weights())) == (3, [3.0, 1.0, 1.0])
    b.add_columns({"x": [7.0]}, [2.5])
    assert list(b.weights()) == [1.0, 1.0, 2.5]
