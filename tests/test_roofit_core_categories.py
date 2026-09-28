"""``RooCategory`` and the categories computed from other values, held to ROOT 6.40.04.

The messages, indices and labels are what ROOT printed for the same
calls. Where the engine is known to part from ROOT, the test that holds it
to ROOT is marked ``xfail(strict=True)``, so that it turns red - to be
unmarked - once the engine is fixed.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.roofit.categories import (
    RooBinningCategory,
    RooCategory,
    RooMappedCategory,
    RooMultiCategory,
    RooSuperCategory,
    RooThresholdCategory,
)
from xrdroot.roofit.variables import RooRealVar


def signs() -> RooCategory:
    c = RooCategory("c", "c")
    c.defineType("Plus", 1)
    c.defineType("Minus", 2)
    c.defineType("Zero", 0)
    return c


def test_a_state_without_an_index_takes_one_more_than_the_largest() -> None:
    """``defineType("C")`` after ``A`` and ``B = 5`` is 6: ``nextAvailableStateIndex``."""
    cat = RooCategory("cat", "cat")
    assert cat.defineType("A") is False
    cat.defineType("B", 5)
    cat.defineType("C")
    assert cat.states() == {"A": 0, "B": 5, "C": 6}
    assert (cat.size(), cat.numTypes(), len(cat)) == (3, 3, 3)
    assert list(cat) == [("A", 0), ("B", 5), ("C", 6)]
    assert cat.isCategory()


def test_a_category_prints_its_label_and_index_as_root_does(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ROOT's ``Print()`` of a category is ``label(idx = n)`` and a blank line."""
    cat = RooCategory("cat", "cat")
    cat.defineTypes(["A", "B"])
    cat.Print()
    assert capsys.readouterr().out == "RooCategory::cat = A(idx = 0)\n\n"


def test_a_bad_label_or_a_taken_index_or_label_is_refused_with_roots_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A semicolon, an index already used and a label already used: ROOT's three errors."""
    cat = RooCategory("cat", "cat")
    cat.defineType("A")
    cat.defineType("B", 5)
    assert cat.defineType("A;b") is True
    assert cat.defineType("D", 5) is True
    assert cat.defineType("A", 9) is True
    assert capsys.readouterr().out == (
        "[#0] ERROR:InputArguments -- RooCategory::defineType(cat): semicolons not allowed "
        "in label name\n"
        "[#0] ERROR:InputArguments -- RooAbsCategory::defineState(cat): index 5 already "
        "assigned\n"
        "[#0] ERROR:InputArguments -- RooAbsCategory::defineState(cat): label A already "
        "assigned or not allowed\n"
    )


def test_setting_an_unknown_state_is_an_error_and_leaves_the_state_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ROOT's errors for ``setIndex(7)`` and ``setLabel("Q")``; ``printError=false`` is silent."""
    cat = RooCategory("cat", "cat", {"A": 0, "B": 5})
    assert cat.setIndex(7) is True
    assert cat.setLabel("Q") is True
    assert cat.setIndex(7, False) is True
    assert cat.setLabel("Q", False) is True
    assert (cat.lookupIndex("Z"), cat.getIndex(), cat.getLabel()) == (-2147483648, 0, "A")
    assert capsys.readouterr().out == (
        "[#0] ERROR:InputArguments -- RooCategory: Trying to set invalid state 7 for "
        "category cat\n"
        "[#0] ERROR:InputArguments -- Trying to set invalid state label 'Q' for category cat\n"
    )


def test_a_state_is_set_by_index_or_label_and_read_back_either_way() -> None:
    """The index is what a dataset keeps and a formula compares; the label what people read."""
    cat = RooCategory("cat", "cat")
    assert (cat.getIndex(), cat.getLabel(), cat.size()) == (0, "", 0)
    cat.defineType("b", 3)
    cat.defineType("a")
    assert (cat.getIndex(), cat.lookupName(4), cat.lookupName(99)) == (3, "a", "")
    assert (cat.hasLabel("a"), cat.hasIndex(3), cat.hasIndex(9)) == (True, True, False)
    assert cat.setLabel("a") is False
    assert (cat.getCurrentIndex(), cat.getCurrentLabel(), cat.getVal()) == (4, "a", 4.0)
    assert cat.setIndex(3) is False
    assert cat.getLabel() == "b"
    cat["c"] = 10
    assert cat["c"] == 10
    assert (cat.isFundamental(), cat.isDerived()) == (True, False)


def test_a_category_reads_its_column_when_evaluated_over_data() -> None:
    """``compute`` takes the category's column from the context, else its own index."""
    cat = signs()
    assert cat.compute({}) == 1.0
    assert list(cat.compute({"c": np.array([0.0, 2.0])})) == [0.0, 2.0]


def test_a_category_can_be_made_constant() -> None:
    """A constant category is left out of what a fit or a generator varies."""
    cat = signs()
    cat.setConstant()
    assert cat.isConstant()
    cat.setConstant(False)
    assert not cat.isConstant()


def test_a_named_range_of_a_category_is_a_list_of_its_labels() -> None:
    """ROOT's ``setRange("R", "Plus,Zero")`` holds Plus, not Minus; ``addToRange`` adds Minus."""
    c = signs()
    c.setRange("R", "Plus,Zero")
    c.setLabel("Plus")
    assert c.inRange("R")
    c.setLabel("Minus")
    assert not c.inRange("R")
    c.addToRange("R", "Minus")
    assert (c.inRange("R"), c.hasRange("R"), c.hasRange("S")) == (True, True, False)
    assert c.range_indices("R") == [1, 0, 2]
    assert c.range_indices("S") == []
    assert c.inRange("S")


def test_a_dataset_keeps_a_categorys_index_and_shows_it_with_its_label() -> None:
    """A data store keeps the index, checks it is a state, and prints ``label(index)``."""
    c = signs()
    c.load_value(2)
    assert (c.stored_value(), c.getLabel()) == (2.0, "Minus")
    assert c.can_hold(0)
    assert not c.can_hold(7)
    assert c.value_text(1.0) == "Plus(1)"
    other = signs()
    other.copy_value_from(c)
    assert other.getLabel() == "Minus"


def test_a_cloned_category_has_its_own_states() -> None:
    """ROOT's clone may be given a state the original does not get."""
    c = signs()
    c.setLabel("Zero")
    copy = c.clone("c2")
    copy.defineType("New", 9)
    assert (c.size(), copy.size(), copy.getLabel()) == (3, 4, "Zero")


def thresholds() -> tuple[RooRealVar, RooThresholdCategory]:
    x = RooRealVar("x", "x", 1, -10, 10)
    t = RooThresholdCategory("t", "t", x, "high", 9)
    t.addThreshold(2, "low")
    t.addThreshold(5, "mid", 4)
    t.addThreshold(1, "low")
    return x, t


def test_a_threshold_category_is_the_state_of_the_first_threshold_above_the_value() -> None:
    """ROOT's indices for 0.5, 3 and 7: ``low`` (10), ``mid`` (4) and the default ``high`` (9)."""
    x, t = thresholds()
    found = []
    for value in (0.5, 3.0, 7.0):
        x.setVal(value)
        found.append(t.getIndex())
    assert found == [10, 4, 9]
    assert list(t.compute({"x": np.array([0.5, 3.0, 7.0])})) == [10.0, 4.0, 9.0]
    assert not t.isFundamental()


def test_a_threshold_category_labels_the_state_it_computes() -> None:
    """ROOT's labels for 0.5, 3 and 7 are ``low``, ``mid`` and ``high``."""
    x, t = thresholds()
    found = []
    for value in (0.5, 3.0, 7.0):
        x.setVal(value)
        found.append(t.getLabel())
    assert found == ["low", "mid", "high"]


def test_a_computed_category_is_kept_in_a_dataset_as_a_category_of_its_states() -> None:
    """A dataset's column of ``t`` is a plain category with the same labels and indices."""
    _, t = thresholds()
    made = t.as_fundamental()
    assert isinstance(made, RooCategory)
    assert made.states() == {"high": 9, "low": 10, "mid": 4}
    assert made.GetName() == "t"


def test_a_binning_category_is_the_bin_a_value_falls_in() -> None:
    """ROOT: 6 in four bins over ``[0, 10]`` is bin 2; the states are named from the binning."""
    xb = RooRealVar("xb", "xb", 0, 10)
    xb.setBins(4)
    bc = RooBinningCategory("bc", "bc", xb)
    xb.setVal(6)
    assert (bc.getIndex(), bc.size()) == (2, 4)
    assert bc.lookupName(2) == "xb_bin2"
    xb.setBins(2, "two")
    assert RooBinningCategory("bc2", "bc2", xb, "two").getIndex() == 1
    assert RooBinningCategory("bc2", "bc2", xb, "two").lookupName(1) == "xb_two_bin1"
    assert RooBinningCategory("bc3", "bc3", xb, "two", "pre").lookupName(1) == "pre1"
    assert list(bc.compute({"xb": np.array([-1.0, 9.9, 11.0])})) == [0.0, 3.0, 3.0]


def test_a_binning_category_labels_the_bin_it_computes() -> None:
    """ROOT's label is ``xb_bin2`` for 6 in four bins over ``[0, 10]``."""
    xb = RooRealVar("xb", "xb", 6, 0, 10)
    xb.setBins(4)
    assert RooBinningCategory("bc", "bc", xb).getLabel() == "xb_bin2"


def mapped() -> tuple[RooCategory, RooMappedCategory]:
    c = RooCategory("c", "c")
    c.defineType("Plus", 1)
    c.defineType("Minus", -1)
    c.defineType("Zero", 0)
    m = RooMappedCategory("m", "m", c, "Other")
    m.map("P*", "Pos")
    m.map("M*", "Neg", 7)
    m.map("Pl*", "Pos")
    return c, m


def test_a_mapped_category_maps_states_by_wildcard_onto_its_own() -> None:
    """ROOT's indices: ``Plus`` to ``Pos`` (1), ``Minus`` to ``Neg`` (7), ``Zero`` unmapped (0)."""
    c, m = mapped()
    found = []
    for label in ("Plus", "Minus", "Zero"):
        c.setLabel(label)
        found.append(m.getIndex())
    assert found == [1, 7, 0]
    assert m.states() == {"Other": 0, "Pos": 1, "Neg": 7}
    assert list(m.compute({"c": np.array([1.0, -1.0, 0.0, 5.0])})) == [1.0, 7.0, 0.0, 0.0]


def test_a_mapped_category_labels_the_state_it_maps_onto() -> None:
    """ROOT's labels: ``Pos``, ``Neg`` and ``Other``."""
    c, m = mapped()
    found = []
    for label in ("Plus", "Minus", "Zero"):
        c.setLabel(label)
        found.append(m.getLabel())
    assert found == ["Pos", "Neg", "Other"]


def test_a_multi_category_has_a_state_for_each_combination_of_its_inputs() -> None:
    """ROOT's index of ``{Minus;b}`` is 4: the first input's position plus three times the next."""
    c = signs()
    d = RooCategory("d", "d", {"a": 0, "b": 1})
    mc = RooMultiCategory("mc", "mc", [d, c])
    c.setLabel("Minus")
    d.setLabel("b")
    assert mc.getIndex() == 4
    assert [one.GetName() for one in mc.inputs] == ["c", "d"]


def test_a_multi_category_names_each_combination_as_root_does() -> None:
    """ROOT's states: ``{Plus;a}`` 0, ``{Minus;a}`` 1, ``{Zero;a}`` 2, ``{Plus;b}`` 3 ..."""
    c = signs()
    d = RooCategory("d", "d", {"a": 0, "b": 1})
    mc = RooMultiCategory("mc", "mc", [c, d])
    assert mc.lookupIndex("{Minus;b}") == 4
    assert mc.size() == 6


def test_a_multi_category_of_one_input_names_each_state_in_braces() -> None:
    """ROOT's ``{Plus}`` 0, ``{Minus}`` 1 and ``{Zero}`` 2, whatever the input's indices."""
    c = signs()
    m = RooMultiCategory("m", "m", [c])
    assert m.states() == {"{Plus}": 0, "{Minus}": 1, "{Zero}": 2}
    c.setLabel("Zero")
    assert m.getIndex() == 2
    assert list(m.compute({"c": np.array([1.0, 2.0])})) == [0.0, 1.0]


def test_a_super_category_sets_its_inputs_when_its_state_is_set() -> None:
    """ROOT's ``setLabel("{Minus}")`` sets ``c`` to ``Minus``; an unknown label is refused."""
    c = signs()
    s = RooSuperCategory("s", "s", [c])
    assert s.setLabel("{Minus}") is False
    assert (c.getLabel(), s.getIndex()) == ("Minus", 1)
    assert s.setLabel("{nope}") is True


def test_variables_the_density_ignores_are_drawn_uniformly_in_the_order_asked_for() -> None:
    """rf406: ROOT's counts for ``{x, b0flav, tagCat}``, and the other way round."""
    from xrdroot.roofit.pdfs.basic import RooPolynomial
    from xrdroot.roofit.rng import generator

    tag = RooCategory("tagCat", "Tagging category")
    for state in ("Lepton", "Kaon", "NetTagger-1", "NetTagger-2"):
        tag.defineType(state)
    flavour = RooCategory("b0flav", "B0 flavour eigenstate")
    flavour.defineType("B0", -1)
    flavour.defineType("B0bar", 1)
    x = RooRealVar("x", "x", 0, 10)
    p = RooPolynomial("p", "p", x)
    counts = []
    for order in ([x, flavour, tag], [x, tag, flavour]):
        generator().SetSeed(4357)
        data = p.generate(order, 10000)
        tags = data.table(tag)
        counts.append((tags.get("Lepton") + tags.get("Kaon"), data.table(flavour).get("B0")))
    assert counts == [(5040, 5058), (5058, 5040)]
