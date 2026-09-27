"""``RooArgSet`` and ``RooArgList``: membership, lookups, selections, values and printing.

A set refuses a second member of a name it holds and a list does not; both
find members by name or position and hand back their values by name. The
printed lines were compared with ROOT 6.40.04, whose ``Print("s")`` numbers
the members and right-aligns their names one wider than the longest.
"""

from __future__ import annotations

import re

import pytest

from xrdroot.roofit.categories import RooCategory
from xrdroot.roofit.collections import RooArgList, RooArgSet, as_list, names_of
from xrdroot.roofit.printing import kCollectionHeader, kName, kStandard
from xrdroot.roofit.variables import RooConstVar, RooRealVar


def variables() -> tuple[RooRealVar, RooRealVar, RooRealVar]:
    return (
        RooRealVar("x", "the x", 1, -10, 10),
        RooRealVar("mean", "m", 0, -1, 1),
        RooRealVar("sigma", "s", 2, 0.1, 5),
    )


def test_a_set_refuses_a_name_it_holds_and_a_list_takes_it() -> None:
    """A set's members are unique by name; a list keeps its order and repeats."""
    x, m, _ = variables()
    other = RooRealVar("x", "another x", 3.0)
    held = RooArgSet(x, m)
    assert not held.add(other)
    assert held.find("x") is x
    assert RooArgList(x, other).names() == ["x", "x"]
    assert held.add([RooRealVar("a", "a", 1.0), RooArgList(RooRealVar("b", "b", 1.0))])
    assert held.names() == ["x", "mean", "a", "b"]
    assert held.addOwned(RooRealVar("c", "c", 1.0))


def test_a_trailing_string_names_the_collection() -> None:
    """``RooArgSet(x, m, "name")`` names the set, and the members may come nested."""
    x, m, s = variables()
    held = RooArgSet([x, (m,)], {s}, "params")
    assert held.GetName() == "params"
    assert held.GetTitle() == "params"
    assert held.names() == ["x", "mean", "sigma"]
    held.SetName("other")
    assert held.printName() == "other" and held.printTitle() == "other"
    assert held.ClassName() == "RooArgSet"


def test_members_are_removed_by_name_one_or_many_at_a_time() -> None:
    """``remove`` matches by name, and says whether anything went."""
    x, m, s = variables()
    held = RooArgList(x, m, s, x)
    assert held.remove(RooRealVar("x", "x", 0.0))
    assert held.names() == ["mean", "sigma"]
    assert not held.remove(x)
    assert held.remove([m, s])
    assert held.empty()
    held.add(x)
    held.removeAll()
    assert held.size() == 0 and held.getSize() == 0


def test_a_member_is_replaced_in_place_by_name() -> None:
    """``replace`` keeps the position of what it replaces, and fails for a name not held."""
    x, m, s = variables()
    held = RooArgList(x, m)
    assert held.replace(m, s)
    assert held.names() == ["x", "sigma"]
    assert not held.replace(m, s)


def test_members_are_found_by_name_by_object_and_by_position() -> None:
    """``find``, ``index``, ``at`` and ``[]`` all find members, and say so when they do not."""
    x, m, s = variables()
    held = RooArgList(x, m)
    assert held.find(m) is m
    assert held.contains("x") and "mean" in held and s not in held
    assert held.containsInstance(x)
    assert not held.containsInstance(RooRealVar("x", "x", 0.0))
    assert held.index("mean") == 1 and held.index(m) == 1 and held.index(s) == -1
    assert held.at(0) is x and held.at(2) is None and held.at(-1) is None
    assert held.first() is x and RooArgList().first() is None
    assert held["mean"] is m and held[1] is m
    with pytest.raises(KeyError, match="RooArgList has no member called 'sigma'"):
        held["sigma"]
    assert list(held) == [x, m] and len(held) == 2
    assert bool(RooArgList())


def test_two_collections_are_compared_by_their_names() -> None:
    """``overlaps``, ``hasSameLayout`` and ``equals``: the first looks at any, the others at all."""
    x, m, s = variables()
    assert RooArgSet(x, m).overlaps([m, s])
    assert not RooArgSet(x).overlaps(RooArgSet(s))
    assert RooArgSet(x, m).hasSameLayout(RooArgList(x, m))
    assert not RooArgSet(x, m).hasSameLayout(RooArgList(m, x))
    assert RooArgSet(x, m).equals(RooArgList(m, x))


def test_selections_by_name_attribute_and_membership_are_new_collections() -> None:
    """``selectByName`` takes comma-separated wildcards; the others an attribute or a set."""
    x, m, s = variables()
    held = RooArgSet(x, m, s)
    chosen = held.selectByName("m*,s?gma,")
    assert isinstance(chosen, RooArgSet)
    assert chosen.names() == ["mean", "sigma"]
    s.setConstant(True)
    assert held.selectByAttrib("Constant", True).names() == ["sigma"]
    assert held.selectByAttrib("Constant", False).names() == ["x", "mean"]
    assert held.selectCommon([s, x]).names() == ["x", "sigma"]


def test_a_snapshot_keeps_the_values_the_members_had() -> None:
    """``snapshot`` copies, so changing the originals afterwards leaves it as it was."""
    x, m, _ = variables()
    saved = RooArgSet(x, m).snapshot()
    x.setVal(5.0)
    assert saved.find("x").getVal() == 1.0
    assert saved.find("x") is not x
    RooArgSet(x, m).assign(saved)
    assert x.getVal() == 1.0
    RooArgSet(x).assign([RooRealVar("absent", "a", 3.0), RooRealVar("x", "x", 2.0, -10, 10)])
    assert x.getVal() == 2.0


def test_real_values_are_set_and_read_by_name() -> None:
    """``setRealValue`` returns true for an error - a name not held - as ROOT does."""
    x, m, _ = variables()
    held = RooArgSet(x, m)
    assert not held.setRealValue("x", 3.5)
    assert held.getRealValue("x") == 3.5
    assert held.setRealValue("absent", 1.0)
    assert held.getRealValue("absent", 7.0) == 7.0


def test_category_states_are_set_and_read_by_name() -> None:
    """The same for a category's index and label, with a default for a name not held."""
    cat = RooCategory("c", "c")
    cat.defineType("a", 1)
    cat.defineType("b", 2)
    held = RooArgSet(cat)
    assert not held.setCatIndex("c", 2)
    assert held.getCatIndex("c") == 2 and held.getCatLabel("c") == "b"
    assert not held.setCatLabel("c", "a")
    assert held.getCatIndex("c") == 1
    assert held.setCatIndex("absent", 1) and held.setCatLabel("absent", "a")
    assert held.getCatIndex("absent", 9) == 9
    assert held.getCatLabel("absent", "none") == "none"


def test_attributes_are_set_on_every_member_and_constness_asks_all() -> None:
    """``setAttribAll("Constant")`` makes the set constant; one free member makes it not."""
    x, m, _ = variables()
    held = RooArgSet(x, m)
    assert not held.isConstant()
    held.setAttribAll("Constant")
    assert held.isConstant()
    held.setAttribAll("Constant", False)
    assert not x.isConstant()


def test_a_collection_sorts_by_name_in_place_or_as_a_copy() -> None:
    """``sort`` orders in place, either way; ``sorted_copy`` keeps the name."""
    x, m, s = variables()
    held = RooArgList(x, m, s, "named")
    copy = held.sorted_copy()
    assert copy.names() == ["mean", "sigma", "x"] and copy.GetName() == "named"
    held.sort(reverse=True)
    assert held.names() == ["x", "sigma", "mean"]
    held.sort()
    assert held.names() == ["mean", "sigma", "x"]


def test_a_named_set_prints_its_class_name_and_members_as_root_does(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``Print()`` is one line of the member names; an unnamed list shows ``RooArgList::``."""
    x, m, s = variables()
    RooArgSet(x, m, s, "myset").Print()
    RooArgList(x, m).Print()
    RooArgList(x, m).Print("i")
    assert capsys.readouterr().out == (
        "RooArgSet::myset = (x,mean,sigma)\nRooArgList:: = (x,mean)\nRooArgList:: = (x,mean)"
    )


def test_the_standard_print_numbers_the_members_in_one_name_width(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``Print("s")`` right-aligns each name one wider than the longest, as ROOT 6.40 does."""
    x, m, s = variables()
    RooArgSet(x, m, s, "myset").Print("s")
    assert capsys.readouterr().out == (
        "  1) RooRealVar::     x = 1\n  2) RooRealVar::  mean = 0\n  3) RooRealVar:: sigma = 2\n"
    )


def test_the_verbose_print_adds_addresses_ranges_and_titles(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``Print("v")`` is ROOT's: address, class, name, value, range, then the title quoted."""
    x, m, s = variables()
    RooArgSet(x, m, s, "myset").Print("v")
    out = re.sub(r"0x[0-9a-f]+", "@", capsys.readouterr().out)
    assert out == (
        '  1) @ RooRealVar::     x = 1  L(-10 - 10)  "the x"\n'
        '  2) @ RooRealVar::  mean = 0  L(-1 - 1)  "m"\n'
        '  3) @ RooRealVar:: sigma = 2  L(0.1 - 5)  "s"\n'
    )


def test_a_collection_header_names_the_collection_above_its_members() -> None:
    """With ``kCollectionHeader`` a named collection is introduced by its class and name."""
    x, _, _ = variables()
    text = RooArgSet(x, "held").printStream(kName | kCollectionHeader, kStandard, "> ")
    assert text == "> RooArgSet::held:\n>   1)  x\n"
    assert RooArgSet().printStream(kName, kStandard) == ""


def test_an_empty_collection_prints_empty_brackets_and_reprs_its_members() -> None:
    """The value of a collection is its member names in brackets, and ``str`` is just that."""
    x, m, _ = variables()
    assert str(RooArgList(x, m)) == "(x,mean)"
    assert repr(RooArgSet(x, m)) == "<RooArgSet (x,mean)>"
    assert RooArgSet().printValue() == "()"


def test_a_name_width_already_set_is_kept_while_members_print() -> None:
    """Nested printing keeps the outer width rather than choosing its own."""
    x, _, _ = variables()
    RooArgSet.name_length[0] = 4
    try:
        assert RooArgSet(x).printStream(kName, kStandard) == "  1)    x\n"
    finally:
        RooArgSet.name_length[0] = 0


def test_anything_iterable_stands_for_a_list_of_arguments() -> None:
    """``as_list`` takes one argument, a collection, nested Python containers, or nothing."""
    x, m, s = variables()
    c = RooConstVar("c", "c", 1.0)
    assert as_list(None) == []
    assert as_list(x) == [x]
    assert as_list([x, (m, frozenset({s}))]) == [x, m, s]
    assert as_list(RooArgList(x, c)) == [x, c]
    assert names_of({x}) == ["x"]
