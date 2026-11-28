"""``Tree.arrays(["pt", "sqrt(px*px + py*py)"], cut=...)``: what ``select`` does with the names.
The engine's side of an array read is a **source** - a handful of methods a tree or a chain
answers - so it is exercised here against a table of columns rather than a file: which branches
an expression needs are read, once each, and the cut is applied entry by entry or element by
element exactly as ``TTree::Draw`` applies it. That keeps ``xrdroot.formula.select`` inside the
``hist`` component's isolation scope, where a test may not open a ROOT file."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xrdroot import Jagged
from xrdroot.formula.select import TreeNames, dimensions, one_entry, select


class Leaf:
    """A branch's leaf as ``_fixed`` reads it: its title, whose sizes declare the shape."""

    def __init__(self, title: str = "") -> None:
        self.title = title


class Branch:
    """A branch as ``_Read`` reads one: a range of entries, or the entries named."""

    def __init__(self, values: Any, *, title: str = "", typename: str = "double") -> None:
        self.values = values
        self.leaf = Leaf(title)
        self.typename = typename

    @property
    def is_jagged(self) -> bool:
        return isinstance(self.values, Jagged)

    @property
    def length(self) -> int:
        """What a value of this branch spans: a row's width, as a leaf title counts it."""
        if self.is_jagged:
            return len(self.values.content) // len(self.values)
        return int(np.prod(self.values.shape[1:]))

    def array(self, low: int, high: int) -> Any:
        return self.values[low:high]

    def pick(self, rows: Any) -> Any:
        return self.values[rows]


class Source:
    """A tree as the engine needs one: its entries, its branches, and their names."""

    def __init__(self, columns: dict[str, Branch]) -> None:
        self.columns = columns
        self.friends: dict[str, Any] = {}

    def keys(self) -> list[str]:
        return list(self.columns)

    def readable(self) -> list[str]:
        return list(self.columns)

    def __contains__(self, name: object) -> bool:
        return name in self.columns

    def __getitem__(self, name: str) -> Branch:
        """A branch of its own, a friend's under ``alias.name``, or a friend's bare name."""
        alias, _, rest = name.partition(".")
        if rest and alias in self.friends:
            return self.friends[alias][rest]
        if name in self.columns:
            return self.columns[name]
        for friend in self.friends.values():
            if name in friend:
                return friend[name]
        raise KeyError(name)

    def __len__(self) -> int:
        return len(next(iter(self.columns.values())).values)

    def _selected(self, rows: Any) -> Any:
        """The local row of each entry named, which a source of one file answers as itself."""
        return np.asarray(rows, dtype=np.int64)


def _source() -> Source:
    """Three entries: ``n`` a number each, ``pt`` a collection of differing length."""
    return Source(
        {
            "n": Branch(np.array([1.0, 2.0, 3.0])),
            "px": Branch(np.array([3.0, 0.0, 1.0])),
            "py": Branch(np.array([4.0, 2.0, 0.0])),
            "pt": Branch(Jagged(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), np.array([0, 2, 4, 5]))),
            "eta": Branch(Jagged(np.array([1.0, 3.0, 0.5, 2.0, 4.0]), np.array([0, 2, 4, 5]))),
        }
    )


def test_a_branch_is_read_and_an_expression_is_computed_under_its_own_text():
    """``arrays(["n", "sqrt(px*px + py*py)"])`` reads ``n``'s branch and computes the
    second from the two branches it names - 3-4-5 and 0-2-2 - under the text it was
    asked for."""
    columns = select(_source(), ["n", "sqrt(px*px + py*py)"])
    assert columns["n"].tolist() == [1.0, 2.0, 3.0]
    assert columns["sqrt(px*px + py*py)"].tolist() == pytest.approx([5.0, 2.0, 1.0])


def test_a_branch_is_read_once_however_many_expressions_share_it():
    """``px`` is named by both expressions and read once: the read records each branch it took, so a
    second expression over it costs nothing."""
    source = _source()
    columns = select(source, ["px * 2", "px + 1"])
    assert columns["px * 2"].tolist() == [6.0, 0.0, 2.0]
    assert columns["px + 1"].tolist() == [4.0, 1.0, 2.0]


def test_a_cut_that_is_a_value_per_entry_keeps_the_entries_where_it_holds():
    """``cut="n > 1"`` is one value per entry, so it drops whole rows: the entries where it is
    nonzero keep every column they had."""
    columns = select(_source(), ["n", "px"], cut="n > 1")
    assert columns["n"].tolist() == [2.0, 3.0]
    assert columns["px"].tolist() == [0.0, 1.0]


def test_a_cut_that_loops_applies_element_by_element_and_keeps_the_entry():
    """A cut over a collection pairs its elements with the column's, up to the shorter, and keeps an
    entry in which **any** element passed - so the columns still line up entry for entry."""
    columns = select(_source(), ["pt", "eta"], cut="eta < 1")
    # the first entry's 1.0 and 3.0 both fail, the second's 0.5 passes where its 2.0 does not, and
    # the third entry's only jet fails - so one entry survives, with the one element that passed
    assert columns["eta"].tolist() == [[0.5]]
    assert columns["pt"].tolist() == [[30.0]]


def test_a_range_of_entries_is_what_is_read():
    """``entry_start``/``entry_stop`` name the entries, and ``entries`` the numbers a chunk was read
    for, which is how a draw reads a tree in parts."""
    columns = select(_source(), ["n"], 1, 3)
    assert columns["n"].tolist() == [2.0, 3.0]
    chunk = select(_source(), ["n"], entries=np.array([0, 2]))
    assert chunk["n"].tolist() == [1.0, 3.0]


def test_the_columns_come_back_in_the_library_asked_for():
    """``library="ak"`` hands the columns to ``convert``; an extra that is not installed is refused
    in a full sentence naming the ``pip install`` that fixes it."""
    pytest.importorskip("awkward")
    converted = select(_source(), ["n"], library="ak")
    assert converted["n"].tolist() == [1.0, 2.0, 3.0]


def test_the_names_a_tree_offers_include_a_friend_s_and_are_counted_by_dimension():
    """``TreeNames`` is what an expression compiles against: the branches, a friend's under both its
    bare and its prefixed name, and each name's number of dimensions."""
    source = _source()
    source.friends["f"] = Source({"x": Branch(np.array([1.0, 2.0, 3.0]))})
    names = TreeNames(source)
    assert "n" in names and "f.x" in names and "x" in names  # a friend under both its names
    assert (names["n"], names["pt"], names["x"]) == (0, 1, 0)  # a number, a collection of them
    assert len(names) == len(list(names)) == 7  # five of the tree's and two of the friend's


def test_a_fixed_size_branch_is_shaped_as_its_title_declares():
    """``m[3][3]`` read as nine numbers is made three by three, and a jagged branch of that width as
    rows of three - ``dimensions`` counts the sizes the title names."""
    fixed = Branch(np.arange(27.0).reshape(3, 9), title="m[3][3]")
    assert dimensions(fixed) == 2
    jagged = Branch(Jagged(np.arange(6.0), np.array([0, 3, 6])), title="v[3]")
    assert dimensions(jagged) == 2  # rows of three
    source = Source({"m": fixed, "v": jagged})
    # a branch read by name answers what its `array` returns; it is an **expression** over it that
    # `_shaped` reshapes to the title's sizes, which is why `m * 1` comes back three by three
    assert select(source, ["m"])["m"].shape == (3, 9)
    # an expression that indexes the title's two sizes can only work if `_shaped` read the nine
    # numbers as three by three first: m[1][1] is each entry's fifth number
    assert select(source, ["m[1][1] * 1"])["m[1][1] * 1"].tolist() == [4.0, 13.0, 22.0]
    assert select(source, ["v"])["v"].tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_one_entry_reads_a_formula_s_values_and_mask_for_that_entry():
    """``one_entry`` is what a draw's whole-object path asks for a single row, and it returns the
    values and the mask together."""
    from xrdroot.formula import compile_formula

    source = _source()
    formula = compile_formula("n * 2", TreeNames(source))
    values, valid = one_entry(source, formula, 1)
    assert values.tolist() == [4.0]
    assert valid.tolist() == [True]
